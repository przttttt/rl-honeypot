from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from ..agents.tabular_q import TabularQAgent
from ..cowrie.parser import iter_cowrie_events
from ..envs import gym
from ..metrics.ci import mean_with_ci
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..transitions.estimate import estimate
from .e4_baselines import _q_of

def _phase2_start_distribution(ground_truth_path: Path) -> dict[str, float]:
    gt = json.loads(ground_truth_path.read_text())
    counts: dict[int, int] = {}
    total = 0
    for (sid, info) in gt['sessions'].items():
        if info.get('phase') != 2:
            continue
        states = info.get('states', [])
        if not states:
            continue
        s0 = int(states[0])
        counts[s0] = counts.get(s0, 0) + 1
        total += 1
    if total == 0:
        return {'0': 1.0}
    return {str(s): c / total for (s, c) in sorted(counts.items())}

def _train_offline(env, episodes: int, seed: int, hp: dict, start_states: np.ndarray | None=None, start_probs: np.ndarray | None=None) -> TabularQAgent:
    agent = TabularQAgent(alpha=hp['alpha'], gamma=hp['gamma'], epsilon=hp['epsilon'], seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        if start_states is not None and start_probs is not None:
            s0 = int(rng.choice(start_states, p=start_probs))
            (obs, _) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)), options={'start_state': s0})
        else:
            (obs, _) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        s = int(obs)
        while not (done or trunc):
            a = agent.act(s)
            (obs2, r, done, trunc, _) = env.step(a)
            agent.update(s, a, r, int(obs2), done or trunc)
            s = int(obs2)
    return agent

def _online_first50(env, q_init, episodes: int, seed: int, hp: dict) -> float:
    agent = TabularQAgent(alpha=hp['alpha'], gamma=hp['gamma'], epsilon=hp['epsilon'], seed=seed, q_init=q_init)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    for _ in range(episodes):
        (obs, _) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        s = int(obs)
        total = 0.0
        while not (done or trunc):
            a = agent.act(s)
            (obs2, r, done, trunc, _) = env.step(a)
            agent.update(s, a, r, int(obs2), done or trunc)
            total += r
            s = int(obs2)
        returns.append(total)
    return float(np.mean(returns[:50]))

def run(cfg=None, overrides=None, out_root: str='results', quick: bool=False) -> dict:
    started = time.time()
    ctx = make_context('e12', out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg['mdp']['hyperoffline']
    hn = cfg['mdp']['hyperonline']
    laplace = cfg['transitions']['laplace_alpha']
    min_support = cfg['transitions']['min_support_flag']
    est1 = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p1.json'))))], laplace_alpha=laplace, min_support=min_support)
    est2 = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p2.json.json'))))], laplace_alpha=laplace, min_support=min_support)
    gt_path = Path('data/raw/ground_truth.json')
    p2_dist = _phase2_start_distribution(gt_path)
    states_arr = np.array([int(s) for s in p2_dist.keys()])
    probs_arr = np.array([p2_dist[s] for s in p2_dist.keys()])
    probs_arr = probs_arr / probs_arr.sum()
    n_seeds = 3 if quick else 10
    pa_episodes = 60 if quick else ho['episodes']
    pb_episodes = 60 if quick else 200
    env1 = gym.make(cfg['env']['id_tabular'], transitions=est1, max_steps=ho['max_steps'])
    env2 = gym.make(cfg['env']['id_tabular'], transitions=est2, max_steps=ho['max_steps'])
    results_by_mode: dict[str, list[float]] = {'cold': [], 'warm_uniform': [], 'warm_matched': []}
    for seed_i in range(n_seeds):
        seed = ctx.seed + seed_i * 7919
        agent_uni = _train_offline(env1, pa_episodes, seed, ho)
        q_uni = _q_of([agent_uni.policy()], 0)
        seed_matched = ctx.seed + 9999 + seed_i * 7919
        agent_mat = _train_offline(env1, pa_episodes, seed_matched, ho, start_states=states_arr, start_probs=probs_arr)
        q_mat = _q_of([agent_mat.policy()], 0)
        seed_pb = ctx.seed + 20000 + seed_i * 7919
        results_by_mode['cold'].append(_online_first50(env2, None, pb_episodes, seed_pb, hn))
        results_by_mode['warm_uniform'].append(_online_first50(env2, q_uni, pb_episodes, seed_pb, hn))
        results_by_mode['warm_matched'].append(_online_first50(env2, q_mat, pb_episodes, seed_pb, hn))
    cold_arr = np.array(results_by_mode['cold'])
    uni_arr = np.array(results_by_mode['warm_uniform'])
    mat_arr = np.array(results_by_mode['warm_matched'])
    modes = {'cold': mean_with_ci(cold_arr, resamples=2000, seed=0), 'warm_uniform_start': mean_with_ci(uni_arr, resamples=2000, seed=0), 'warm_matched_start': mean_with_ci(mat_arr, resamples=2000, seed=0)}
    deltas = {'uniform_vs_cold': mean_with_ci(uni_arr - cold_arr, resamples=2000, seed=0), 'matched_vs_cold': mean_with_ci(mat_arr - cold_arr, resamples=2000, seed=0), 'matched_vs_uniform': mean_with_ci(mat_arr - uni_arr, resamples=2000, seed=0)}
    d_mat = deltas['matched_vs_cold']
    d_uni = deltas['uniform_vs_cold']
    occ_explains = bool(d_mat['ci_lo'] > 0 and d_mat['mean'] > d_uni['mean'])
    results: dict = {'experiment': 'e12_occupancy', 'provenance': 'synthetic', 'status': 'ok', 'design': {'source_mdp': 'phase-1 transition estimate (est1)', 'target_mdp': 'phase-2 transition estimate (est2)', 'phase_a_episodes': pa_episodes, 'phase_b_episodes': pb_episodes, 'n_seeds': n_seeds, 'start_distribution_source': 'data/raw/ground_truth.json (phase-2 empirical)'}, 'phase2_start_distribution': p2_dist, 'modes': modes, 'deltas': deltas, 'occupancy_explains_penalty': occ_explains, 'note': 'warm_uniform_start = Q initialised from phase-A policy trained with S0 starts (matches E4 warm arm). warm_matched_start = same but phase-A trained from the phase-2 empirical start-state distribution. If matched_vs_cold CI excludes 0 on the positive side AND beats uniform, occupancy shift was the mechanism behind D9. If not, the penalty originates elsewhere.', 'decision_refs': ['D9', 'D13', 'D18']}
    return finish(ctx, results, started)
if __name__ == '__main__':
    r = run(quick=True)
    print('Occupancy explains penalty:', r['occupancy_explains_penalty'])
    for (k, v) in r['modes'].items():
        print(f"  {k:25s}: {v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    for (k, v) in r['deltas'].items():
        print(f"  Δ {k:30s}: {v['mean']:+.3f} [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]")
