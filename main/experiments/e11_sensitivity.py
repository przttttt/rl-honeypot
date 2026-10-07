from __future__ import annotations
import tempfile
import time
from pathlib import Path
import numpy as np
from ..agents.dqn import DQNAgent
from ..corpus.generator import generate_corpus
from ..cowrie.parser import iter_cowrie_events
from ..envs.scaled_env import FEATURE_DIM, ScaledResponseEnv, ScaledTabularQ, build_scaled
from ..metrics.ci import mean_with_ci
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..transitions.estimate import estimate
N_STATES_TESTED = [8, 128]
CORPUS_CONFIGS = [('default', 60000, 40000, 0), ('small', 20000, 15000, 1), ('large', 120000, 80000, 2)]

def _train_tabular(scaled: dict, n_states: int, ho: dict, seed: int, episodes: int, max_steps: int, noise: float) -> tuple[list[float], object]:
    env = ScaledResponseEnv(scaled, max_steps=max_steps, obs_mode='discrete', reward_noise_sigma=noise, seed=seed)
    agent = ScaledTabularQ(n_states=n_states, alpha=ho['alpha'], gamma=ho['gamma'], epsilon=ho['epsilon'], seed=seed)
    rng = np.random.default_rng(seed)
    curve: list[float] = []
    for _ in range(episodes):
        (obs, _) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            a = agent.act(int(obs))
            (obs2, r, done, trunc, _) = env.step(a)
            agent.update(int(obs), a, r, int(obs2), done or trunc)
            total += r
            obs = obs2
        curve.append(total)
    return (curve, agent)

def _eval_tabular(scaled: dict, agent: ScaledTabularQ, n_eval: int, max_steps: int, noise: float, seed: int) -> float:
    env = ScaledResponseEnv(scaled, max_steps=max_steps, obs_mode='discrete', reward_noise_sigma=noise, seed=seed)
    rng = np.random.default_rng(seed + 99991)
    returns = []
    for _ in range(n_eval):
        (obs, _) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            a = agent.act_greedy(int(obs))
            (obs2, r, done, trunc, _) = env.step(a)
            total += r
            obs = obs2
        returns.append(total)
    return float(np.mean(returns))

def _train_dqn(scaled: dict, dqn_cfg: dict, seed: int, episodes: int, max_steps: int, noise: float) -> tuple[list[float], DQNAgent]:
    env = ScaledResponseEnv(scaled, max_steps=max_steps, obs_mode='features', reward_noise_sigma=noise, seed=seed)
    agent = DQNAgent(obs_dim=FEATURE_DIM, n_actions=5, cfg=dqn_cfg, seed=seed)
    rng = np.random.default_rng(seed)
    curve: list[float] = []
    for _ in range(episodes):
        (obs, _) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            a = agent.act(context=np.asarray(obs, dtype=np.float32))
            (obs2, r, done, trunc, _) = env.step(a)
            agent.observe(obs, a, r, obs2, done or trunc)
            total += r
            obs = obs2
        curve.append(total)
    return (curve, agent)

def _eval_dqn(scaled: dict, agent: DQNAgent, n_eval: int, max_steps: int, noise: float, seed: int) -> float:
    env = ScaledResponseEnv(scaled, max_steps=max_steps, obs_mode='features', reward_noise_sigma=noise, seed=seed)
    rng = np.random.default_rng(seed + 88881)
    returns = []
    for _ in range(n_eval):
        (obs, _) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            a = agent.act(context=np.asarray(obs, dtype=np.float32), greedy=True)
            (obs2, r, done, trunc, _) = env.step(a)
            total += r
            obs = obs2
        returns.append(total)
    return float(np.mean(returns))

def run(cfg=None, overrides=None, out_root: str='results', quick: bool=False) -> dict:
    started = time.time()
    ctx = make_context('e11', out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg['mdp']['hyperoffline']
    dqn_cfg = cfg['agents']['dqn']
    scale_cfg = cfg.get('scale', {})
    noise = float(scale_cfg.get('reward_noise_sigma', 4.0))
    max_steps = int(ho.get('max_steps', 100))
    n_seeds = 3 if quick else 5
    episodes = 100 if quick else 800
    n_eval = 30 if quick else 200
    sweep: list[dict] = []
    for (label, n_phase1, n_phase2, seed_offset) in CORPUS_CONFIGS:
        corpus_seed = ctx.seed + seed_offset * 131071
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_corpus(out_dir=Path(tmpdir), n_phase1=n_phase1, n_phase2=n_phase2, seed=corpus_seed, anomaly_rate=0.011)
            labelled = label_events(list(iter_cowrie_events(paths.phase2)))
            est = estimate([ls.transitions for ls in labelled], laplace_alpha=cfg['transitions']['laplace_alpha'], min_support=cfg['transitions']['min_support_flag'])
        for n_states in N_STATES_TESTED:
            k = n_states // 8
            scaled = build_scaled(est, k=k, structure='shared', seed=ctx.seed)
            (tab_evals, dqn_evals) = ([], [])
            for seed_i in range(n_seeds):
                seed = ctx.seed + 50000 + seed_offset * 10007 + seed_i * 7919
                (_, tab_agent) = _train_tabular(scaled, n_states, ho, seed, episodes, max_steps, noise)
                tab_evals.append(_eval_tabular(scaled, tab_agent, n_eval, max_steps, noise, seed))
                (_, dqn_agent) = _train_dqn(scaled, dqn_cfg, seed, episodes, max_steps, noise)
                dqn_evals.append(_eval_dqn(scaled, dqn_agent, n_eval, max_steps, noise, seed))
            tab_ret = mean_with_ci(np.array(tab_evals), resamples=2000, seed=0)
            dqn_ret = mean_with_ci(np.array(dqn_evals), resamples=2000, seed=0)
            delta = mean_with_ci(np.array(dqn_evals) - np.array(tab_evals), resamples=2000, seed=0)
            sweep.append({'corpus_config': label, 'n_phase1': n_phase1, 'n_phase2': n_phase2, 'corpus_seed': corpus_seed, 'n_states': n_states, 'structure': 'shared', 'tabular_return': tab_ret, 'dqn_return': dqn_ret, 'delta_dqn_minus_tabular': delta, 'drl_wins': bool(delta['ci_lo'] > 0), 'tie': bool(delta['ci_lo'] <= 0 <= delta['ci_hi'])})
    n8_rows = [r for r in sweep if r['n_states'] == 8]
    n128_rows = [r for r in sweep if r['n_states'] == 128]
    n8_tie = all((r['tie'] for r in n8_rows))
    n128_drl = all((r['drl_wins'] for r in n128_rows))
    results: dict = {'experiment': 'e11_sensitivity', 'provenance': 'synthetic', 'status': 'ok', 'design': {'corpus_configs': [{'label': lbl, 'n_phase1': n1, 'n_phase2': n2} for (lbl, n1, n2, _) in CORPUS_CONFIGS], 'n_states_tested': N_STATES_TESTED, 'structure': 'shared', 'episodes_per_learner': episodes, 'n_seeds': n_seeds, 'n_eval_episodes': n_eval}, 'sweep': sweep, 'robustness_verdict': {'n8_always_tie': n8_tie, 'n128_always_drl_wins': n128_drl, 'overall': 'robust' if n8_tie and n128_drl else 'not_robust', 'note': 'At N=8 the two learners should tie (matching E5) across ALL corpus configs; at N=128 DRL should win (matching E10 shared regime). If both hold, the crossover finding is not a corpus-calibration artefact.'}, 'decision_refs': ['D3', 'D15', 'D33']}
    return finish(ctx, results, started)
if __name__ == '__main__':
    r = run(quick=True)
    print('Robustness verdict:', r['robustness_verdict']['overall'])
    for row in r['sweep']:
        d = row['delta_dqn_minus_tabular']
        print(f"  {row['corpus_config']:8s} N={row['n_states']:4d}: Δ = {d['mean']:+.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]  {('DRL wins' if row['drl_wins'] else 'tie/loss')}")
