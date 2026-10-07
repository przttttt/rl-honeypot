from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from ..agents.tabular_q import TabularQAgent
from ..cowrie.parser import iter_cowrie_events
from ..envs import gym
from ..metrics.ci import mean_with_ci
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..transitions.estimate import TransitionEstimate, estimate
from .e4_baselines import _q_of
SHIFT_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

def _escalation_target(n_states: int) -> np.ndarray:
    tgt = np.zeros((n_states, n_states, n_states), dtype=np.float64)
    for s in range(n_states):
        tgt[s, :, min(s + 1, n_states - 1)] = 1.0
    return tgt

def shifted_estimate(src: TransitionEstimate, level: float) -> TransitionEstimate:
    if not 0.0 <= level <= 1.0:
        raise ValueError(f'level must be in [0, 1], got {level}')
    n_states = src.probs.shape[0]
    tgt = _escalation_target(n_states)
    blend = (1.0 - level) * src.probs + level * tgt
    probs = np.where(src.support_mask[:, :, None], blend, 0.0)
    probs = probs / np.clip(probs.sum(axis=2, keepdims=True), 1e-12, None)
    return TransitionEstimate(probs=probs, counts=src.counts, support_mask=src.support_mask, observed_support=src.observed_support, n_sessions=src.n_sessions, n_transitions=src.n_transitions)

def mean_tv(a: TransitionEstimate, b: TransitionEstimate) -> float:
    diff = 0.5 * np.abs(a.probs - b.probs).sum(axis=2)
    mask = a.support_mask.astype(bool) & b.support_mask.astype(bool)
    return float(diff[mask].mean()) if mask.any() else 0.0

def _train_offline(env, episodes: int, seed: int, hp: dict) -> TabularQAgent:
    agent = TabularQAgent(alpha=hp['alpha'], gamma=hp['gamma'], epsilon=hp['epsilon'], seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
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
    ctx = make_context('e9', out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg['mdp']['hyperoffline']
    hn = cfg['mdp']['hyperonline']
    laplace = cfg['transitions']['laplace_alpha']
    min_support = cfg['transitions']['min_support_flag']
    est1 = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p1.json'))))], laplace_alpha=laplace, min_support=min_support)
    est2 = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p2.json.json'))))], laplace_alpha=laplace, min_support=min_support)
    n_seeds = 3 if quick else cfg['eval']['num_seeds']
    pa_episodes = 60 if quick else ho['episodes']
    pb_episodes = 60 if quick else 200
    env_src = gym.make(cfg['env']['id_tabular'], transitions=est1, max_steps=ho['max_steps'])
    q_inits: list[np.ndarray] = []
    policies_a: list[np.ndarray] = []
    for seed_i in range(n_seeds):
        seed = ctx.seed + seed_i * 7919
        agent = _train_offline(env_src, pa_episodes, seed, ho)
        policies_a.append(agent.policy())
        q_inits.append(_q_of(policies_a, seed_i))
    d_full = mean_tv(est1, shifted_estimate(est1, 1.0))
    per_level: dict[str, dict] = {}
    for level in SHIFT_LEVELS:
        env_s = gym.make(cfg['env']['id_tabular'], transitions=shifted_estimate(est1, level), max_steps=ho['max_steps'])
        (cold, warm) = ([], [])
        for seed_i in range(n_seeds):
            seed = ctx.seed + 5000 + seed_i * 7919
            cold.append(_online_first50(env_s, None, pb_episodes, seed, hn))
            warm.append(_online_first50(env_s, q_inits[seed_i], pb_episodes, seed, hn))
        delta = mean_with_ci(np.array(warm) - np.array(cold), resamples=2000, seed=0)
        per_level[f'{level:.2f}'] = {'shift_level': level, 'divergence_tv_from_source': mean_tv(est1, shifted_estimate(est1, level)), 'cold_first50': mean_with_ci(np.array(cold), resamples=2000, seed=0), 'warm_first50': mean_with_ci(np.array(warm), resamples=2000, seed=0), 'delta_warm_minus_cold': delta, 'warm_helps': bool(delta['ci_lo'] > 0), 'warm_hurts': bool(delta['ci_hi'] < 0)}
    d_corpus = mean_tv(est1, est2)
    matched_level = d_corpus / d_full if d_full > 0 else 0.0
    nearest = min(SHIFT_LEVELS, key=lambda lv: abs(lv - matched_level))
    matched = per_level[f'{nearest:.2f}']
    crossover: float | None = None
    for lv in SHIFT_LEVELS:
        if per_level[f'{lv:.2f}']['warm_hurts']:
            crossover = lv
            break
    results: dict = {'experiment': 'e9_phase_shift', 'provenance': 'synthetic', 'status': 'ok', 'design': {'source': 'estimated phase-1 transition tensor (E2)', 'target': 'convex blend toward one-hot escalation (next state = s+1)', 'shift_levels': list(SHIFT_LEVELS), 'n_seeds': n_seeds, 'phase_a_episodes': pa_episodes, 'phase_b_episodes': pb_episodes}, 'per_level': per_level, 'divergence_full_shift_tv': d_full, 'corpus_phase1_to_phase2_tv': d_corpus, 'corpus_matched_level': matched_level, 'corpus_matched_nearest_level': nearest, 'corpus_matched_delta': matched['delta_warm_minus_cold'], 'corpus_matched_warm_hurts': matched['warm_hurts'], 'first_hurting_level': crossover, 'note': "Δ(warm−cold) on first-50 return as a function of a controlled dynamics shift; the corpus's own phase-1→phase-2 divergence is placed on the same axis to test whether the D9 negative finding is attributable to the size of the two-phase shift.", 'decision_refs': ['D9', 'D13', 'D18']}
    return finish(ctx, results, started)
if __name__ == '__main__':
    r = run(quick=True)
    for (lv, d) in r['per_level'].items():
        dc = d['delta_warm_minus_cold']
        print(f"level {lv}: Δ = {dc['mean']:+.3f} [{dc['ci_lo']:+.3f}, {dc['ci_hi']:+.3f}]")
