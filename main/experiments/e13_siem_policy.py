from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from ..agents.tabular_q import TabularQAgent
from ..cowrie.parser import iter_cowrie_events
from ..envs import gym
from ..mdp.rewards import reward_table
from ..metrics.ci import mean_with_ci
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..transitions.estimate import estimate
_FALLBACK_LIFT = [1.06, 3.07, 1.52, 1.96, 1.04, 0.53, 0.31, 0.61]

def _load_lift(out_root: str) -> list[float]:
    p = Path(out_root) / 'e6' / 'result.json'
    if p.exists():
        try:
            e6 = json.loads(p.read_text())
            states = e6['lift']['states']
            lift = [s['lift'] for s in sorted(states, key=lambda x: x['state'])]
            if len(lift) == 8 and all((v is not None for v in lift)):
                return lift
        except Exception:
            pass
    return list(_FALLBACK_LIFT)

def _train_q(env, episodes: int, seed: int, hp: dict) -> TabularQAgent:
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

def _eval_greedy(env_eval, agent: TabularQAgent, n_eval: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    (returns, hijack_flags) = ([], [])
    for _ in range(n_eval):
        (obs, _) = env_eval.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        s = int(obs)
        total = 0.0
        seen_s5 = False
        allowed_at_s5 = False
        while not (done or trunc):
            a = agent.act_greedy(s)
            (obs2, r, done, trunc, _) = env_eval.step(a)
            total += r
            if s == 5:
                seen_s5 = True
                if a == 0:
                    allowed_at_s5 = True
            s = int(obs2)
        returns.append(total)
        hijack_flags.append(int(seen_s5 and (not allowed_at_s5)))
    return (float(np.mean(returns)), float(np.mean(hijack_flags)))

def run(cfg=None, overrides=None, out_root: str='results', quick: bool=False) -> dict:
    started = time.time()
    ctx = make_context('e13', out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg['mdp']['hyperoffline']
    laplace = cfg['transitions']['laplace_alpha']
    min_support = cfg['transitions']['min_support_flag']
    alpha_siem = float(cfg.get('siem', {}).get('reward_shaping_alpha', 0.5))
    est2 = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p2.json.json'))))], laplace_alpha=laplace, min_support=min_support)
    lift = _load_lift(out_root)
    lift_arr = np.array(lift, dtype=np.float64)
    R_base = reward_table()
    R_aug = R_base.copy()
    for s in range(8):
        R_aug[s, :, :] += alpha_siem * lift_arr[s]
    R_siem_only = np.zeros_like(R_base)
    for s in range(8):
        R_siem_only[s, :, :] += alpha_siem * lift_arr[s]
    n_seeds = 3 if quick else 10
    episodes = 300 if quick else 1000
    n_eval = 100 if quick else 400
    env_base = gym.make(cfg['env']['id_tabular'], transitions=est2, max_steps=ho['max_steps'])
    env_aug = gym.make(cfg['env']['id_tabular'], transitions=est2, max_steps=ho['max_steps'], reward_tensor=R_aug)
    env_only = gym.make(cfg['env']['id_tabular'], transitions=est2, max_steps=ho['max_steps'], reward_tensor=R_siem_only)
    env_eval = gym.make(cfg['env']['id_tabular'], transitions=est2, max_steps=ho['max_steps'])
    eval_returns: dict[str, list[float]] = {'base': [], 'siem_augmented': [], 'siem_only': []}
    hijack_rates: dict[str, list[float]] = {'base': [], 'siem_augmented': [], 'siem_only': []}
    greedy_policies: dict[str, list] = {'base': [], 'siem_augmented': [], 'siem_only': []}
    for seed_i in range(n_seeds):
        seed = ctx.seed + seed_i * 7919
        for (scheme, env_train) in (('base', env_base), ('siem_augmented', env_aug), ('siem_only', env_only)):
            agent = _train_q(env_train, episodes, seed, ho)
            (ret, hjr) = _eval_greedy(env_eval, agent, n_eval, seed=ctx.seed + 40000 + seed_i * 7919)
            eval_returns[scheme].append(ret)
            hijack_rates[scheme].append(hjr)
            greedy_policies[scheme].append(agent.policy().tolist())
    base_arr = np.array(eval_returns['base'])
    aug_arr = np.array(eval_returns['siem_augmented'])
    only_arr = np.array(eval_returns['siem_only'])
    policies_out = {}
    for scheme in ('base', 'siem_augmented', 'siem_only'):
        arr = np.array(eval_returns[scheme])
        pols = greedy_policies[scheme]
        consensus = [int(np.bincount([p[s] for p in pols], minlength=5).argmax()) for s in range(8)]
        policies_out[scheme] = {'eval_return': mean_with_ci(arr, resamples=2000, seed=0), 'hijack_prevented_rate': float(np.mean(hijack_rates[scheme])), 'greedy_policy': consensus}
    delta_aug = mean_with_ci(aug_arr - base_arr, resamples=2000, seed=0)
    delta_only = mean_with_ci(only_arr - base_arr, resamples=2000, seed=0)
    pol_base = policies_out['base']['greedy_policy']
    pol_aug = policies_out['siem_augmented']['greedy_policy']
    pol_only = policies_out['siem_only']['greedy_policy']
    agree_aug = float(sum((1 for s in range(8) if pol_base[s] == pol_aug[s])) / 8)
    agree_only = float(sum((1 for s in range(8) if pol_base[s] == pol_only[s])) / 8)
    siem_improves = bool(delta_aug['ci_lo'] > 0)
    results: dict = {'experiment': 'e13_siem_policy', 'provenance': 'synthetic', 'status': 'ok', 'alpha_siem': alpha_siem, 'lift_by_state': lift, 'lift_source': 'results/e6/result.json' if (Path(out_root) / 'e6' / 'result.json').exists() else 'hardcoded fallback from published E6 run', 'policies': policies_out, 'delta_siem_aug_vs_base': delta_aug, 'delta_siem_only_vs_base': delta_only, 'policy_agreement': {'base_vs_siem_augmented': agree_aug, 'base_vs_siem_only': agree_only}, 'siem_improves_return': siem_improves, 'note': 'Training reward includes SIEM shaping; evaluation always uses base reward. siem_improves_return=True means the SIEM-shaped policy achieves a higher deployment-time return than the base policy, closing the E6 loop.', 'decision_refs': ['D4', 'D11', 'D12', 'D13']}
    return finish(ctx, results, started)
if __name__ == '__main__':
    r = run(quick=True)
    print('SIEM improves return:', r['siem_improves_return'])
    for (scheme, d) in r['policies'].items():
        ev = d['eval_return']
        print(f"  {scheme:20s}: {ev['mean']:.3f} [{ev['ci_lo']:.3f}, {ev['ci_hi']:.3f}]  hijack_prevented={d['hijack_prevented_rate']:.3f}  policy={d['greedy_policy']}")
    dv = r['delta_siem_aug_vs_base']
    print(f"  Δ(aug - base): {dv['mean']:+.3f} [{dv['ci_lo']:+.3f}, {dv['ci_hi']:+.3f}]")
