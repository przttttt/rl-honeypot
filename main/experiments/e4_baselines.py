from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from ..agents.bandits import ContextualBandit, EpsilonGreedyBandit, RandomAgent
from ..agents.supervised import SupervisedImitator
from ..agents.tabular_q import TabularQAgent
from ..cowrie.parser import iter_cowrie_events
from ..envs import gym
from ..metrics.ci import mean_with_ci, wilson_interval
from ..mdp.rewards import COMPROMISE_STATES
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..safety.shim import SafeAgentWrapper
from ..transitions.estimate import estimate

def _eval_policy(env, act_fn, n_episodes, rng, safe=None) -> dict:
    (returns, steps, succ_flags, hijack_flags) = ([], [], [], [])
    hijack_episodes = 0
    for ep in range(n_episodes):
        (obs, info) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        if safe is not None:
            safe.reset_session()
        (total, done, trunc) = (0.0, False, False)
        t = 0
        seen_hijack = False
        prevented = True
        prev_state = None
        while not (done or trunc):
            s_int = int(obs)
            a = act_fn(s_int, env)
            (obs, r, done, trunc, info) = env.step(a)
            total += r
            t += 1
            if s_int == 5:
                seen_hijack = True
            nxt = int(info['next_state'])
            if nxt in COMPROMISE_STATES and info['attacker_action'] is not None:
                if a == 0:
                    prevented = False
            if s_int == 5 and a == 0:
                prevented = False
        returns.append(total)
        steps.append(t)
        hijack_episodes += int(seen_hijack)
        hijack_flags.append(int(seen_hijack and prevented))
    out = {'return': mean_with_ci(np.array(returns), resamples=2000, seed=0), 'steps': mean_with_ci(np.array(steps), resamples=2000, seed=0), 'episodes_with_hijack': hijack_episodes, 'hijack_prevented_rate': hijack_flags and float(np.mean(hijack_flags)) or 0.0}
    return out

def _make_act(agent, contextual: bool, obs_dim: int | None=None):
    if contextual:

        def act(state, env):
            return int(agent.act(state, context=np.asarray(observation_of(env))))
        return act

    def act2(state, env):
        return int(agent.act(state))
    return act2

def observation_of(env) -> np.ndarray:
    from ..mdp.rewards import HARM_BY_NEXT_STATE
    s = env.unwrapped._s
    t = env.unwrapped._t
    onehot = np.zeros(8, dtype=np.float32)
    onehot[s] = 1.0
    ctx = np.array([t / env.unwrapped.max_steps, HARM_BY_NEXT_STATE[s] / 5.0, 1.0 if s in (5, 6, 7) else 0.0], dtype=np.float32)
    return np.concatenate([onehot, ctx])

def run(cfg=None, overrides=None, out_root: str='results', quick: bool=False) -> dict:
    started = time.time()
    ctx = make_context('e4', out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg['mdp']['hyperoffline']
    hn = cfg['mdp']['hyperonline']
    est1 = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p1.json'))))], laplace_alpha=cfg['transitions']['laplace_alpha'], min_support=cfg['transitions']['min_support_flag'])
    est2 = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p2.json.json'))))], laplace_alpha=cfg['transitions']['laplace_alpha'], min_support=cfg['transitions']['min_support_flag'])
    n_seeds = 3 if quick else cfg['eval']['num_seeds']
    n_eval = 120 if quick else cfg['eval']['eval_episodes']
    episodes = 300 if quick else cfg['mdp']['hyperoffline']['episodes']
    env1 = gym.make(cfg['env']['id_tabular'], transitions=est1, max_steps=ho['max_steps'])
    env2 = gym.make(cfg['env']['id_tabular'], transitions=est2, max_steps=ho['max_steps'])
    results: dict = {'experiment': 'e4_baselines', 'provenance': 'synthetic', 'status': 'ok', 'agents': {}}
    curves: dict[str, list] = {'cold': [], 'warm': []}
    policies_a = []
    for seed_i in range(n_seeds):
        seed = ctx.seed + seed_i * 7919
        agent = TabularQAgent(alpha=ho['alpha'], gamma=ho['gamma'], epsilon=ho['epsilon'], seed=seed)
        rng = np.random.default_rng(seed)
        curve = []
        for ep in range(episodes):
            (obs, _) = env1.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
            done = trunc = False
            ep_ret = 0.0
            s_prev = int(obs)
            while not (done or trunc):
                a = agent.act(s_prev)
                (obs2, r, done, trunc, info) = env1.step(a)
                agent.update(s_prev, a, r, int(obs2), done or trunc)
                ep_ret += r
                s_prev = int(obs2)
            curve.append(ep_ret)
        policies_a.append(agent.policy())
        results.setdefault('agents', {}).setdefault('tabular_q_phaseA', {})[f'seed{seed_i}'] = {'final50_mean': float(np.mean(curve[-50:]))}
        curves['cold'].append(curve)
    consensus = np.array([int(np.bincount([p[s] for p in policies_a], minlength=5).argmax()) for s in range(8)])
    (X, y) = ([], [])
    rng = np.random.default_rng(ctx.seed + 1)
    for ep in range(1500):
        (obs, _) = env1.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        s_prev = int(obs)
        for t in range(50):
            X.append(observation_of(env1))
            y.append(int(consensus[s_prev]))
            a = int(consensus[s_prev])
            (obs2, r, done, trunc, info) = env1.step(a)
            if done or trunc:
                break
            s_prev = int(obs2)
    X = np.array(X)
    y = np.array(y)
    imit = SupervisedImitator(model=cfg['agents']['supervised']['model'], C=cfg['agents']['supervised']['C'], seed=ctx.seed).fit(X, y)
    for mode in ('cold', 'warm'):
        for seed_i in range(n_seeds):
            seed = ctx.seed + 1000 + seed_i * 7919
            q_init = None if mode == 'cold' else _q_of(policies_a, seed_i)
            agent = TabularQAgent(alpha=hn['alpha'], gamma=hn['gamma'], epsilon=hn['epsilon'], seed=seed, q_init=q_init)
            rng = np.random.default_rng(seed)
            curve = []
            for ep in range(episodes):
                (obs, _) = env2.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
                done = trunc = False
                ep_ret = 0.0
                s_prev = int(obs)
                while not (done or trunc):
                    a = agent.act(s_prev)
                    (obs2, r, done, trunc, info) = env2.step(a)
                    agent.update(s_prev, a, r, int(obs2), done or trunc)
                    ep_ret += r
                    s_prev = int(obs2)
                curve.append(ep_ret)
            curves[mode].append(curve)
            results['agents'].setdefault('tabular_q_phaseB', {}).setdefault(mode, {})[f'seed{seed_i}'] = {'first50_mean': float(np.mean(curve[:50])), 'final50_mean': float(np.mean(curve[-50:]))}
    _agent = RandomAgent(seed=ctx.seed)
    _agent_b = EpsilonGreedyBandit(epsilon=cfg['agents']['bandit']['epsilon'], seed=ctx.seed)
    _agent_c = ContextualBandit(dim=11, epsilon=cfg['agents']['contextual_bandit']['epsilon'], lr=cfg['agents']['contextual_bandit']['lr'], seed=ctx.seed)
    for (slot, (name, agent, contextual)) in enumerate([('random', _agent, False), ('bandit_egreedy', _agent_b, False), ('contextual_bandit', _agent_c, True), ('supervised_imitation', imit, True)]):
        safe = SafeAgentWrapper(agent)
        safe.reset_session()
        if name == 'supervised_imitation':

            def raw_act(s, env, _imit=imit):
                return int(_imit.act(context=observation_of(env)))
        elif contextual:

            def raw_act(s, env, _agent=agent):
                return _agent.act(s, context=observation_of(env))
        else:

            def raw_act(s, env, _agent=agent):
                return _agent.act(s)

        def filtered_act(s, env):
            return int(safe.act(s, context=observation_of(env)) if contextual else safe.act(s))
        res = _eval_policy(env2, filtered_act, n_eval, np.random.default_rng(ctx.seed + 3000 + slot * 7919))
        res['safety_filtered'] = safe.violation_counts
        results['agents'][name] = res
    cold_first50 = [results['agents']['tabular_q_phaseB']['cold'][f'seed{i}']['first50_mean'] for i in range(n_seeds)]
    warm_first50 = [results['agents']['tabular_q_phaseB']['warm'][f'seed{i}']['first50_mean'] for i in range(n_seeds)]
    results['cold_vs_warm'] = {'cold_first50': mean_with_ci(np.array(cold_first50), resamples=2000, seed=0), 'warm_first50': mean_with_ci(np.array(warm_first50), resamples=2000, seed=0), 'delta_warm_minus_cold': mean_with_ci(np.array(warm_first50) - np.array(cold_first50), resamples=2000, seed=0), 'note': "warm start = Q initialised from phase-A policy (paper's cold-start mitigation)"}
    results['q_table_phaseA_consensus'] = consensus.tolist()
    results['decision_refs'] = ['D4', 'D9', 'D13']
    return finish(ctx, results, started)

def _q_of(policies, idx):
    q = np.zeros((8, 5))
    pol = policies[idx % len(policies)]
    q[pol, pol] = 1.0
    return q
if __name__ == '__main__':
    r = run(quick=True)
    print({k: v['return']['mean'] for (k, v) in r['agents'].items() if 'return' in v})
