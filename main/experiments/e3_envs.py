from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from ..agents.bandits import RandomAgent
from ..envs import gym
from ..repro.harness import finish, make_context
from ..transitions.estimate import estimate
from ..cowrie.parser import iter_cowrie_events
from ..pipeline.label import label_events
from ..safety.shim import SafeAgentWrapper

def run(cfg=None, overrides=None, out_root: str='results') -> dict:
    started = time.time()
    ctx = make_context('e3', out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg['mdp']['hyperoffline']
    events = list(iter_cowrie_events(Path('data/raw/p1.json')))
    labelled = label_events(events)
    est = estimate([ls.transitions for ls in labelled], laplace_alpha=cfg['transitions']['laplace_alpha'], min_support=cfg['transitions']['min_support_flag'])
    ho = cfg['mdp']['hyperoffline']
    env = gym.make(cfg['env']['id_tabular'], transitions=est, max_steps=ho['max_steps'], seed=42)
    rng = np.random.default_rng(ctx.seed)
    n_ep = 300
    (returns, steps, causes) = ([], [], {'terminal': 0, 'terminate_resp': 0, 'truncated': 0})
    state_occ = np.zeros(8)
    agent = RandomAgent(seed=ctx.seed)
    safe = SafeAgentWrapper(agent)
    for ep in range(n_ep):
        (obs, info) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        safe.reset_session()
        (total, done, trunc) = (0.0, False, False)
        t = 0
        while not (done or trunc):
            a = safe.act(int(obs))
            (obs, r, done, trunc, info) = env.step(a)
            total += r
            t += 1
            state_occ[int(obs)] += 1
        returns.append(total)
        steps.append(t)
        if trunc:
            causes['truncated'] += 1
        elif info['next_state'] == 7:
            causes['terminal'] += 1
        else:
            causes['terminate_resp'] += 1
    env1 = gym.make(cfg['env']['id_tabular'], transitions=est, max_steps=ho['max_steps'], seed=42)
    env2 = gym.make(cfg['env']['id_tabular'], transitions=est, max_steps=ho['max_steps'], seed=42)
    (t1, t2) = ([], [])
    for (env_, tl) in ((env1, t1), (env2, t2)):
        rng = np.random.default_rng(42)
        (obs, _) = env_.reset()
        d = tr = False
        while not (d or tr):
            a = int(rng.integers(0, env_.action_space.n))
            (obs, r, d, tr, _) = env_.step(a)
            tl.append((int(obs), float(r)))
    result = {'experiment': 'e3_envs', 'provenance': 'synthetic', 'status': 'ok', 'random_policy': {'mean_return': float(np.mean(returns)), 'std_return': float(np.std(returns)), 'mean_steps': float(np.mean(steps)), 'episodes': n_ep, 'termination_causes': causes, 'state_occupancy_normalised': (state_occ / state_occ.sum()).tolist()}, 'determinism_check': {'episodes_equal': t1 == t2, 'len': len(t1)}, 'floor_note': 'learning agents in E4/E5 must beat this return to matter', 'decision_refs': ['D13', 'D17']}
    return finish(ctx, result, started)
if __name__ == '__main__':
    print(run()['random_policy']['mean_return'])
