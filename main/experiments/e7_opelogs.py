from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from ..agents.tabular_q import TabularQAgent
from ..cowrie.parser import iter_cowrie_events
from ..mdp.rewards import reward_table
from ..opelogs.logger import log_randomised_policy, support_check
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..transitions.estimate import estimate

def run(cfg=None, overrides=None, out_root: str='results', quick: bool=False) -> dict:
    started = time.time()
    ctx = make_context('e7', out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg['mdp']['hyperoffline']
    est = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p2.json.json'))))], laplace_alpha=cfg['transitions']['laplace_alpha'], min_support=cfg['transitions']['min_support_flag'])
    agent = TabularQAgent(alpha=ho['alpha'], gamma=ho['gamma'], epsilon=ho['epsilon'], seed=ctx.seed)
    env_eps = cfg['mdp']['hyperoffline']['episodes']
    from ..envs import gym
    env = gym.make(cfg['env']['id_tabular'], transitions=est, max_steps=ho['max_steps'])
    rng_tr = np.random.default_rng(ctx.seed)
    for ep in range(300 if quick else env_eps):
        (obs, _) = env.reset(seed=int(rng_tr.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        s_prev = int(obs)
        while not (done or trunc):
            a = agent.act(s_prev)
            (obs2, r, done, trunc, info) = env.step(a)
            agent.update(s_prev, a, r, int(obs2), done or trunc)
            s_prev = int(obs2)
    labelled = [ls for ls in label_events(list(iter_cowrie_events(Path('data/raw/p2.json.json')))) if ls.transitions]
    n_sessions = 3000 if quick else min(60000, len(labelled))
    eps = cfg['ope']['logging_epsilon']
    rng = np.random.default_rng(ctx.seed + 7)
    log = log_randomised_policy([ls.states_actions for ls in labelled[:n_sessions]], est, agent.q, epsilon=eps, rng=rng, reward_tensor=reward_table(), max_steps=ho['max_steps'])
    support = support_check(log, n_actions=5, min_actions_covered=5)
    if not support['support_ok']:
        support['status'] = 'FAIL: logging policy lacks action support; OPE would be biased'
    from collections import defaultdict
    per_session = defaultdict(float)
    for (s, r) in zip(log.sessions, log.rewards):
        per_session[s] += r
    vals = np.array(list(per_session.values()))
    out = {'experiment': 'e7_opelogs', 'provenance': 'synthetic', 'status': 'ok', 'n_sessions_logged': len(set(log.sessions)), 'n_steps_logged': len(log.states), 'logging_epsilon': eps, 'support_check': support, 'mean_session_reward': float(vals.mean()) if vals.size else None, 'session_reward_std': float(vals.std()) if vals.size else None, 'q_table': agent.q.tolist(), 'greedy_policy': agent.policy().tolist(), 'records_sample': log.to_records()[:5], 'decision_refs': ['D16']}
    return finish(ctx, out, started)
if __name__ == '__main__':
    r = run(quick=True)
    print(r['support_check'])
