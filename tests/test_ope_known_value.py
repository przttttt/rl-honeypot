import numpy as np
import pytest

from rlhoneypot.ope.estimators import doubly_robust, ips, known_value_validation, snips

def test_known_value_ips_unbiased():
    est = known_value_validation(seed=1234, n=40000)
    assert abs(est["abs_error"]) < 0.02, est

def test_known_value_ips_ci_contains_truth():
    est = known_value_validation(seed=5678, n=40000)
    assert abs(est["ips"] - 0.5) < 0.02, est


def test_doubly_robust_with_model_based_q_v():
    from rlhoneypot.agents.tabular_q import TabularQAgent
    from rlhoneypot.envs import ResponseEnv
    from rlhoneypot.mdp.rewards import reward_table
    from rlhoneypot.ope.estimators import doubly_robust, ips, snips
    from rlhoneypot.ope.model_based import policy_evaluation
    from rlhoneypot.transitions.estimate import estimate
    from rlhoneypot.opelogs.logger import log_randomised_policy, support_check

    sessions = []
    for _ in range(60):
        sessions.append([(0, 0, 1)] * 2 + [(1, 1, 7)])
    est = estimate(sessions)
    R = reward_table()
    agent = TabularQAgent(seed=0)
    rng = np.random.default_rng(0)
    env = ResponseEnv(est, seed=0, reward_tensor=R)
    for ep in range(120):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        s_prev = int(obs)
        while not (done or trunc):
            a = agent.act(s_prev)
            obs2, r, done, trunc, info = env.step(a)
            agent.update(s_prev, a, r, int(obs2), done or trunc)
            s_prev = int(obs2)

    log = log_randomised_policy([(s[0], s[1]) for s in sessions], est, agent.q, epsilon=0.3, rng=np.random.default_rng(7), reward_tensor=R, max_steps=100)
    assert support_check(log, 5, 5)["support_ok"]
    log_dict = {"states": log.states, "actions": log.actions, "propensities": log.propensities, "rewards": log.rewards, "next_states": log.next_states, "dones": log.dones, "sessions": log.sessions}
    pi = np.zeros((8, 5))
    for s in range(8):
        pi[s, agent.policy()[s]] = 1.0

    env = ResponseEnv(est, seed=5, reward_tensor=R)
    rets = []
    rng_mc = np.random.default_rng(11)
    for _ in range(2000):
        obs, _ = env.reset(seed=int(rng_mc.integers(0, 2**31 - 1)))
        done = trunc = False
        tot = 0.0
        while not (done or trunc):
            obs, r, done, trunc, info = env.step(int(agent.policy()[int(obs)]))
            tot += r
        rets.append(tot)
    truth = float(np.mean(rets))

    V, Q = policy_evaluation(est.probs, R, pi, gamma=1.0)
    assert abs(V[0] - truth) < 2.0, (V[0], truth)

    dr = doubly_robust(log_dict, pi, resamples=300, seed=0, q_values=Q, v_values=V,
                       gamma=1.0)
    ip = ips(log_dict, pi, resamples=300, seed=0)
    sn = snips(log_dict, pi, resamples=300, seed=0)
    for e in (ip, sn, dr):
        assert np.isfinite(e["estimate"]), e
        assert all(np.isfinite(c) for c in e["ci"]), e
        assert e["ess"] > 0 and e["max_weight"] >= 1.0, e
    assert abs(dr["estimate"] - V[0]) < 1.0, (dr, V[0])
    assert dr["ci"][0] - 1.0 <= truth <= dr["ci"][1] + 1.0, (dr, truth)
