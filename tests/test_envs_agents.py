import numpy as np
import pytest

from rlhoneypot.agents.bandits import ContextualBandit, EpsilonGreedyBandit, RandomAgent
from rlhoneypot.agents.tabular_q import TabularQAgent
from rlhoneypot.envs import ResponseEnv
from rlhoneypot.safety.shim import SafeAgentWrapper
from rlhoneypot.transitions.estimate import estimate

@pytest.fixture
def est():

    sessions = []
    for _ in range(50):
        sessions.append([(0, 0, 1)] * 3 + [(1, 1, 7)])
    for _ in range(20):
        sessions.append([(0, 0, 2)] + [(2, 2, 4)])
    for _ in range(20):
        sessions.append([(3, 2, 4), (4, 4, 5), (5, 5, 0)])
    return estimate(sessions)

def test_gym_api_contract(est):
    env = ResponseEnv(est, max_steps=10, seed=1)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    done = trunc = False
    steps = 0
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        steps += 1
        assert env.observation_space.contains(obs)
        assert isinstance(r, float)
    assert steps <= 10

def test_env_same_seed_same_trajectory(est):
    t1, t2 = [], []
    for tl in (t1, t2):
        env = ResponseEnv(est, max_steps=20, seed=123)
        obs, _ = env.reset(seed=123)
        done = trunc = False
        while not (done or trunc):
            a = int(np.argmax(env.unwrapped.T.probs[obs])) % 5
            obs, r, done, trunc, _ = env.step(a)
            tl.append((int(obs), r))
    assert t1 == t2

def test_tabular_q_learns_to_block_compromise(est):
    env = ResponseEnv(est, max_steps=30, seed=0)
    agent = TabularQAgent(alpha=0.2, gamma=0.9, epsilon=0.15, seed=3)
    rng = np.random.default_rng(3)
    rets = []
    for ep in range(150):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        s_prev = int(obs)
        tot = 0.0
        while not (done or trunc):
            a = agent.act(s_prev)
            obs2, r, done, trunc, info = env.step(a)
            agent.update(s_prev, a, r, int(obs2), done or trunc)
            tot += r
            s_prev = int(obs2)
        rets.append(tot)
    pol = agent.policy()
    assert pol[5] != 0, "learned policy allows in visited compromise state S5"
    q = agent.q
    assert q[5, 0] < q[5].max(), "allow in S5 should not be the best action"

def test_epsilon_greedy_props_sum_to_one():
    agent = TabularQAgent(seed=0)
    p = agent.props_for(2)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p > 0).all()

def test_bandit_updates_and_safe_wrapper():
    b = EpsilonGreedyBandit(epsilon=0.0, seed=0)
    b.update(1.0, 1)
    b.update(-1.0, 1)
    b.update(0.5, 0)
    assert b.act_greedy() == 0
    cb = ContextualBandit(dim=11, epsilon=0.0, seed=0)
    x = np.zeros(11); x[5] = 1
    cb.W[2] += 1.0
    assert cb.act_greedy(context=x) == 2
    w = SafeAgentWrapper(RandomAgent(seed=1))
    for s in range(8):
        a = w.act(s)
        assert 0 <= a < 5
