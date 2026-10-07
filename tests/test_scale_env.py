from __future__ import annotations

import numpy as np

from rlhoneypot.envs.scaled_env import (
    FEATURE_DIM,
    ScaledResponseEnv,
    ScaledTabularQ,
    build_scaled,
)
from rlhoneypot.mdp.definitions import N_DEFENDER_RESPONSES, action_support
from rlhoneypot.transitions.estimate import TransitionEstimate

def _chain_estimate() -> TransitionEstimate:
    """Deterministic attack chain S0→S1→…→S7 (no corpus needed)."""
    probs = np.zeros((8, 7, 8), dtype=float)
    counts = np.zeros((8, 7, 8), dtype=np.int64)
    support = np.zeros((8, 7), dtype=bool)
    for s in range(8):
        for a in action_support(s):
            support[s, a] = True
            nxt = min(s + 1, 7)
            probs[s, a, nxt] = 1.0
            counts[s, a, nxt] = 10
    return TransitionEstimate(probs=probs, counts=counts, support_mask=support, observed_support=support.copy())

def test_scaled_rows_are_distributions() -> None:
    est = _chain_estimate()
    for k in (1, 4, 16):
        sc = build_scaled(est, k)
        n = 8 * k
        assert sc["n_states"] == n
        assert sc["probs"].shape == (n, 7, n)
        for s in range(n):
            supported = np.flatnonzero(sc["support"][s])
            assert supported.size >= 1
            for a in supported:
                assert np.isclose(sc["probs"][s, a].sum(), 1.0)

def test_scaled_expectation_ignores_sub_state_when_shared() -> None:
    est = _chain_estimate()
    k = 4
    sc = build_scaled(est, k, structure="shared")
    for p in range(8):
        for a in range(N_DEFENDER_RESPONSES):
            vals = []
            for j in range(k):
                s = p * k + j
                exp_r = float(np.sum(sc["probs"][s, np.flatnonzero(sc["support"][s])[0]] * sc["reward"][s, a]))
                vals.append(exp_r)
            assert np.allclose(vals, vals[0])

def test_randomised_regime_breaks_the_shared_structure() -> None:
    est = _chain_estimate()
    k = 4
    shared = build_scaled(est, k, structure="shared", seed=0)
    rand = build_scaled(est, k, structure="randomised", seed=0)
    p, a = 3, 0
    shared_vals = [shared["reward"][p * k + j, a, p * k] for j in range(k)]
    rand_vals = [rand["reward"][p * k + j, a, p * k] for j in range(k)]
    assert np.allclose(shared_vals, shared_vals[0])
    assert not np.allclose(rand_vals, rand_vals[0])

def test_observation_modes_and_shapes() -> None:
    est = _chain_estimate()
    sc = build_scaled(est, 4)
    env_d = ScaledResponseEnv(sc, obs_mode="discrete", seed=0)
    env_f = ScaledResponseEnv(sc, obs_mode="features", seed=0)
    obs_d, _ = env_d.reset(seed=0)
    obs_f, _ = env_f.reset(seed=0)
    assert isinstance(obs_d, (int, np.integer))
    assert obs_f.shape == (FEATURE_DIM,)
    assert np.isclose(obs_f[:8].sum(), 1.0)
    assert np.all(obs_f >= 0.0) and np.all(obs_f <= 1.0)
    for env in (env_d, env_f):
        obs, r, term, trunc, info = env.step(0)
        assert env.observation_space.contains(obs)
        assert "next_state" in info

def test_scaled_env_determinism() -> None:
    est = _chain_estimate()
    sc = build_scaled(est, 4)
    traces = []
    for _ in range(2):
        env = ScaledResponseEnv(sc, obs_mode="discrete", reward_noise_sigma=1.0, seed=7)
        obs, _ = env.reset(seed=7)
        trace = []
        done = trunc = False
        while not (done or trunc):
            obs, r, done, trunc, _ = env.step(1)
            trace.append((int(obs), round(float(r), 12)))
        traces.append(trace)
    assert traces[0] == traces[1]

def test_scaled_tabular_beats_random() -> None:
    est = _chain_estimate()
    sc = build_scaled(est, 2, structure="shared")
    env = ScaledResponseEnv(sc, max_steps=50, obs_mode="discrete", seed=3)
    agent = ScaledTabularQ(n_states=sc["n_states"], alpha=0.2, gamma=0.9, epsilon=0.1, seed=3)
    rng = np.random.default_rng(3)
    for _ in range(300):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        while not (done or trunc):
            a = agent.act(int(obs))
            obs2, r, done, trunc, _ = env.step(a)
            agent.update(int(obs), a, r, int(obs2), done or trunc)
            obs = obs2

    def _mean_return(policy_greedy: bool) -> float:
        env2 = ScaledResponseEnv(sc, max_steps=50, obs_mode="discrete", seed=11)
        rng2 = np.random.default_rng(11)
        rets = []
        for _ in range(60):
            obs, _ = env2.reset(seed=int(rng2.integers(0, 2**31 - 1)))
            done = trunc = False
            total = 0.0
            while not (done or trunc):
                a = (agent.act_greedy(int(obs)) if policy_greedy
                     else int(rng2.integers(0, N_DEFENDER_RESPONSES)))
                obs, r, done, trunc, _ = env2.step(a)
                total += r
            rets.append(total)
        return float(np.mean(rets))

    assert _mean_return(True) > _mean_return(False)
