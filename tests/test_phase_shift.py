import numpy as np
import pytest

from rlhoneypot.envs import ResponseEnv
from rlhoneypot.experiments.e9_phase_shift import (
    _escalation_target,
    _online_first50,
    _train_offline,
    mean_tv,
    shifted_estimate,
)
from rlhoneypot.transitions.estimate import estimate

HP = {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1}

def _tiny_est():
    sessions = [[(0, 0, 1)] * 3 + [(1, 1, 7)] for _ in range(50)]
    sessions += [[(0, 0, 2)] + [(2, 2, 4)] for _ in range(20)]
    return estimate(sessions)

def test_escalation_target_is_one_hot_escalation():
    tgt = _escalation_target(8)
    assert tgt.shape == (8, 8, 8)
    for s in range(8):
        # every column is one-hot at min(s+1, 7), independent of the row (attacker action)
        assert np.allclose(tgt[s].sum(axis=1), 1.0)
        assert tgt[s, :, min(s + 1, 7)].all()
    assert tgt[7, 0, 7] == 1.0  # S7 stays

def test_shift_level_zero_returns_source_dynamics():
    est = _tiny_est()
    out = shifted_estimate(est, 0.0)
    mask = est.support_mask.astype(bool)
    # supported rows are unchanged; unsupported rows stay zero
    assert np.allclose(out.probs[mask], est.probs[mask])
    assert not out.probs[~mask].any()

def test_shift_level_one_is_pure_escalation_on_supported_rows():
    est = _tiny_est()
    out = shifted_estimate(est, 1.0)
    for s in range(out.probs.shape[0]):
        for a in range(out.probs.shape[1]):
            if est.support_mask[s, a]:
                assert np.allclose(out.probs[s, a], _escalation_target(8)[s, a])

def test_shifted_rows_are_valid_distributions():
    est = _tiny_est()
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = shifted_estimate(est, level)
        mask = est.support_mask.astype(bool)
        assert np.allclose(out.probs[mask].sum(axis=1), 1.0)
        assert (out.probs >= -1e-12).all()

def test_shift_rejects_out_of_range_level():
    est = _tiny_est()
    with pytest.raises(ValueError):
        shifted_estimate(est, 1.5)
    with pytest.raises(ValueError):
        shifted_estimate(est, -0.1)

def test_mean_tv_zero_for_identical_and_bounded():
    est = _tiny_est()
    assert mean_tv(est, est) == pytest.approx(0.0)
    full = mean_tv(est, shifted_estimate(est, 1.0))
    assert 0.0 < full <= 1.0
    # TV grows with the shift level (linear in the blend)
    assert mean_tv(est, shifted_estimate(est, 0.5)) < full

def test_offline_then_online_loop_runs_on_tiny_mdp():
    est = _tiny_est()
    env = ResponseEnv(est, max_steps=6, seed=0)
    agent = _train_offline(env, episodes=5, seed=1, hp=HP)
    assert agent.q.shape == (8, 5)
    q_init = np.zeros((8, 5))
    pol = agent.policy()
    q_init[pol, pol] = 1.0
    cold = _online_first50(env, None, episodes=5, seed=2, hp=HP)
    warm = _online_first50(env, q_init, episodes=5, seed=2, hp=HP)
    assert np.isfinite(cold) and np.isfinite(warm)
