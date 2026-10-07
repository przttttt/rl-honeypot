import numpy as np
import pytest

from rlhoneypot.ope.estimators import doubly_robust, ips, snips, known_value_validation
from rlhoneypot.ope.weight_diagnostics import WeightDiagnostics, summary

def _analytic_blocks():
    from rlhoneypot.transitions.estimate import estimate
    from rlhoneypot.mdp.rewards import reward_table

    sessions = []
    for _ in range(200):
        sessions.append([(0, 0, 0)])
    for _ in range(200):
        sessions.append([(1, 1, 1)])
    est = estimate(sessions, laplace_alpha=0.0)
    probs2 = est.probs[0:2, 0:2, 0:2]
    R2 = reward_table()[0:2, 0:2, 0:2]
    assert np.isclose(probs2[0, 0, 0], 1.0, atol=1e-9)
    assert np.isclose(probs2[1, 1, 1], 1.0, atol=1e-9)
    assert np.isclose(R2[0, 0, 0], 1.0, atol=1e-9)
    assert np.isclose(R2[1, 1, 1], -0.1, atol=1e-9)
    return est, probs2, R2

def _full_surface(probs2, R2):
    full_probs = np.zeros((8, 8, 8))
    full_probs[0:2, 0:2, 0:2] = probs2
    full_R = np.zeros((8, 5, 8))
    full_R[0:2, 0:2, 0:2] = R2
    return full_probs, full_R

def _target_policy():
    pi = np.zeros((8, 5))
    pi[0, 0] = 1.0  
    pi[1, 1] = 1.0  
    return pi

def _policy_evaluation(probs, R, gamma=0.95):
    from rlhoneypot.ope.model_based import policy_evaluation
    return policy_evaluation(probs, R, _target_policy(), gamma=gamma)

def _make_log(probs, R, n=400, seed=0, epsilon=0.0, max_steps=1):
    from rlhoneypot.opelogs.logger import log_randomised_policy
    from rlhoneypot.transitions.estimate import estimate
    rng = np.random.default_rng(seed)
    sessions = []
    for i in range(n):
        s = i % 2
        sessions.append([(s, s, s)])  # per-state supported action: state 0 -> a=0, state 1 -> a=1
    est = estimate(sessions, laplace_alpha=0.0)
    log = log_randomised_policy(sessions, est, _target_policy(), epsilon=epsilon,
                                 rng=rng, reward_tensor=R, max_steps=max_steps)
    return log

def _mc_truth(n_episodes=2000, seed=0):
    from rlhoneypot.transitions.estimate import estimate
    from rlhoneypot.mdp.rewards import reward_table
    from rlhoneypot.ope.model_based import policy_evaluation

    rng = np.random.default_rng(seed)
    sessions = []
    for i in range(n_episodes):
        s = i % 2
        sessions.append([(s, s, s)])
    est = estimate(sessions, laplace_alpha=0.0)
    full_probs = np.zeros((8, 8, 8))
    full_probs[0:2, 0:2, 0:2] = est.probs[0:2, 0:2, 0:2]
    full_R = np.zeros((8, 5, 8))
    full_R[0:2, 0:2, 0:2] = reward_table()[0:2, 0:2, 0:2]
    pi = _target_policy()
    V, _ = policy_evaluation(full_probs, full_R, pi, gamma=0.95)
    return V

def test_analytic_mdp_known_truth():
    _, probs2, R2 = _analytic_blocks()
    full_probs, full_R = _full_surface(probs2, R2)
    V, Q = _policy_evaluation(full_probs, full_R, gamma=0.95)

    assert np.isclose(V[0], 1.0 / (1 - 0.95), atol=1e-6)
    assert np.isclose(V[1], -0.1 / (1 - 0.95), atol=1e-6)

def test_dr_with_true_model_equals_truth():
    _, probs2, R2 = _analytic_blocks()
    full_probs, full_R = _full_surface(probs2, R2)
    V, Q = _policy_evaluation(full_probs, full_R, gamma=0.95)
    assert np.isclose(V[0], 1.0 / (1 - 0.95), atol=1e-6)
    assert np.isclose(V[1], -0.1 / (1 - 0.95), atol=1e-6)
    log = _make_log(full_probs, full_R, n=400, seed=0, epsilon=0.0, max_steps=1)
    dr = doubly_robust(log, _target_policy(), v_values=V, q_values=Q, gamma=0.95)
    assert abs(dr["estimate"] - 0.5 * (V[0] + V[1])) < 0.5, dr
    assert np.isclose(dr["estimate"], 9.0, atol=0.5), dr

def test_mc_ground_truth_matches_analytic():
    V_mc = _mc_truth(n_episodes=2000, seed=0)
    assert np.isclose(V_mc[0], 1.0 / (1 - 0.95), atol=1e-2)
    assert np.isclose(V_mc[1], -0.1 / (1 - 0.95), atol=1e-2)

def _chain_mdp():
    from rlhoneypot.transitions.estimate import TransitionEstimate
    R = np.zeros((8, 5, 8))
    R[0, 0, :] = R[1, 0, :] = R[2, 0, :] = 1.0
    probs = np.zeros((8, 8, 8))
    probs[0, 0, 1] = 1.0
    probs[1, 1, 2] = 1.0
    probs[2, 2, 7] = 1.0
    probs[7, 6, 7] = 1.0
    mask = np.zeros((8, 8), bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = mask[7, 6] = True
    est = TransitionEstimate(probs=probs, counts=np.zeros((8, 8, 8), dtype=int),support_mask=mask, observed_support=mask)
    return est, probs, R

def test_snips_is_trajectory_self_normalised():
    from rlhoneypot.opelogs.logger import log_randomised_policy
    from rlhoneypot.ope.estimators import snips

    est, probs, R = _chain_mdp()
    q = np.zeros((8, 5))
    q[:, 0] = 1.0
    sessions = [[(0, 0)] for _ in range(4000)]
    log = log_randomised_policy(sessions, est, q, epsilon=0.5, rng=np.random.default_rng(0), reward_tensor=R, max_steps=100)
    pi_det = np.zeros((8, 5))
    pi_det[:, 0] = 1.0
    ld = {"states": log.states, "actions": log.actions, "propensities": log.propensities,
          "rewards": log.rewards, "next_states": log.next_states, "dones": log.dones,
          "sessions": log.sessions}
    sn = snips(ld, pi_det, resamples=200, seed=0)
    assert np.isclose(sn["estimate"], 3.0, atol=1e-9), sn
    assert sn["max_weight"] > 1.0, sn
    assert sn["ess"] < sn["n_trajectories"], sn

def test_dr_tracks_truth_on_deterministic_target():
    from rlhoneypot.opelogs.logger import log_randomised_policy
    from rlhoneypot.ope.estimators import doubly_robust, ips
    from rlhoneypot.ope.model_based import policy_evaluation

    est, probs, R = _chain_mdp()
    q = np.zeros((8, 5))
    q[:, 1] = 1.0  
    sessions = [[(0, 0)] for _ in range(4000)]
    log = log_randomised_policy(sessions, est, q, epsilon=0.5, rng=np.random.default_rng(0), reward_tensor=R, max_steps=100)
    pi_det = np.zeros((8, 5))
    pi_det[:, 0] = 1.0
    V, Q = policy_evaluation(probs, R, pi_det, gamma=1.0)
    assert np.isclose(V[0], 3.0, atol=1e-9), V
    ld = {"states": log.states, "actions": log.actions, "propensities": log.propensities,
          "rewards": log.rewards, "next_states": log.next_states, "dones": log.dones,
          "sessions": log.sessions}
    dr = doubly_robust(ld, pi_det, resamples=200, seed=0, q_values=Q, v_values=V, gamma=1.0)
    assert np.isclose(dr["estimate"], 3.0, atol=1e-9), dr
    assert dr["ci"][0] <= 3.0 <= dr["ci"][1], dr
    ip = ips(ld, pi_det, resamples=200, seed=0)
    assert abs(ip["estimate"] - 3.0) < 0.5, ip

def test_model_based_q_keeps_terminal_reward():
    from rlhoneypot.ope.model_based import policy_evaluation

    _, probs, R = _chain_mdp()
    pi_det = np.zeros((8, 5))
    pi_det[:, 0] = 1.0
    V, Q = policy_evaluation(probs, R, pi_det, gamma=1.0)
    assert np.isclose(V[0], 3.0, atol=1e-9), V
    assert np.isclose(Q[2, 0], 1.0, atol=1e-9), Q
    assert np.isclose(Q[2, 4], R[2, 4, 7], atol=1e-9), Q

def test_logger_rolls_out_complete_episodes():
    """The offline log must contain complete episodes (D34), not session-length prefixes."""
    from rlhoneypot.opelogs.logger import log_randomised_policy

    est, probs, R = _chain_mdp()
    q = np.zeros((8, 5))
    q[:, 0] = 1.0
    sessions = [[(0, 0)] for _ in range(200)]
    log = log_randomised_policy(sessions, est, q, epsilon=0.5,
                                rng=np.random.default_rng(0), reward_tensor=R, max_steps=100)
    per_session: dict[str, int] = {}
    last_done: dict[str, bool] = {}
    for sid, d in zip(log.sessions, log.dones):
        per_session[sid] = per_session.get(sid, 0) + 1
        last_done[sid] = bool(d)
    assert set(per_session) == set(last_done)
    assert all(last_done.values()), "every logged episode must terminate"
    assert max(per_session.values()) >= 2, "episode length must exceed the 1-step corpus session"


def test_ips_known_value_validation():
    val = known_value_validation(seed=1, n=20000)
    assert abs(val["ips"] - val["truth"]) < 0.02
    assert val["abs_error"] < 0.02


def test_snips_known_value_validation():
    val = known_value_validation(seed=1, n=20000)
    assert abs(val["ips"] - val["truth"]) < 0.02


def test_weight_diagnostics_report_ess_max_weight():
    from rlhoneypot.opelogs.logger import log_randomised_policy
    from rlhoneypot.transitions.estimate import estimate

    rng = np.random.default_rng(0)
    n = 200
    sessions = [[(i % 2, 0, i % 2)] for i in range(n)]
    est = estimate(sessions, laplace_alpha=0.0)
    R_full = np.zeros((8, 5, 8))
    R_full[0, 0, 0] = 1.0
    R_full[0, 1, 1] = -0.1
    R_full[1, 0, 0] = 1.0
    R_full[1, 1, 1] = -0.1
    log = log_randomised_policy(sessions, est, _target_policy(), epsilon=0.3,
                                 rng=rng, reward_tensor=R_full, max_steps=1)
    diag = summary(log, _target_policy())
    assert diag["ess"] > 0
    assert diag["max_weight"] >= 1.0
    assert diag["n_steps"] == n
    assert not diag["high_variance"], diag


def test_termination_contract_shared_by_env_logger_est():
    from rlhoneypot.envs.response_env import ResponseEnv

    _, probs2, R2 = _analytic_blocks()
    full_probs, full_R = _full_surface(probs2, R2)
    V, Q = _policy_evaluation(full_probs, full_R, gamma=0.95)

    from rlhoneypot.opelogs.logger import log_randomised_policy
    from rlhoneypot.transitions.estimate import estimate

    rng = np.random.default_rng(0)
    n = 200
    sessions = []
    for i in range(n):

        sessions.append([(i % 2, i % 2, i % 2)])
    est = estimate(sessions, laplace_alpha=0.0)
    R_full = np.zeros((8, 5, 8))
    R_full[0, 0, 0] = 1.0
    R_full[0, 1, 1] = -0.1
    R_full[1, 0, 0] = 1.0
    R_full[1, 1, 1] = -0.1
    log = log_randomised_policy(sessions, est, _target_policy(), epsilon=0.0,
                                 rng=rng, reward_tensor=R_full, max_steps=1)
    do = np.asarray(log.dones)
    assert not do.any(), "shared termination contract broken: S7 reachable in 2-state block"
    assert np.array_equal(np.asarray(log.states), np.asarray(log.next_states)), \
        "states must equal next_states in a recurrent 2-state block"
