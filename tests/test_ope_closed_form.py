import numpy as np
import pytest

from rlhoneypot.ope.estimators import ips, snips, doubly_robust

def _single_step_log(n_allow=10000, n_term=10000):
    N = n_allow + n_term
    states = np.zeros(N, int)
    actions = np.zeros(N, int)
    actions[n_allow:] = 1
    rewards = np.zeros(N)
    rewards[:n_allow] = 1.0
    dones = np.ones(N, bool)
    return {
        "states": states.tolist(),
        "actions": actions.tolist(),
        "propensities": [0.5] * N,
        "rewards": rewards.tolist(),
        "next_states": [0] * N,
        "dones": dones.tolist(),
        "sessions": [f"t{i}" for i in range(N)],
    }

def test_closed_form_single_step():
    log = _single_step_log(n_allow=10000, n_term=10000)
    pi = np.array([[1.0, 0.0]])          
    truth = 1.0
    e = ips(log, pi, resamples=200, seed=1)
    assert abs(e["estimate"] - truth) < 1e-9, (e["estimate"], truth)
    sn = snips(log, pi, resamples=200, seed=1)
    assert abs(sn["estimate"] - truth) < 1e-9, (sn["estimate"], truth)
    dr = doubly_robust(log, pi, resamples=200, seed=1, q_values=np.array([[1.0, 0.0]]), v_values=np.array([1.0]))
    assert abs(dr["estimate"] - truth) < 1e-9, (dr["estimate"], truth)
