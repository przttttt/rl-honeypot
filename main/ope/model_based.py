from __future__ import annotations
import numpy as np
from ..mdp.definitions import DEFENDER_RESPONSES, N_STATES, action_support
from ..transitions.estimate import TransitionEstimate

def policy_evaluation(probs: np.ndarray, reward_tensor: np.ndarray, policy: np.ndarray, gamma: float=0.95) -> tuple[np.ndarray, np.ndarray]:
    n_def = reward_tensor.shape[1]
    if policy.ndim == 1:
        pi_def = np.zeros((N_STATES, n_def))
        for s in range(N_STATES):
            pi_def[s, policy[s]] = 1.0
    else:
        pi_def = policy
    V = np.zeros(N_STATES)
    Q = np.zeros((N_STATES, n_def))
    a_att = np.array([action_support(s)[0] for s in range(N_STATES)])
    R = reward_tensor
    P = probs
    base_continue = np.zeros(N_STATES)
    base_continue[7] = 1.0
    Rbar = np.zeros(N_STATES)
    for s in range(N_STATES):
        att = a_att[s]
        P_s_att = P[s, att, :]
        R_s = R[s]
        Rbar[s] = float(np.sum(pi_def[s][:, None] * P_s_att[None, :] * R_s, axis=(0, 1)))
    M = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        p_s = P[s, a_att[s], :]
        for a in range(n_def):
            action_continue = base_continue.copy()
            if DEFENDER_RESPONSES[a] == 'terminate':
                action_continue[:] = 1.0
            M[s] += pi_def[s, a] * p_s * (1.0 - action_continue)
    A_mat = np.eye(N_STATES) - gamma * M
    b = Rbar
    V = np.linalg.solve(A_mat, b)
    V[7] = 0.0
    for s in range(N_STATES):
        att = a_att[s]
        P_s_att = P[s, att, :]
        for a in range(n_def):
            action_continue = base_continue.copy()
            if DEFENDER_RESPONSES[a] == 'terminate':
                action_continue[:] = 1.0
            cont = P_s_att * (1.0 - action_continue)
            Q[s, a] = float(np.sum(P_s_att * R[s, a, :])) + gamma * float(cont @ V)
    return (V, Q)
