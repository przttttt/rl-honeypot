from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ..mdp.definitions import N_STATES, action_support

@dataclass
class TransitionEstimate:
    probs: np.ndarray             
    counts: np.ndarray                      
    support_mask: np.ndarray              
    observed_support: np.ndarray            
    low_confidence_rows: list[tuple[int, int, int]] = field(default_factory=list)
    n_sessions: int = 0
    n_transitions: int = 0
    unmapped_commands: int = 0
    out_of_support_counts: int = 0
    laplace_alpha: float = 1.0

    def row(self, s: int, a: int) -> np.ndarray:
        return self.probs[s, a]

def label_session(states_actions: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    if not states_actions:
        return out
    cur_state, cur_action = states_actions[0]
    for state, action in states_actions[1:]:
        if state != cur_state:  
            out.append((cur_state, cur_action, state))
            cur_state, cur_action = state, action
    return out

def estimate(
    session_transitions: list[list[tuple[int, int, int]]],
    laplace_alpha: float = 1.0,
    min_support: int = 30,
) -> TransitionEstimate:
    counts = np.zeros((N_STATES, N_STATES, N_STATES), dtype=np.int64)
    n_out_of_support = 0
    n_sessions = len(session_transitions)
    n_transitions = 0
    for session in session_transitions:
        for s, a, s2 in session:
            if a in action_support(s):
                counts[s, a, s2] += 1
            else:
                n_out_of_support += 1

    probs = np.zeros_like(counts, dtype=np.float64)
    support_mask = np.zeros((N_STATES, N_STATES), dtype=bool)
    observed_support = np.zeros((N_STATES, N_STATES), dtype=bool)
    low_conf: list[tuple[int, int, int]] = []

    for s in range(N_STATES):
        for a in action_support(s):
            row = counts[s, a]
            total = int(row.sum())
            support_mask[s, a] = True
            if total > 0:
                observed_support[s, a] = True
            if total < min_support:
                low_conf.append((s, a, total))
            probs[s, a] = (row + laplace_alpha) / (total + laplace_alpha * N_STATES)

    for s in range(N_STATES):
        for a in action_support(s):
            if counts[s, a].sum() == 0:
                probs[s, a] = 1.0 / N_STATES

    n_transitions = int(counts.sum())

    return TransitionEstimate(
        probs=probs,
        counts=counts,
        support_mask=support_mask,
        observed_support=observed_support,
        low_confidence_rows=low_conf,
        n_sessions=n_sessions,
        n_transitions=n_transitions,
        out_of_support_counts=n_out_of_support,
        laplace_alpha=laplace_alpha,
    )

def heldout_logloss(train: TransitionEstimate, eval_transitions: list[list[tuple[int, int, int]]]) -> dict:
    n = 0
    nll_model = 0.0
    nll_uniform = 0.0
    for session in eval_transitions:
        for s, a, s2 in session:
            p = float(train.probs[s, a, s2]) if a < N_STATES else 1.0 / N_STATES
            nll_model += -np.log(max(p, 1e-12))
            nll_uniform += -np.log(1.0 / N_STATES)
            n += 1
    if n == 0:
        return {"n": 0, "nll_model": None, "nll_uniform": None, "improvement_nats": None}
    return {
        "n": n,
        "nll_model": nll_model / n,
        "nll_uniform": nll_uniform / n,
        "improvement_nats": (nll_uniform - nll_model) / n,
    }
