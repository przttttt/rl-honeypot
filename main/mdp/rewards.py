from __future__ import annotations

import numpy as np

from .definitions import DEFENDER_RESPONSES, RESPONSE_INDEX

HARM_BY_NEXT_STATE = np.array([1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=float)

COMPROMISE_STATES = (5, 6, 7)

_OUTCOME = {
    "allow":      {"benign": +1.0, "compromise": -1.0},
    "block":      {"benign": 0.0,  "compromise": +1.0},
    "delay":      {"benign": +0.5, "compromise": +0.5},
    "insult":     {"benign": 0.0,  "compromise": +0.5},
    "terminate":  {"benign": -0.5, "compromise": +1.0},
}

COST = {
    "allow": 0.0,
    "block": 0.1,     # stops intelligence flow
    "delay": 0.1,     # small, buys observation time (paper: delay extends engagement)
    "insult": 0.5,    # detectability risk (paper §7.1.5)
    "terminate": 1.0, # ends episode: maximal intelligence loss
}

def harm(next_state: int) -> float:
    return HARM_BY_NEXT_STATE[next_state]

def reward(state: int, response: int, next_state: int) -> float:
    
    comp = next_state in COMPROMISE_STATES
    key = "compromise" if comp else "benign"
    resp = DEFENDER_RESPONSES[response]
    oc = _OUTCOME[resp][key]
    return oc * HARM_BY_NEXT_STATE[next_state] - COST[resp]

def reward_table() -> np.ndarray:
    t = np.zeros((8, len(DEFENDER_RESPONSES), 8), dtype=np.float64)
    for s in range(8):
        for a in range(len(DEFENDER_RESPONSES)):
            for s2 in range(8):
                t[s, a, s2] = reward(s, a, s2)
    return t

def severity_of_transition(next_state: int) -> str:
    h = HARM_BY_NEXT_STATE[next_state]
    return {5.0: "high(+5)", 2.0: "medium(+2)", 1.0: "low(+1)"}[h]
