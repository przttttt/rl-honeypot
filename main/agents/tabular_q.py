from __future__ import annotations

import numpy as np

from ..mdp.definitions import DEFENDER_RESPONSES, N_DEFENDER_RESPONSES, N_STATES

class TabularQAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
                 seed: int | None = None, q_init: np.ndarray | None = None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((N_STATES, N_DEFENDER_RESPONSES)) if q_init is None else np.array(q_init, dtype=float)

    def act(self, state: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_DEFENDER_RESPONSES))
        q = self.q[state]
        return int(np.flatnonzero(q == q.max())[0])

    def act_greedy(self, state: int) -> int:
        return self.act(state, greedy=True)

    def update(self, s: int, a: int, r: float, s2: int, terminal: bool) -> float:
        target = r if terminal else r + self.gamma * float(self.q[s2].max())
        td = target - float(self.q[s, a])
        self.q[s, a] += self.alpha * td
        return abs(td)

    def policy(self) -> np.ndarray:
        return np.array([int(np.flatnonzero(self.q[s] == self.q[s].max())[0]) for s in range(N_STATES)])

    def props_for(self, state: int) -> np.ndarray:
        p = np.full(N_DEFENDER_RESPONSES, self.epsilon / N_DEFENDER_RESPONSES)
        best = int(np.flatnonzero(self.q[state] == self.q[state].max())[0])
        p[best] += 1.0 - self.epsilon
        return p
