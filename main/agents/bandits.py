from __future__ import annotations
import numpy as np
from ..mdp.definitions import N_DEFENDER_RESPONSES

class RandomAgent:

    def __init__(self, seed: int | None=None):
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, context: np.ndarray | None=None) -> int:
        return int(self.rng.integers(0, N_DEFENDER_RESPONSES))

    def act_greedy(self, state: int, context: np.ndarray | None=None) -> int:
        return self.act(state, context)

    def update(self, *args, **kwargs) -> None:
        return None

class EpsilonGreedyBandit:

    def __init__(self, epsilon: float=0.1, seed: int | None=None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.means = np.zeros(N_DEFENDER_RESPONSES)
        self.counts = np.zeros(N_DEFENDER_RESPONSES)

    def act(self, state: int=0, context: np.ndarray | None=None) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_DEFENDER_RESPONSES))
        return int(np.argmax(self.means))

    def act_greedy(self, state: int=0, context: np.ndarray | None=None) -> int:
        return int(np.argmax(self.means))

    def update(self, reward: float, action: int | None=None, *args, **kwargs) -> None:
        if action is None:
            return
        self.counts[action] += 1
        n = self.counts[action]
        self.means[action] += (reward - self.means[action]) / n

class ContextualBandit:

    def __init__(self, dim: int, epsilon: float=0.1, lr: float=0.05, l2: float=0.01, seed: int | None=None):
        self.epsilon = epsilon
        self.lr = lr
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((N_DEFENDER_RESPONSES, dim))

    def _features(self, context: np.ndarray) -> np.ndarray:
        return np.asarray(context, dtype=float)

    def act(self, state: int=0, context: np.ndarray | None=None) -> int:
        x = self._features(context if context is not None else np.zeros(self.W.shape[1]))
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_DEFENDER_RESPONSES))
        return int(np.argmax(self.W @ x))

    def act_greedy(self, state: int=0, context: np.ndarray | None=None) -> int:
        x = self._features(context if context is not None else np.zeros(self.W.shape[1]))
        return int(np.argmax(self.W @ x))

    def update(self, reward: float, action: int | None=None, context: np.ndarray | None=None, *args, **kwargs) -> None:
        if action is None or context is None:
            return
        x = self._features(context)
        pred = float(self.W[action] @ x)
        err = reward - pred
        self.W[action] += self.lr * (err * x - self.l2 * self.W[action])
