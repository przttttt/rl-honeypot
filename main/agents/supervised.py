from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class SupervisedImitator:

    def __init__(self, model: str='logistic', C: float=1.0, seed: int=0):
        self.kind = model
        self.seed = seed
        if model == 'logistic':
            self.clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        elif model == 'gradient_boosting':
            self.clf = GradientBoostingClassifier(random_state=seed)
        else:
            raise ValueError(f'unknown supervised model: {model}')
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupervisedImitator':
        if len(np.unique(y)) < 2:
            self.clf = _ConstantClassifier(int(y[0]) if len(y) else 0)
        else:
            self.clf.fit(X, y)
        self._fitted = True
        return self

    def act(self, state: int=0, context: np.ndarray | None=None) -> int:
        assert self._fitted, 'call fit() first'
        x = np.asarray(context if context is not None else np.zeros(1)).reshape(1, -1)
        return int(self.clf.predict(x)[0])

    def act_greedy(self, state: int=0, context: np.ndarray | None=None) -> int:
        return self.act(state, context)

class _ConstantClassifier:

    def __init__(self, c: int):
        self.c = c

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.c)
