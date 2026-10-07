from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from ..metrics.ci import auroc_ap, precision_at_k

def alert_features(alerts: list[dict]) -> np.ndarray:
    import datetime as dt
    X = np.zeros((len(alerts), 4), dtype=np.float32)
    for (i, a) in enumerate(alerts):
        X[i, 0] = float(a.get('rule', {}).get('level', 0))
        ts = a.get('@timestamp') or ''
        try:
            X[i, 1] = dt.datetime.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S').hour / 23.0
        except Exception:
            X[i, 1] = 0.0
        X[i, 2] = float(a.get('rlhoneypot', {}).get('mdp_state', 0))
        X[i, 3] = 1.0 if a.get('rule', {}).get('name') == 'cowrie_noise_alert' else 0.0
    return X

def baseline_scores(kind: str, X: np.ndarray, alerts: list[dict]) -> np.ndarray:
    if kind == 'recency':
        ts = [a.get('@timestamp') or '' for a in alerts]
        order = np.argsort(np.argsort(np.array(ts)))
        return order.astype(float) / max(1, len(ts) - 1)
    if kind == 'severity':
        return X[:, 0].astype(float)
    raise ValueError(kind)

class SupervisedRanker:

    def __init__(self, model: str='logistic', seed: int=0):
        if model == 'logistic':
            self.clf = LogisticRegression(max_iter=2000, random_state=seed)
        elif model == 'gradient_boosting':
            self.clf = GradientBoostingClassifier(random_state=seed)
        else:
            raise ValueError(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupervisedRanker':
        if len(np.unique(y)) < 2:

            class _Const:

                def __init__(self, c):
                    self.c = c

                def predict_proba(self, X):
                    p = np.full((len(X), 2), 0.5)
                    p[:, 1] = self.c
                    return p
            self.clf = _Const(float(np.mean(y) if len(y) else 0.5))
        else:
            self.clf.fit(X, y)
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        proba = self.clf.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.full(len(X), 0.5)

def evaluate_ranking(y_true: np.ndarray, scores: np.ndarray, ks: list[int]) -> dict:
    out = auroc_ap(y_true, scores)
    for k in ks:
        out[f'precision_at_{k}'] = precision_at_k(y_true, scores, k)
    return out

def run(cfg=None, alerts=None, y=None, ks=(10, 50, 100)):
    X = alert_features(alerts)
    if y is None:
        from rlhoneypot.experiments.e6_siem import run as e6
        y = np.zeros(len(X))
    rankings = {}
    for kind in ('recency', 'severity'):
        scores = baseline_scores(kind, X, alerts)
        rankings[kind] = evaluate_ranking(y, scores, ks)
    n_train = int(0.7 * len(X))
    rng2 = np.random.default_rng(0)
    order = rng2.permutation(len(X))
    (tr, te) = (order[:n_train], order[n_train:])
    for model in ('logistic', 'gradient_boosting'):
        rk = SupervisedRanker(model=model, seed=0).fit(X[tr], y[tr])
        rankings[model] = evaluate_ranking(y[te], rk.scores(X[te]), ks)
        rankings[model]['n_test'] = int(len(te))
    return rankings
