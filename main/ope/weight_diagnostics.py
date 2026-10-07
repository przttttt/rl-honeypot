from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class WeightDiagnostics:
    ess: float
    max_weight: float
    topk_share: float
    n_steps: int

    @classmethod
    def from_log(cls, log: dict, pi_probs: np.ndarray) -> 'WeightDiagnostics':
        if hasattr(log, 'states'):
            log_states = np.asarray(log.states)
            log_actions = np.asarray(log.actions)
            log_props = np.asarray(log.propensities, dtype=float)
        else:
            log_states = np.asarray(log['states'])
            log_actions = np.asarray(log['actions'])
            log_props = np.asarray(log['propensities'], dtype=float)
        mu = np.maximum(log_props, 1e-12)
        pi = pi_probs[log_states, log_actions]
        rho = pi / mu
        ess = float(rho.sum() ** 2 / np.maximum((rho ** 2).sum(), 1e-12))
        max_weight = float(np.max(rho))
        k = max(1, int(0.05 * len(rho)))
        topk = np.partition(rho, -k)[-k:]
        topk_share = float(topk.sum() / np.maximum(rho.sum(), 1e-12))
        return cls(ess=ess, max_weight=max_weight, topk_share=topk_share, n_steps=len(rho))

def summary(log: dict, pi_probs: np.ndarray) -> dict:
    d = WeightDiagnostics.from_log(log, pi_probs)
    return {'ess': d.ess, 'max_weight': d.max_weight, 'topk_share': d.topk_share, 'n_steps': d.n_steps, 'high_variance': d.ess < 0.5 * d.n_steps}
