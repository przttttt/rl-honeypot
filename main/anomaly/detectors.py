from __future__ import annotations

import numpy as np

from ..transitions.estimate import TransitionEstimate

def session_features(labelled_sessions, max_states: int = 8) -> np.ndarray:
    
    X = np.zeros((len(labelled_sessions), max_states + 3), dtype=np.float32)
    for i, ls in enumerate(labelled_sessions):
        for s, _ in ls.states_actions:
            X[i, s] += 1.0
        X[i, max_states] = len(ls.states_actions)
        X[i, max_states + 1] = ls.n_unmapped
        X[i, max_states + 2] = ls.n_commands
        n = max(1, len(ls.states_actions))
        X[i, :max_states] /= n
    return X

class TrajectoryLikelihood:
    def __init__(self, transitions: TransitionEstimate, uniform_smooth: float = 1e-3):
        self.T = transitions
        self.eps = uniform_smooth

    def scores(self, labelled_sessions) -> np.ndarray:
        out = np.zeros(len(labelled_sessions))
        for i, ls in enumerate(labelled_sessions):
            ll = 0.0
            sa = ls.states_actions
            if not sa:
                out[i] = 0.0
                continue
            cur = sa[0][0]
            for s, a in sa[1:]:
                att = a
                p = float(self.T.probs[cur, att, s])
                ll += -np.log(max(p, self.eps))
                cur = s
            out[i] = ll / max(1, len(sa))  # per-step NLL: length-robust
        return out

class IsolationForestDetector:
    def __init__(self, contamination: float = 0.02, seed: int = 0):
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(contamination=contamination, random_state=seed)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        
        return -self.model.decision_function(X)

class AutoencoderDetector:
    def __init__(self, dim: int, latent: int = 4, epochs: int = 40, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        self.epochs = epochs
        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, latent), nn.ReLU(),
            nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, dim),
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        
        torch = self.torch
        xt = torch.as_tensor(X, dtype=torch.float32)
        for _ in range(self.epochs):
            recon = self.model(xt)
            loss = ((recon - xt) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32)
            recon = self.model(xt)
            return ((recon - xt) ** 2).mean(dim=1).cpu().numpy()
