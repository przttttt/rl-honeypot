from __future__ import annotations

import numpy as np

from ..metrics.ci import bootstrap_ci

DEFAULT_GAMMA = 1.0

def _states_actions(log: dict) -> tuple[np.ndarray, np.ndarray]:
    
    if hasattr(log, "states"):
        return np.asarray(log.states), np.asarray(log.actions)
    return np.asarray(log["states"]), np.asarray(log["actions"])

def _as_dict(log: dict) -> dict:
    
    if isinstance(log, dict):
        return log
    return {
        "states": log.states, "actions": log.actions,
        "propensities": log.propensities, "rewards": log.rewards,
        "next_states": log.next_states, "dones": log.dones,
        "sessions": log.sessions, "q_values": log.q_values,
        "v_values": log.state_values,
    }

def _pi_at(log: dict, pi_probs: np.ndarray) -> np.ndarray:
    states, actions = _states_actions(log)
    return pi_probs[states, actions]

def _trajectory_groups(log: dict):
    if hasattr(log, "sessions"):
        traj_ids = np.asarray(log.sessions)
    else:
        traj_ids = np.asarray(_as_dict(log)["sessions"])
    unique, inv = np.unique(traj_ids, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    return unique, inv, order

def _trajectory_steps(log: dict) -> list[np.ndarray]:
    
    _unique, inv, order = _trajectory_groups(log)
    dones = np.asarray(_as_dict(log)["dones"], dtype=bool)
    steps: list[np.ndarray] = []
    for i in range(int(inv.max()) + 1 if inv.size else 0):
        idx = order[inv[order] == i]
        stop = idx.size
        for k, t in enumerate(idx):
            if dones[t]:
                stop = k + 1
                break
        steps.append(idx[:stop])
    return steps

def _ess(w: np.ndarray) -> float:
    
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12))

def _diagnostics(rho: np.ndarray, weights: np.ndarray, n_traj: int, warn: bool = True) -> dict:
    out = {
        "ess": _ess(weights),
        "max_weight": float(np.max(weights)) if weights.size else 0.0,
        "n_trajectories": int(n_traj),
        "n_steps": int(rho.size),
    }
    if warn:
        out["high_variance_warning"] = bool(_ess(weights) < 0.05 * max(rho.size, 1))
    return out

def _trajectory_weights(log: dict, pi_probs: np.ndarray):
    d = _as_dict(log)
    mu = np.maximum(np.asarray(d["propensities"], dtype=float), 1e-12)
    rho = _pi_at(log, pi_probs) / mu
    steps = _trajectory_steps(log)
    W = np.ones(len(steps))
    for i, idx in enumerate(steps):
        w = 1.0
        for t in idx:
            w = w * rho[t]
        W[i] = w
    return rho, steps, W

def ips(log: dict, pi_probs: np.ndarray, level: float = 0.95, resamples: int = 10000,
        seed: int = 0, gamma: float = DEFAULT_GAMMA) -> dict:
    d = _as_dict(log)
    rewards = np.asarray(d["rewards"], dtype=float)
    rho, steps, W = _trajectory_weights(log, pi_probs)
    G = np.empty(len(steps))
    for i, idx in enumerate(steps):
        w = 1.0
        g = 0.0
        for k, t in enumerate(idx):
            w = w * rho[t]
            g += (gamma ** k) * w * rewards[t]
        G[i] = g
    lo, hi = bootstrap_ci(G, level, resamples, seed)
    out = {"estimate": float(G.mean()) if G.size else 0.0, "ci": [float(lo), float(hi)],
           "gamma": float(gamma), "normaliser": None}
    out.update(_diagnostics(rho, W, G.size))
    return out

def snips(log: dict, pi_probs: np.ndarray, level: float = 0.95, resamples: int = 10000,
          seed: int = 0, gamma: float = DEFAULT_GAMMA) -> dict:
    
    d = _as_dict(log)
    rewards = np.asarray(d["rewards"], dtype=float)
    rho, steps, W = _trajectory_weights(log, pi_probs)
    G = np.empty(len(steps))
    for i, idx in enumerate(steps):
        g = 0.0
        for k, t in enumerate(idx):
            g += (gamma ** k) * rewards[t]
        G[i] = g
    den = float(W.sum())
    estimate = float((W * G).sum() / den) if den > 0 else 0.0

    rng = np.random.default_rng(seed)
    n = W.size
    vals = np.empty(resamples)
    for r in range(resamples):
        b = rng.integers(0, n, n)
        dd = W[b].sum()
        vals[r] = (W[b] * G[b]).sum() / dd if dd > 0 else 0.0
    lo, hi = np.percentile(vals, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    out = {"estimate": estimate, "ci": [float(lo), float(hi)], "gamma": float(gamma),
           "normaliser": den}
    out.update(_diagnostics(rho, W, n))
    return out

def doubly_robust(log: dict, pi_probs: np.ndarray, level: float = 0.95,
                  resamples: int = 10000, seed: int = 0,
                  q_values: np.ndarray | None = None,
                  v_values: np.ndarray | None = None,
                  gamma: float = DEFAULT_GAMMA) -> dict:
    
    states, actions = _states_actions(log)
    d = _as_dict(log)
    rewards = np.asarray(d["rewards"], dtype=float)
    next_states = np.asarray(d["next_states"], dtype=int)
    dones = np.asarray(d["dones"], dtype=bool)
    _q = d.get("q_values")
    _v = d.get("v_values")
    if q_values is None:
        q_values = np.asarray(_q, dtype=float) if _q is not None else None
    if v_values is None:
        v_values = np.asarray(_v, dtype=float) if _v is not None else None
    if q_values is None:
        q_values = np.asarray(_q, dtype=float)
    if v_values is None:
        v_values = np.asarray(_v, dtype=float)
    Q = np.asarray(q_values, dtype=float)
    V = np.asarray(v_values, dtype=float)

    rho, steps, W = _trajectory_weights(log, pi_probs)
    values = np.empty(len(steps))
    for i, idx in enumerate(steps):
        total = V[states[idx[0]]]              # model term at trajectory start
        w = 1.0
        for t in idx:
            residual = rewards[t] + (0.0 if dones[t] else gamma * V[next_states[t]]) \
                - Q[states[t], actions[t]]
            total += w * residual
            w = w * rho[t]
        values[i] = total
    lo, hi = bootstrap_ci(values, level, resamples, seed)
    out = {"estimate": float(values.mean()) if values.size else 0.0,
           "ci": [float(lo), float(hi)], "gamma": float(gamma), "normaliser": None}
    out.update(_diagnostics(rho, W, values.size, warn=False))
    return out

def audit_against(reference: float, estimate: dict) -> dict:
    lo, hi = estimate["ci"]
    return {
        "bias": float(estimate["estimate"] - reference),
        "covers_truth": bool(lo <= reference <= hi),
        "ci_width": float(hi - lo),
    }

def known_value_validation(seed: int = 0, n: int = 20000) -> dict:
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 2, n)
    mu = np.full((2, 2), 0.5)
    actions = np.array([rng.choice(2, p=mu[s]) for s in states])
    rewards = np.where((states == 0) & (actions == 0), 1.0, 0.0)
    pi = np.array([[1.0, 0.0], [0.0, 1.0]])
    log = {"states": states.tolist(), "actions": actions.tolist(),
           "propensities": [mu[s, a] for s, a in zip(states, actions)],
           "rewards": rewards.tolist(), "next_states": np.zeros(n, dtype=int).tolist(),
           "dones": [True] * n, "sessions": [f"t{i}" for i in range(n)]}
    est = ips(log, pi, resamples=200, seed=seed)
    return {"truth": 0.5, "ips": est["estimate"], "abs_error": abs(est["estimate"] - 0.5),
            "snips": snips(log, pi, resamples=200, seed=seed)["estimate"]}
