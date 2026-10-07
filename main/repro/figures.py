from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt  
import numpy as np  

C_MAIN = "#2b6cb0"      
C_ALT = "#c05621"       
C_TRUTH = "#2f855a"    
C_BAD = "#c53030"      
C_GREY = "#718096"     
PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b7791f", "#718096"]

STATE_LABELS = [f"S{i}" for i in range(8)]

def _load(results_root: Path, exp: str) -> dict:
    p = results_root / exp / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run `make reproduce` (or `make smoke`) first.")
    return json.loads(p.read_text())

def _finish(fig, out_dir: Path, name: str, source: str, top: float = 1.0) -> Path:
    
    fig.text(0.01, 0.005, f"rlhoneypot · synthetic corpus (D3) · source: {source}", fontsize=7, color=C_GREY)
    fig.tight_layout(rect=(0, 0.03, 1, top))
    out = out_dir / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def _ci_bars(ax, labels, means, los, his, colors, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, zorder=3)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)], fmt="none", ecolor="#2d3748", capsize=4, lw=1.2, zorder=4)
    ax.set_xticks(x, labels)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, zorder=0)

def fig_state_histogram(results_root: Path, out_dir: Path) -> Path:
    e1 = _load(results_root, "e1")
    counts = [e1["state_histogram"][str(i)] for i in range(8)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(STATE_LABELS, counts, color=C_MAIN, zorder=3)
    ax.bar_label(bars, fmt="%d", fontsize=8)
    ax.set_title("Attacker-state occupancy across parsed commands (E1)", fontsize=11, fontweight="bold")
    ax.set_ylabel("commands mapped to state")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    return _finish(fig, out_dir, "01_state_histogram.png", "results/e1/result.json")

def fig_transition_matrix(results_root: Path, out_dir: Path) -> Path:
    e2 = _load(results_root, "e2")
    M = np.array(e2["phases"]["phase1"]["transition_matrix"], dtype=float)
    supported = M.sum(axis=2) > 0                      
    n_sup = supported.sum(axis=1)                     
    P = np.einsum("sax,sa->sx", M, supported) / np.maximum(n_sup, 1)[:, None]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            v = P[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else "#1a365d")
    ax.set_xticks(range(8), STATE_LABELS)
    ax.set_yticks(range(8), STATE_LABELS)
    ax.set_xlabel("attacker next state")
    ax.set_ylabel("current state")
    ax.set_title("Transition matrix, phase 1 — action-averaged P(s′|s)\n"
                 "(paper Alg. 2, D7)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="probability")
    return _finish(fig, out_dir, "02_transition_matrix_phase1.png", "results/e2/result.json")

def fig_agent_returns(results_root: Path, out_dir: Path) -> Path:
    e4 = _load(results_root, "e4")
    rows = [(k, v["return"]) for k, v in e4["agents"].items() if "return" in v]
    rows.sort(key=lambda kv: kv[1]["mean"])
    labels = [k.replace("_", " ") for k, _ in rows]
    means = [r["mean"] for _, r in rows]
    los = [r["ci_lo"] for _, r in rows]
    his = [r["ci_hi"] for _, r in rows]
    colors = [C_GREY if lab == "random" else C_MAIN for lab in labels]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _ci_bars(ax, labels, means, los, his, colors,
             "Response-selection baselines: mean return ± 95% CI (E4)", "mean episodic return (safety-filtered)")
    for i, m in enumerate(means):
        ax.text(i, his[i] + 0.25, f"{m:.2f}", ha="center", fontsize=9)
    return _finish(fig, out_dir, "03_agent_returns_e4.png", "results/e4/result.json")

def fig_cold_vs_warm(results_root: Path, out_dir: Path) -> Path:
    e4 = _load(results_root, "e4")
    cw = e4["cold_vs_warm"]
    cold, warm, d = cw["cold_first50"], cw["warm_first50"], cw["delta_warm_minus_cold"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    _ci_bars(ax, ["cold start", "warm start"], [cold["mean"], warm["mean"]], [cold["ci_lo"], warm["ci_lo"]], [cold["ci_hi"], warm["ci_hi"]], [C_MAIN, C_ALT],
             "Paper's central claim tested: warm vs cold start, first 50 episodes (E4)", "mean return, first 50 episodes")
    ax.annotate(f"Δ(warm−cold) = {d['mean']:.3f}\n95% CI [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] — excludes 0\n(warm start hurts: negative finding, D9)",
                xy=(1, warm["mean"]), xytext=(0.42, 0.92), textcoords="axes fraction",
                fontsize=9, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD))
    return _finish(fig, out_dir, "04_cold_vs_warm_e4.png", "results/e4/result.json")

def fig_tabular_vs_dqn(results_root: Path, out_dir: Path) -> Path:
    
    e5 = _load(results_root, "e5")
    t = e5["tabular"].get("eval_return") or e5["tabular"]["final50_mean_across_seeds"]
    dqn = e5["dqn"].get("eval_return") or e5["dqn"]["final50_mean_across_seeds"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    _ci_bars(ax1, ["tabular Q", "DQN"], [t["mean"], dqn["mean"]], [t["ci_lo"], dqn["ci_lo"]], [t["ci_hi"], dqn["ci_hi"]], [C_MAIN, C_ALT], "Fair greedy evaluation of both final policies (E5)", "mean return, greedy")
    dv = e5.get("delta_eval_dqn_minus_tabular")
    if dv:
        label = (f"Δ = {dv['mean']:+.2f} [{dv['ci_lo']:+.2f}, {dv['ci_hi']:+.2f}]" + ("\nCI spans 0 → tie at this scale" if dv["ci_lo"] <= 0 <= dv["ci_hi"] else ""))
        ax1.text(0.5, 0.97, label, transform=ax1.transAxes, ha="center", va="top", fontsize=9, color=C_GREY, fontweight="bold")
    gs = e5["gamma_sensitivity"]
    _ci_bars(ax2, [f"γ={g}" for g in gs], [v["mean"] for v in gs.values()], [v["ci_lo"] for v in gs.values()], [v["ci_hi"] for v in gs.values()], [C_TRUTH] * len(gs), "γ sensitivity (D1): unresolved paper inconsistency", "return")
    fig.suptitle("E5 — at 8×5 states the two learners tie; scale is what separates them (E10)", fontsize=12, fontweight="bold")
    return _finish(fig, out_dir, "05_tabular_vs_dqn_e5.png", "results/e5/result.json")

def fig_learning_curves(results_root: Path, out_dir: Path) -> Path:
    
    e5 = _load(results_root, "e5")
    tab_curves = e5["tabular"]["learning_curves"]
    dqn_curves = e5["dqn"]["learning_curves"]
    step = int(e5["dqn"].get("learning_curve_step", 1))
    episodes = int(e5["dqn"].get("learning_curve_episodes", len(tab_curves[0])))
    x_tab = np.arange(1, len(tab_curves[0]) + 1)
    x_dqn = np.arange(1, len(dqn_curves[0]) + 1) * step
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for c in tab_curves:
        ax.plot(x_tab, c, color=C_MAIN, alpha=0.35, lw=1.0)
    for c in dqn_curves:
        ax.plot(x_dqn, c, color=C_ALT, alpha=0.35, lw=1.0)
    tab_mean = np.mean(tab_curves, axis=0)
    dqn_mean = np.mean(dqn_curves, axis=0)
    ax.plot(x_tab, tab_mean, color=C_MAIN, lw=2.2, label=f"tabular Q (mean, n={len(tab_curves)})")
    ax.plot(x_dqn, dqn_mean, color=C_ALT, lw=2.2, label=f"DQN (mean, n={len(dqn_curves)})")
    ax.set_title("Learning curves: tabular Q vs DQN (E5, synthetic corpus, D3)", fontsize=11, fontweight="bold")
    ax.set_xlabel("episode")
    ax.set_ylabel("episodic return")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir, "10_learning_curves_e5.png", "results/e5/result.json")

def fig_siem_lift(results_root: Path, out_dir: Path) -> Path:
    e6 = _load(results_root, "e6")
    st = e6["lift"]["states"]
    lifts = [s["lift"] for s in st]
    colors = [C_TRUTH if v >= 1 else C_ALT for v in lifts]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar([f"S{s['state']}" for s in st], lifts, color=colors, zorder=3)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.axhline(1.0, color=C_GREY, ls="--", lw=1.2, zorder=4, label="no lift (SIEM rate = marginal rate)")
    ax.set_title("SIEM alert lift per honeypot state (E6)", fontsize=11, fontweight="bold")
    ax.set_ylabel("lift (alert rate with state conditioning / marginal)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    return _finish(fig, out_dir, "06_siem_lift_e6.png", "results/e6/result.json")

def fig_prioritisation(results_root: Path, out_dir: Path) -> Path:
    e6 = _load(results_root, "e6")
    pr = {k: v for k, v in e6["prioritisation"].items()
          if isinstance(v, dict) and "auroc" in v}
    order = sorted(pr, key=lambda k: pr[k]["auroc"])
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x - w / 2, [pr[k]["auroc"] for k in order], w,
                label="AUROC", color=C_MAIN, zorder=3)
    b2 = ax.bar(x + w / 2, [pr[k].get("precision_at_50", float("nan")) for k in order],
                w, label="Precision@50", color=C_ALT, zorder=3)
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [k.replace("_", " ") for k in order])
    ax.set_ylim(0, 1.12)
    ax.set_title("Alert prioritisation: can an analyst triage better? (E6)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("score (higher = better)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    return _finish(fig, out_dir, "07_prioritisation_e6.png", "results/e6/result.json")

def fig_ope_vs_truth(results_root: Path, out_dir: Path) -> Path:
    
    e8 = _load(results_root, "e8")
    targets = e8["ope"].get("targets") or {
        "stochastic_eps_greedy": e8["ope"]["stochastic_eps_greedy"]}
    names = list(targets.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 4.6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        d = targets[name]
        rows = [("IPS", "ips", C_MAIN), ("SNIPS", "snips", C_ALT),
                ("DR", "dr", C_GREY)]
        labels = [r[0] for r in rows] + ["truth"]
        means = [d[k]["estimate"] for _, k, _ in rows] + [d["mc_truth"]["mean"]]
        los = [d[k]["ci"][0] for _, k, _ in rows] + [d["mc_truth"]["ci"][0]]
        his = [d[k]["ci"][1] for _, k, _ in rows] + [d["mc_truth"]["ci"][1]]
        colors = [c for _, _, c in rows] + [C_TRUTH]
        _ci_bars(ax, labels, means, los, his, colors, name.replace("_", " "),
                 "value (episodic return)")
        t_lo, t_hi = d["mc_truth"]["ci"]
        ax.axhspan(t_lo, t_hi, color=C_TRUTH, alpha=0.12, zorder=1)
        for i, (_, k, _) in enumerate(rows):
            covers = d[k].get("covers_truth", True)
            ax.patches[i].set_edgecolor("none" if covers else C_BAD)
            ax.patches[i].set_linewidth(0.0 if covers else 2.0)
            ax.text(i, 0.93, "ok" if covers else "fail", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=C_TRUTH if covers else C_BAD, fontweight="bold")
        ax.set_ylim(min(los) - 1.0, max(his) + 1.0)
    fig.suptitle("Off-policy estimators vs on-policy MC truth by target policy (E8, γ = 1)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.99, 0.02, "red border + fail = 95% CI excludes truth (estimator failure); "
                         "estimator disagreements are reported, not hidden (D20, D34)", ha="right", fontsize=8, color=C_GREY)
    return _finish(fig, out_dir, "08_ope_vs_truth_e8.png", "results/e8/result.json", top=0.92)

def fig_anomaly_detection(results_root: Path, out_dir: Path) -> Path:
    e8 = _load(results_root, "e8")
    an = e8["anomaly"]
    methods = ["trajectory_likelihood", "isolation_forest", "autoencoder"]
    labels = [m.replace("_", " ") for m in methods]
    auroc = [an[m]["auroc"] for m in methods]
    ap = [an[m]["average_precision"] for m in methods]
    base = an["n_anomalies"] / an["n_sessions"]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x - w / 2, auroc, w, label="AUROC", color=C_MAIN, zorder=3)
    b2 = ax.bar(x + w / 2, ap, w, label="average precision", color=C_ALT, zorder=3)
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.axhline(0.5, color=C_GREY, ls="--", lw=1, zorder=4, label="AUROC chance")
    ax.axhline(base, color=C_BAD, ls=":", lw=1, zorder=4, label=f"AP base rate ({base:.1%} anomalies)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_title("Anomaly detection with real ground truth (E8)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    return _finish(fig, out_dir, "09_anomaly_detection_e8.png", "results/e8/result.json")

def fig_phase_shift(results_root: Path, out_dir: Path) -> Path:
    e9 = _load(results_root, "e9")
    lv = sorted(e9["per_level"].values(), key=lambda d: d["shift_level"])
    x = np.array([d["shift_level"] for d in lv])
    delta = np.array([d["delta_warm_minus_cold"]["mean"] for d in lv])
    lo = np.array([d["delta_warm_minus_cold"]["ci_lo"] for d in lv])
    hi = np.array([d["delta_warm_minus_cold"]["ci_hi"] for d in lv])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.axhline(0.0, color=C_GREY, ls="--", lw=1.2, zorder=2, label="no effect (Δ = 0)")
    ax.errorbar(x, delta, yerr=[delta - lo, hi - delta], marker="o", color=C_MAIN, lw=1.8, capsize=4, zorder=4, label="Δ(warm−cold) first-50 return")
    corpus_lv = e9.get("corpus_matched_level")
    if corpus_lv is not None:
        ax.axvline(corpus_lv, color=C_BAD, ls=":", lw=1.4, zorder=3, label=f"real corpus shift (s ≈ {corpus_lv:.2f})")
    ax.set_title("Warm-start effect vs controlled phase shift (E9)", fontsize=11, fontweight="bold")
    ax.set_xlabel("shift level s  (0 = source dynamics → 1 = pure escalation)")
    ax.set_ylabel("Δ(warm − cold), first-50 return")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, zorder=0)
    return _finish(fig, out_dir, "11_phase_shift_e9.png", "results/e9/result.json")

def fig_scale_crossover(results_root: Path, out_dir: Path) -> Path:
    e10 = _load(results_root, "e10")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhline(0.0, color=C_GREY, ls="--", lw=1.2, zorder=2, label="no difference (Δ = 0)")
    styles = {"shared": (C_ALT, "o", "shared structure (generalisation possible)"), "randomised": (C_MAIN, "s", "randomised (control: no exploitable structure)")}
    for structure, cx in e10["crossover"].items():
        rows = sorted([r for r in e10["sweep"] if r["structure"] == structure], key=lambda r: r["n_states"])
        n = [r["n_states"] for r in rows]
        d = [r["delta_eval_dqn_minus_tabular"]["mean"] for r in rows]
        lo = [r["delta_eval_dqn_minus_tabular"]["ci_lo"] for r in rows]
        hi = [r["delta_eval_dqn_minus_tabular"]["ci_hi"] for r in rows]
        col, marker, lab = styles.get(structure, (C_GREY, "^", structure))
        ax.errorbar(n, d, yerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)], marker=marker, color=col, lw=1.8, capsize=4, zorder=4, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 32, 128, 512, 2048], ["8", "32", "128", "512", "2048"])
    ax.set_title("Where does deep RL overtake tabular Q? (E10)", fontsize=11, fontweight="bold")
    ax.set_xlabel("state-space size N (states)")
    ax.set_ylabel("Δ(DQN − tabular Q), mean greedy return")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, zorder=0)
    fw = e10["crossover"].get("shared", {}).get("first_drl_win_n_states")
    if fw is not None:
        ax.axvline(fw, color=C_BAD, ls=":", lw=1.4, zorder=3, label=f"first DRL win (N = {fw})")
        ax.legend(fontsize=8, loc="upper left")
    return _finish(fig, out_dir, "12_scale_crossover_e10.png", "results/e10/result.json")

def fig_sensitivity(results_root: Path, out_dir: Path) -> Path:
    
    e11 = _load(results_root, "e11")
    sweep = e11["sweep"]
    configs = sorted({r["corpus_config"] for r in sweep})
    n_vals = sorted({r["n_states"] for r in sweep})
    x = np.arange(len(configs))
    w = 0.35
    fig, axes = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, n_vals):
        rows = {r["corpus_config"]: r for r in sweep if r["n_states"] == n}
        means = [rows[c]["delta_dqn_minus_tabular"]["mean"] for c in configs]
        los = [rows[c]["delta_dqn_minus_tabular"]["ci_lo"] for c in configs]
        his = [rows[c]["delta_dqn_minus_tabular"]["ci_hi"] for c in configs]
        colors = [C_ALT if rows[c].get("drl_wins") else
                  (C_GREY if rows[c].get("tie") else C_BAD) for c in configs]
        ax.axhline(0, color=C_GREY, ls="--", lw=1.2, zorder=2)
        ax.bar(x, means, color=colors, zorder=3)
        ax.errorbar(x, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)], fmt="none", ecolor="#2d3748", capsize=4, lw=1.2, zorder=4)
        ax.set_xticks(x, configs)
        ax.set_title(f"N = {n}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Δ(DQN − tabular), eval return")
        ax.grid(axis="y", alpha=0.3, zorder=0)
    rv = e11.get("robustness_verdict", {})
    fig.suptitle(f"E11 — corpus sensitivity: crossover finding is {rv.get('overall', 'unknown').upper()}"
                 f"(N=8 tie={rv.get('n8_always_tie')} · N=128 DRL wins={rv.get('n128_always_drl_wins')})", fontsize=11, fontweight="bold", y=1.02)
    return _finish(fig, out_dir, "13_sensitivity_e11.png", "results/e11/result.json", top=0.93)

def fig_occupancy(results_root: Path, out_dir: Path) -> Path:
    
    e12 = _load(results_root, "e12")
    modes = e12["modes"]
    labels = list(modes.keys())
    means = [modes[k]["mean"] for k in labels]
    los = [modes[k]["ci_lo"] for k in labels]
    his = [modes[k]["ci_hi"] for k in labels]
    colors = [C_GREY, C_BAD, C_TRUTH]  # cold, warm_uniform, warm_matched
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _ci_bars(ax, [l.replace("_", " ") for l in labels], means, los, his, colors, "Warm-start mechanism: occupancy shift axis (E12)", "mean first-50 return, phase B")
    occ = e12.get("occupancy_explains_penalty", False)
    ax.text(0.5, 0.97, f"Occupancy shift {'DOES' if occ else 'does NOT'} explain the penalty (D9)", transform=ax.transAxes, ha="center", va="top", fontsize=9, color=C_TRUTH if occ else C_BAD, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    return _finish(fig, out_dir, "14_occupancy_e12.png", "results/e12/result.json")

def fig_siem_policy(results_root: Path, out_dir: Path) -> Path:
    
    e13 = _load(results_root, "e13")
    policies = e13["policies"]
    labels = list(policies.keys())
    means = [policies[k]["eval_return"]["mean"] for k in labels]
    los = [policies[k]["eval_return"]["ci_lo"] for k in labels]
    his = [policies[k]["eval_return"]["ci_hi"] for k in labels]
    colors = [C_MAIN, C_ALT, C_GREY]  # base, siem_augmented, siem_only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    _ci_bars(ax1, [l.replace("_", " ") for l in labels], means, los, his, colors, "Policy returns (base reward eval) ± 95% CI (E13)", "mean episodic return (base reward)")
    siem_helps = e13.get("siem_improves_return", False)
    ax1.text(0.5, 0.97, f"SIEM shaping {'IMPROVES' if siem_helps else 'does NOT improve'} return", transform=ax1.transAxes, ha="center", va="top", fontsize=9, color=C_TRUTH if siem_helps else C_BAD, fontweight="bold")
    pa = e13.get("policy_agreement", {})
    agree_labels = ["vs siem_augmented", "vs siem_only"]
    agree_vals = [pa.get("base_vs_siem_augmented", 0), pa.get("base_vs_siem_only", 0)]
    ax2.bar(agree_labels, agree_vals, color=[C_ALT, C_GREY], zorder=3)
    ax2.bar_label(ax2.patches, fmt="%.2f", fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Policy agreement with base (fraction of states)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("fraction agreeing")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    lift = e13.get("lift_by_state", [])
    if lift:
        ax2.text(0.5, 0.05, f"α_siem={e13.get('alpha_siem')} · lift={[round(v,2) for v in lift]}", transform=ax2.transAxes, ha="center", fontsize=7, color=C_GREY)
    fig.suptitle("E13 — SIEM-augmented policy: closing the E6 loop", fontsize=12, fontweight="bold")
    return _finish(fig, out_dir, "15_siem_policy_e13.png", "results/e13/result.json", top=0.93)

FIGURES = [
    ("01_state_histogram", fig_state_histogram),
    ("02_transition_matrix_phase1", fig_transition_matrix),
    ("03_agent_returns_e4", fig_agent_returns),
    ("04_cold_vs_warm_e4", fig_cold_vs_warm),
    ("05_tabular_vs_dqn_e5", fig_tabular_vs_dqn),
    ("06_siem_lift_e6", fig_siem_lift),
    ("07_prioritisation_e6", fig_prioritisation),
    ("08_ope_vs_truth_e8", fig_ope_vs_truth),
    ("09_anomaly_detection_e8", fig_anomaly_detection),
    ("10_learning_curves_e5", fig_learning_curves),
    ("11_phase_shift_e9", fig_phase_shift),
    ("12_scale_crossover_e10", fig_scale_crossover),
    ("13_sensitivity_e11", fig_sensitivity),
    ("14_occupancy_e12", fig_occupancy),
    ("15_siem_policy_e13", fig_siem_policy),
]

def main() -> int:
    ap = argparse.ArgumentParser(description="rlhoneypot figure generator")
    ap.add_argument("--results", default="results", help="results root directory")
    ap.add_argument("--out", default=None, help="output directory (default: <results>/figures)")
    args = ap.parse_args()

    results_root = Path(args.results)
    if not results_root.exists():
        sys.stderr.write(
            f"figures: {results_root}/ does not exist — run `make reproduce` first.\n")
        return 1
    out_dir = Path(args.out) if args.out else results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for name, fn in FIGURES:
        try:
            written.append(fn(results_root, out_dir))
        except FileNotFoundError as exc:
            skipped.append((name, str(exc)))
        except Exception as exc:  # noqa: BLE001 — one bad chart must not kill the batch;
            skipped.append((name, f"{type(exc).__name__}: {exc} — "
                                  "results may be incomplete; re-run `make reproduce`"))

    for p in written:
        print(f"[figures] wrote {p}")
    for name, why in skipped:
        print(f"[figures] SKIPPED {name}: {why}")
    print(f"[figures] {len(written)} written, {len(skipped)} skipped → {out_dir}")
    if not written:
        return 1
    return 0 if not skipped else 2  # 2 = partial: figure set is incomplete

if __name__ == "__main__":
    sys.exit(main())
