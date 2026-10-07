
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..agents.dqn import DQNAgent
from ..cowrie.parser import iter_cowrie_events
from ..envs.scaled_env import FEATURE_DIM, ScaledResponseEnv, ScaledTabularQ, build_scaled
from ..metrics.ci import mean_with_ci
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..transitions.estimate import estimate

def _make_env(scaled: dict, mode: str, max_steps: int, noise: float, seed: int) -> ScaledResponseEnv:
    return ScaledResponseEnv(scaled, max_steps=max_steps, obs_mode=mode,
                             reward_noise_sigma=noise, seed=seed)

def _pilot_mean_length(scaled: dict, max_steps: int, noise: float, episodes: int,
                       seed: int) -> float:
    env = _make_env(scaled, "discrete", max_steps, noise, seed)
    rng = np.random.default_rng(seed)
    lens = []
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        t = 0
        while not (done or trunc):
            _, _, done, trunc, _ = env.step(int(rng.integers(0, 5)))
            t += 1
        lens.append(t)
    return float(np.mean(lens)) if lens else 1.0

def _train_tabular(scaled: dict, n_states: int, ho: dict, seed: int, episodes: int,
                   max_steps: int, noise: float) -> tuple[list[float], float, ScaledTabularQ]:
    env = _make_env(scaled, "discrete", max_steps, noise, seed)
    agent = ScaledTabularQ(n_states=n_states, alpha=ho["alpha"], gamma=ho["gamma"],
                           epsilon=ho["epsilon"], seed=seed)
    rng = np.random.default_rng(seed)
    curve: list[float] = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            a = agent.act(int(obs))
            obs2, r, done, trunc, _ = env.step(a)
            agent.update(int(obs), a, r, int(obs2), done or trunc)
            obs = obs2
            total += r
        curve.append(total)
    return curve, agent.coverage, agent

def _train_dqn(scaled: dict, ho: dict, dqn_cfg: dict, seed: int, episodes: int,
               max_steps: int, noise: float, warmup_steps: int) -> tuple[list[float], DQNAgent]:
    env = _make_env(scaled, "features", max_steps, noise, seed)
    agent = DQNAgent(obs_dim=FEATURE_DIM, n_actions=5, cfg=dqn_cfg, seed=seed)
    rng = np.random.default_rng(seed)
    curve: list[float] = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            a = agent.act(context=np.asarray(obs, dtype=np.float32))
            obs2, r, done, trunc, _ = env.step(a)
            agent.observe(obs, a, r, obs2, done or trunc)
            obs = obs2
            total += r
        curve.append(total)
    return curve, agent

def _evaluate(scaled: dict, mode: str, agent, max_steps: int, noise: float,
              seeds: list[int]) -> float:
    
    env = _make_env(scaled, mode, max_steps, noise, seeds[0])
    returns = []
    for sd in seeds:
        obs, _ = env.reset(seed=int(sd))
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            if mode == "discrete":
                a = int(agent.act_greedy(int(obs)))
            else:
                a = int(agent.act_greedy(context=np.asarray(obs, dtype=np.float32)))
            obs, r, done, trunc, _ = env.step(a)
            total += r
        returns.append(total)
    return float(np.mean(returns))

def run(cfg=None, overrides=None, out_root: str = "results", quick: bool = False) -> dict:
    started = time.time()
    ctx = make_context("e10", out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg["mdp"]["hyperoffline"]
    sc = cfg.get("scale", {})
    max_steps = int(ho["max_steps"])

    events = list(iter_cowrie_events(Path("data/raw/p1.json")))
    est = estimate([ls.transitions for ls in label_events(events)],
                   laplace_alpha=cfg["transitions"]["laplace_alpha"],
                   min_support=cfg["transitions"]["min_support_flag"])

    structures = list(sc.get("structures", ["shared", "randomised"]))
    ks = list(sc.get("k_values_quick", [1, 16])) if quick else list(sc.get("k_values", [1, 4, 16, 64, 256]))
    n_seeds = int(sc.get("seeds_quick", 2)) if quick else int(sc.get("seeds", 5))
    episodes = int(sc.get("episodes_quick", 120)) if quick else int(sc.get("episodes", 800))
    n_eval = int(sc.get("eval_episodes_quick", 30)) if quick else int(sc.get("eval_episodes", 200))
    decay_frac = float(sc.get("epsilon_decay_frac", 0.5))
    sigma = float(sc.get("randomised_reward_sigma", 6.0))
    noise = float(sc.get("reward_noise_sigma", 0.0))
    warmup = int(sc.get("dqn_warmup_steps", 200))

    out: dict = {
        "experiment": "e10_scale",
        "provenance": "synthetic",
        "status": "ok",
        "protocol": {
            "k_values": ks, "n_states": [8 * k for k in ks], "structures": structures,
            "seeds_per_cell": n_seeds, "episodes_per_learner": episodes,
            "eval_episodes": n_eval, "max_steps": max_steps,
            "reward_noise_sigma": noise, "randomised_reward_sigma": sigma,
            "observation": "information-equivalent: Discrete(N) vs Box(11) feature view",
            "budget_rule": "tabular and DQN receive the SAME episode budget in every cell",
        },
        "sweep": [],
    }

    for structure in structures:
        for k in ks:
            n_states = 8 * k
            scaled = build_scaled(est, k, structure=structure, seed=ctx.seed + 7919 * k,
                                  randomised_sigma=sigma)
            mean_len = _pilot_mean_length(scaled, max_steps, noise, 10, ctx.seed + k)
            decay_steps = max(200, int(decay_frac * episodes * max(mean_len, 1.0)))
            dqn_cfg = dict(cfg["agents"]["dqn"])
            dqn_cfg["gamma"] = ho["gamma"]
            dqn_cfg["epsilon_decay_steps"] = decay_steps
            dqn_cfg["warmup_steps"] = warmup

            tab_curves, tab_cov = [], []
            dqn_curves = []
            tab_eval_seeds, dqn_eval_seeds = [], []
            eval_seeds = [int(ctx.seed + 900000 + i * 104729) for i in range(n_eval)]
            for seed_i in range(n_seeds):
                curve, cov, tagent = _train_tabular(scaled, n_states, ho,
                                                    ctx.seed + 100000 + seed_i * 7919,
                                                    episodes, max_steps, noise)
                tab_curves.append(curve)
                tab_cov.append(cov)
                dcurve, dagent = _train_dqn(scaled, ho, dqn_cfg, ctx.seed + 500000 + seed_i * 7919,
                                            episodes, max_steps, noise, warmup)
                dqn_curves.append(dcurve)
                tab_eval_seeds.append(
                    _evaluate(scaled, "discrete", tagent, max_steps, noise, eval_seeds))
                dqn_eval_seeds.append(
                    _evaluate(scaled, "features", dagent, max_steps, noise, eval_seeds))

            tab_final = [float(np.mean(c[-50:])) for c in tab_curves]
            dqn_final = [float(np.mean(c[-50:])) for c in dqn_curves]
            row = {
                "structure": structure,
                "k": k,
                "n_states": n_states,
                "pilot_mean_episode_len": mean_len,
                "epsilon_decay_steps": decay_steps,
                "tabular": {
                    "final50_mean_across_seeds": mean_with_ci(np.array(tab_final), resamples=2000, seed=0),
                    "eval_return": mean_with_ci(np.array(tab_eval_seeds), resamples=2000, seed=0),
                    "state_coverage": mean_with_ci(np.array(tab_cov), resamples=2000, seed=0),
                },
                "dqn": {
                    "final50_mean_across_seeds": mean_with_ci(np.array(dqn_final), resamples=2000, seed=0),
                    "eval_return": mean_with_ci(np.array(dqn_eval_seeds), resamples=2000, seed=0),
                },
                "delta_eval_dqn_minus_tabular": mean_with_ci(
                    np.array(dqn_eval_seeds) - np.array(tab_eval_seeds), resamples=2000, seed=0),
            }
            out["sweep"].append(row)
            d = row["delta_eval_dqn_minus_tabular"]
            print(f"[e10] {structure:<10} k={k:>3} N={n_states:>5} "
                  f"tab={row['tabular']['eval_return']['mean']:6.2f} "
                  f"dqn={row['dqn']['eval_return']['mean']:6.2f} "
                  f"delta={d['mean']:6.2f} [{d['ci_lo']:.2f},{d['ci_hi']:.2f}] "
                  f"cov={row['tabular']['state_coverage']['mean']:.2f}", flush=True)

    crossover = {}
    for structure in structures:
        rows = sorted([r for r in out["sweep"] if r["structure"] == structure],
                      key=lambda r: r["n_states"])
        first_win = next((r for r in rows
                          if r["delta_eval_dqn_minus_tabular"]["ci_lo"] > 0), None)
        crossover[structure] = {
            "n_states": [r["n_states"] for r in rows],
            "tabular_eval": [r["tabular"]["eval_return"]["mean"] for r in rows],
            "dqn_eval": [r["dqn"]["eval_return"]["mean"] for r in rows],
            "delta": [r["delta_eval_dqn_minus_tabular"]["mean"] for r in rows],
            "first_drl_win_n_states": first_win["n_states"] if first_win else None,
        }
    out["crossover"] = crossover
    out["decision_refs"] = ["D15", "D33"]
    out["note"] = (
        "Both arms see an information-equivalent observation of the same environment under "
        "the same episode budget; the only difference is enumeration (N x 5 table) versus "
        "function approximation (2x64 MLP). A win in 'shared' but not in 'randomised' "
        "attributes the effect to generalisation over exploitable structure rather than to "
        "model capacity alone."
    )
    return finish(ctx, out, started)

if __name__ == "__main__":
    r = run(quick=True)
    print(r["crossover"])
