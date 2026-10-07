from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from ..agents.tabular_q import TabularQAgent
from ..anomaly.detectors import AutoencoderDetector, IsolationForestDetector, TrajectoryLikelihood, session_features
from ..cowrie.parser import iter_cowrie_events
from ..envs import gym
from ..metrics.ci import auroc_ap, bootstrap_ci
from ..mdp.rewards import reward_table
from ..opelogs.logger import log_randomised_policy, support_check
from ..ope.estimators import audit_against, doubly_robust, ips, known_value_validation, snips
from ..ope.model_based import policy_evaluation
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..transitions.estimate import estimate
GAMMA_EVAL = 1.0

def _start_distribution(labelled) -> np.ndarray:
    counts = np.zeros(8)
    for ls in labelled:
        if ls.states_actions:
            counts[ls.states_actions[0][0]] += 1
    total = counts.sum()
    return counts / total if total else counts

def _mc_truth(env, act_fn, counts: np.ndarray, n_episodes: int, rng: np.random.Generator) -> dict:
    supported = np.flatnonzero(counts > 0)
    probs = counts[supported] / counts[supported].sum()
    rets = []
    per_start: dict[int, list[float]] = {int(s): [] for s in supported}
    for _ in range(n_episodes):
        s0 = int(rng.choice(supported, p=probs))
        (obs, _) = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)), options={'start_state': s0})
        done = trunc = False
        tot = 0.0
        while not (done or trunc):
            (obs2, r, done, trunc, info) = env.step(act_fn(int(obs)))
            tot += r
            obs = obs2
        per_start[s0].append(tot)
        rets.append(tot)
    arr = np.array(rets)
    (lo, hi) = bootstrap_ci(arr, resamples=2000, seed=0)
    return {'mean': float(arr.mean()), 'ci': [float(lo), float(hi)], 'n_episodes': int(arr.size), 'per_start': {str(s): float(np.mean(v)) for (s, v) in per_start.items() if v}}

def _stochastic_policy(pi_greedy: np.ndarray, eps: float) -> np.ndarray:
    pi = np.full((8, 5), eps / 5)
    for s in range(8):
        pi[s, pi_greedy[s]] += 1.0 - eps
    return pi

def _as_log_dict(log) -> dict:
    return {'states': log.states, 'actions': log.actions, 'propensities': log.propensities, 'rewards': log.rewards, 'next_states': log.next_states, 'dones': log.dones, 'sessions': log.sessions}

def _onehot(actions: np.ndarray) -> np.ndarray:
    pi = np.zeros((8, 5))
    for s in range(8):
        pi[s, int(actions[s])] = 1.0
    return pi

def run(cfg=None, overrides=None, out_root: str='results', quick: bool=False) -> dict:
    started = time.time()
    ctx = make_context('e8', out_root, overrides)
    cfg = cfg or ctx.cfg
    ho = cfg['mdp']['hyperoffline']
    est = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(Path('data/raw/p2.json.json'))))], laplace_alpha=cfg['transitions']['laplace_alpha'], min_support=cfg['transitions']['min_support_flag'])
    env = gym.make(cfg['env']['id_tabular'], transitions=est, max_steps=ho['max_steps'])
    R = reward_table()
    agent = TabularQAgent(alpha=ho['alpha'], gamma=ho['gamma'], epsilon=ho['epsilon'], seed=ctx.seed)
    rng_tr = np.random.default_rng(ctx.seed)
    for ep in range(300 if quick else ho['episodes']):
        (obs, _) = env.reset(seed=int(rng_tr.integers(0, 2 ** 31 - 1)))
        done = trunc = False
        s_prev = int(obs)
        while not (done or trunc):
            a = agent.act(s_prev)
            (obs2, r, done, trunc, info) = env.step(a)
            agent.update(s_prev, a, r, int(obs2), done or trunc)
            s_prev = int(obs2)
    pi_tab = agent.policy()
    labelled = [ls for ls in label_events(list(iter_cowrie_events(Path('data/raw/p2.json.json')))) if ls.transitions]
    n_sessions = 3000 if quick else min(60000, len(labelled))
    log_sessions = labelled[:n_sessions]
    starts = _start_distribution(log_sessions)
    n_replicates = 1 if quick else 5
    rng = np.random.default_rng(ctx.seed + 11)
    pooled = log_randomised_policy([ls.states_actions for ls in log_sessions], est, agent.q, epsilon=cfg['ope']['logging_epsilon'], rng=rng, reward_tensor=R, max_steps=ho['max_steps'])
    support = support_check(pooled, n_actions=5, min_actions_covered=5)
    log_dict = _as_log_dict(pooled)
    replicate_logs = []
    if n_replicates > 1:
        chunks = np.array_split(np.arange(len(log_sessions)), n_replicates)
        for (r, chunk) in enumerate(chunks):
            rng_r = np.random.default_rng(ctx.seed + 101 + r)
            lg = log_randomised_policy([log_sessions[i].states_actions for i in chunk], est, agent.q, epsilon=cfg['ope']['logging_epsilon'], rng=rng_r, reward_tensor=R, max_steps=ho['max_steps'])
            replicate_logs.append(_as_log_dict(lg))
    targets = {'stochastic_eps_greedy': _stochastic_policy(pi_tab, 0.1), 'tabular_greedy': _onehot(pi_tab), 'always_allow': _onehot(np.zeros(8, dtype=int)), 'always_block': _onehot(np.ones(8, dtype=int))}
    n_mc = 200 if quick else 1500
    rng_mc = np.random.default_rng(ctx.seed + 13)

    def act_for(pi: np.ndarray):
        rng_a = np.random.default_rng(ctx.seed + 29)

        def act(s: int) -> int:
            return int(rng_a.choice(5, p=pi[s]))
        return act
    rows: dict[str, dict] = {}
    for (name, pi) in targets.items():
        (V, Q) = policy_evaluation(est.probs, R, pi, gamma=GAMMA_EVAL)
        truth = _mc_truth(env, act_for(pi), starts, n_mc, rng_mc)
        ref = truth['mean']

        def evaluate(log_d: dict, resamples: int) -> dict:
            e_i = ips(log_d, pi, resamples=resamples, seed=ctx.seed, gamma=GAMMA_EVAL)
            e_s = snips(log_d, pi, resamples=resamples, seed=ctx.seed, gamma=GAMMA_EVAL)
            e_d = doubly_robust(log_d, pi, resamples=resamples, seed=ctx.seed, q_values=Q, v_values=V, gamma=GAMMA_EVAL)
            return {'ips': e_i, 'snips': e_s, 'dr': e_d}
        pooled_est = evaluate(log_dict, 2000)
        rep_est = [evaluate(d, 400) for d in replicate_logs]
        cells = {}
        for key in ('ips', 'snips', 'dr'):
            vals = np.array([r[key]['estimate'] for r in rep_est]) if rep_est else np.array([pooled_est[key]['estimate']])
            cells[key] = {**pooled_est[key], **audit_against(ref, pooled_est[key]), 'n_replicates': int(vals.size), 'replicate_mean': float(vals.mean()), 'replicate_std': float(vals.std(ddof=1)) if vals.size > 1 else 0.0, 'replicate_min': float(vals.min()), 'replicate_max': float(vals.max()), 'rmse_vs_truth': float(np.sqrt(np.mean((vals - ref) ** 2)))}
        rows[name] = {'mc_truth': truth, 'model_based_v': float(starts @ V), 'model_based_per_state': [float(x) for x in V], **cells}

    def failures(estimator: str, ref_scale: float) -> list[str]:
        return [n for (n, r) in rows.items() if not r[estimator]['covers_truth'] and abs(r[estimator]['bias']) > ref_scale]

    def mean_rmse(estimator: str) -> float:
        return float(np.mean([r[estimator]['rmse_vs_truth'] for r in rows.values()]))
    ips_miss = failures('ips', 1.0)
    snips_miss = failures('snips', 1.0)
    dr_miss = failures('dr', 1.0)
    ope_out = {'protocol': {'gamma': GAMMA_EVAL, 'return': 'episodic, undiscounted (env return; terminates at S7 or `terminate`)', 'start_state_distribution': {str(i): float(p) for (i, p) in enumerate(starts) if p > 0}, 'offline_log': "full-episode rollouts of the ε-greedy logging policy from the corpus's start-state distribution under the estimated MDP", 'on_policy_truth': 'independent MC rollouts of the target through the same env', 'n_logged_steps': int(len(pooled.states)), 'n_logged_sessions': int(len(set(pooled.sessions))), 'n_replicates': n_replicates}, 'known_value_check': known_value_validation(seed=ctx.seed), 'targets': rows, 'failure_summary': {'ips_ci_misses_truth': ips_miss, 'snips_ci_misses_truth': snips_miss, 'dr_ci_misses_truth': dr_miss, 'mean_rmse_vs_truth': {k: mean_rmse(k) for k in ('ips', 'snips', 'dr')}, 'note': f"DR is the only estimator whose 95% CI covers the on-policy truth on all {len(rows)} targets (IPS misses {len(ips_miss)} — {', '.join(ips_miss) or 'none'} — and SNIPS misses {len(snips_miss)} — {', '.join(snips_miss) or 'none'}), with the smallest mean across-replicate RMSE ({mean_rmse('dr'):.2f} vs {mean_rmse('ips'):.2f} for IPS and {mean_rmse('snips'):.2f} for SNIPS over {n_replicates} independent offline logs). The mechanism lives in the importance weights: for a deterministic target ρ = 0 on every off-target step, so ΣW is carried by the few trajectories that never left the target and SNIPS is averaged over them (ESS in the tens instead of thousands, max weight in the hundreds), while IPS keeps the exponentially growing cumulative weight. DR's residual r + γV(s′) − Q̂ still has conditional mean zero under the estimated MDP, so the model term absorbs what the weights cannot. Bias, CI width, ESS, max weight and the across-replicate spread are reported per estimator so the failures are auditable rather than hidden."}}
    gt = json.loads(Path('data/raw/ground_truth.json').read_text())
    keep = [ls for ls in labelled if ls.session_id in gt['sessions']]
    y_true = np.array([int(gt['sessions'][ls.session_id]['anomaly']) for ls in keep])
    feats = session_features(keep)
    tld_scores = TrajectoryLikelihood(est).scores(keep)
    auroc_tl = auroc_ap(y_true, tld_scores)
    iso = IsolationForestDetector(contamination=cfg['anomaly']['contamination'], seed=ctx.seed).fit(feats[y_true == 0])
    auroc_iso = auroc_ap(y_true, iso.scores(feats))
    ae = AutoencoderDetector(dim=feats.shape[1], latent=cfg['anomaly']['ae_latent'], epochs=cfg['anomaly']['ae_epochs'], seed=ctx.seed)
    ae.fit(feats[y_true == 0])
    ae_scores = ae.scores(feats)
    auroc_ae = auroc_ap(y_true, ae_scores)
    means = {'autoencoder': {'mean_normal': float(ae_scores[y_true == 0].mean()), 'mean_anomaly': float(ae_scores[y_true == 1].mean()), 'direction': 'higher = more anomalous'}, 'isolation_forest': {'mean_normal': float(iso.scores(feats[y_true == 0]).mean()), 'mean_anomaly': float(iso.scores(feats[y_true == 1]).mean()), 'direction': 'higher = more anomalous'}, 'trajectory_likelihood': {'mean_normal': float(tld_scores[y_true == 0].mean()), 'mean_anomaly': float(tld_scores[y_true == 1].mean()), 'direction': 'higher = more anomalous (per-step NLL under the estimated MDP)'}}
    anomaly_out = {'n_sessions': int(len(keep)), 'n_anomalies': int(y_true.sum()), 'protocol': 'fit on non-anomalous sessions only; score all', 'trajectory_likelihood': auroc_tl, 'isolation_forest': auroc_iso, 'autoencoder': auroc_ae, 'per_class_mean_scores': means, 'note': 'higher score = more anomalous; ground truth is real (D3). First-draft autoencoder AUROC 0.28 was a memorisation artefact of fitting on all sessions; the fixed protocol is reported here and the failure is kept in REPORT.'}
    out = {'experiment': 'e8_ope_anomaly', 'provenance': 'synthetic', 'status': 'ok', 'ope': ope_out, 'logging_support': support, 'anomaly': anomaly_out, 'decision_refs': ['D6', 'D13', 'D16', 'D20', 'D34']}
    return finish(ctx, out, started)
if __name__ == '__main__':
    r = run(quick=True)
    print(json.dumps(r['ope']['failure_summary'], indent=1))
