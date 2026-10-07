from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from ..cowrie.parser import iter_cowrie_events
from ..ecs.normalise import ecs_session_summary, to_ecs
from ..metrics.ci import bootstrap_ci
from ..pipeline.label import label_events
from ..prioritise.ranker import SupervisedRanker, alert_features, baseline_scores, evaluate_ranking
from ..repro.harness import finish, make_context
from ..siem.index import OfflineIndex, WazuhRules, build_index_from_sessions, correlation_lift, siem_alert_stream

def run(cfg=None, overrides=None, out_root: str='results') -> dict:
    started = time.time()
    ctx = make_context('e6', out_root, overrides)
    cfg = cfg or ctx.cfg
    ph2 = list(iter_cowrie_events(Path('data/raw/p2.json.json')))
    labelled = label_events(ph2)
    from ..configs import load_config
    siem_cfg = load_config('siem')
    index = OfflineIndex(name=siem_cfg['offline']['index'])
    timestamps: dict[str, str] = {}
    for ev in ph2:
        if ev.event_name == 'cowrie.command.input':
            ts_map = timestamps.setdefault(ev.session_id, ev.timestamp)
        if ev.event_name in ('cowrie.command.input', 'cowrie.login.failed', 'cowrie.login.success', 'cowrie.session.file_download'):
            if ev.event_name == 'cowrie.command.input':
                try:
                    from ..mdp.mapping import map_command
                    m = map_command(ev.raw.get('input', ''))
                    index.add(to_ecs(ev, mdp_state=m.state, mdp_action=m.attacker_action, mapping_rule=m.rule, mapped=True))
                    continue
                except Exception:
                    pass
            index.add(to_ecs(ev))
    rules = WazuhRules.from_config(siem_cfg)
    rng = np.random.default_rng(ctx.seed)
    rates = cfg['siem']['alert_rate_per_state']
    noise = cfg['siem']['noise_rate']
    alerts = siem_alert_stream(index, rules, rates, noise, rng)
    honeypot_counts = index.count_by_state()
    alert_counts = [0] * 8
    for a in alerts:
        st = a.get('rlhoneypot', {}).get('mdp_state')
        if st is not None:
            alert_counts[int(st)] += 1
    lift = correlation_lift(honeypot_counts, alert_counts, noise)
    gt = json.loads(Path('data/raw/ground_truth.json').read_text())
    deep_sessions = {sid for (sid, v) in gt['sessions'].items() if v.get('phase') == 2 and len(v.get('states', [])) >= 3}
    X = alert_features(alerts)
    y = np.array([1 if a.get('session', {}).get('id') in deep_sessions else 0 for a in alerts])
    rankings: dict = {}
    for kind in ('recency', 'severity'):
        scores = baseline_scores(kind, X, alerts)
        rankings[kind] = evaluate_ranking(y, scores, cfg['eval']['precision_at_k'])
    n_train = int(0.7 * len(X))
    rng2 = np.random.default_rng(ctx.seed + 1)
    order = rng2.permutation(len(X))
    (tr, te) = (order[:n_train], order[n_train:])
    for model in ('logistic', 'gradient_boosting'):
        rk = SupervisedRanker(model=model, seed=ctx.seed).fit(X[tr], y[tr])
        rankings[model] = evaluate_ranking(y[te], rk.scores(X[te]), cfg['eval']['precision_at_k'])
        rankings[model]['n_test'] = int(len(te))
    rankings['_note'] = 'supervised models evaluated on held-out 30% split'
    from ..siem.index import OpensearchClient
    client = OpensearchClient(index=index.name, dry_run=True, sink_path='data/siem_bulk.ndjson')
    bulk = client.bulk_index(alerts[:1000])
    out = {'experiment': 'e6_siem', 'provenance': 'synthetic', 'status': 'ok', 'ecs_documents': len(index.docs), 'alerts_emulated': len(alerts), 'alert_counts_by_state': alert_counts, 'honeypot_counts_by_state': honeypot_counts, 'lift': lift, 'prioritisation': rankings, 'bulk_write': bulk, 'decision_refs': ['D11', 'D12', 'D17']}
    return finish(ctx, out, started)
if __name__ == '__main__':
    r = run()
    print(r['lift']['marginal_alert_rate'], r['alerts_emulated'])
