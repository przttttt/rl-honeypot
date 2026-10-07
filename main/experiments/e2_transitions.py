from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from ..pipeline.label import label_events
from ..repro.harness import finish, make_context
from ..transitions.estimate import estimate, heldout_logloss
from ..cowrie.parser import iter_cowrie_events

def run(cfg=None, overrides=None, out_root: str='results') -> dict:
    started = time.time()
    ctx = make_context('e2', out_root, overrides)
    cfg = cfg or ctx.cfg
    laplace = cfg.get('transitions', {}).get('laplace_alpha', 1.0)
    min_support = cfg.get('transitions', {}).get('min_support_flag', 30)
    out: dict = {'experiment': 'e2_transitions', 'provenance': 'synthetic', 'status': 'ok', 'phases': {}}
    gt = json.loads(Path('data/raw/ground_truth.json').read_text())
    for (phase, fname) in ((1, 'p1.json'), (2, 'p2.json.json')):
        events = list(iter_cowrie_events(Path('data/raw') / fname))
        labelled = label_events(events)
        rng = np.random.default_rng(ctx.seed)
        order = rng.permutation(len(labelled))
        cut = int(0.8 * len(order))
        train = [labelled[i] for i in order[:cut]]
        eval_ = [labelled[i] for i in order[cut:]]
        est = estimate([ls.transitions for ls in train], laplace_alpha=laplace, min_support=min_support)
        from rlhoneypot.mdp.definitions import action_support
        (n, nll_model, nll_uniform) = (0, 0.0, 0.0)
        for ls in eval_:
            for (s, a, s2) in ls.transitions:
                att = int(est.support_mask[s].argmax())
                p = float(est.probs[s, att, s2])
                nll_model += -np.log(max(p, 1e-12))
                nll_uniform += -np.log(1.0 / 8)
                n += 1
        nll = {'n': n, 'nll_model': nll_model / n if n else None, 'nll_uniform': nll_uniform / n if n else None, 'improvement_nats': (nll_uniform - nll_model) / n if n else None}
        truth_sessions = {k: v for (k, v) in gt['sessions'].items() if v['phase'] == phase}
        (n_top1, n_tot) = (0, 0)
        for ls in eval_:
            gtl = truth_sessions.get(ls.session_id)
            if not gtl or not gtl['states']:
                continue
            gstates = gtl['states']
            gstates = [s for s in gstates if s != gtl['states'][0] or True]
            distinct = []
            for s in gstates:
                if not distinct or s != distinct[-1]:
                    distinct.append(s)
            gstart = ls.states_actions[0][0] if ls.states_actions else None
            if gstart is None:
                continue
            for (j, s2) in enumerate(distinct[1:], start=1):
                a_supported = int(est.support_mask[gstart].argmax())
                pred_next = int(np.argmax(est.probs[gstart, a_supported]))
                n_tot += 1
                n_top1 += int(pred_next == s2)
                gstart = s2
        top1 = n_top1 / n_tot if n_tot else None
        out['phases'][f'phase{phase}'] = {'n_sessions_total': len(labelled), 'n_train': len(train), 'n_eval': len(eval_), 'n_transitions': est.n_transitions, 'out_of_support_counts': est.out_of_support_counts, 'unmapped_commands': sum((ls.n_unmapped for ls in labelled)), 'low_confidence_rows': est.low_confidence_rows, 'heldout': nll, 'top1_recovery_vs_ground_truth': top1, 'argmax_holdout_accuracy': _argmax_accuracy(est, eval_), 'transition_matrix': est.probs.tolist(), 'counts': est.counts.tolist()}
    out['decision_refs'] = ['D7', 'D2']
    return finish(ctx, out, started)

def _argmax_accuracy(est, eval_sessions):
    import numpy as np
    from ..pipeline.label import label_events
    from ..cowrie.parser import iter_cowrie_events
    (n, n_ok) = (0, 0)
    for ls in eval_sessions:
        for (s, a, s2) in ls.transitions:
            n += 1
            if int(np.argmax(est.probs[s, a])) == int(s2):
                n_ok += 1
    return float(n_ok / n) if n else None
if __name__ == '__main__':
    r = run()
    print({k: v.get('top1_recovery_vs_ground_truth') for (k, v) in r['phases'].items()})
