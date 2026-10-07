from __future__ import annotations
import time
from collections import Counter
from pathlib import Path
from ..corpus.generator import generate_corpus
from ..repro.harness import finish, make_context

def run(cfg=None, overrides=None, out_root: str='results') -> dict:
    started = time.time()
    ctx = make_context('e0', out_root, overrides)
    cfg = cfg or ctx.cfg
    c = cfg['corpus']
    paths = generate_corpus(out_dir=Path('data/raw'), n_phase1=c['phase1']['sessions_default'], n_phase2=c['phase2']['sessions_default'], seed=ctx.seed % 2 ** 31)
    import json
    gt = json.loads(Path(paths.ground_truth).read_text())
    counts = {'phase1': 0, 'phase2': 0}
    states_central = Counter()
    anomalies = 0
    n_cmds = []
    for s in gt['sessions'].values():
        counts[f"phase{s['phase']}"] += 1
        n_cmds.append(s['n_commands'])
        states_central.update(s['states'])
        anomalies += int(s['anomaly'])
    n_sessions = sum(counts.values())
    result = {'experiment': 'e0_corpus', 'provenance': 'synthetic', 'status': 'ok', 'paths': {k: str(v) for (k, v) in paths.__dict__.items()}, 'sessions_generated': counts, 'paper_sessions_full': {'phase1': c['phase1']['sessions_full'], 'phase2': c['phase2']['sessions_full']}, 'anomaly_rate': round(anomalies / max(1, n_sessions), 5), 'mean_commands_per_session': round(sum(n_cmds) / max(1, len(n_cmds)), 3), 'state_histogram': dict(sorted(states_central.items())), 'decision_refs': ['D3', 'D18']}
    return finish(ctx, result, started)
if __name__ == '__main__':
    print(run()['sessions_generated'])
