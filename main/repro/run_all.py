from __future__ import annotations
import argparse
import json
import time
import traceback
from pathlib import Path
from ..logging_util import sha256_of, write_json

def _sha256(path: Path) -> str:
    return sha256_of(path)
from .harness import require_sandbox
EXPERIMENTS = [('e0_corpus', dict()), ('e1_parse', dict()), ('e2_transitions', dict()), ('e3_envs', dict()), ('e4_baselines', dict()), ('e5_deep_rl', dict()), ('e6_siem', dict()), ('e7_opelogs', dict()), ('e8_ope_anomaly', dict()), ('e9_phase_shift', dict()), ('e10_scale', dict()), ('e11_sensitivity', dict()), ('e12_occupancy', dict()), ('e13_siem_policy', dict())]

def main() -> int:
    ap = argparse.ArgumentParser(description='rlhoneypot one-command reproduction')
    ap.add_argument('--out', default='results')
    ap.add_argument('--smoke', action='store_true', help='fast low-budget run (CI)')
    ap.add_argument('--only', default=None, help='comma-separated experiment names')
    args = ap.parse_args()
    require_sandbox()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    smoke = args.smoke
    quick = smoke
    only = set(args.only.split(',')) if args.only else None
    failures: list[dict] = []
    durations: dict[str, float] = {}
    t0 = time.time()
    (out_root / 'RUN_MANIFEST.json').unlink(missing_ok=True)
    for exp in ('e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13'):
        try:
            (out_root / exp / 'manifest_entry.json').unlink(missing_ok=True)
        except Exception:
            pass
    for (name, kwargs) in EXPERIMENTS:
        if only and name not in only:
            continue
        mod = __import__(f'rlhoneypot.experiments.{name}', fromlist=['run'])
        t1 = time.time()
        print(f'[run_all] {name} ...', flush=True)
        try:
            res = mod.run(quick=quick) if name in ('e4_baselines', 'e5_deep_rl', 'e7_opelogs', 'e8_ope_anomaly', 'e9_phase_shift', 'e10_scale', 'e11_sensitivity', 'e12_occupancy', 'e13_siem_policy') else mod.run()
            status = res.get('status', 'ok')
            print(f'[run_all] {name} done ({time.time() - t1:.1f}s) status={status}', flush=True)
        except Exception as exc:
            status = f'FAILED: {exc}'
            failures.append({'experiment': name, 'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'[run_all] {name} FAILED: {exc}', flush=True)
        durations[name] = round(time.time() - t1, 1)
    summary = {'smoke': smoke, 'total_duration_s': round(time.time() - t0, 1), 'per_experiment_durations_s': durations, 'failures': failures, 'status': 'ok' if not failures else 'with_failures'}
    write_json(out_root / 'RUN_SUMMARY.json', summary)
    if not failures:
        from .report import generate_report
        generate_report(out_root)
        print(f'[run_all] report written to {out_root}/REPORT.md', flush=True)
        report_p = out_root / 'REPORT.md'
        report_entry = {'config_sha256': '', 'duration_s': round(time.time() - t0, 2), 'experiment': 'report', 'git_describe': '', 'notes': {'provenance': 'synthetic', 'status': 'ok'}, 'python': '', 'report_sha256': _sha256(report_p), 'seeds': {'experiment': 0}, 'started_utc': time.time()}
        mf = out_root / 'RUN_MANIFEST.json'
        entries = []
        if mf.exists():
            entries = json.loads(mf.read_text())
        entries.append(report_entry)
        write_json(mf, entries)
    else:
        print('[run_all] failures present; REPORT.md not regenerated (fix and re-run)', flush=True)
    print(f"[run_all] finished in {summary['total_duration_s']}s", flush=True)
    return 0 if not failures else 1
if __name__ == '__main__':
    raise SystemExit(main())
