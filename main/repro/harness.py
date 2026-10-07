from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from pathlib import Path
sys.setrecursionlimit(10000)
from ..configs import config_sha256, load_config
from ..logging_util import RunManifest, append_manifest, derive_seed, git_describe, write_json, sha256_of, sha256_of_file, sha256_of_string, canonical_json, Result
from ..safety.shim import require_sandbox

@dataclass
class ExperimentContext:
    name: str
    seed: int
    cfg: dict
    out_dir: Path

def make_context(name: str, out_root: Path | str='results', overrides: dict | None=None) -> ExperimentContext:
    require_sandbox()
    cfg = load_config('experiment')
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    idx = int(name[1:]) if name.startswith('e') and name[1:].isdigit() else 0
    seed = derive_seed(cfg['seeds']['master'], name)
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentContext(name=name, seed=seed, cfg=cfg, out_dir=out_dir)

def finish(ctx: ExperimentContext, result: dict, started: float) -> dict:
    mf = RunManifest(experiment=ctx.name, config_sha256=config_sha256('experiment'), seeds={'experiment': ctx.seed}, git_describe=git_describe(), python=sys.version.split()[0], notes={k: result.get(k) for k in ('provenance', 'status')})
    entry = mf.seal()
    entry['duration_s'] = round(time.time() - started, 2)
    result_path = ctx.out_dir / 'result.json'
    body = dict(result)
    body.pop('sha256', None)
    body_text = canonical_json(body)
    result_sha256 = sha256_of_string(body_text)
    out = dict(result)
    out['sha256'] = result_sha256
    write_json(result_path, out)
    entry['result_sha256'] = result_sha256
    write_json(ctx.out_dir / 'manifest_entry.json', dict(entry))
    append_manifest('results', entry)
    return out

def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for (k, v) in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
