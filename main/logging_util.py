from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

def sha256_of(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def derive_seed(master: int, experiment: str, salts: dict[str, int] | None=None) -> int:
    h = hashlib.sha256()
    h.update(f'rlhoneypot:{master}:{experiment}'.encode())
    for k in sorted((salts or {}).items()):
        h.update(f'|{k[0]}={k[1]}'.encode())
    return int.from_bytes(h.digest()[:8], 'big') % (2 ** 31 - 1)

def rng_for(master: int, experiment: str, salts: dict[str, int] | None=None) -> np.random.Generator:
    return np.random.default_rng(derive_seed(master, experiment, salts))

def torch_seed(master: int, experiment: str, salts: dict[str, int] | None=None) -> int:
    return derive_seed(master, experiment, salts) % (2 ** 31 - 1)

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, default=_json_default)

def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not JSON-serialisable: {type(o)}')

def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(obj) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)

def sha256_of_file(path: Path | str) -> str:
    return sha256_of(path)

def sha256_of_string(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

@dataclass
class Result:
    experiment: str
    status: str
    provenance: str
    decision_refs: list[str]
    notes: dict[str, Any] | None = None
    ope: dict[str, Any] | None = None
    model_based_vs_mc: dict[str, Any] | None = None
    anomaly: dict[str, Any] | None = None
    logging_support: dict[str, Any] | None = None
    config_sha256: str = ''
    seeds: dict[str, int] = field(default_factory=dict)
    git_describe: str = ''
    python: str = ''
    started_utc: float = field(default_factory=time.time)
    duration_s: float = 0.0
    sha256: str = field(repr=False, default='')
    result_sha256: str = ''
    report_sha256: str = ''

    def to_dict(self) -> dict[str, Any]:
        out = dataclass_to_dict(self)
        out.pop('_field_defaults', None)
        out.pop('_fields', None)
        out.pop('_name', None)
        return out

@dataclass
class RunManifest:
    experiment: str
    config_sha256: str
    seeds: dict[str, int] = field(default_factory=dict)
    started_utc: float = field(default_factory=time.time)
    duration_s: float = 0.0
    git_describe: str = ''
    python: str = ''
    notes: dict[str, Any] = field(default_factory=dict)

    def seal(self) -> dict[str, Any]:
        self.duration_s = round(time.time() - self.started_utc, 3)
        return dataclass_to_dict(self)

def dataclass_to_dict(dc: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (k, v) in dc.__dict__.items():
        out[k] = v
    return out

def append_manifest(out_dir: Path | str, entry: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mf = out_dir / 'RUN_MANIFEST.json'
    entries = []
    if mf.exists():
        try:
            entries = json.loads(mf.read_text())
        except Exception:
            entries = []
    entries.append(entry)
    write_json(mf, entries)

def git_describe() -> str:
    try:
        import subprocess
        return subprocess.run(['git', 'describe', '--always', '--dirty'], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
    except Exception:
        return 'nogit'
