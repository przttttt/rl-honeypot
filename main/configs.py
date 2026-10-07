from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import yaml
_CONFIGS: dict[str, dict[str, Any]] = {}
_HASHES: dict[str, str] = {}

def load_config(name: str, config_dir: str | Path='configs') -> dict[str, Any]:
    if name in _CONFIGS:
        return _CONFIGS[name]
    path = Path(config_dir) / f'{name}.yaml'
    if not path.exists():
        raise FileNotFoundError(f'config not found: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'config {path} is not a mapping')
    _schema_check(name, data)
    _CONFIGS[name] = data
    _HASHES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data

def config_sha256(name: str, config_dir: str | Path='configs') -> str:
    load_config(name, config_dir)
    return _HASHES[name]
_REQUIRED: dict[str, list[str]] = {'experiment': ['seeds', 'mdp', 'corpus', 'env', 'agents', 'ope', 'eval', 'siem'], 'safety': ['sandbox', 'action_filter', 'insult_rate_cap'], 'siem': ['mode', 'offline', 'live', 'wazuh'], 'transitions': []}

def _schema_check(name: str, data: dict[str, Any]) -> None:
    for key in _REQUIRED.get(name, []):
        if key not in data:
            raise ValueError(f"config '{name}' missing required key: {key}")

def merged_config(config_dir: str | Path='configs') -> dict[str, Any]:
    exp = load_config('experiment', config_dir)
    return exp
