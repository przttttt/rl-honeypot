from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable
from ..ecs.normalise import to_ecs
from ..logging_util import write_json
from ..mdp.definitions import N_STATES

class OpensearchClient:

    def __init__(self, endpoint: str | None=None, index: str='rlhoneypot-cowrie-ecs', dry_run: bool=True, sink_path: str | None=None):
        self.endpoint = endpoint
        self.index = index
        self.dry_run = dry_run
        self.sink_path = sink_path or f'data/siem_bulk_{index}.ndjson'
        self._sink: list[dict[str, Any]] = []

    def bulk_index(self, docs: Iterable[dict[str, Any]]) -> dict[str, Any]:
        n = 0
        with open(self.sink_path, 'a', encoding='utf-8') if self.dry_run else _Noop() as fh:
            for doc in docs:
                action = json.dumps({'index': {'_index': self.index}})
                body = json.dumps(doc, sort_keys=True)
                if self.dry_run:
                    fh.write(action + '\n' + body + '\n')
                else:
                    if not self.endpoint:
                        raise RuntimeError('live mode requires an endpoint (configs/siem.yaml live.endpoint_env)')
                    raise NotImplementedError('network writes intentionally disabled in this build; use dry_run=True and replay the NDJSON bulk file with curl/_bulk (see README.md, Reproduce)')
                n += 1
        return {'indexed': n, 'dry_run': self.dry_run, 'sink': self.sink_path}

class _Noop:

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, *a):
        pass

@dataclass
class OfflineIndex:
    name: str = 'rlhoneypot-cowrie-ecs'
    docs: list[dict[str, Any]] = field(default_factory=list)

    def add(self, doc: dict[str, Any]) -> None:
        self.docs.append(doc)

    def add_many(self, docs: Iterable[dict[str, Any]]) -> int:
        n = 0
        for d in docs:
            self.add(d)
            n += 1
        return n

    def count_by_state(self) -> list[int]:
        out = [0] * N_STATES
        for d in self.docs:
            st = d.get('rlhoneypot', {}).get('mdp_state')
            if st is not None:
                out[int(st)] += 1
        return out

    def command_docs(self) -> list[dict[str, Any]]:
        return [d for d in self.docs if d.get('event', {}).get('action') == 'cowrie.command.input']

@dataclass
class WazuhRules:
    rule_id_base: int = 100100
    by_state: dict[int, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def from_config(cfg: dict) -> 'WazuhRules':
        base = int(cfg['wazuh']['rule_id_base'])
        by_state = {}
        for r in cfg['wazuh']['rules']:
            by_state[int(r['mapped_state'])] = {'id': base + int(r['id_offset']), 'name': r['name'], 'level': int(r['level'])}
        return WazuhRules(rule_id_base=base, by_state=by_state)

    def alert(self, state: int, ecs_doc: dict[str, Any]) -> dict[str, Any]:
        rule = self.by_state.get(state)
        if rule is None:
            return {}
        return {'@timestamp': ecs_doc.get('@timestamp'), 'rule': {'id': rule['id'], 'name': rule['name'], 'level': rule['level'], 'description': f"rlhoneypot {rule['name']}"}, 'agent': {'type': 'rlhoneypot'}, 'session': ecs_doc.get('session'), 'source': ecs_doc.get('source'), 'event': {'kind': 'alert', 'module': 'rlhoneypot', 'dataset': 'wazuh.alerts'}, 'rlhoneypot': {'mdp_state': int(state)}}

def build_index_from_sessions(labelled_sessions, ecs_timestamps: dict[str, str] | None=None) -> OfflineIndex:
    idx = OfflineIndex()
    ts_map = ecs_timestamps or {}
    for ls in labelled_sessions:
        for ((s, a), cmd_no) in zip(ls.states_actions, range(len(ls.states_actions))):
            doc = {'@timestamp': ts_map.get(ls.session_id, ''), 'event': {'kind': 'event', 'category': ['process'], 'type': ['info'], 'module': 'rlhoneypot', 'dataset': 'cowrie.session', 'action': 'cowrie.command.input'}, 'source': {'ip': ls.src_ip}, 'session': {'id': ls.session_id}, 'process': {'command_line': None}, 'rlhoneypot': {'mdp_state': int(s), 'mdp_action': int(a), 'mapped': True}}
            idx.add(doc)
    return idx

def siem_alert_stream(index: OfflineIndex, rules: WazuhRules, alert_rate_per_state: list[float], noise_rate: float, rng: 'np.random.Generator') -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for doc in index.docs:
        st = doc.get('rlhoneypot', {}).get('mdp_state')
        if st is None:
            continue
        p_alert = alert_rate_per_state[int(st)]
        if rng.random() < p_alert:
            a = rules.alert(int(st), doc)
            if a:
                if rng.random() < noise_rate:
                    a['rlhoneypot']['mdp_state'] = int(rng.integers(0, N_STATES))
                    a['rule']['name'] = 'cowrie_noise_alert'
                alerts.append(a)
    return alerts

def correlation_lift(state_counts_honeypot: list[int], alert_counts_siem: list[int], noise_rate: float) -> dict[str, Any]:
    import numpy as np
    from ..metrics.ci import wilson_interval
    f_H = np.asarray(state_counts_honeypot, dtype=float)
    a_S = np.asarray(alert_counts_siem, dtype=float)
    total_h = f_H.sum()
    total_a = a_S.sum()
    out = {'states': [], 'note': 'lift > 1 = state over-represented in SIEM alerts'}
    marginal = total_a / max(total_h, 1.0)
    for s in range(N_STATES):
        rate = a_S[s] / f_H[s] if f_H[s] else float('nan')
        lift = rate / marginal if marginal and f_H[s] else float('nan')
        (lo, hi) = wilson_interval(a_S[s], f_H[s]) if f_H[s] else (float('nan'), float('nan'))
        out['states'].append({'state': s, 'honeypot_events': int(f_H[s]), 'siem_alerts': int(a_S[s]), 'alert_rate': None if not f_H[s] else float(rate), 'alert_rate_ci95': [lo, hi], 'lift': None if not f_H[s] else float(lift)})
    out['marginal_alert_rate'] = float(marginal)
    out['noise_rate'] = float(noise_rate)
    return out
