from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path
from ..cowrie.parser import group_sessions, iter_cowrie_events
from ..ecs.normalise import to_ecs
from ..mdp.mapping import map_command
from ..repro.harness import finish, make_context
from ..siem.index import OfflineIndex, WazuhRules

def run(cfg=None, overrides=None, out_root: str='results') -> dict:
    started = time.time()
    ctx = make_context('e1', out_root, overrides)
    cfg = cfg or ctx.cfg
    ph1 = list(iter_cowrie_events(Path('data/raw/p1.json')))
    ph2 = list(iter_cowrie_events(Path('data/raw/p2.json.json')))
    ev_counts = Counter((e.event_name for e in ph1 + ph2))
    mapped_states = Counter()
    mapped_actions = Counter()
    unmapped = 0
    total_cmds = 0
    from ..configs import load_config
    rules = WazuhRules.from_config(load_config('siem'))
    index = OfflineIndex(name=cfg.get('siem', {}).get('offline', {}).get('index', 'rlhoneypot-cowrie-ecs'))
    for ev in ph1 + ph2:
        if ev.event_name == 'cowrie.command.input':
            total_cmds += 1
            try:
                m = map_command(ev.raw.get('input', ''))
                mapped_states[m.state] += 1
                mapped_actions[m.attacker_action] += 1
                doc = to_ecs(ev, mdp_state=m.state, mdp_action=m.attacker_action, mapping_rule=m.rule, mapped=True)
            except Exception:
                unmapped += 1
                doc = to_ecs(ev, mdp_state=0, mdp_action=0, mapped=False, mapping_rule=None)
            index.add(doc)
        elif ev.event_name in ('cowrie.login.failed', 'cowrie.login.success'):
            index.add(to_ecs(ev))
    ecs_docs = len(index.docs)
    result = {'experiment': 'e1_parse', 'provenance': 'synthetic', 'status': 'ok', 'sessions_phase1': len(group_sessions(ph1)), 'sessions_phase2': len(group_sessions(ph2)), 'cowrie_event_counts': dict(ev_counts.most_common()), 'commands_total': total_cmds, 'commands_mapped': total_cmds - unmapped, 'commands_unmapped': unmapped, 'unmapped_rate': round(unmapped / max(1, total_cmds), 5), 'state_histogram': dict(sorted(mapped_states.items())), 'action_histogram': {f'A{a + 1}' if a < 6 else 'A_stay': n for (a, n) in sorted(mapped_actions.items())}, 'ecs_documents': ecs_docs, 'wazuh_rules': {str(s): rules.by_state.get(s, {}).get('id') for s in range(8)}, 'decision_refs': ['D2', 'D10']}
    return finish(ctx, result, started)
if __name__ == '__main__':
    print(run()['ecs_documents'])
