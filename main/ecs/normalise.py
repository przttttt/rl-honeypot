from __future__ import annotations
from typing import Any
EVENT_CATEGORY_BY_COWRIE = {'cowrie.session.connect': ['network'], 'cowrie.login.failed': ['authentication'], 'cowrie.login.success': ['authentication'], 'cowrie.session.closed': ['network'], 'cowrie.command.input': ['process'], 'cowrie.session.file_download': ['file'], 'cowrie.session.input': ['process'], 'cowrie.command.command_failed': ['process']}
EVENT_TYPE_BY_COWRIE = {'cowrie.session.connect': ['connection'], 'cowrie.login.failed': ['start'], 'cowrie.login.success': ['start'], 'cowrie.session.closed': ['end'], 'cowrie.command.input': ['info'], 'cowrie.session.file_download': ['creation'], 'cowrie.session.input': ['info'], 'cowrie.command.command_failed': ['info']}

def to_ecs(event, *, mdp_state: int | None=None, mdp_action: int | None=None, response: int | None=None, mapping_rule: str | None=None, mapped: bool=True) -> dict[str, Any]:
    name = event.event_name
    doc: dict[str, Any] = {'@timestamp': event.timestamp, 'event': {'kind': 'event', 'category': EVENT_CATEGORY_BY_COWRIE.get(name, ['network']), 'type': EVENT_TYPE_BY_COWRIE.get(name, ['info']), 'module': 'rlhoneypot', 'dataset': 'cowrie.session', 'action': name, 'sequence': getattr(event, 'seq', None)}, 'source': {'ip': event.src_ip or None, 'port': event.src_port or None}, 'destination': {'port': event.dest_port or None}, 'session': {'id': event.session_id}, 'ecs': {'version': '8.0.0'}, 'agent': {'type': 'rlhoneypot', 'version': '0.1.0'}}
    if name in ('cowrie.login.failed', 'cowrie.login.success'):
        doc['user'] = {'name': event.raw.get('username'), 'name_raw': None}
        doc['rlhoneypot'] = {'auth': {'success': name == 'cowrie.login.success', 'password': event.raw.get('password')}}
        doc['rlhoneypot']['auth']['password'] = bool(event.raw.get('password'))
    if name == 'cowrie.command.input':
        doc['process'] = {'command_line': event.raw.get('input', '')}
        if mdp_state is not None:
            doc['rlhoneypot'] = {'mdp_state': int(mdp_state), 'mdp_action': None if mdp_action is None else int(mdp_action), 'mapping_rule': mapping_rule, 'mapped': bool(mapped)}
    if name == 'cowrie.session.file_download':
        doc['file'] = {'path': event.raw.get('outfile') or event.raw.get('url')}
        doc['url'] = {'full': event.raw.get('url')}
    if response is not None:
        doc.setdefault('rlhoneypot', {})['response'] = int(response)
    return doc

def ecs_session_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {'@timestamp': summary.get('@timestamp'), 'session': {'id': summary['session_id']}, 'source': {'ip': summary['src_ip']}, 'destination': {'port': summary['dest_port']}, 'event': {'module': 'rlhoneypot', 'dataset': 'cowrie.session', 'kind': 'session_summary'}, 'rlhoneypot': {'n_commands': summary['n_commands'], 'n_login_attempts': summary['n_login_attempts'], 'n_success_logins': summary['n_success_logins'], 'n_downloads': summary['n_downloads'], 'states_visited': summary.get('states_visited', []), 'mdp_final_state': summary.get('mdp_final_state')}}
