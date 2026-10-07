from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
COWRIE_EVENT_NAMES = ('cowrie.session.connect', 'cowrie.login.failed', 'cowrie.login.success', 'cowrie.session.closed', 'cowrie.command.input', 'cowrie.session.file_download', 'cowrie.session.input', 'cowrie.command.command_failed')

@dataclass
class ParsedEvent:
    timestamp: str
    session_id: str
    src_ip: str
    src_port: int
    dest_port: int
    event_name: str
    raw: dict[str, Any] = field(default_factory=dict)

def iter_cowrie_events(path: Path | str) -> Iterator[ParsedEvent]:
    path = Path(path)
    malformed = 0
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            ev = obj.get('eventid', '')
            yield ParsedEvent(timestamp=str(obj.get('timestamp', '')), session_id=str(obj.get('session', '')), src_ip=str(obj.get('src_ip', '')), src_port=int(obj.get('src_port', 0) or 0), dest_port=int(obj.get('dest_port', 0) or 0), event_name=ev, raw=obj)
    if malformed:
        iter_cowrie_events.malformed_lines = malformed

def group_sessions(events: list[ParsedEvent]) -> dict[str, list[ParsedEvent]]:
    sessions: dict[str, list[ParsedEvent]] = {}
    for e in events:
        sessions.setdefault(e.session_id, []).append(e)
    for sid in sessions:
        sessions[sid].sort(key=lambda x: (x.timestamp, x.src_port))
    return sessions

def extract_commands(session_events: list[ParsedEvent]) -> list[str]:
    return [e.raw.get('input', '') for e in session_events if e.event_name == 'cowrie.command.input']

def extract_credentials(session_events: list[ParsedEvent]) -> list[tuple[str, str, bool]]:
    out = []
    for e in session_events:
        if e.event_name in ('cowrie.login.failed', 'cowrie.login.success'):
            out.append((str(e.raw.get('username', '')), str(e.raw.get('password', '')), e.event_name == 'cowrie.login.success'))
    return out

def extract_downloads(session_events: list[ParsedEvent]) -> list[str]:
    return [str(e.raw.get('outfile', e.raw.get('url', ''))) for e in session_events if e.event_name == 'cowrie.session.file_download']

def summarise_session(session_events: list[ParsedEvent]) -> dict[str, Any]:
    cmds = extract_commands(session_events)
    creds = extract_credentials(session_events)
    dls = extract_downloads(session_events)
    return {'session_id': session_events[0].session_id if session_events else '', 'src_ip': session_events[0].src_ip if session_events else '', 'dest_port': session_events[0].dest_port if session_events else 0, 'n_events': len(session_events), 'n_commands': len(cmds), 'n_login_attempts': len(creds), 'n_success_logins': sum((1 for (_, _, ok) in creds if ok)), 'n_downloads': len(dls), 'commands': cmds, 'duration_hint': len(session_events)}
