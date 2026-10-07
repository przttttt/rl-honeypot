from __future__ import annotations
import re
from dataclasses import dataclass
from .definitions import ATTACKER_ACTIONS, STATE_TECHNIQUE
RULES: tuple[tuple[str, int, int], ...] = tuple([('\\b(curl|wget|fetch|tftp)\\b|\\bscp\\b\\s+-\\w*\\s+\\S+@', 3, 6), ('(xmrig|minerd|cryptonight|\\bminer\\b|cpulimit|nice\\s+-n)', 5, 5), ('^\\s*(top|htop|free(\\s+-m)?|uptime|lscpu|nproc)\\b', 5, 5), ('\\bchattr\\b|\\bsystemctl\\s+(stop|disable)\\b|\\bservice\\s+\\w+\\s+stop\\b|\\biptables\\b|\\bsetenforce\\b|\\bkillall\\b', 4, 4), ('\\bcrontab\\b|\\buseradd\\b|\\badduser\\b|\\busermod\\b|^\\s*passwd\\b|\\bchpasswd\\b|authorized_keys|sshd_config', 6, 7), ('/etc/(shadow|passwd|sudoers)', 2, 3), ('\\bssh\\b|\\bssh-keygen\\b|\\bsshpass\\b', 1, 1), ('^\\s*sudo\\b|\\bsu\\s+\\S+\\b|\\bchmod\\s+[0-7x+\\-]+\\s+/etc/|\\bchmod\\s+\\+?s\\b', 2, 3), ('^\\s*(chmod|chown|chgrp|mv|cp|rm|mkdir|touch|tee)\\b|^\\s*echo\\b.*>>', 2, 2), ('^(uname|hostname|ifconfig|whoami|id|pwd|ls|env|history|dmesg|netstat|ps)\\b', 0, 0), ('^cat\\s+/proc/cpuinfo|grep\\s+.?model\\s+name|^cat\\s+/etc/(os-release|issue)|^cat\\s+/etc/hostname', 0, 0), ('^(cat|echo|cd)\\b', 0, 0), ('^\\s*login\\b', 1, 1)])
_COMPILED = tuple(((re.compile(pat), a, s) for (pat, a, s) in RULES))

@dataclass(frozen=True)
class Mapping:
    state: int
    attacker_action: int
    rule: str
    mitre: str

class UnmappedCommand(Exception):
    pass

def map_command(command_line: str) -> Mapping:
    line = (command_line or '').strip()
    if not line:
        raise UnmappedCommand('empty command line')
    for (rx, action, state) in _COMPILED:
        if rx.search(line):
            return Mapping(state=state, attacker_action=action, rule=rx.pattern, mitre=STATE_TECHNIQUE[state])
    raise UnmappedCommand(f'no mapping rule for: {line!r}')

def map_or_unknown(command_line: str) -> tuple[int, int, bool]:
    try:
        m = map_command(command_line)
        return (m.state, m.attacker_action, True)
    except UnmappedCommand:
        return (0, 0, False)

def describe_state(state: int) -> str:
    from .definitions import STATE_NAMES
    return f'S{state}:{STATE_NAMES[state]}'

def describe_action(action: int) -> str:
    if action == 6:
        return 'A_stay'
    return f'A{action + 1}:{ATTACKER_ACTIONS[action]}'
