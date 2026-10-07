from __future__ import annotations
import os
from typing import Protocol
import numpy as np
from ..mdp.definitions import DEFENDER_RESPONSES, RESPONSE_INDEX

class SandboxViolation(RuntimeError):
    pass

class SafetyViolation(RuntimeError):
    pass

def require_sandbox(env_var: str='rlhoneypot_SANDBOX') -> None:
    if os.environ.get(env_var, '') != '1':
        raise SandboxViolation(f'{env_var}=1 must be set: rlhoneypot simulates attacker/defender behaviour and must never be run against live infrastructure.')

class _PolicyLike(Protocol):

    def act(self, state: int, context: np.ndarray | None=None) -> int:
        ...

class SafeAgentWrapper:

    def __init__(self, agent: _PolicyLike, block_allow_states: tuple[int, ...]=(5, 6, 7), insult_cap: int | None=2, enabled: bool=True):
        self.agent = agent
        self.block_allow_states = set(block_allow_states) if enabled else set()
        self.insult_cap = insult_cap if enabled and insult_cap is not None else None
        self._insults_this_session = 0
        self.violation_counts = {'allow_filtered': 0, 'insult_filtered': 0}

    def reset_session(self) -> None:
        self._insults_this_session = 0

    def act(self, state: int, context: np.ndarray | None=None) -> int:
        a = int(self.agent.act(state, context))
        a = self._filter(state, a)
        return a

    def act_greedy(self, state: int, context: np.ndarray | None=None) -> int:
        a = int(getattr(self.agent, 'act_greedy', self.agent.act)(state, context))
        return self._filter(state, a)

    def _filter(self, state: int, a: int) -> int:
        name = DEFENDER_RESPONSES[a]
        if name == 'allow' and state in self.block_allow_states:
            self.violation_counts['allow_filtered'] += 1
            return RESPONSE_INDEX['block']
        if name == 'insult' and self.insult_cap is not None:
            if self._insults_this_session >= self.insult_cap:
                self.violation_counts['insult_filtered'] += 1
                return RESPONSE_INDEX['delay']
            self._insults_this_session += 1
        return a
INSULT_MESSAGES: dict[str, str] = {'chmod': "Changing permissions won't change your chances", 'file_modification': "Trying to modify files? Too bad it's not allowed", 'deletion': "Delete away; it's not like you'll succeed", 'top': "Monitoring processes? Here's one: 'YOU' failed", 'resource_query': 'System resources? Better save some dignity instead', 'general': "Nice try, but that's not going to work"}

def insult_for(command_line: str) -> str:
    c = (command_line or '').lower()
    if c.startswith('chmod'):
        return INSULT_MESSAGES['chmod']
    if 'rm' in c:
        return INSULT_MESSAGES['deletion']
    if c.startswith('top') or c.startswith('htop'):
        return INSULT_MESSAGES['top']
    if 'free' in c or 'lscpu' in c or 'cpu' in c:
        return INSULT_MESSAGES['resource_query']
    if any((k in c for k in ('/etc/', 'crontab', 'passwd', 'shadow'))):
        return INSULT_MESSAGES['file_modification']
    return INSULT_MESSAGES['general']
