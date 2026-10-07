from __future__ import annotations
from dataclasses import dataclass
N_STATES = 8
N_ATTACKER_ACTIONS = 6
N_DEFENDER_RESPONSES = 5
DEFENDER_RESPONSES: tuple[str, ...] = ('allow', 'block', 'delay', 'insult', 'terminate')
RESPONSE_INDEX: dict[str, int] = {r: i for (i, r) in enumerate(DEFENDER_RESPONSES)}
STATE_NAMES: tuple[str, ...] = ('initial_reconnaissance', 'initial_compromise', 'establish_foothold', 'privilege_escalation', 'defence_evasion', 'resource_hijacking', 'command_and_control', 'persistence')
ATTACKER_ACTIONS: tuple[str, ...] = ('system_reconnaissance', 'ssh_configuration', 'modify_system_files', 'execute_remote_scripts', 'disable_security_measures', 'resource_manipulation')
A_STAY = 6
STATE_TECHNIQUE: dict[int, str] = {0: 'T1082', 1: 'T1133', 2: 'T1546', 3: 'T1059', 4: 'T1562', 5: 'T1496', 6: 'T1105', 7: 'T1098'}
_SUPPORT: dict[int, tuple[int, ...]] = {0: (0,), 1: (1,), 2: (2,), 3: (2,), 4: (4,), 5: (5,), 6: (3,), 7: (A_STAY,)}

def action_support(state: int) -> tuple[int, ...]:
    return _SUPPORT[state]

def supported(state: int, attacker_action: int) -> bool:
    return attacker_action in action_support(state)

@dataclass(frozen=True)
class Step:
    session: str
    t: int
    state: int
    attacker_action: int
    response: int | None
    next_state: int
    reward: float | None = None
    propensity: float | None = None
    ecs_event_id: str | None = None
