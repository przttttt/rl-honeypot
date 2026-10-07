
from __future__ import annotations

from dataclasses import dataclass

from ..cowrie.parser import ParsedEvent, extract_commands, group_sessions
from ..mdp.definitions import Step
from ..mdp.mapping import map_or_unknown

@dataclass
class LabelledSession:
    session_id: str
    src_ip: str
    dest_port: int
    states_actions: list[tuple[int, int]]      
    transitions: list[tuple[int, int, int]]    
    n_unmapped: int
    n_commands: int

def label_events(events: list[ParsedEvent]) -> list[LabelledSession]:
    sessions = group_sessions(events)
    out: list[LabelledSession] = []
    for sid, evs in sessions.items():
        cmds = extract_commands(evs)
        sa: list[tuple[int, int]] = []
        unmapped = 0
        for c in cmds:
            s, a, ok = map_or_unknown(c)
            sa.append((s, a))
            if not ok:
                unmapped += 1
        tr: list[tuple[int, int, int]] = []
        if sa:
            cur_s, cur_a = sa[0]
            for s, a in sa[1:]:
                if s != cur_s:
                    tr.append((cur_s, cur_a, s))
                    cur_s, cur_a = s, a
        out.append(LabelledSession(
            session_id=sid,
            src_ip=evs[0].src_ip,
            dest_port=evs[0].dest_port,
            states_actions=sa,
            transitions=tr,
            n_unmapped=unmapped,
            n_commands=len(cmds),
        ))
    return out

def steps_from_logged(session_id: str, labelled: LabelledSession,
                      responses: list[int], rewards: list[float],
                      propensities: list[float] | None = None) -> list[Step]:
    assert len(responses) == len(rewards) == max(0, len(labelled.transitions))
    steps: list[Step] = []
    for i, (s, a, s2) in enumerate(labelled.transitions):
        steps.append(Step(
            session=session_id, t=i, state=s, attacker_action=a,
            response=responses[i], next_state=s2, reward=float(rewards[i]),
            propensity=None if propensities is None else float(propensities[i]),
        ))
    return steps
