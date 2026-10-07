from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from ..mdp.definitions import Step, action_support
from ..mdp.rewards import reward_table
from ..transitions.estimate import TransitionEstimate

@dataclass
class OPELog:
    states: list[int] = None
    actions: list[int] = None
    propensities: list[float] = None
    rewards: list[float] = None
    next_states: list[int] = None
    dones: list[bool] = None
    sessions: list[str] = None
    q_values: list[list[float]] = None
    state_values: list[float] = None

    def __post_init__(self):
        defaults = (self.states, self.actions, self.propensities, self.rewards, self.next_states, self.dones, self.sessions, self.q_values, self.state_values)
        if all((d is None for d in defaults)):
            self.states = []
            self.actions = []
            self.propensities = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.sessions = []
            self.q_values = []
            self.state_values = []
        else:
            for name in ('states', 'actions', 'propensities', 'rewards', 'next_states', 'dones', 'sessions', 'q_values', 'state_values'):
                if getattr(self, name) is None:
                    setattr(self, name, [])

    def to_records(self) -> list[dict]:
        n = len(self.states)
        return [{'session': self.sessions[i], 't': i, 'state': self.states[i], 'action': self.actions[i], 'propensity': self.propensities[i], 'reward': self.rewards[i], 'next_state': self.next_states[i], 'done': self.dones[i], 'q_values': self.q_values[i], 'v_state': self.state_values[i]} for i in range(n)]

def log_randomised_policy(sessions_states_actions: list[list[tuple[int, int]]], transitions: TransitionEstimate, policy_q: np.ndarray, epsilon: float, rng: np.random.Generator, reward_tensor: np.ndarray | None=None, session_prefix: str='log', max_steps: int=100) -> OPELog:
    R = reward_table() if reward_tensor is None else reward_tensor
    n_actions = policy_q.shape[1]
    log = OPELog([], [], [], [], [], [], [], [])
    for (si, sa) in enumerate(sessions_states_actions):
        sid = f'{session_prefix}-{si:06d}'
        cur = sa[0][0] if sa else 0
        for t in range(max_steps):
            q = policy_q[cur]
            best = int(np.flatnonzero(q == q.max())[0])
            p = np.full(n_actions, epsilon / n_actions)
            p[best] += 1.0 - epsilon
            a = int(rng.choice(n_actions, p=p))
            att = _sample_attacker(transitions, cur, rng)
            s2 = int(rng.choice(transitions.probs.shape[2], p=transitions.probs[cur, att]))
            r = float(R[cur, a, s2])
            done = s2 == 7 or DEFENDER_TERMINATE(a)
            log.states.append(int(cur))
            log.actions.append(a)
            log.propensities.append(float(p[a]))
            log.rewards.append(r)
            log.next_states.append(int(s2))
            log.dones.append(bool(done))
            log.sessions.append(sid)
            log.q_values.append([float(x) for x in q])
            log.state_values.append(float(q.max()))
            if done:
                break
            cur = s2
    return log

def DEFENDER_TERMINATE(a: int) -> bool:
    from ..mdp.definitions import DEFENDER_RESPONSES
    return DEFENDER_RESPONSES[a] == 'terminate'

def _sample_attacker(transitions: TransitionEstimate, s: int, rng: np.random.Generator) -> int:
    support = np.flatnonzero(transitions.support_mask[s])
    if support.size == 0:
        return 0
    return int(rng.choice(support))

def support_check(log: OPELog, n_actions: int, min_actions_covered: int) -> dict:
    counts = np.bincount(log.actions, minlength=n_actions)
    covered = int((counts > 0).sum())
    return {'action_counts': counts.tolist(), 'actions_covered': covered, 'min_required': min_actions_covered, 'support_ok': covered >= min_actions_covered, 'min_propensity': min(log.propensities) if log.propensities else 0.0}
