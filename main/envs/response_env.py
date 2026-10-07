from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ..mdp.definitions import DEFENDER_RESPONSES, N_DEFENDER_RESPONSES, N_STATES
from ..mdp.rewards import reward_table
from ..transitions.estimate import TransitionEstimate

class ResponseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, transitions: TransitionEstimate, max_steps: int=100, start_state: int=0, end_states: tuple[int, ...]=(7,), reward_tensor: np.ndarray | None=None, seed: int | None=None):
        super().__init__()
        self.T = transitions
        self.max_steps = max_steps
        self.start_state = start_state
        self.end_states = set(end_states)
        self.R = reward_table() if reward_tensor is None else reward_tensor
        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_DEFENDER_RESPONSES)
        (self.np_random, _) = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed: int | None=None, options: dict | None=None):
        super().reset(seed=seed)
        self._t = 0
        self._s = int(options['start_state']) if options and 'start_state' in options else self.start_state
        self._terminated = False
        return (self._s, {'steps': 0})

    def step(self, action: int):
        assert self.action_space.contains(action), f'bad action {action}'
        self._t += 1
        a_att = self._sample_attacker_action(self._s)
        row = self.T.probs[self._s, a_att]
        next_s = int(self.np_random.choice(N_STATES, p=row))
        r = float(self.R[self._s, action, next_s])
        terminated = next_s in self.end_states or DEFENDER_RESPONSES[action] == 'terminate'
        truncated = not terminated and self._t >= self.max_steps
        info = {'attacker_action': a_att, 'prev_state': self._s, 'next_state': next_s}
        self._s = next_s
        return (next_s, r, bool(terminated), bool(truncated), info)

    def _sample_attacker_action(self, s: int) -> int:
        support = self.T.support_mask[s]
        idx = np.flatnonzero(support)
        return int(self.np_random.choice(idx))

class ContextualResponseEnv(ResponseEnv):
    CONTEXT_DIM = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(N_STATES + self.CONTEXT_DIM,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        onehot = np.zeros(N_STATES, dtype=np.float32)
        onehot[self._s] = 1.0
        from ..mdp.rewards import HARM_BY_NEXT_STATE
        ctx = np.array([self._t / float(self.max_steps), HARM_BY_NEXT_STATE[self._s] / 5.0, 1.0 if self._s in (5, 6, 7) else 0.0], dtype=np.float32)
        return np.concatenate([onehot, ctx])

    def reset(self, *, seed: int | None=None, options: dict | None=None):
        (s, info) = super().reset(seed=seed, options=options)
        return (self._obs(), info)

    def step(self, action: int):
        (s2, r, term, trunc, info) = super().step(action)
        return (self._obs(), r, term, trunc, info)
