from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ..mdp.definitions import DEFENDER_RESPONSES, N_DEFENDER_RESPONSES, N_STATES, action_support
from ..mdp.rewards import HARM_BY_NEXT_STATE, reward_table
from ..transitions.estimate import TransitionEstimate
FEATURE_DIM = N_STATES + 3

def parent_of(s: int, k: int) -> int:
    return int(s) // int(k)

def build_scaled(base: TransitionEstimate, k: int, structure: str='shared', seed: int=0, randomised_sigma: float=6.0) -> dict:
    k = int(k)
    if k < 1:
        raise ValueError('k must be >= 1')
    if structure not in ('shared', 'randomised'):
        raise ValueError(f'unknown structure {structure!r}')
    n = N_STATES * k
    probs = np.zeros((n, 7, n), dtype=np.float64)
    support = np.zeros((n, 7), dtype=bool)
    for s in range(n):
        p = s // k
        for a in action_support(p):
            support[s, a] = True
            for p2 in range(N_STATES):
                row_p = float(base.probs[p, a, p2])
                if row_p <= 0.0:
                    continue
                for j2 in range(k):
                    probs[s, a, p2 * k + j2] += row_p / k
    base_r = reward_table()
    reward = np.zeros((n, N_DEFENDER_RESPONSES, n), dtype=np.float64)
    for s in range(n):
        p = s // k
        for a in range(N_DEFENDER_RESPONSES):
            for s2 in range(n):
                reward[s, a, s2] = base_r[p, a, s2 // k]
    if structure == 'randomised':
        rng = np.random.default_rng(seed)
        offset = rng.normal(0.0, randomised_sigma, size=(n, N_DEFENDER_RESPONSES))
        reward += offset[:, :, None]
    return {'probs': probs, 'support': support, 'reward': reward, 'n_states': n, 'k': k}

class ScaledResponseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scaled: dict, max_steps: int=100, obs_mode: str='discrete', start_state: int=0, end_states: tuple[int, ...] | None=None, reward_noise_sigma: float=0.0, seed: int | None=None):
        super().__init__()
        if obs_mode not in ('discrete', 'features'):
            raise ValueError(f'unknown obs_mode {obs_mode!r}')
        self.reward_noise_sigma = float(reward_noise_sigma)
        self.probs = scaled['probs']
        self.support = scaled['support']
        self.R = scaled['reward']
        self.n = int(scaled['n_states'])
        self.k = int(scaled['k'])
        self.max_steps = int(max_steps)
        self.obs_mode = obs_mode
        self.start_state = int(start_state)
        if end_states is None:
            end_states = tuple(range((N_STATES - 1) * self.k, N_STATES * self.k))
        self.end_states = set((int(s) for s in end_states))
        if obs_mode == 'discrete':
            self.observation_space = spaces.Discrete(self.n)
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(FEATURE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_DEFENDER_RESPONSES)
        (self.np_random, _) = gym.utils.seeding.np_random(seed)

    def _obs(self, s: int) -> np.ndarray | int:
        if self.obs_mode == 'discrete':
            return int(s)
        p = s // self.k
        onehot = np.zeros(N_STATES, dtype=np.float32)
        onehot[p] = 1.0
        sub = s % self.k / (self.k - 1) if self.k > 1 else 0.0
        ctx = np.array([sub, self._t / float(self.max_steps), HARM_BY_NEXT_STATE[p] / 5.0], dtype=np.float32)
        return np.concatenate([onehot, ctx])

    def reset(self, *, seed: int | None=None, options: dict | None=None):
        super().reset(seed=seed)
        self._t = 0
        self._s = int(options['start_state']) if options and 'start_state' in options else self.start_state
        return (self._obs(self._s), {'steps': 0})

    def step(self, action: int):
        assert self.action_space.contains(action), f'bad action {action}'
        self._t += 1
        idx = np.flatnonzero(self.support[self._s])
        a_att = int(self.np_random.choice(idx))
        row = self.probs[self._s, a_att]
        next_s = int(self.np_random.choice(self.n, p=row))
        r = float(self.R[self._s, action, next_s])
        if self.reward_noise_sigma > 0.0:
            r += float(self.np_random.normal(0.0, self.reward_noise_sigma))
        terminated = next_s in self.end_states or DEFENDER_RESPONSES[action] == 'terminate'
        truncated = not terminated and self._t >= self.max_steps
        info = {'attacker_action': a_att, 'prev_state': self._s, 'next_state': next_s}
        self._s = next_s
        return (self._obs(next_s), r, bool(terminated), bool(truncated), info)

class ScaledTabularQ:

    def __init__(self, n_states: int, alpha: float=0.1, gamma: float=0.9, epsilon: float=0.1, seed: int | None=None):
        self.n_states = int(n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((self.n_states, N_DEFENDER_RESPONSES), dtype=float)
        self.visited = np.zeros(self.n_states, dtype=bool)

    def act(self, state: int, greedy: bool=False) -> int:
        self.visited[int(state)] = True
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_DEFENDER_RESPONSES))
        q = self.q[int(state)]
        return int(np.flatnonzero(q == q.max())[0])

    def act_greedy(self, state: int) -> int:
        return self.act(state, greedy=True)

    def update(self, s: int, a: int, r: float, s2: int, terminal: bool) -> float:
        target = r if terminal else r + self.gamma * float(self.q[s2].max())
        td = target - float(self.q[s, a])
        self.q[s, a] += self.alpha * td
        return abs(td)

    @property
    def coverage(self) -> float:
        return float(self.visited.mean())
