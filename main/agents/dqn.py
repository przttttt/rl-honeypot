from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False
        self.rng = random.Random(seed)

    def add(self, o, a, r, o2, d) -> None:
        p = self.pos
        self.obs[p] = o
        self.act[p] = a
        self.rew[p] = r
        self.next_obs[p] = o2
        self.done[p] = float(d)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch: int):
        idx = np.array([self.rng.randrange(self.size) for _ in range(batch)])
        return (self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx])

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.pos

class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, seed: int = 0):
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device("cpu")  # D15: CPU for determinism
        hidden = tuple(cfg.get("hidden", (64, 64)))
        self.q = QNet(obs_dim, n_actions, hidden).to(self.device)
        self.target = copy.deepcopy(self.q)
        self.opt = optim.Adam(self.q.parameters(), lr=float(cfg.get("lr", 3e-4)))
        self.buf = ReplayBuffer(int(cfg.get("replay_capacity", 50000)), obs_dim, seed=seed)
        self.gamma = float(cfg.get("gamma", 0.9))
        self.batch = int(cfg.get("batch_size", 64))
        self.warmup = int(cfg.get("warmup_steps", 500))
        self.sync_every = int(cfg.get("target_sync", 200))
        self.grad_clip = float(cfg.get("grad_clip", 10.0))
        self.double_q = bool(cfg.get("double_q", True))
        self.eps_start, self.eps_final = 0.5, float(cfg.get("epsilon_final", 0.05))
        self.eps_decay_steps = int(cfg.get("epsilon_decay_steps", 5000))
        self._step = 0
        self._loss_fn = nn.SmoothL1Loss()

    def _eps(self) -> float:
        frac = min(1.0, self._step / max(1, self.eps_decay_steps))
        return self.eps_start + (self.eps_final - self.eps_start) * frac

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            return self.q(x).squeeze(0).cpu().numpy()

    def act(self, state: int = 0, context: np.ndarray | None = None,
            greedy: bool = False) -> int:
        assert context is not None, "DQN needs a contextual observation"
        if not greedy and random.random() < self._eps():
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_values(context)))

    def act_greedy(self, state: int = 0, context: np.ndarray | None = None) -> int:
        return self.act(state, context, greedy=True)

    def observe(self, o, a, r, o2, d) -> None:
        self.buf.add(o, a, r, o2, d)
        self._step += 1
        if self.buf.size >= self.warmup:
            self._train_step()
        if self._step % self.sync_every == 0:
            self.target.load_state_dict(self.q.state_dict())

    def _train_step(self) -> None:
        o, a, r, o2, d = self.buf.sample(self.batch)
        to = torch.as_tensor(o)
        ta = torch.as_tensor(a)
        tr = torch.as_tensor(r)
        to2 = torch.as_tensor(o2)
        td = torch.as_tensor(d)
        with torch.no_grad():
            if self.double_q:
                best = self.q(to2).argmax(dim=1)
                q2 = self.target(to2).gather(1, best.unsqueeze(1)).squeeze(1)
            else:
                q2 = self.target(to2).max(dim=1).values
            y = tr + self.gamma * (1.0 - td) * q2
        qsa = self.q(to).gather(1, ta.unsqueeze(1)).squeeze(1)
        loss = self._loss_fn(qsa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()

    def behaviour_clone(self, X: np.ndarray, y: np.ndarray, steps: int | None = None,
                        lr: float | None = None) -> list[float]:
        steps = steps or int(self.cfg.get("warm_start_bc_steps", 300))
        lr = lr or float(self.cfg.get("warm_start_lr", 1e-3))
        opt = optim.Adam(self.q.parameters(), lr=lr)
        rng = np.random.default_rng(0)
        losses: list[float] = []
        n = len(X)
        if n == 0:
            return losses
        for _ in range(steps):
            idx = rng.integers(0, n, size=self.batch)
            x = torch.as_tensor(X[idx], dtype=torch.float32)
            a = torch.as_tensor(y[idx], dtype=torch.long)
            loss = nn.functional.cross_entropy(self.q(x), a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.target.load_state_dict(self.q.state_dict())
        return losses

    def epsilon_greedy_probs(self, obs: np.ndarray, epsilon: float) -> np.ndarray:
        qv = self.q_values(obs)
        p = np.full(self.n_actions, epsilon / self.n_actions)
        p[int(np.argmax(qv))] += 1.0 - epsilon
        return p
