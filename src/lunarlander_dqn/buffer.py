"""
Replay buffer for LunarLander DQN.

Design choices:
    - Store observations as float32 — LunarLander returns them that way and
      the state is only 8 numbers, so there is no memory pressure like in
      Atari (100_000 * 8 * 4 bytes = 3.2 MB total for the obs array).
    - Store observation and next_observation separately (naive layout).
    - Backing storage is preallocated numpy arrays with a rolling pointer.
      O(1) push, O(batch_size) sample.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", "obs actions rewards next_obs dones")


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: str = "cpu") -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs_dim = int(obs_dim)

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self._idx] = obs
        self.next_obs[self._idx] = next_obs
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        # `done` stored as float32: 1.0 for a real terminal state (Bellman
        # target drops the bootstrap), 0.0 otherwise. Time-limit truncation
        # must be passed as 0.0 by the caller so the bootstrap survives.
        self.dones[self._idx] = float(done)

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        idx = np.random.randint(0, self._size, size=batch_size)
        d = self.device
        return Transition(
            torch.from_numpy(self.obs[idx]).to(d, non_blocking=True),
            torch.from_numpy(self.actions[idx]).to(d, non_blocking=True),
            torch.from_numpy(self.rewards[idx]).to(d, non_blocking=True),
            torch.from_numpy(self.next_obs[idx]).to(d, non_blocking=True),
            torch.from_numpy(self.dones[idx]).to(d, non_blocking=True),
        )
