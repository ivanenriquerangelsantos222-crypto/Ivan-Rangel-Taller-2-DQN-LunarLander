"""
Replay buffer for Atari DQN.

Design choices:
    - Store observations as uint8 (0-255), not float32. Cuts memory 4x, which is
      the difference between "fits in Colab" and "OOM crash".
    - Store observation and next_observation separately (naive layout). More
      memory than a per-frame circular buffer, but 20 lines of code instead of
      100 and easier to reason about.
    - Backing storage is preallocated numpy arrays, indexed with a rolling
      pointer. O(1) push, O(batch_size) sample.

Sizing (Pong):
    - capacity 100_000 transitions
    - each obs: (4, 84, 84) uint8 = 28_224 bytes = ~28 KB
    - obs + next_obs storage: 100_000 * 28_224 * 2 ≈ 5.6 GB
    - fits in Colab T4 host RAM (12-16 GB) with room to spare.
    - if RAM is tight, drop capacity to 50_000.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", "obs actions rewards next_obs dones")


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: str = "cpu") -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs_shape = tuple(obs_shape)

        self.obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
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
        # Store `done` as a float: 1.0 for a true terminal (Bellman target
        # drops the bootstrap), 0.0 otherwise. Time-limit truncation must be
        # passed as 0.0 by the caller so the bootstrap survives.
        self.dones[self._idx] = float(done)

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        idx = np.random.randint(0, self._size, size=batch_size)
        obs_t = torch.from_numpy(self.obs[idx]).to(self.device, non_blocking=True)
        next_obs_t = torch.from_numpy(self.next_obs[idx]).to(self.device, non_blocking=True)
        actions_t = torch.from_numpy(self.actions[idx]).to(self.device, non_blocking=True)
        rewards_t = torch.from_numpy(self.rewards[idx]).to(self.device, non_blocking=True)
        dones_t = torch.from_numpy(self.dones[idx]).to(self.device, non_blocking=True)
        return Transition(obs_t, actions_t, rewards_t, next_obs_t, dones_t)
