"""
DQN agent for Atari.

Encapsulates:
    - Online Q-network (trained) and target Q-network (frozen, sync'd every N steps).
    - Epsilon-greedy action selection with linear decay.
    - Replay buffer (imported).
    - The Bellman learning step (target computed from target_net, no gradients).

Deliberately kept model-agnostic to make Double DQN a one-line change later
(see `_compute_target`).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .network import NatureCNN


@dataclass
class DQNConfig:
    # Environment
    env_id: str = "ALE/Pong-v5"
    n_actions: int = 6
    frame_stack: int = 4

    # Optimization
    lr: float = 2.5e-4
    gamma: float = 0.99
    batch_size: int = 32
    max_grad_norm: float = 10.0  # gradient clipping — DQN targets can spike

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 250_000  # linear anneal length in *agent* steps

    # Replay buffer
    buffer_capacity: int = 100_000
    replay_start_size: int = 10_000  # collect this many transitions before learning

    # Training cadence
    learn_every: int = 4        # gradient step every 4 env steps
    target_update_every: int = 1_000  # hard copy of online → target

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class DQNAgent:
    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.q_net = NatureCNN(in_channels=cfg.frame_stack, n_actions=cfg.n_actions).to(cfg.device)
        self.target_net = deepcopy(self.q_net).to(cfg.device)
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_capacity,
            obs_shape=(cfg.frame_stack, 84, 84),
            device=cfg.device,
        )

        self.step_count = 0  # total env interactions (not learn steps)

    # ------------------------------------------------------------------ action

    def epsilon(self) -> float:
        """Linear anneal from eps_start to eps_end over eps_decay_steps env steps."""
        frac = min(1.0, self.step_count / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, *, deterministic: bool = False) -> int:
        """Epsilon-greedy at train time; pure argmax at eval time."""
        if not deterministic and np.random.random() < self.epsilon():
            return int(np.random.randint(self.cfg.n_actions))
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.cfg.device)
        q = self.q_net(obs_t)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------ learn

    def _compute_target(self, batch) -> torch.Tensor:
        with torch.no_grad():
            next_q = self.target_net(batch.next_obs).max(dim=1).values
            target = batch.rewards + self.cfg.gamma * next_q * (1.0 - batch.dones)
        return target

    def learn(self) -> float | None:
        """One gradient step on a mini-batch. Returns loss, or None if not warm."""
        if len(self.buffer) < self.cfg.replay_start_size:
            return None

        batch = self.buffer.sample(self.cfg.batch_size)
        current_q = self.q_net(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        target_q = self._compute_target(batch)

        loss = F.smooth_l1_loss(current_q, target_q)  # Huber — robust to target spikes
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return float(loss.item())

    def sync_target(self) -> None:
        """Hard copy: target ← online."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------ i/o

    def save(self, path: str) -> None:
        torch.save({
            "cfg": self.cfg.__dict__,
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
