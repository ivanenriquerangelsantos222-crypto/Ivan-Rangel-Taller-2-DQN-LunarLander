"""
Q-network for LunarLander-v3 — a compact MLP.

Input:  (batch, 8) float32       # 8 state components (x, y, vx, vy, angle, ω, contact_L, contact_R)
Output: (batch, 4)               # 1 Q-value per discrete action (nothing, left, main, right)

Architecture:
    Linear(  8 → 128) → ReLU     →  (batch, 128)
    Linear(128 → 128) → ReLU     →  (batch, 128)
    Linear(128 →   4)            →  (batch, 4)

Total parameters: 17_924 — three orders of magnitude smaller than the Nature CNN
we would need for Atari. Because the state is already a vector of structured
features (position, velocity, angle...), there is nothing that a convolutional
layer could exploit; a plain MLP is the natural fit.

Why 128 neurons per hidden layer:
    - Enough capacity to model Q over 8 continuous inputs and 4 actions without
      overfitting the transients of DQN training.
    - Two layers, not one: single-layer MLPs cannot express the non-trivial
      interactions between components (e.g. the effect of firing the main
      engine depends on the current angle and vertical velocity together).
    - Not more than 128 or 256: measurable overkill on this problem — the
      literature reports diminishing returns past 128 units on LunarLander.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 8, n_actions: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input arrives as float32 — the LunarLander observation space already
        # gives us continuous values in that dtype, no cast needed.
        return self.net(x)
