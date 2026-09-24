"""
Q-network for Atari — the Nature 2015 architecture.

Input:  (batch, 4, 84, 84) uint8 (four stacked grayscale frames)
Output: (batch, n_actions) real-valued Q(s, ·)

Architecture (Mnih et al., 2015, Nature 518):
    Conv2d(4  →  32, kernel 8, stride 4)  → ReLU   →  (batch, 32, 20, 20)
    Conv2d(32 →  64, kernel 4, stride 2)  → ReLU   →  (batch, 64,  9,  9)
    Conv2d(64 →  64, kernel 3, stride 1)  → ReLU   →  (batch, 64,  7,  7)
    Flatten                                        →  (batch, 3136)
    Linear(3136 → 512)                    → ReLU   →  (batch, 512)
    Linear(512  → n_actions)                       →  (batch, n_actions)

Why these numbers:
    - Kernel 8/stride 4 at the front is a *large* receptive field per pixel;
      the paddle and ball are small, but their trajectories span the frame.
    - Progressive stride reduction (4 → 2 → 1) coarse-to-fine: capture rough
      motion first, then local structure.
    - No pooling: strided convs already downsample, and pooling would throw away
      spatial precision the paddle needs.
    - No batch norm: DQN targets are non-stationary, batch norm is unstable in
      this setting (a well-known result). Layer norm sometimes helps but is not
      standard for the Nature baseline.
    - The final linear head outputs one Q-value per discrete action.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """The Nature 2015 architecture, verbatim."""

    def __init__(self, in_channels: int = 4, n_actions: int = 6) -> None:
        super().__init__()
        self.n_actions = n_actions

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # 84 → 20 → 9 → 7  (each conv shrinks by (input − kernel) / stride + 1)
        # 7 * 7 * 64 = 3136 flat features.
        self.head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frames arrive as uint8 in [0, 255]. Normalise on the fly so the replay
        # buffer stays in uint8 (4x cheaper memory-wise).
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.head(self.features(x))
