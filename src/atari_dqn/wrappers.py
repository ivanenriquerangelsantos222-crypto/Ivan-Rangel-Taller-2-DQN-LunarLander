"""
Atari preprocessing wrappers.

These are the standard preprocessing steps from Mnih et al. (2015). The raw Atari
frame is 210x160 RGB at 60Hz; feeding that to a network wastes compute and hides
the useful signal. The wrappers turn it into something a CNN can learn from:

    raw (210, 160, 3) RGB uint8
    ├── AtariPreprocessing:
    │     ├── Noop reset: 0-30 random no-ops so the agent doesn't memorize a start
    │     ├── Frame skip 4: agent picks an action, env repeats it 4 times,
    │     │                 max over the last two frames to defeat sprite flicker
    │     ├── Grayscale + resize to 84x84: 3x less input, none of the useful signal lost
    │     └── (Life loss = episode end): does nothing on Pong (no lives), harmless
    ├── FrameStackObservation (k=4):
    │     └── Stack the last 4 processed frames so the agent can see motion.
    │         A single frame tells you where the ball IS but not where it is GOING.
    └── final observation: (4, 84, 84) uint8, dtype-cast in the network to float32/255
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py


def make_atari_env(env_id: str = "ALE/Pong-v5", seed: int | None = None,
                   render_mode: str | None = None) -> gym.Env:
    """Build the fully wrapped Atari environment used for both training and eval.

    We ask gymnasium for the raw ROM (frameskip=1) because AtariPreprocessing does
    its own skip. Passing frameskip=4 here AND letting the wrapper skip would
    double-skip and drop information.
    """
    gym.register_envs(ale_py)

    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        noop_max=30,          # up to 30 no-ops at start; randomises the initial state
        frame_skip=4,         # agent decides once per 4 frames (Atari standard)
        screen_size=84,       # resize final image to 84x84
        terminal_on_life_loss=False,  # Pong has no lives, keeps generality for other games
        grayscale_obs=True,   # single channel, RGB is redundant for Pong
        grayscale_newaxis=False,
        scale_obs=False,      # keep uint8 for cheap replay-buffer storage
    )
    env = FrameStackObservation(env, stack_size=4)
    # After FrameStackObservation the observation is (4, 84, 84) uint8, arranged
    # as (channels, height, width) which is what PyTorch conv layers expect.

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
