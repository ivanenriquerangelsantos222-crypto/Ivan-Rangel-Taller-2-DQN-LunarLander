"""
Evaluate a trained DQN checkpoint on Pong.

For each of `n_episodes`, run the agent in pure-greedy mode from a fresh env
reset and record the raw score (opponent misses minus agent misses; range -21 to +21).
"""
from __future__ import annotations

import numpy as np

from .agent import DQNAgent, DQNConfig
from .wrappers import make_atari_env


def evaluate(agent: DQNAgent, n_episodes: int = 10, seed_offset: int = 10_000,
             env_id: str = "ALE/Pong-v5") -> dict:
    env = make_atari_env(env_id, seed=seed_offset)
    returns = []
    lengths = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        obs = np.asarray(obs)
        done = False
        ep_return = 0.0
        ep_length = 0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs)
            ep_return += float(reward)
            ep_length += 1
            done = terminated or truncated
        returns.append(ep_return)
        lengths.append(ep_length)
    env.close()

    return {
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "wins": int(np.sum(np.array(returns) > 0)),
        "n_episodes": n_episodes,
    }


def evaluate_from_checkpoint(checkpoint_path: str, n_episodes: int = 10, **kwargs) -> dict:
    cfg = DQNConfig()
    agent = DQNAgent(cfg)
    agent.load(checkpoint_path)
    return evaluate(agent, n_episodes=n_episodes, **kwargs)
