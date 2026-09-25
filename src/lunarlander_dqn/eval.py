"""
Evaluate a trained DQN agent on LunarLander-v3 under a fully greedy policy.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .agent import DQNAgent, DQNConfig


def evaluate(agent: DQNAgent, n_episodes: int = 20, seed_offset: int = 10_000,
             env_id: str = "LunarLander-v3") -> dict:
    env = gym.make(env_id)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        R, done = 0.0, False
        while not done:
            a = agent.select_action(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            R += float(r)
            done = term or trunc
        returns.append(R)
    env.close()
    arr = np.array(returns, dtype=np.float32)
    return {
        "returns": arr,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "landings": int((arr >= 200).sum()),
        "n_episodes": n_episodes,
    }


def evaluate_from_checkpoint(checkpoint_path: str, n_episodes: int = 20, **kwargs) -> dict:
    cfg = DQNConfig()
    agent = DQNAgent(cfg)
    agent.load(checkpoint_path)
    return evaluate(agent, n_episodes=n_episodes, **kwargs)
