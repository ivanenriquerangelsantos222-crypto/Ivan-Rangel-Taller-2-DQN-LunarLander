"""
Training loop for DQN on LunarLander-v3.

Records per-episode return, running mean of the last 100 episodes, and periodic
loss/epsilon so you can plot the curve at the end. Writes checkpoints every
`checkpoint_every` steps. Stops recording a "solved" milestone when the running
mean of the last 100 episodes first crosses `target_solved` (default 200,
the official threshold for LunarLander).
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np

from .agent import DQNAgent, DQNConfig


def train(cfg: DQNConfig,
          total_steps: int = 300_000,
          log_every: int = 2_000,
          checkpoint_every: int = 50_000,
          save_dir: str = "saves",
          target_solved: float = 200.0) -> tuple[DQNAgent, int | None]:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    env = gym.make(cfg.env_id)
    agent = DQNAgent(cfg)

    episode_returns: list[float] = []
    log_steps: list[int] = []
    log_running: list[float] = []
    log_eps: list[float] = []
    log_loss: list[float] = []

    running = deque(maxlen=100)
    recent_losses = deque(maxlen=1000)

    obs, _ = env.reset(seed=cfg.seed)
    ep_return = 0.0
    ep_length = 0
    solved_at: int | None = None
    t0 = time.time()

    for step in range(1, total_steps + 1):
        a = agent.select_action(obs)
        next_obs, r, term, trunc, _ = env.step(a)
        # Only real termination collapses the Bellman bootstrap. Time-limit
        # truncation must be stored as done=0 so the target still bootstraps.
        agent.buffer.push(obs, a, float(r), next_obs, term)
        obs = next_obs
        ep_return += float(r)
        ep_length += 1
        agent.step_count = step

        if step % cfg.learn_every == 0:
            loss = agent.learn()
            if loss is not None:
                recent_losses.append(loss)

        if step % cfg.target_update_every == 0:
            agent.sync_target()

        if term or trunc:
            episode_returns.append(ep_return)
            running.append(ep_return)
            ep_return = 0.0
            ep_length = 0
            obs, _ = env.reset()

        if step % log_every == 0:
            eps = agent.epsilon()
            rmean = float(np.mean(running)) if running else float("nan")
            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            log_steps.append(step)
            log_running.append(rmean)
            log_eps.append(eps)
            log_loss.append(avg_loss)
            elapsed = time.time() - t0
            fps = step / max(elapsed, 1e-9)
            marker = ""
            if rmean >= target_solved and solved_at is None:
                solved_at = step
                marker = " ★ SOLVED (mean100 ≥ 200)"
            print(f"step {step:>7} | ep {len(episode_returns):>4} | "
                  f"return(mean100) {rmean:>+7.2f} | eps {eps:.3f} | "
                  f"loss {avg_loss:.4f} | buffer {len(agent.buffer):>6} | "
                  f"{fps:.0f} step/s | {elapsed/60:.1f} min{marker}", flush=True)

        if step % checkpoint_every == 0:
            agent.save(f"{save_dir}/lunarlander_dqn_{step}.pt")
            _save_history(save_dir, episode_returns, log_steps, log_running,
                          log_eps, log_loss, solved_at)

    agent.save(f"{save_dir}/lunarlander_dqn_final.pt")
    _save_history(save_dir, episode_returns, log_steps, log_running,
                  log_eps, log_loss, solved_at)
    env.close()
    return agent, solved_at


def _save_history(save_dir, episode_returns, log_steps, log_running,
                  log_eps, log_loss, solved_at):
    np.savez(f"{save_dir}/history.npz",
             episode_returns=np.array(episode_returns, dtype=np.float32),
             log_steps=np.array(log_steps, dtype=np.int64),
             log_running_mean=np.array(log_running, dtype=np.float32),
             log_epsilon=np.array(log_eps, dtype=np.float32),
             log_loss=np.array(log_loss, dtype=np.float32),
             solved_at=int(solved_at) if solved_at else -1)
