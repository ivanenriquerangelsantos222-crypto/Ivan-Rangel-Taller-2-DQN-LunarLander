"""
Training loop for DQN on Atari.

Records per-episode return, running mean of the last 100 episodes, and periodic
loss/epsilon so you can plot the curve at the end. Writes checkpoints every
`checkpoint_every` steps so a Colab session cut mid-run is not fatal.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np

from .agent import DQNAgent, DQNConfig
from .wrappers import make_atari_env


def train(cfg: DQNConfig,
          total_steps: int = 1_500_000,
          log_every: int = 5_000,
          checkpoint_every: int = 100_000,
          save_dir: str = "saves",
          history_path: str | None = "saves/history.npz") -> dict:
    """Full training loop. Returns a dict of arrays ready to plot."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    env = make_atari_env(cfg.env_id, seed=cfg.seed)
    agent = DQNAgent(cfg)

    # Log buffers
    episode_returns: list[float] = []      # one entry per episode
    episode_lengths: list[int] = []
    log_steps: list[int] = []
    log_running_mean: list[float] = []     # running mean of last 100 returns
    log_epsilon: list[float] = []
    log_loss: list[float] = []

    ep_return = 0.0
    ep_length = 0
    running = deque(maxlen=100)
    recent_losses = deque(maxlen=1000)

    obs, _ = env.reset(seed=cfg.seed)
    obs = np.asarray(obs)  # (4, 84, 84) uint8
    t0 = time.time()

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.asarray(next_obs)

        # Only real termination should zero out the bootstrap; time-limit
        # truncation must not, otherwise the agent learns a wrong horizon.
        done_bootstrap = terminated

        agent.buffer.push(obs, action, float(reward), next_obs, done_bootstrap)
        obs = next_obs
        ep_return += float(reward)
        ep_length += 1
        agent.step_count = step

        # Learn
        if step % cfg.learn_every == 0:
            loss = agent.learn()
            if loss is not None:
                recent_losses.append(loss)

        # Sync target
        if step % cfg.target_update_every == 0:
            agent.sync_target()

        # Episode boundary
        if terminated or truncated:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            running.append(ep_return)
            ep_return = 0.0
            ep_length = 0
            obs, _ = env.reset()
            obs = np.asarray(obs)

        # Log
        if step % log_every == 0:
            eps = agent.epsilon()
            rmean = float(np.mean(running)) if running else float("nan")
            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            log_steps.append(step)
            log_running_mean.append(rmean)
            log_epsilon.append(eps)
            log_loss.append(avg_loss)
            elapsed = time.time() - t0
            fps = step / max(elapsed, 1e-9)
            print(f"step {step:>8} | ep {len(episode_returns):>5} | "
                  f"return(mean100) {rmean:>+6.2f} | eps {eps:.3f} | "
                  f"loss {avg_loss:.4f} | buffer {len(agent.buffer):>6} | "
                  f"{fps:.0f} steps/s | {elapsed/60:.1f} min", flush=True)

        # Checkpoint
        if step % checkpoint_every == 0:
            agent.save(f"{save_dir}/pong_dqn_{step}.pt")
            _save_history(history_path, episode_returns, episode_lengths,
                          log_steps, log_running_mean, log_epsilon, log_loss)

    # Final save
    agent.save(f"{save_dir}/pong_dqn_final.pt")
    history = _save_history(history_path, episode_returns, episode_lengths,
                            log_steps, log_running_mean, log_epsilon, log_loss)
    env.close()
    return history


def _save_history(path, episode_returns, episode_lengths,
                  log_steps, log_running_mean, log_epsilon, log_loss) -> dict:
    hist = {
        "episode_returns": np.array(episode_returns, dtype=np.float32),
        "episode_lengths": np.array(episode_lengths, dtype=np.int32),
        "log_steps": np.array(log_steps, dtype=np.int64),
        "log_running_mean": np.array(log_running_mean, dtype=np.float32),
        "log_epsilon": np.array(log_epsilon, dtype=np.float32),
        "log_loss": np.array(log_loss, dtype=np.float32),
    }
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **hist)
    return hist
