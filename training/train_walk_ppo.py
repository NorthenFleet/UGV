import os
import csv
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from hexapod_robot.sim.pybullet_env.envs.hexapod_walk_env import HexapodWalkEnv
from hexapod_robot.sim.pybullet_env.envs.vec_env import SimpleVecEnv
from hexapod_robot.rl.policies.actor_critic import ActorCritic
from hexapod_robot.rl.algorithms.ppo_agent import PPOAgent

def rollout(env, agent, steps=1024):
    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    val_buf = []
    done_buf = []
    obs = env.reset()
    for t in range(steps):
        a, logp, v = agent.act(obs)
        next_obs, r, d, i = env.step(a)
        if obs.ndim == 1:
            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(logp)
            rew_buf.append(r)
            val_buf.append(v)
            done_buf.append(float(d))
        else:
            obs_buf.extend(list(obs))
            act_buf.extend(list(a))
            logp_buf.extend(list(logp))
            rew_buf.extend(list(r))
            val_buf.extend(list(v))
            done_buf.extend(list(d.astype(np.float32)))
        obs = next_obs
    last_v = agent.act(obs)[2]
    if last_v.ndim == 0:
        val_buf.append(last_v)
    else:
        val_buf.extend(list(last_v))
    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.float32),
        np.asarray(logp_buf, dtype=np.float32),
        np.asarray(rew_buf, dtype=np.float32),
        np.asarray(val_buf, dtype=np.float32),
        np.asarray(done_buf, dtype=np.float32),
    )

def main():
    n_env = int(os.environ.get("HEXAPOD_N_ENV", "1"))
    backend = os.environ.get("HEXAPOD_BACKEND", "pybullet")
    if n_env > 1:
        env = SimpleVecEnv(lambda: HexapodWalkEnv(gui=False, dt=0.02, backend=backend), n_env)
    else:
        env = HexapodWalkEnv(gui=False, dt=0.02, backend=backend)
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_dim
    model = ActorCritic(obs_dim, act_dim)
    agent = PPOAgent(model)
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "training_metrics.csv")
    best = None
    writer = SummaryWriter(os.path.join(os.path.dirname(__file__), "../logs/tensorboard")) if SummaryWriter else None
    for it in range(10):
        obs_b, act_b, logp_b, rew_b, val_b, done_b = rollout(env, agent, steps=1024)
        adv_b, ret_b = agent.compute_gae(rew_b, val_b, done_b)
        stats = agent.update(obs_b, act_b, logp_b, ret_b, adv_b, epochs=10, batch_size=128)
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([it, np.mean(rew_b), stats["policy_loss"], stats["value_loss"], stats["entropy"]])
        avg = float(np.mean(rew_b))
        if writer:
            writer.add_scalar("train/avg_reward", avg, it)
            writer.add_scalar("train/policy_loss", stats["policy_loss"], it)
            writer.add_scalar("train/value_loss", stats["value_loss"], it)
            writer.add_scalar("train/entropy", stats["entropy"], it)
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints/walk")
        os.makedirs(ckpt_dir, exist_ok=True)
        path_latest = os.path.join(ckpt_dir, "ckpt_latest.pt")
        torch.save(agent.model.state_dict(), path_latest)
        if best is None or avg > best:
            best = avg
            path_best = os.path.join(ckpt_dir, "best.pt")
            torch.save(agent.model.state_dict(), path_best)
    if writer:
        writer.close()

if __name__ == "__main__":
    main()