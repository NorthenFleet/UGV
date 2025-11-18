import os
import csv
import numpy as np
import torch

from hexapod_robot.sim.pybullet_env.envs.hexapod_walk_env import HexapodWalkEnv
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
        obs_buf.append(obs)
        act_buf.append(a)
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        done_buf.append(float(d))
        obs = next_obs if not d else env.reset()
    val_buf.append(agent.act(obs)[2])
    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.float32),
        np.asarray(logp_buf, dtype=np.float32),
        np.asarray(rew_buf, dtype=np.float32),
        np.asarray(val_buf, dtype=np.float32),
        np.asarray(done_buf, dtype=np.float32),
    )

def main():
    env = HexapodWalkEnv(gui=False, dt=0.02)
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_dim
    model = ActorCritic(obs_dim, act_dim)
    agent = PPOAgent(model)
    log_path = os.path.join(os.path.dirname(__file__), "../logs/training_metrics.csv")
    for it in range(10):
        obs_b, act_b, logp_b, rew_b, val_b, done_b = rollout(env, agent, steps=1024)
        adv_b, ret_b = agent.compute_gae(rew_b, val_b, done_b)
        stats = agent.update(obs_b, act_b, logp_b, ret_b, adv_b, epochs=10, batch_size=128)
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([it, np.mean(rew_b), stats["policy_loss"], stats["value_loss"], stats["entropy"]])

if __name__ == "__main__":
    main()