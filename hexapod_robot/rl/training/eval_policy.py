import os
import numpy as np
import torch

from hexapod_robot.sim.pybullet_env.envs.hexapod_walk_env import HexapodWalkEnv
from hexapod_robot.rl.policies.actor_critic import ActorCritic

def run_episode(env, agent, max_steps=1000):
    obs = env.reset()
    total = 0.0
    for t in range(max_steps):
        a, _, _ = agent.act(obs)
        obs, r, d, i = env.step(a)
        total += r
        if d:
            break
    return total

def main():
    env = HexapodWalkEnv(gui=False, dt=0.02)
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_dim
    model = ActorCritic(obs_dim, act_dim)
    agent = type("EvalWrap", (), {})()
    agent.model = model
    agent.device = "cpu"
    def act_fn(o):
        with torch.no_grad():
            o_t = torch.as_tensor(o, dtype=torch.float32)
            a, _ = model.act(o_t)
            v = model.value(o_t)
        return a.cpu().numpy(), None, v.cpu().numpy()
    agent.act = act_fn
    ckpt_dir = os.path.join(os.path.dirname(__file__), "../checkpoints/walk")
    best = os.path.join(ckpt_dir, "best.pt")
    latest = os.path.join(ckpt_dir, "ckpt_latest.pt")
    path = best if os.path.isfile(best) else latest
    if os.path.isfile(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    scores = []
    for _ in range(5):
        scores.append(run_episode(env, agent, max_steps=1000))
    avg = float(np.mean(scores))
    print("eval_avg_reward", avg)

if __name__ == "__main__":
    main()