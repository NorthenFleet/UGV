import os
import argparse
import numpy as np
import torch
from sim.pybullet_env.envs.hexapod_walk_env import HexapodWalkEnv
from hexapod_robot.rl.policies.actor_critic import ActorCritic

class SinusoidGaitAgent:
    def __init__(self, act_dim: int, freq_hz: float = 1.5):
        self.act_dim = act_dim
        self.freq = float(freq_hz)
        self.t = 0.0
        self.dt = 0.02
        self.phases = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi], dtype=np.float32)
        self.amp = np.array([0.2, 0.6, 0.4], dtype=np.float32)

    def act(self, _obs: np.ndarray):
        a = np.zeros(self.act_dim, dtype=np.float32)
        for leg in range(min(6, self.act_dim // 3)):
            s = np.sin(2.0 * np.pi * self.freq * self.t + float(self.phases[leg]))
            c = np.cos(2.0 * np.pi * self.freq * self.t + float(self.phases[leg]))
            base = leg * 3
            a[base + 0] = 0.1 * c * self.amp[0]
            a[base + 1] = s * self.amp[1]
            a[base + 2] = -0.5 * s * self.amp[2]
        self.t += self.dt
        return np.clip(a, -1.0, 1.0)

def load_rl_agent(obs_dim: int, act_dim: int, ckpt_path: str | None):
    model = ActorCritic(obs_dim, act_dim)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        def act_fn(o):
            with torch.no_grad():
                o_t = torch.as_tensor(o, dtype=torch.float32)
                a, _ = model.act(o_t)
                v = model.value(o_t)
            return a.cpu().numpy(), None, v.cpu().numpy()
        agent = type("EvalWrap", (), {})()
        agent.act = act_fn
        return agent
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default=os.environ.get("HEXAPOD_BACKEND", "pybullet"))
    parser.add_argument("--gui", type=int, default=int(os.environ.get("HEXAPOD_GUI", "1")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("HEXAPOD_STEPS", "2000")))
    parser.add_argument("--freq", type=float, default=float(os.environ.get("HEXAPOD_GAIT_FREQ", "1.5")))
    parser.add_argument("--ckpt", type=str, default=os.environ.get("HEXAPOD_CKPT", ""))
    args = parser.parse_args()

    env = HexapodWalkEnv(gui=bool(args.gui), dt=0.02, backend=args.backend)
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_dim
    ckpt = args.ckpt or (
        os.path.join(os.path.dirname(__file__), "../../training/checkpoints/walk/best.pt")
        if os.path.isdir(os.path.join(os.path.dirname(__file__), "../../training/checkpoints/walk")) else ""
    )
    rl_agent = load_rl_agent(obs_dim, act_dim, ckpt)
    gait_agent = SinusoidGaitAgent(act_dim, freq_hz=args.freq)
    for _ in range(args.steps):
        if rl_agent is not None:
            a, _, _ = rl_agent.act(obs)
        else:
            a = gait_agent.act(obs)
        obs, r, d, i = env.step(a)
        if d:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()