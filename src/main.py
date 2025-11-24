import os
import argparse
import yaml
import time
import numpy as np

def _try_load_rl(obs_dim: int, act_dim: int, ckpt: str | None):
    try:
        import torch
        from hexapod_robot.rl.policies.actor_critic import ActorCritic
    except Exception:
        return None
    if not ckpt or not os.path.isfile(ckpt):
        return None
    model = ActorCritic(obs_dim, act_dim)
    state = torch.load(ckpt, map_location="cpu")
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

class TripodAgent:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join("src", "config", "runtime.yaml"))
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        cfg = {}
    else:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    backend = str(cfg.get("backend", "pybullet"))
    gui = bool(cfg.get("gui", True))
    steps = int(cfg.get("steps", 2000))
    dt = float(cfg.get("dt", 0.02))
    target_speed = float(cfg.get("target_speed", 0.2))
    freq = float(cfg.get("gait_freq_hz", 1.5))
    ckpt = str(cfg.get("ckpt", ""))
    model = str(cfg.get("model", ""))
    ue = cfg.get("ue_udp", {}) or {}
    mesh = cfg.get("mesh", {}) or {}
    run_forever = bool(cfg.get("run_forever", False))
    real_time = bool(cfg.get("real_time", True if gui else False))
    cam = cfg.get("camera", {}) or {}

    if model:
        os.environ["HEXAPOD_MODEL"] = model
    ip = str(ue.get("ip", ""))
    port = int(ue.get("port", 50051))
    if ip:
        os.environ["UE_UDP_IP"] = ip
        os.environ["UE_UDP_PORT"] = str(port)
    if "scale" in mesh:
        os.environ["HEXAPOD_MESH_SCALE"] = str(mesh.get("scale"))
    if "mass" in mesh:
        os.environ["HEXAPOD_MESH_MASS"] = str(mesh.get("mass"))
    if "pos" in mesh and isinstance(mesh.get("pos"), (list, tuple)):
        os.environ["HEXAPOD_MESH_POS"] = ",".join(str(float(x)) for x in mesh.get("pos"))
    if "rpy" in mesh and isinstance(mesh.get("rpy"), (list, tuple)):
        os.environ["HEXAPOD_MESH_RPY"] = ",".join(str(float(x)) for x in mesh.get("rpy"))
    if "collision" in mesh:
        os.environ["HEXAPOD_MESH_COLLISION"] = str(mesh.get("collision"))
    if "box_size" in mesh and isinstance(mesh.get("box_size"), (list, tuple)):
        os.environ["HEXAPOD_MESH_BOX_SIZE"] = ",".join(str(float(x)) for x in mesh.get("box_size"))
    if "auto" in cam:
        os.environ["HEXAPOD_CAM_AUTO"] = "1" if cam.get("auto") else "0"
    if "distance" in cam:
        os.environ["HEXAPOD_CAM_DISTANCE"] = str(cam.get("distance"))
    if "yaw" in cam:
        os.environ["HEXAPOD_CAM_YAW"] = str(cam.get("yaw"))
    if "pitch" in cam:
        os.environ["HEXAPOD_CAM_PITCH"] = str(cam.get("pitch"))
    os.environ["HEXAPOD_AABB_WIREFRAME"] = "1"

    from sim.pybullet_env.envs.hexapod_walk_env import HexapodWalkEnv
    env = HexapodWalkEnv(gui=gui, dt=dt, backend=backend)
    env.target_speed = target_speed
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_dim
    rl_agent = _try_load_rl(obs_dim, act_dim, ckpt)
    gait_agent = TripodAgent(act_dim, freq_hz=freq)
    try:
        print("loaded_model", os.environ.get("HEXAPOD_MODEL", ""))
        bi = getattr(env.backend, "joint_indices", [])
        print("backend_info", {"joints": len(bi), "robot_id": getattr(env.backend, "robot", None)})
        dbg = getattr(env.backend, "get_debug_info", lambda: {})()
        print("debug_info", dbg)
    except Exception:
        pass
    try:
        if run_forever:
            while True:
                if rl_agent is not None:
                    a, _, _ = rl_agent.act(obs)
                else:
                    a = gait_agent.act(obs)
                obs, r, d, i = env.step(a)
                try:
                    print("step", {"vx": float(obs[4])})
                except Exception:
                    pass
                if d:
                    obs = env.reset()
                if real_time:
                    time.sleep(dt)
        else:
            for _ in range(steps):
                if rl_agent is not None:
                    a, _, _ = rl_agent.act(obs)
                else:
                    a = gait_agent.act(obs)
                obs, r, d, i = env.step(a)
                try:
                    print("step", {"vx": float(obs[4])})
                except Exception:
                    pass
                if d:
                    obs = env.reset()
                if real_time:
                    time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()