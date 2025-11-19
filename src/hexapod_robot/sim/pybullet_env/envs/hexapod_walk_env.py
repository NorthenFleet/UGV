import os
import numpy as np
from hexapod_robot.sim.pybullet_env.backend import PyBulletBackend
from hexapod_robot.sim.ue_bridge.udp_sender import UDPSender

class HexapodWalkEnv:
    def __init__(self, gui: bool = False, target_speed: float = 0.2, dt: float = 0.02, backend: str = "pybullet"):
        self.dt = dt
        if backend == "pybullet":
            self.backend = PyBulletBackend()
            self.urdf = os.path.join(os.path.dirname(__file__), "../assets/urdf/hexapod.urdf")
            self.backend.load_model(self.urdf, gui=gui)
        elif backend == "ue":
            from hexapod_robot.sim.ue_backend import UEBackend
            self.backend = UEBackend()
            self.backend.load_model("ue", gui=gui)
        else:
            raise ValueError("unknown backend")
        self.action_dim = len(getattr(self.backend, "joint_indices", list(range(18))))
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.target_speed = target_speed
        self.max_tilt = np.deg2rad(15)
        self.q_min, self.q_max = self._load_joint_limits()
        if hasattr(self.backend, "set_friction"):
            self.backend.set_friction(0.8)
        self.max_joint_speed = self._load_max_joint_speed()
        self.max_joint_accel = self._load_max_joint_accel()
        self.slide_threshold = self._load_slide_threshold()
        self.prev_q_cmd = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_dq_cmd = np.zeros(self.action_dim, dtype=np.float32)
        ip = os.environ.get("UE_UDP_IP", "")
        port = int(os.environ.get("UE_UDP_PORT", "50051"))
        self.udp = UDPSender(ip, port, src="sim") if ip else None

    def reset(self, seed: int | None = None):
        obs = self.backend.reset(seed=seed)
        if hasattr(self.backend, "get_foot_positions"):
            self.prev_foot = self.backend.get_foot_positions(relative_to_base=True)
        else:
            self.prev_foot = np.zeros((6, 3), dtype=np.float32)
        if self.udp:
            self.udp.send(obs)
        return self._pack_obs(obs)

    def step(self, action: np.ndarray):
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        q_cmd = 0.5 * (act + 1.0) * (self.q_max - self.q_min) + self.q_min
        dq_des = (q_cmd - self.prev_q_cmd) / max(self.dt, 1e-6)
        ddq_des = (dq_des - self.prev_dq_cmd) / max(self.dt, 1e-6)
        ddq_lim = np.clip(ddq_des, -self.max_joint_accel, self.max_joint_accel)
        dq_cmd = self.prev_dq_cmd + ddq_lim * self.dt
        dq_cmd = np.clip(dq_cmd, -self.max_joint_speed, self.max_joint_speed)
        q_cmd = self.prev_q_cmd + dq_cmd * self.dt
        obs = self.backend.step(q_cmd, dt=self.dt)
        foot = self.backend.get_foot_positions(relative_to_base=True) if hasattr(self.backend, "get_foot_positions") else np.zeros((6, 3), dtype=np.float32)
        o = self._pack_obs(obs)
        r = self._reward(obs, q_cmd, foot)
        d = self._done(obs)
        i = {}
        if self.udp:
            self.udp.send(obs)
        self.prev_action = act
        self.prev_q_cmd = q_cmd
        self.prev_dq_cmd = dq_cmd
        self.prev_foot = foot
        return o, r, d, i

    def close(self):
        try:
            self.backend.disconnect()
        except Exception:
            pass
        if self.udp:
            try:
                self.udp.close()
            except Exception:
                pass

    def _pack_obs(self, obs: dict) -> np.ndarray:
        base_ori = obs["base_ori"]
        lin_vel = obs["lin_vel"]
        ang_vel = obs["ang_vel"]
        q = obs["q"]
        dq = obs["dq"]
        contacts = obs["contacts"]
        foot = self.backend.get_foot_positions(relative_to_base=True).reshape(-1) if hasattr(self.backend, "get_foot_positions") else np.zeros(18, dtype=np.float32)
        return np.concatenate([base_ori, lin_vel, ang_vel, q, dq, contacts, foot], axis=0).astype(np.float32)

    def _reward(self, obs: dict, q_cmd: np.ndarray, foot: np.ndarray) -> float:
        vx = obs["lin_vel"][0]
        r_v = -abs(self.target_speed - vx)
        e = obs["base_euler"]
        tilt = abs(e[0]) + abs(e[1])
        r_ori = -tilt
        r_smooth = -np.mean((q_cmd - self._q_from_action(self.prev_action)) ** 2)
        r_energy = -np.mean(np.abs(obs["dq"]))
        slide = np.linalg.norm((foot - self.prev_foot), axis=1)
        slide_mask = obs["contacts"][: len(slide)]
        slide_excess = np.maximum(0.0, slide - self.slide_threshold)
        r_slide = -np.mean(slide_excess * slide_mask)
        return float(1.0 * r_v + 0.1 * r_ori + 0.01 * r_smooth + 0.001 * r_energy + 0.02 * r_slide)

    def _done(self, obs: dict) -> bool:
        base_pos = obs["base_pos"]
        if base_pos[2] < 0.05:
            return True
        e = obs["base_euler"]
        tilt = abs(e[0]) + abs(e[1])
        return tilt > self.max_tilt

    def _q_from_action(self, act: np.ndarray) -> np.ndarray:
        a = np.clip(act, -1.0, 1.0)
        return 0.5 * (a + 1.0) * (self.q_max - self.q_min) + self.q_min

    def _load_joint_limits(self):
        path = os.path.join(os.path.dirname(__file__), "../config/hexapod_joint_limits.yaml")
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            hip = data.get("hip", {})
            knee = data.get("knee", {})
            ankle = data.get("ankle", {})
            mins = []
            maxs = []
            for _ in range(6):
                mins.extend([hip.get("min", -1.0), knee.get("min", -1.5), ankle.get("min", -1.0)])
                maxs.extend([hip.get("max", 1.0), knee.get("max", 1.5), ankle.get("max", 1.0)])
            return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)
        except Exception:
            mins = -np.ones(self.action_dim, dtype=np.float32)
            maxs = np.ones(self.action_dim, dtype=np.float32)
            return mins, maxs

    def _load_max_joint_speed(self):
        path = os.path.join(os.path.dirname(__file__), "../config/sim_params.yaml")
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return float(data.get("max_joint_speed", 4.0))
        except Exception:
            return 4.0

    def _load_max_joint_accel(self):
        path = os.path.join(os.path.dirname(__file__), "../config/sim_params.yaml")
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return float(data.get("max_joint_accel", 50.0))
        except Exception:
            return 50.0

    def _load_slide_threshold(self):
        path = os.path.join(os.path.dirname(__file__), "../config/sim_params.yaml")
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return float(data.get("slide_threshold", 0.002))
        except Exception:
            return 0.002
 
