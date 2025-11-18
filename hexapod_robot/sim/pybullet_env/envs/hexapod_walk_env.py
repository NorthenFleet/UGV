import os
import numpy as np

from hexapod_robot.sim.pybullet_env.backend import PyBulletBackend

class HexapodWalkEnv:
    def __init__(self, gui: bool = False, target_speed: float = 0.2, dt: float = 0.02):
        self.dt = dt
        self.backend = PyBulletBackend()
        self.urdf = os.path.join(os.path.dirname(__file__), "../assets/urdf/hexapod.urdf")
        self.backend.load_model(self.urdf, gui=gui)
        self.action_dim = len(self.backend.joint_indices)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.target_speed = target_speed
        self.max_tilt = np.deg2rad(15)

    def reset(self, seed: int | None = None):
        obs = self.backend.reset(seed=seed)
        return self._pack_obs(obs)

    def step(self, action: np.ndarray):
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        q_min = -np.ones(self.action_dim, dtype=np.float32)
        q_max = np.ones(self.action_dim, dtype=np.float32)
        q_cmd = 0.5 * (act + 1.0) * (q_max - q_min) + q_min
        obs = self.backend.step(q_cmd, dt=self.dt)
        o = self._pack_obs(obs)
        r = self._reward(obs, q_cmd)
        d = self._done(obs)
        i = {}
        self.prev_action = act
        return o, r, d, i

    def _pack_obs(self, obs: dict) -> np.ndarray:
        base_ori = obs["base_ori"]
        lin_vel = obs["lin_vel"]
        ang_vel = obs["ang_vel"]
        q = obs["q"]
        dq = obs["dq"]
        contacts = obs["contacts"]
        return np.concatenate([base_ori, lin_vel, ang_vel, q, dq, contacts], axis=0).astype(np.float32)

    def _reward(self, obs: dict, q_cmd: np.ndarray) -> float:
        vx = obs["lin_vel"][0]
        r_v = -abs(self.target_speed - vx)
        ori = obs["base_ori"]
        tilt = abs(ori[0]) + abs(ori[1])
        r_ori = -tilt
        r_smooth = -np.mean((q_cmd - 0.0) ** 2)
        return float(1.0 * r_v + 0.1 * r_ori + 0.01 * r_smooth)

    def _done(self, obs: dict) -> bool:
        base_pos = obs["base_pos"]
        if base_pos[2] < 0.05:
            return True
        ori = obs["base_ori"]
        tilt = abs(ori[0]) + abs(ori[1])
        return tilt > self.max_tilt