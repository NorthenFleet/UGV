import numpy as np

try:
    from hexapod_robot.sim.pybullet_env.backend import ISimBackend
except Exception:
    class ISimBackend:
        def load_model(self, path: str, gui: bool = False):
            raise NotImplementedError
        def reset(self, seed: int | None = None):
            raise NotImplementedError
        def step(self, action: np.ndarray, dt: float):
            raise NotImplementedError
        def get_obs(self) -> dict:
            raise NotImplementedError
        def set_joint_positions(self, q: np.ndarray):
            raise NotImplementedError
        def get_joint_positions(self) -> np.ndarray:
            raise NotImplementedError

class UEBackend(ISimBackend):
    def __init__(self):
        self.client = None
        self.dt = 0.02
        self.n_joints = 18
        self.joint_indices = list(range(self.n_joints))

    def load_model(self, path: str, gui: bool = False):
        try:
            from hexapod_robot.sim.ue_bridge.client import UEClient
        except Exception as e:
            raise RuntimeError("ue bridge not available") from e
        self.client = UEClient()
        self.client.connect()

    def reset(self, seed: int | None = None):
        self.client.reset(seed)
        return self.get_obs()

    def step(self, action: np.ndarray, dt: float):
        self.dt = dt
        self.client.send_action(np.asarray(action, dtype=np.float32))
        return self.get_obs()

    def get_obs(self) -> dict:
        s = self.client.get_state()
        return s

    def set_joint_positions(self, q: np.ndarray):
        self.client.send_action(np.asarray(q, dtype=np.float32))

    def get_joint_positions(self) -> np.ndarray:
        s = self.client.get_state()
        return np.asarray(s.get("q", np.zeros(self.n_joints, dtype=np.float32)), dtype=np.float32)

    def disconnect(self):
        if self.client:
            self.client.close()