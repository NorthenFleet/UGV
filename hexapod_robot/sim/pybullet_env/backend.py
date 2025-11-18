import os
import math
import numpy as np

class ISimBackend:
    def load_model(self, urdf_path: str, gui: bool = False):
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

class PyBulletBackend(ISimBackend):
    def __init__(self):
        self.p = None
        self.robot = None
        self.plane = None
        self.joint_indices = []
        self.dt = 0.002

    def load_model(self, urdf_path: str, gui: bool = False):
        try:
            import pybullet as p
            import pybullet_data
        except Exception as e:
            raise RuntimeError("pybullet not available") from e
        self.p = p
        cid = p.connect(p.GUI if gui else p.DIRECT)
        if cid < 0:
            raise RuntimeError("pybullet connect failed")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf")
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(urdf_path)
        self.robot = p.loadURDF(urdf_path, [0, 0, 0.2])
        p.setGravity(0, 0, -9.81)
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        base_pos = [0, 0, 0.2]
        base_ori = self.p.getQuaternionFromEuler([0, 0, 0])
        self.p.resetBasePositionAndOrientation(self.robot, base_pos, base_ori)
        for j in self.joint_indices:
            self.p.resetJointState(self.robot, j, 0.0, 0.0)
        return self.get_obs()

    def step(self, action: np.ndarray, dt: float):
        self.dt = dt
        for k, j in enumerate(self.joint_indices):
            self.p.setJointMotorControl2(self.robot, j, self.p.POSITION_CONTROL, targetPosition=float(action[k]))
        self.p.stepSimulation()
        return self.get_obs()

    def get_obs(self) -> dict:
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot)
        q = []
        dq = []
        contacts = []
        for j in self.joint_indices:
            s = self.p.getJointState(self.robot, j)
            q.append(s[0])
            dq.append(s[1])
        q = np.array(q, dtype=np.float32)
        dq = np.array(dq, dtype=np.float32)
        contacts = np.zeros(6, dtype=np.float32)
        return {
            "base_pos": np.array(base_pos, dtype=np.float32),
            "base_ori": np.array(base_ori, dtype=np.float32),
            "lin_vel": np.array(lin_vel, dtype=np.float32),
            "ang_vel": np.array(ang_vel, dtype=np.float32),
            "q": q,
            "dq": dq,
            "contacts": contacts,
        }

    def set_joint_positions(self, q: np.ndarray):
        for k, j in enumerate(self.joint_indices):
            self.p.resetJointState(self.robot, j, float(q[k]), 0.0)

    def get_joint_positions(self) -> np.ndarray:
        vals = []
        for j in self.joint_indices:
            vals.append(self.p.getJointState(self.robot, j)[0])
        return np.array(vals, dtype=np.float32)