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
        self.cid = None
        self.robot = None
        self.plane = None
        self.joint_indices = []
        self.foot_links = []
        self.dt = 0.002
        self.gravity = -9.81
        self.friction = 0.8
        self.friction_range = None

    def load_model(self, urdf_path: str, gui: bool = False):
        try:
            import pybullet as p
            import pybullet_data
        except Exception as e:
            raise RuntimeError("pybullet not available") from e
        self.p = p
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if self.cid < 0:
            raise RuntimeError("pybullet connect failed")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(urdf_path)
        ext = os.path.splitext(urdf_path)[1].lower()
        if ext == ".urdf":
            self.robot = p.loadURDF(urdf_path, [0, 0, 0.2], physicsClientId=self.cid)
        elif ext in (".stl", ".obj"):
            self._load_mesh_model(urdf_path)
        else:
            self.robot = p.loadURDF(urdf_path, [0, 0, 0.2], physicsClientId=self.cid)
        self._apply_sim_params()
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.cid)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        if len(self.joint_indices) >= 18:
            nlegs = min(6, len(self.joint_indices) // 3)
            self.foot_links = [self.joint_indices[i * 3 + 2] for i in range(nlegs)]

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        base_pos = [0, 0, 0.2]
        base_ori = self.p.getQuaternionFromEuler([0, 0, 0])
        self.p.resetBasePositionAndOrientation(self.robot, base_pos, base_ori, physicsClientId=self.cid)
        for j in self.joint_indices:
            self.p.resetJointState(self.robot, j, 0.0, 0.0, physicsClientId=self.cid)
        if self.friction_range is not None:
            f = float(np.random.uniform(self.friction_range[0], self.friction_range[1]))
            self.set_friction(f)
        else:
            self._apply_sim_params()
        return self.get_obs()

    def step(self, action: np.ndarray, dt: float):
        self.dt = dt
        for k, j in enumerate(self.joint_indices):
            self.p.setJointMotorControl2(self.robot, j, self.p.POSITION_CONTROL, targetPosition=float(action[k]), physicsClientId=self.cid)
        self.p.stepSimulation(physicsClientId=self.cid)
        return self.get_obs()

    def get_obs(self) -> dict:
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot, physicsClientId=self.cid)
        q = []
        dq = []
        contacts = []
        for j in self.joint_indices:
            s = self.p.getJointState(self.robot, j, physicsClientId=self.cid)
            q.append(s[0])
            dq.append(s[1])
        q = np.array(q, dtype=np.float32)
        dq = np.array(dq, dtype=np.float32)
        contacts = np.zeros(6, dtype=np.float32)
        for i, link_idx in enumerate(self.foot_links[:6]):
            pts = self.p.getContactPoints(self.robot, self.plane, linkIndexA=link_idx, physicsClientId=self.cid)
            contacts[i] = 1.0 if len(pts) > 0 else 0.0
        euler = np.array(self.p.getEulerFromQuaternion(base_ori), dtype=np.float32)
        return {
            "base_pos": np.array(base_pos, dtype=np.float32),
            "base_ori": np.array(base_ori, dtype=np.float32),
            "base_euler": euler,
            "lin_vel": np.array(lin_vel, dtype=np.float32),
            "ang_vel": np.array(ang_vel, dtype=np.float32),
            "q": q,
            "dq": dq,
            "contacts": contacts,
        }

    def set_friction(self, lateral=0.8):
        self.p.changeDynamics(self.plane, -1, lateralFriction=float(lateral), physicsClientId=self.cid)
        for j in range(-1, self.p.getNumJoints(self.robot, physicsClientId=self.cid)):
            self.p.changeDynamics(self.robot, j, lateralFriction=float(lateral), physicsClientId=self.cid)

    def get_foot_positions(self, relative_to_base=True):
        pos = []
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
        for link_idx in self.foot_links[:6]:
            ls = self.p.getLinkState(self.robot, link_idx, computeForwardKinematics=True, physicsClientId=self.cid)
            p_w = np.array(ls[0], dtype=np.float32)
            if relative_to_base:
                pos.append(p_w - np.array(base_pos, dtype=np.float32))
            else:
                pos.append(p_w)
        return np.stack(pos, axis=0) if pos else np.zeros((6, 3), dtype=np.float32)

    def _apply_sim_params(self):
        try:
            import yaml
            cfg_path = os.path.join(os.path.dirname(__file__), "../config/sim_params.yaml")
            with open(cfg_path, "r") as f:
                data = yaml.safe_load(f)
            self.gravity = float(data.get("gravity", -9.81))
            self.dt = float(data.get("time_step", 0.002))
            self.friction = float(data.get("friction", 0.8))
            fr = data.get("friction_range", None)
            if isinstance(fr, list) and len(fr) == 2:
                self.friction_range = [float(fr[0]), float(fr[1])]
        except Exception:
            pass
        self.p.setGravity(0, 0, self.gravity, physicsClientId=self.cid)
        try:
            self.p.setTimeStep(self.dt, physicsClientId=self.cid)
        except Exception:
            pass
        self.set_friction(self.friction)

    def set_joint_positions(self, q: np.ndarray):
        for k, j in enumerate(self.joint_indices):
            self.p.resetJointState(self.robot, j, float(q[k]), 0.0, physicsClientId=self.cid)

    def get_joint_positions(self) -> np.ndarray:
        vals = []
        for j in self.joint_indices:
            vals.append(self.p.getJointState(self.robot, j, physicsClientId=self.cid)[0])
        return np.array(vals, dtype=np.float32)

    def disconnect(self):
        if self.p is not None and self.cid is not None:
            try:
                self.p.disconnect(self.cid)
            except Exception:
                pass

    def _load_mesh_model(self, mesh_path: str, scale: float = 1.0, mass: float = 10.0):
        p = self.p
        vs = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=mesh_path, meshScale=[scale, scale, scale], physicsClientId=self.cid)
        cs = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=mesh_path, meshScale=[scale, scale, scale], physicsClientId=self.cid)
        self.robot = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=cs, baseVisualShapeIndex=vs, basePosition=[0, 0, 0.2], physicsClientId=self.cid)
        self.joint_indices = []
        self.foot_links = []