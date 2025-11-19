import numpy as np
from hexapod_robot.sim.pybullet_env.envs.hexapod_walk_env import HexapodWalkEnv

def main():
    env = HexapodWalkEnv(gui=True, dt=0.02)
    obs = env.reset()
    for _ in range(500):
        a = np.zeros(env.action_dim, dtype=np.float32)
        obs, r, d, i = env.step(a)
        if d:
            break
    env.close()

if __name__ == "__main__":
    main()