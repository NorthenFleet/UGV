import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", type=str, default=os.path.join(os.getcwd(), "src", "data", "六足机器人3d模型.stl"))
    parser.add_argument("--gui", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=10.0)
    args = parser.parse_args()

    from hexapod_robot.sim.pybullet_env.backend import PyBulletBackend
    be = PyBulletBackend()
    # dispatch via load_model, which now supports .stl
    if not os.path.isfile(args.stl):
        raise FileNotFoundError(args.stl)
    be.load_model(args.stl, gui=bool(args.gui))
    # Optionally adjust scale/mass via private method if needed
    # This demo relies on the default scale/mass in backend
    obs = be.get_obs()
    print("loaded base_pos", obs.get("base_pos"))
    for _ in range(240):
        be.step([], dt=1.0/120.0)
    be.disconnect()

if __name__ == "__main__":
    main()