import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", type=str, default=os.path.join(os.getcwd(), "src", "data", "六足机器人3d模型.stl"))
    parser.add_argument("--gui", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=10.0)
    parser.add_argument("--rpy", type=str, default="0,0,0", help="base orientation in radians: roll,pitch,yaw")
    parser.add_argument("--pos", type=str, default="0,0,0.2", help="base position: x,y,z")
    args = parser.parse_args()

    sys.path.append(os.path.join(os.getcwd(), "src"))
    from sim.pybullet_env.backend import PyBulletBackend
    be = PyBulletBackend()
    if not os.path.isfile(args.stl):
        raise FileNotFoundError(args.stl)
    os.environ["HEXAPOD_MESH_SCALE"] = str(args.scale)
    os.environ["HEXAPOD_MESH_MASS"] = str(args.mass)
    os.environ["HEXAPOD_MESH_RPY"] = args.rpy
    os.environ["HEXAPOD_MESH_POS"] = args.pos
    be.load_model(args.stl, gui=bool(args.gui))
    obs = be.get_obs()
    print("loaded base_pos", obs.get("base_pos"))
    try:
        while True:
            be.step([], dt=1.0/120.0)
    except KeyboardInterrupt:
        pass
    finally:
        be.disconnect()

if __name__ == "__main__":
    main()