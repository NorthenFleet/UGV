import importlib
import sys

base_mod = importlib.import_module('hexapod_robot.sim')
sys.modules['sim'] = base_mod

for name in [
    'pybullet_env',
    'pybullet_env.backend',
    'pybullet_env.envs',
    'pybullet_env.scripts',
    'pybullet_env.config',
    'pybullet_env.assets',
    'ue_bridge',
    'ue_bridge.client',
    'ue_bridge.udp_sender',
    'ue_backend',
    'mujoco_env',
]:
    try:
        m = importlib.import_module(f'hexapod_robot.sim.{name}')
        sys.modules[f'sim.{name}'] = m
    except Exception:
        pass