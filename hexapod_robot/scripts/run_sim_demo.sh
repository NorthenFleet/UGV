#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)"
python -m hexapod_robot.sim.pybullet_env.scripts.visualize_hexapod