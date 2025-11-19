#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/src:$(pwd)"
python -m sim.pybullet_env.scripts.visualize_hexapod --gui ${HEXAPOD_GUI:-0} --steps ${HEXAPOD_STEPS:-1000} --freq ${HEXAPOD_GAIT_FREQ:-1.5}