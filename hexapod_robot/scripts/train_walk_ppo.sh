#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)"
python -m hexapod_robot.rl.training.train_walk_ppo