#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/src:$(pwd)"
python training/train_walk_ppo.py