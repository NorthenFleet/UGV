#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)"
python training/train_walk_ppo.py