#!/usr/bin/env bash
# run a command inside the WSL lab venv from the repo root
cd /mnt/c/Pranoy/HTN/Final/starter
source ~/${LAB:-lab}/bin/activate
eval "$@"
