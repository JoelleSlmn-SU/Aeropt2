#!/bin/bash
set -euo pipefail
source ~/.bashrc
"/lustrehome/home/s.2268086/scratch/s.2268086/mamba-root/envs/aeropt-py310/bin/python" -m pip install -U pip==23.2.1 wheel setuptools
"/lustrehome/home/s.2268086/scratch/s.2268086/mamba-root/envs/aeropt-py310/bin/python" -m pip install -r "/lustrehome/home/s.2268086/aeropt/Scripts/requirements_hpc.txt" --upgrade --no-cache-dir
