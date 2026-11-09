#!/bin/bash -l
# (Re)create the aeropt-hpc env on the cluster from environment.yml.
# Run on a LOGIN node (needs internet). Safe to re-run: updates in place.
set -eo pipefail
unset PYTHONHOME
module purge
module load miniconda3/26.5.3-2          # keep in sync with Remote/cluster_config.py
eval "$(conda shell.bash hook)"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if conda env list | awk '{print $1}' | grep -qx aeropt-hpc; then
    conda env update -n aeropt-hpc -f "$HERE/environment.yml" --prune
else
    conda env create -n aeropt-hpc -f "$HERE/environment.yml"
fi
conda activate aeropt-hpc
python -c "import numpy, scipy, vtk, pyvista, geomdl; print('aeropt-hpc OK:', __import__('sys').executable)"
