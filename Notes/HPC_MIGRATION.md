# HPC migration (Nov 2026: /home -> /pixhome, new module tree)

Everything cluster-specific now lives in **`Remote/cluster_config.py`**.
On the cluster you can override any key without editing code by writing
`~/aeropt/cluster_config.json` (outside the repo, so `git pull` never clobbers it).

## One-off setup on the new cluster (login node)

```bash
# 1. stop the old self-built Python hijacking jobs
grep -nE "localpython|\.opt/python|PYTHONHOME" ~/.bashrc ~/.bash_profile ~/.profile
#    -> comment those lines out, log out and back in

# 2. main env (python 3.10, same pins as before)
#    (code: apply aeropt_hpc_migration.patch with `git am --keep-cr`, or pull once merged)
cd ~/aeropt/Scripts && bash install_hpc_reqs.sh

# 3. ParaView for the solver monitors (no paraview module on this system)
module load miniconda3/26.5.3-2 && eval "$(conda shell.bash hook)"
conda create -n aeropt-pv -c conda-forge --override-channels "paraview=*=*osmesa*" -y
#    if that build string doesn't resolve, try "paraview=*=*egl*" or plain paraview;
#    the preflight tells you whether it runs headless.

# 4. find the FLITE / mesher binaries and record any that moved
ls -l /home/s.o.hassan /pixhome/s.o.hassan /home/s.engevabj /pixhome/s.engevabj 2>&1 | head
#    then e.g. ~/aeropt/cluster_config.json:
#    { "solver_exe": "/pixhome/s.engevabj/codes/FLITE_uns/UnsMgnsg3d", ... }

# 5. check everything, on a compute node
srun -n1 -t 00:15:00 bash -l Utilities/hpc_preflight.sh
```

Do not submit a morph/optimisation until the preflight shows `FAILS=0`.

## What changed in code

| Where | Before | After |
|---|---|---|
| Batch header (`pipeline_cluster`, `pipeline_remote`) | `module load anaconda/2024.06; source activate; conda activate` (no fail-fast) | `set -eo pipefail; module purge; module load miniconda3; conda shell hook; activate; set -u` |
| `pipeline_remote.Batchfile` | built `#!`/`#SBATCH` header but never wrote it | header emitted |
| GUI orchestrators (morph / prelim / opt / mesh-clf) | `source ~/.bashrc` + `/home/<user>/.conda/envs/.../python` | env setup lines + `python $HOME/aeropt/Scripts/...` |
| `runSimRemote` morph (batch + immediate) | `source ~/.bashrc` + env python path | env setup + `python` |
| Modules | `compiler/intel/2020/0`, `compiler/gnu/12/1.0`, `mpi/intel/2020/0`, `mpi/mpich/3.2.1` | `intel/2026.1`, `gnu/12.3.0-x86_64`, `intel/2026.1 intel-mpi/2021.18`, OpenMPI 5 |
| Monitors | `module load paraview/2019 \|\| true` + bare `pvpython` (failed silently) | `$PVPYTHON` from config, warning in job stderr if missing |
| Executables | hard-coded in 4 files | `cluster_config.py` (+ JSON override) |
| Prelim screening binaries | `s.o.hassan/bin/Gen3d_jj`, `UnsMgnsg3d` | same as optimisation: `s.engevabj/.../Gen3d`, `UnsMgnsg3d` |

## Open questions to settle before trusting new results

- **Binary rebuild**: FLITE tools were built against the Intel 2020 runtime. If
  `ldd` shows missing `libif*/libimf/libmpi*`, they need recompiling with
  `intel/2026.1` + `intel-mpi/2021.18`. A rebuilt solver is a *different solver*:
  re-run the baseline intake and confirm PR / distortion / mass-flow match the
  old-system baseline before mixing new evaluations into an existing BO history.
- **Prelim vs optimisation binaries** were different before this change. Any
  earlier Morris/prelim screening was run with the s.o.hassan builds.
