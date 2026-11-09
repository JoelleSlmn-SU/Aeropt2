# cluster_config.py
# ----------------------------------------------------------------------
# Single source of truth for everything that is specific to the HPC system:
#   - how a batch job gets a working Python (module + conda env)
#   - which compiler / MPI modules the FLITE tools need
#   - where the mesher / solver / utility executables live
#   - how pvpython is found for the in-run monitors
#
# Every batch-file generator (pipeline_cluster, pipeline_remote, runSimRemote,
# remoteOpt, prelimResponseSurface, GUI orchestrators) reads from here.
#
# Overrides (cluster side, no code edit or git conflict needed):
#   1. $AEROPT_CLUSTER_CONFIG  -> path to a JSON file
#   2. ~/aeropt/cluster_config.json
# Keys in the JSON replace the defaults below one-for-one. Unknown keys are
# kept, so you can add entries without touching this file.
#
# NOTE: the GUI runs on your own machine and never sees the cluster-side JSON.
# It only uses env_setup_lines(), so if you change env_module / conda_env,
# change them HERE (and pull on both sides), not only in the JSON.
#
# Check the whole thing on the cluster with:  bash Utilities/hpc_preflight.sh
# ----------------------------------------------------------------------

import json
import os

# --- Defaults for the current (2026) cluster -------------------------------
CLUSTER_DEFAULTS = {
    # Python environment for every batch job
    "env_module": "miniconda3/26.5.3-2",
    "conda_env": "aeropt-hpc",

    # Compiler / MPI modules (full shell lines, appended verbatim to batch files)
    # Old system: compiler/intel/2020/0, compiler/gnu/12/1.0, mpi/intel/2020/0, mpi/mpich/3.2.1
    "intel_module": "module load intel/2026.1",
    "gnu_module": "module load gnu/12.3.0-x86_64",
    # Solver stage only loads this line, so it must bring the Intel runtime too.
    "mpi_intel_module": "module load intel/2026.1 intel-mpi/2021.18",
    "mpi_module": "module load gnu/12.3.0-x86_64 mpi/openmpi/5.0.10-gnu-12.3.0-ucx-mt-x86_64",

    # ParaView for the solver-side monitors. There is no paraview module on the
    # new system, so pvpython comes from its own conda env (see preflight).
    # paraview_setup is an optional shell line run before the monitor loop.
    "paraview_setup": "",
    "pvpython": "$HOME/.conda/envs/aeropt-pv/bin/pvpython",

    # Aeropt scripts on the cluster
    "scripts_root": "$HOME/aeropt/Scripts",

    # FLITE / mesher executables. These are the values remoteOpt.py actually
    # passed on the old system. Verify with the preflight; override in JSON.
    "surface_mesher": "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Surf3D",
    "volume_mesher": "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Mesh3D",
    "prepro_exe": "/home/s.engevabj/codes/PrePro_uns/Gen3d",
    "solver_exe": "/home/s.engevabj/codes/FLITE_uns/UnsMgnsg3d",
    "combine_exe": "/home/s.engevabj/codes/utilities/makeplot2",
    "ensight_exe": "/home/s.engevabj/codes/utilities/engen_tet",
    "splitplot_exe": "/home/s.engevabj/codes/utilities/splitplot2",
    "makeplot_exe": "/home/s.engevabj/codes/utilities/makeplot2",
    "hyb_plt_converter": "/home/s.engevabj/codes/utilities/hyb_plt_converter",
}

# Keys that are executables (used by the preflight check)
EXECUTABLE_KEYS = (
    "surface_mesher", "volume_mesher", "prepro_exe", "solver_exe",
    "combine_exe", "ensight_exe", "splitplot_exe", "makeplot_exe",
    "hyb_plt_converter",
)

# Keys that hold module lines
MODULE_KEYS = ("intel_module", "gnu_module", "mpi_intel_module", "mpi_module")


def _override_path():
    p = os.environ.get("AEROPT_CLUSTER_CONFIG")
    if p:
        return os.path.expanduser(os.path.expandvars(p))
    return os.path.expanduser("~/aeropt/cluster_config.json")


def load_cluster_config(read_overrides=True):
    """Return the cluster config dict (defaults + optional JSON overrides)."""
    cfg = dict(CLUSTER_DEFAULTS)
    if read_overrides:
        path = _override_path()
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    user = json.load(f)
                if not isinstance(user, dict):
                    raise ValueError("top-level JSON must be an object")
                cfg.update(user)
            except Exception as e:  # a broken override must not silently pass
                raise RuntimeError(f"[cluster_config] Could not read {path}: {e}") from e
    return cfg


def env_setup_lines(cfg=None):
    """
    Shell lines that give a batch job the aeropt Python environment.

    - set -eo pipefail first, so a failed module load/activate stops the job
      instead of falling through to whatever python3 is on PATH.
    - PYTHONHOME is cleared: a stale one from ~/.bashrc breaks every Python.
    - `conda shell.bash hook` works in non-interactive batch shells, where
      `source activate` / plain `conda activate` do not.
    - set -u only after activation: conda's activate scripts reference unset
      variables and abort under -u.
    """
    cfg = cfg or load_cluster_config(read_overrides=False)
    return [
        "set -eo pipefail",
        "unset PYTHONHOME",
        "module purge",
        f"module load {cfg['env_module']}",
        'eval "$(conda shell.bash hook)"',
        f"conda activate {cfg['conda_env']}",
        "set -u",
    ]


def env_setup_inline(cfg=None):
    """env_setup_lines() joined for a single `bash -lc '...'` command."""
    return "; ".join(env_setup_lines(cfg))


def executables_from_config(cfg=None, overrides=None):
    """
    Executable / module dict in the shape ClusterTestManager expects.
    `overrides` (e.g. GUI settings) wins over cfg for keys it actually sets.
    """
    cfg = cfg or load_cluster_config()
    out = {k: cfg[k] for k in EXECUTABLE_KEYS + MODULE_KEYS}
    for k, v in (overrides or {}).items():
        if k in out and v not in (None, ""):
            out[k] = v
    return out


if __name__ == "__main__":
    # `python cluster_config.py` prints the resolved config (used by preflight)
    import sys
    cfg = load_cluster_config()
    if len(sys.argv) > 1 and sys.argv[1] == "--shell":
        # KEY='value' lines that bash can eval
        for k, v in cfg.items():
            if isinstance(v, str):
                print(f"CFG_{k.upper()}='{v}'")
        print("CFG_EXECUTABLE_KEYS='" + " ".join(EXECUTABLE_KEYS) + "'")
        print("CFG_MODULE_KEYS='" + " ".join(MODULE_KEYS) + "'")
    else:
        print(json.dumps(cfg, indent=2))
