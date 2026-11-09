#!/bin/bash -l
# hpc_preflight.sh
# ----------------------------------------------------------------------
# Checks, on the cluster, everything Remote/cluster_config.py assumes:
#   1. login environment is not hijacked by the old ~/.localpython
#   2. module + conda env activate exactly as batch jobs do it (incl. set -u)
#   3. the Python packages the pipeline imports are present
#   4. each compiler/MPI module line loads, and intel+gnu do not swap each other
#   5. every FLITE/mesher executable exists, is executable, and has no
#      missing shared libraries under the modules its stage loads
#   6. pvpython (monitors -> monitors.csv -> objective) works headless
#
# Run it where the jobs run, i.e. on a compute node:
#   cd ~/aeropt/Scripts
#   srun -n1 -t 00:15:00 bash -l Utilities/hpc_preflight.sh
# (Login-node runs are fine for a first pass but ldd/libs can differ.)
# Exit code = number of FAILs.
# ----------------------------------------------------------------------

# srun/non-login shells may not define `module`
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1

SCRIPTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILS=0; WARNS=0
pass() { echo "  [PASS] $*"; }
warn() { echo "  [WARN] $*"; WARNS=$((WARNS+1)); }
fail() { echo -e "  [FAIL] $*"; FAILS=$((FAILS+1)); }
hdr()  { echo; echo "== $* =="; }

hdr "0. Where am I"
echo "  host=$(hostname)  user=$USER  HOME=$HOME  scripts=$SCRIPTS_ROOT"
[ -d "$HOME/aeropt/Scripts" ] && pass "\$HOME/aeropt/Scripts exists" \
                              || fail "\$HOME/aeropt/Scripts missing (all generated jobs use this path)"
if [ -d "/scratch/$USER" ]; then pass "/scratch/$USER exists (GUI run dirs)"
else warn "/scratch/$USER missing - the GUI puts run dirs there (GUI/Aeropt.py)"; fi

hdr "1. Leftovers from the old system"
case ":$PATH:" in *localpython*) fail "PATH still contains .localpython - remove it from ~/.bashrc / ~/.bash_profile";;
                  *) pass "PATH has no .localpython";; esac
[ -n "${PYTHONHOME:-}" ] && fail "PYTHONHOME is set ($PYTHONHOME) - remove from login files" || pass "PYTHONHOME unset"
case "${PYTHONPATH:-}" in *localpython*|*/.opt/python*) fail "PYTHONPATH points at the old Python: $PYTHONPATH";;
                         *) pass "PYTHONPATH clean (${PYTHONPATH:-<empty>})";; esac
if grep -nE "localpython|\.opt/python|PYTHONHOME" ~/.bashrc ~/.bash_profile ~/.profile 2>/dev/null; then
  warn "login files still reference the old Python (lines above)"
fi

hdr "2. Python env, activated exactly like a batch job"
# Same order as env_setup_lines() in Remote/cluster_config.py
CFG_SHELL=$(
  set -eo pipefail
  unset PYTHONHOME
  module purge
  module load miniconda3/26.5.3-2
  eval "$(conda shell.bash hook)"
  conda activate aeropt-hpc
  set -u
  cd "$SCRIPTS_ROOT"
  python Remote/cluster_config.py --shell
) || { fail "module load / conda activate aeropt-hpc failed (create the env first: see Notes/HPC_MIGRATION.md)"; echo; echo "FAILS=$FAILS WARNS=$WARNS"; exit $FAILS; }
eval "$CFG_SHELL"
pass "env activates under set -u; config read from Remote/cluster_config.py"
[ -f "$HOME/aeropt/cluster_config.json" ] && echo "  (override file in use: ~/aeropt/cluster_config.json)"
if [ "$CFG_ENV_MODULE" != "miniconda3/26.5.3-2" ] || [ "$CFG_CONDA_ENV" != "aeropt-hpc" ]; then
  warn "config env ($CFG_ENV_MODULE / $CFG_CONDA_ENV) differs from what this script tested"
fi

activate() {  # used inside subshells below
  unset PYTHONHOME; module purge >/dev/null 2>&1
  module load "$CFG_ENV_MODULE"; eval "$(conda shell.bash hook)"; conda activate "$CFG_CONDA_ENV"
}

hdr "3. Python packages"
(
  activate
  echo "  python: $(command -v python)  ($(python -c 'import sys;print(sys.version.split()[0])'))"
  python - <<'PY'
import importlib
mods = ["numpy", "scipy", "sklearn", "vtk", "pyvista", "shapely", "alphashape",
        "matplotlib", "pandas", "networkx", "cma", "tqdm", "paramiko", "geomdl"]
bad = 0
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  [PASS] {m:<11} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  [FAIL] {m:<11} {type(e).__name__}: {e}"); bad += 1
raise SystemExit(bad)
PY
) ; n=$?; FAILS=$((FAILS+n))

hdr "4. Module lines"
for key in $CFG_MODULE_KEYS; do
  var="CFG_${key^^}"; line="${!var}"
  out=$( ( activate; set -u; eval "$line" ) 2>&1 ) && pass "$key: $line" \
      || fail "$key: '$line' failed:\n$(echo "$out" | sed 's/^/        /')"
done
# Stages append intel_module then gnu_module. If Lmod treats them as one
# 'compiler' family, the second load silently swaps out Intel.
loaded=$( ( activate; eval "$CFG_INTEL_MODULE"; eval "$CFG_GNU_MODULE"; module -t list ) 2>&1 )
if echo "$loaded" | grep -q "^intel/"; then pass "intel still loaded after gnu_module (no family swap)"
else fail "gnu_module swapped out intel - set gnu_module to \"\" in ~/aeropt/cluster_config.json or load gnu first"; fi

hdr "5. Executables (exists, executable, shared libraries)"
for key in $CFG_EXECUTABLE_KEYS; do
  var="CFG_${key^^}"; exe=$(eval echo "${!var}")
  if [ ! -e "$exe" ]; then fail "$key: not found: $exe"; continue; fi
  if [ ! -x "$exe" ]; then fail "$key: not executable/readable by you: $exe"; continue; fi
  case "$key" in solver_exe) mods="$CFG_MPI_INTEL_MODULE";; *) mods="$CFG_INTEL_MODULE; $CFG_GNU_MODULE";; esac
  missing=$( ( activate; eval "$mods"; ldd "$exe" 2>&1 ) | grep "not found" )
  if [ -n "$missing" ]; then
    fail "$key: missing libraries under [$mods] -> needs a rebuild or a compat module:"
    echo "$missing" | sed 's/^/        /'
  else pass "$key: $exe"; fi
done

hdr "6. pvpython (in-run monitors -> monitors.csv -> objective)"
PV=$(eval echo "$CFG_PVPYTHON")
if [ -n "$CFG_PARAVIEW_SETUP" ]; then eval "$CFG_PARAVIEW_SETUP" || warn "paraview_setup line failed"; fi
if command -v "$PV" >/dev/null 2>&1; then
  tmp=$(mktemp --suffix=.py); echo 'import paraview.simple, paraview.servermanager; print("PV_OK")' > "$tmp"
  if "$PV" --force-offscreen-rendering "$tmp" 2>&1 | grep -q PV_OK; then pass "pvpython imports paraview.simple headless: $PV"
  else fail "pvpython found but could not run headless: $PV  (run it by hand to see the error)"; fi
  rm -f "$tmp"
else
  fail "pvpython not found at $PV - monitors (incl. pressure recovery) will not be written"
fi

echo; echo "== Summary: FAILS=$FAILS  WARNS=$WARNS =="
exit $FAILS
