# pipeline_cluster.py
# ----------------------------------------------------------------------
# CLUSTER-SIDE pipeline manager (no SSH/SFTP - direct filesystem access)
# - Used when the optimization batch job is already running ON the cluster
# - Mirrors HPCPipelineManager API but uses subprocess.run() instead of SSH
# ----------------------------------------------------------------------

import sys, os, posixpath, shutil, re, textwrap, subprocess, json
from datetime import datetime

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = script_dir
while True:
    # We want the directory that *contains* FileRW
    if os.path.isdir(os.path.join(project_root, "FileRW")):
        break
    parent = os.path.dirname(project_root)
    if parent == project_root:
        # Reached filesystem root; give up
        break
    project_root = parent

# Put the project root on sys.path if we found FileRW
if os.path.isdir(os.path.join(project_root, "FileRW")):
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "ConvertFileType"))
    sys.path.insert(0, os.path.join(project_root, "MeshGeneration"))
else:
    # Fallback: at least add script_dir so relative imports still have a chance
    sys.path.insert(0, script_dir)

from ShapeParameterization.controlNodeDisp import estimate_normals, getDisplacements

class Batchfile:
    def __init__(self, name: str):
        self.name = name
        self.lines = []
        self.sbatch_params = {
            "job-name": name,
            "output": f"{name}.%J.out",
            "error":  f"{name}.%J.err",
            "nodes":  "1",
            "ntasks": "1",
            "time":   "1-00:00",
        }
    def __str__(self):
        hdr = ["#!/bin/bash -l", "#SBATCH --export=NONE"]
        for k, v in self.sbatch_params.items():
            hdr.append(f"#SBATCH --{k}={v}")
        return "\n".join(hdr + ['module load anaconda/2024.06',
            'source activate',
            'conda activate aeropt-hpc',
            'set -euo pipefail',
        ] + self.lines + [""])

class ClusterPipelineManager:
    """
    Pipeline manager for use WITHIN cluster batch jobs.
    No SSH - uses local filesystem and subprocess for sbatch.
    """
    
    def __init__(self, config_dict, gen=0, n=0):
        """
        config_dict should contain:
        - remote_output: base output directory (already on cluster FS)
        - base_name: model base name
        - input_dir: directory with input files
        - executables paths
        - modal_coeffs: the BO design vector
        """
        self.gen = gen
        self.n = n
        self.base_name = config_dict.get("base_name", "model")
        self.remote_output = config_dict["remote_output"]
        self.input_dir = config_dict.get("input_dir", "")
        
        # Executables
        self.morph_script = config_dict.get("morph_script", "$HOME/aeropt/Scripts/Remote/remoteMorph.py")
        self.paraview_script = "$HOME/aeropt/Scripts/Remote/paraview_cluster.py"
        self.surface_mesher = config_dict.get("surface_mesher", "/home/s.o.hassan/XieZ/work/Meshers/surface/src/a.Surf3D")
        self.volume_mesher = config_dict.get("volume_mesher", "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Mesh3D")
        self.prepro_exe = config_dict.get("prepro_exe", "/home/s.engevabj/codes/PrePro_uns/Gen3d")
        self.solver_exe = config_dict.get("solver_exe", "/home/s.engevabj/codes/FLITE_uns/UnsMgnsg3d")
        self.combine_exe = config_dict.get("combine_exe", "/home/s.engevabj/codes/utilities/makeplot2")
        self.ensight_exe = config_dict.get("ensight_exe", "/home/s.engevabj/codes/utilities/engen_tet")
        self.splitplot_exe = config_dict.get("splitplot_exe", "/home/s.engevabj/codes/utilities/splitplot2")
        self.makeplot_exe = config_dict.get("makeplot_exe", "/home/s.engevabj/codes/utilities/makeplot2")
        
        self.orig_dir = os.path.join(self.remote_output, "orig")
        
        # Modules
        self.python_module = "/home/s.2268086/.conda/envs/aeropt-hpc/bin/python"
        self.intel_module = config_dict.get("intel_module", "module load compiler/intel/2020/0")
        self.gnu_module = config_dict.get("gnu_module", "module load compiler/gnu/12/1.0")
        self.mpi_module = config_dict.get("mpi_module", "module load mpi/mpich/3.2.1")
        self.mpi_intel_module = config_dict.get("mpi_intel_module", "module load mpi/intel/2020/0")
        self.hyb_plt_converter = config_dict.get("hyb_plt_converter", "/home/s.engevabj/codes/utilities/hyb_plt_converter")
        # Design parameters
        self.modal_coeffs = config_dict.get("modal_coeffs", [])
        self.morph_basis_json = config_dict.get("morph_basis_json", "")
        self.units = config_dict.get("cad_units", "mm")
        
        self.monitor_config_json = config_dict.get("monitor_config_json", "")
        self.monitor_interval = int(config_dict.get("monitor_interval", 50))
        
        self.previous_solution = config_dict.get("previous_solution", {}) or {}
        self.restart_from_previous = bool(self.previous_solution.get("enabled", False))
        self.interpu_script = config_dict.get(
            "interpu_script",
            "$HOME/aeropt/Scripts/Utilities/interpu.py"
        )
        
        self.job_ids = {}
        pp = int(config_dict.get("parallel_processes", 160))
        self.sol_parallel_domains = max(1, pp)
        
        # Setup logging
        log_dir = os.path.join(self.remote_output, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"pipeline_n{gen}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    def _log(self, msg: str):
        """Log to both stdout and file."""
        print(msg, flush=True)
        with open(
            self.log_file,
            "a",
            encoding="utf-8",
            errors="replace",
        ) as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
            
    def _state_path(self) -> str:
        st_dir = os.path.join(self.remote_output, "logs", "state", f"n_{self.gen}")
        os.makedirs(st_dir, exist_ok=True)
        return os.path.join(st_dir, f"sample_{self.n}.json")

    def _load_state(self) -> dict:
        p = self._state_path()
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self, **updates):
        st = self._load_state()
        st.update(updates)
        st.setdefault("gen", self.gen)
        st.setdefault("n", self.n)
        st["updated"] = datetime.now().isoformat()
        with open(self._state_path(), "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)

    
    def _submit_batch(self, batchfile_path: str, cwd: str) -> str:
        """Submit a batch file and return job ID"""
        cmd = f"sbatch {batchfile_path}"
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")
        
        # Parse job ID from output
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from: {result.stdout}")
        
        return match.group(1)
    
    def _ensure_convergence_script(self, sol_dir: str) -> str:
        """
        Ensure convergenceCheck.py exists in the solution folder on the cluster.
        Returns the path to the script in sol_dir.
        """
        script_target = os.path.join(sol_dir, "convergenceCheck.py")

        # Look for a local copy relative to this file or CWD
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "convergenceCheck.py"),
            os.path.join(os.path.dirname(here), "Remote", "convergenceCheck.py"),
            os.path.join(os.getcwd(), "Remote", "convergenceCheck.py"),
            os.path.join(os.getcwd(), "convergenceCheck.py"),
        ]
        src = next((p for p in candidates if os.path.exists(p)), None)
        if not src:
            raise FileNotFoundError("Could not find convergenceCheck.py on the cluster.")

        shutil.copyfile(src, script_target)
        return script_target
    
    def _submit_convergence_guard(
        self,
        tag_slug: str,
        sol_dir: str,
        stdout_name: str = "solver_output",
        solver_job_id: str | None = None,
        res_threshold: float = -3.0,
        residual_csv: str | None = None,
    ):
        """
        Two-step convergence guard:
          1) guardcheck: run convergenceCheck.py -> last_convergence.json
          2) guarddecide: read last_convergence.json, bump restart if needed,
             run makeplot/splitplot, and re-sbatch solver batchfile.
        """
        residual_csv = residual_csv or f"{self.base_name}.rsd"

        # ensure convergenceCheck.py is present
        self._ensure_convergence_script(sol_dir)

        # ---------- 1) guardcheck ----------
        check_name = f"guardcheck_{tag_slug}"
        bf1 = Batchfile(check_name)
        bf1.sbatch_params["output"] = f"{check_name}.out"
        bf1.sbatch_params["error"] = f"{check_name}.err"
        if solver_job_id:
            bf1.sbatch_params["dependency"] = f"afterany:{solver_job_id}"

        bf1.lines.append(f"cd {sol_dir}")
        bf1.lines.append("ATTEMPT=0")
        bf1.lines.append('if [ -f convergence_state.json ]; then ATTEMPT=$(python3 - <<PY\n'
                         'import json; print(json.load(open("convergence_state.json")).get("attempt",0))\n'
                         'PY\n); fi')
        bf1.lines.append(
            'python3 ./convergenceCheck.py '
            f'"{sol_dir}" "$ATTEMPT" "2" "{float(res_threshold)}" '
            "'' "
            f'"{residual_csv}" "{stdout_name}" > last_convergence.json'
        )

        check_path = os.path.join(sol_dir, f"batchfile_{check_name}")
        with open(check_path, "w") as f:
            f.write(str(bf1))
        check_job = self._submit_batch(check_path, cwd=sol_dir)
        self._log(f"[CLUSTER] Guard-check job {check_job}")

        # ---------- 2) guarddecide ----------
        decide_name = f"guarddecide_{tag_slug}"
        bf2 = Batchfile(decide_name)
        bf2.sbatch_params["output"] = f"{decide_name}.out"
        bf2.sbatch_params["error"] = f"{decide_name}.err"
        bf2.sbatch_params["dependency"] = f"afterany:{check_job}"

        bf2.lines.append(f"cd {sol_dir}")
        bf2.lines.append('CONV=$(python3 - <<PY\n'
                         'import json; d=json.load(open("last_convergence.json"))\n'
                         'print("1" if d.get("converged") else "0")\nPY\n)')
        bf2.lines.append('REASON=$(python3 - <<PY\n'
                         'import json; d=json.load(open("last_convergence.json"))\n'
                         'print(d.get("reason",""))\nPY\n)')
        bf2.lines.append('echo "[GUARD] converged=$CONV reason=$REASON"')
        bf2.lines.append('if [ "$CONV" = "1" ]; then')
        bf2.lines.append('  echo "[GUARD] Converged. Nothing to do."')
        bf2.lines.append('  exit 0')
        bf2.lines.append('fi')

        bf2.lines.append('NEXT=1')
        bf2.lines.append('if [ -f convergence_state.json ]; then')
        bf2.lines.append("  NEXT=$(python3 - <<'PY'\n"
                         "import json\n"
                         "with open('convergence_state.json','r',encoding='utf-8') as f:\n"
                         "    d = json.load(f)\n"
                         "print(d.get('attempt',0) + 1)\n"
                         "PY\n"
                         ")")
        bf2.lines.append("fi")

        bf2.lines.append("python3 - <<PY\n"
                         "import json\n"
                         "data = {'attempt': int('" + "${NEXT}" + "'), 'last_reason': '" + "${REASON}" + "'}\n"
                         "with open('convergence_state.json','w',encoding='utf-8') as f:\n"
                         "    json.dump(data, f)\n"
                         "PY")

        # bump ivd%restartNumber
        bf2.lines.append("python3 - <<'PY'")
        bf2.lines.append("import re")
        bf2.lines.append(f"p = '{self.base_name}.inp'")
        bf2.lines.append("with open(p,'r',encoding='utf-8',errors='ignore') as f:")
        bf2.lines.append("    txt = f.read()")
        bf2.lines.append("")
        bf2.lines.append("def bump(match):")
        bf2.lines.append("    prefix = match.group('p')")
        bf2.lines.append("    old_s = match.group('n')")
        bf2.lines.append("    try:")
        bf2.lines.append("        old = int(old_s)")
        bf2.lines.append("    except ValueError:")
        bf2.lines.append("        old = 0")
        bf2.lines.append("    new = old + 1")
        bf2.lines.append("    print(f'[GUARD] restart number {old} -> {new} in {p}')")
        bf2.lines.append("    return prefix + str(new) + match.group('trail')")
        bf2.lines.append("")
        bf2.lines.append(r"pattern = r'(?im)^(?P<p>\s*ivd%restartNumber\s*=\s*)(?P<n>\d+)(?P<trail>\s*,?)'")
        bf2.lines.append("txt2, n = re.subn(pattern, bump, txt, count=1)")
        bf2.lines.append("if n == 0:")
        bf2.lines.append("    print('[GUARD] WARNING: ivd%restartNumber line not found in file')")
        bf2.lines.append("else:")
        bf2.lines.append("    with open(p,'w',encoding='utf-8',newline='\\n') as f:")
        bf2.lines.append("        f.write(txt2)")
        bf2.lines.append("PY")

        # makeplot/splitplot + re-sbatch solver batch
        bf2.lines.append(f'"{self.makeplot_exe}" <<INPUT1')
        bf2.lines.append("plotreg.reg")
        bf2.lines.append(f"{self.base_name}.res")
        bf2.lines.append(f"{self.base_name}.unk")
        bf2.lines.append("F")
        bf2.lines.append("T")
        bf2.lines.append("INPUT1")

        bf2.lines.append(f'"{self.splitplot_exe}" <<INPUT2')
        bf2.lines.append("plotreg.reg")
        bf2.lines.append(f"{self.base_name}.unk")
        bf2.lines.append(f"{self.base_name}.rst")
        bf2.lines.append("T")
        bf2.lines.append("INPUT2")

        bf2.lines.append(f"sbatch batchfile_sol_n{self.gen}_{tag_slug}")

        decide_path = os.path.join(sol_dir, f"batchfile_{decide_name}")
        with open(decide_path, "w") as f:
            f.write(str(bf2))
        decide_job = self._submit_batch(decide_path, cwd=sol_dir)
        self._log(f"[CLUSTER] Guard-decide job {decide_job}")

        self.job_ids[f"guardcheck_{tag_slug}"] = check_job
        self.job_ids[f"guarddecide_{tag_slug}"] = decide_job
        return decide_job
    
    def _append_previous_solution_initialisation(self, bf, sol_dir: str):
        """
        Add interpu-based initialisation lines to solver batchfile.
        Must be called after the new .inp has been copied/patched.
        """
        if not self.restart_from_previous:
            return

        prev = self.previous_solution
        old_dir = prev.get("directory", "").strip()
        old_base = prev.get("base", "").strip()
        boundary_mode = prev.get("boundary_mode", "same_id")
        num_comp = int(prev.get("num_comp", 7))

        new_base = f"{self.base_name}_{self.n}"

        if not old_dir or not old_base:
            bf.lines.append('echo "[INTERPU][WARN] Previous solution enabled but old_dir/old_base missing."')
            bf.lines.append("exit 2")
            return

        bf.lines.append("")
        bf.lines.append(f"cp base.plt {new_base}.plt")
        bf.lines.append("# ---------------- Previous-solution interpolation ----------------")
        bf.lines.append('echo "[INTERPU] Initialising solution from previous run..."')
        bf.lines.append(f'OLD_DIR="{old_dir}"')
        bf.lines.append(f'OLD_BASE="{old_base}"')
        bf.lines.append(f'NEW_DIR="{sol_dir}"')
        bf.lines.append(f'NEW_BASE="{new_base}"')
        bf.lines.append(f'SCRIPT="{self.interpu_script}"')
        bf.lines.append(f'NUM_COMP="{num_comp}"')
        bf.lines.append(f'BOUNDARY_MODE="{boundary_mode}"')
        bf.lines.append(f'MAKEPLOT_EXE="{self.makeplot_exe}"')
        bf.lines.append(f'SPLITPLOT_EXE="{self.splitplot_exe}"')

        bf.lines.append('if [ ! -d "$OLD_DIR" ]; then echo "[INTERPU][ERROR] OLD_DIR missing: $OLD_DIR"; exit 2; fi')
        bf.lines.append('if [ ! -f "$OLD_DIR/${OLD_BASE}.unk" ]; then')
        bf.lines.append('  echo "[INTERPU] ${OLD_BASE}.unk not found; creating from ${OLD_BASE}.res"')
        bf.lines.append('  cd "$OLD_DIR"')
        bf.lines.append('  "$MAKEPLOT_EXE" <<EOF')
        bf.lines.append('plotreg.reg')
        bf.lines.append('${OLD_BASE}.res')
        bf.lines.append('${OLD_BASE}.unk')
        bf.lines.append('F')
        bf.lines.append('T')
        bf.lines.append('EOF')
        bf.lines.append('fi')

        bf.lines.append('cd "$NEW_DIR"')
        bf.lines.append('python "$SCRIPT" \\')
        bf.lines.append('  --old-dir "$OLD_DIR" \\')
        bf.lines.append('  --old-base "$OLD_BASE" \\')
        bf.lines.append('  --new-dir "$NEW_DIR" \\')
        bf.lines.append('  --new-base "$NEW_BASE" \\')
        bf.lines.append('  --boundary-mode "$BOUNDARY_MODE" \\')
        bf.lines.append('  --num-comp "$NUM_COMP"')

        bf.lines.append('if [ ! -s "${NEW_BASE}.unk" ]; then')
        bf.lines.append('  echo "[INTERPU][ERROR] Failed to create non-empty ${NEW_BASE}.unk"')
        bf.lines.append('  exit 3')
        bf.lines.append('fi')
        
        bf.lines.append('  "$SPLITPLOT_EXE" <<EOF')
        bf.lines.append('plotreg.reg')
        bf.lines.append('${NEW_BASE}.unk')
        bf.lines.append('${NEW_BASE}.rst')
        bf.lines.append('T')
        bf.lines.append('EOF')

        bf.lines.append("# Force solver restartNumber = 1 only after .unk exists")
        bf.lines.append("python3 - <<'PY'")
        bf.lines.append("import re")
        bf.lines.append(f"p = '{new_base}.inp'")
        bf.lines.append("with open(p, 'r', encoding='utf-8', errors='ignore') as f:")
        bf.lines.append("    txt = f.read()")
        bf.lines.append(r"pat = r'(?im)^(?P<p>\s*ivd%restartNumber\s*=\s*)(?P<n>\d+)(?P<trail>\s*,?)'")
        bf.lines.append("txt2, n = re.subn(pat, r'\\g<p>1\\g<trail>', txt, count=1)")
        bf.lines.append("if n == 0:")
        bf.lines.append("    print('[INTERPU][WARN] ivd%restartNumber not found; appending it')")
        bf.lines.append("    txt2 = txt.rstrip() + '\\nivd%restartNumber = 1,\\n'")
        bf.lines.append("with open(p, 'w', encoding='utf-8', newline='\\n') as f:")
        bf.lines.append("    f.write(txt2)")
        bf.lines.append("print('[INTERPU] restartNumber set to 1')")
        bf.lines.append("PY")

    def _append_paraview_monitoring_section(self, bf, mach_for_post: float):
        """
        Append ParaView/monitor logic to the solver batchfile.

        Steady:
            Uses BASE.res* and BASE.rsd as before.

        Transient:
            Uses BASE_1.res* and BASE_1.rsd with the same monitoring logic as
            the steady-state path. We do not cycle through BASE_2, BASE_3, ...
            here because the monitor should track the first transient solution
            family in the same way the steady monitor tracks BASE.
        """
        bf.lines.append("")
        bf.lines.append("# ---------------- Pressure-recovery / monitors ----------------")
        bf.lines.append("module load paraview/2019 || true")

        bf.lines.append(f"BASE='{self.base_name}_{self.n}'")
        bf.lines.append(f"INTERVAL={int(self.monitor_interval)}")
        bf.lines.append("SLEEP_S=20")

        bf.lines.append("PR_DIR=Monitors")
        bf.lines.append("WORK_DIR=\"$PR_DIR/work\"")
        bf.lines.append("UNK_DIR=\"$PR_DIR/unk\"")
        bf.lines.append("CSV_OUT=\"$PR_DIR/monitors.csv\"")
        bf.lines.append("MON_LOG=\"$PR_DIR/monitor.log\"")
        bf.lines.append("MON_JSON_SRC=\"{0}\"".format(self.monitor_config_json or ""))
        bf.lines.append("MON_JSON=\"$PR_DIR/monitors.json\"")
        bf.lines.append("STATE_FILE=\"$PR_DIR/state_abs_iter.txt\"")
        bf.lines.append("LOCK_DIR=\"$PR_DIR/lockdir\"")
        bf.lines.append("PV_SCRIPT=\"$PR_DIR/paraview_cluster.py\"")
        bf.lines.append("mkdir -p \"$PR_DIR\" \"$UNK_DIR\"")
        bf.lines.append("touch \"$MON_LOG\"")

        bf.lines.append(f"cp {self.paraview_script} \"$PV_SCRIPT\"")

        bf.lines.append("if [ -n \"$MON_JSON_SRC\" ] && [ -f \"$MON_JSON_SRC\" ]; then")
        bf.lines.append("  cp \"$MON_JSON_SRC\" \"$MON_JSON\"")
        bf.lines.append("fi")
        bf.lines.append("if [ ! -f \"$MON_JSON\" ]; then")
        bf.lines.append("  echo '{\"interval\":50,\"enabled\":true,\"monitors\":[]}' > \"$MON_JSON\"")
        bf.lines.append("fi")

        bf.lines.append("rm -f SOLVER_DONE")

        # Parse solver mode from the already-patched input file.
        bf.lines.append("")
        bf.lines.append("# ---- Parse steady/unsteady settings from solver input ----")
        bf.lines.append("FLOW_TYPE=$(python3 - <<'PY'")
        bf.lines.append("import re")
        bf.lines.append(f"txt=open('{self.base_name}_{self.n}.inp', encoding='utf-8', errors='ignore').read()")
        bf.lines.append("m=re.search(r'(?im)^\\s*ivd%flowType\\s*=\\s*([-+]?\\d+)', txt)")
        bf.lines.append("print(int(m.group(1)) if m else 0)")
        bf.lines.append("PY")
        bf.lines.append(")")
        bf.lines.append('echo "[MON] FLOW_TYPE=$FLOW_TYPE interval=$INTERVAL" >> "$MON_LOG"')

        # Map solver mode to actual .res base.
        # Steady:    BASE.res*
        # Transient: BASE_1.res*  (same logic as steady, just different base name)
        bf.lines.append("")
        bf.lines.append("post_base_for_abs_iter() {")
        bf.lines.append("  if [ \"$FLOW_TYPE\" -eq 1 ]; then")
        bf.lines.append("    echo \"${BASE}_1\"")
        bf.lines.append("  else")
        bf.lines.append("    echo \"${BASE}\"")
        bf.lines.append("  fi")
        bf.lines.append("}")

        bf.lines.append("local_iter_for_abs_iter() {")
        bf.lines.append("  abs_it=\"$1\"")
        bf.lines.append("  echo \"$abs_it\"")
        bf.lines.append("}")

        # For steady, use BASE.rsd.
        # For unsteady, use POST_BASE.rsd if it exists; otherwise allow the post attempt
        # once matching .res files exist.
        bf.lines.append("")
        bf.lines.append("latest_local_iter_for_base() {")
        bf.lines.append("  pb=\"$1\"")
        bf.lines.append("  required_it=\"${2:-0}\"")
        bf.lines.append("  rsd=\"${pb}.rsd\"")
        bf.lines.append("  if [ -f \"$rsd\" ]; then")
        bf.lines.append("    awk 'NF>0 && $1 ~ /^[0-9]+$/ {it=$1} END{print it+0}' \"$rsd\" 2>/dev/null || echo 0")
        bf.lines.append("    return")
        bf.lines.append("  fi")
        bf.lines.append("  if ls \"${pb}.res\"* >/dev/null 2>&1; then")
        bf.lines.append("    # If no .rsd exists yet but .res files exist, allow one post attempt.")
        bf.lines.append("    echo \"$required_it\"")
        bf.lines.append("  else")
        bf.lines.append("    echo 0")
        bf.lines.append("  fi")
        bf.lines.append("}")

        bf.lines.append("")
        bf.lines.append("run_post() {")
        bf.lines.append("  abs_it=\"$1\"")
        bf.lines.append("  [ -n \"$abs_it\" ] || return 0")
        bf.lines.append("  [ \"$abs_it\" -gt 0 ] || return 0")

        bf.lines.append("  POST_BASE=$(post_base_for_abs_iter \"$abs_it\")")
        bf.lines.append("  LOCAL_IT=$(local_iter_for_abs_iter \"$abs_it\")")
        bf.lines.append("  HAVE_IT=$(latest_local_iter_for_base \"$POST_BASE\" \"$LOCAL_IT\")")

        bf.lines.append("  if [ \"$HAVE_IT\" -lt \"$LOCAL_IT\" ]; then")
        bf.lines.append("    echo \"[MON] skip abs_it=$abs_it POST_BASE=$POST_BASE local=$LOCAL_IT have=$HAVE_IT\" >> \"$MON_LOG\"")
        bf.lines.append("    return 0")
        bf.lines.append("  fi")

        bf.lines.append("  mkdir \"$LOCK_DIR\" 2>/dev/null || { echo \"[MON] lock active, skip abs_it=$abs_it\" >> \"$MON_LOG\"; return 0; }")
        bf.lines.append("  trap 'rm -rf \"$WORK_DIR\" 2>/dev/null || true; rmdir \"$LOCK_DIR\" 2>/dev/null || true' RETURN")
        bf.lines.append("  MON_LOG_ABS=\"$(pwd)/$MON_LOG\"")
        bf.lines.append("  echo \"[MON] abs_it=$abs_it POST_BASE=$POST_BASE local_it=$LOCAL_IT: makeplot2 -> engen_tet -> pvpython\" >> \"$MON_LOG_ABS\"")

        bf.lines.append("  rm -rf \"$WORK_DIR\" && mkdir -p \"$WORK_DIR\"")
        bf.lines.append("  ln -sf \"$(pwd)/${POST_BASE}.res\"* \"$WORK_DIR/\" 2>/dev/null || true")
        bf.lines.append("  ln -sf \"$(pwd)/base.plt\" \"$WORK_DIR/${POST_BASE}.plt\" 2>/dev/null || true")
        bf.lines.append("  ln -sf \"$(pwd)/plotreg.reg\" \"$WORK_DIR/\" 2>/dev/null || true")
        bf.lines.append("  [ -f \"${POST_BASE}.rsd\" ] && ln -sf \"$(pwd)/${POST_BASE}.rsd\" \"$WORK_DIR/\" || true")

        bf.lines.append("  pushd \"$WORK_DIR\" >/dev/null || return 0")

        bf.lines.append(f"  \"{self.makeplot_exe}\" <<INPUT1 >> \"$MON_LOG_ABS\" 2>&1")
        bf.lines.append("plotreg.reg")
        bf.lines.append("${POST_BASE}.res")
        bf.lines.append("${POST_BASE}.unk")
        bf.lines.append("F")
        bf.lines.append("T")
        bf.lines.append("INPUT1")

        bf.lines.append("  if [ -s \"${POST_BASE}.unk\" ]; then")
        bf.lines.append("    cp -f \"${POST_BASE}.unk\" \"../unk/${BASE}_abs_${abs_it}_${POST_BASE}.unk\"")
        bf.lines.append("  else")
        bf.lines.append("    echo \"[MON][WARN] missing/non-empty ${POST_BASE}.unk at abs_it=$abs_it\" >> \"$MON_LOG_ABS\"")
        bf.lines.append("  fi")

        bf.lines.append(f"  \"{self.ensight_exe}\" <<INPUT2 >> \"$MON_LOG_ABS\" 2>&1")
        bf.lines.append("${POST_BASE}")
        bf.lines.append("T")
        bf.lines.append(f"{mach_for_post:.8f}")
        bf.lines.append("298")
        bf.lines.append("106000")
        bf.lines.append("1006")
        bf.lines.append("INPUT2")

        bf.lines.append("  case_path=\"${POST_BASE}.case\"")
        bf.lines.append("  if [ ! -f \"$case_path\" ]; then")
        bf.lines.append("    case_path=$(find . -maxdepth 1 -name '*.case' | head -n 1)")
        bf.lines.append("  fi")
        bf.lines.append("  echo \"[MON] case_path=$case_path csv_out=../monitors.csv\" >> \"$MON_LOG_ABS\"")
        bf.lines.append("  if [ ! -f \"$case_path\" ]; then echo \"[MON][WARN] missing case for abs_it=$abs_it\" >> \"$MON_LOG_ABS\"; popd >/dev/null || true; return 0; fi")

        bf.lines.append("  set +e")
        bf.lines.append("  pvpython --force-offscreen-rendering \"../paraview_cluster.py\" \\")
        bf.lines.append("    --case \"$case_path\" \\")
        bf.lines.append("    --iter \"$abs_it\" \\")
        bf.lines.append(f"    --mach {mach_for_post:.8f} \\")
        bf.lines.append("    --monitors \"../monitors.json\" \\")
        bf.lines.append("    --out \"../monitors.csv\" \\")
        bf.lines.append("    --append >> \"$MON_LOG_ABS\" 2>&1")
        bf.lines.append("  rc=$?")
        bf.lines.append("  set -e")
        bf.lines.append("  echo \"[MON] pvpython rc=$rc for abs_it=$abs_it POST_BASE=$POST_BASE\" >> \"$MON_LOG_ABS\"")
        bf.lines.append("  popd >/dev/null || true")
        bf.lines.append("}")

        bf.lines.append("")
        bf.lines.append("monitor_pr() {")
        bf.lines.append("  last_abs=0")
        bf.lines.append("  [ -f \"$STATE_FILE\" ] && last_abs=$(cat \"$STATE_FILE\" 2>/dev/null || echo 0)")
        bf.lines.append("  echo \"[MON] start: last_abs=$last_abs interval=$INTERVAL\" >> \"$MON_LOG\"")
        bf.lines.append("  while true; do")
        bf.lines.append("    next=$(( (last_abs / INTERVAL + 1) * INTERVAL ))")
        bf.lines.append("    run_post \"$next\" || true")

        bf.lines.append("    POST_BASE=$(post_base_for_abs_iter \"$next\")")
        bf.lines.append("    LOCAL_IT=$(local_iter_for_abs_iter \"$next\")")
        bf.lines.append("    HAVE_IT=$(latest_local_iter_for_base \"$POST_BASE\" \"$LOCAL_IT\")")
        bf.lines.append("    if [ \"$HAVE_IT\" -ge \"$LOCAL_IT\" ]; then")
        bf.lines.append("      echo \"$next\" > \"$STATE_FILE\"")
        bf.lines.append("      last_abs=\"$next\"")
        bf.lines.append("    fi")

        bf.lines.append("    [ -f SOLVER_DONE ] && break")
        bf.lines.append("    sleep \"$SLEEP_S\"")
        bf.lines.append("  done")
        bf.lines.append("}")

        bf.lines.append("")
        bf.lines.append("monitor_pr &")
        bf.lines.append("PR_PID=$!")
        bf.lines.append("# -----------------------------------------------------------")
        bf.lines.append("")

    def _write_morph_config(self):
        """Write morph configuration JSON for one design (self.n)."""
        # 1) Find the baseline mesh
        surf_dir = os.path.join(self.remote_output, "surfaces", f"n_{self.gen}")
        vtm_path = None
        for ext in [".vtm", ".vtk", ".fro", ".case"]:
            candidate = os.path.join(self.input_dir, f"{self.base_name}{ext}")
            if os.path.exists(candidate):
                vtm_path = candidate
                break

        if not vtm_path:
            raise FileNotFoundError(f"No baseline mesh found in {self.input_dir}")

        # 2) Copy baseline to surf_dir if not already there
        mesh_name = os.path.basename(vtm_path)
        target = os.path.join(surf_dir, mesh_name)
        if not os.path.exists(target):
            if os.path.splitext(vtm_path)[1] == ".vtm":
                shutil.copy2(vtm_path, target)
                sidecar = os.path.join(os.path.dirname(vtm_path),
                                       os.path.splitext(mesh_name)[0])
                if os.path.isdir(sidecar):
                    target_sidecar = os.path.join(surf_dir,
                                                  os.path.basename(sidecar))
                    if os.path.exists(target_sidecar):
                        shutil.rmtree(target_sidecar)
                    shutil.copytree(sidecar, target_sidecar)
            else:
                shutil.copy2(vtm_path, target)

        # 3) Apply morph basis: surfaces, control nodes, d_ctrl
        t_surfaces = []
        u_surfaces = []
        c_surfaces = []
        control_nodes = []
        d_ctrl = []

        basis_path = (self.morph_basis_json or "").strip()
        if basis_path:
            try:
                with open(basis_path, "r") as bf:
                    basis = json.load(bf)

                cn = np.asarray(basis["control_nodes"], float)
                cn_normals = np.asarray(basis["control_normals"], float)
                t_surfaces = basis["TSurfaces"]
                if cn.size > 0:
                    cn = cn.reshape((-1, 3))
                    control_nodes = cn.tolist()

                    t_surfaces = list(map(int, basis.get("TSurfaces", [])))
                    u_surfaces = list(map(int, basis.get("USurfaces", [])))
                    c_surfaces = list(map(int, basis.get("CSurfaces", [])))
                    
                                        # -----------------------------
                    # top-level parameterisation info
                    # -----------------------------
                    parameterisation_method = str(
                        basis.get("parameterisation_method", "modal")
                    ).strip().lower()

                    direct_subtype = basis.get("direct_parameterisation_subtype", None)

                    use_pca = bool(basis.get("use_pca", False))
                    pca_cache_path = basis.get("pca_cache_path", None)

                    normal_project = bool(basis.get("normal_project", True))
                    vector_mode = str(basis.get("vector_mode", "local_frame"))
                    frame_knn = basis.get("frame_knn", 12)

                    global_modes = bool(basis.get("global_modes", False))
                    global_mode_config = basis.get("global_mode_config", [])
                    basis_axes = basis.get("basis_axes", None)

                    use_local_modes = bool(basis.get("use_local_modes", True))
                    global_only = bool(basis.get("global_only", False))
                    if global_only:
                        use_local_modes = False

                    k = int(basis.get("k_modes", 5))
                    # design coeffs (from BO / optimiser)
                    coeffs = np.asarray(
                        [] if self.modal_coeffs is None else self.modal_coeffs,
                        dtype=float
                    ).reshape(-1)

                    self._log(
                        f"[PIPELINE] gen={self.gen} n={self.n} "
                        f"param={parameterisation_method} "
                        f"use_pca={use_pca} coeffs_len={int(coeffs.size)}"
                    )
                    
                    use_protection = bool(basis.get("use_protection", False))
                    protected_nodes = [int(i) for i in basis.get("protected_control_nodes", [])]
                    protection_radius = basis.get("protection_radius", None)

                    if protection_radius is not None:
                        protection_radius = float(protection_radius)

                    if not use_protection:
                        protected_nodes = []
                        protection_radius = None

                    # -------------------------------------------------
                    # DIRECT PARAMETERISATION
                    # -------------------------------------------------
                    if parameterisation_method == "direct":
                        subtype = str(direct_subtype or "").strip().lower()

                        if subtype == "xyz":
                            expected_len = 3 * len(cn)
                        elif subtype == "normal":
                            expected_len = len(cn)
                        else:
                            raise RuntimeError(
                                f"Unknown direct_parameterisation_subtype: {direct_subtype}"
                            )

                        if coeffs.size < expected_len:
                            coeffs = np.pad(coeffs, (0, expected_len - coeffs.size))
                        elif coeffs.size > expected_len:
                            coeffs = coeffs[:expected_len]

                        d_ctrl = getDisplacements(
                            self.remote_output,
                            control_nodes=cn,
                            normals=cn_normals,
                            coeffs=coeffs,
                            t_patch_scale=basis.get("t_patch_scale", None),
                            amp_alpha=float(basis.get("amp_alpha", 0.005)),
                            parameterisation_method="direct",
                            direct_parameterisation_subtype=subtype,
                            protected_nodes=protected_nodes,
                            radius=protection_radius,
                        )

                    # -------------------------------------------------
                    # PCA-REDUCED MODAL PARAMETERISATION
                    # -------------------------------------------------
                    elif use_pca:
                        if not pca_cache_path:
                            raise RuntimeError(
                                "use_pca=True but no pca_cache_path provided in morph_basis.json"
                            )

                        # try to size from morph_basis first, otherwise pass through
                        pca_k_final = basis.get("pca_k_final", None)
                        if pca_k_final is not None:
                            pca_k_final = int(pca_k_final)
                            if coeffs.size < pca_k_final:
                                coeffs = np.pad(coeffs, (0, pca_k_final - coeffs.size))
                            elif coeffs.size > pca_k_final:
                                coeffs = coeffs[:pca_k_final]

                        d_ctrl = getDisplacements(
                            self.remote_output,
                            control_nodes=cn,
                            normals=cn_normals,
                            use_pca=True,
                            pca_cache_path=pca_cache_path,
                            pca_coeffs=coeffs,
                            normal_project=normal_project,
                            t_patch_scale=basis.get("t_patch_scale", None),
                            amp_alpha=float(basis.get("amp_alpha", 0.005)),
                            vector_mode=vector_mode,
                            frame_knn=frame_knn,
                            global_modes=global_modes,
                            global_mode_config=global_mode_config,
                            basis_axes=basis_axes,
                            use_local_modes=use_local_modes,
                            global_only=global_only,
                            protected_nodes=protected_nodes,
                            radius=protection_radius,
                        )

                    # -------------------------------------------------
                    # STANDARD MODAL PARAMETERISATION
                    # -------------------------------------------------
                    else:
                        n_global = (
                            len(global_mode_config)
                            if global_modes and global_mode_config
                            else (8 if global_modes else 0)
                        )

                        if use_local_modes:
                            if normal_project:
                                valid_local = (k,)
                                default_local = k
                            else:
                                if vector_mode == "xyz":
                                    valid_local = (3 * k,)
                                    default_local = 3 * k
                                else:
                                    valid_local = (k, 2 * k, 3 * k)
                                    default_local = 3 * k
                        else:
                            valid_local = (0,)
                            default_local = 0

                        valid_full = tuple(n_global + v for v in valid_local)
                        if coeffs.size in valid_local or coeffs.size in valid_full:
                            expected_len = int(coeffs.size)
                        else:
                            expected_len = n_global + default_local

                        if coeffs.size < expected_len:
                            coeffs = np.pad(coeffs, (0, expected_len - coeffs.size))
                        elif coeffs.size > expected_len:
                            coeffs = coeffs[:expected_len]

                        d_ctrl = getDisplacements(
                            self.remote_output,
                            seed=int(basis.get("seed", 0)),
                            control_nodes=cn,
                            normals=cn_normals,
                            coeffs=coeffs,
                            k_modes=k,
                            normal_project=normal_project,
                            t_patch_scale=basis.get("t_patch_scale", None),
                            amp_alpha=float(basis.get("amp_alpha", 0.005)),
                            vector_mode=vector_mode,
                            frame_knn=frame_knn,
                            global_modes=global_modes,
                            global_mode_config=global_mode_config,
                            basis_axes=basis_axes,
                            parameterisation_method="modal",
                            use_local_modes=use_local_modes,
                            global_only=global_only,
                            protected_nodes=protected_nodes,
                            radius=protection_radius,
                        )

                    d_ctrl = np.asarray(d_ctrl, dtype=float)
                else:
                    self._log("[PIPELINE] morph_basis_json has no control_nodes; leaving morph zero.")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._log(f"[PIPELINE][ERROR] Failed morph basis '{basis_path}': {e}\n{tb}")
                raise 

        # 4) Write final morph config
        self._log(
            f"[PIPELINE] Wrote morph_config_n_{self.n}.json with "
            f"{len(control_nodes)} CNs, "
            f"d_ctrl_norm={float(np.linalg.norm(d_ctrl)) if len(d_ctrl) else 0.0}"
        )
        
        rigid_flag = bool(basis.get("rigid_translation", False))
        
        config = {
            "mesh filetype": os.path.splitext(mesh_name)[1],
            "vtk_name": mesh_name,
            "output_directory": self.remote_output,
            "n": self.n,
            "gen": self.gen,
            "debug": True,
            "morph_kind": "mesh",
            "modal_coeffs": self.modal_coeffs,
            "t_surfaces": t_surfaces,
            "u_surfaces": u_surfaces,
            "c_surfaces": c_surfaces,
            "control_nodes": control_nodes,
            "displacement_vector": d_ctrl.tolist() if len(d_ctrl) else [],
            "rigid_translation": rigid_flag,
        }

        config_path = os.path.join(surf_dir, f"morph_config_n_{self.n}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return config_path, mesh_name
    
    def morph(self, n=0, predir=None, runafter=None):
        """Submit morph job"""
        surf_dir = os.path.join(self.remote_output, "surfaces", f"n_{self.gen}")
        os.makedirs(surf_dir, exist_ok=True)
        
        config_path, mesh_name = self._write_morph_config()
        fro_target = os.path.join(surf_dir, f"{self.base_name}_{n}.fro")
        
        if os.path.exists(fro_target) and os.path.getsize(fro_target) > 1024:
            self._log(f"[CLUSTER] Morph output exists, skipping submit: {fro_target}")
            self.job_ids["morph"] = None
            self._save_state(stage="morph_done", morph_output=fro_target)
            return None
        
        # Create batch file
        batch_name = f"morph_n{self.gen}_{n}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["time"] = "00-01:00"
        bf.sbatch_params["mem"] = "0"
        
        bf.lines.append(f"cd {surf_dir}")
        bf.lines.append(f"python3 {self.morph_script} {fro_target} {config_path}")
        
        batch_path = os.path.join(surf_dir, f"batchfile_{batch_name}")
        with open(batch_path, "w") as f:
            f.write(str(bf))
        
        dep_arg = f"--dependency=afterany:{runafter}" if runafter else ""
        cmd = f"sbatch {dep_arg} {batch_path}"
        result = subprocess.run(cmd, shell=True, cwd=surf_dir, capture_output=True, text=True)
        
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        jobid = match.group(1) if match else None
        
        self.job_ids["morph"] = jobid
        self._log(f"[CLUSTER] Morph job {jobid}")
        self._save_state(stage="morph_submitted", morph_job=jobid)

        return jobid
    
    def volume(self, predir=None, units="mm", runafter=None):
        """Submit volume mesher job"""
        vol_dir = os.path.join(self.remote_output, "volumes", f"n_{self.gen}")
        surf_dir = os.path.join(self.remote_output, "surfaces", f"n_{self.gen}")
        self.orig_dir = os.path.join(self.remote_output, "orig")
        os.makedirs(vol_dir, exist_ok=True)
        
        # Stage control files
        '''for fname in [f"{self.base_name}.bac", f"{self.base_name}.bpp"]:
            src = os.path.join(self.input_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, vol_dir)'''
        
        # Find and copy control file
        '''import glob
        ctl_files = glob.glob(os.path.join(self.input_dir, "Mesh3D_v*.ctl"))
        if ctl_files:
            shutil.copy2(ctl_files[-1], vol_dir)'''
        
        # Create batch file
        batch_name = f"vol_n{self.n}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["mem"] = "0"
        bf.sbatch_params["time"] = "04:00:00"
        
        bf.lines.append(self.intel_module)
        bf.lines.append(self.gnu_module)
        bf.lines.append(f"cd {vol_dir}/")
        bf.lines.append(f"cp {surf_dir}/{self.base_name}_{self.n}.fro {vol_dir}/ || true")
        bf.lines.append(f"cp {self.orig_dir}/{self.base_name}.bpp {vol_dir}/{self.base_name}_{self.n}.bpp || true")
        bf.lines.append(f"cp {self.orig_dir}/{self.base_name}.bac {vol_dir}/{self.base_name}_{self.n}.bac || true")
        bf.lines.append(f"cp {self.orig_dir}/Mesh3D_v50.ctl {vol_dir}/Mesh3D_v50.ctl || true")
        base = f"{self.base_name}_{self.n}"
        
        plt_path = os.path.join(vol_dir, f"{base}.plt")
        if os.path.exists(plt_path) and os.path.getsize(plt_path) > 1024:
            self._log(f"[CLUSTER] Volume output exists, skipping submit: {plt_path}")
            self.job_ids["volume"] = None
            self._save_state(stage="volume_done", volume_output=plt_path)
            return None

        # Run volume mesher (allow non-zero exit but keep outputs)
        bf.lines.append("")
        bf.lines.append("# ---- RUN VOLUME MESHER ----")
        bf.lines.append("set +e")  # do not abort batchfile if Mesh3D returns non-zero after writing outputs
        bf.lines.append(f"srun {self.volume_mesher} {base} &> volume_output_{self.n}")
        bf.lines.append("MESH_RC=$?")
        bf.lines.append("set -e")
        bf.lines.append("echo \"[VOL] Mesh3D return code: ${MESH_RC}\"")

        # UNIT CONVERSION (mm -> m or whatever your converter does) 
        if (self.units or "").lower() == "mm":
            self._log("[PIPELINE] CAD units = mm > adding PLT conversion to volume batchfile")

            bf.lines.append("")
            bf.lines.append("# ---- UNIT CONVERSION: run converter if PLT exists ----")
            bf.lines.append(f"if [ -f \"{base}.plt\" ]; then")
            bf.lines.append(f"  echo \"[VOL] Found {base}.plt -> running hyb_plt_converter\"")
            bf.lines.append(f"  {self.hyb_plt_converter} <<INPUT1")
            bf.lines.append(f"{base}")   # base name ONLY (no extension)
            bf.lines.append("INPUT1")
            bf.lines.append("")
            bf.lines.append(f"  mv {base}.plt {base}_mm.plt || true")
            bf.lines.append(f"  mv {base}_new.plt {base}.plt || true")
            bf.lines.append("else")
            bf.lines.append(f"  echo \"[VOL][ERROR] Missing {base}.plt; not converting. Mesh3D rc=${{MESH_RC}}\"")
            bf.lines.append("  exit ${MESH_RC}")
            bf.lines.append("fi")

        # Decide final exit code behavior:
        bf.lines.append("")
        bf.lines.append("# ---- FINALIZE ----")
        bf.lines.append(f"if [ -f \"{base}.plt\" ]; then")
        bf.lines.append("  exit 0")
        bf.lines.append("else")
        bf.lines.append("  exit ${MESH_RC}")
        bf.lines.append("fi")
        
        batch_path = os.path.join(vol_dir, f"batchfile_{self.gen}_{batch_name}")
        with open(batch_path, "w") as f:
            f.write(str(bf))
        
        dep_id = runafter or self.job_ids.get("morph")
        dep_arg = f"--dependency=afterany:{dep_id}" if dep_id else ""
        cmd = f"sbatch {dep_arg} {batch_path}"
        result = subprocess.run(cmd, shell=True, cwd=vol_dir, capture_output=True, text=True)
        
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        jobid = match.group(1) if match else None
        
        self.job_ids["volume"] = jobid
        self._log(f"[CLUSTER] Volume job {jobid}")
        self._save_state(stage="volume_submitted", volume_job=jobid)

        return jobid
    
    def prepro(self, runafter=None):
        """Submit preprocessor job"""
        pre_dir = os.path.join(self.remote_output, "preprocessed", f"n_{self.gen}", f"{self.n}")
        vol_dir = os.path.join(self.remote_output, "volumes", f"n_{self.gen}")
        os.makedirs(pre_dir, exist_ok=True)
        
        sol_glob_ok = any(
            fn.startswith(f"{self.base_name}_{self.n}.sol") for fn in os.listdir(pre_dir)
        )
        if sol_glob_ok:
            self._log(f"[CLUSTER] Prepro outputs exist, skipping submit: {pre_dir}")
            self.job_ids["prepro"] = None
            self._save_state(stage="prepro_done", prepro_dir=pre_dir)
            return None
        
        # Create batch file
        batch_name = f"pre_n{self.gen}_{self.n}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["ntasks"] = 1
        bf.sbatch_params["mem"] = "0"
        bf.sbatch_params["time"] = "06:00:00"
        bf.sbatch_params.pop("nodes", None)
        
        bf.lines.append(self.intel_module)
        bf.lines.append(self.gnu_module)
        bf.lines.append(f"cd {pre_dir}")
        bf.lines.append(f"ln {vol_dir}/{self.base_name}_{self.n}.plt {pre_dir}/ || true")
        bf.lines.append(f"cp {self.orig_dir}/rungen.inp {pre_dir}/rungen.inp || true")
        bf.lines.append(f"cp {self.orig_dir}/{self.base_name}.bco {pre_dir}/{self.base_name}_{self.n}.bco || true")
        bf.lines.append(f"sed -i '1s/.*/{self.base_name}_{self.n}/' rungen.inp")
        bf.lines.append(f"srun {self.prepro_exe} < rungen.inp &> prepro_output_{self.n}")
        
        batch_path = os.path.join(pre_dir, f"batchfile_{batch_name}")
        with open(batch_path, "w") as f:
            f.write(str(bf))
        
        dep_id = runafter or self.job_ids.get("volume")
        dep_arg = f"--dependency=afterany:{dep_id}" if dep_id else ""
        cmd = f"sbatch {dep_arg} {batch_path}"
        result = subprocess.run(cmd, shell=True, cwd=pre_dir, capture_output=True, text=True)
        
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        jobid = match.group(1) if match else None
        
        self.job_ids["prepro"] = jobid
        self._log(f"[CLUSTER] Prepro job {jobid}")
        self._save_state(stage="prepro_submitted", prepro_job=jobid)
        
        return jobid
    
    def solver(self, cond: dict = None, nc=1):
        """Submit solver job"""
        cond = cond or {}
        tag_dir = f"cond_{nc}/"
        tag_slug = f"cond_{nc}"
        
        sol_dir = os.path.join(self.remote_output, "solutions", f"n_{self.gen}", tag_dir, f"{self.n}/")
        pre_dir = os.path.join(self.remote_output, "preprocessed", f"n_{self.gen}", f"{self.n}")
        os.makedirs(sol_dir, exist_ok=True)
        
        # Create batch file
        batch_name = f"sol_n{self.gen}_{self.n}_{tag_slug}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["ntasks"] = str(self.sol_parallel_domains) if self.sol_parallel_domains != 1 else 80
        bf.sbatch_params["mem"] = "0"
        bf.sbatch_params["time"] = "3-00:00"
        bf.sbatch_params.pop("nodes", None)
        
        bf.lines.append(self.mpi_intel_module)
        bf.lines.append(f"cd {sol_dir}")
        bf.lines.append(f"ln {pre_dir}/base.plt {sol_dir}/ || true")
        bf.lines.append(f"cp {pre_dir}/plotreg.reg {sol_dir}/ || true")
        bf.lines.append(f"ln {pre_dir}/{self.base_name}_{self.n}.sol* {sol_dir}/ || true")
        
        # Copy files from orig_dir
        bf.lines.append(f"cp {self.orig_dir}/{self.base_name}.inp {sol_dir}/{self.base_name}_{self.n}.inp || true")
        bf.lines.append(f"cp {self.orig_dir}/run.inp {sol_dir}/run.inp || true")
        
        # Modify first line of run_{self.n}.inp
        bf.lines.append(f"sed -i '1s/.*/{self.base_name}_{self.n}/' run.inp")
        
        # Patch solver .inp file with conditions using inline Python
        bf.lines.append("python3 - <<'PY'")
        bf.lines.append("import re")
        bf.lines.append(f"inp_path = '{self.base_name}_{self.n}.inp'")
        bf.lines.append(f"processes = {self.sol_parallel_domains}")
        bf.lines.append(f"sol_dir = '{sol_dir}'")
        bf.lines.append("")
        bf.lines.append("with open(inp_path, 'r', encoding='utf-8', errors='ignore') as f:")
        bf.lines.append("    txt = f.read()")
        bf.lines.append("")
        
        # Patch numberOfProcesses
        bf.lines.append("txt = re.sub(")
        bf.lines.append("    r'(?im)^(\\s*ivd%numberOfProcesses\\s*=\\s*)\\d+',")
        bf.lines.append("    rf'\\g<1>{processes}',")
        bf.lines.append("    txt")
        bf.lines.append(")")
        bf.lines.append("")
        
        # Patch dataDirectory
        bf.lines.append("txt = re.sub(")
        bf.lines.append("    r\"(?im)^(\\s*ivd%dataDirectory\\s*=\\s*)'[^']*'\",")
        bf.lines.append("    rf\"\\g<1>'{sol_dir}'\",")
        bf.lines.append("    txt")
        bf.lines.append(")")
        bf.lines.append("")
        
        # Patch conditions if provided
        if "AoA" in cond or "alpha" in cond:
            aoa = cond.get("AoA", cond.get("alpha"))
            bf.lines.append("txt = re.sub(")
            bf.lines.append("    r'(?im)^(\\s*ivd%alpha\\s*=\\s*)[\\d.eE+-]+',")
            bf.lines.append(f"    rf'\\g<1>{float(aoa):.8f}',")
            bf.lines.append("    txt")
            bf.lines.append(")")
            bf.lines.append("")
        
        if "Mach" in cond or "M" in cond:
            mach = cond.get("Mach", cond.get("M"))
            bf.lines.append("txt = re.sub(")
            bf.lines.append("    r'(?im)^(\\s*ivd%MachNumber\\s*=\\s*)[\\d.eE+-]+',")
            bf.lines.append(f"    rf'\\g<1>{float(mach):.8f}',")
            bf.lines.append("    txt")
            bf.lines.append(")")
            bf.lines.append("")
        
        if "Re" in cond:
            bf.lines.append("txt = re.sub(")
            bf.lines.append("    r'(?im)^(\\s*ivd%ReynoldsNumber\\s*=\\s*)[\\d.eE+-]+',")
            bf.lines.append(f"    rf'\\g<1>{float(cond['Re']):.8e}',")
            bf.lines.append("    txt")
            bf.lines.append(")")
            bf.lines.append("")
        
        if "TurbModel" in cond:
            bf.lines.append("txt = re.sub(")
            bf.lines.append("    r'(?im)^(\\s*ivd%turbulenceModel\\s*=\\s*)\\d+',")
            bf.lines.append(f"    rf'\\g<1>{int(cond['TurbModel'])}',")
            bf.lines.append("    txt")
            bf.lines.append(")")
            bf.lines.append("")
        
        if "EngineFlow" in cond:
            bf.lines.append("txt = re.sub(")
            bf.lines.append("    r'(?im)^(\\s*ivd%engineFlowType\\s*=\\s*)\\d+',")
            bf.lines.append(f"    rf'\\g<1>{int(cond['EngineFlow'])}',")
            bf.lines.append("    txt")
            bf.lines.append(")")
            bf.lines.append("")
        
        if "MassFlow" in cond:
            bf.lines.append("txt = re.sub(")
            bf.lines.append("    r'(?im)^(\\s*ivd%enginesFrontMassFlow\\(1\\)\\s*=\\s*)[\\d.eE+-]+',")
            bf.lines.append(f"    rf'\\g<1>{float(cond['MassFlow']):.8f}',")
            bf.lines.append("    txt")
            bf.lines.append(")")
            bf.lines.append("")
        
        bf.lines.append("with open(inp_path, 'w', encoding='utf-8', newline='\\n') as f:")
        bf.lines.append("    f.write(txt)")
        bf.lines.append("PY")
        bf.lines.append("")
        
        self._append_previous_solution_initialisation(bf, sol_dir)
        
        # Safe defaults used later by monitor / ensight conversion
        mach_for_post = float(cond.get("Mach", cond.get("M", 1.0)))
        self._append_paraview_monitoring_section(bf, mach_for_post)
        
        # Run solver
        bf.lines.append(f"mpirun {self.solver_exe} < {self.base_name}_{self.n}.inp &> solver_output")
        bf.lines.append("touch SOLVER_DONE")

        bf.lines.append("wait $PR_PID 2>/dev/null || true")

        bf.lines.append(f"{self.combine_exe} <<INPUT1")
        bf.lines.append("plotreg.reg")
        bf.lines.append(f"{self.base_name}_{self.n}.res")
        bf.lines.append(f"{self.base_name}_{self.n}.unk")
        bf.lines.append("F")
        bf.lines.append("T")
        bf.lines.append("INPUT1")
        
        batch_path = os.path.join(sol_dir, f"batchfile_{batch_name}")
        with open(batch_path, "w") as f:
            f.write(str(bf))
        
        dep_id = self.job_ids.get("prepro")
        dep_arg = f"--dependency=afterany:{dep_id}" if dep_id else ""
        cmd = f"sbatch {dep_arg} {batch_path}"
        result = subprocess.run(cmd, shell=True, cwd=sol_dir, capture_output=True, text=True)
        
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        jobid = match.group(1) if match else None
        
        self.job_ids[f"solver_{tag_slug}"] = jobid
        self._log(f"[CLUSTER] Solver job {jobid}")

        # Convergence guard
        guard_job = self._submit_convergence_guard(
            tag_slug=tag_slug,
            sol_dir=sol_dir,
            stdout_name="solver_output",
            solver_job_id=jobid,
            res_threshold=-3.0,
            residual_csv=f"{self.base_name}.rsd",
        )
        self.job_ids[f"solver_guard_{tag_slug}"] = guard_job
        self._save_state(stage="solver_submitted", solver_job=jobid, solver_guard_job=guard_job)

        return jobid
    
    def _patch_solver_inp(self, inp_path: str, processes: int, sol_dir: str, cond: dict):
        """Patch solver input file with condition parameters"""
        with open(inp_path, "r") as f:
            txt = f.read()
        
        # Patch number of processes
        txt = re.sub(
            r"(?im)^(\s*ivd%numberOfProcesses\s*=\s*)\d+",
            rf"\g<1>{processes}",
            txt
        )
        
        # Patch data directory
        txt = re.sub(
            r"(?im)^(\s*ivd%dataDirectory\s*=\s*)'[^']*'",
            rf"\g<1>'{sol_dir}'",
            txt
        )
        
        # Patch conditions
        if "AoA" in cond or "alpha" in cond:
            aoa = cond.get("AoA", cond.get("alpha"))
            txt = re.sub(
                r"(?im)^(\s*ivd%alpha\s*=\s*)[\d.eE+-]+",
                rf"\g<1>{float(aoa):.8f}",
                txt
            ) 
        
        if "Mach" in cond or "M" in cond:
            mach = cond.get("Mach", cond.get("M"))
            txt = re.sub(
                r"(?im)^(\s*ivd%MachNumber\s*=\s*)[\d.eE+-]+",
                rf"\g<1>{float(mach):.8f}",
                txt
            )
        
        if "Re" in cond:
            txt = re.sub(
                r"(?im)^(\s*ivd%ReynoldsNumber\s*=\s*)[\d.eE+-]+",
                rf"\g<1>{float(cond['Re']):.8e}",
                txt
            )
        
        with open(inp_path, "w") as f:
            f.write(txt)