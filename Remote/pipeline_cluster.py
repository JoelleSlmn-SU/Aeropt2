# pipeline_cluster.py
# ----------------------------------------------------------------------
# CLUSTER-SIDE pipeline manager (no SSH/SFTP - direct filesystem access)
# - Used when the optimization batch job is already running ON the cluster
# - Mirrors HPCPipelineManager API but uses subprocess.run() instead of SSH
# ----------------------------------------------------------------------

import os, posixpath, shutil, re, textwrap, subprocess, json
from datetime import datetime

import numpy as np
from MeshGeneration.controlNodeDisp import estimate_normals, getDisplacements

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
        self.prepro_exe = config_dict.get("prepro_exe", "/home/s.o.hassan/bin/Gen3d_jj")
        self.solver_exe = config_dict.get("solver_exe", "/home/s.o.hassan/bin/UnsMgnsg3d") # TODO: CHANGE TO BENS SOLVER OR USE MAKEPLOT AND SPLITPLOT FROM OUBAY
        self.combine_exe = config_dict.get("combine_exe", "/home/s.engevabj/codes/utilities/makeplot2")
        self.ensight_exe = config_dict.get("ensight_exe", "/home/s.engevabj/codes/utilities/engen_tet")
        self.splitplot_exe = config_dict.get("splitplot_exe", "/home/s.engevabj/codes/utilities/splitplot2")
        self.makeplot_exe = config_dict.get("makeplot_exe", "/home/s.engevabj/codes/utilities/makeplot2")
        
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
        
        self.job_ids = {}
        if config_dict.get("parallel_domains") == 1:
            self.sol_parallel_domains = 80
        else:
            self.sol_parallel_domains = config_dict.get("parallel_domains", 80)
        
        # Setup logging
        log_dir = os.path.join(self.remote_output, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"pipeline_n{gen}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    def _log(self, msg: str):
        """Log to both stdout and file"""
        print(msg, flush=True)
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    
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


    def _write_morph_config(self, surf_dir: str):
        """Write morph configuration JSON for one design (self.n)."""
        # 1) Find the baseline mesh
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

        # ------------------------------------------------------
        # 3) Apply morph basis: surfaces, control nodes, d_ctrl
        # ------------------------------------------------------
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
                t_surfaces = basis["TSurfaces"]
                if cn.size > 0:
                    cn = cn.reshape((-1, 3))
                    control_nodes = cn.tolist()

                    t_surfaces = list(map(int, basis.get("TSurfaces", [])))
                    u_surfaces = list(map(int, basis.get("USurfaces", [])))
                    c_surfaces = list(map(int, basis.get("CSurfaces", [])))

                    # normals at control nodes
                    cn_normals = basis.get("control_normals", None)
                    if cn_normals is None:
                        cn_normals = estimate_normals(cn, knn=12)  # fallback for old jsons
                    cn_normals = np.asarray(cn_normals, float).reshape((-1, 3))

                    # design coeffs (from BO)
                    coeffs = np.asarray(self.modal_coeffs, dtype=float)
                    k_modes = int(basis.get("k_modes", max(1, coeffs.size)))
                    if coeffs.ndim > 1:
                        coeffs = coeffs.reshape(-1)
                    if coeffs.size < k_modes:
                        coeffs = np.pad(coeffs, (0, k_modes - coeffs.size))
                    elif coeffs.size > k_modes:
                        coeffs = coeffs[:k_modes]
                        
                    self._log(
                        f"[PIPELINE] gen={self.gen} n={self.n} "
                        f"modal_coeffs={self.modal_coeffs} (used coeffs={coeffs.tolist()})"
                    )

                    d_ctrl = getDisplacements(
                        self.remote_output,
                        seed=int(basis.get("seed", 0)),
                        control_nodes=cn,
                        normals=cn_normals,
                        coeffs=coeffs,
                        k_modes=k_modes,
                        normal_project=bool(basis.get("normal_project", True)),
                        t_patch_scale=basis.get("t_patch_scale", None),
                        amp_alpha=float(basis.get("amp_alpha", 0.02)),  # 2% default
                    )
                    d_ctrl = np.asarray(d_ctrl, dtype=float)
                else:
                    self._log("[PIPELINE] morph_basis_json has no control_nodes; leaving morph zero.")
            except Exception as e:
                self._log(f"[PIPELINE] Failed to use morph_basis_json '{basis_path}': {e}")

        # ------------------------------------------------------
        # 4) Write final morph config
        # ------------------------------------------------------
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
        
        config_path, mesh_name = self._write_morph_config(surf_dir)
        fro_target = os.path.join(surf_dir, f"{self.base_name}_{n}.fro")
        
        # Create batch file
        batch_name = f"morph_n{self.gen}_{n}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["time"] = "02-00:00"
        bf.sbatch_params["mem"] = "0"
        
        bf.lines.append(f"cd {surf_dir}")
        bf.lines.append(f"python3 {self.morph_script} {fro_target} {config_path}")
        
        batch_path = os.path.join(surf_dir, f"batchfile_{batch_name}")
        with open(batch_path, "w") as f:
            f.write(str(bf))
        
        dep_arg = f"--dependency=afterok:{runafter}" if runafter else ""
        cmd = f"sbatch {dep_arg} {batch_path}"
        result = subprocess.run(cmd, shell=True, cwd=surf_dir, capture_output=True, text=True)
        
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        jobid = match.group(1) if match else None
        
        self.job_ids["morph"] = jobid
        self._log(f"[CLUSTER] Morph job {jobid}")
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
        bf.lines.append(f"srun {self.volume_mesher} {self.base_name}_{self.n} &> volume_output_{self.n}")
        
        if (self.units or "").lower() == "mm":
            self._log("[PIPELINE] CAD units = mm → adding PLT conversion to volume batchfile")

            bf.lines.append("")
            bf.lines.append("# ---- UNIT CONVERSION: convert mesh to mm ----")
            bf.lines.append(f"{self.hyb_plt_converter} <<INPUT1")
            bf.lines.append(f"{self.base_name}_{self.n}")   # base name ONLY (no extension)
            bf.lines.append("INPUT1")
            bf.lines.append("")

            # rename outputs
            bf.lines.append(f"mv {self.base_name}_{self.n}.plt {self.base_name}_{self.n}_mm.plt || true")
            bf.lines.append(f"mv {self.base_name}_{self.n}_new.plt {self.base_name}_{self.n}.plt || true")
        
        batch_path = os.path.join(vol_dir, f"batchfile_{self.gen}_{batch_name}")
        with open(batch_path, "w") as f:
            f.write(str(bf))
        
        dep_id = runafter or self.job_ids.get("morph")
        dep_arg = f"--dependency=afterok:{dep_id}" if dep_id else ""
        cmd = f"sbatch {dep_arg} {batch_path}"
        result = subprocess.run(cmd, shell=True, cwd=vol_dir, capture_output=True, text=True)
        
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        jobid = match.group(1) if match else None
        
        self.job_ids["volume"] = jobid
        self._log(f"[CLUSTER] Volume job {jobid}")

        return jobid
    
    def prepro(self, runafter=None):
        """Submit preprocessor job"""
        pre_dir = os.path.join(self.remote_output, "preprocessed", f"n_{self.gen}", f"{self.n}")
        vol_dir = os.path.join(self.remote_output, "volumes", f"n_{self.gen}")
        os.makedirs(pre_dir, exist_ok=True)
        
        # Copy necessary files
        '''for fname in [f"{self.base_name}_{self.n}.bco", "rungen.inp"]:
            src = os.path.join(self.input_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, pre_dir)'''
        
        # Create batch file
        batch_name = f"pre_n{self.gen}_{self.n}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["ntasks"] = str(self.sol_parallel_domains) if self.sol_parallel_domains != 1 else 80
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
        dep_arg = f"--dependency=afterok:{dep_id}" if dep_id else ""
        cmd = f"sbatch {dep_arg} {batch_path}"
        result = subprocess.run(cmd, shell=True, cwd=pre_dir, capture_output=True, text=True)
        
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        jobid = match.group(1) if match else None
        
        self.job_ids["prepro"] = jobid
        self._log(f"[CLUSTER] Prepro job {jobid}")
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
        
        # --- background PR monitor (runs during solver) ---
        bf.lines.append("") 
        bf.lines.append("# ---------------- Pressure-recovery monitor ----------------")
        bf.lines.append("module load paraview/2019 || true")  # keep harmless if module already loaded

        bf.lines.append(f"BASE='{self.base_name}_{self.n}'")
        bf.lines.append("RSD=\"${BASE}.rsd\"")  # typically BASE.rsd; adjust if needed
        bf.lines.append("INTERVAL=50")
        bf.lines.append("SLEEP_S=20")

        bf.lines.append("PR_DIR=pr_monitor")
        bf.lines.append("ENS_DIR=\"$PR_DIR/ensight\"")
        bf.lines.append("CSV_OUT=\"$PR_DIR/pressure_recovery.csv\"")
        bf.lines.append("STATE_FILE=\"$PR_DIR/state_iter.txt\"")
        bf.lines.append("LOCK_DIR=\"$PR_DIR/lockdir\"")
        bf.lines.append("PV_SCRIPT=\"$PR_DIR/paraview_cluster.py\"")
        bf.lines.append("mkdir -p \"$ENS_DIR\"")

        # write pvpython script into the run dir (so each case has its own copy)
        bf.lines.append(f"cp {self.paraview_script} \"$PV_SCRIPT\"")


        # helper: get latest iter from rsd
        bf.lines.append("latest_iter() { [ -f \"$RSD\" ] || { echo 0; return; }; tail -n 1 \"$RSD\" | awk '{print $1}' || echo 0; }")

        # helper: one post step
        bf.lines.append("run_post() {")
        bf.lines.append("  it=\"$1\"")
        bf.lines.append("  mkdir \"$LOCK_DIR\" 2>/dev/null || { echo \"[PR] lock active, skip it=$it\"; return 0; }")
        bf.lines.append("  trap 'rmdir \"$LOCK_DIR\" 2>/dev/null || true' RETURN")
        bf.lines.append("  echo \"[PR] it=$it: makeplot2 -> engen_tet_mesh -> pvpython\"")

        # safer than reading half-written files: stage .res into temp and run utilities there
        bf.lines.append("  stage=\"$PR_DIR/stage_${it}\"")
        bf.lines.append("  rm -rf \"$stage\" && mkdir -p \"$stage\"")
        bf.lines.append("  rsync -a --include='*.res' --include='plotreg.reg' --exclude='*' ./ \"$stage/\" || true")
        bf.lines.append("  pushd \"$stage\" >/dev/null")

        # 1) combine region res -> unk
        bf.lines.append(f"  \"{self.makeplot_exe}\" <<INPUT1 > makeplot2_${{it}}.log 2>&1")
        bf.lines.append("plotreg.reg")
        bf.lines.append("${BASE}.res")
        bf.lines.append("${BASE}.unk")
        bf.lines.append("F")
        bf.lines.append("T")
        bf.lines.append("INPUT1")

        # 2) unk -> ensight
        bf.lines.append(f"  \"{self.ensight_exe}\" \"${{BASE}}.unk\" > engen_${{it}}.log 2>&1")

        # copy ensight outputs into ENS_DIR/it_XXXX
        bf.lines.append("  out_it=\"../ensight/it_${it}\"")
        bf.lines.append("  rm -rf \"$out_it\" && mkdir -p \"$out_it\"")
        bf.lines.append("  rsync -a ./*.case ./*.geo ./*velocity* ./*density* ./*energy* \"$out_it/\" 2>/dev/null || true")
        bf.lines.append("  popd >/dev/null")

        bf.lines.append("  case_path=\"$ENS_DIR/it_${it}/$ENSIGHT${BASE}.case\"")
        bf.lines.append("  if [ ! -f \"$case_path\" ]; then echo \"[PR][WARN] missing case: $case_path\"; return 0; fi")

        # 3) pvpython: append one line
        bf.lines.append("  pvpython --force-offscreen-rendering \"$PV_SCRIPT\" --case \"$case_path\" --iter \"$it\" --out \"$CSV_OUT\" --append || echo \"[PR][WARN] pvpython failed it=$it\"")
        bf.lines.append("}")

        # monitor loop in background
        bf.lines.append("monitor_pr() {")
        bf.lines.append("  last=0; [ -f \"$STATE_FILE\" ] && last=$(cat \"$STATE_FILE\" 2>/dev/null || echo 0)")
        bf.lines.append("  while true; do")
        bf.lines.append("    [ -f SOLVER_DONE ] && break")
        bf.lines.append("    it=$(latest_iter)")
        bf.lines.append("    if [ \"$it\" -ge \"$INTERVAL\" ] && [ $((it % INTERVAL)) -eq 0 ] && [ \"$it\" -gt \"$last\" ]; then")
        bf.lines.append("      run_post \"$it\" || true")
        bf.lines.append("      echo \"$it\" > \"$STATE_FILE\"")
        bf.lines.append("      last=\"$it\"")
        bf.lines.append("    fi")
        bf.lines.append("    sleep \"$SLEEP_S\"")
        bf.lines.append("  done")
        bf.lines.append("}")
        bf.lines.append("monitor_pr &")
        bf.lines.append("PR_PID=$!")
        bf.lines.append("# -----------------------------------------------------------")
        bf.lines.append("")

        
        # Run solver
        bf.lines.append(f"mpirun {self.solver_exe} < {self.base_name}_{self.n}.inp &> solver_output")
        bf.lines.append("touch SOLVER_DONE")
        #bf.lines.append("kill $PR_PID 2>/dev/null || true")
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
        dep_arg = f"--dependency=afterok:{dep_id}" if dep_id else ""
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