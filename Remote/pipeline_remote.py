# pipeline_remote.py
# ----------------------------------------------------------------------
# Remote (HPC) pipeline manager
# - Probes input-folder BAC/BPP/CTL/SURFACE in __init__
# - Each stage calls _stage_file(...) to ensure inputs exist
# - morph(): submits morph job that must write <base>.fro in surfaces/n_<n>
# - surface(): stages CAD+BAC+BPP+CTL, uploads, submits surface mesher
# - volume(): can start after surface OR cold-start from input surface mesh
# - prepro(), solver(): unchanged submission pattern
# ----------------------------------------------------------------------

import os, posixpath, shutil, re, textwrap
from Remote.runSimRemote import runSurfMorph
from ConvertFileType.convertVtmtoFro import vtm_to_fro
from ConvertFileType.convertEnGeoToFro import engeo_to_fro

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
        return "\n".join(hdr + [
            'source ~/.bashrc',
            ': "${PROMPT_COMMAND:=}"',
            ': "${PYTHONPATH:=}"',
            'set -euo pipefail',
        ] + self.lines + [""])

class HPCPipelineManager:
    def __init__(self, main_window, n=0):
        self.main_window = main_window
        self.mesh_viewer = main_window.mesh_viewer
        self.geo_viewer = main_window.geo_viewer
        self.n = n

        # cluster handles (adapt to your app wiring)
        self.ssh_client    = getattr(main_window, "ssh_client", None)
        self.remote_output = getattr(main_window, "remote_output_dir", None)
        self.intel_module  = getattr(self, "intel_module", "module load compiler/intel/2020/0")
        self.gnu_module    = getattr(self, "gnu_module",   "module load compiler/gnu/12/1.0")
        self.mpi_module    = getattr(self, "mpi_module",   "module load mpi/mpich/3.2.1")
        self.mpi_intel_module    = getattr(self, "mpi_intel_module",   "module load mpi/intel/2020/0")

        # executables / wrappers
        self.morph_cmd       = getattr(self, "morph_cmd",       "remoteMorph.py")  # Python entrypoint on cluster PATH
        self.surface_mesher  = getattr(self, "surface_mesher",  "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Surf3D")
        self.volume_mesher   = getattr(self, "volume_mesher",   "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Mesh3D")
        self.prepro_exe      = getattr(self, "prepro_exe",      "/home/s.o.hassan/bin/Gen3d_jj")
        self.solver_exe      = getattr(self, "solver_exe",      "/home/s.o.hassan/bin/UnsMgnsg3d")
        self.combine_exe     = getattr(self, "combine_exe",     "/home/s.engevabj/codes/utilities/makeplot2")
        self.ensight_exe     = getattr(self, "ensight_exe",     "/home/s.engevabj/codes/utilities/engen_tet")
        
        self.splitplot_exe = getattr(self, "splitplot_exe", "/home/s.engevabj/codes/utilities/splitplot2")
        self.makeplot_exe  = getattr(self, "makeplot_exe",  "/home/s.engevabj/codes/utilities/makeplot2")

        # paths
        self.input_path = getattr(self.main_window, "input_file_path", "")
        self.input_dir  = getattr(self.main_window, "input_directory", "") or os.path.dirname(self.input_path)
        self.base_name  = os.path.splitext(os.path.basename(self.input_path))[0] if self.input_path else "model"
        self.output_dir = getattr(self.main_window, "output_directory", "") or os.getcwd()
        self.templates_dir = os.path.join(os.getcwd(), "templates")
        os.makedirs(self.output_dir, exist_ok=True)

        # surfacedir/vol/pre/sol roots
        self.job_ids = {}

        # probe input-folder assets (highest precedence)
        def _probe(p): return p if (p and os.path.exists(p)) else None
        self.bac_src  = _probe(os.path.join(self.input_dir, f"{self.base_name}.bac"))
        self.bpp_src  = _probe(os.path.join(self.input_dir, f"{self.base_name}.bpp"))
        self.ctl_surf_src = self._find_ctl("surf")
        self.ctl_vol_src  = self._find_ctl("vol")
        # surface mesh candidates for cold-start volume
        self.fro_src  = _probe(os.path.join(self.input_dir, f"{self.base_name}.fro"))
        self.vtk_src  = _probe(os.path.join(self.input_dir, f"{self.base_name}.vtk"))
        self.vtm_src  = _probe(os.path.join(self.input_dir, f"{self.base_name}.vtm"))

        # original upload dir (if you have one elsewhere, keep as is)
        self.orig_dir = getattr(self, "orig_dir", posixpath.join(self.remote_output or "~", "orig"))

        self._log(f"[HPC] Pipeline init: base='{self.base_name}' n={self.n}")

    # ---------------------------- helpers ------------------------------
    def _log(self, msg: str):
        try:    self.main_window.logger.log(msg)
        except: print(msg)
        
    def _jobs_log_path(self):
        # local file alongside your run outputs
        return os.path.join(os.getcwd(), "jobs_log.json")

    def _append_job_log(self, entry: dict):
        import json
        path = self._jobs_log_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []
        data.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
    def _find_ctl(self, kind: str) -> str | None:
        """
        kind: 'surf' -> Surf3D_v*.ctl
            'vol'  -> Mesh3D_v*.ctl
        Search order: input_dir -> output_dir -> templates_dir.
        Returns absolute path or None.
        """
        import glob
        pat = "Surf3D_v*.ctl" if kind == "surf" else "Mesh3D_v*.ctl"
        for root in (self.input_dir, self.output_dir, self.templates_dir):
            if not root:
                continue
            matches = sorted(glob.glob(os.path.join(root, pat)))
            if matches:
                # pick the "latest" by lexical order (v50 > v25 etc.)
                return matches[-1]
        return None

    def _stage_file(self, kind: str, stage_dir: str) -> str:
        """
        Ensure the requested file exists in stage_dir (local).
        kinds:
        - 'bac' -> <base>.bac
        - 'bpp' -> <base>.bpp
        - 'bco' -> <base>.bco
        - 'ctl_surf' -> Surf3D_v*.ctl (surface mesher control)
        - 'ctl_vol'  -> Mesh3D_v*.ctl (volume mesher control)
        Precedence for BAC/BPP: input-folder -> output_dir -> templates -> generated
        Precedence for CTL: _find_ctl(kind) across input/output/templates -> fallback stub
        """
        os.makedirs(stage_dir, exist_ok=True)

        if kind == "bac":
            target = os.path.join(stage_dir, f"{self.base_name}.bac")
            src_candidates = [
                getattr(self, "bac_src", None),
                os.path.join(self.output_dir, f"{self.base_name}.bac"),
                os.path.join(self.templates_dir, "filename.bac"),
            ]
            for src in src_candidates:
                if src and os.path.exists(src):
                    shutil.copyfile(src, target); return target
            # last resort: generate
            from FileRW.BacFile import BacFile
            fn = getattr(BacFile, "defaultCRMFineMesh", None)
            if callable(fn):
                with open(target, "w", newline="\n") as f: f.write(str(fn(name=self.base_name)))
            else:
                with open(target, "w", newline="\n") as f: f.write("# BAC (fallback)\n")
            return target

        if kind == "bpp":
            target = os.path.join(stage_dir, f"{self.base_name}.bpp")
            src_candidates = [
                getattr(self, "bpp_src", None),
                os.path.join(self.output_dir, f"{self.base_name}.bpp"),
                os.path.join(self.templates_dir, "filename.bpp"),
            ]
            for src in src_candidates:
                if src and os.path.exists(src):
                    shutil.copyfile(src, target); return target
            # last resort: generate
            from FileRW.BppFile import BppFile
            with open(target, "w", newline="\n") as f: f.write(str(BppFile(self.base_name)))
            return target
        
        if kind == "bco":
            target = os.path.join(stage_dir, f"{self.base_name}.bco")
            # search order: input_dir → output_dir → templates
            candidates = [
                os.path.join(self.input_dir or "", f"{self.base_name}.bco"),
                os.path.join(self.output_dir, f"{self.base_name}.bco"),
                os.path.join(self.templates_dir, "filename.bco"),
            ]
            src = next((p for p in candidates if p and os.path.exists(p)), None)
            if src:
                shutil.copyfile(src, target)
            else:
                # minimal placeholder if nothing found (prepro may require it)
                with open(target, "w", newline="\n") as f:
                    f.write("# BCO placeholder\n")
            return target

        if kind in ("ctl_surf", "ctl_vol"):
            target_name = None
            # try to find a real control file
            ctl_src = self._find_ctl("surf" if kind == "ctl_surf" else "vol")
            if ctl_src:
                target_name = os.path.basename(ctl_src)
                target = os.path.join(stage_dir, target_name)
                shutil.copyfile(ctl_src, target)
                return target
            # fallback stub if nothing found
            target_name = "Surf3D_v00.ctl" if kind == "ctl_surf" else "Mesh3D_v00.ctl"
            target = os.path.join(stage_dir, target_name)
            with open(target, "w", newline="\n") as f:
                f.write("# auto-generated placeholder control file\n")
            return target

        raise ValueError(f"unknown kind={kind}")

    def _pick_surface_mesh(self):
        """Find a usable surface mesh for cold-start. Returns absolute local path or None."""
        import glob, os
        tried = []

        def _add(p):
            if p and p not in tried:
                tried.append(p)

        # 1) the loaded file itself (if it is a surface mesh)
        p = getattr(self, "input_path", None)
        if p and os.path.exists(p) and os.path.splitext(p)[1].lower() in (".fro", ".vtk", ".vtm"):
            return p

        # 2) explicit sources probed at __init__
        for p in (getattr(self, "fro_src", None), getattr(self, "vtk_src", None), getattr(self, "vtm_src", None)):
            if p and os.path.exists(p):
                return p
            _add(p)

        # 3) glob the input directory
        inp = getattr(self, "input_dir", "") or ""
        for pat in ("*.fro", "*.vtk", "*.vtm"):
            for p in glob.glob(os.path.join(inp, pat)):
                if os.path.exists(p):
                    return p
                _add(p)

        # 4) glob the output directory and surfaces folder (user may have saved there already)
        out = getattr(self, "output_dir", "") or ""
        for pat in ("*.fro", "*.vtk", "*.vtm"):
            for p in glob.glob(os.path.join(out, pat)):
                if os.path.exists(p):
                    return p
                _add(p)
        surf_out = os.path.join(out, "surfaces", f"n_{self.n}")
        for pat in ("*.fro", "*.vtk", "*.vtm"):
            for p in glob.glob(os.path.join(surf_out, pat)):
                if os.path.exists(p):
                    return p
                _add(p)

        # Log what we tried to make debugging easier
        self._log("[HPC][volume] Could not find a surface mesh. Tried:")
        for t in tried:
            if t:
                self._log(f"    {t}")
        return None
    
    def _upload_batch_and_submit(self, batchfile_local: str, remote_dir: str, batch_name: str, dep_jobid: str = None):
        sftp = self.ssh_client.open_sftp()
        try:
            # ensure remote_dir exists
            parts = remote_dir.strip("/").split("/")
            path = ""
            for p in parts:
                path += "/" + p
                try:    sftp.mkdir(path)
                except: pass
            sftp.put(batchfile_local, posixpath.join(remote_dir, f"batchfile_{batch_name}"))
        finally:
            sftp.close()

        dep = f"--dependency=afterany:{dep_jobid}" if dep_jobid else ""
        cmd = f"bash -lc 'source ~/.bashrc; cd {remote_dir}; sbatch {dep} batchfile_{batch_name}'"
        self._log(f"[HPC] submit: {cmd}")
        _in,_out,_err = self.ssh_client.exec_command(cmd)
        out = _out.read().decode().strip()
        err = _err.read().decode().strip()
        if err: self._log(f"[HPC][stderr] {err}")
        if "Submitted batch job" not in out:
            raise RuntimeError(f"Submit failed. out='{out}' err='{err}'")
        return out.split()[-1]

    def _ensure_remote_dir(self, remote_dir: str):
        sftp = self.ssh_client.open_sftp()
        try:
            parts = remote_dir.strip("/").split("/")
            path = ""
            for p in parts:
                path += "/" + p
                try:    sftp.mkdir(path)
                except: pass
        finally:
            sftp.close()

    def _mkdir_p_remote(self, path: str):
        # Use the login shell to expand ~ and $HOME and create parents
        cmd = f"bash -lc 'mkdir -p {path}'"
        _in, _out, _err = self.ssh_client.exec_command(cmd)
        err = _err.read().decode().strip()
        if err:
            # Not fatal for mkdir -p, but log it so we see permission issues
            try:
                self.main_window.logger.log(f"[HPC mkdir -p stderr] {err}")
            except Exception:
                pass
    
    def _ensure_remote_convergence_script(self, sol_dir: str) -> str:
        """
        Ensure convergenceCheck.py exists in the solution folder on the cluster.
        Returns the remote path to the script.
        """
        script_remote = posixpath.join(sol_dir, "convergenceCheck.py")
        # Try common local locations; fall back to CWD.
        local_candidates = [
            os.path.join(os.getcwd(), "Remote", "convergenceCheck.py"),
            os.path.join(os.getcwd(), "convergenceCheck.py"),
        ]
        local_script = next((p for p in local_candidates if os.path.exists(p)), None)
        if not local_script:
            # Last-resort: the user put it somewhere else; keep name but raise a clear error.
            raise FileNotFoundError("Could not find a local convergenceCheck.py to upload.")
        sftp = self.ssh_client.open_sftp()
        try:
            # overwrite each time (script is tiny, avoids caching stale versions)
            sftp.put(local_script, script_remote)
        finally:
            sftp.close()
        return script_remote
    
    def _here_doc(self, tag: str, lines: list[str]) -> str:
        body = "\n".join(lines)
        return f"<<{tag}\n{body}\n{tag}"
    
    def _submit_convergence_guard(
        self,
        aoa_or_tag,
        sol_dir,
        stdout_name="solver_output",
        solver_job_id=None,
        res_threshold=-3,
        residual_csv=None,
        runafter = None,
    ):
        """
        Simple two-step guard:
        1) guardcheck: run convergenceCheck.py → writes last_convergence.json
        2) guarddecide: read last_convergence.json → if not converged:
            - bump 'restart number' in <base>.inp
            - run splitplot2
            - sbatch batchfile_solver (your standard solver batch)
        """
        residual_csv = residual_csv or f"{self.base_name}.rsd"

        # --- ensure convergenceCheck.py is on the cluster (same dir as results)
        self._ensure_remote_convergence_script(sol_dir)

        # ------------------------------------------------------------------
        # 1) guardcheck: write last_convergence.json
        # ------------------------------------------------------------------
        check_name = f"guardcheck_{aoa_or_tag}"
        bf1 = Batchfile(check_name)
        bf1.sbatch_params["output"] = f"{check_name}.out"
        bf1.sbatch_params["error"]  = f"{check_name}.err"
        if solver_job_id:
            bf1.sbatch_params["dependency"] = f"afterany:{solver_job_id}"

        bf1.lines.append(f"cd {sol_dir}")
        # ATTEMPT: from convergence_state.json if present, else 0
        bf1.lines.append('ATTEMPT=0')
        bf1.lines.append('if [ -f convergence_state.json ]; then ATTEMPT=$(python3 - <<PY\n'
                        'import json; print(json.load(open("convergence_state.json")).get("attempt",0))\n'
                        'PY\n); fi')
        # Convergence check → JSON file
        bf1.lines.append(
            'python3 ./convergenceCheck.py '
            f'"{sol_dir}" "$ATTEMPT" "2" "{float(res_threshold)}" '
            "'' "  # empty force_csv (we read Cl/Cd from .rsd)
            f'"{residual_csv}" "{stdout_name}" > last_convergence.json'
        )

        # write/upload/submit
        local1 = os.path.join(self.output_dir, f"batchfile_{check_name}")
        with open(local1, "w", newline="\n") as f: f.write(str(bf1))
        check_job = self._upload_batch_and_submit(local1, sol_dir, check_name, dep_jobid=solver_job_id)
        self._log(f"[HPC] Guard-check job {check_job}")

        # ------------------------------------------------------------------
        # 2) guarddecide: read last_convergence.json and act
        # ------------------------------------------------------------------
        decide_name = f"guarddecide_{aoa_or_tag}"
        bf2 = Batchfile(decide_name)
        bf2.sbatch_params["output"] = f"{decide_name}.out"
        bf2.sbatch_params["error"]  = f"{decide_name}.err"
        bf2.sbatch_params["dependency"] = f"afterany:{check_job}"

        bf2.lines.append(f"cd {sol_dir}")
        # Parse JSON (very small inline Python to be robust)
        bf2.lines.append('CONV=$(python3 - <<PY\n'
                        'import json; d=json.load(open("last_convergence.json"))\n'
                        'print("1" if d.get("converged") else "0")\nPY\n)')
        bf2.lines.append('REASON=$(python3 - <<PY\n'
                        'import json; d=json.load(open("last_convergence.json"))\n'
                        'print(d.get("reason",""))\nPY\n)')
        bf2.lines.append('echo "[GUARD] converged=$CONV reason=$REASON"')

        # If converged → exit; else bump restart, run splitplot2, re-sbatch solver
        bf2.lines.append('if [ "$CONV" = "1" ]; then')
        bf2.lines.append('  echo "[GUARD] Converged. Nothing to do."')
        bf2.lines.append('  exit 0')
        bf2.lines.append('fi')

        # Next attempt number (persist attempt in file)
        bf2.lines.append('NEXT=1')
        bf2.lines.append('if [ -f convergence_state.json ]; then')
        bf2.lines.append("  NEXT=$(python3 - <<'PY'\n"
                        "import json\n"
                        "with open('convergence_state.json','r',encoding='utf-8') as f:\n"
                        "    d = json.load(f)\n"
                        "print(d.get('attempt',0) + 1)\n"
                        "PY\n"
                        ")")
        bf2.lines.append('fi')

        # Write convergence_state.json using json.dump (always valid JSON)
        bf2.lines.append("python3 - <<PY\n"
                        "import json\n"
                        "data = {'attempt': int('" + "${NEXT}" + "'), 'last_reason': '" + "${REASON}" + "'}\n"
                        "with open('convergence_state.json','w',encoding='utf-8') as f:\n"
                        "    json.dump(data, f)\n"
                        "PY")

        # Patch restart number in <base>.inp to $NEXT
                # Patch restart number in <base>.inp by bumping the existing value (0→1, 1→2, ...)
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
        bf2.lines.append("    return prefix + str(new)")
        bf2.lines.append("")
        # allow:  "restart number 0", "restart number = 0", "restart   number:    0"
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
        # Match: "ivd%restartNumber = 0," with optional spaces and optional comma
        bf2.lines.append(r"pattern = r'(?im)^(?P<p>\s*ivd%restartNumber\s*=\s*)(?P<n>\d+)(?P<trail>\s*,?)'")
        bf2.lines.append("txt2, n = re.subn(pattern, bump, txt, count=1)")
        bf2.lines.append("if n == 0:")
        bf2.lines.append("    print('[GUARD] WARNING: ivd%restartNumber line not found in file')")
        bf2.lines.append("else:")
        bf2.lines.append("    with open(p,'w',encoding='utf-8',newline='\\n') as f:")
        bf2.lines.append("        f.write(txt2)")
        bf2.lines.append("PY")

        # Run makeplot2 and splitplot2 with your quick here-doc, then re-submit the standard solver batch
        bf2.lines.append(f'"{self.makeplot_exe}" <<INPUT1')
        bf2.lines.append('plotreg.reg')
        bf2.lines.append(f'{self.base_name}.res')
        bf2.lines.append(f'{self.base_name}.unk')
        bf2.lines.append('F')
        bf2.lines.append('T')
        bf2.lines.append('INPUT1')
        
        bf2.lines.append(f'"{self.splitplot_exe}" <<INPUT2')
        bf2.lines.append('plotreg.reg')
        bf2.lines.append(f'{self.base_name}.unk')
        bf2.lines.append(f'{self.base_name}.rst')
        bf2.lines.append('T')
        bf2.lines.append('INPUT2')

        # Finally: re-run your solver by re-submitting the same solver batchfile
        bf2.lines.append(f"sbatch batchfile_sol_n{self.n}_{aoa_or_tag}")

        # write/upload/submit
        local2 = os.path.join(self.output_dir, f"batchfile_{decide_name}")
        with open(local2, "w", newline="\n") as f: f.write(str(bf2))
        decide_job = self._upload_batch_and_submit(local2, sol_dir, decide_name, dep_jobid=check_job)
        self._log(f"[HPC] Guard-decide job {decide_job}")

        self.job_ids[f"guardcheck_{aoa_or_tag}"] = check_job
        self.job_ids[f"guarddecide_{aoa_or_tag}"] = decide_job
        return decide_job
    
    def job_status(self, jobid: str) -> str:
        """Return a compact Slurm status line for a job."""
        cmd = f"bash -lc \"sacct -j {jobid} --format=JobID,JobName,Partition,AllocCPUS,State,Elapsed --parsable2 --noheader | head -n1\""
        _in, _out, _err = self.ssh_client.exec_command(cmd)
        out = _out.read().decode().strip()
        err = _err.read().decode().strip()
        if err and "job has no step" not in err.lower():
            self._log(f"[HPC][status stderr] {err}")
        if not out:
            # fallback to squeue if job hasn't started accounting yet
            cmd2 = f"bash -lc \"squeue -j {jobid} -o '%i|%j|%T|%M|%R' --noheader | head -n1\""
            _in2, _out2, _err2 = self.ssh_client.exec_command(cmd2)
            out2 = _out2.read().decode().strip()
            err2 = _err2.read().decode().strip()
            if err2:
                self._log(f"[HPC][status stderr] {err2}")
            return out2 or "UNKNOWN"
        return out
    
    '''def list_jobs_me(self) -> str:
        """
        Return a compact table of all your pending/running jobs.
        Uses squeue --me, falls back to sacct if needed.
        """
        fmt = "%i|%j|%T|%M|%D|%R"  # JobID|Name|State|Elapsed|Nodes|Reason/NodeList
        cmd = f"bash -lc \"squeue --me -o '{fmt}' --noheader | sort\""
        _in, _out, _err = self.ssh_client.exec_command(cmd)
        out = _out.read().decode().strip()
        err = _err.read().decode().strip()
        if not out:
            # maybe nothing in queue yet → show last few finished today
            cmd2 = "bash -lc \"sacct --user=$USER --state=ALL --format=JobID,JobName,State,Elapsed -X --parsable2 --noheader | tail -n 10\""
            _in2, _out2, _err2 = self.ssh_client.exec_command(cmd2)
            out = _out2.read().decode().strip()
            err = _err2.read().decode().strip()
        if err:
            self.main_window.logger.log(f"[HPC][status stderr] {err}")
        return out


    def cancel_job(self, jobid: str) -> bool:
        """
        Cancel a job by ID. Returns True on success, False on error.
        """
        jobid = jobid.strip()
        if not jobid.isdigit():
            self.main_window.logger.log(f"[HPC] Invalid job id: '{jobid}'")
            return False
        cmd = f"bash -lc 'scancel {jobid}'"
        _in, _out, _err = self.ssh_client.exec_command(cmd)
        err = _err.read().decode().strip()
        if err:
            self.main_window.logger.log("[HPC][cancel stderr] {err}")
            return False
        return True'''

    def cancel_job(self, jobid: str) -> bool:
        """Attempt to cancel a job; returns True if scancel returned no error."""
        cmd = f"bash -lc 'scancel {jobid}'"
        _in, _out, _err = self.ssh_client.exec_command(cmd)
        err = _err.read().decode().strip()
        if err:
            self._log(f"[HPC][cancel stderr] {err}")
            return False
        return True

    def _loaded_surface_path(self):
        """Return a usable surface-mesh file path if the LOADED file itself is .fro/.vtk/.vtm."""
        p = self.input_path
        if not p:
            return None
        ext = os.path.splitext(p)[1].lower()
        return p if ext in (".fro", ".vtk", ".vtm", ".case") and os.path.exists(p) else None
    
    # ---------------------------- stages -------------------------------

    def morph(self, predir=None, runafter=None):
        assert self.remote_output and self.ssh_client
        surf_dir = posixpath.join(self.remote_output, "surfaces", f"n_{self.n}/")
        self._mkdir_p_remote(surf_dir)

        morph_id = runSurfMorph(self.mesh_viewer, n=self.n, debug=True, run_as_batch=True)
        
        batch_name = f"morph_n{self.n}"
        bf = Batchfile(batch_name)
        bf.lines.append(self.intel_module)
        bf.lines.append(self.gnu_module)
        bf.lines.append(f"cd {surf_dir}")
        bf.lines.append(f"{self.morph_cmd} {self.base_name} &> morph_output")

        local_batch = os.path.join(self.output_dir, f"batchfile_{batch_name}")
        with open(local_batch, "w", newline="\n") as f:
            f.write(str(bf))

        jobid = self._upload_batch_and_submit(local_batch, surf_dir, batch_name, dep_jobid=runafter)
        self.job_ids["morph"] = jobid
        self._log(f"[HPC] Morph job {jobid}")
        return jobid

    def surface(self, predir=None, runafter=None):
        """
        Stage CAD+BAC+BPP+CTL locally, upload, submit surface mesher.
        Writes <base>.fro in surfaces/n_<n>.
        """
        assert self.remote_output and self.ssh_client
        surf_dir = posixpath.join(self.remote_output, "surfaces", f"n_{self.n}/")
        self._mkdir_p_remote(surf_dir)

        # local staging
        local_stage = os.path.join(self.output_dir, f"stage_surface_n_{self.n}")
        os.makedirs(local_stage, exist_ok=True)
        cad_ext = os.path.splitext(self.input_path)[1].lower()
        shutil.copyfile(self.input_path, os.path.join(local_stage, f"{self.base_name}{cad_ext}"))
        self._stage_file("bac",       local_stage)
        self._stage_file("bpp",       local_stage)
        ctl_surf_path = self._stage_file("ctl_surf", local_stage)
        ctl_surf_name = os.path.basename(ctl_surf_path)

        # upload staged
        sftp = self.ssh_client.open_sftp()
        try:
            for fname in [f"{self.base_name}{cad_ext}", f"{self.base_name}.bac", f"{self.base_name}.bpp", ctl_surf_name]:
                sftp.put(os.path.join(local_stage, fname), posixpath.join(surf_dir, fname))
        finally:
            sftp.close()

        # batch
        batch_name = f"surf_n{self.n}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["mem"] = "0"
        bf.lines.append(self.intel_module)
        bf.lines.append(self.gnu_module)
        bf.lines.append(f"cd {surf_dir}")
        bf.lines.append(f"mpirun {self.surface_mesher} {self.base_name} &> surface_output")

        local_batch = os.path.join(self.output_dir, f"batchfile_{batch_name}")
        with open(local_batch, "w", newline="\n") as f: f.write(str(bf))
        jobid = self._upload_batch_and_submit(local_batch, surf_dir, batch_name, dep_jobid=runafter)
        self.job_ids["surface"] = jobid
        self._log(f"[HPC] Surface job {jobid}")
        return jobid

    def volume(self, predir=None, units="mm", runafter=None):
        """
        Submit volume mesher.
        - If a surface or morph stage ran, copy data from surfaces/n_<n>.
        - Otherwise, cold-start using the loaded surface mesh (.fro/.vtk/.vtm)
        or one found nearby.
        - If the source is .vtm/.vtk, convert to .fro and use/save that.
        - Always place the surface mesh into BOTH surfaces/n_<n> and volumes/n_<n>.
        - Always stage BAC/BPP/CTL into volumes/n_<n> when cold-starting.
        """
        import os
        import posixpath
        from ConvertFileType.convertVtmtoFro import vtm_to_fro  # <-- ensure this exists

        assert self.remote_output and self.ssh_client
        
        units = self.main_window.cad_units
        

        vol_dir  = posixpath.join(self.remote_output, "volumes",  f"n_{self.n}/")
        surf_dir = posixpath.join(self.remote_output, "surfaces", f"n_{self.n}/")
        self._mkdir_p_remote(vol_dir)
        self._mkdir_p_remote(surf_dir)

        have_surface_stage = ("surface" in self.job_ids) or ("morph" in self.job_ids)

        # ---- helper: enumerate candidates and log what we’ll try
        tried = []
        def _add_try(p):
            if p and p not in tried:
                tried.append(p)

        # 1) the LOADED file itself, if it is a surface mesh
        input_dir = getattr(self, "input_dir", "") or ""
        base_fro = os.path.join(input_dir, f"{self.base_name}.fro")

        if os.path.exists(base_fro):
            self._log(f"[HPC][volume] Found pre-existing FRO file in input directory: {base_fro}")
            local_surf = base_fro
        else:
            # 1) the LOADED file itself
            loaded = getattr(self, "input_path", None)
            if loaded and os.path.exists(loaded) and os.path.splitext(loaded)[1].lower() in (".fro", ".vtk", ".vtm"):
                local_surf = loaded
            else:
                # 2) sources probed at __init__
                local_surf = None
                for p in (getattr(self, "fro_src", None), getattr(self, "vtk_src", None), getattr(self, "vtm_src", None)):
                    if p and os.path.exists(p):
                        ext = os.path.splitext(p)[1].lower()
                        if ext in (".vtk", ".vtm"):
                            self._log(f"[HPC][volume] No .fro found — converting {p} to .fro (temporary).")
                            from ConvertFileType.convertVtmtoFro import vtm_to_fro
                            fro_out = os.path.join(self.output_dir, "surfaces", f"n_{self.n}", f"{self.base_name}.fro")
                            os.makedirs(os.path.dirname(fro_out), exist_ok=True)
                            vtm_to_fro(p, fro_out)
                            local_surf = fro_out
                        else:
                            local_surf = p
                        break

        # If we’re cold-starting, we may need to convert
        sftp = self.ssh_client.open_sftp()
        try:
            if have_surface_stage:
                # Surface/morph will have produced {base}.fro under surfaces/n_<n>.
                pass
            else:
                if not local_surf or not os.path.exists(local_surf):
                    # verbose log of what we attempted to find
                    self._log("[HPC][volume] No prior surface/morph; could not find a surface mesh. Tried:")
                    for t in tried:
                        if t: self._log(f"    {t}")
                    raise FileNotFoundError(
                        "No prior surface/morph and no surface mesh found. "
                        "Load or provide a .fro/.vtk/.vtm, or run surface/morph first."
                    )

                src_ext = os.path.splitext(local_surf)[1].lower()
                self._log(f"[HPC][volume] Cold-start with surface mesh: {local_surf}")

                # If VTK/VTM, convert to FRO and use that instead
                if src_ext in (".vtm", ".vtk"):
                    # Convert into the local run’s surfaces/n_<n>/ folder with canonical name
                    fro_out_local = os.path.join(self.input_dir, f"{self.base_name}.fro")
                    self._log(f"[HPC][volume] Converting {src_ext} -> FRO at {fro_out_local}")
                    m0 = vtm_to_fro(local_surf, fro_out_local)  # returns a FroFile; we only need the path
                    
                    if not os.path.exists(fro_out_local):
                        try:
                            # If converter returned a FroFile-like object, write it
                            m0.write_file(fro_out_local)  # no-op if already written
                        except Exception as e:
                            raise RuntimeError(f"VTM→FRO conversion did not produce '{fro_out_local}': {e}")

                    local_surf = fro_out_local
                    src_ext = ".fro"
                    
                elif src_ext in (".case"):
                    # Convert into the local run’s surfaces/n_<n>/ folder with canonical name
                    fro_out_local = os.path.join(self.input_dir, f"{self.base_name}.fro")
                    self._log(f"[HPC][volume] Converting {src_ext} -> FRO at {fro_out_local}")
                    m0 = engeo_to_fro(local_surf, fro_out_local)  # returns a FroFile; we only need the path
                    
                    if not os.path.exists(fro_out_local):
                        try:
                            # If converter returned a FroFile-like object, write it
                            m0.write_file(fro_out_local)  # no-op if already written
                        except Exception as e:
                            raise RuntimeError(f"VTM→FRO conversion did not produce '{fro_out_local}': {e}")

                    local_surf = fro_out_local
                    src_ext = ".fro"

                # Upload the (now .fro) surface mesh to BOTH surfaces/n_<n> and volumes/n_<n>
                basename = f"{self.base_name}.fro" if src_ext != ".fro" else os.path.basename(local_surf)
                sftp.put(local_surf, posixpath.join(surf_dir, basename))
                sftp.put(local_surf, posixpath.join(vol_dir,  basename))

                # Stage BAC/BPP/CTL into volumes/n_<n>
                local_stage = os.path.join(self.output_dir, f"stage_volume_n_{self.n}")
                os.makedirs(local_stage, exist_ok=True)
                self._stage_file("bac",       local_stage)
                self._stage_file("bpp",       local_stage)
                ctl_vol_path = self._stage_file("ctl_vol", local_stage)
                ctl_vol_name = os.path.basename(ctl_vol_path)

                for fname in (f"{self.base_name}.bac", f"{self.base_name}.bpp", ctl_vol_name):
                    sftp.put(os.path.join(local_stage, fname), posixpath.join(vol_dir, fname))
                
                # Automate????
                local_dat = os.path.join(self.input_dir, f"{self.base_name}.dat")
                if os.path.exists(local_dat):
                    sftp.put(local_dat, posixpath.join(vol_dir, f"{self.base_name}.dat"))
        finally:
            sftp.close()

        # --- Batchfile: always try to pull from surfaces stage at runtime (harmless if absent)
        batch_name = f"vol_n{self.n}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["mem"] = "0"
        bf.lines.append("")
        bf.lines.append("")
        bf.lines.append("module purge")
        bf.lines.append(self.intel_module)
        bf.lines.append(self.gnu_module)
        bf.lines.append("")
        bf.lines.append(f"cd {vol_dir}")
        bf.lines.append(f"cp {surf_dir}/{self.base_name}.fro {vol_dir} || true")
        bf.lines.append(f"cp {surf_dir}/{self.base_name}.bpp {vol_dir} || true")
        bf.lines.append(f"cp {surf_dir}/{self.base_name}.bac {vol_dir} || true")
        bf.lines.append("")
        bf.lines.append(f"time  srun {self.volume_mesher} {self.base_name} &> volume_output")

        local_batch = os.path.join(self.output_dir, f"batchfile_{batch_name}")
        with open(local_batch, "w", newline="\n") as f:
            f.write(str(bf))

        dep_id = runafter or self.job_ids.get("surface") or self.job_ids.get("morph")
        jobid = self._upload_batch_and_submit(local_batch, vol_dir, batch_name, dep_jobid=dep_id)
        self.job_ids["volume"] = jobid
        self._log(f"[HPC] Volume job {jobid}")
        
        if units == "mm":
            jobid = self.conv_plt(units, runafter=jobid)
            
        return jobid
    
    def conv_plt(self, units="mm", runafter=None):
        vol_dir  = posixpath.join(self.remote_output, "volumes",  f"n_{self.n}/")
        
        batch_name = f"plt_conv_n{self.n}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["mem"] = "0"
        bf.lines.append("")
        bf.lines.append("")
        bf.lines.append("module purge")
        bf.lines.append(self.intel_module)
        bf.lines.append("")
        bf.lines.append("")
        bf.lines.append(f"/home/s.engevabj/codes/utilities/hyb_plt_converter <<INPUT1")
        bf.lines.append(f"corner")
        bf.lines.append(f"INPUT1")
        bf.lines.append("")
        bf.lines.append("mv corner.plt corner_mm.plt")
        bf.lines.append("mv corner_new.plt corner.plt")

        local_batch = os.path.join(self.output_dir, f"batchfile_{batch_name}")
        with open(local_batch, "w", newline="\n") as f:
            f.write(str(bf))

        dep_id = runafter or self.job_ids.get("volume")
        jobid = self._upload_batch_and_submit(local_batch, vol_dir, batch_name, dep_jobid=dep_id)
        self.job_ids["conv_plt"] = jobid
        self._log(f"[HPC] Volume job {jobid}")
        
        return jobid

    def prepro(self, runafter=None):
        assert self.remote_output and self.ssh_client

        pre_dir = posixpath.join(self.remote_output, "preprocessed", f"n_{self.n}/")
        vol_dir = posixpath.join(self.remote_output, "volumes", f"n_{self.n}/")
        self._mkdir_p_remote(pre_dir)

        # --- Stage locally
        stage_dir = os.path.join(self.output_dir, f"stage_prepro_n_{self.n}")
        os.makedirs(stage_dir, exist_ok=True)

        # 1) Stage BCO: prefer a user-provided file from input_dir, otherwise use _stage_file('bco', ...)
        #    (Assumes you’ve added 'bco' support inside _stage_file; see snippet below.)
        input_bco = os.path.join(self.input_dir or "", f"{self.base_name}.bco")
        if os.path.exists(input_bco):
            local_bco = os.path.join(stage_dir, f"{self.base_name}.bco")
            shutil.copyfile(input_bco, local_bco)
            self._log(f"[HPC][prepro] Using BCO from input_dir: {input_bco}")
        else:
            local_bco = self._stage_file("bco", stage_dir)  # returns stage path
            self._log(f"[HPC][prepro] Staged BCO: {local_bco}")

        # 2) Find rungen.inp (as you already do)
        from FileRW.RungenInpFile import RungenInpFile
        local_rungen = getattr(self.main_window, "rungen_local_path", None)
        if not local_rungen:
            cand1 = os.path.join(self.output_dir, "rungen.inp")
            cand2 = os.path.join(self.output_dir, "preprocessed", f"n_{self.n}", "rungen.inp")
            local_rungen = cand2 if os.path.exists(cand2) else (cand1 if os.path.exists(cand1) else None)

        # --- Uploads (single SFTP session)
        sftp = self.ssh_client.open_sftp()
        try:
            # Put BCO into PREPRO dir (not volumes)
            if local_bco and os.path.exists(local_bco):
                sftp.put(local_bco, posixpath.join(pre_dir, f"{self.base_name}.bco"))

            # Put rungen.inp into PREPRO dir if present
            self.sol_parallel_domains = 1
            if local_rungen and os.path.exists(local_rungen):
                try:
                    rg_inp = RungenInpFile.read(local_rungen)
                    pd = getattr(rg_inp, "parallel_domains", 1)
                    self.sol_parallel_domains = int(pd) if pd not in (None, "") else 1
                except Exception as e:
                    self._log(f"[HPC][prepro] Warning: could not parse rungen.inp for ntasks: {e}")
                sftp.put(local_rungen, posixpath.join(pre_dir, "rungen.inp"))
                self._log(f"[HPC][prepro] Uploaded rungen.inp")
            else:
                self._log("[HPC][prepro] No local rungen.inp found; proceeding with Gen3d defaults.")

        finally:
            sftp.close()

        # --- Batchfile
        batch_name = f"pre_n{self.n}"
        bf = Batchfile(batch_name)
        if self.sol_parallel_domains and self.sol_parallel_domains > 0:
            bf.sbatch_params["ntasks"] = str(self.sol_parallel_domains)
        bf.sbatch_params["mem"] = "0"
        bf.sbatch_params["time"] = "2-00:00"
        bf.sbatch_params.pop("nodes", None)

        bf.lines.append(self.intel_module)
        bf.lines.append(self.gnu_module)
        bf.lines.append(f"cd {pre_dir}")

        # If your prepro needs the PLT from volume, keep this copy (harmless if absent)
        bf.lines.append(f"cp {vol_dir}/{self.base_name}.plt {pre_dir} || true")

        # Run: Gen3d_jj reads rungen.inp from CWD; BCO is also in CWD
        bf.lines.append(f"time srun {self.prepro_exe} < rungen.inp &> prepro_output")

        local_batch = os.path.join(self.output_dir, f"batchfile_{batch_name}")
        with open(local_batch, "w", newline="\n") as f:
            f.write(str(bf))

        dep_id = runafter or self.job_ids.get("conv_plt") or self.job_ids.get("volume")
        jobid = self._upload_batch_and_submit(local_batch, pre_dir, batch_name, dep_jobid=dep_id)
        self.job_ids["prepro"] = jobid
        self._log(f"[HPC] Prepro job {jobid}")
        return jobid

    def solver_multi(self, conds):
        if "prepro" not in self.job_ids:
            raise RuntimeError("prepro must be submitted before solver_multi.")
        for i, cond in enumerate(conds or [], 1):
            self.solver(cond, nc=i)

    def solver(self, cond: dict = None, nc=1):
        from FileRW.CaseInpFile import CaseInpFile
        from FileRW.RungenInpFile import RungenInpFile
        assert self.remote_output and self.ssh_client
        cond = cond or {}
        tag = self._cond_tag(cond)
        tag_dir = f"cond_{nc}/"
        tag_slug = f"cond_{nc}"
        sol_dir = posixpath.join(self.remote_output, "solutions", f"n_{self.n}", tag_dir)
        pre_dir = posixpath.join(self.remote_output, "preprocessed", f"n_{self.n}/")
        self._mkdir_p_remote(sol_dir)

        # --- read parallel_domains from rungen.inp (same pattern as prepro)
        local_rungen = getattr(self.main_window, "rungen_local_path", None)
        if not local_rungen:
            cand1 = os.path.join(self.output_dir, "preprocessed", f"n_{self.n}", "rungen.inp")
            cand2 = os.path.join(self.output_dir, "rungen.inp")
            local_rungen = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)

        # --- locate user's solver .inp (UI may have set this)
        local_inp = getattr(self.main_window, "solver_input_path", None)
        if not local_inp or not os.path.exists(local_inp):
            guesses = [
                os.path.join(self.output_dir, f"{self.base_name}.inp"),
                os.path.join(self.input_dir or "", f"{self.base_name}.inp"),
                next((os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f and f.lower().endswith(".inp")), None)
            ]
            for g in guesses:
                if g and os.path.exists(g):
                    local_inp = g
                    break

        stage_dir  = os.path.join(self.output_dir, f"stage_solver_n_{self.n}_{tag_slug}")
        os.makedirs(stage_dir, exist_ok=True)
        staged_inp = os.path.join(stage_dir, f"{self.base_name}.inp")

        if local_inp and os.path.exists(local_inp):
            # 1) User-provided file exists -> copy it and patch in-place (no CaseInpFile)
            import shutil
            shutil.copyfile(local_inp, staged_inp)
            # This call must accept 'cond' and patch ivd% fields as well as processes/dataDirectory
            self._patch_solver_inp(staged_inp, self.sol_parallel_domains, sol_dir, cond)
            self._log(f"[HPC][solver] Using UI .inp; patched → {staged_inp}")
        else:
            # 2) Fallback -> render from CaseInpFile to match a template exactly
            from FileRW.CaseInpFile import CaseInpFile
            template_guess = getattr(self.main_window, "solver_template_path", None)
            if not template_guess or not os.path.exists(template_guess):
                # try common locations
                guess_list = [
                    os.path.join(self.input_dir or "", f"{self.base_name}.inp"),
                    os.path.join(self.output_dir,       f"{self.base_name}.inp"),
                    os.path.join(self.templates_dir,    f"{self.base_name}.inp"),
                    os.path.join(self.templates_dir,    "filename.inp"),  # last resort
                ]
                template_guess = next((p for p in guess_list if p and os.path.exists(p)), None)

            ci = CaseInpFile(name=self.base_name, filepath=template_guess)
            # set required params (floats where appropriate!)
            if cond.get("AoA")        is not None: ci.set_param("alpha",               float(cond["AoA"]))
            if cond.get("Mach")       is not None: ci.set_param("MachNumber",          float(cond["Mach"]))
            if cond.get("Re")         is not None: ci.set_param("ReynoldsNumber",      float(cond["Re"]))
            if cond.get("TurbModel")  is not None: ci.set_param("turbulenceModel",     int(cond["TurbModel"]))
            if cond.get("EngineFlow") is not None: ci.set_param("engineFlowType",      int(cond["EngineFlow"]))
            if cond.get("MassFlow")   is not None: ci.set_param("enginesFrontMassFlow(1)", float(cond["MassFlow"]))

            # always force these two
            ci.set_param("numberOfProcesses", int(self.sol_parallel_domains))
            ci.set_param("dataDirectory",     sol_dir)

            # render preserving the template's look
            ci.write_preserving_source(staged_inp)
            self._log(f"[HPC][solver] Rendered from template; wrote → {staged_inp}")
        
        run_file_dir = os.path.join(self.main_window.input_directory, "run.inp")
        with open(run_file_dir, "a") as f:
            f.write(f"{self.base_name}")
        
        if not staged_inp and local_inp and os.path.exists(local_inp):
            # Fall back to raw upload if structured staging failed
            sftp = self.ssh_client.open_sftp()
            try:
                self._log("[HPC][solver] Falling back to raw .inp upload (no staging edits).")
                sftp.put(local_inp, posixpath.join(sol_dir, os.path.basename(local_inp)))
                sftp.put(run_file_dir, posixpath.join(sol_dir, os.path.basename(run_file_dir)))
            finally:
                sftp.close()
        
        # --- upload staged .inp if present
        sftp = self.ssh_client.open_sftp()
        try:
            if staged_inp and os.path.exists(staged_inp):
                sftp.put(staged_inp, posixpath.join(sol_dir, os.path.basename(staged_inp)))
        finally:
            sftp.close()
        
        batch_name = f"sol_n{self.n}_{tag_slug}"
        bf = Batchfile(batch_name)
        if self.sol_parallel_domains and self.sol_parallel_domains > 0:
            bf.sbatch_params["ntasks"] = str(self.sol_parallel_domains)
        bf.sbatch_params["mem"] = "0"
        bf.sbatch_params["time"] = "3-00:00"
        bf.sbatch_params.pop("nodes")
        bf.lines.append("")
        bf.lines.append("")
        bf.lines.append(self.mpi_intel_module)
        bf.lines.append("")
        bf.lines.append(f"cd {sol_dir}")
        bf.lines.append(f"cp {pre_dir}/base.plt {sol_dir} || true")
        bf.lines.append(f"cp {pre_dir}/plotreg.reg {sol_dir} || true")
        bf.lines.append(f"ln -sf {pre_dir}/{self.base_name}.sol* {sol_dir} || true")
        bf.lines.append("")
        bf.lines.append(f"mpirun {self.solver_exe} < {self.base_name}.inp &> solver_output")
        bf.lines.append("")
        bf.lines.append(f"{self.combine_exe} <<INPUT1")
        bf.lines.append("plotreg.reg")
        bf.lines.append(f"{self.base_name}.res")
        bf.lines.append(f"{self.base_name}.unk")
        bf.lines.append("F")
        bf.lines.append("T")
        bf.lines.append("INPUT1")

        local_batch = os.path.join(self.output_dir, f"batchfile_{batch_name}")
        with open(local_batch, "w", newline="\n") as f: f.write(str(bf))
        dep_id = self.job_ids.get("prepro")
        jobid = self._upload_batch_and_submit(local_batch, sol_dir, batch_name, dep_jobid=dep_id)
        self.job_ids[f"solver_{tag_slug}"] = jobid
        self.job_ids["solver"] = jobid
        self._log(f"[HPC] Solver job {jobid} ({tag})")
        
        job_id_s2 = self._submit_convergence_guard(
            aoa_or_tag=tag_slug,
            sol_dir=sol_dir,
            stdout_name="solver_output",
            solver_job_id=jobid,
            res_threshold=-3,        # your “-3” threshold
            residual_csv=f"{self.base_name}.rsd",  # or "" if unavailable
            runafter = None,
        )
        self.job_ids[f"solver2_{tag_slug}"] = job_id_s2
        self.job_ids["solver2"] = job_id_s2
        
        return jobid

    def _cond_tag(self, cond: dict) -> str:
        if not cond: return "default"
        parts = []
        if "AoA" in cond: parts.append(f"AoA{cond['AoA']}")
        if "M"   in cond: parts.append(f"M{cond['M']}")
        if "Re"  in cond: parts.append(f"Re{cond['Re']}")
        if "T"   in cond: parts.append(f"T{cond['T']}")
        if not parts:
            parts = [f"{k}{cond[k]}" for k in sorted(cond.keys())]
        token = "_".join(parts)
        return re.sub(r"[^A-Za-z0-9_.-]", "-", token)[:60]
    
    import re

    def _patch_solver_inp(self, inp_path: str, processes: int, remote_sol_dir: str, cond: dict | None = None) -> str:
        """
        Patch a solver .inp file in-place:
        - numberOfProcesses / ivd%numberOfProcesses
        - dataDirectory / ivd%dataDirectory  (ALWAYS single-quoted)
        - ivd%alpha, ivd%MachNumber, ivd%ReynoldsNumber, ivd%turbulenceModel,
            ivd%engineFlowType, ivd%enginesFrontMassFlow(1) from 'cond' (if provided)
        """
        import re

        with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        # ---- small helpers
        def _ensure_nl(s: str) -> str:
            return s if s.endswith("\n") else s + "\n"

        # top-level k=v   (preserve spacing + case; append if missing)
        def _set_kv(src: str, key_regex: str, value: str) -> str:
            pat = re.compile(rf"(?im)^(?P<k>\s*{key_regex}\s*[:=]\s*)(?P<v>.+?)\s*$")
            if pat.search(src):
                return pat.sub(lambda m: f"{m.group('k')}{value}", src, count=1)
            # append if not present
            pretty = {
                r"(number[_\s]*of[_\s]*processes?)": "numberOfProcesses",
                r"(data[_\s]*directory)": "dataDirectory",
            }.get(key_regex, key_regex)
            src = _ensure_nl(src)
            return src + f"{pretty} = {value}\n"

        # ivd%Name = value[,]
        def _set_ivd(src: str, name: str, value: str) -> str:
            pat = re.compile(rf"(?im)^(?P<p>\s*ivd%{re.escape(name)}\s*=\s*)(?P<v>[^,/\n]+)(?P<c>,?)")
            if pat.search(src):
                return pat.sub(lambda m: f"{m.group('p')}{value}{m.group('c')}", src, count=1)
            src = _ensure_nl(src)
            return src + f"ivd%{name} = {value},\n"

        # ---- processes + dataDirectory (patch both places)
        procs_str = str(int(processes))
        # always single-quote the directory for safety
        dd_str = f"'{remote_sol_dir}'"

        # top-level
        #txt = _set_kv(txt, r"(number[_\s]*of[_\s]*Processes?)", procs_str)
        #txt = _set_kv(txt, r"(data[_\s]*Directory)", dd_str)
        # ivd-variants
        txt = _set_ivd(txt, "numberOfProcesses", procs_str)
        txt = _set_ivd(txt, "dataDirectory", dd_str)

        # ---- conditions → ivd%...
        cond = cond or {}
        def _first(*keys):
            for k in keys:
                if k in cond and cond[k] is not None:
                    return cond[k]
            return None

        aoa  = _first("AoA", "alpha")
        mach = _first("Mach", "M", "MachNumber")
        Re   = _first("Re", "Reynolds", "ReynoldsNumber")
        tm   = _first("TurbModel", "TM", "turbulenceModel")
        eft  = _first("EngineFlow", "engineFlowType")
        mf   = _first("MassFlow", "enginesFrontMassFlow", "enginesFrontMassFlow(1)")

        if aoa  is not None: txt = _set_ivd(txt, "alpha",                   f"{float(aoa):.8f}")
        if mach is not None: txt = _set_ivd(txt, "MachNumber",              f"{float(mach):.10g}")
        if Re   is not None: txt = _set_ivd(txt, "ReynoldsNumber",          f"{float(Re):.10g}")
        if tm   is not None: txt = _set_ivd(txt, "turbulenceModel",         f"{int(tm)}")
        if eft  is not None: txt = _set_ivd(txt, "engineFlowType",          f"{int(eft)}")
        if mf   is not None: txt = _set_ivd(txt, "enginesFrontMassFlow(1)", f"{float(mf):.10g}")

        with open(inp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
        return inp_path

    def postproc_multi(self, conds):
        if "solver" not in self.job_ids:
            raise RuntimeError("solver must be submitted before postproc_multi.")
        for i, cond in enumerate(conds or [], 1):
            self.postproc(cond, nc=i)

    def postproc(self, predir=None, runafter=None, cond: dict = None, nc=1):
        from ambiance import Atmosphere

        tag = self._cond_tag(cond)
        tag_dir = f"cond_{nc}/"
        tag_slug = f"cond_{nc}"
        sol_dir = posixpath.join(self.remote_output, "solutions", f"n_{self.n}", tag_dir)
        post_dir = posixpath.join(self.remote_output, "postprocessed", f"n_{self.n}", tag_dir)
        self._mkdir_p_remote(post_dir)
        cond = cond or {}
        
        altitude_m = cond.get('altitude', 10972)
        Mach = cond.get('M')
        atm = Atmosphere(altitude_m)
        
        temp_atm  = atm.temperature[0]   # in K
        press_atm = atm.pressure[0]      # in Pa
        cp_atm = 1004.5                      # in J/(kgK)
        
        sftp = self.ssh_client.open_sftp()
        
        batch_name = f"post_n{self.n}_{tag_slug}"
        bf = Batchfile(batch_name)
        bf.sbatch_params["ntasks"] = 1
        bf.sbatch_params["mem"] = "0"
        bf.sbatch_params["time"] = "01:00"
        bf.sbatch_params.pop("nodes")
        bf.lines.append("")
        bf.lines.append("")
        bf.lines.append(self.intel_module)
        bf.lines.append("")
        bf.lines.append(f"cd {post_dir}")
        bf.lines.append(f"cp {sol_dir}/{self.base_name}.unk {post_dir} || true")
        bf.lines.append(f"cp {sol_dir}/base.plt {post_dir}/{self.base_name}.plt || true")
        bf.lines.append("")
        bf.lines.append(f"{self.ensight_exe} <<INPUT1")
        bf.lines.append(f"{self.base_name}")
        bf.lines.append("T")
        bf.lines.append(f"{Mach}")
        bf.lines.append(f"{temp_atm}")
        bf.lines.append(f"{press_atm}")
        bf.lines.append(f"{cp_atm}")
        bf.lines.append("INPUT1")

        local_batch = os.path.join(self.output_dir, f"batchfile_{batch_name}")
        with open(local_batch, "w", newline="\n") as f: f.write(str(bf))
        dep_id = runafter or self.job_ids.get(f"solver2_{tag_slug}") or self.job_ids.get("solver2") or self.job_ids.get(f"solver_{tag_slug}") or self.job_ids.get("solver")
        jobid = self._upload_batch_and_submit(local_batch, post_dir, batch_name, dep_jobid=dep_id)
        self.job_ids[f"post_{tag_slug}"] = jobid
        self._log(f"[HPC] Solver job {jobid} ({tag})")
        return jobid
