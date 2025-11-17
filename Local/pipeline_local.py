# pipeline_local.py
# ----------------------------------------------------------------------
# Local pipeline manager
# - Probes input-folder BAC/BPP/CTL/SURFACE in __init__
# - morph(): writes <base>.fro into surfaces/n_<n> (calls morph_exe)
# - surface(): stages CAD+BAC+BPP+CTL and runs surface mesher
# - volume(): can run after surface OR cold-start from input surface mesh
# - prepro(), solver(): simple local commands
# ----------------------------------------------------------------------

import os, shutil, subprocess, re

class PipelineManager:
    def __init__(self, main_window, n=0):
        self.main_window = main_window
        self.n = n

        # executables
        self.morph_exe   = getattr(self, "morph_exe",   "Morph.exe")
        self.surface_exe = getattr(self, "surface_exe", "Surface.exe")
        self.volume_exe  = getattr(self, "volume_exe",  "Volume.exe")
        self.prepro_exe  = getattr(self, "prepro_exe",  "PrePro.exe")
        self.solver_exe  = getattr(self, "solver_exe",  "Solver.exe")

        # dirs
        self.output_dir = getattr(self.main_window, "output_directory", "") or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        self.surf_dir = os.path.join(self.output_dir, "surfaces",     f"n_{self.n}")
        self.vol_dir  = os.path.join(self.output_dir, "volumes",      f"n_{self.n}")
        self.pre_dir  = os.path.join(self.output_dir, "preprocessed", f"n_{self.n}")
        self.sol_dir  = os.path.join(self.output_dir, "solutions",    f"n_{self.n}")

        # input/base
        self.input_path = getattr(self.main_window, "input_file_path", "")
        self.input_dir  = getattr(self.main_window, "input_directory", "") or os.path.dirname(self.input_path)
        self.base_name  = os.path.splitext(os.path.basename(self.input_path))[0] if self.input_path else "model"
        self.templates_dir = os.path.join(os.getcwd(), "templates")

        # probe input-folder assets
        def _probe(p): return p if (p and os.path.exists(p)) else None
        self.bac_src = _probe(os.path.join(self.input_dir, f"{self.base_name}.bac"))
        self.bpp_src = _probe(os.path.join(self.input_dir, f"{self.base_name}.bpp"))
        self.ctl_src = _probe(os.path.join(self.input_dir, "Surf3D_v25.ctl"))
        self.fro_src = _probe(os.path.join(self.input_dir, f"{self.base_name}.fro"))
        self.vtk_src = _probe(os.path.join(self.input_dir, f"{self.base_name}.vtk"))
        self.vtm_src = _probe(os.path.join(self.input_dir, f"{self.base_name}.vtm"))

        # logger
        self.viewer_geom = getattr(self.main_window, "mesh_viewer", None)
        self._log(f"[LOCAL] Pipeline init: base='{self.base_name}' n={self.n}")

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        if getattr(self.viewer_geom, "logger", None):
            try: self.viewer_geom.logger.log(msg); return
            except: pass
        if getattr(self.main_window, "logger", None):
            try: self.main_window.logger.log(msg); return
            except: pass
        print(msg)

    def _stage_file(self, kind: str, stage_dir: str) -> str:
        os.makedirs(stage_dir, exist_ok=True)
        if kind == "ctl":
            target = os.path.join(stage_dir, "Surf3D_v25.ctl")
            out_candidate = os.path.join(self.output_dir, "Surf3D_v25.ctl")
            tpl_candidate = os.path.join(self.templates_dir, "Surf3D_v25.ctl")
            src_attr = "ctl_src"
        elif kind == "bac":
            target = os.path.join(stage_dir, f"{self.base_name}.bac")
            out_candidate = os.path.join(self.output_dir, f"{self.base_name}.bac")
            tpl_candidate = os.path.join(self.templates_dir, "filename.bac")
            src_attr = "bac_src"
        elif kind == "bpp":
            target = os.path.join(stage_dir, f"{self.base_name}.bpp")
            out_candidate = os.path.join(self.output_dir, f"{self.base_name}.bpp")
            tpl_candidate = os.path.join(self.templates_dir, "filename.bpp")
            src_attr = "bpp_src"
        else:
            raise ValueError(f"unknown kind={kind}")

        src = getattr(self, src_attr, None)
        if src and os.path.exists(src): shutil.copyfile(src, target); return target
        if os.path.exists(out_candidate): shutil.copyfile(out_candidate, target); return target
        if os.path.exists(tpl_candidate): shutil.copyfile(tpl_candidate, target); return target

        if kind == "bac":
            from FileRW.BacFile import BacFile
            fn = getattr(BacFile, "defaultCRMFineMesh", None)
            if callable(fn):
                with open(target, "w", newline="\n") as f: f.write(str(fn(name=self.base_name)))
                return target
        if kind == "bpp":
            from FileRW.BppFile import BppFile
            with open(target, "w", newline="\n") as f: f.write(str(BppFile(self.base_name)))
            return target
        if kind == "ctl":
            with open(target, "w", newline="\n") as f: f.write("# Surf control\n")
            return target
        return target

    # --------------- stages -------------------
    def morph(self):
        """Run local morph to create <base>.fro in surfaces/n_<n>."""
        self._log(f"[LOCAL] Morph (n={self.n})…")
        os.makedirs(self.surf_dir, exist_ok=True)

        # Validate executable
        exe = self.morph_exe
        if not shutil.which(exe):
            raise FileNotFoundError(f"Morph executable not found on PATH: '{exe}'")

        # Run and check output
        cmd = [exe, os.path.join(self.surf_dir, self.base_name)]
        self._log(f"[LOCAL] Exec: {cmd}")
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Morph failed to start (bad path?): {e}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Morph exited with non-zero status: {e}")

        out_fro = os.path.join(self.surf_dir, f"{self.base_name}.fro")
        if not os.path.exists(out_fro):
            raise FileNotFoundError(f"Morph did not create {out_fro}")
        self._log("[LOCAL] Morph complete")

    def surface(self):
        """Run surface mesher to produce <base>.fro."""
        self._log(f"[LOCAL] Surface (n={self.n})…")
        os.makedirs(self.surf_dir, exist_ok=True)
        cad_ext = os.path.splitext(self.input_path)[1].lower()
        cad_target = os.path.join(self.surf_dir, f"{self.base_name}{cad_ext}")
        shutil.copyfile(self.input_path, cad_target)
        self._stage_file("bac", self.surf_dir)
        self._stage_file("bpp", self.surf_dir)
        self._stage_file("ctl", self.surf_dir)
        cmd = [self.surface_exe, os.path.join(self.surf_dir, self.base_name)]
        self._log(f"[LOCAL] Exec: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        out_fro = os.path.join(self.surf_dir, f"{self.base_name}.fro")
        if not os.path.exists(out_fro):
            raise FileNotFoundError(f"Surface mesher did not create {out_fro}")
        self._log("[LOCAL] Surface complete")

    def volume(self):
        """
        Run the volume mesher locally.
        - If a surface/morph stage ran, copy data from surfaces/n_<n>.
        - Otherwise, cold-start using the loaded surface mesh (.fro/.vtk/.vtm)
        or one found nearby.
        - Always place the surface mesh into BOTH surfaces/n_<n> and volumes/n_<n>.
        - Always stage BAC/BPP/CTL into volumes/n_<n> when cold-starting.
        """
        self._log(f"[LOCAL] Volume (n={self.n})…")
        os.makedirs(self.vol_dir,  exist_ok=True)
        os.makedirs(self.surf_dir, exist_ok=True)

        have_surface_stage = os.path.exists(os.path.join(self.surf_dir, f"{self.base_name}.fro"))

        tried = []
        def _add_try(p):
            if p and p not in tried:
                tried.append(p)

        # Prefer the LOADED file if it's a surface mesh
        loaded = getattr(self, "input_path", None)
        if loaded and os.path.exists(loaded) and os.path.splitext(loaded)[1].lower() in (".fro", ".vtk", ".vtm"):
            local_surf = loaded
        else:
            local_surf = None
            for p in (self.fro_src, self.vtk_src, self.vtm_src):
                _add_try(p)
                if p and os.path.exists(p):
                    local_surf = p
                    break
            if not local_surf:
                for root in (self.input_dir, self.output_dir, os.path.join(self.output_dir, "surfaces", f"n_{self.n}")):
                    for ext in (".fro", ".vtk", ".vtm"):
                        cand = os.path.join(root, f"{self.base_name}{ext}")
                        _add_try(cand)
                        if os.path.exists(cand):
                            local_surf = cand
                            break
                    if local_surf: break

        if have_surface_stage:
            # Copy forward from surface output
            for name in (f"{self.base_name}.fro", f"{self.base_name}.bpp", f"{self.base_name}.bac"):
                src = os.path.join(self.surf_dir, name)
                dst = os.path.join(self.vol_dir,  name)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
        else:
            if not local_surf or not os.path.exists(local_surf):
                self._log("[LOCAL][volume] No prior surface/morph; could not find a surface mesh. Tried:")
                for t in tried:
                    if t: self._log(f"    {t}")
                raise FileNotFoundError("No prior surface/morph and no surface mesh found. Load or provide a .fro/.vtk/.vtm, or run surface/morph first.")

            basename = os.path.basename(local_surf)
            self._log(f"[LOCAL][volume] Cold-start with surface mesh: {local_surf}")

            # Copy to BOTH surfaces/n_<n> and volumes/n_<n>
            shutil.copyfile(local_surf, os.path.join(self.surf_dir, basename))
            shutil.copyfile(local_surf, os.path.join(self.vol_dir,  basename))

            # Stage BAC/BPP/CTL into volumes/n_<n>
            self._stage_file("bac", self.vol_dir)
            self._stage_file("bpp", self.vol_dir)
            self._stage_file("ctl", self.vol_dir)

        # Run
        cmd = [self.volume_exe, os.path.join(self.vol_dir, self.base_name)]
        self._log(f"[LOCAL] Exec: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        self._log("[LOCAL] Volume complete")

    def prepro(self):
        self._log(f"[LOCAL] PrePro (n={self.n})…")
        os.makedirs(self.pre_dir, exist_ok=True)
        cmd = [self.prepro_exe, os.path.join(self.pre_dir, self.base_name)]
        self._log(f"[LOCAL] Exec: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        self._log("[LOCAL] PrePro complete")

    def solver(self, cond: dict = None):
        self._log(f"[LOCAL] Solver (n={self.n})…")
        tag = self._cond_tag(cond or {})
        sol_dir = os.path.join(self.sol_dir, tag)
        os.makedirs(sol_dir, exist_ok=True)
        cmd = [self.solver_exe, "--case", self.base_name, "--tag", tag]
        self._log(f"[LOCAL] Exec: {' '.join(cmd)} in {sol_dir}")
        subprocess.run(cmd, check=True, cwd=sol_dir)
        self._log(f"[LOCAL] Solver complete ({tag})")

    def _cond_tag(self, cond: dict) -> str:
        if not cond: return "default"
        parts = []
        if "AoA" in cond: parts.append(f"AoA{cond['AoA']}")
        if "M"   in cond: parts.append(f"M{cond['M']}")
        if "Re"  in cond: parts.append(f"Re{cond['Re']}")
        if "T"   in cond: parts.append(f"T{cond['T']}")
        if not parts: parts = [f"{k}{cond[k]}" for k in sorted(cond.keys())]
        return re.sub(r"[^A-Za-z0-9_.-]", "-", "_".join(parts))[:60]
