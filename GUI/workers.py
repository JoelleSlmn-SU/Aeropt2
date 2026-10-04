from PyQt5.QtCore import QObject, pyqtSignal
import os
import shutil
import json

class MorphWorker(QObject):
    finished = pyqtSignal(object)   # morphed mesh (or None for HPC path)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, viewer, debug=True):
        super().__init__()
        self.viewer = viewer
        self.debug = debug

    def run(self):
        try:
            main = self.viewer.main_window

            # sanity
            if getattr(main, "run_mode", "LOCAL") != "HPC":
                raise RuntimeError("This Morph button mode is intended for HPC runs only.")

            # Use your pipeline manager that talks to the cluster from the GUI machine
            from Remote.pipeline_remote import HPCPipelineManager

            submitted = []
            for i in range(self.n_morphs):
                pipe = HPCPipelineManager(main_window=main, n=0)  # n here is “gen” in some places; see note below

                self.log.emit(f"[MORPH] Submitting morph case n={i} ...")
                morph_job = pipe.morph(n=i)

                vol_job = None
                if self.do_volume:
                    self.log.emit(f"[VOLUME] Submitting volume for n={i} (after morph={morph_job}) ...")
                    vol_job = pipe.volume(runafter=morph_job)

                submitted.append({"n": i, "morph": morph_job, "volume": vol_job})

            self.log.emit(f"[DONE] Submitted {len(submitted)} morph(s).")
            self.finished.emit(submitted)

        except Exception as e:
            self.failed.emit(str(e))


class SurfaceWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pipeline, debug=True):
        super().__init__()
        self.pipeline = pipeline
        self.debug = debug

    def run(self):
        result = None
        try:
            self.log.emit("[INFO] Starting surface mesh...")
            if getattr(self.pipeline.geo_viewer.main_window, "run_mode", "Local") == "HPC":
                self.pipeline.surface()
            else:
                self.pipeline.surface()
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
            
   
class VolumeWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pipeline, debug=True):
        super().__init__()
        self.pipeline = pipeline
        self.debug = debug

    def run(self):
        result = None
        try:
            self.log.emit("[INFO] Starting surface mesh...")

            # Handle both local (viewer_geom) and HPC (geo_viewer)
            viewer_geom = getattr(self.pipeline, "viewer_geom", None)
            if viewer_geom is None:
                viewer_geom = getattr(self.pipeline, "geo_viewer", None)

            run_mode = getattr(getattr(viewer_geom, "main_window", None), "run_mode", "Local")
            # At the moment you call volume() in both branches anyway
            self.pipeline.volume()

            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
            

class SimulationWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, main_window, debug=True):
        super().__init__()
        self.w = main_window
        self.debug = debug

    def _is_geometry(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in (".stp", ".step", ".igs", ".iges")

    def _is_surface_mesh(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in (".vtk", ".vtm", ".fro")

    def run(self):
        try:
            self.log.emit("[SIM] Submitting jobs via pipeline…")
            pipeline = getattr(self.w, "pipeline", None)
            assert pipeline is not None, "PipelineManager/HPCPipelineManager not initialised."
            inpath = getattr(self.w, "input_file_path", None)
            assert inpath and os.path.exists(inpath), "No input file loaded."

            # Geometry → Surface → Volume → Prepro → Solver
            if self._is_geometry(inpath):
                self.log.emit("[SIM] Detected geometry input.")
                surf_id = pipeline.surface()
                self.log.emit(f"[SIM] Surface submitted: {surf_id}")
                vol_id  = pipeline.volume(runafter=surf_id)
                self.log.emit(f"[SIM] Volume submitted:  {vol_id}")
                pre_id  = pipeline.prepro(runafter=vol_id)
                self.log.emit(f"[SIM] Prepro submitted:  {pre_id}")

            # Surface mesh → Volume → Prepro → Solver
            elif self._is_surface_mesh(inpath):
                self.log.emit("[SIM] Detected surface-mesh input.")
                vol_id  = pipeline.volume()
                self.log.emit(f"[SIM] Volume submitted:  {vol_id}")
                pre_id  = pipeline.prepro(runafter=vol_id)
                self.log.emit(f"[SIM] Prepro submitted:  {pre_id}")

            else:
                raise RuntimeError("Input must be geometry (.stp/.step/.igs) or surface mesh (.vtk/.vtm/.fro).")

            # Solver for each flow condition
            conds =  getattr(self, "conds", [])
            if hasattr(pipeline, "solver_multi") and hasattr(pipeline, "postproc_multi"):
                pipeline.solver_multi(conds)
                pipeline.postproc_multi(conds)
            else:
                for i, cond in enumerate(conds, 1):
                    jid_solv = pipeline.solver(cond, nc=i)
                    jid_post = pipeline.postproc(cond, nc=i)
                    self.log.emit(f"[SIM] Solver submitted: {jid_solv}")

            self.log.emit("[SIM] All jobs submitted.")
            self.finished.emit()

        except Exception as e:
            self.failed.emit(str(e))
            
            
class Simulation2DWorker(QObject):
    """Runs the local FLITE 2D chain (PrePro.exe, then Solver.exe per flow condition)
    for one design off the GUI thread. finished(list of metrics dicts, one per condition)."""
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, output_dir, base, prepro_options, control_text, conds, boundary_flags,
                 time_limit_s, n=0, solver_dir=None, prepro_cmd=None, solver_cmd=None):
        super().__init__()
        self.output_dir, self.base, self.n = output_dir, base, int(n)
        self.opts, self.control_text = prepro_options, control_text
        self.conds = list(conds) if conds else [{}]
        self.boundary_flags, self.time_limit_s = boundary_flags, time_limit_s
        self.solver_dir, self.prepro_cmd, self.solver_cmd = solver_dir, prepro_cmd, solver_cmd
        self._stop = False
        self._discard = False
        self.stopped = False              # True once a stop request has ended the run

    def request_stop(self, discard=False):
        """Called from the GUI thread; only sets flags (the worker thread polls them twice
        a second). discard=True: the window is closing, so skip the pictures as well."""
        self._discard = self._discard or bool(discard)
        self._stop = True

    def run(self):
        from Local.flite2d_local import Flite2DCase, RunStopped
        try:
            case = Flite2DCase(self.output_dir, self.base, n=self.n, solver_dir=self.solver_dir,
                               log=self._log, prepro_cmd=self.prepro_cmd, solver_cmd=self.solver_cmd)
            self._case = case
            results = case.run(self.opts, self.control_text, self.conds, self.boundary_flags,
                               solver_time_limit_s=self.time_limit_s, should_stop=self._should_stop)
            self.stopped = self._stop
            self.finished.emit(results)
        except RunStopped:
            self.stopped = True
            self.finished.emit([])
        except Exception as e:
            self.failed.emit(str(e))

    def _should_stop(self):
        if self._discard and getattr(self, "_case", None) is not None:
            self._case.make_plots = False
        return self._stop

    def _log(self, msg):
        if not self._discard:             # the window is closing: nothing left to log to
            self.log.emit(msg)


class OptimisationWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, optimiser):
        super().__init__()
        self.optimiser = optimiser

    def run(self):
        try:
            self.log.emit("[OPT] Starting Bayesian optimisation…")
            X_best, Y_best = self.optimiser.optimise(cont=True)
            self.log.emit(f"[OPT] Finished. Best X = {X_best}, Objective = {Y_best}")
            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))