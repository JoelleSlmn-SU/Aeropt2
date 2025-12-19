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
        result = None
        try:
            self.log.emit("[INFO] Starting mesh deformation...")
            # Resolve the pipeline robustly
            pipeline = getattr(self.viewer, "pipeline", None)
            if pipeline is None:
                pipeline = getattr(self.viewer.main_window, "pipeline", None)
            if pipeline is None:
                raise RuntimeError("HPCPipelineManager not initialised (viewer.pipeline is missing).")
            if getattr(self.viewer.main_window, "run_mode", "Local") == "HPC":
                pipeline.morph()
                pipeline.volume()
            else:
                self.log.emit("[INFO] Running morph via Local PipelineManager...")
                pipeline.morph()
                 
            self.finished.emit(result)
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