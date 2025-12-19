# remoteOpt.py - FIXED FOR CLUSTER EXECUTION
# ----------------------------------------------------------------------
# This script runs ON THE CLUSTER (not your local machine)
# It uses ClusterPipelineManager instead of HPCPipelineManager
# ----------------------------------------------------------------------

import os, json, time, sys, re, subprocess
import numpy as np

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Assuming remoteOpt.py is in Scripts/

for subdir in ["", "Optimisation", "FileRW", "Remote", "MeshGeneration"]:
    path = os.path.join(project_root, subdir) if subdir else project_root
    if path not in sys.path:
        sys.path.insert(0, path)

# Import cluster-side pipeline (NO SSH)
from pipeline_cluster import ClusterPipelineManager


def _log(msg, log_path):
    """Log to both stdout and file"""
    print(msg, flush=True)
    try:
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _cond_tag(cond):
    """Create filesystem-safe tag for condition"""
    return f"AoA{cond.get('AoA',0)}_M{cond.get('Mach',1.0)}_Re{int(cond.get('Re',0))}_T{cond.get('TurbModel',0)}"


def _metrics_path(remote_root, n, cond_index: int):
    """Path to metrics file for a given test and condition index (1-based)"""
    return os.path.join(
        remote_root,
        "solutions",
        f"n_{n}",
        f"cond_{cond_index}",
        "corner.rsd",
    )

class ClusterTestManager:
    """
    Test manager that runs ON THE CLUSTER.
    Uses ClusterPipelineManager (no SSH/SFTP).
    """
    def __init__(self, remote_root, base_name, input_dir, executables, poll_s=120, morph_basis_json="", units="mm"):
        self.remote_root = os.path.abspath(remote_root)
        self.base_name = base_name
        self.input_dir = input_dir
        self.executables = executables
        self.poll_s = int(max(10, poll_s))
        self.jobs = {}
        self.morph_basis_json = morph_basis_json or ""
        self.units = units
        
        # Create logs directory
        self.log_dir = os.path.join(self.remote_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _alloc_n_index(self, gen_num, local_idx):
        """Generate unique n-index for (generation, design) pair"""
        return int(gen_num) * 1_000_000 + int(local_idx)
    
    def _start_one(self, n_index, x, conds):
        """Start pipeline for one design point"""
        print(f"[CLUSTER-TM] Starting n={n_index} with x={x}", flush=True)
        
        self.remote_output = self.remote_root
        # Build config for ClusterPipelineManager
        config = {
            "remote_output": self.remote_output,
            "base_name": self.base_name,
            "input_dir": self.input_dir,
            "modal_coeffs": list(map(float, x)),  # BO design vector
            "morph_basis_json": self.morph_basis_json,
            "cad_units": self.units,
            **self.executables,
        }

        # Use n_index as the pipeline's "gen"/n-directory
        pipe = ClusterPipelineManager(config, gen=0, n=n_index)
        
        # Submit jobs in sequence with dependencies
        try:
            morph_id = pipe.morph(n=n_index)           # <- pass n_index through
            vol_id = pipe.volume(runafter=morph_id)
            pre_id = pipe.prepro(runafter=vol_id)
            
            # Submit solver for each condition
            sol_ids = []
            for i, cond in enumerate(conds, 1):
                jid = pipe.solver(cond, nc=i)
                sol_ids.append(jid)
            
            self.jobs[n_index] = {
                "morph": morph_id,
                "volume": vol_id,
                "prepro": pre_id,
                "solvers": sol_ids
            }
            
            print(f"[CLUSTER-TM] Submitted n={n_index} â†’ jobs={self.jobs[n_index]}", flush=True)
            return sol_ids[-1]  # Return last solver job for dependency chaining
            
        except Exception as e:
            print(f"[CLUSTER-TM] ERROR starting n={n_index}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    def init_generation(self, X_list, gen_num, conds):
        """Submit all designs for a generation"""
        print(f"[CLUSTER-TM] Initializing generation {gen_num} with {len(X_list)} designs", flush=True)
        
        for i, x in enumerate(X_list):
            n_index = self._alloc_n_index(gen_num, i+1)
            self._start_one(n_index, x, conds)
    
    def evaluate_generation(self, X_list, gen_num, conds):
        """Wait for all results and parse them"""
        num_conds = len(conds)
        tags = [_cond_tag(c) for c in conds]  # still useful for logging if you want
        
        # Build list of required result files
        need = []
        for i, x in enumerate(X_list, 1):
            n_index = self._alloc_n_index(gen_num, i)
            for nc in range(1, num_conds + 1):
                path = _metrics_path(self.remote_root, n_index, nc)
                need.append((path, n_index, nc))
        
        print(f"[CLUSTER-TM] Waiting for {len(need)} result files...", flush=True)
        
        # Poll until all files exist
        unfinished = set(p for (p, _, _) in need)
        while unfinished:
            done = {p for p in list(unfinished) if os.path.exists(p)}
            unfinished -= done
            if unfinished:
                print(f"[CLUSTER-TM] Still waiting for {len(unfinished)} files...", flush=True)
                time.sleep(self.poll_s)
        
        print(f"[CLUSTER-TM] All results ready!", flush=True)
        
        def parse_one(path):
            try:
                with open(path, "r") as f:
                    lines = f.read().splitlines()
                # Find last valid line with numbers
                last = None
                for raw in reversed(lines):
                    toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                    if len(toks) >= 4:
                        last = toks
                        break
                if not last:
                    return {"CL": 0.0, "CD": 1e9, "CM": 0.0}
                # adjust column mapping if needed
                CL = float(last[1])
                CD = float(last[2])
                CM = float(last[3])
                return {"CL": CL, "CD": CD, "CM": CM}
            except Exception as e:
                print(f"[CLUSTER-TM] Error parsing {path}: {e}", flush=True)
                return {"CL": 0.0, "CD": 1e9, "CM": 0.0}
        
        # Collect results per design
        results = []
        for i, x in enumerate(X_list, 1):
            n_index = self._alloc_n_index(gen_num, i)
            per_cond = []
            for nc in range(1, num_conds + 1):
                path = _metrics_path(self.remote_root, n_index, nc)
                m = parse_one(path)
                per_cond.append(m)
            results.append(per_cond)
        
        return results


def main():
    if len(sys.argv) < 2:
        print("Usage: remoteOpt.py <run_directory>", flush=True)
        sys.exit(2)
    
    run_dir = os.path.abspath(sys.argv[1])
    log_path = os.path.join(run_dir, "remote_opt.log")
    os.makedirs(run_dir, exist_ok=True)
    
    _log(f"[REMOTE-OPT] Starting in {run_dir}", log_path)
    
    # Load configurations
    settings_path = os.path.join(run_dir, "bo_settings.json")
    objective_path = os.path.join(run_dir, "objective.json")
    
    if not os.path.exists(settings_path):
        _log(f"[ERROR] Settings file not found: {settings_path}", log_path)
        sys.exit(1)
    
    if not os.path.exists(objective_path):
        _log(f"[ERROR] Objective file not found: {objective_path}", log_path)
        sys.exit(1)
    
    with open(settings_path) as f:
        settings_json = json.load(f)
    
    with open(objective_path) as f:
        objective = json.load(f)
        
    morph_basis_json = settings_json.get("morph_basis_json", "")
    
    _log(f"[REMOTE-OPT] Loaded settings: {settings_json}", log_path)
    _log(f"[REMOTE-OPT] Loaded objective: {objective}", log_path)
    
    # Import BO components
    from Optimisation.BayesianOptimisation.optimiser import BayesianOptimiser
    from Optimisation.BayesianOptimisation.kernels import (
        RBFKernel, SquaredExponentialKernel, ExponentialKernel, 
        Mat12Kern, Mat32Kern, Mat52Kern
    )
    from Optimisation.BayesianOptimisation.acquisition_functions import EI, POI, UCB
    
    # Map string names to classes
    kern_map = {
        "RBFKernel": RBFKernel,
        "Squared Exponential Kernel": SquaredExponentialKernel,
        "Exponential Kernel": ExponentialKernel,
        "Mat12Kern": Mat12Kern,
        "Mat32Kern": Mat32Kern,
        "Mat52Kern": Mat52Kern
    }
    
    acq_map = {
        "Expected Improvement": EI,
        "Probability of Improvement": POI,
        "Upper Confidence Bound": UCB
    }
    
    # Prepare settings
    settings = dict(settings_json)
    settings["kernel"] = kern_map[settings_json["kernel"]]
    settings["acquisition_function"] = acq_map[settings_json["acquisition_function"]]
    settings["sim_dir"] = run_dir
    
    # Get conditions and weights
    conds = objective.get("conditions", [])
    weights = [c.get("Weight", 1.0) for c in conds]
    
    _log(f"[REMOTE-OPT] Conditions: {conds}", log_path)
    _log(f"[REMOTE-OPT] Weights: {weights}", log_path)
    
    # Determine remote root (parent of run_dir usually)
    # Adjust this based on your directory structure
    remote_root = os.path.dirname(run_dir)
    base_name = settings_json.get("base_name", "model")
    input_dir = settings_json.get("input_dir", os.path.join(remote_root, "orig"))
    cad_units = settings_json.get("units", "mm")
    
    # Executable paths (customize for your cluster)
    executables = {
        "parallel_domains": settings_json.get("parallel_domains", 1),
        "surface_mesher": "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Surf3D",
        "volume_mesher": "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Mesh3D",
        "prepro_exe": "/home/s.o.hassan/bin/Gen3d_jj",
        "solver_exe": "/home/s.o.hassan/bin/UnsMgnsg3d",
        "combine_exe": "/home/s.engevabj/codes/utilities/makeplot2",
        "ensight_exe": "/home/s.engevabj/codes/utilities/engen_tet",
        "splitplot_exe": "/home/s.engevabj/codes/utilities/splitplot2",
        "makeplot_exe": "/home/s.engevabj/codes/utilities/makeplot2",
        "intel_module": "module load compiler/intel/2020/0",
        "gnu_module": "module load compiler/gnu/12/1.0",
        "mpi_intel_module": "module load mpi/intel/2020/0",
    }
    
    # Create test manager (uses ClusterPipelineManager internally)
    tm = ClusterTestManager(
        remote_root=remote_root,
        base_name=base_name,
        input_dir=input_dir,
        executables=executables,
        poll_s=settings_json.get("poll_interval", 120),
        morph_basis_json=morph_basis_json,
        units = cad_units,
    )
    
    # Define init and eval functions for BO
    def init_func(X_list, gen_num):
        _log(f"[REMOTE-OPT] Initializing generation {gen_num}: {len(X_list)} designs", log_path)
        tm.init_generation(X_list, gen_num, conds)
    
    def eval_func(X_list, gen_num):
        _log(f"[REMOTE-OPT] Evaluating generation {gen_num}", log_path)
        per_design = tm.evaluate_generation(X_list, gen_num, conds)
        
        # Reduce per-condition metrics to scalar objective
        Y = []
        for metrics in per_design:
            y = 0.0
            for w, m in zip(weights, metrics):
                y += float(w) * float(m.get("CD", 1e9))
            Y.append(y)
        
        _log(f"[REMOTE-OPT] Generation {gen_num} objectives: {Y}", log_path)
        return np.array(Y, dtype=float)
    
    # Run Bayesian Optimization
    _log("[REMOTE-OPT] Starting Bayesian Optimization...", log_path)
    bo = BayesianOptimiser(settings, eval_func=eval_func, init_func=init_func)
    X_best, Y_best = bo.optimise(cont=True)
    
    _log(f"[REMOTE-OPT] OPTIMIZATION COMPLETE!", log_path)
    _log(f"[REMOTE-OPT] Best X = {X_best}", log_path)
    _log(f"[REMOTE-OPT] Best Y = {Y_best}", log_path)
    
    # Save final results
    results_file = os.path.join(run_dir, "optimization_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "X_best": X_best.tolist() if hasattr(X_best, 'tolist') else X_best,
            "Y_best": float(Y_best),
            "settings": settings_json,
            "objective": objective
        }, f, indent=2)
    
    _log(f"[REMOTE-OPT] Results saved to {results_file}", log_path)


if __name__ == "__main__":
    main()