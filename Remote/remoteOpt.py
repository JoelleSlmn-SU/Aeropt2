# remoteOpt.py - FIXED FOR CLUSTER EXECUTION
# This script runs ON THE CLUSTER (not your local machine)
# It uses ClusterPipelineManager instead of HPCPipelineManager

import os, json, time, sys, re, subprocess
import numpy as np

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  

for subdir in ["", "Optimisation", "FileRW", "Remote", "MeshGeneration"]:
    path = os.path.join(project_root, subdir) if subdir else project_root
    if path not in sys.path:
        sys.path.insert(0, path)


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


def _metrics_path(remote_root, n, gen, cond_index: int, base_name):
    """Path to metrics file for a given test and condition index (1-based)"""
    return os.path.join(
        remote_root,
        "solutions",
        f"n_{gen}",
        f"cond_{cond_index}",
        f"{n}",
        f"{base_name}_{n}.rsd",
    )
    
def _strip_outer_minmax(expr_raw: str):
    """Return (sense, inner_expr). sense is min or max."""
    expr_raw = (expr_raw or "").strip()
    m = re.match(r"^\s*(min|max)\s*\(\s*(.*)\s*\)\s*$", expr_raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return "min", expr_raw


def _build_objective_callable(objective_dict: dict):
    """
    Build a safe scalar objective evaluator.

    New format:
      objective_dict["expression"] may reference any symbols produced by:
        - .rsd parser: CL, CD, CM, CL_over_CD
        - monitor parser: objective symbols such as PR_s5, DC60_s5, CD_s2_3_4

    Backwards compatible with old objective_type values: Drag, Lift, Lift-to-Drag.
    Returned objective is always in minimisation form for BO.
    """
    obj_type = (objective_dict.get("objective_type", "") or "").strip()
    expr_raw = (objective_dict.get("expression", "") or "").strip()

    if not expr_raw or expr_raw.lower() == "drag" or obj_type.lower() == "drag":
        expr_raw = "CD"
    elif expr_raw.lower() == "lift" or obj_type.lower() == "lift":
        expr_raw = "-CL"
    elif expr_raw.lower() in ("lift-to-drag", "lift to drag") or obj_type.lower() in ("lift-to-drag", "lift to drag"):
        expr_raw = "-(CL/CD)"

    sense, inner = _strip_outer_minmax(expr_raw)
    # expression symbols cannot contain '/', so also expose CL_over_CD
    inner = inner.replace("CL/CD", "CL_over_CD")

    allowed_funcs = {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": lambda x: float(np.sqrt(x)),
        "log": lambda x: float(np.log(x)),
        "exp": lambda x: float(np.exp(x)),
    }
    safe_globals = {"__builtins__": {}}
    safe_globals.update(allowed_funcs)

    def obj_func(mdict: dict) -> float:
        try:
            local_vars = {}
            for k, v in (mdict or {}).items():
                if re.match(r"^[A-Za-z_]\w*$", str(k)):
                    try:
                        local_vars[str(k)] = float(v)
                    except Exception:
                        pass
            local_vars.setdefault("CL", 0.0)
            local_vars.setdefault("CD", 1e9)
            local_vars.setdefault("CM", 0.0)
            if "CL_over_CD" not in local_vars:
                local_vars["CL_over_CD"] = local_vars["CL"] / max(local_vars["CD"], 1e-30)
            val = float(eval(inner, safe_globals, local_vars))
            return -val if sense == "max" else val
        except Exception as e:
            print(f"[OBJECTIVE][ERROR] Could not evaluate '{inner}' with metrics={mdict}: {e}", flush=True)
            return 1e9

    pretty = f"-{inner}" if sense == "max" else inner
    return obj_func, pretty


def _reduce_values(values, reduction="last", default=1e9, window_frac=0.30, min_window=5, osc_rel_tol=0.02, unstable_policy="last"):
    vals = []
    for v in values:
        try:
            fv = float(v)
            if np.isfinite(fv):
                vals.append(fv)
        except Exception:
            pass
    if not vals:
        return float(default)
    reduction = str(reduction or "last").lower()
    if reduction == "time_average":
        n = len(vals)
        w = max(int(np.ceil(window_frac * n)), min_window)
        w = min(w, n)

        tail = np.asarray(vals[-w:], dtype=float)
        mean = float(np.mean(tail))

        amp = float(np.max(tail) - np.min(tail))
        scale = max(abs(mean), 1e-12)
        rel_amp = amp / scale

        if rel_amp <= osc_rel_tol:
            return mean

        if unstable_policy == "penalty":
            return float(default)

        return float(vals[-1])
    return float(vals[-1])


def _metric_aliases(metric: str):
    m = str(metric or "").strip().lower()
    aliases = {
        "pressure_recovery": ["pressure_recovery", "pr", "p0_recovery"],
        "distortion": ["distortion", "dc60", "DC60"],
        "drag": ["drag", "CD", "cd", "duct_drag"],
        "lift": ["lift", "CL", "cl"],
        "moment": ["moment", "CM", "cm"],
        "CD": ["CD", "cd", "drag", "duct_drag"],
        "CL": ["CL", "cl", "lift"],
        "CM": ["CM", "cm", "moment"],
    }
    return aliases.get(metric, aliases.get(m, [metric, m]))

class ClusterTestManager:
    """
    Test manager that runs ON THE CLUSTER.
    Uses ClusterPipelineManager (no SSH/SFTP).
    """
    def __init__(self, remote_root, base_name, input_dir, executables, poll_s=120, morph_basis_json="", units="mm", parallel=80, monitor_config_json="", previous_solution=None):
        self.remote_root = os.path.abspath(remote_root)
        self.base_name = base_name
        self.input_dir = input_dir
        self.executables = executables
        self.poll_s = int(max(10, poll_s))
        self.jobs = {}
        self.morph_basis_json = morph_basis_json or ""
        self.units = units
        self.parallel = parallel
        self.monitor_config_json = monitor_config_json or ""
        self.previous_solution = previous_solution or {}
        
        # Create logs directory
        self.log_dir = os.path.join(self.remote_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _alloc_n_index(self, gen_num, local_idx):
        """Generate unique n-index for (generation, design) pair"""
        return int(local_idx)
    
    def _start_one(self, gen_num, n_index, x, conds):
        """Start pipeline for one design point"""
        print(f"[CLUSTER-TM] Starting gen={gen_num} n={n_index} with x={x}", flush=True)

        self.remote_output = self.remote_root

        config = {
            "remote_output": self.remote_output,
            "base_name": self.base_name,
            "input_dir": self.input_dir,
            "modal_coeffs": list(map(float, x)),
            "morph_basis_json": self.morph_basis_json,
            "cad_units": self.units,
            "parallel_processes": self.parallel,
            "monitor_config_json": self.monitor_config_json,
            "previous_solution": self.previous_solution,
            **self.executables,
        }

        # gen must be the BO generation number
        pipe = ClusterPipelineManager(config, gen=int(gen_num), n=int(n_index))

        try:
            morph_id = pipe.morph(n=n_index)
            vol_id   = pipe.volume(runafter=morph_id)
            pre_id   = pipe.prepro(runafter=vol_id)

            sol_ids = []
            for i, cond in enumerate(conds, 1):
                jid = pipe.solver(cond, nc=i)
                sol_ids.append(jid)

            self.jobs[n_index] = {
                "gen": int(gen_num),
                "morph": morph_id,
                "volume": vol_id,
                "prepro": pre_id,
                "solvers": sol_ids,
            }

            print(f"[CLUSTER-TM] Submitted gen={gen_num} n={n_index} -> jobs={self.jobs[n_index]}", flush=True)
            return sol_ids[-1] if sol_ids else None

        except Exception as e:
            print(f"[CLUSTER-TM] ERROR starting gen={gen_num} n={n_index}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    def init_generation(self, X_list, gen_num, conds):
        """Submit all designs for a generation"""
        print(f"[CLUSTER-TM] Initializing generation {gen_num} with {len(X_list)} designs", flush=True)

        for i, x in enumerate(X_list):
            n_index = self._alloc_n_index(gen_num, i + 1)
            self._start_one(gen_num, n_index, x, conds)
    
    def evaluate_generation(self, X_list, gen_num, conds):
        """Wait for solver completion markers, then parse .rsd plus monitor CSVs."""
        num_conds = len(conds)

        def _sol_dir(n_index: int, nc: int) -> str:
            return os.path.join(self.remote_root, "solutions", f"n_{gen_num}", f"cond_{nc}", f"{n_index}")

        def _done_path(n_index: int, nc: int) -> str:
            return os.path.join(_sol_dir(n_index, nc), "SOLVER_DONE")

        def _rsd_path(n_index: int, nc: int) -> str:
            return os.path.join(_sol_dir(n_index, nc), f"{self.base_name}_{n_index}.rsd")

        need_done = []
        for i, _x in enumerate(X_list, 1):
            n_index = self._alloc_n_index(gen_num, i)
            for nc in range(1, num_conds + 1):
                need_done.append((_done_path(n_index, nc), n_index, nc))

        print(f"[CLUSTER-TM] Waiting for {len(need_done)} SOLVER_DONE markers.", flush=True)
        unfinished = set(p for (p, _, _) in need_done)
        while unfinished:
            done_now = {p for p in list(unfinished) if os.path.exists(p)}
            unfinished -= done_now
            if unfinished:
                example = next(iter(unfinished))
                print(f"[CLUSTER-TM] Still waiting for {len(unfinished)} markers... e.g. {example}", flush=True)
                time.sleep(self.poll_s)

        print("[CLUSTER-TM] All SOLVER_DONE markers present. Parsing objective metrics.", flush=True)

        def parse_rsd(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
                last = None
                for raw in reversed(lines):
                    toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                    if len(toks) >= 4:
                        last = toks
                        break
                if not last:
                    return {"CL": 0.0, "CD": 1e9, "CM": 0.0, "CL_over_CD": 0.0}
                # Existing convention in your code: CL=tokens[2], CD=tokens[3], CM=tokens[4] if present.
                CL = float(last[2]) if len(last) > 2 else 0.0
                CD = float(last[3]) if len(last) > 3 else 1e9
                CM = float(last[4]) if len(last) > 4 else 0.0
                return {"CL": CL, "CD": CD, "CM": CM, "CL_over_CD": CL / max(CD, 1e-30)}
            except Exception as e:
                print(f"[CLUSTER-TM] Error parsing {path}: {e}", flush=True)
                return {"CL": 0.0, "CD": 1e9, "CM": 0.0, "CL_over_CD": 0.0}

        def read_monitor_rows(sol_dir):
            import csv
            candidates = [
                os.path.join(sol_dir, "Monitors", "monitors.csv"),
                os.path.join(sol_dir, "Monitors", "pressure_recovery.csv"),
            ]
            rows = []
            for path in candidates:
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                        sample = f.read(4096)
                        f.seek(0)
                        if "," in sample and any(h in sample.lower() for h in ["iter", "pressure", "drag", "distortion", "dc60"]):
                            rdr = csv.DictReader(f)
                            rows.extend([{k: v for k, v in row.items()} for row in rdr])
                        else:
                            # fallback for simple CSV with no reliable header
                            for line in f:
                                toks = [x.strip() for x in line.split(",")]
                                if len(toks) >= 2:
                                    rows.append({"pressure_recovery": toks[-1]})
                except Exception as e:
                    print(f"[CLUSTER-TM][WARN] Failed reading monitor csv {path}: {e}", flush=True)
            return rows

        def parse_monitors(sol_dir, base_metrics):
            metrics = dict(base_metrics)
            rows = read_monitor_rows(sol_dir)
            if not rows:
                return metrics

            # Add last numeric value for every column using both original and sanitised names.
            by_col = {}
            for row in rows:
                for k, v in row.items():
                    if k is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    by_col.setdefault(str(k).strip(), []).append(fv)
            for k, vals in by_col.items():
                safe = re.sub(r"\W+", "_", k).strip("_")
                metrics[safe] = _reduce_values(vals, "last", default=0.0)

            # Objective terms define the symbols we actually need.
            for term in getattr(self, "objective_terms", []):
                if str(term.get("source", "")).lower() != "monitor":
                    continue
                symbol = str(term.get("symbol", "")).strip()
                metric = str(term.get("metric", "")).strip()
                reduction = term.get("reduction", "last")
                vals = []
                # Prefer exact objective_symbol/name matching if paraview_cluster writes it.
                for row in rows:
                    row_name = str(row.get("name", row.get("monitor", row.get("objective_symbol", "")))).strip()
                    if row_name and row_name == symbol:
                        for alias in _metric_aliases(metric) + ["value", symbol]:
                            if alias in row:
                                vals.append(row.get(alias))
                    else:
                        for alias in _metric_aliases(metric) + [symbol]:
                            if alias in row:
                                vals.append(row.get(alias))
                if symbol:
                    metrics[symbol] = _reduce_values(vals, reduction, default=metrics.get(symbol, 1e9))
            return metrics

        results = []
        for i, _x in enumerate(X_list, 1):
            n_index = self._alloc_n_index(gen_num, i)
            per_cond = []
            for nc in range(1, num_conds + 1):
                sol_dir = _sol_dir(n_index, nc)
                m = parse_rsd(_rsd_path(n_index, nc))
                m = parse_monitors(sol_dir, m)
                per_cond.append(m)
                print(f"[CLUSTER-TM] gen={gen_num} n={n_index} cond={nc} metrics={m}", flush=True)
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
    monitor_config_json = settings_json.get("monitor_config_json", "")
    
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
    
    # Build objective function from GUI config (Drag/Lift/Lift-to-Drag/Custom)
    obj_func, obj_expr = _build_objective_callable(objective)
    objective_terms = objective.get("terms", []) or []
    _log(f"[REMOTE-OPT] Objective expression (minimised): {obj_expr}", log_path)
    _log(f"[REMOTE-OPT] Objective terms: {objective_terms}", log_path)    
    
    _log(f"[REMOTE-OPT] Conditions: {conds}", log_path)
    _log(f"[REMOTE-OPT] Weights: {weights}", log_path)
    
    # Determine remote root (parent of run_dir usually)
    # Adjust this based on your directory structure
    remote_root = settings_json.get("remote_root", os.path.dirname(run_dir))
    base_name = settings_json.get("base_name", "model")
    input_dir = settings_json.get("input_dir", os.path.join(remote_root, "orig"))
    parallel = settings_json.get("parallel", 80)
    cad_units = settings_json.get("units", "mm")
    previous_solution = settings_json.get("previous_solution", {}) or {}
    
    # Executable paths (customize for your cluster)
    executables = {
        "parallel_domains": settings_json.get("parallel_domains", 1),
        "surface_mesher": "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Surf3D",
        "volume_mesher": "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Mesh3D",
        "prepro_exe": "/home/s.engevabj/codes/PrePro_uns/Gen3d",
        "solver_exe": "/home/s.engevabj/codes/FLITE_uns/UnsMgnsg3d",
        "combine_exe": "/home/s.engevabj/codes/utilities/makeplot2",
        "ensight_exe": "/home/s.engevabj/codes/utilities/engen_tet",
        "splitplot_exe": "/home/s.engevabj/codes/utilities/splitplot2",
        "makeplot_exe": "/home/s.engevabj/codes/utilities/makeplot2",
        "intel_module": "module load compiler/intel/2020/0",
        "gnu_module": "module load compiler/gnu/12/1.0",
        "mpi_intel_module": "module load mpi/intel/2020/0",
        "interpu_script": "$HOME/aeropt/Scripts/Utilities/interpu.py",
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
        parallel = parallel,
        monitor_config_json=monitor_config_json,
        previous_solution=previous_solution
    )
    tm.objective_terms = objective_terms
    
    # Define init and eval functions for BO
    def init_func(X_list, gen_num):
        _log(f"[REMOTE-OPT] Initializing generation {gen_num}: {len(X_list)} designs", log_path)
        tm.init_generation(X_list, gen_num, conds)
    
    def eval_func(X_list, gen_num):
        _log(f"[REMOTE-OPT] Evaluating generation {gen_num}", log_path)
        per_design = tm.evaluate_generation(X_list, gen_num, conds)

        # Reduce per-condition metrics -> scalar objective via expression
        Y = []
        for metrics_per_cond in per_design:
            y = 0.0
            for cond, m in zip(conds, metrics_per_cond):
                w = float(cond.get("Weight", 1.0))
                y += w * obj_func(m)
            Y.append(float(y))

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