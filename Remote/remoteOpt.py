# remote_opt.py
import os, json, time, posixpath, sys, re, subprocess
import numpy as np

# --- Helpers
def _log(msg, log_path):
    print(msg, flush=True)
    try:
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def _cond_tag(cond):
    return f"AoA{cond.get('AoA')}_M{cond.get('Mach',1.0)}_Re{int(cond.get('Re',0))}_T{cond.get('TurbModel',0)}"

def _solutions_dir(remote_root, n):
    return posixpath.join(remote_root, "solutions", f"n_{n}")

def _metrics_path(remote_root, n, tag):
    return posixpath.join(_solutions_dir(remote_root, n), f"{tag}", "liftdrag.res")

def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, check=False, capture_output=True, text=True)

# --- Headless TestManager (no Qt, no SSH)
class HeadlessTestManager:
    def __init__(self, remote_root, poll_s=120, concurrent_tests=0, logger=None):
        self.remote_root = remote_root.rstrip("/") + "/"
        self.poll_s = int(max(10, poll_s))
        self.concurrent_tests = int(max(0, concurrent_tests))
        self.jobs = {}
        self.logger = logger or (lambda m: None)

    def _alloc_n_index(self, gen_num, local_idx):
        return int(gen_num) * 1_000_000 + int(local_idx)

    def _start_one(self, n_index, x, conds):
        # Compose a minimal “submit one design” script by calling your pipeline remotely.
        # Here we just write a tiny Python driver that imports your pipeline and submits jobs.
        driver = f"""#!/usr/bin/env python3
import os, json
from pipeline_remote import HPCPipelineManager

# Fake a tiny main_window-like shim
class MW: 
    ssh_client=None; remote_output_dir="{self.remote_root}"; logger=type("L",(),{{"log":print}})()
    input_file_path=os.path.join("{self.remote_root}","orig","DUMMY.vtm")  # not used if you cold-start
    input_directory="{self.remote_root}"
    output_directory="{self.remote_root}"

mw = MW()
pipe = HPCPipelineManager(mw, n={n_index})
# Make BO X drive the deformation on the remote—assuming your morph step reads these from mesh_viewer;
# if needed you can persist them to a file and have morph read them.
# For now we just run volume→prepro and solver over all conditions; morph() if your workflow requires.
# pipe.morph()
v = pipe.volume()
p = pipe.prepro(runafter=v)
for i,cond in enumerate({json.dumps(conds)},1):
    pipe.solver(cond, nc=i)
"""
        run_dir = os.path.join(self.remote_root, "headless", f"n_{n_index}")
        _safe_mkdir(run_dir)
        drv_path = os.path.join(run_dir, "submit_one.py")
        with open(drv_path, "w") as f:
            f.write(driver)

        bf = os.path.join(run_dir, "batchfile_submit_one")
        with open(bf, "w") as f:
            f.write("\n".join([
                "#!/bin/bash -l",
                "#SBATCH --job-name=opt_n{}".format(n_index),
                "#SBATCH --output=opt_n{}.out".format(n_index),
                "#SBATCH --error=opt_n{}.err".format(n_index),
                "#SBATCH --time=3-00:00",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "source ~/.bashrc",
                "set -euo pipefail",
                f"python3 {drv_path}"
            ]) + "\n")

        out = run_cmd(f"sbatch {bf}", cwd=run_dir)
        jid = ""
        if "Submitted batch job" in out.stdout:
            jid = out.stdout.strip().split()[-1]
        self.jobs[n_index] = jid
        self.logger(f"[HEADLESS] Submitted n={n_index} → job {jid}")

    # Exposed to BO
    def init_generation(self, X_list, gen_num, conds):
        plan = [(self._alloc_n_index(gen_num, i+1), list(map(float, x))) for i, x in enumerate(X_list)]
        if self.concurrent_tests > 0:
            for i in range(0, len(plan), self.concurrent_tests):
                chunk = plan[i:i+self.concurrent_tests]
                for n_index, x in chunk:
                    self._start_one(n_index, x, conds)
        else:
            for n_index, x in plan:
                self._start_one(n_index, x, conds)

    def evaluate_generation(self, X_list, gen_num, conds):
        # Wait for all liftdrag.res files to appear, then parse.
        tags = [_cond_tag(c) for c in conds]
        need = []
        for i,_x in enumerate(X_list, 1):
            n_index = self._alloc_n_index(gen_num, i)
            for t in tags:
                need.append((_metrics_path(self.remote_root, n_index, t), n_index, t))

        self.logger(f"[HEADLESS] Waiting for {len(need)} results…")
        unfinished = set(p for (p,_,_) in need)
        while unfinished:
            done = {p for p in list(unfinished) if os.path.exists(p)}
            unfinished -= done
            if unfinished:
                time.sleep(self.poll_s)

        def parse_one(path):
            try:
                with open(path, "r") as f:
                    lines = f.read().splitlines()
                last = None
                for raw in reversed(lines):
                    toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                    if len(toks) >= 4:
                        last = toks; break
                if not last: return {"CL":0.0,"CD":1e9,"CM":0.0}
                drag_over_q = float(last[3])
                CD = drag_over_q
                CL = float(last[2]) if len(last)>2 else 0.0
                CM = float(last[1]) if len(last)>1 else 0.0
                return {"CL":CL,"CD":CD,"CM":CM}
            except Exception:
                return {"CL":0.0,"CD":1e9,"CM":0.0}

        results = []
        for i,_x in enumerate(X_list, 1):
            n_index = self._alloc_n_index(gen_num, i)
            per_cond = [parse_one(_metrics_path(self.remote_root, n_index, t)) for t in tags]
            results.append(per_cond)
        return results

def main():
    if len(sys.argv) < 2:
        print("usage: remote_opt.py <remote_run_dir>", flush=True)
        sys.exit(2)
    run_dir = sys.argv[1]
    log_path = os.path.join(run_dir, "remote_opt.log")
    _safe_mkdir(run_dir)

    # Load configs written by the GUI
    with open(os.path.join(run_dir, "bo_settings.json")) as f:
        settings_json = json.load(f)
    with open(os.path.join(run_dir, "objective.json")) as f:
        objective = json.load(f)

    # Map strings → classes for kernel & acquisition (simple dispatch)
    from Optimisation.BayesianOptimisation.optimiser import BayesianOptimiser
    from Optimisation.BayesianOptimisation.kernels import (
        RBFKernel, SquaredExponentialKernel, ExponentialKernel, Mat12Kern, Mat32Kern, Mat52Kern
    )
    from Optimisation.BayesianOptimisation.acquisition_functions import EI, POI, UCB

    kern_map = {
        "RBFKernel": RBFKernel, "Squared Exponential Kernel": SquaredExponentialKernel,
        "Exponential Kernel": ExponentialKernel, "Mat12Kern": Mat12Kern,
        "Mat32Kern": Mat32Kern, "Mat52Kern": Mat52Kern
    }
    acq_map = {"Expected Improvement": EI, "Probability of Improvement": POI, "Upper Confidence Bound": UCB}

    settings = dict(settings_json)
    settings["kernel"] = kern_map[settings_json["kernel"]]
    settings["acquisition_function"] = acq_map[settings_json["acquisition_function"]]
    settings["sim_dir"] = run_dir if settings.get("sim_dir","") == "" else settings["sim_dir"]

    conds = objective.get("conditions", [])
    weights = [c.get("Weight", 1.0) for c in conds]

    tm = HeadlessTestManager(remote_root=run_dir.rsplit("/postprocessed/",1)[0] if "/postprocessed/" in run_dir else run_dir)

    def init_func(X_list, gen_num):
        _log(f"[REMOTE-OPT] init gen {gen_num}: {len(X_list)} designs", log_path)
        tm.init_generation(X_list, gen_num, conds)

    def eval_func(X_list, gen_num):
        _log(f"[REMOTE-OPT] eval gen {gen_num}", log_path)
        per_design = tm.evaluate_generation(X_list, gen_num, conds)
        # Reduce per-condition metrics → scalar objective using weights
        Y = []
        for metrics in per_design:
            # Simple example: sum(weights * CD); customise as needed
            y = 0.0
            for w, m in zip(weights, metrics):
                y += float(w) * float(m.get("CD", 1e9))
            Y.append(y)
        return np.array(Y, dtype=float)

    bo = BayesianOptimiser(settings, eval_func=eval_func, init_func=init_func)
    X_best, Y_best = bo.optimise(cont=True)
    _log(f"[REMOTE-OPT] DONE. Best X={X_best}  Y={Y_best}", log_path)

if __name__ == "__main__":
    main()
