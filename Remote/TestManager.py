"""
TestManagerAeropt
-----------------
A lightweight test manager that reuses the student's orchestration *structure* but
routes all execution through your AerOpt pipeline (`pipeline_remote.HPCPipelineManager`).

Use-cases
---------
- As `init_func` for `optimiser.BayesianOptimiser`: starts a batch of remote tests
  for the unevaluated design points of generation `g`.
- As `eval_func` for `optimiser.BayesianOptimiser`: blocks/polls the HPC until all
  tests from that generation finish, then parses `liftdrag.res` to return drag values.

Assumptions (per your confirmation)
-----------------------------------
1) We run remotely on an HPC cluster.
2) Your `main_window` provides:
   - `ssh_client` (paramiko client)
   - `ssh_creds["username"]`
   - `remote_output_dir` (e.g. "/home/<user>/aeropt/aeropt_out/<run>/")
   - `logger` with `.log(str)` method
3) Results are written to:
   "<remote_output_dir>/solutions/n_<n>/liftdrag.res"
"""

import os
import posixpath
import time
import re
from typing import List, Dict, Tuple

try:
    from pipeline_remote import HPCPipelineManager
except Exception as _err:
    raise ImportError("Could not import pipeline_remote.HPCPipelineManager. "
                      "Ensure this file lives in the same environment as AerOpt.")


class TestManager:
    def __init__(self, main_window, poll_interval_s=90, concurrent_tests=0):
        """
        Parameters
        ----------
        main_window : object
            Your GUI main window. Must expose: ssh_client, ssh_creds, remote_output_dir, logger.
        poll_interval_s : int
            Seconds between HPC polling checks while waiting for results.
        concurrent_tests : int
            0 to submit all at once; otherwise submit in waves of this size.
        """
        self.main_window = main_window
        self.poll_s = int(max(10, poll_interval_s))
        self.concurrent_tests = int(max(0, concurrent_tests))

        self.ssh = self.main_window.ssh_client
        self.username = self.main_window.ssh_creds.get("username", "unknown")
        self.remote_root = self.main_window.remote_output_dir.rstrip("/") + "/"
        self.log = getattr(self.main_window, "logger", None)

        self._gen_tests: Dict[int, List[Tuple[str, int, Dict[str, str]]]] = {}
        self._test_to_x: Dict[str, List[float]] = {}

        if self.log:
            self._safe_log(f"[TM-Aeropt] Using remote root: {self.remote_root}")

    # ---------- Helpers ----------

    def _safe_log(self, msg: str):
        try:
            self.log.log(msg)
        except Exception:
            print(msg)

    @staticmethod
    def _extract_int_suffix(name: str, default: int = 0) -> int:
        m = re.search(r"(\d+)$", name.strip())
        return int(m.group(1)) if m else default

    def _alloc_n_index(self, gen_num: int, local_idx: int) -> int:
        """Generate a unique n-index for a (generation, within-gen index) pair."""
        return int(gen_num) * 1_000_000 + int(local_idx)

    def _solutions_dir_for(self, n: int) -> str:
        return posixpath.join(self.remote_root, "solutions", f"n_{n}")

    def _liftdrag_path_for(self, n: int) -> str:
        return posixpath.join(self._solutions_dir_for(n), "liftdrag.res")

    # ---------- Public API for BO ----------

    def init_generation(self, X_list: List[List[float]], gen_num: int) -> None:
        """Submit all tests for this generation."""
        tests = []
        plan = []
        for i, x in enumerate(X_list):
            testname = f"g{gen_num}_Test_{i+1}"
            n_index = self._alloc_n_index(gen_num, i+1)
            plan.append((testname, n_index, x))

        if self.concurrent_tests > 0:
            submit_ptr = 0
            while submit_ptr < len(plan):
                batch = plan[submit_ptr: submit_ptr + self.concurrent_tests]
                self._submit_batch(batch, tests, gen_num)
                submit_ptr += self.concurrent_tests
        else:
            self._submit_batch(plan, tests, gen_num)

        self._gen_tests[gen_num] = tests
        for (tname, _n, _ids) in tests:
            self._test_to_x[tname] = next(x for (tn, n, x) in plan if tn == tname)

        if self.log:
            self._safe_log(f"[TM-Aeropt] Submitted gen {gen_num}: {len(tests)} tests.")

    __call__ = init_generation  # so the class can be passed directly as init_func

    def evaluate_generation(self, X_list, gen_num, conds):
        """
        Wait until all condition-specific results exist, then return
        a list (per design) of per-condition metrics dicts:
        [ [ {"CL":..,"CD":..,"CM":..}, ... ], ... ]
        """
        # sanity: we must have submitted this gen already
        if gen_num not in self._gen_tests:
            self._safe_log(f"[TM-Aeropt][WARN] No submitted tests for gen {gen_num}.")
            return [ [] for _ in X_list ]

        tests = self._gen_tests[gen_num]
        # Build per-test required files based on condition tags
        required = {}
        for (tname, n_idx, _ids) in tests:
            tags = [self._cond_tag(c) for c in conds]
            req_files = [self._metrics_path_for(n_idx, tag) for tag in tags]
            required[tname] = req_files

        # Poll until every required file exists
        unfinished = set(required.keys())
        self._safe_log(f"[TM-Aeropt] Polling HPC for gen {gen_num} results...")
        while unfinished:
            done_now = []
            for tname in list(unfinished):
                if all(self._remote_file_exists(p) for p in required[tname]):
                    done_now.append(tname)
            for t in done_now:
                unfinished.remove(t)
            if unfinished:
                self._safe_log(f"[TM-Aeropt] Waiting ({len(unfinished)} remaining)...")
                time.sleep(self.poll_s)

        # Parse metrics per design, in the same order as X_list
        results = []
        for i, _x in enumerate(X_list):
            n_idx = self._alloc_n_index(gen_num, i+1)
            tags  = [self._cond_tag(c) for c in conds]
            per_cond = []
            for tag in tags:
                m = self._parse_metrics(self._metrics_path_for(n_idx, tag))
                per_cond.append(m)
            results.append(per_cond)

        return results

    # ---------- Internal submission & HPC helpers ----------

    def _submit_batch(self, plan, tests_store, gen_num):
        for (testname, n_index, x) in plan:
            job_ids = self._start_one(testname, n_index, x, gen_num)
            tests_store.append((testname, n_index, job_ids))

    def _start_one(self, testname: str, n_index: int, x: List[float], gen_num: int):
        self._safe_log(f"[TM-Aeropt] Starting {testname} (n={n_index}) with x={x}")

        # MAKE BO X DRIVE THE MORPH
        self.main_window.mesh_viewer.modal_coeffs = list(map(float, x))

        pipe = HPCPipelineManager(self.main_window, n=gen_num, debug=True)
        pipe.morph(n=n_index)
        vol_id = pipe.volume()
        pre_id = pipe.prepro()

        # Use flow conditions & weights from the GUI (see section B)
        conds = getattr(self.main_window, "objective_config", {}).get("conditions", [])
        sol_ids = {}
        for i, cond in enumerate(conds):
            jid = pipe.solver(cond)   # now passes full condition dict
            sol_ids[f"solver_{i}"] = jid

        self._safe_log(f"[TM-Aeropt] Submitted {testname} → jobs: {sol_ids}")
        job_ids = {"volume": vol_id, "prepro": pre_id, **sol_ids}
        return job_ids

    def _remote_file_exists(self, remote_path: str) -> bool:
        try:
            sftp = self.ssh.open_sftp()
            try:
                sftp.stat(remote_path)
                return True
            except IOError:
                return False
            finally:
                sftp.close()
        except Exception as e:
            self._safe_log(f"[TM-Aeropt][WARN] sftp.stat failed for {remote_path}: {e}")
            return False

    def _metrics_path_for(self, n: int, tag: str) -> str:
        return posixpath.join(self._solutions_dir_for(n), tag, "liftdrag.res") ## FIX THE PATH FOR THIS!!!

    def _parse_metrics(self, remote_res_path: str) -> dict:
        try:
            sftp = self.ssh.open_sftp()
            with sftp.file(remote_res_path, "r") as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()
            sftp.close()
        except Exception as e:
            self._safe_log(f"[TM-Aeropt][ERROR] read {remote_res_path}: {e}")
            return {"CL": 0.0, "CD": 1e9, "CM": 0.0}

        last = None
        import re
        for raw in reversed(lines):
            toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            if len(toks) >= 4:
                last = toks
                break
        if not last:
            return {"CL": 0.0, "CD": 1e9, "CM": 0.0}

        # Example mapping:
        #   col4 → Drag/Q (then CD = (Drag/Q) / ??? if needed)
        # If your file already gives CD in a known column, map it directly.
        try:
            drag_over_q = float(last[3])  # 4th column
            # If liftdrag.res already contains CD in a known position, replace this mapping accordingly
            CD = drag_over_q  # or convert to CD if needed
            CL = float(last[2]) if len(last) > 2 else 0.0
            CM = float(last[1]) if len(last) > 1 else 0.0
            return {"CL": CL, "CD": CD, "CM": CM}
        except Exception:
            return {"CL": 0.0, "CD": 1e9, "CM": 0.0}
    
    def _cond_tag(self, cond):
        # short, filesystem-safe tag
        return f"AoA{cond['AoA']}_M{cond.get('Mach',1.0)}_Re{int(cond.get('Re',0))}_T{cond.get('TurbModel',0)}"

