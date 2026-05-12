# pipelineMorph.py
import os, sys, json, time, subprocess
import numpy as np
from pipeline_cluster import ClusterPipelineManager  # already in your file :contentReference[oaicite:3]{index=3}

def _log(msg, log_path=None):
    print(msg, flush=True)
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

def _jobs_still_in_queue(job_ids):
    """
    Return the subset of job_ids that are still visible to squeue.
    """
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return []

    # squeue returns active/pending/running jobs; finished jobs vanish
    cmd = ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        alive = set(x.strip() for x in out if x.strip())
        return [j for j in job_ids if j in alive]
    except Exception:
        # If squeue fails for some reason, be conservative and assume still running
        return job_ids

def _wait_for_jobs(job_ids, poll_s=60, log_path=None, tag="volume"):
    """
    Block until all given job IDs have left the queue.
    """
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return

    _log(f"[MORPH-ORCH] Waiting for {tag} jobs to finish: {job_ids}", log_path)
    while True:
        alive = _jobs_still_in_queue(job_ids)
        if not alive:
            _log(f"[MORPH-ORCH] All {tag} jobs finished.", log_path)
            return
        _log(f"[MORPH-ORCH] {tag} still running/pending: {alive} (sleep {poll_s}s)", log_path)
        time.sleep(int(max(10, poll_s)))

def orchestrate_run(run_dir: str):
    run_dir = os.path.abspath(run_dir)
    settings_path = os.path.join(run_dir, "morph_settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Missing morph_settings.json in: {run_dir}")

    with open(settings_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"morph_orchestrator_{time.strftime('%Y%m%d_%H%M%S')}.log")

    remote_output = s.get("remote_output")
    base_name     = s.get("base_name", "model")
    input_dir     = s.get("input_dir", "")
    morph_basis   = s.get("morph_basis_json", "")
    units         = s.get("cad_units", "mm")
    n_cases       = int(s.get("n_cases", 1))
    sigma         = float(s.get("coeff_sigma", 0.5))
    seed          = s.get("seed", None)
    parallel      = int(s.get("parallel_domains", 80))
    run_mode      = s.get("run_mode", "morph")

    # NEW:
    batch_size    = int(s.get("batch_size", 10))      # run 10 at a time
    poll_s        = int(s.get("poll_s", 60))          # how often to check

    with open(morph_basis, "r", encoding="utf-8") as f:
        b = json.load(f)

    if not remote_output:
        raise ValueError("morph_settings.json must include 'remote_output'")

    rng = np.random.default_rng(seed if seed is not None else None)
    parameterisation_method = str(b.get("parameterisation_method", "modal")).strip().lower()
    direct_subtype = str(b.get("direct_parameterisation_subtype", "") or "").strip().lower()

    use_pca = bool(b.get("use_pca", False))
    pca_k_final = b.get("pca_k_final", None)

    k = int(b.get("k_modes", 5))
    vector_mode = b.get("vector_mode", "local_frame")
    normal_project = bool(b.get("normal_project", True))
    global_modes = bool(b.get("global_modes", False))
    global_mode_config = b.get("global_mode_config", [])
    use_local_modes = bool(b.get("use_local_modes", True))
    global_only = bool(b.get("global_only", False))
    if global_only:
        use_local_modes = False

    n_cn = len(b.get("control_nodes", []))
    n_global = len(global_mode_config) if global_modes and global_mode_config else (8 if global_modes else 0)

    if parameterisation_method == "direct":
        if direct_subtype == "xyz":
            coeff_len = 3 * n_cn
        elif direct_subtype == "normal":
            coeff_len = n_cn
        else:
            raise ValueError(f"Unknown direct_parameterisation_subtype: {direct_subtype}")

    elif use_pca:
        if pca_k_final is None:
            raise ValueError("use_pca=True but pca_k_final is missing in morph_basis.json")
        coeff_len = int(pca_k_final)

    else:
        if use_local_modes:
            if normal_project:
                local_len = k
            else:
                if vector_mode == "xyz":
                    local_len = 3 * k
                else:
                    local_len = 3 * k
        else:
            local_len = 0

        coeff_len = n_global + local_len

    _log(f"[MORPH-ORCH] run_dir={run_dir}", log_path)
    _log(f"[MORPH-ORCH] remote_output={remote_output}", log_path)
    _log(f"[MORPH-ORCH] n_cases={n_cases}, sigma={sigma}, k={k}", log_path)
    _log(f"[MORPH-ORCH] batch_size={batch_size}, poll_s={poll_s}", log_path)
    _log(f"[MORPH-ORCH] morph_basis_json={morph_basis}", log_path)

    # Submit in WAVES of batch_size
    i = 1
    while i <= n_cases:
        j_end = min(n_cases, i + batch_size - 1)
        _log(f"[MORPH-ORCH] Submitting batch: n={i}..{j_end}", log_path)

        jobs_to_wait = []
        wait_tag = None

        for n in range(i, j_end + 1):
            #modal_coeffs = (rng.normal(0.0, sigma, size=coeff_len)).tolist()
            modal_coeffs = (rng.normal(1.0, 1.0, size=coeff_len)).tolist()

            config = {
                "remote_output": remote_output,
                "base_name": base_name,
                "input_dir": input_dir,
                "parallel_domains": parallel,
                "modal_coeffs": modal_coeffs,
                "morph_basis_json": morph_basis,
                "cad_units": units,
            }

            pipe = ClusterPipelineManager(config_dict=config, gen=0, n=n)

            try:
                if run_mode == "disp":
                    cfg_path = pipe._write_morph_config()
                    _log(f"[MORPH-ORCH] n={n}: wrote displacement config {cfg_path}", log_path)

                elif run_mode == "morph":
                    morph_id = pipe.morph(n=n)
                    jobs_to_wait.append(morph_id)
                    wait_tag = "morph"
                    _log(f"[MORPH-ORCH] n={n}: morph={morph_id}", log_path)

                elif run_mode == "vol":
                    morph_id = pipe.morph(n=n)
                    vol_id = pipe.volume(runafter=morph_id)
                    jobs_to_wait.append(vol_id)
                    wait_tag = "volume"
                    _log(f"[MORPH-ORCH] n={n}: morph={morph_id}, volume={vol_id}", log_path)

                else:
                    raise ValueError(f"Unknown run_mode: {run_mode}")

            except Exception as e:
                _log(f"[MORPH-ORCH][ERROR] n={n} failed submit: {e}", log_path)

        if jobs_to_wait:
            _wait_for_jobs(jobs_to_wait, poll_s=poll_s, log_path=log_path, tag=wait_tag or "jobs")

        i = j_end + 1

    _log("[MORPH-ORCH] Submitted and completed all batches.", log_path)

if __name__ == "__main__":
    import sys
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    orchestrate_run(run_dir)