# remoteMorph.py
import os, sys, json, time
import numpy as np

# add project paths (same pattern as remoteOpt.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

for subdir in ["", "Optimisation", "FileRW", "Remote", "MeshGeneration"]:
    p = os.path.join(project_root, subdir) if subdir else project_root
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline_cluster import ClusterPipelineManager


def _log(msg, log_path=None):
    print(msg, flush=True)
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


# ----------------------------
# (A) Orchestrator mode
# ----------------------------
def orchestrate_run(run_dir: str):
    run_dir = os.path.abspath(run_dir)
    settings_path = os.path.join(run_dir, "morph_settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Missing morph_settings.json in: {run_dir}")

    with open(settings_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    # logs
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"morph_orchestrator_{time.strftime('%Y%m%d_%H%M%S')}.log")

    remote_output = s.get("remote_output")  # base aeropt_out/<case>
    base_name     = s.get("base_name", "model")
    input_dir     = s.get("input_dir", "")
    morph_basis   = s.get("morph_basis_json", "")
    units         = s.get("cad_units", "mm")
    n_cases       = int(s.get("n_cases", 1))
    sigma         = float(s.get("coeff_sigma", 0.5))
    seed          = s.get("seed", None)
    parallel      = int(s.get("parallel_domains", 80))

    if not remote_output:
        raise ValueError("morph_settings.json must include 'remote_output'")

    k = int(s.get("k_modes", 5))

    rng = np.random.default_rng(seed if seed is not None else None)

    _log(f"[MORPH-ORCH] run_dir={run_dir}", log_path)
    _log(f"[MORPH-ORCH] remote_output={remote_output}", log_path)
    _log(f"[MORPH-ORCH] n_cases={n_cases}, sigma={sigma}, k={k}", log_path)
    _log(f"[MORPH-ORCH] morph_basis_json={morph_basis}", log_path)

    # submit each case
    for i in range(1, n_cases + 1):
        modal_coeffs = (rng.normal(0.0, sigma, size=k)).tolist()

        config = {
            "remote_output": remote_output,
            "base_name": base_name,
            "input_dir": input_dir,
            "parallel_domains": parallel,
            "modal_coeffs": modal_coeffs,
            "morph_basis_json": morph_basis,
            "cad_units": units,
        }

        pipe = ClusterPipelineManager(config_dict=config, gen=0, n=i)

        try:
            morph_id = pipe.morph(n=i)  # writes morph_config_n_i.json + sbatch
            vol_id   = pipe.volume(runafter=morph_id)
            _log(f"[MORPH-ORCH] n={i}: morph={morph_id}, volume={vol_id}", log_path)
        except Exception as e:
            _log(f"[MORPH-ORCH][ERROR] n={i} failed: {e}", log_path)

    _log("[MORPH-ORCH] Submitted all cases.", log_path)


def main():
    orchestrate_run(sys.argv[1])
    return


if __name__ == "__main__":
    main()