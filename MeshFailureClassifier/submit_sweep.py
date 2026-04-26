#!/usr/bin/env python3
"""
submit_sweep.py
Driver/orchestrator for mesh-feasibility classifier dataset generation.

Expected layout:
  <run_dir>/sweep_settings.json      (written by UI)
  <remote_output>/orig/             (baseline inputs; baseline .fro must be here)
  <run_dir>/morph/morph_basis.json  (written by export_morph_basis_for_opt)

Outputs:
  <run_dir>/configs.jsonl           (one line per case: coeffs, ids, dirs)
  <run_dir>/dataset.jsonl           (one line per case: label + features + config)
  <run_dir>/logs/submit_sweep_*.log
  <run_dir>/model.joblib (optional, if sklearn available and train_at_end=True)
"""

import os
import sys
import json
import time
import glob
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# .../Scripts/MeshFailureClassifier/sweep
# go up 2 levels -> .../Scripts
SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

REMOTE_DIR = os.path.join(SCRIPTS_DIR, "Remote")
if REMOTE_DIR not in sys.path:
    sys.path.insert(0, REMOTE_DIR)
    
MESH_DIR = os.path.join(SCRIPTS_DIR, "MeshGeneration")
if MESH_DIR not in sys.path:
    sys.path.insert(0, MESH_DIR)

from Remote.pipeline_cluster import ClusterPipelineManager


# ----------------------------
# small utils
# ----------------------------
def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _log(msg: str, log_path: Optional[str] = None) -> None:
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _jobs_still_in_queue(job_ids: List[str]) -> List[str]:
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return []
    cmd = ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        alive = set(x.strip() for x in out if x.strip())
        return [j for j in job_ids if j in alive]
    except Exception:
        # if squeue fails, be conservative
        return job_ids


def _wait_for_jobs(job_ids: List[str], poll_s: int, log_path: Optional[str], tag: str) -> None:
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return
    _log(f"[SWEEP] Waiting for {tag} jobs to finish: {job_ids}", log_path)
    while True:
        alive = _jobs_still_in_queue(job_ids)
        if not alive:
            _log(f"[SWEEP] All {tag} jobs finished.", log_path)
            return
        _log(f"[SWEEP] {tag} still running/pending: {alive} (sleep {poll_s}s)", log_path)
        time.sleep(int(max(10, poll_s)))


def _infer_base_name_from_orig(input_dir: str) -> str:
    # Prefer <something>.fro in orig/
    cands = sorted(glob.glob(os.path.join(input_dir, "*.fro")))
    if cands:
        return os.path.splitext(os.path.basename(cands[0]))[0]
    # Fall back to any known mesh type
    for ext in ("*.vtk", "*.vtm", "*.case"):
        c = sorted(glob.glob(os.path.join(input_dir, ext)))
        if c:
            return os.path.splitext(os.path.basename(c[0]))[0]
    return "model"


def _default_success_check(remote_output: str, n: int):
    """
    Heuristic success criterion until you specify the definitive artifact.
    Marks success if volumes/n_<n> exists and contains any file with plausible mesh extensions.
    """
    vol_dir = os.path.join(remote_output, "volumes", f"n_{n}")
    if not os.path.isdir(vol_dir):
        return 0, "no_volume_dir"

    exts = (".msh", ".cgns", ".vtk", ".vtm", ".plt", ".grid", ".p3d", ".cas", ".h5", ".dat")
    files = []
    for root, _, fnames in os.walk(vol_dir):
        for fn in fnames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    if files:
        return 1, f"found:{os.path.basename(files[0])}"
    return 0, "no_mesh_artifact"


def _try_compute_features(run_dir: str, remote_output: str, input_dir: str, base_name: str, n: int) -> Dict[str, Any]:
    """
    Hook: if you later add features.py with compute_features(...), this will use it.
    For now it returns {} if unavailable.
    """
    try:
        # You will create this later:
        #   ~/aeropt/Scripts/Remote/features.py
        # with: compute_features(orig_fro_path, morphed_fro_path, **kwargs) -> dict
        from features import compute_features  # type: ignore
    except Exception:
        return {}

    orig_fro = os.path.join(input_dir, f"{base_name}.fro")
    morphed_fro = os.path.join(remote_output, "surfaces", f"n_{n}", f"{base_name}.fro")
    return compute_features(orig_fro, morphed_fro, run_dir=run_dir, case_id=n)


@dataclass
class SweepSettings:
    remote_output: str
    run_dir: str
    input_dir: str
    morph_basis_json: str
    cad_units: str = "mm"
    parallel_domains: int = 80
    n_cases: int = 1
    batch_size: int = 10
    poll_s: int = 200

    # sampling knobs
    coeff_sigma: float = 0.5
    seed: Optional[int] = None

    # dataset/training
    dataset_path: str = ""
    train_at_end: bool = False  # set True later when you’re ready
    model_out: str = ""         # default <run_dir>/model.joblib


def _load_settings(run_dir: str) -> SweepSettings:
    p = os.path.join(run_dir, "sweep_settings.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing sweep_settings.json in {run_dir}")

    s = _read_json(p)

    ss = SweepSettings(
        remote_output=s["remote_output"],
        run_dir=s.get("run_dir", run_dir),
        input_dir=s["input_dir"],
        morph_basis_json=s["morph_basis_json"],
        cad_units=s.get("cad_units", "mm"),
        parallel_domains=int(s.get("parallel_domains", 80)),
        n_cases=int(s.get("n_cases", 1)),
        batch_size=int(s.get("batch_size", 10)),
        poll_s=int(s.get("poll_s", 200)),
        coeff_sigma=float(s.get("coeff_sigma", 0.5)),
        seed=s.get("seed", None),
        dataset_path=s.get("dataset_path", os.path.join("/scratch/$USER/aeropt/aeropt_out/", "dataset.jsonl")),
        train_at_end=bool(s.get("train_at_end", False)),
        model_out=s.get("model_out", os.path.join(run_dir, "model.joblib")),
    )
    return ss


def main():
    if len(sys.argv) < 2:
        print("Usage: submit_sweep.py <run_dir>")
        sys.exit(2)

    run_dir = os.path.abspath(sys.argv[1])
    settings = _load_settings(run_dir)

    log_dir = os.path.join(run_dir, "logs")
    _ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"submit_sweep_{_now()}.log")

    _log(f"[SWEEP] run_dir={run_dir}", log_path)
    _log(f"[SWEEP] remote_output={settings.remote_output}", log_path)
    _log(f"[SWEEP] input_dir={settings.input_dir}", log_path)
    _log(f"[SWEEP] n_cases={settings.n_cases} batch_size={settings.batch_size} poll_s={settings.poll_s}", log_path)
    _log(f"[SWEEP] morph_basis_json={settings.morph_basis_json}", log_path)

    # Read k_modes from morph_basis.json (uploaded by UI)
    if not os.path.exists(settings.morph_basis_json):
        raise FileNotFoundError(f"morph_basis_json not found: {settings.morph_basis_json}")
    mb = _read_json(settings.morph_basis_json)
    k_modes = int(mb.get("k_modes", 6))
    _log(f"[SWEEP] k_modes={k_modes}", log_path)

    base_name = mb.get("base_name", None) or _infer_base_name_from_orig(settings.input_dir)
    _log(f"[SWEEP] base_name={base_name}", log_path)

    # Prepare outputs
    configs_path = os.path.join(run_dir, "configs.jsonl")
    dataset_path = settings.dataset_path
    _ensure_dir(os.path.dirname(dataset_path) or run_dir)

    # deterministic RNG if seed provided
    rng = np.random.default_rng(settings.seed if settings.seed is not None else None)

    # Generate configs once and persist (so dataset is reproducible)
    if not os.path.exists(configs_path):
        _log(f"[SWEEP] Writing configs → {configs_path}", log_path)
        for n in range(1, settings.n_cases + 1):
            modal_coeffs = (rng.normal(0.0, settings.coeff_sigma, size=k_modes)).tolist()
            _append_jsonl(configs_path, {
                "case_id": n,
                "gen": 0,
                "k_modes": k_modes,
                "coeff_sigma": settings.coeff_sigma,
                "modal_coeffs": modal_coeffs,
            })
    else:
        _log(f"[SWEEP] Using existing configs → {configs_path}", log_path)

    # Load configs into memory (small enough)
    configs = []
    with open(configs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                configs.append(json.loads(line))

    # Submit in waves
    idx = 0
    while idx < len(configs):
        batch = configs[idx: idx + settings.batch_size]
        n0 = batch[0]["case_id"]
        n1 = batch[-1]["case_id"]
        _log(f"[SWEEP] Submitting batch cases {n0}..{n1}", log_path)

        vol_job_ids: List[str] = []
        submitted_cases: List[int] = []

        # Submit morph+volume for each case
        for cfg in batch:
            n = int(cfg["case_id"])
            modal_coeffs = cfg["modal_coeffs"]

            config_dict = {
                "remote_output": settings.remote_output,
                "base_name": base_name,
                "input_dir": settings.input_dir,
                "parallel_domains": int(settings.parallel_domains),
                "modal_coeffs": modal_coeffs,
                "morph_basis_json": settings.morph_basis_json,
                "cad_units": settings.cad_units,
            }

            pipe = ClusterPipelineManager(config_dict=config_dict, gen=0, n=n)

            try:
                morph_id = pipe.morph(n=n)
                vol_id = pipe.volume(runafter=morph_id)
                vol_job_ids.append(str(vol_id))
                submitted_cases.append(n)
                _log(f"[SWEEP] n={n}: morph={morph_id} volume={vol_id}", log_path)
            except Exception as e:
                _log(f"[SWEEP][ERROR] n={n}: submit failed: {e}", log_path)

        # Wait for the volumes in this batch to complete
        _wait_for_jobs(vol_job_ids, poll_s=settings.poll_s, log_path=log_path, tag="volume")

        # Post-process this batch: compute label + features, append dataset rows
        for n in submitted_cases:
            label, reason = _default_success_check(settings.remote_output, n)

            feats = _try_compute_features(
                run_dir=run_dir,
                remote_output=settings.remote_output,
                input_dir=settings.input_dir,
                base_name=base_name,
                n=n,
            )

            # Write one dataset row
            row = {
                "case_id": n,
                "label": int(label),
                "reason": reason,
                "base_name": base_name,
                "remote_output": settings.remote_output,
                "input_dir": settings.input_dir,
                "morph_basis_json": settings.morph_basis_json,
                "config": next((c for c in batch if int(c["case_id"]) == int(n)), None),
                "features": feats,
                "paths": {
                    "orig_fro": os.path.join(settings.input_dir, f"{base_name}.fro"),
                    "morphed_fro": os.path.join(settings.remote_output, "surfaces", f"n_{n}", f"{base_name}.fro"),
                    "volume_dir": os.path.join(settings.remote_output, "volumes", f"n_{n}"),
                },
                "timestamp": time.time(),
            }
            _append_jsonl(dataset_path, row)
            _log(f"[SWEEP] dataset row written: n={n} label={label} ({reason}) feats={len(feats)}", log_path)

        idx += settings.batch_size

    _log("[SWEEP] All batches completed.", log_path)

    # Optional: train a quick classifier at the end (only if features exist and sklearn is available)
    if settings.train_at_end:
        _log("[SWEEP] train_at_end=True → attempting to train a classifier", log_path)
        try:
            import joblib
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, roc_auc_score
            from sklearn.ensemble import RandomForestClassifier

            X = []
            y = []
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    feats = r.get("features", {})
                    if not feats:
                        continue
                    X.append([float(v) for v in feats.values()])
                    y.append(int(r["label"]))

            if len(X) < 20:
                _log("[SWEEP][WARN] Not enough feature rows to train (need ~20+).", log_path)
                return

            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=int)

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=0,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(Xtr, ytr)

            p = clf.predict_proba(Xte)[:, 1]
            yhat = (p >= 0.5).astype(int)

            try:
                auc = roc_auc_score(yte, p)
                _log(f"[SWEEP] ROC-AUC: {auc:.4f}", log_path)
            except Exception:
                pass

            rep = classification_report(yte, yhat, digits=4)
            _log("[SWEEP] Classification report:\n" + rep, log_path)

            joblib.dump({"model": clf, "feature_order": list(feats.keys())}, settings.model_out)
            _log(f"[SWEEP] Saved model → {settings.model_out}", log_path)
        except Exception as e:
            _log(f"[SWEEP][WARN] Training failed: {e}", log_path)


if __name__ == "__main__":
    main()
