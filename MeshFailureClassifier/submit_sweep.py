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
from pathlib import Path

import numpy as np

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# .../Scripts/MeshFailureClassifier/sweep
# go up 2 levels -> .../Scripts
SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
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


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _case_state_path(run_dir: str, n: int) -> str:
    return os.path.join(run_dir, "state", f"case_{int(n)}.json")


def _load_case_state(run_dir: str, n: int) -> Dict[str, Any]:
    path = _case_state_path(run_dir, n)
    if not os.path.exists(path):
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _save_case_state(run_dir: str, n: int, **updates: Any) -> Dict[str, Any]:
    state = _load_case_state(run_dir, n)
    state.update(updates)
    state["case_id"] = int(n)
    state["updated"] = time.time()
    _write_json_atomic(_case_state_path(run_dir, n), state)
    return state


def _case_is_terminal(run_dir: str, n: int) -> bool:
    state = _load_case_state(run_dir, n)
    return state.get("status") in {"passed", "failed", "recorded"}


def _load_recorded_case_ids(dataset_path: str) -> set[int]:
    recorded: set[int] = set()
    if not os.path.exists(dataset_path):
        return recorded

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("case_id") is not None:
                    recorded.add(int(row["case_id"]))
            except Exception:
                continue
    return recorded


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


def _default_success_check(remote_output: str, n: int, gen: int = 0):
    """
    Return:
      (1, reason) for a successful volume mesh,
      (0, reason) for a completed failed attempt,
      (None, reason) if the attempt is not yet classifiable.

    Both pass and fail are terminal outcomes for classifier-data generation.
    """
    vol_dir = os.path.join(remote_output, "volumes", f"n_{int(gen)}")
    if not os.path.isdir(vol_dir):
        return None, "missing_volume_directory"

    for pattern in (
        os.path.join(vol_dir, f"*_{int(n)}.plt"),
        os.path.join(vol_dir, f"*_{int(n)}.msh"),
        os.path.join(vol_dir, f"*_{int(n)}.cgns"),
        os.path.join(vol_dir, f"*_{int(n)}.vtk"),
        os.path.join(vol_dir, f"*_{int(n)}.vtm"),
    ):
        for path in glob.glob(pattern):
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return 1, f"found:{os.path.basename(path)}"

    logs = glob.glob(os.path.join(vol_dir, f"volume_output_{int(n)}*"))
    for log_path in logs:
        if not os.path.isfile(log_path):
            continue
        txt = Path(log_path).read_text(encoding="utf-8", errors="ignore")
        if "Error Stop" in txt:
            return 0, "error_stop_found"
        if os.path.getsize(log_path) > 0:
            return 0, "volume_log_present_without_mesh_artifact"

    return None, "volume_attempt_not_complete"


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
    morphed_fro = os.path.join(remote_output, "surfaces", "n_0", f"{base_name}_{n}.fro")
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
    dataset_path = os.path.expandvars(settings.dataset_path)
    _ensure_dir(os.path.dirname(dataset_path) or run_dir)
    _ensure_dir(os.path.join(run_dir, "state"))
    recorded_case_ids = _load_recorded_case_ids(dataset_path)
    _log(f"[SWEEP] dataset already contains {len(recorded_case_ids)} case IDs", log_path)

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
        raw_batch = configs[idx: idx + settings.batch_size]
        batch = []
        for cfg in raw_batch:
            n = int(cfg["case_id"])
            if n in recorded_case_ids:
                _save_case_state(run_dir, n, status="recorded")
                _log(f"[SWEEP][RESUME] n={n}: already in dataset; skipping", log_path)
                continue
            batch.append(cfg)

        if not batch:
            idx += settings.batch_size
            continue

        n0 = batch[0]["case_id"]
        n1 = batch[-1]["case_id"]
        _log(f"[SWEEP] Processing batch cases {n0}..{n1}", log_path)

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

            if _case_is_terminal(run_dir, n):
                submitted_cases.append(n)
                _log(f"[SWEEP][RESUME] n={n}: terminal state found; no resubmission", log_path)
                continue

            # Backfill state from artifacts/logs created by an earlier run.
            # This is essential when restarting a run that predates state files.
            prior_label, prior_reason = _default_success_check(
                settings.remote_output,
                n,
                gen=0,
            )
            if prior_label is not None:
                _save_case_state(
                    run_dir,
                    n,
                    status=("passed" if int(prior_label) == 1 else "failed"),
                    label=int(prior_label),
                    reason=prior_reason,
                    recovered_from_existing_outputs=True,
                )
                submitted_cases.append(n)
                _log(
                    f"[SWEEP][RESUME] n={n}: recovered existing terminal result "
                    f"label={prior_label} ({prior_reason}); no resubmission",
                    log_path,
                )
                continue

            pipe = ClusterPipelineManager(config_dict=config_dict, gen=0, n=n)

            try:
                _save_case_state(run_dir, n, status="submitting")
                morph_id = pipe.morph(n=n)
                vol_id = pipe.volume(runafter=morph_id)
                if vol_id:
                    vol_job_ids.append(str(vol_id))
                submitted_cases.append(n)
                _save_case_state(
                    run_dir, n,
                    status="submitted",
                    morph_job=morph_id,
                    volume_job=vol_id,
                )
                _log(f"[SWEEP] n={n}: morph={morph_id} volume={vol_id}", log_path)
            except Exception as e:
                _save_case_state(run_dir, n, status="submission_error", error=repr(e))
                _log(f"[SWEEP][ERROR] n={n}: submit failed: {e}", log_path)

        # Wait for the volumes in this batch to complete
        _wait_for_jobs(vol_job_ids, poll_s=settings.poll_s, log_path=log_path, tag="volume")

        # Post-process this batch: compute label + features, append dataset rows
        for n in submitted_cases:
            if n in recorded_case_ids:
                continue

            label, reason = _default_success_check(settings.remote_output, n, gen=0)
            if label is None:
                _save_case_state(run_dir, n, status="incomplete", reason=reason)
                _log(f"[SWEEP][WARN] n={n}: {reason}; leaving for future restart", log_path)
                continue

            _save_case_state(
                run_dir,
                n,
                status=("passed" if int(label) == 1 else "failed"),
                label=int(label),
                reason=reason,
            )

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
                    "morphed_fro": os.path.join(settings.remote_output, "surfaces", "n_0", f"{base_name}_{n}.fro"),
                    "volume_dir": os.path.join(settings.remote_output, "volumes", "n_0"),
                },
                "timestamp": time.time(),
            }
            _append_jsonl(dataset_path, row)
            recorded_case_ids.add(int(n))
            _save_case_state(
                run_dir,
                n,
                status="recorded",
                label=int(label),
                reason=reason,
                dataset_path=dataset_path,
            )
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