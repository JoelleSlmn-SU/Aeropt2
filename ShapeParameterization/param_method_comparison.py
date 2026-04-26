import os
import re
import json
import glob
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

@dataclass
class MethodConfig:
    name: str
    folder: str
    dof: int


MAIN_DIR = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples"
CASE = "0"   # e.g. 0 -> n_0, "ell" -> n_ell, etc.

OUT_DIR = os.path.join(MAIN_DIR, f"analysis_results_{CASE}")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = [
    MethodConfig(name="xyz",        folder="param1", dof=6),
    MethodConfig(name="dn",         folder="param2", dof=6),
    MethodConfig(name="lap",        folder="param3", dof=6),
    MethodConfig(name="lap_global", folder="param4", dof=6),
]

# Toggle this on to additionally compare morph smoothness / robustness
# using features.py on the morphed .fro meshes.
COMPARE_MORPH = True

# Baseline/original .fro mesh used by features.py.
# If left as None, the script will try a few automatic guesses.
ORIG_FRO: Optional[str] = None

# Optional path to morph_basis.json for region-aware features.
# If None, the script tries to auto-discover one per method.
MORPH_BASIS_JSON: Optional[str] = None

# Optional filename stem of the baseline mesh, e.g. "corner" or "crm".
# If None, the script tries to infer it from morph_config*.json.
BASE_MESH_STEM: Optional[str] = None

# Common comparison space when methods use different control-node sets
# Options:
#   "union"   -> union of unique control nodes across methods
#   "largest" -> control nodes of the method with the largest control-node count
#   "<name>"  -> use control nodes from a specific method name, e.g. "xyz"
REFERENCE_MODE = "union"

# Interpolation settings for mapping each method's CN displacement field
# onto the common reference points
INTERP_METHOD = "idw"   # "idw" or "nearest"
IDW_K = 8
IDW_POWER = 2.0
UNIQUE_TOL = 1e-8

# Morph comparison metric groups.
# Lower-is-better metrics go in SMOOTHNESS_METRICS and ROBUSTNESS_LOWER_BETTER.
# Higher-is-better metrics go in ROBUSTNESS_HIGHER_BETTER.
SMOOTHNESS_METRICS = [
    "lap_energy",
    "strain_p50",
    "strain_p95",
    "strain_max",
]

ROBUSTNESS_LOWER_BETTER = [
    "tri_deg_frac_q1_lt_1e-6",
    "flipped_frac",
    "flipped_tri_frac",
    "u_anchor_max_ratio",
    "u_anchor_mean_ratio",
]

ROBUSTNESS_HIGHER_BETTER = [
    "tri_q1_p01",
    "tri_q1_min",
    "tri_q_ratio_p01",
    "tri_q_ratio_min",
    "edge_ratio_min",
    "edge_ratio_p01",
    "area_ratio_min",
    "area_ratio_p01",
]


# ============================================================
# Utilities
# ============================================================


def ensure_2d_xyz(arr: np.ndarray, name: str = "array") -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be (N,3), got {arr.shape}")
    return arr



def flatten_disp(arr: np.ndarray) -> np.ndarray:
    return ensure_2d_xyz(arr, "displacement").reshape(-1)



def relative_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))



def pca_spectrum(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    denom = max(X.shape[0] - 1, 1)
    return (S ** 2) / denom



def average_pairwise_distance(X: np.ndarray, max_pairs: int = 25000, seed: int = 0) -> float:
    n = X.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        vals = []
        for i in range(n):
            d = np.linalg.norm(X[i + 1:] - X[i], axis=1)
            vals.append(d)
        return float(np.concatenate(vals).mean())

    vals = []
    for _ in range(max_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        vals.append(np.linalg.norm(X[i] - X[j]))
    return float(np.mean(vals))



def build_basis(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if S.size == 0:
        return np.zeros((X.shape[1], 0), dtype=float)
    tol = 1e-10 * max(S[0], 1e-12)
    r = int(np.sum(S > tol))
    return Vt[:r].T



def project(u: np.ndarray, mu: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B.shape[1] == 0:
        return mu.copy()
    return mu + B @ (B.T @ (u - mu))



def unique_rows_tol(arr: np.ndarray, tol: float = UNIQUE_TOL) -> np.ndarray:
    arr = ensure_2d_xyz(arr, "points")
    if len(arr) == 0:
        return arr
    key = np.round(arr / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    return arr[idx]



def nearest_interp(src_pts: np.ndarray, src_disp: np.ndarray, qry_pts: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    src_pts = ensure_2d_xyz(src_pts, "src_pts")
    src_disp = ensure_2d_xyz(src_disp, "src_disp")
    qry_pts = ensure_2d_xyz(qry_pts, "qry_pts")

    nn = NearestNeighbors(n_neighbors=1).fit(src_pts)
    idx = nn.kneighbors(qry_pts, return_distance=False).reshape(-1)
    return src_disp[idx]



def idw_interp(src_pts: np.ndarray, src_disp: np.ndarray, qry_pts: np.ndarray,
               k: int = IDW_K, power: float = IDW_POWER) -> np.ndarray:
    """
    Inverse-distance-weight interpolation from source control nodes to
    common reference points.
    """
    from sklearn.neighbors import NearestNeighbors

    src_pts = ensure_2d_xyz(src_pts, "src_pts")
    src_disp = ensure_2d_xyz(src_disp, "src_disp")
    qry_pts = ensure_2d_xyz(qry_pts, "qry_pts")

    n_src = len(src_pts)
    k_eff = min(max(1, k), n_src)

    nn = NearestNeighbors(n_neighbors=k_eff).fit(src_pts)
    dists, idx = nn.kneighbors(qry_pts, return_distance=True)

    out = np.zeros((len(qry_pts), 3), dtype=float)

    exact = dists[:, 0] < 1e-14
    if np.any(exact):
        out[exact] = src_disp[idx[exact, 0]]

    nonexact = ~exact
    if np.any(nonexact):
        dd = dists[nonexact]
        ii = idx[nonexact]
        w = 1.0 / np.maximum(dd, 1e-12) ** power
        w /= w.sum(axis=1, keepdims=True)
        out[nonexact] = np.einsum("ij,ijk->ik", w, src_disp[ii])

    return out



def interpolate_displacement(src_pts: np.ndarray, src_disp: np.ndarray, qry_pts: np.ndarray) -> np.ndarray:
    if INTERP_METHOD == "nearest":
        return nearest_interp(src_pts, src_disp, qry_pts)
    if INTERP_METHOD == "idw":
        return idw_interp(src_pts, src_disp, qry_pts, k=IDW_K, power=IDW_POWER)
    raise ValueError(f"Unknown INTERP_METHOD={INTERP_METHOD}")



def load_features_module():
    import os
    import sys
    import importlib.util

    this_dir = os.path.dirname(os.path.abspath(__file__))   # .../ShapeParameterization
    project_root = os.path.dirname(this_dir)                # .../Aeropt2
    mfc_dir = os.path.join(project_root, "MeshFailureClassifier")
    path = os.path.join(mfc_dir, "features.py")

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if not os.path.exists(path):
        raise FileNotFoundError(f"features.py not found: {path}")

    spec = importlib.util.spec_from_file_location("_param_compare_features", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import features.py from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def stem_without_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]



def parse_case_number_from_name(name: str) -> Optional[int]:
    stem = stem_without_ext(name)
    m = re.search(r"(?:^|[_-])(\d+)$", stem)
    return int(m.group(1)) if m else None



def parse_case_number_from_cfg(cfg: Dict, fallback_name: str) -> Optional[int]:
    for key in ("n", "case", "case_idx", "index"):
        val = cfg.get(key, None)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, str) and val.strip().isdigit():
            return int(val.strip())
    return parse_case_number_from_name(fallback_name)



def infer_base_mesh_stem(cfg: Dict, folder: str) -> Optional[str]:
    if BASE_MESH_STEM:
        return BASE_MESH_STEM

    vtk_name = cfg.get("vtk_name", None)
    if isinstance(vtk_name, str) and vtk_name.strip():
        return stem_without_ext(vtk_name)

    fros = sorted(glob.glob(os.path.join(folder, "*.fro")))
    indexed = []
    plain = []
    for fp in fros:
        stem = stem_without_ext(fp)
        if re.search(r"(?:^|[_-])\d+$", stem):
            indexed.append(fp)
        else:
            plain.append(fp)
    if plain:
        return stem_without_ext(plain[0])
    return None



def guess_orig_fro(method_folder_path: str, cfg: Dict) -> str:
    candidates: List[str] = []
    if ORIG_FRO:
        candidates.append(ORIG_FRO)

    stem = infer_base_mesh_stem(cfg, method_folder_path)
    if stem:
        candidates.extend([
            os.path.join(method_folder_path, f"{stem}.fro"),
            os.path.join(MAIN_DIR, "orig", f"{stem}.fro"),
            os.path.join(MAIN_DIR, "orig", stem, f"{stem}.fro"),
            os.path.join(os.path.dirname(method_folder_path), "orig", f"{stem}.fro"),
            os.path.join(os.path.dirname(os.path.dirname(method_folder_path)), "orig", f"{stem}.fro"),
        ])

    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand

    msg = "\n".join(f"  - {c}" for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate the original/baseline .fro mesh. "
        "Set ORIG_FRO explicitly near the top of the script.\n"
        f"Tried:\n{msg if msg else '  (no candidates generated)'}"
    )



def guess_basis_json(method_folder_path: str) -> Optional[str]:
    candidates: List[str] = []
    if MORPH_BASIS_JSON:
        candidates.append(MORPH_BASIS_JSON)

    candidates.extend([
        os.path.join(method_folder_path, "morph_basis.json"),
        os.path.join(os.path.dirname(method_folder_path), "morph_basis.json"),
        os.path.join(os.path.dirname(os.path.dirname(method_folder_path)), "morph_basis.json"),
        os.path.join(MAIN_DIR, "morph_basis.json"),
    ])

    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None



def guess_morphed_fro_path(folder: str, cfg: Dict, config_path: str) -> str:
    case_num = parse_case_number_from_cfg(cfg, os.path.basename(config_path))
    stem = infer_base_mesh_stem(cfg, folder)

    candidates: List[str] = []
    if stem and case_num is not None:
        candidates.extend([
            os.path.join(folder, f"{stem}_{case_num}.fro"),
            os.path.join(folder, f"{stem}-{case_num}.fro"),
        ])

    if case_num is not None:
        candidates.extend(sorted(glob.glob(os.path.join(folder, f"*_{case_num}.fro"))))
        candidates.extend(sorted(glob.glob(os.path.join(folder, f"*-{case_num}.fro"))))

    stem_cfg = stem_without_ext(config_path)
    digits = parse_case_number_from_name(stem_cfg)
    if digits is not None:
        candidates.extend(sorted(glob.glob(os.path.join(folder, f"*_{digits}.fro"))))

    seen = set()
    unique_candidates = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            unique_candidates.append(cand)

    for cand in unique_candidates:
        if os.path.exists(cand):
            return cand

    msg = "\n".join(f"  - {c}" for c in unique_candidates if c)
    raise FileNotFoundError(
        f"Could not find morphed .fro for config '{config_path}'.\n"
        f"Tried:\n{msg if msg else '  (no candidates generated)'}"
    )



def aggregate_metric_columns(df: pd.DataFrame, group_cols: List[str], metrics: List[str], prefix: str) -> pd.DataFrame:
    available = [m for m in metrics if m in df.columns]
    if not available:
        return pd.DataFrame(index=sorted(df[group_cols[0]].unique()))

    agg = df.groupby(group_cols)[available].agg(["mean", "median", "std"])
    agg.columns = [f"{prefix}{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()



def minmax_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    xmin = np.nanmin(x.values)
    xmax = np.nanmax(x.values)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return pd.Series(np.nan, index=series.index)
    if abs(xmax - xmin) < 1e-15:
        return pd.Series(1.0, index=series.index)
    score = (x - xmin) / (xmax - xmin)
    return score if higher_is_better else (1.0 - score)


# ============================================================
# Loading
# ============================================================


def method_folder(method: MethodConfig) -> str:
    return os.path.join(MAIN_DIR, method.folder, "surfaces", f"n_{CASE}")



def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)



def load_method_raw(method: MethodConfig) -> Dict:
    folder = method_folder(method)
    print(f"\n[INFO] Loading method: {method.name}")
    print(f"[INFO] Folder: {folder}")

    files = sorted(glob.glob(os.path.join(folder, "morph_config*.json")))
    if not files:
        raise FileNotFoundError(f"No configs found in {folder}")
    print(f"[INFO] Found {len(files)} configs")

    control_nodes_ref = None
    case_names = []
    disps = []

    for fp in files:
        cfg = load_config(fp)

        if "control_nodes" not in cfg:
            raise KeyError(f"No 'control_nodes' in {fp}")
        if "displacement_vector" not in cfg:
            raise KeyError(f"No 'displacement_vector' in {fp}")

        cn = ensure_2d_xyz(np.asarray(cfg["control_nodes"], dtype=float), "control_nodes")
        disp = ensure_2d_xyz(np.asarray(cfg["displacement_vector"], dtype=float), "displacement_vector")

        if len(cn) != len(disp):
            raise ValueError(f"control_nodes and displacement_vector length mismatch in {fp}")

        if control_nodes_ref is None:
            control_nodes_ref = cn
        else:
            if cn.shape != control_nodes_ref.shape or not np.allclose(cn, control_nodes_ref, atol=1e-10, rtol=0.0):
                raise ValueError(
                    f"Method '{method.name}' has varying control-node sets across cases. "
                    f"This script assumes one fixed CN set per method."
                )

        disps.append(disp)
        case_names.append(os.path.basename(fp))

    print(f"[INFO] Control nodes: {control_nodes_ref.shape[0]}")
    return {
        "name": method.name,
        "folder": folder,
        "files": files,
        "case_names": case_names,
        "control_nodes": control_nodes_ref,
        "disps_native": disps,
        "n_samples": len(disps),
        "n_nodes_native": control_nodes_ref.shape[0],
        "dof": method.dof,
    }



def choose_reference_points(raw_data: Dict[str, Dict]) -> Tuple[np.ndarray, str]:
    if REFERENCE_MODE == "union":
        pts = np.vstack([d["control_nodes"] for d in raw_data.values()])
        ref = unique_rows_tol(pts, tol=UNIQUE_TOL)
        return ref, f"union({len(ref)})"

    if REFERENCE_MODE == "largest":
        key = max(raw_data.keys(), key=lambda k: raw_data[k]["n_nodes_native"])
        return raw_data[key]["control_nodes"].copy(), f"largest={key}"

    if REFERENCE_MODE in raw_data:
        return raw_data[REFERENCE_MODE]["control_nodes"].copy(), f"method={REFERENCE_MODE}"

    raise ValueError(f"Unknown REFERENCE_MODE={REFERENCE_MODE}")



def reproject_to_common_space(raw: Dict, ref_points: np.ndarray) -> Dict:
    src_pts = raw["control_nodes"]
    X = []
    for disp in raw["disps_native"]:
        disp_ref = interpolate_displacement(src_pts, disp, ref_points)
        X.append(flatten_disp(disp_ref))
    X = np.vstack(X)

    return {
        "name": raw["name"],
        "X": X,
        "n_samples": raw["n_samples"],
        "n_nodes_native": raw["n_nodes_native"],
        "n_nodes_ref": ref_points.shape[0],
        "dof": raw["dof"],
    }



def load_morph_feature_records(method: MethodConfig, features_module) -> pd.DataFrame:
    folder = method_folder(method)
    files = sorted(glob.glob(os.path.join(folder, "morph_config*.json")))
    if not files:
        raise FileNotFoundError(f"No configs found in {folder}")

    first_cfg = load_config(files[0])
    orig_fro = guess_orig_fro(folder, first_cfg)
    basis_json = guess_basis_json(folder)

    print(f"\n[INFO] Morph features for method: {method.name}")
    print(f"[INFO] Baseline .fro : {orig_fro}")
    print(f"[INFO] Basis json    : {basis_json if basis_json else '(none)'}")

    rows = []
    for fp in files:
        cfg = load_config(fp)
        morphed_fro = guess_morphed_fro_path(folder, cfg, fp)
        case_num = parse_case_number_from_cfg(cfg, os.path.basename(fp))
        rec = features_module.compute_features(
            orig_fro=orig_fro,
            morphed_fro=morphed_fro,
            run_dir=folder,
            case_id=str(case_num) if case_num is not None else stem_without_ext(fp),
            morph_basis_json=basis_json,
            extra={
                "method": method.name,
                "config_file": os.path.basename(fp),
                "orig_fro": orig_fro,
                "basis_json": basis_json or "",
            },
        )
        flat = {
            "method": method.name,
            "config_file": os.path.basename(fp),
            "morphed_fro": rec.get("morphed_fro", morphed_fro),
            "case_id": rec.get("case_id", ""),
            "dof": method.dof,
            "n_features": 0,
        }
        feats = rec.get("features", {}) or {}
        flat.update(feats)
        flat["n_features"] = len(feats)
        rows.append(flat)

    return pd.DataFrame(rows)


# ============================================================
# Metrics
# ============================================================


def compute_metrics(data: Dict) -> Dict:
    X = data["X"]
    eig = pca_spectrum(X)
    return {
        "method": data["name"],
        "dof": data["dof"],
        "n_nodes_native": data["n_nodes_native"],
        "n_nodes_ref": data["n_nodes_ref"],
        "trace_cov": float(np.sum(eig)),
        "avg_dist": float(average_pairwise_distance(X)),
        "first_pc": float(eig[0]) if len(eig) else 0.0,
    }



def cross_representability(all_data: Dict[str, Dict]) -> pd.DataFrame:
    names = list(all_data.keys())
    mean_err = pd.DataFrame(index=names, columns=names, dtype=float)

    bases = {}
    means = {}
    for n in names:
        X = all_data[n]["X"]
        bases[n] = build_basis(X)
        means[n] = X.mean(axis=0)

    for A in names:
        XA = all_data[A]["X"]
        for B in names:
            errs = []
            for u in XA:
                u_hat = project(u, means[B], bases[B])
                errs.append(relative_l2_error(u, u_hat))
            mean_err.loc[A, B] = float(np.mean(errs))
    return mean_err



def summarise_morph_metrics(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    smooth_avail = [m for m in SMOOTHNESS_METRICS if m in df_features.columns]
    robust_low_avail = [m for m in ROBUSTNESS_LOWER_BETTER if m in df_features.columns]
    robust_high_avail = [m for m in ROBUSTNESS_HIGHER_BETTER if m in df_features.columns]

    summary_rows = []
    methods = list(df_features["method"].drop_duplicates())
    for method in methods:
        sub = df_features[df_features["method"] == method].copy()
        row = {
            "method": method,
            "dof": float(sub["dof"].iloc[0]) if "dof" in sub.columns else np.nan,
            "n_cases": int(len(sub)),
        }

        for metric in smooth_avail + robust_low_avail + robust_high_avail:
            row[f"{metric}_mean"] = float(pd.to_numeric(sub[metric], errors="coerce").mean())
            row[f"{metric}_median"] = float(pd.to_numeric(sub[metric], errors="coerce").median())
            row[f"{metric}_std"] = float(pd.to_numeric(sub[metric], errors="coerce").std())

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        smooth_score_cols = []
        robust_score_cols = []

        for metric in smooth_avail:
            col = f"{metric}_mean"
            score_col = f"{metric}_score"
            summary[score_col] = minmax_score(summary[col], higher_is_better=False)
            smooth_score_cols.append(score_col)

        for metric in robust_low_avail:
            col = f"{metric}_mean"
            score_col = f"{metric}_score"
            summary[score_col] = minmax_score(summary[col], higher_is_better=False)
            robust_score_cols.append(score_col)

        for metric in robust_high_avail:
            col = f"{metric}_mean"
            score_col = f"{metric}_score"
            summary[score_col] = minmax_score(summary[col], higher_is_better=True)
            robust_score_cols.append(score_col)

        summary["smoothness_score"] = summary[smooth_score_cols].mean(axis=1) if smooth_score_cols else np.nan
        summary["robustness_score"] = summary[robust_score_cols].mean(axis=1) if robust_score_cols else np.nan
    else:
        summary["smoothness_score"] = []
        summary["robustness_score"] = []

    smooth_cols = ["method", "dof", "n_cases"] + [c for c in summary.columns if c.startswith(tuple(m + "_" for m in smooth_avail))] + ["smoothness_score"]
    robust_cols = ["method", "dof", "n_cases"] + [c for c in summary.columns if c.startswith(tuple(m + "_" for m in robust_low_avail + robust_high_avail))] + ["robustness_score"]

    smooth_summary = summary.loc[:, [c for c in smooth_cols if c in summary.columns]].copy()
    robust_summary = summary.loc[:, [c for c in robust_cols if c in summary.columns]].copy()
    return summary, smooth_summary, robust_summary


# ============================================================
# Visualisation
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_param_method_pareto(
    shape_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    smoothness_df: pd.DataFrame,
    outdir: str,
    shape_col: str = "trace_cov",
    robustness_col: str = "robustness_score",
    smoothness_col: str = "smoothness_score",
    filename: str = "pareto_tradeoff.png",
):
    """
    Create a 3-metric trade-off plot:
        x-axis   = shape-space size
        y-axis   = robustness score
        colour   = smoothness score
        size     = smoothness score (for emphasis)

    Expected input dataframes:
        shape_df      : columns ['method', shape_col]
        robustness_df : columns ['method', robustness_col]
        smoothness_df : columns ['method', smoothness_col]
    """

    # -----------------------------
    # Merge inputs
    # -----------------------------
    df = shape_df[["method", shape_col]].merge(
        robustness_df[["method", robustness_col]],
        on="method",
        how="inner"
    ).merge(
        smoothness_df[["method", smoothness_col]],
        on="method",
        how="inner"
    )

    if df.empty:
        raise ValueError("Merged dataframe is empty. Check method names and column names.")

    # -----------------------------
    # Bubble size from smoothness
    # -----------------------------
    smin = df[smoothness_col].min()
    smax = df[smoothness_col].max()

    if np.isclose(smin, smax):
        sizes = np.full(len(df), 500.0)
    else:
        sizes = 300 + 900 * (df[smoothness_col] - smin) / (smax - smin)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(
        df[shape_col],
        df[robustness_col],
        c=df[smoothness_col],
        s=sizes,
        alpha=0.85,
        edgecolors="black",
        linewidths=1.0,
    )

    # Annotate each method
    for _, row in df.iterrows():
        ax.annotate(
            row["method"],
            (row[shape_col], row[robustness_col]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11
        )

    # Colour bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Smoothness score")

    # Axes labels / title
    ax.set_xlabel("Shape-space size")
    ax.set_ylabel("Robustness score")
    ax.set_title("Parameterisation trade-off plot")

    # -----------------------------
    # Optional 'better direction' guide
    # -----------------------------
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    ax.annotate(
        "better",
        xy=(x1, y1),
        xytext=(x1 - 0.15 * (x1 - x0), y1 - 0.08 * (y1 - y0)),
        arrowprops=dict(arrowstyle="->", lw=2),
        fontsize=12,
        ha="center"
    )

    ax.grid(True, alpha=0.3)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return df, outpath


def plot_heatmap(df: pd.DataFrame, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(df.values.astype(float), cmap="magma_r")
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center")

    ax.set_title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()



def plot_bar(df: pd.DataFrame, col: str, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["method"], df[col])
    ax.set_title(title)
    ax.set_ylabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()



def plot_pca_scatter(all_data: Dict[str, Dict], path: str) -> None:
    from sklearn.decomposition import PCA

    plt.figure(figsize=(7, 6))
    for name, data in all_data.items():
        X = data["X"]
        X2 = PCA(n_components=2).fit_transform(X)
        plt.scatter(X2[:, 0], X2[:, 1], label=name, alpha=0.5)

    plt.legend()
    plt.title("Shape-space PCA projection (common reference points)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()



def plot_metric_bars_from_summary(summary: pd.DataFrame, metrics: List[str], path_prefix: str, title_prefix: str) -> None:
    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in summary.columns:
            continue
        plot_bar(summary, mean_col, f"{title_prefix}: {metric}", f"{path_prefix}_{metric}.png")


# ============================================================
# Main
# ============================================================


def main():
    print("\n==============================")
    print("Loading Raw Data")
    print("==============================")

    raw_data = {}
    for m in METHODS:
        raw_data[m.name] = load_method_raw(m)

    ref_points, ref_desc = choose_reference_points(raw_data)
    print("\n==============================")
    print("Common Comparison Space")
    print("==============================")
    print(f"[INFO] Reference mode : {REFERENCE_MODE}")
    print(f"[INFO] Reference desc : {ref_desc}")
    print(f"[INFO] Reference nodes: {len(ref_points)}")

    all_data = {}
    metrics = []
    for name, raw in raw_data.items():
        data = reproject_to_common_space(raw, ref_points)
        all_data[name] = data
        metrics.append(compute_metrics(data))
        print(
            f"[INFO] {name}: native_nodes={data['n_nodes_native']}, "
            f"ref_nodes={data['n_nodes_ref']}, samples={data['n_samples']}"
        )

    df_metrics = pd.DataFrame(metrics)

    print("\n==============================")
    print("Metrics")
    print("==============================")
    print(df_metrics)

    print("\n==============================")
    print("Cross Representability")
    print("==============================")
    repr_df = cross_representability(all_data)
    print(repr_df)

    plot_bar(df_metrics, "trace_cov", "Shape-space size", os.path.join(OUT_DIR, "trace.png"))
    plot_bar(df_metrics, "avg_dist", "Average shape distance", os.path.join(OUT_DIR, "dist.png"))
    plot_heatmap(repr_df, "Cross-method representability", os.path.join(OUT_DIR, "repr.png"))
    plot_pca_scatter(all_data, os.path.join(OUT_DIR, "pca.png"))

    df_metrics.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    repr_df.to_csv(os.path.join(OUT_DIR, "representability.csv"))
    np.save(os.path.join(OUT_DIR, "reference_points.npy"), ref_points)

    with open(os.path.join(OUT_DIR, "comparison_space_info.txt"), "w") as f:
        f.write(f"REFERENCE_MODE = {REFERENCE_MODE}\n")
        f.write(f"REFERENCE_DESC = {ref_desc}\n")
        f.write(f"REFERENCE_NODES = {len(ref_points)}\n")
        f.write(f"INTERP_METHOD = {INTERP_METHOD}\n")
        f.write(f"IDW_K = {IDW_K}\n")
        f.write(f"IDW_POWER = {IDW_POWER}\n")
        f.write(f"COMPARE_MORPH = {COMPARE_MORPH}\n")
        f.write(f"ORIG_FRO = {ORIG_FRO}\n")
        f.write(f"MORPH_BASIS_JSON = {MORPH_BASIS_JSON}\n")

    if COMPARE_MORPH:
        print("\n==============================")
        print("Morph Smoothness / Robustness")
        print("==============================")
        features_module = load_features_module()

        feat_frames = []
        for m in METHODS:
            feat_frames.append(load_morph_feature_records(m, features_module))

        df_features = pd.concat(feat_frames, ignore_index=True)
        summary, smooth_summary, robust_summary = summarise_morph_metrics(df_features)

        print("\n[INFO] Available smoothness metrics:", [m for m in SMOOTHNESS_METRICS if m in df_features.columns])
        print("[INFO] Available robustness metrics (lower better):", [m for m in ROBUSTNESS_LOWER_BETTER if m in df_features.columns])
        print("[INFO] Available robustness metrics (higher better):", [m for m in ROBUSTNESS_HIGHER_BETTER if m in df_features.columns])

        print("\n[INFO] Smoothness summary")
        print(smooth_summary)
        print("\n[INFO] Robustness summary")
        print(robust_summary)

        df_features.to_csv(os.path.join(OUT_DIR, "morph_features_all_cases.csv"), index=False)
        summary.to_csv(os.path.join(OUT_DIR, "morph_feature_summary.csv"), index=False)
        smooth_summary.to_csv(os.path.join(OUT_DIR, "smoothness_summary.csv"), index=False)
        robust_summary.to_csv(os.path.join(OUT_DIR, "robustness_summary.csv"), index=False)

        if "smoothness_score" in summary.columns:
            plot_bar(summary, "smoothness_score", "Smoothness score", os.path.join(OUT_DIR, "smoothness_score.png"))
        if "robustness_score" in summary.columns:
            plot_bar(summary, "robustness_score", "Robustness score", os.path.join(OUT_DIR, "robustness_score.png"))

        plot_metric_bars_from_summary(summary, [m for m in SMOOTHNESS_METRICS if f"{m}_mean" in summary.columns], os.path.join(OUT_DIR, "smoothness"), "Smoothness")
        plot_metric_bars_from_summary(
            summary,
            [m for m in (ROBUSTNESS_LOWER_BETTER + ROBUSTNESS_HIGHER_BETTER) if f"{m}_mean" in summary.columns],
            os.path.join(OUT_DIR, "robustness"),
            "Robustness",
        )
        
        # ============================================
        # Pareto trade-off plot
        # ============================================
        try:
            if "robustness_score" in summary.columns and "smoothness_score" in summary.columns:

                pareto_df, pareto_path = plot_param_method_pareto(
                    shape_df=df_metrics,
                    robustness_df=summary,
                    smoothness_df=summary,
                    outdir=OUT_DIR,
                    shape_col="trace_cov",
                    robustness_col="robustness_score",
                    smoothness_col="smoothness_score",
                )

                print("\n[INFO] Pareto trade-off plot saved to:", pareto_path)
                print("\n[INFO] Pareto data:")
                print(pareto_df)

        except Exception as e:
            print("[WARNING] Failed to generate Pareto plot:", e)

    print(f"\nSaved results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
