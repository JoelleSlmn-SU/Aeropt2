#!/usr/bin/env python3
"""
response_surface_analysis.py

Preliminary regional sensitivity analysis for AerOpt.

Purpose
-------
After a preliminary DOE has been run using one representative control node per
region, this script:

1. reads the preliminary design matrix X and objective values Y,
2. fits a response surface model,
3. ranks regions by sensitivity/importance,
4. keeps the most important regions,
5. distributes the final control nodes inside those selected regions using
   farthest-point sampling,
6. recomputes normals at the final control nodes,
7. writes an updated morph_basis.json and, optionally, bo_settings.json.

Expected prelim data formats
----------------------------
Preferred JSON:
    prelim_results.json
    {
      "X": [[...], [...]],
      "Y": [0.1, 0.2, ...]
    }

Alternative CSV:
    prelim_results.csv
    x0,x1,x2,...,Y
    ...

Required morph_basis fields
---------------------------
The input morph_basis.json should contain at least:
    control_nodes              # representative region centres, shape (R, 3)
    control_normals            # optional, shape (R, 3)
    prelim_candidate_nodes     # candidate nodes over T surface, shape (Ncand, 3)

Recommended prelim fields:
    prelim_enabled
    prelim_regions
    prelim_keep_fraction
    prelim_final_control_nodes
    prelim_region_radius

If prelim_region_radius is not provided, the script estimates one from nearest
region-centre spacing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ResponseSurfaceResult:
    model: str
    intercept: float
    coefficients: np.ndarray
    scores: np.ndarray
    selected_regions: np.ndarray
    r2: float


def _as_array(data, name: str, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {arr.shape}")
    return arr


def load_prelim_results(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load DOE matrix X and objective vector Y from JSON or CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prelim results not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        X = _as_array(data["X"], "X", ndim=2)
        Y = _as_array(data["Y"], "Y").reshape(-1)
        return X, Y

    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError(f"CSV is empty: {path}")

        cols = reader.fieldnames or []
        y_col = "Y" if "Y" in cols else cols[-1]
        x_cols = [c for c in cols if c != y_col]
        X = np.array([[float(row[c]) for c in x_cols] for row in rows], dtype=float)
        Y = np.array([float(row[y_col]) for row in rows], dtype=float)
        return X, Y

    raise ValueError(f"Unsupported prelim result format: {ext}. Use .json or .csv")


def fit_response_surface(X: np.ndarray, Y: np.ndarray, model: str = "linear") -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Fit response surface.

    For model='linear':
        Y = b0 + sum_i b_i x_i

    For model='quadratic_diag':
        Y = b0 + sum_i b_i x_i + sum_i q_i x_i^2

    Region score is based on absolute standardised coefficients.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.shape}")
    if len(Y) != X.shape[0]:
        raise ValueError(f"len(Y)={len(Y)} does not match X rows={X.shape[0]}")

    # Standardise X for fair coefficient comparison.
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    Xs = (X - x_mean) / x_std

    if model == "linear":
        A = np.column_stack([np.ones(Xs.shape[0]), Xs])
        region_terms = slice(1, 1 + X.shape[1])
    elif model == "quadratic_diag":
        A = np.column_stack([np.ones(Xs.shape[0]), Xs, Xs**2])
        region_terms = slice(1, 1 + X.shape[1])
    else:
        raise ValueError("model must be 'linear' or 'quadratic_diag'")

    coef_all, *_ = np.linalg.lstsq(A, Y, rcond=None)
    Y_hat = A @ coef_all

    ss_res = float(np.sum((Y - Y_hat) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

    linear_coeffs = coef_all[region_terms]

    if model == "quadratic_diag":
        quad_coeffs = coef_all[1 + X.shape[1]:]
        scores = np.abs(linear_coeffs) + 0.5 * np.abs(quad_coeffs)
    else:
        scores = np.abs(linear_coeffs)

    return float(coef_all[0]), np.asarray(coef_all), scores, r2


def estimate_region_radius(region_centres: np.ndarray, scale: float = 0.75) -> float:
    """Estimate region radius from nearest centre spacing."""
    C = np.asarray(region_centres, dtype=float)
    if len(C) <= 1:
        return 1.0

    d2 = np.sum((C[:, None, :] - C[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nearest = np.sqrt(np.min(d2, axis=1))
    return float(scale * np.median(nearest))


def assign_candidates_to_regions(
    candidate_nodes: np.ndarray,
    region_centres: np.ndarray,
    region_radius: Optional[float] = None,
) -> Dict[int, np.ndarray]:
    """
    Assign candidate nodes to nearest region centre.

    If region_radius is provided, only nodes within that radius are included.
    Otherwise every candidate belongs to its nearest region.
    """
    P = np.asarray(candidate_nodes, dtype=float)
    C = np.asarray(region_centres, dtype=float)

    d2 = np.sum((P[:, None, :] - C[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1)
    nearest_dist = np.sqrt(np.min(d2, axis=1))

    region_map: Dict[int, List[int]] = {i: [] for i in range(len(C))}
    for idx, (rid, dist) in enumerate(zip(nearest, nearest_dist)):
        if region_radius is None or dist <= region_radius:
            region_map[int(rid)].append(idx)

    return {rid: np.asarray(ids, dtype=int) for rid, ids in region_map.items()}


def farthest_point_sample(points: np.ndarray, n_select: int, seed_point: Optional[np.ndarray] = None) -> np.ndarray:
    """Return indices of n_select farthest-point samples from points."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n == 0:
        return np.array([], dtype=int)
    if n_select >= n:
        return np.arange(n, dtype=int)

    selected = []
    if seed_point is None:
        first = int(np.argmin(np.sum((P - P.mean(axis=0)) ** 2, axis=1)))
    else:
        first = int(np.argmin(np.sum((P - np.asarray(seed_point, dtype=float)) ** 2, axis=1)))
    selected.append(first)

    min_d2 = np.sum((P - P[first]) ** 2, axis=1)
    for _ in range(1, n_select):
        nxt = int(np.argmax(min_d2))
        selected.append(nxt)
        d2 = np.sum((P - P[nxt]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    return np.asarray(selected, dtype=int)


def estimate_normals_from_candidates(
    control_nodes: np.ndarray,
    candidate_nodes: np.ndarray,
    candidate_normals: Optional[np.ndarray] = None,
    k: int = 12,
) -> np.ndarray:
    """
    Estimate final control-node normals.

    If candidate_normals are provided, average nearest candidate normals.
    Otherwise estimate normals using local PCA on candidate_nodes.
    """
    from sklearn.neighbors import NearestNeighbors

    C = np.asarray(control_nodes, dtype=float)
    P = np.asarray(candidate_nodes, dtype=float)
    kk = min(max(3, int(k)), max(1, len(P)))

    if candidate_normals is None:
        # PCA normals at candidate nodes.
        nn = NearestNeighbors(n_neighbors=kk).fit(P)
        idx = nn.kneighbors(P, return_distance=False)
        N = np.zeros_like(P)
        for i, row in enumerate(idx):
            Q = P[row] - P[row].mean(axis=0, keepdims=True)
            cov = Q.T @ Q
            _, V = np.linalg.eigh(cov)
            n = V[:, 0]
            n /= np.linalg.norm(n) + 1e-12
            N[i] = n
        centroid = P.mean(axis=0)
        s = np.sign(np.sum((P - centroid) * N, axis=1))
        s[s == 0] = 1.0
        candidate_normals = N * s[:, None]
    else:
        candidate_normals = np.asarray(candidate_normals, dtype=float)

    nn = NearestNeighbors(n_neighbors=kk).fit(P)
    idx = nn.kneighbors(C, return_distance=False)
    out = []
    for row in idx:
        n = candidate_normals[row].mean(axis=0)
        n /= np.linalg.norm(n) + 1e-12
        out.append(n)
    return np.asarray(out, dtype=float)


def select_final_control_nodes(
    candidate_nodes: np.ndarray,
    region_centres: np.ndarray,
    selected_regions: np.ndarray,
    final_n: int,
    region_radius: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select final control nodes from selected regions using farthest-point sampling.

    Returns:
        final_nodes, final_candidate_indices
    """
    P = np.asarray(candidate_nodes, dtype=float)
    C = np.asarray(region_centres, dtype=float)
    selected_regions = np.asarray(selected_regions, dtype=int)

    region_map = assign_candidates_to_regions(P, C, region_radius=region_radius)

    selected_candidate_ids = []
    for rid in selected_regions:
        ids = region_map.get(int(rid), np.array([], dtype=int))
        selected_candidate_ids.extend(ids.tolist())

    selected_candidate_ids = np.unique(np.asarray(selected_candidate_ids, dtype=int))

    if selected_candidate_ids.size == 0:
        raise RuntimeError(
            "No candidate nodes found inside selected regions. Try increasing prelim_region_radius."
        )

    region_points = P[selected_candidate_ids]
    sampled_local = farthest_point_sample(region_points, int(final_n))
    final_candidate_ids = selected_candidate_ids[sampled_local]
    final_nodes = P[final_candidate_ids]
    return final_nodes, final_candidate_ids


def update_bo_settings(bo_path: str, n_dim: int, output_path: Optional[str] = None) -> Optional[str]:
    """Update BO dimensionality and default bounds in bo_settings.json."""
    if not bo_path:
        return None
    if not os.path.exists(bo_path):
        raise FileNotFoundError(f"bo_settings.json not found: {bo_path}")

    with open(bo_path, "r", encoding="utf-8") as f:
        bo = json.load(f)

    bo["n_dim"] = int(n_dim)
    old_lb = np.asarray(bo.get("lb", [-1.0] * n_dim), dtype=float).reshape(-1)
    old_ub = np.asarray(bo.get("ub", [1.0] * n_dim), dtype=float).reshape(-1)

    lb_default = float(old_lb[0]) if old_lb.size else -1.0
    ub_default = float(old_ub[0]) if old_ub.size else 1.0
    bo["lb"] = [lb_default] * int(n_dim)
    bo["ub"] = [ub_default] * int(n_dim)

    out = output_path or bo_path
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bo, f, indent=2)
    return out


def run_analysis(
    prelim_results: str,
    morph_basis: str,
    output_morph_basis: str,
    bo_settings: Optional[str] = None,
    output_bo_settings: Optional[str] = None,
    model: str = "linear",
    keep_fraction: Optional[float] = None,
    keep_regions: Optional[int] = None,
    final_control_nodes: Optional[int] = None,
    region_radius: Optional[float] = None,
) -> ResponseSurfaceResult:
    X, Y = load_prelim_results(prelim_results)

    with open(morph_basis, "r", encoding="utf-8") as f:
        basis = json.load(f)

    region_centres = _as_array(basis.get("control_nodes"), "control_nodes", ndim=2)
    R = region_centres.shape[0]

    if X.shape[1] != R:
        raise ValueError(
            f"DOE dimension ({X.shape[1]}) must match number of prelim regions/control nodes ({R})."
        )

    candidate_nodes_data = basis.get("prelim_candidate_nodes", None)
    if candidate_nodes_data is None:
        # Fallback: allow user to pass candidate nodes as all_t_nodes for early testing.
        candidate_nodes_data = basis.get("all_t_nodes", None)
    if candidate_nodes_data is None:
        raise KeyError(
            "morph_basis.json must contain 'prelim_candidate_nodes' containing candidate nodes over the T surface."
        )
    candidate_nodes = _as_array(candidate_nodes_data, "prelim_candidate_nodes", ndim=2)

    candidate_normals = basis.get("prelim_candidate_normals", None)
    if candidate_normals is not None:
        candidate_normals = _as_array(candidate_normals, "prelim_candidate_normals", ndim=2)

    intercept, coef_all, scores, r2 = fit_response_surface(X, Y, model=model)

    if keep_regions is None:
        if keep_fraction is None:
            keep_fraction = float(basis.get("prelim_keep_fraction", 2.0 / 3.0))
        keep_regions = int(np.ceil(float(keep_fraction) * R))
    keep_regions = max(1, min(int(keep_regions), R))

    selected_regions = np.argsort(scores)[::-1][:keep_regions]
    selected_regions = np.sort(selected_regions)

    if final_control_nodes is None:
        final_control_nodes = int(basis.get("prelim_final_control_nodes", max(keep_regions, R)))

    if region_radius is None:
        region_radius = basis.get("prelim_region_radius", None)
    if region_radius is None:
        region_radius = estimate_region_radius(region_centres)

    final_nodes, final_candidate_ids = select_final_control_nodes(
        candidate_nodes=candidate_nodes,
        region_centres=region_centres,
        selected_regions=selected_regions,
        final_n=int(final_control_nodes),
        region_radius=float(region_radius),
    )

    final_normals = estimate_normals_from_candidates(
        control_nodes=final_nodes,
        candidate_nodes=candidate_nodes,
        candidate_normals=candidate_normals,
        k=12,
    )

    # Update basis for normal optimisation.
    updated = dict(basis)
    updated["prelim_completed"] = True
    updated["prelim_model"] = model
    updated["prelim_r2"] = float(r2)
    updated["prelim_region_scores"] = scores.tolist()
    updated["prelim_selected_regions"] = selected_regions.astype(int).tolist()
    updated["prelim_region_radius"] = float(region_radius)
    updated["prelim_final_candidate_ids"] = final_candidate_ids.astype(int).tolist()

    updated["control_nodes"] = final_nodes.tolist()
    updated["control_normals"] = final_normals.tolist()

    # Important: after prelim, BO should use the final number of modes/control nodes.
    old_k = int(updated.get("k_modes", min(6, len(final_nodes))))
    updated["k_modes"] = int(min(old_k, max(1, len(final_nodes) - 1))) if len(final_nodes) > 1 else 1

    os.makedirs(os.path.dirname(os.path.abspath(output_morph_basis)), exist_ok=True)
    with open(output_morph_basis, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)

    if bo_settings:
        # Design dimension is k for normal projection, 3k for local-frame vector mode.
        normal_project = bool(updated.get("normal_project", True))
        k_modes = int(updated["k_modes"])
        n_dim = k_modes if normal_project else 3 * k_modes
        update_bo_settings(bo_settings, n_dim=n_dim, output_path=output_bo_settings)

    summary_path = os.path.join(os.path.dirname(os.path.abspath(output_morph_basis)), "prelim_response_surface_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "intercept": intercept,
                "coefficients": coef_all.tolist(),
                "scores": scores.tolist(),
                "selected_regions": selected_regions.astype(int).tolist(),
                "r2": float(r2),
                "region_radius": float(region_radius),
                "final_control_nodes": int(len(final_nodes)),
                "output_morph_basis": os.path.abspath(output_morph_basis),
                "output_bo_settings": os.path.abspath(output_bo_settings) if output_bo_settings else None,
            },
            f,
            indent=2,
        )

    return ResponseSurfaceResult(
        model=model,
        intercept=intercept,
        coefficients=coef_all,
        scores=scores,
        selected_regions=selected_regions,
        r2=float(r2),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="AerOpt preliminary response-surface analysis")
    ap.add_argument("--prelim-results", required=True, help="Path to prelim_results.json or prelim_results.csv")
    ap.add_argument("--morph-basis", required=True, help="Input morph_basis.json")
    ap.add_argument("--output-morph-basis", required=True, help="Output updated morph_basis.json")
    ap.add_argument("--bo-settings", default=None, help="Optional bo_settings.json to update")
    ap.add_argument("--output-bo-settings", default=None, help="Optional output bo_settings.json path")
    ap.add_argument("--model", default="linear", choices=["linear", "quadratic_diag"])
    ap.add_argument("--keep-fraction", type=float, default=None, help="Fraction of regions to retain, e.g. 0.67")
    ap.add_argument("--keep-regions", type=int, default=None, help="Exact number of regions to retain")
    ap.add_argument("--final-control-nodes", type=int, default=None, help="Number of final CNs after screening")
    ap.add_argument("--region-radius", type=float, default=None, help="Candidate inclusion radius around region centres")

    args = ap.parse_args()

    result = run_analysis(
        prelim_results=args.prelim_results,
        morph_basis=args.morph_basis,
        output_morph_basis=args.output_morph_basis,
        bo_settings=args.bo_settings,
        output_bo_settings=args.output_bo_settings,
        model=args.model,
        keep_fraction=args.keep_fraction,
        keep_regions=args.keep_regions,
        final_control_nodes=args.final_control_nodes,
        region_radius=args.region_radius,
    )

    print("[PRELIM-RSA] Completed response-surface analysis")
    print(f"[PRELIM-RSA] model={result.model}, R2={result.r2:.6f}")
    print(f"[PRELIM-RSA] selected_regions={result.selected_regions.tolist()}")
    print(f"[PRELIM-RSA] scores={result.scores.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
