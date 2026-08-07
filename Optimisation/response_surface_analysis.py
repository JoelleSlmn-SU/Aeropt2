#!/usr/bin/env python3
"""
response_surface_analysis.py
 
Preliminary regional sensitivity analysis for AerOpt - Morris (elementary
effects) version.
 
Purpose
-------
After a Morris trajectory design (see morris_design.py) has been run
through the CFD pipeline, this script:
 
1. reads the design matrix X, objective values Y, and trajectory metadata,
2. computes elementary effects per region: mu (mean signed effect),
   mu_star (mean absolute effect - the importance ranking), and sigma
   (std of effect across trajectories - the interaction/nonlinearity flag),
3. ranks regions by mu_star,
4. keeps the most important regions,
5. distributes the final control nodes inside those selected regions using
   farthest-point sampling,
6. recomputes normals at the final control nodes,
7. writes an updated morph_basis.json and, optionally, bo_settings.json,
   including mu_star/sigma so a downstream step can turn them into an
   informative GP lengthscale prior for BO.
 
Why elementary effects instead of a single-baseline regression
----------------------------------------------------------------
The previous version of this script fit one linear model to a single
one-region-at-a-time perturbation around one baseline geometry. That
gives one finite-difference-like slope per region, with two problems:
it only reflects sensitivity local to that one baseline (irrelevant once
the optimizer moves elsewhere in a nonlinear regime like shock-dominated
intake flow), and it cannot see interaction between regions (two regions
that only matter jointly look unimportant individually).
 
Morris trajectories fix both: r independent, randomized base points give
mu_star that is averaged over the design space rather than measured at one
point, and sigma - the spread of a region's elementary effect across
trajectories - is a direct, cheap signal of interaction/nonlinearity that
a single regression coefficient cannot provide.
 
Expected prelim data format
----------------------------
JSON, produced by running morris_design.py's output X through your CFD
pipeline and appending the resulting Y:
    prelim_results.json
    {
      "X": [[...], [...]],       # from morris_design.py, unmodified order
      "Y": [0.1, 0.2, ...],      # objective, same row order as X
      "trajectories": [...],     # copied from morris_design.py's output
      "k": R, "r": ..., "p": ..., "delta": ...,
      "lb": [...], "ub": [...]
    }
 
Required morph_basis fields
---------------------------
The input morph_basis.json should contain at least:
    control_nodes              # representative region centres, shape (R, 3)
    control_normals             # optional, shape (R, 3)
    prelim_candidate_nodes      # candidate nodes over T surface, shape (Ncand, 3)
 
Recommended prelim fields:
    prelim_enabled
    prelim_regions
    prelim_keep_fraction
    prelim_final_control_nodes
    prelim_region_radius
"""
 
from __future__ import annotations
 
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
 
import numpy as np
 
 
@dataclass
class MorrisResult:
    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    selected_regions: np.ndarray
    n_trajectories_used: np.ndarray  # per-factor count of EE samples actually available
 
 
def _as_array(data, name: str, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {arr.shape}")
    return arr
 
 
def load_prelim_results(path: str) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Load DOE matrix X, objective vector Y, and Morris trajectory metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prelim results not found: {path}")
 
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    X = _as_array(data["X"], "X", ndim=2)
    Y = _as_array(data["Y"], "Y").reshape(-1)
 
    if "trajectories" not in data:
        raise KeyError(
            "prelim_results.json must contain 'trajectories' metadata produced by "
            "morris_design.py. This version of response_surface_analysis.py requires "
            "a Morris trajectory design, not an arbitrary DOE - the elementary-effects "
            "computation needs to know which factor changed at each step, and in which "
            "direction, for every trajectory."
        )
    trajectories = data["trajectories"]
 
    if len(Y) != X.shape[0]:
        raise ValueError(f"len(Y)={len(Y)} does not match X rows={X.shape[0]}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("X and Y must contain only finite values")
 
    return X, Y, trajectories
 
 
def compute_elementary_effects(
    X: np.ndarray,
    Y: np.ndarray,
    trajectories: List[dict],
) -> MorrisResult:
    """
    Compute Morris elementary effects (mu, mu_star, sigma) per region.
 
    For each trajectory and each step within it, exactly one factor
    changes; the elementary effect for that factor on that trajectory is
    the finite difference in Y divided by the actual (physical) step taken
    in X - read directly from X rather than reconstructed from delta/sign,
    so this is robust to any clipping that occurred during design
    generation near the domain boundary.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(Y) != len(X):
        raise ValueError("X must be 2-D and Y must contain one value per X row")
    k = X.shape[1]
 
    ee_by_factor: List[List[float]] = [[] for _ in range(k)]
 
    for traj_index, traj in enumerate(trajectories):
        rows = [int(v) for v in traj["rows"]]
        order = [int(v) for v in traj["order"]]
        if len(rows) != len(order) + 1:
            raise ValueError("Malformed trajectory metadata: len(rows) must equal len(order)+1")
        if any(r < 0 or r >= len(X) for r in rows):
            raise ValueError(f"Trajectory {traj_index} contains an out-of-range row index")
        if sorted(order) != list(range(k)):
            raise ValueError(f"Trajectory {traj_index} order must be a permutation of 0..{k - 1}")
 
        for step, factor in enumerate(order):
            r0, r1 = rows[step], rows[step + 1]
            changed = np.flatnonzero(np.abs(X[r1] - X[r0]) > 1e-12)
            if changed.size != 1 or int(changed[0]) != factor:
                raise ValueError(
                    f"Trajectory {traj_index}, step {step} is inconsistent: metadata says "
                    f"factor {factor}, but changed columns are {changed.tolist()}"
                )
            dx = X[r1, factor] - X[r0, factor]
            if abs(dx) < 1e-14:
                # Step landed on itself after boundary clipping; skip rather
                # than divide by ~0 and inject a spurious huge effect.
                continue
            dy = Y[r1] - Y[r0]
            ee_by_factor[factor].append(dy / dx)
 
    mu = np.zeros(k)
    mu_star = np.zeros(k)
    sigma = np.zeros(k)
    n_used = np.zeros(k, dtype=int)
 
    for j in range(k):
        vals = np.asarray(ee_by_factor[j], dtype=float)
        n_used[j] = len(vals)
        if len(vals) == 0:
            continue
        mu[j] = vals.mean()
        mu_star[j] = np.abs(vals).mean()
        sigma[j] = vals.std(ddof=1) if len(vals) > 1 else 0.0
 
    return MorrisResult(mu=mu, mu_star=mu_star, sigma=sigma,
                         selected_regions=np.array([]), n_trajectories_used=n_used)
 
 
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
    keep_fraction: Optional[float] = None,
    keep_regions: Optional[int] = None,
    final_control_nodes: Optional[int] = None,
    region_radius: Optional[float] = None,
    sigma_flag_ratio: float = 0.5,
) -> MorrisResult:
    X, Y, trajectories = load_prelim_results(prelim_results)
 
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
        candidate_nodes_data = basis.get("all_t_nodes", None)
    if candidate_nodes_data is None:
        raise KeyError(
            "morph_basis.json must contain 'prelim_candidate_nodes' containing candidate nodes over the T surface."
        )
    candidate_nodes = _as_array(candidate_nodes_data, "prelim_candidate_nodes", ndim=2)
 
    candidate_normals = basis.get("prelim_candidate_normals", None)
    if candidate_normals is not None:
        candidate_normals = _as_array(candidate_normals, "prelim_candidate_normals", ndim=2)
 
    result = compute_elementary_effects(X, Y, trajectories)
 
    if np.any(result.n_trajectories_used < 2):
        weak = np.where(result.n_trajectories_used < 2)[0].tolist()
        print(f"[PRELIM-MORRIS][WARNING] regions {weak} have fewer than 2 usable elementary-effect "
              f"samples (likely boundary clipping during design generation). sigma for these regions "
              f"is unreliable; consider more trajectories or narrower bounds.")
 
    if keep_regions is None:
        if keep_fraction is None:
            keep_fraction = float(basis.get("prelim_keep_fraction", 2.0 / 3.0))
        keep_regions = int(np.ceil(float(keep_fraction) * R))
    keep_regions = max(1, min(int(keep_regions), R))
 
    selected_regions = np.argsort(result.mu_star)[::-1][:keep_regions]
    selected_regions = np.sort(selected_regions)
    result.selected_regions = selected_regions
 
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(result.mu_star > 1e-12, result.sigma / result.mu_star, 0.0)
    interaction_flagged = np.where(ratio > sigma_flag_ratio)[0].tolist()
 
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
 
    updated = dict(basis)
    updated["prelim_completed"] = True
    updated["prelim_method"] = "morris_elementary_effects"
    updated["prelim_mu"] = result.mu.tolist()
    updated["prelim_mu_star"] = result.mu_star.tolist()
    updated["prelim_sigma"] = result.sigma.tolist()
    updated["prelim_n_trajectories_used"] = result.n_trajectories_used.tolist()
    updated["prelim_interaction_flagged_regions"] = interaction_flagged
    updated["prelim_selected_regions"] = selected_regions.astype(int).tolist()
    updated["prelim_region_radius"] = float(region_radius)
    updated["prelim_final_candidate_ids"] = final_candidate_ids.astype(int).tolist()
 
    updated["control_nodes"] = final_nodes.tolist()
    updated["control_normals"] = final_normals.tolist()
 
    old_k = int(updated.get("k_modes", min(6, len(final_nodes))))
    updated["k_modes"] = int(min(old_k, max(1, len(final_nodes) - 1))) if len(final_nodes) > 1 else 1
 
    os.makedirs(os.path.dirname(os.path.abspath(output_morph_basis)), exist_ok=True)
    with open(output_morph_basis, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
 
    if bo_settings:
        normal_project = bool(updated.get("normal_project", True))
        k_modes = int(updated["k_modes"])
        n_dim = k_modes if normal_project else 3 * k_modes
        update_bo_settings(bo_settings, n_dim=n_dim, output_path=output_bo_settings)
 
    summary_path = os.path.join(os.path.dirname(os.path.abspath(output_morph_basis)), "prelim_morris_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "morris_elementary_effects",
                "mu": result.mu.tolist(),
                "mu_star": result.mu_star.tolist(),
                "sigma": result.sigma.tolist(),
                "n_trajectories_used": result.n_trajectories_used.tolist(),
                "interaction_flagged_regions": interaction_flagged,
                "selected_regions": selected_regions.astype(int).tolist(),
                "region_radius": float(region_radius),
                "final_control_nodes": int(len(final_nodes)),
                "output_morph_basis": os.path.abspath(output_morph_basis),
                "output_bo_settings": os.path.abspath(output_bo_settings) if output_bo_settings else None,
            },
            f,
            indent=2,
        )
 
    return result
 
 
def main() -> int:
    ap = argparse.ArgumentParser(description="AerOpt preliminary Morris elementary-effects screening")
    ap.add_argument("--prelim-results", required=True, help="Path to prelim_results.json (X, Y, trajectories)")
    ap.add_argument("--morph-basis", required=True, help="Input morph_basis.json")
    ap.add_argument("--output-morph-basis", required=True, help="Output updated morph_basis.json")
    ap.add_argument("--bo-settings", default=None, help="Optional bo_settings.json to update")
    ap.add_argument("--output-bo-settings", default=None, help="Optional output bo_settings.json path")
    ap.add_argument("--keep-fraction", type=float, default=None, help="Fraction of regions to retain, e.g. 0.67")
    ap.add_argument("--keep-regions", type=int, default=None, help="Exact number of regions to retain")
    ap.add_argument("--final-control-nodes", type=int, default=None, help="Number of final CNs after screening")
    ap.add_argument("--region-radius", type=float, default=None, help="Candidate inclusion radius around region centres")
    ap.add_argument("--sigma-flag-ratio", type=float, default=0.5,
                     help="Flag a region as interaction-heavy if sigma/mu_star exceeds this ratio")
 
    args = ap.parse_args()
 
    result = run_analysis(
        prelim_results=args.prelim_results,
        morph_basis=args.morph_basis,
        output_morph_basis=args.output_morph_basis,
        bo_settings=args.bo_settings,
        output_bo_settings=args.output_bo_settings,
        keep_fraction=args.keep_fraction,
        keep_regions=args.keep_regions,
        final_control_nodes=args.final_control_nodes,
        region_radius=args.region_radius,
        sigma_flag_ratio=args.sigma_flag_ratio,
    )
 
    print("[PRELIM-MORRIS] Completed elementary-effects screening")
    print(f"[PRELIM-MORRIS] mu_star={np.round(result.mu_star, 4).tolist()}")
    print(f"[PRELIM-MORRIS] sigma={np.round(result.sigma, 4).tolist()}")
    print(f"[PRELIM-MORRIS] selected_regions={result.selected_regions.tolist()}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())