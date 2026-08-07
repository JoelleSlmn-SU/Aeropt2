#!/usr/bin/env python3
"""
morris_design.py
 
Generates a Morris (elementary effects) trajectory design for the AerOpt
preliminary regional screening step.
 
This replaces the single-baseline, one-region-at-a-time DOE that
response_surface_analysis.py used to assume was already produced elsewhere.
Instead of one perturbation per region around a single baseline, this
generates r randomized trajectories through the R-dimensional region-
displacement space. Each trajectory is a path of R+1 points where every
step changes exactly one region's displacement by a fixed amount, with the
order of regions and the direction of each step randomized per trajectory.
 
This gives you, per region, r independent "elementary effect" samples
instead of one finite-difference slope from a single baseline - which is
what lets the analysis step separate genuine importance (mu*) from
interaction/nonlinearity (sigma), rather than reporting a single regression
coefficient that conflates the two.
 
Reference construction: Morris (1991); trajectory sampling matrix as given
in Saltelli et al., "Global Sensitivity Analysis: The Primer" (2008), ch. 3.
 
Output
------
Writes a design file (JSON) containing:
    X                physical design matrix, shape (r*(R+1), R)
    trajectories     list of r dicts describing step structure, needed by
                      response_surface_analysis.py to compute elementary
                      effects (which factor changed at each step, and in
                      which direction)
    k, r, p, delta    design parameters
    lb, ub            physical bounds used, shape (R,)
 
This file's X should be run through your existing prelim CFD evaluation
pipeline (one CFD run per row) to produce the corresponding Y vector.
The resulting (X, Y, trajectories) triple is what
response_surface_analysis.py now expects as prelim_results.
"""
 
from __future__ import annotations
 
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional
 
import numpy as np
 
 
@dataclass
class MorrisDesign:
    X: np.ndarray                  # (r*(k+1), k) physical design matrix
    trajectories: List[dict]       # per-trajectory step metadata
    k: int
    r: int
    p: int
    delta: float                   # step size in unit [0,1] space
    lb: np.ndarray
    ub: np.ndarray
 
 
def _build_single_trajectory(k: int, p: int, delta: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Build one Morris trajectory of k+1 points in unit [0,1]^k space.
 
    Follows the standard B* construction:
        B*  = (J x*  +  (delta/2)[(2B - J_k) D* + J_k]) P*
    where B is (k+1)xk strictly-lower-triangular of ones, D* is a random
    +-1 diagonal, P* is a random permutation matrix, x* is a random base
    point on the coarse grid.
 
    Returns
    -------
    traj_points : (k+1, k) array, unit space
    meta : dict with:
        "order": list[int], length k - order[i] = index of factor that
                 changes going from traj_points[i] to traj_points[i+1]
        "sign":  list[int], length k - sign[j] = +1 or -1, the direction
                 factor j moves whenever it is the one being stepped
    """
    # Construct the path explicitly.  This is algebraically equivalent to
    # the usual B* construction, but avoids an easy-to-miss permutation
    # convention bug where the metadata says factor j changed while the
    # generated matrix actually changed another column.
    levels = np.arange(p) / (p - 1)
    signs = rng.choice([-1, 1], size=k)
    order = rng.permutation(k)
    x0 = np.empty(k, dtype=float)
    for j, sign in enumerate(signs):
        allowed = levels[levels <= 1.0 - delta + 1e-12] if sign > 0 else levels[levels >= delta - 1e-12]
        if allowed.size == 0:
            raise RuntimeError(f"No valid Morris base level for p={p}, delta={delta}")
        x0[j] = float(rng.choice(allowed))

    B_star = np.empty((k + 1, k), dtype=float)
    B_star[0] = x0
    for step, factor in enumerate(order):
        B_star[step + 1] = B_star[step]
        B_star[step + 1, factor] += signs[factor] * delta
 
    meta = {"order": order.tolist(), "sign": signs.tolist()}
    return B_star, meta
 
 
def generate_morris_trajectories(
    k: int,
    r: int,
    p: int = 4,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> MorrisDesign:
    """
    Generate r Morris trajectories over k factors.
 
    Parameters
    ----------
    k : number of regions/factors
    r : number of trajectories (repetitions). Total CFD runs = r*(k+1).
        r ~ 8-12 is a reasonable starting point; more gives cleaner mu*/sigma
        estimates at proportional extra cost.
    p : number of grid levels per factor. Must be even (Morris 1991
        recommends this so the step delta = p/(2(p-1)) lands exactly on a
        grid point). Default 4.
    lb, ub : physical lower/upper bounds per factor, shape (k,). Defaults
        to [-1, 1] per factor to match the bo_settings.json convention
        already used elsewhere in this pipeline.
    seed : RNG seed for reproducibility.
    """
    if int(k) < 1 or int(r) < 1:
        raise ValueError("k and r must both be positive integers")
    if int(p) < 2:
        raise ValueError("p must be at least 2")
    if p % 2 != 0:
        raise ValueError("p should be even for the standard Morris step size to land on-grid.")
 
    delta = p / (2.0 * (p - 1))
    rng = np.random.default_rng(seed)
 
    if lb is None:
        lb = -np.ones(k)
    if ub is None:
        ub = np.ones(k)
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    if lb.shape[0] != k or ub.shape[0] != k:
        raise ValueError("lb/ub must have length k")
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("lb/ub must contain only finite values")
    if np.any(ub <= lb):
        raise ValueError("Every upper bound must be greater than its lower bound")
 
    all_points = []
    trajectories = []
    row_cursor = 0
    for _ in range(r):
        traj_unit, meta = _build_single_trajectory(k, p, delta, rng)
        traj_phys = lb[None, :] + traj_unit * (ub - lb)[None, :]
        all_points.append(traj_phys)
        meta["rows"] = list(range(row_cursor, row_cursor + k + 1))
        trajectories.append(meta)
        row_cursor += k + 1
 
    X = np.vstack(all_points)
    return MorrisDesign(X=X, trajectories=trajectories, k=k, r=r, p=p,
                         delta=delta, lb=lb, ub=ub)
 
 
def save_design(design: MorrisDesign, output_path: str) -> None:
    payload = {
        "X": design.X.tolist(),
        "trajectories": design.trajectories,
        "k": design.k,
        "r": design.r,
        "p": design.p,
        "delta": design.delta,
        "lb": design.lb.tolist(),
        "ub": design.ub.tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
 
 
def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Morris trajectory design for prelim regional screening")
    ap.add_argument("--k", type=int, required=True, help="Number of regions/factors")
    ap.add_argument("--r", type=int, default=10, help="Number of trajectories (repetitions)")
    ap.add_argument("--p", type=int, default=4, help="Number of grid levels (even)")
    ap.add_argument("--lb", type=float, default=-1.0, help="Lower bound (applied to all factors unless --bounds-file given)")
    ap.add_argument("--ub", type=float, default=1.0, help="Upper bound (applied to all factors unless --bounds-file given)")
    ap.add_argument("--bounds-file", default=None, help="Optional JSON with {'lb': [...], 'ub': [...]} per-factor bounds")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", required=True, help="Output design JSON path")
    args = ap.parse_args()
 
    if args.bounds_file:
        with open(args.bounds_file, "r", encoding="utf-8") as f:
            b = json.load(f)
        lb = np.asarray(b["lb"], dtype=float)
        ub = np.asarray(b["ub"], dtype=float)
    else:
        lb = np.full(args.k, args.lb)
        ub = np.full(args.k, args.ub)
 
    design = generate_morris_trajectories(k=args.k, r=args.r, p=args.p, lb=lb, ub=ub, seed=args.seed)
    save_design(design, args.output)
 
    n_runs = args.r * (args.k + 1)
    print(f"[MORRIS-DESIGN] k={args.k}, r={args.r}, p={args.p}, delta={design.delta:.4f}")
    print(f"[MORRIS-DESIGN] total CFD runs required: {n_runs}")
    print(f"[MORRIS-DESIGN] wrote design to {args.output}")
    print("[MORRIS-DESIGN] run each row of X through your prelim CFD pipeline, "
          "collect Y in the same row order, then pass both to response_surface_analysis.py")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 