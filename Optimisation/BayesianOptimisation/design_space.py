"""
design_space.py -- initial sampling and geometric (pre-CFD) constraints for BayesianOptimiser.

Sampling methods (bo_settings.json "sampling_method")
-----------------------------------------------------
  "lhs"          Latin hypercube, random within strata   (previous behaviour, default)
  "lhs_maximin"  Latin hypercube, best of 20 by minimum pairwise distance
  "random"       independent uniform draws in the box
  "sobol"        scrambled Sobol' sequence (scipy.stats.qmc)
All draws come from ONE numpy Generator seeded by bo_settings.json "seed", so a
run with the same seed reproduces its sample exactly.

Geometric constraints
---------------------
A constraint whose metric starts with "geom_" (e.g. "geom_envelope >= 0") is not a
CFD output: it is computed from the morphed geometry, before any CFD, by a function
the driver supplies,

    geom_func(X (m, n)) -> {metric: values (m,)}

The optimiser then
  * keeps only designs that satisfy them in the initial sample and in random search
    (rejection sampling; the acceptance rate is logged -- a low rate means the box
    is mostly infeasible and should be reconsidered),
  * gives the acquisition search a penalty > any feasible value outside them
    (so CMA-ES stays inside the feasible set), and
  * never fits a GP to them (they are exact; a GP would only blur them).

Many of these functions are linear (or very nearly) in the design vector: normal
displacements are linear in the modal coefficients and the morph is a linear
interpolation. `Linearised` probes geom_func at n+1 points, checks the linear model
at a few random designs, and from then on evaluates g0 + A x (cheap enough for the
~10^4 calls of one acquisition search). If the check fails it falls back to calling
geom_func directly and says so.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from Utilities.lhs import lhs as _lhs

SAMPLING_METHODS = ("lhs", "lhs_maximin", "random", "sobol")
GEOM_PREFIX = "geom_"


def is_geometric(metric: str) -> bool:
    return str(metric).strip().startswith(GEOM_PREFIX)


def sample_box(method: str, n: int, lb, ub, rng: np.random.Generator) -> np.ndarray:
    """n points in the box [lb, ub] (shape (n, dim))."""
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    d = lb.size
    n = int(n)
    if n <= 0:
        return np.zeros((0, d))
    method = str(method or "lhs").lower().strip()
    if method in ("lhs", "lhs_maximin"):
        # Utilities.lhs draws from the global numpy state: seed it from our generator
        state = np.random.get_state()
        try:
            np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))
            U = _lhs(d, samples=n, criterion=("maximin" if method == "lhs_maximin" else None),
                     iterations=(20 if method == "lhs_maximin" else None))
        finally:
            np.random.set_state(state)
        U = np.asarray(U, float).reshape(n, d)
    elif method == "random":
        U = rng.random((n, d))
    elif method == "sobol":
        from scipy.stats import qmc
        U = qmc.Sobol(d, scramble=True, seed=rng).random(n)
    else:
        raise ValueError(f"Unknown sampling_method {method!r}; use one of {SAMPLING_METHODS}")
    return lb + U * (ub - lb)


class Linearised:
    """g(x) ~= g0 + A x, built from n+1 calls of an expensive geometric function."""

    def __init__(self, func: Callable[[np.ndarray], Dict[str, np.ndarray]], lb, ub,
                 rng: np.random.Generator, rel_tol: float = 1e-3, n_check: int = 4,
                 log: Callable[[str], None] = print):
        self.func = func
        lb = np.asarray(lb, float)
        ub = np.asarray(ub, float)
        d = lb.size
        # small probe step: the model is built around the baseline, the check below
        # tells whether it holds across the box
        h = 0.05 * (ub - lb)
        X = np.vstack([np.zeros(d), np.diag(h)])
        G = {k: _as2d(v, X.shape[0]) for k, v in func(X).items()}
        self.names = list(G)
        self.g0 = {k: G[k][0] for k in self.names}                           # (p,)
        self.A = {k: (G[k][1:] - self.g0[k]) / h[:, None] for k in self.names}  # (d, p)
        Xc = sample_box("random", n_check, lb, ub, rng)
        exact = {k: _as2d(v, n_check) for k, v in func(Xc).items()}
        self.max_rel_err = {}
        for k in self.names:
            lin = self.g0[k] + Xc @ self.A[k]
            ex = exact[k]
            if not (np.all(np.isfinite(ex)) and np.all(np.isfinite(lin))):
                self.max_rel_err[k] = float("inf")
                continue
            scale = max(float(np.max(np.abs(ex - self.g0[k]))), float(np.max(np.abs(self.g0[k]))) * 1e-6, 1e-300)
            self.max_rel_err[k] = float(np.max(np.abs(lin - ex)) / scale)
        # per metric: linear model where it holds, the exact function where it does not
        self.exact_names = [k for k in self.names if self.max_rel_err[k] > rel_tol]
        self.exact = bool(self.exact_names)
        for k in self.names:
            how = (f"NOT linear (relative error {self.max_rel_err[k]:.2e} > {rel_tol:.0e}): evaluated exactly "
                   "at every check (slower acquisition search)") if k in self.exact_names else \
                  f"linear to {self.max_rel_err[k]:.1e}: using g0 + A x"
            log(f"[GEOM] {k}: {how} (checked at {n_check} random designs)")

    def __call__(self, X) -> Dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, float))
        out = {k: self.g0[k] + X @ self.A[k] for k in self.names if k not in self.exact_names}
        if self.exact_names:
            G = self.func(X)
            for k in self.exact_names:
                out[k] = _as2d(G[k], X.shape[0])
        return out


def _as2d(v, m):
    """(m,) or (m, p) -> (m, p): a metric may be one number per design or one per
    station (e.g. thickness at 100 chord positions); the constraint applies to all."""
    a = np.asarray(v, float)
    return a.reshape(m, -1)


class GeometricConstraints:
    """The geom_* constraints of a run and the function that evaluates them."""

    def __init__(self, specs: Sequence[dict], func: Optional[Callable], lb, ub,
                 rng: np.random.Generator, linearise: bool = True,
                 log: Callable[[str], None] = print):
        self.specs = [dict(s) for s in specs]
        self.log = log
        if self.specs and func is None:
            names = [s["metric"] for s in self.specs]
            raise ValueError(f"Geometric constraint(s) {names} given, but this driver supplies no "
                             "geometry function for them.")
        self.func = None
        if self.specs:
            self.func = Linearised(func, lb, ub, rng, log=log) if linearise else func
            missing = [s["metric"] for s in self.specs
                       if s["metric"] not in (self.func.names if linearise else self.func(np.zeros((1, len(lb)))))]
            if missing:
                raise ValueError(f"The geometry function does not provide {missing}.")

    def __bool__(self):
        return bool(self.specs)

    def raw(self, X) -> Dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, float))
        if not self.specs:
            return {}
        return {k: _as2d(v, X.shape[0]) for k, v in self.func(X).items()}

    def values(self, X) -> Dict[str, np.ndarray]:
        """Worst case over stations, one number per design and constraint (for logs)."""
        G = self.raw(X)
        out = {}
        for s in self.specs:
            g = G[s["metric"]]
            out[s["metric"]] = g.max(axis=1) if s["sense"] == "<=" else g.min(axis=1)
        return out

    def violation(self, X) -> np.ndarray:
        """Sum over constraints and stations of the violations, scaled by |limit|
        (or 1); 0 where feasible."""
        X = np.atleast_2d(np.asarray(X, float))
        v = np.zeros(X.shape[0])
        if not self.specs:
            return v
        G = self.raw(X)
        for s in self.specs:
            g = G[s["metric"]]
            lim = float(s["limit"])
            excess = (g - lim) if s["sense"] == "<=" else (lim - g)
            excess = np.where(np.isfinite(excess), excess, np.inf)
            v += np.sum(np.maximum(excess, 0.0), axis=1) / max(abs(lim), 1.0)
        return v

    def feasible(self, X, tol: float = 1e-12) -> np.ndarray:
        return self.violation(X) <= tol

    def sample_feasible(self, method, n, lb, ub, rng, max_factor: int = 200):
        """n feasible designs by rejection; returns (X, acceptance_rate). Fewer than n
        are returned (with a warning) if max_factor * n candidates are not enough."""
        n = int(n)
        if not self.specs:
            return sample_box(method, n, lb, ub, rng), 1.0
        cand = sample_box(method, n, lb, ub, rng)
        ok = self.feasible(cand)
        kept = [cand[ok]]
        tried, got = n, int(ok.sum())
        while got < n and tried < max_factor * n:
            # top up with uniform draws (a stratified design cannot be extended)
            m = max(n, 4 * (n - got))
            c = sample_box("random", m, lb, ub, rng)
            ok = self.feasible(c)
            kept.append(c[ok])
            tried += m
            got += int(ok.sum())
        X = np.vstack(kept)[:n]
        rate = got / max(tried, 1)
        self.log(f"[GEOM] {len(X)} of {n} requested designs satisfy the geometric constraints "
                 f"(acceptance {100 * rate:.2f} % of {tried} candidates).")
        if len(X) < n:
            self.log("[GEOM][WARN] could not find enough feasible designs: the box is almost entirely "
                     "infeasible. Tighten the bounds or reconsider the constraint.")
        return X, rate
