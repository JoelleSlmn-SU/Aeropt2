#!/usr/bin/env python3
"""
surface_sensitivity.py
======================

Data-driven surface sensitivity analysis for a set of morphed surfaces
(e.g. an LHS over Laplacian-mode coefficients) and their CFD outputs.

Why surface space, not mode space
---------------------------------
In Aeropt2 the Laplacian modes live on the *control nodes* (normalised graph
Laplacian, modalBasis.build_laplacian_basis) and are propagated to the surface
by RBF. Within ONE parameterisation the coefficient -> surface map is linear
(checked per batch in the report), but different parameterisation versions
(e.g. the DSI DOE cases 15-28 vs the rest) use different bases, so the same
coefficient vector means a different shape. Everything here therefore works on
what the solver actually saw: the node-wise displacement
d_i(x) = X_i(x) - X_0(x) of each deformed .fro against the baseline .fro
(the morph preserves node numbering and connectivity; this is checked on load).
That lets designs from different parameterisations and from the optimiser be
pooled legitimately.

What it computes, per objective (expressed as % improvement vs baseline,
positive = better)
------------------------------------------------------------------------
Nodal (continuous) maps on the deformable part of the surface
  r(x)      Pearson correlation between normal displacement dn(x) and the
            improvement, with a family-wise significance threshold from a
            max-|r| permutation test (Westfall-Young maxT) -> handles the
            thousands of spatially-correlated nodes without Bonferroni.
  beta(x)   Kernel-ridge "data-driven adjoint": the area-weighted linear
            functional  dJ ~ sum_x a(x) beta(x) dn(x)  that best predicts the
            improvement. Ridge lambda picked by exact leave-one-out (with
            re-centring inside each fold). Bootstrap sign stability is
            reported; unstable regions are masked in the figures.
  contrast  mean dn of the best third of designs minus the worst third
            ("what did good designs do to the surface").
  Q2_LOO    leave-one-out predictive skill of the linear model. If Q2 <= ~0.2
            the maps are NOT interpretable for that objective - this is
            printed and written to the report.

Patch (grid) analysis - coarse, interpretable regions
  The deformable region is projected onto a 2-D plane (principal axes of the
  region, or user-chosen axes) and split into an nu x nv grid. Each patch
  feature = area-weighted mean normal displacement inside it.
  Per patch: Spearman rho, permutation p, Benjamini-Hochberg q, ridge
  coefficient on standardised features (LOO lambda) + bootstrap sign
  stability. The whole analysis is repeated on a half-cell-shifted grid; a
  patch that is "critical" on one grid but not the other is a gridding
  artefact.
  The effective rank of the patch matrix is reported: the LHS only spans as
  many independent shape directions as there are modes, so patches that are
  always moved together by the modes cannot be separated by ANY method.

Outputs (in --outdir)
  sensitivity_surface.vtk      baseline surface with all nodal fields
                               (open in ParaView)
  fig_<obj>.png                4-panel map per objective
  fig_objectives.png           correlation between objectives
  fig_loo_<obj>.png            LOO predicted vs actual
  patch_stats_<obj>.csv        per-patch statistics (main grid)
  patch_stats_<obj>_shifted.csv
  design_summary.csv           per-design displacement diagnostics
  model_skill.csv              Q2_LOO, lambda, n per objective
  report.md                    human-readable summary

Usage (Aeropt2 DOE folder: <root>/1/ baseline .fro, <root>/2..N/ morphed .fro + morph_config_n_*.json)
-----
1) Build the table (reads the 'Results' sheet: design # in column A = folder number;
   pulls t/u/c surfaces, n, gen, batch and modal_coeffs from each morph config, and checks
   that every batch uses the same modal basis -> basis_check.txt):

   python surface_sensitivity.py from-doe "DSI DOE" --results "DSI DOE/DSI_Optimisation_Results.xlsx"

2) Analyse the T surface (default), LHS designs only, raw and size-controlled:

   python surface_sensitivity.py analyse "DSI DOE/designs.csv" --outdir sens_T_raw  --lhs-only
   python surface_sensitivity.py analyse "DSI DOE/designs.csv" --outdir sens_T_ctrl --lhs-only --control-magnitude

   Useful switches: --surfaces TU | all | 14,15    --rank-y (outlier-robust)
                    --grid 8 4    --normal-convention fluid|file

   Normals are oriented INTO THE FLUID by default (ray-parity test on the closed .fro
   boundary), so dn > 0 always means "surface pushed into the flow".

(Legacy: `xlsx2csv` converts the old 'FP M1.x' sheet layout.)

Dependencies: numpy, scipy, pandas, matplotlib (openpyxl for xlsx2csv).
No Aeropt2 imports - runs on a laptop or a login node.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("surfsens")


# =============================================================================
# .fro IO (same layout as FileRW/FroFile.py, vectorised)
# =============================================================================
@dataclass
class Surface:
    nodes: np.ndarray            # (N,3) float
    tris: np.ndarray             # (T,3) int, 0-based  (quads split into 2 tris)
    tri_sid: np.ndarray          # (T,)  surface id per triangle
    path: str = ""


def read_fro(path: str) -> Surface:
    """Read a FLITE .fro surface.

    Header with 8 fields  -> ntri nnode ... (no quads)       [FroFile.read_file]
    Header otherwise      -> nquad ntri nnode ...
    Then nnode lines 'id x y z ...', nquad lines 'id v1 v2 v3 v4 sid',
    ntri lines 'id v1 v2 v3 sid' (1-based vertex ids). Tail is ignored.
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError(f"{path}: empty file")
    h = lines[0].split()
    if len(h) == 8:
        nquad, ntri, nnode = 0, int(h[0]), int(h[1])
    else:
        nquad, ntri, nnode = int(h[0]), int(h[1]), int(h[2])

    def block(i0, n, cols):
        if n == 0:
            return np.zeros((0, len(cols)))
        txt = lines[i0:i0 + n]
        if len(txt) != n:
            raise ValueError(f"{path}: truncated file (expected {n} lines from line {i0+1})")
        if any(("D" in t) or ("d" in t) for t in txt):   # Fortran double exponent
            txt = [t.replace("D", "E").replace("d", "e") for t in txt]
        return np.loadtxt(txt, usecols=cols, ndmin=2)

    i = 1
    nodes = block(i, nnode, (1, 2, 3)).astype(float); i += nnode
    quads = block(i, nquad, (1, 2, 3, 4, 5)).astype(np.int64); i += nquad
    tris = block(i, ntri, (1, 2, 3, 4)).astype(np.int64)

    T = [tris[:, :3] - 1]
    S = [tris[:, 3]]
    if nquad:
        q = quads[:, :4] - 1
        T += [q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
        S += [quads[:, 4], quads[:, 4]]
    tri = np.vstack(T)
    sid = np.concatenate(S)
    if tri.size and (tri.min() < 0 or tri.max() >= nnode):
        raise ValueError(f"{path}: connectivity references node outside 1..{nnode}")
    return Surface(nodes=nodes, tris=tri, tri_sid=sid, path=path)


# =============================================================================
# Geometry
# =============================================================================
def tri_normals_areas(nodes, tris):
    p0, p1, p2 = nodes[tris[:, 0]], nodes[tris[:, 1]], nodes[tris[:, 2]]
    c = np.cross(p1 - p0, p2 - p0)
    a2 = np.linalg.norm(c, axis=1)
    return c, 0.5 * a2          # c is area-weighted (unnormalised) normal*2


def vertex_normals_areas(nodes, tris):
    c, area = tri_normals_areas(nodes, tris)
    N = len(nodes)
    vn = np.zeros((N, 3))
    va = np.zeros(N)
    for k in range(3):
        np.add.at(vn, tris[:, k], c)
        np.add.at(va, tris[:, k], area / 3.0)
    nrm = np.linalg.norm(vn, axis=1)
    ok = nrm > 0
    vn[ok] /= nrm[ok, None]
    return vn, va


def _ray_crossings(P0, E1, E2, o, d):
    """Number of triangle hits of the ray o + t d (t > 0), Moller-Trumbore, vectorised."""
    h = np.cross(d, E2)
    a = np.einsum("ij,ij->i", E1, h)
    ok = np.abs(a) > 1e-12
    f = np.zeros_like(a); f[ok] = 1.0 / a[ok]
    s = o - P0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, E1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", E2, q)
    return int(np.sum(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)))


def fluid_side_sign(nodes, tris, vn, probe_nodes, n_probe=7, n_rays=3, seed=0):
    """+1 if +n points into the fluid, -1 if into the solid, 0 if inconclusive.
    Uses point-in-closed-surface parity: the .fro holds the whole domain boundary
    (walls + farfield), so a point is in the fluid iff a ray from it crosses the
    boundary an odd number of times."""
    rng = np.random.default_rng(seed)
    P0 = nodes[tris[:, 0]]; E1 = nodes[tris[:, 1]] - P0; E2 = nodes[tris[:, 2]] - P0
    ext = np.linalg.norm(nodes.max(0) - nodes.min(0))
    votes = []
    for i in rng.choice(probe_nodes, min(n_probe, len(probe_nodes)), replace=False):
        # step off the surface by a small fraction of the local edge length
        eps = 1e-3 * ext if len(nodes) < 10 else None
        nb = tris[np.any(tris == i, axis=1)]
        h = np.mean(np.linalg.norm(nodes[nb[:, 1]] - nodes[nb[:, 0]], axis=1)) if len(nb) else 1e-3 * ext
        eps = 0.1 * h
        par = []
        for sgn in (+1, -1):
            o = nodes[i] + sgn * eps * vn[i]
            c = []
            for _ in range(n_rays):
                d = rng.normal(size=3); d /= np.linalg.norm(d)
                c.append(_ray_crossings(P0, E1, E2, o, d) % 2)
            par.append(np.mean(c) > 0.5)
        if par[0] and not par[1]:
            votes.append(+1)
        elif par[1] and not par[0]:
            votes.append(-1)
    if not votes:
        return 0
    m = np.mean(votes)
    return int(np.sign(m)) if abs(m) >= 0.7 else 0


# =============================================================================
# Statistics helpers
# =============================================================================
def _rank(a, axis=0):
    from scipy.stats import rankdata
    return rankdata(a, axis=axis)


def col_corr(X, y):
    """Pearson correlation of each column of X (n,p) with y (n,). NaN for constant cols."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den <= 0] = np.nan
    return r


def maxT_threshold(X, y, n_perm=2000, alpha=0.05, rng=None):
    """Westfall-Young max-|r| permutation threshold for column-wise correlation.
    Returns r_crit such that P(max|r| >= r_crit | H0) = alpha, and the null maxima."""
    rng = np.random.default_rng(rng)
    Xc = X - X.mean(0)
    sx = np.sqrt((Xc ** 2).sum(0))
    good = sx > 0
    Xn = Xc[:, good] / sx[good]
    yc = y - y.mean()
    yc = yc / np.linalg.norm(yc)
    mx = np.empty(n_perm)
    for b in range(n_perm):
        mx[b] = np.max(np.abs(Xn.T @ yc[rng.permutation(len(y))]))
    return float(np.quantile(mx, 1 - alpha)), mx


def perm_pvalues(X, y, n_perm=5000, rng=None, spearman=True):
    """Per-column two-sided permutation p-values (+ maxT adjusted p) for (Spearman) corr."""
    rng = np.random.default_rng(rng)
    if spearman:
        X = _rank(X, 0).astype(float)
        y = _rank(y).astype(float)
    r_obs = col_corr(X, y)
    Xc = X - X.mean(0)
    sx = np.sqrt((Xc ** 2).sum(0))
    sx[sx == 0] = np.inf
    Xn = Xc / sx
    yc = y - y.mean()
    yc /= np.linalg.norm(yc)
    cnt = np.zeros(X.shape[1])
    cnt_max = np.zeros(X.shape[1])
    a_obs = np.abs(r_obs)
    for _ in range(n_perm):
        rp = np.abs(Xn.T @ yc[rng.permutation(len(y))])
        cnt += rp >= a_obs - 1e-12
        cnt_max += rp.max() >= a_obs - 1e-12
    p = (cnt + 1) / (n_perm + 1)
    p_fwer = (cnt_max + 1) / (n_perm + 1)
    return r_obs, p, p_fwer


def bh_fdr(p):
    p = np.asarray(p, float)
    n = np.sum(~np.isnan(p))
    q = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


# -----------------------------------------------------------------------------
# Kernel ridge in the area-weighted L2 inner product on the surface
# -----------------------------------------------------------------------------
def _center_kernel_train(G):
    n = G.shape[0]
    C = np.eye(n) - 1.0 / n
    return C @ G @ C


def _center_kernel_test(g_test, G_train):
    # g_test: (m, n) raw kernel between test and train rows
    return (g_test - g_test.mean(1, keepdims=True)
            - G_train.mean(0, keepdims=True) + G_train.mean())


def kernel_ridge_loo(G, y, lambdas):
    """Exact LOO (with re-centring of X and y in every fold) for linear kernel ridge.
    G = raw Gram matrix  X W X^T  (n,n). Returns (Q2 per lambda, loo predictions per lambda)."""
    n = len(y)
    preds = np.zeros((len(lambdas), n))
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        Gt = G[np.ix_(tr, tr)]
        Kc = _center_kernel_train(Gt)
        kt = _center_kernel_test(G[i:i + 1, tr], Gt)
        yt = y[tr]
        ym = yt.mean()
        w, V = np.linalg.eigh(Kc)
        w = np.clip(w, 0, None)
        Vty = V.T @ (yt - ym)
        kV = (kt @ V).ravel()
        for li, lam in enumerate(lambdas):
            alpha_proj = Vty / (w + lam)
            preds[li, i] = ym + float(kV @ alpha_proj)
    ss = np.sum((y - y.mean()) ** 2)
    q2 = 1.0 - np.sum((preds - y[None, :]) ** 2, axis=1) / ss
    return q2, preds


def kernel_ridge_fit(X, wts, y, lam):
    """Fit on all data. X (n,p) features, wts (p,) quadrature weights (areas).
    Returns beta (p,) such that yhat = ym + sum_j wts_j beta_j (x_j - xm_j)."""
    xm = X.mean(0)
    Xc = X - xm
    ym = y.mean()
    K = (Xc * wts) @ Xc.T
    alpha = np.linalg.solve(K + lam * np.eye(len(y)), y - ym)
    beta = Xc.T @ alpha
    return beta, xm, ym


def choose_lambda(G, y, n_grid=40):
    scale = max(np.trace(_center_kernel_train(G)) / len(y), 1e-300)
    lambdas = scale * np.logspace(-4, 3, n_grid)
    q2, preds = kernel_ridge_loo(G, y, lambdas)
    k = int(np.nanargmax(q2))
    return lambdas[k], q2[k], preds[k], lambdas, q2


def _ols_fit_pred(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ coef, coef


def magnitude_loo(M, y):
    """LOO Q2 of y ~ a + b m (deformation-size-only model)."""
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        pred[i] = _ols_fit_pred(M[tr], y[tr], M[i:i + 1])[0][0]
    return 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2), pred


def combined_loo(G, M, y, lambdas):
    """LOO Q2 of  y ~ [a + b m]  +  linear functional of dn  (ridge on the residual).
    Magnitude part refitted inside every fold, so there is no leakage."""
    n = len(y)
    preds = np.zeros((len(lambdas), n))
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        p_tr, coef = _ols_fit_pred(M[tr], y[tr], M[tr])
        p_te = np.column_stack([np.ones(1), M[i:i + 1]]) @ coef
        rtr = y[tr] - p_tr
        Gt = G[np.ix_(tr, tr)]
        Kc = _center_kernel_train(Gt)
        kt = _center_kernel_test(G[i:i + 1, tr], Gt)
        w, V = np.linalg.eigh(Kc)
        w = np.clip(w, 0, None)
        Vty = V.T @ (rtr - rtr.mean())
        kV = (kt @ V).ravel()
        for li, lam in enumerate(lambdas):
            preds[li, i] = p_te[0] + rtr.mean() + float(kV @ (Vty / (w + lam)))
    q2 = 1 - np.sum((preds - y[None]) ** 2, axis=1) / np.sum((y - y.mean()) ** 2)
    k = int(np.nanargmax(q2))
    return q2[k], lambdas[k], preds[k]


def normal_scores(y):
    from scipy.stats import norm
    r = _rank(y)
    return norm.ppf((r - 0.5) / len(y))


def bootstrap_sign_stability(X, wts, y, lam, beta_full, n_boot=500, rng=None):
    rng = np.random.default_rng(rng)
    n = len(y)
    agree = np.zeros(X.shape[1])
    done = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(y[idx]) == 0:
            continue
        b, _, _ = kernel_ridge_fit(X[idx], wts, y[idx], lam)
        agree += np.sign(b) == np.sign(beta_full)
        done += 1
    return agree / max(done, 1)


# =============================================================================
# Patch / grid construction
# =============================================================================
def plane_axes(P, w, axes_spec="auto"):
    """Return (e1, e2, origin) of the projection plane."""
    unit = {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}
    o = np.average(P, axis=0, weights=w)
    if axes_spec != "auto":
        a, b = axes_spec[0], axes_spec[1]
        return unit[a], unit[b], o
    Q = (P - o) * np.sqrt(w)[:, None]
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    # intake convention: put the axis most aligned with x (streamwise) first
    if abs(e2[0]) > abs(e1[0]):
        e1, e2 = e2, e1
    # make orientation deterministic: largest component positive
    if e1[np.argmax(np.abs(e1))] < 0:
        e1 = -e1
    if e2[np.argmax(np.abs(e2))] < 0:
        e2 = -e2
    return e1, e2, o


def grid_patches(uv, nu, nv, shift=(0.0, 0.0), pad=1e-9):
    """Assign each point to a grid cell; shift is in fractions of a cell."""
    umin, vmin = uv.min(0) - pad
    umax, vmax = uv.max(0) + pad
    du, dv = (umax - umin) / nu, (vmax - vmin) / nv
    # a shifted grid needs one more cell per direction to cover the region
    u0, v0 = umin - shift[0] * du, vmin - shift[1] * dv
    nu_e = nu + (1 if shift[0] else 0)
    nv_e = nv + (1 if shift[1] else 0)
    iu = np.clip(((uv[:, 0] - u0) // du).astype(int), 0, nu_e - 1)
    iv = np.clip(((uv[:, 1] - v0) // dv).astype(int), 0, nv_e - 1)
    pid = iu * nv_e + iv
    edges_u = u0 + du * np.arange(nu_e + 1)
    edges_v = v0 + dv * np.arange(nv_e + 1)
    return pid, nu_e, nv_e, edges_u, edges_v


def patch_features(Dn, area, pid, n_patch, min_area_frac=0.01):
    """Area-weighted mean normal displacement per patch. Dn (n,N)."""
    A = np.zeros(n_patch)
    np.add.at(A, pid, area)
    F = np.zeros((Dn.shape[0], n_patch))
    for j in range(Dn.shape[0]):
        s = np.zeros(n_patch)
        np.add.at(s, pid, Dn[j] * area)
        F[j] = s
    keep = A > min_area_frac * A.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        F = F / A
    return F, A, keep


def effective_rank(F, tol=1e-2):
    Fc = F - F.mean(0)
    s = np.linalg.svd(Fc, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0, s
    return int(np.sum(s / s[0] > tol)), s


def patch_analysis(F, y, n_perm, n_boot, rng):
    """Univariate (Spearman + permutation + BH) and multivariate ridge per patch."""
    rho, p, p_fwer = perm_pvalues(F, y, n_perm=n_perm, rng=rng, spearman=True)
    q = bh_fdr(p)
    # ridge on standardised features (so coefficients are comparable)
    sd = F.std(0, ddof=1)
    sd[sd == 0] = np.inf
    Z = (F - F.mean(0)) / sd
    wts = np.ones(Z.shape[1])
    G = (Z * wts) @ Z.T
    lam, q2, _, _, _ = choose_lambda(G, y)
    beta, _, _ = kernel_ridge_fit(Z, wts, y, lam)
    stab = bootstrap_sign_stability(Z, wts, y, lam, beta, n_boot=n_boot, rng=rng)
    return dict(rho=rho, p=p, p_fwer=p_fwer, q=q, ridge_coef=beta,
                ridge_stab=stab, ridge_q2=q2, ridge_lambda=lam)


# =============================================================================
# Objectives
# =============================================================================
@dataclass
class Objective:
    name: str
    sense: str               # 'max' or 'min'
    absolute: bool = False   # improvement as absolute difference instead of % of baseline
    raw: np.ndarray = field(repr=False, default=None)
    base: float = np.nan
    y: np.ndarray = field(repr=False, default=None)   # % improvement, +ve better


def parse_objectives(spec, df, weights):
    objs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        name, sense = parts[0], parts[1].lower()
        absolute = len(parts) > 2 and parts[2].lower() == "abs"
        if sense not in ("max", "min"):
            raise ValueError(f"objective {tok}: sense must be max or min")
        if name == "J" and "J" not in df.columns:
            if not {"CD", "PR"} <= set(df.columns):
                raise ValueError("J requested but CD/PR columns missing")
            w1, w2 = weights
            df["J"] = w1 * df["CD"] - w2 * df["PR"]     # as in the workbook: w1*cd - w2*pr
        if name not in df.columns:
            raise ValueError(f"objective column '{name}' not in table (have {list(df.columns)})")
        objs.append(Objective(name=name, sense=sense, absolute=absolute))
    return objs


def resolve_surface_selection(spec, df, resolve):
    """'all' -> None; 'T','U','C' or combos like 'TU' -> ids from the morph_config JSON column;
    '14,15' -> explicit ids."""
    spec = str(spec).strip()
    if spec.lower() == "all":
        return None
    if all(ch.isdigit() or ch in ", " for ch in spec):
        return sorted({int(t) for t in spec.replace(" ", ",").split(",") if t})
    if "morph_config" not in df.columns:
        raise SystemExit(f"--surfaces {spec} needs a 'morph_config' column (from-doe writes it)")
    key = {"T": "t_surfaces", "U": "u_surfaces", "C": "c_surfaces"}
    ids_seen = None
    for p in df.loc[df["morph_config"].notna(), "morph_config"]:
        with open(resolve(p)) as f:
            cfg = json.load(f)
        ids = sorted({int(i) for ch in spec.upper() for i in cfg.get(key[ch], [])})
        if ids_seen is None:
            ids_seen = ids
        elif ids != ids_seen:
            raise SystemExit(f"{p}: {spec} surfaces {ids} differ from other configs {ids_seen}")
    if not ids_seen:
        raise SystemExit(f"morph configs define no {spec} surfaces")
    return ids_seen


# =============================================================================
# VTK writer (legacy ASCII, no dependencies)
# =============================================================================
def write_vtk(path, nodes, tris, point_fields: dict, cell_fields: dict | None = None):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nsurface sensitivity\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(nodes)} double\n")
        np.savetxt(f, nodes, fmt="%.10e")
        f.write(f"POLYGONS {len(tris)} {4 * len(tris)}\n")
        np.savetxt(f, np.column_stack([np.full(len(tris), 3), tris]), fmt="%d")
        if cell_fields:
            f.write(f"CELL_DATA {len(tris)}\n")
            for k, v in cell_fields.items():
                f.write(f"SCALARS {k} double 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, np.nan_to_num(np.asarray(v, float), nan=0.0), fmt="%.6e")
        f.write(f"POINT_DATA {len(nodes)}\n")
        for k, v in point_fields.items():
            v = np.nan_to_num(np.asarray(v, float), nan=0.0)
            if v.ndim == 2:
                f.write(f"VECTORS {k} double\n")
                np.savetxt(f, v, fmt="%.6e")
            else:
                f.write(f"SCALARS {k} double 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, v, fmt="%.6e")


# =============================================================================
# Plotting
# =============================================================================
def _div_cmap():
    import matplotlib
    return matplotlib.colormaps["viridis"]


def plot_objective(out_png, obj, uv, tri_act, res, patch_main, grid_main, axes_lbl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from matplotlib.colors import TwoSlopeNorm

    cmap = _div_cmap()
    T = Triangulation(uv[:, 0], uv[:, 1], tri_act)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    title_skill = f"Q²(LOO) = {res['q2']:.2f}"
    fig.suptitle(f"{obj.name} ({'maximise' if obj.sense == 'max' else 'minimise'}); "
                 f"improvement = {'abs. diff' if obj.absolute else '% vs baseline'} [{res['transform']}], + is better;  nodal linear model {title_skill}"
                 + ("   [LOW SKILL - maps not interpretable]" if res["q2"] < 0.2 else ""),
                 fontsize=11)

    def panel(ax, val, ttl, cbl, mask=None, sig=None):
        v = np.asarray(val, float).copy()
        vmax = np.nanmax(np.abs(v)) if np.any(np.isfinite(v)) else 1.0
        vmax = vmax if vmax > 0 else 1.0
        v = np.nan_to_num(v, nan=0.0)
        tp = ax.tripcolor(T, v, shading="gouraud", cmap=cmap,
                          norm=TwoSlopeNorm(0.0, -vmax, vmax))
        if mask is not None:
            m = np.asarray(mask, float)
            ax.tricontourf(T, m, levels=[-0.5, 0.5], colors=["#fcfcfb"], alpha=0.75)
        if sig is not None and np.any(sig):
            ax.tricontour(T, sig.astype(float), levels=[0.5], colors="#1a1a19", linewidths=1.0)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel(axes_lbl[0], fontsize=8)
        ax.set_ylabel(axes_lbl[1], fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(tp, ax=ax, shrink=0.85)
        cb.set_label(cbl, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    panel(axs[0, 0], res["r"],
          f"(a) corr(δn, improvement)\ncontour: |r| > {res['r_crit']:.2f} (maxT, FWER 5%)",
          "r  (+: moving along +n improves; +n into fluid if oriented)", sig=np.abs(res["r"]) > res["r_crit"])
    bn = res["beta"] / (np.nanmax(np.abs(res["beta"])) or 1.0)
    panel(axs[0, 1], bn,
          f"(b) ridge sensitivity β(x), normalised\nfaded: bootstrap sign agreement < {res['stab_thr']:.0%}",
          "β/max|β|  (+: moving along +n improves)", mask=res["stab"] >= res["stab_thr"])
    panel(axs[1, 0], res["contrast"],
          f"(c) mean δn, best {res['n_top']} minus worst {res['n_top']} designs\n(descriptive, no significance test)",
          "Δ(δn)  [mesh length units]")

    # (d) patch grid
    ax = axs[1, 1]
    ax.triplot(T, color="#c9c8c4", lw=0.2)
    pm, (eu, ev, nu_e, nv_e) = patch_main, grid_main
    vmax = 1.0
    for k in range(nu_e * nv_e):
        if not pm["keep"][k]:
            continue
        iu, iv = divmod(k, nv_e)
        rho = pm["rho"][k]
        col = cmap((rho / vmax + 1) / 2)
        ax.add_patch(plt.Rectangle((eu[iu], ev[iv]), eu[iu + 1] - eu[iu], ev[iv + 1] - ev[iv],
                                   facecolor=col, edgecolor="#fcfcfb", lw=2, alpha=0.9))
        star = "**" if pm["q"][k] < 0.05 else ("*" if pm["p"][k] < 0.05 else "")
        stab = pm["ridge_stab"][k]
        ax.text((eu[iu] + eu[iu + 1]) / 2, (ev[iv] + ev[iv + 1]) / 2,
                f"{k}\n{rho:+.2f}{star}\n{stab:.0%}", ha="center", va="center", fontsize=6.5,
                color="#1a1a19")
    ax.set_xlim(eu[0], eu[-1]); ax.set_ylim(ev[0], ev[-1]); ax.set_aspect("equal")
    ax.set_title("(d) patches: id / Spearman ρ / ridge sign stability\n* p<.05 uncorrected, ** BH-FDR q<.05",
                 fontsize=10)
    ax.set_xlabel(axes_lbl[0], fontsize=8); ax.set_ylabel(axes_lbl[1], fontsize=8)
    ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(0, -1, 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.85); cb.set_label("Spearman ρ", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_loo(out_png, obj, y, yhat, labels, groups, transform="raw"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    cat = ["#2a78d6", "#e34948", "#0ca30c", "#8a5cd1", "#d98a12", "#1a9e9e"]
    ug = list(dict.fromkeys(groups))
    for gi, g in enumerate(ug):
        m = np.array([gg == g for gg in groups])
        ax.scatter(y[m], yhat[m], s=36, color=cat[gi % len(cat)], edgecolor="#fcfcfb",
                   linewidth=1.5, label=str(g), zorder=3)
    lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], color="#8a8985", lw=1, zorder=1)
    ax.axhline(0, color="#d9d8d4", lw=0.8, zorder=0); ax.axvline(0, color="#d9d8d4", lw=0.8, zorder=0)
    ax.set_xlabel(f"actual {obj.name} improvement [{'abs' if obj.absolute else '%'}; {transform}]")
    ax.set_ylabel("LOO predicted [%]")
    ax.set_title(f"{obj.name}: surface-linear model, leave-one-out")
    if len(ug) > 1:
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_objective_corr(out_png, df_obj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    C = df_obj.corr(method="spearman").values
    names = list(df_obj.columns)
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 1.0 * len(names) + 1.5),
                           constrained_layout=True)
    im = ax.imshow(C, cmap=_div_cmap(), norm=TwoSlopeNorm(0, -1, 1))
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="#fcfcfb" if abs(C[i, j]) > 0.6 else "#1a1a19")
    ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
    ax.set_title("Spearman correlation between improvements (+ = better)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================
def analyse(args):
    import pandas as pd
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.table)
    base_dir = os.path.dirname(os.path.abspath(args.table))
    for c in ("design_id", "fro_path", "is_baseline"):
        if c not in df.columns:
            raise SystemExit(f"table must contain column '{c}'")
    if "group" not in df.columns:
        df["group"] = "all"
    if int(df["is_baseline"].sum()) != 1:
        raise SystemExit("exactly one row must have is_baseline = 1")
    objs = parse_objectives(args.objectives, df, args.weights)
    if df["fro_path"].duplicated().any():
        raise SystemExit("duplicate fro_path entries: " +
                         str(df.loc[df["fro_path"].duplicated(keep=False), "fro_path"].tolist()))

    # drop rows with missing objective values (failed CFD) - reported, not silent
    obj_cols = [o.name for o in objs]
    bad = df[obj_cols].isna().any(axis=1) & (df["is_baseline"] == 0)
    if bad.any():
        log.warning("dropping %d designs with missing objectives: %s",
                    bad.sum(), df.loc[bad, "design_id"].tolist())
        df = df.loc[~bad].reset_index(drop=True)

    if args.lhs_only:
        if "is_lhs" not in df.columns:
            raise SystemExit("--lhs-only needs an 'is_lhs' column (from-doe writes it)")
        drop = (df["is_lhs"] == 0) & (df["is_baseline"] == 0)
        log.info("--lhs-only: excluding %d non-LHS designs: %s", drop.sum(),
                 df.loc[drop, "design_id"].tolist())
        df = df.loc[~drop].reset_index(drop=True)

    if args.include_batch or args.exclude_batch:
        if "batch" not in df.columns:
            raise SystemExit("--include-batch/--exclude-batch need a 'batch' column (from-doe writes it)")
        keep = df["is_baseline"] == 1
        if args.include_batch:
            keep |= df["batch"].isin(args.include_batch)
        else:
            keep |= ~df["batch"].isin(args.exclude_batch)
        log.info("batch filter: keeping %d of %d designs", int(keep.sum()) - 1,
                 int((df["is_baseline"] == 0).sum()))
        df = df.loc[keep].reset_index(drop=True)
    if int((df["is_baseline"] == 0).sum()) < 8:
        raise SystemExit("fewer than 8 designs left after filtering - nothing meaningful to fit")

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    surface_ids = resolve_surface_selection(args.surfaces, df, resolve)
    log.info("surface selection '%s' -> ids %s", args.surfaces,
             surface_ids if surface_ids is not None else "all")

    # ------------------------------------------------------------------ meshes
    ib = int(np.where(df["is_baseline"] == 1)[0][0])
    base = read_fro(resolve(df.loc[ib, "fro_path"]))
    log.info("baseline: %s  nodes=%d tris=%d", base.path, len(base.nodes), len(base.tris))
    vn, va = vertex_normals_areas(base.nodes, base.tris)
    if args.flip_normals:
        vn = -vn
    normal_note = "as stored in the .fro" + (" (flipped by --flip-normals)" if args.flip_normals else "")

    designs = df.loc[df["is_baseline"] == 0].reset_index(drop=True)
    n_d = len(designs)
    D = np.zeros((n_d, len(base.nodes), 3))
    for j, row in designs.iterrows():
        s = read_fro(resolve(row["fro_path"]))
        if s.nodes.shape != base.nodes.shape:
            raise SystemExit(f"{s.path}: node count {len(s.nodes)} != baseline {len(base.nodes)}")
        if s.tris.shape != base.tris.shape or not np.array_equal(s.tris, base.tris):
            raise SystemExit(f"{s.path}: connectivity differs from baseline - node-wise "
                             "differencing is invalid (was the surface remeshed?)")
        D[j] = s.nodes - base.nodes
        log.info("  read %-40s max|d|=%.3e", os.path.basename(s.path),
                 np.linalg.norm(D[j], axis=1).max())

    # restrict to selected surface IDs (e.g. the T surface from the morph config)
    node_ok = np.ones(len(base.nodes), bool)
    tri_sel = np.ones(len(base.tris), bool)
    if surface_ids is not None:
        tri_sel = np.isin(base.tri_sid, surface_ids)
        if not tri_sel.any():
            raise SystemExit(f"no triangles carry surface ids {surface_ids}")
        node_ok = np.zeros(len(base.nodes), bool)
        node_ok[np.unique(base.tris[tri_sel])] = True
    # make +n point into the FLUID, so '+' always means 'surface pushed into the flow'
    if args.normal_convention == "fluid":
        probe = np.where(node_ok)[0]
        sgn = fluid_side_sign(base.nodes, base.tris, vn, probe)
        if sgn == -1:
            vn = -vn
            normal_note = "flipped so +n points into the fluid (ray-parity test: file normals point into the solid)"
        elif sgn == +1:
            normal_note = "+n points into the fluid (ray-parity test; file orientation kept)"
        else:
            normal_note = ("fluid side INCONCLUSIVE (open surface or no farfield in the .fro) - "
                           "normals as stored; check the 'normal' vectors in the VTK")
        log.info("normals: %s", normal_note)

    # how much of the total shape change happens OUTSIDE the analysed surfaces
    # (normal component only: tangential node sliding does not change the shape)
    dn_all2 = np.einsum("jnk,nk->jn", D, vn) ** 2
    e_tot = (dn_all2 * va).sum(1)
    e_in = (dn_all2[:, node_ok] * va[node_ok]).sum(1)
    frac_outside = 1.0 - e_in / np.maximum(e_tot, 1e-300)
    del dn_all2

    dmag = np.linalg.norm(D, axis=2)                    # (n_d, N)
    dmax_node = dmag.max(0)
    active = node_ok & (dmax_node > args.active_tol * dmax_node.max())
    if active.sum() < 10:
        raise SystemExit("fewer than 10 active (moving) nodes - check inputs")
    Dn_full = np.einsum("jnk,nk->jn", D, vn)            # (n_d, N) normal disp
    Dt_full = np.linalg.norm(D - Dn_full[..., None] * vn[None], axis=2)

    # rigid-shift sanity: inactive nodes should not move
    moving_inactive = (~active) & node_ok & (dmax_node > 0)
    act_idx = np.where(active)[0]
    Dn = Dn_full[:, act_idx]
    a_act = va[act_idx]

    # per-design summary
    tan_ratio = (np.sqrt((Dt_full[:, act_idx] ** 2 * a_act).sum(1))
                 / np.maximum(np.sqrt((Dn ** 2 * a_act).sum(1)), 1e-300))
    dsum = pd.DataFrame({
        "design_id": designs["design_id"], "group": designs["group"],
        "max_disp": dmag.max(1), "rms_dn_active": np.sqrt((Dn ** 2 * a_act).sum(1) / a_act.sum()),
        "mean_dn_active": (Dn * a_act).sum(1) / a_act.sum(),
        "tangential_to_normal_L2": tan_ratio,
        "disp_energy_frac_outside_selection": frac_outside,
    })
    dsum.to_csv(os.path.join(args.outdir, "design_summary.csv"), index=False)

    # include baseline as a sample at d = 0, y = 0 (it is a genuine CFD point)
    labels = designs["design_id"].astype(str).tolist()
    groups = designs["group"].astype(str).tolist()
    if args.include_baseline:
        Dn = np.vstack([Dn, np.zeros((1, Dn.shape[1]))])
        labels.append("baseline"); groups.append("baseline")

    # objectives -> % improvement
    for o in objs:
        base_val = float(df.loc[ib, o.name])
        raw = designs[o.name].to_numpy(float)
        sgn = 1.0 if o.sense == "max" else -1.0
        if o.absolute or abs(base_val) < 1e-12:
            y = sgn * (raw - base_val)
        else:
            y = sgn * (raw - base_val) / abs(base_val) * 100.0
        if args.include_baseline:
            y = np.append(y, 0.0)
        o.raw, o.base, o.y = raw, base_val, y

    df_imp = pd.DataFrame({o.name: o.y for o in objs})
    plot_objective_corr(os.path.join(args.outdir, "fig_objectives.png"), df_imp)

    # ---------------------------------------------------------- projection/grid
    P = base.nodes[act_idx]
    e1, e2, o3 = plane_axes(P, a_act, args.grid_axes)
    uv = np.column_stack([(P - o3) @ e1, (P - o3) @ e2])
    axes_lbl = (f"u  along e1=({e1[0]:+.2f},{e1[1]:+.2f},{e1[2]:+.2f}), centred",
                f"v  along e2=({e2[0]:+.2f},{e2[1]:+.2f},{e2[2]:+.2f})")
    # triangles fully inside the active set, remapped to local ids
    loc = -np.ones(len(base.nodes), int); loc[act_idx] = np.arange(len(act_idx))
    tri_in = np.all(active[base.tris], axis=1)
    tri_act = loc[base.tris[tri_in]]
    # projection fold-over check (region not a graph over the plane)
    if len(tri_act):
        pu, pv = uv[tri_act[:, 0]], uv[tri_act[:, 1]]
        pw = uv[tri_act[:, 2]]
        s_area = 0.5 * ((pv[:, 0] - pu[:, 0]) * (pw[:, 1] - pu[:, 1])
                        - (pw[:, 0] - pu[:, 0]) * (pv[:, 1] - pu[:, 1]))
        folded = min((s_area > 0).mean(), (s_area < 0).mean())
    else:
        folded = 0.0

    grids = {}
    for tag, shift in (("main", (0.0, 0.0)), ("shifted", (0.5, 0.5))):
        pid, nu_e, nv_e, eu, ev = grid_patches(uv, args.grid[0], args.grid[1], shift)
        F, A, keep = patch_features(Dn, a_act, pid, nu_e * nv_e, args.min_patch_area)
        grids[tag] = dict(pid=pid, nu=nu_e, nv=nv_e, eu=eu, ev=ev, F=F, A=A, keep=keep)
    erank, svals = effective_rank(grids["main"]["F"][:, grids["main"]["keep"]])

    # ---------------------------------------------------------- per objective
    skill_rows = []
    point_fields = {"normal": vn, "vertex_area": va, "active": active.astype(float),
                    "patch_id_main": _scatter(len(base.nodes), act_idx, grids["main"]["pid"], -1),
                    "max_disp": dmax_node}
    report_obj = []
    # deformation magnitude per sample (area-weighted rms normal displacement on the
    # analysed surface); baseline sample has m = 0
    m_rms = np.sqrt((Dn ** 2 * a_act).sum(1) / a_act.sum())
    M = m_rms[:, None]          # a + b*m ; the quadratic term was unstable under LOO with n~20-35
    G_nodal = (Dn * a_act) @ Dn.T
    lam_grid = max(np.trace(_center_kernel_train(G_nodal)) / len(m_rms), 1e-300) * np.logspace(-4, 3, 40)
    for o in objs:
        y_raw = o.y.copy()
        log.info("objective %s: n=%d, improvement range [%.3f, %.3f]", o.name, len(y_raw),
                 y_raw.min(), y_raw.max())
        q2_mag, _ = magnitude_loo(M, y_raw)
        q2_comb, _, _ = combined_loo(G_nodal, M, y_raw, lam_grid)
        rho_mag = float(np.corrcoef(_rank(m_rms), _rank(y_raw))[0, 1])
        y = y_raw
        if args.control_magnitude:
            y = y - _ols_fit_pred(M, y, M)[0]      # direction effect at fixed deformation size
        if args.rank_y:
            y = normal_scores(y)
        o.y_used = y
        r = col_corr(Dn, y)
        r_crit, _ = maxT_threshold(Dn, y, n_perm=args.n_perm_nodal, rng=rng)
        G = G_nodal
        lam, q2, yhat, lams, q2s = choose_lambda(G, y)
        beta, _, _ = kernel_ridge_fit(Dn, a_act, y, lam)
        stab = bootstrap_sign_stability(Dn, a_act, y, lam, beta, n_boot=args.n_boot, rng=rng)
        order = np.argsort(y)
        n_top = max(3, len(y) // 3)
        contrast = Dn[order[-n_top:]].mean(0) - Dn[order[:n_top]].mean(0)
        res = dict(r=r, r_crit=r_crit, beta=beta, stab=stab, stab_thr=args.stab_thr,
                   contrast=contrast, q2=q2, n_top=n_top, q2_mag=q2_mag, q2_comb=q2_comb,
                   rho_mag=rho_mag, transform=_transform_label(args))

        # patches
        pm = {}
        for tag, g in grids.items():
            k = g["keep"]
            pa = patch_analysis(g["F"][:, k], y, args.n_perm_patch, args.n_boot, rng)
            full = {kk: _expand(v, k) for kk, v in pa.items() if isinstance(v, np.ndarray)}
            full.update(keep=k, ridge_q2=pa["ridge_q2"], ridge_lambda=pa["ridge_lambda"])
            pm[tag] = full
            iu, iv = np.divmod(np.arange(g["nu"] * g["nv"]), g["nv"])
            tab = pd.DataFrame({
                "patch": np.arange(g["nu"] * g["nv"]), "iu": iu, "iv": iv,
                "u_lo": g["eu"][iu], "u_hi": g["eu"][iu + 1],
                "v_lo": g["ev"][iv], "v_hi": g["ev"][iv + 1],
                "area_active": g["A"], "used": k,
                "mean_dn_range": _nanrange(g["F"]),
                "spearman_rho": full["rho"], "p_perm": full["p"], "p_fwer_maxT": full["p_fwer"],
                "q_BH": full["q"], "ridge_coef_std": full["ridge_coef"],
                "ridge_sign_stability": full["ridge_stab"],
            })
            tab = tab[tab["used"]].sort_values("p_perm")
            suffix = "" if tag == "main" else "_shifted"
            tab.to_csv(os.path.join(args.outdir, f"patch_stats_{o.name}{suffix}.csv"), index=False)
            pm[tag]["table"] = tab

        gm = grids["main"]
        plot_objective(os.path.join(args.outdir, f"fig_{o.name}.png"), o, uv, tri_act, res,
                       pm["main"], (gm["eu"], gm["ev"], gm["nu"], gm["nv"]), axes_lbl)
        plot_loo(os.path.join(args.outdir, f"fig_loo_{o.name}.png"), o, y, yhat, labels, groups,
                 res["transform"])

        point_fields[f"{o.name}_r"] = _scatter(len(base.nodes), act_idx, r, 0.0)
        point_fields[f"{o.name}_r_signif"] = _scatter(len(base.nodes), act_idx,
                                                      (np.abs(r) > r_crit).astype(float), 0.0)
        point_fields[f"{o.name}_beta_norm"] = _scatter(len(base.nodes), act_idx,
                                                       beta / (np.abs(beta).max() or 1), 0.0)
        point_fields[f"{o.name}_beta_stab"] = _scatter(len(base.nodes), act_idx, stab, 0.0)
        point_fields[f"{o.name}_contrast"] = _scatter(len(base.nodes), act_idx, contrast, 0.0)
        point_fields[f"{o.name}_patch_rho"] = _scatter(
            len(base.nodes), act_idx, pm["main"]["rho"][gm["pid"]], 0.0)

        # grid-robustness: fraction of significant nodes (by patch) agreeing across grids
        sig_main = _node_flag(pm["main"], grids["main"]["pid"], args.patch_alpha)
        sig_shift = _node_flag(pm["shifted"], grids["shifted"]["pid"], args.patch_alpha)
        union = sig_main | sig_shift
        agree = (sig_main & sig_shift).sum() / union.sum() if union.any() else np.nan

        skill_rows.append(dict(objective=o.name, sense=o.sense, baseline=o.base, n=len(y),
                               response=_transform_label(args),
                               rho_vs_deformation_size=rho_mag,
                               q2_loo_size_only=q2_mag,
                               q2_loo_size_plus_shape=q2_comb,
                               q2_loo_nodal=q2, lambda_nodal=lam, r_crit_maxT=r_crit,
                               frac_area_signif=float((a_act * (np.abs(r) > r_crit)).sum() / a_act.sum()),
                               q2_loo_patch=pm["main"]["ridge_q2"],
                               grid_agreement_jaccard=agree))
        report_obj.append((o, res, pm, agree))

    skill = pd.DataFrame(skill_rows)
    skill.to_csv(os.path.join(args.outdir, "model_skill.csv"), index=False)

    # --------------------------------------------- optional mode-coefficient check
    # Coefficients only mean the same shape WITHIN one parameterisation (batch), so the
    # linearity check and the empirical mode shapes are computed per batch, never pooled.
    coef_cols = [c for c in designs.columns if c.lower().startswith(args.coef_prefix)
                 and c[len(args.coef_prefix):].isdigit()]
    lin_parts = []
    if coef_cols:
        bcol = designs["batch"].astype(str) if "batch" in designs.columns else pd.Series(["all"] * n_d)
        k = len(coef_cols)
        Dn_d = Dn[:n_d]
        for b in dict.fromkeys(bcol):
            m = (bcol == b).to_numpy() & designs[coef_cols].notna().all(axis=1).to_numpy()
            if m.sum() < k + 4:
                lin_parts.append(f"{b}: n={m.sum()} too few for a {k}-mode fit")
                continue
            C1 = np.column_stack([np.ones(m.sum()), designs.loc[m, coef_cols].to_numpy(float)])
            coef, *_ = np.linalg.lstsq(C1, Dn_d[m], rcond=None)
            resid = Dn_d[m] - C1 @ coef
            den = (((Dn_d[m] - Dn_d[m].mean(0)) ** 2) * a_act).sum()
            r2 = 1 - (resid ** 2 * a_act).sum() / den
            lin_parts.append(f"{b}: n={m.sum()}, in-sample area-weighted R² of a linear "
                             f"coefficient->δn map = {r2:.4f}"
                             + (" (effectively linear)" if r2 > 0.99 else " (NONLINEAR: RBF/"
                                "propagation adds shape not in the modes)"))
            tag = "".join(ch if ch.isalnum() else "_" for ch in b)
            for j, c in enumerate(coef_cols):
                point_fields[f"emode_{tag}_{c}"] = _scatter(len(base.nodes), act_idx,
                                                            coef[j + 1], 0.0)
    lin_msg = ("; ".join(lin_parts) if lin_parts else "no coefficient columns supplied")

    # VTK: selected surfaces only (the full corner mesh would be ~100 MB of ASCII)
    sel_nodes = np.unique(base.tris[tri_sel])
    remap = -np.ones(len(base.nodes), int); remap[sel_nodes] = np.arange(len(sel_nodes))
    write_vtk(os.path.join(args.outdir, "sensitivity_surface.vtk"), base.nodes[sel_nodes],
              remap[base.tris[tri_sel]],
              {k: (v[sel_nodes]) for k, v in point_fields.items()},
              cell_fields={"surface_id": base.tri_sid[tri_sel]})

    # ------------------------------------------------------------------ report
    args._normal_note = normal_note
    _write_report(args, df, designs, objs, report_obj, skill, dsum, active, a_act, va,
                  erank, svals, folded, moving_inactive, lin_msg, e1, e2, n_d)
    log.info("done in %.1f s -> %s", time.time() - t0, args.outdir)


def _transform_label(args):
    t = []
    if args.control_magnitude:
        t.append("size-controlled")
    if args.rank_y:
        t.append("rank")
    return "+".join(t) if t else "raw"


def _scatter(N, idx, vals, fill):
    out = np.full(N, float(fill))
    out[idx] = vals
    return out


def _nanrange(F):
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return np.nanmax(F, 0) - np.nanmin(F, 0)


def _expand(v, keep):
    out = np.full(len(keep), np.nan)
    out[keep] = v
    return out


def _node_flag(pm, pid, alpha):
    flag_patch = (pm["p"] < alpha) & (pm["ridge_stab"] >= 0.9)
    flag_patch = np.nan_to_num(flag_patch.astype(float)) > 0
    return flag_patch[pid]


def _write_report(args, df, designs, objs, report_obj, skill, dsum, active, a_act, va,
                  erank, svals, folded, moving_inactive, lin_msg, e1, e2, n_d):
    normal_note = getattr(args, "_normal_note", "as stored")
    L = []
    L.append("# Surface sensitivity report\n")
    L.append(f"- designs analysed: {n_d} (+ baseline as a zero-displacement sample: "
             f"{'yes' if args.include_baseline else 'no'})")
    L.append(f"- active (moving) nodes: {active.sum()} / {len(active)}; "
             f"active area fraction {a_act.sum() / va.sum():.3f}")
    L.append(f"- projection axes: e1={np.round(e1, 3).tolist()}, e2={np.round(e2, 3).tolist()}; "
             f"fold-over fraction of active triangles in projection = {folded:.3f}"
             + ("  **WARNING: region is not a graph over this plane - 2-D maps/patches "
                "mix overlapping surfaces; use --grid-axes or --surface-ids**" if folded > 0.05 else ""))
    L.append(f"- patch grid {args.grid[0]}x{args.grid[1]}: effective rank of patch-displacement "
             f"matrix = {erank} (singular values: {np.round(svals[:10] / svals[0], 3).tolist()} ...). "
             "Patches cannot be separated beyond this many independent directions.")
    if moving_inactive.any():
        L.append(f"- note: {moving_inactive.sum()} nodes move by less than the active tolerance "
                 "but are non-zero (RBF tail / centroid correction).")
    L.append(f"- tangential/normal displacement ratio (area-L2), median over designs: "
             f"{dsum['tangential_to_normal_L2'].median():.3f}. Tangential motion is node sliding "
             "within the surface: it does not change the shape except where it moves the edges "
             "of the analysed surface, which the normal component cannot see.")
    L.append(f"- analysed surfaces: {args.surfaces}; share of normal-displacement energy "
             f"(area-weighted δn²) OUTSIDE them: median {dsum['disp_energy_frac_outside_selection'].median():.1%}, "
             f"max {dsum['disp_energy_frac_outside_selection'].max():.1%}"
             + ("  **large - the U surfaces carry a real part of the shape change that this "
                "analysis ignores; rerun with --surfaces TU to check**"
                if dsum['disp_energy_frac_outside_selection'].median() > 0.25 else ""))
    L.append(f"- LHS only: {'yes' if args.lhs_only else 'no (all designs incl. optimiser generations)'}")
    L.append(f"- **normal convention: {normal_note}.** δn > 0 = {'surface moved into the flow' if 'into the fluid' in normal_note and 'INCONCLUSIVE' not in normal_note else 'along the stored normal'}.")
    L.append(f"- mode coefficients (per parameterisation batch): {lin_msg}\n")
    L.append("## Predictive skill (read this first)\n")
    L.append(skill.to_markdown(index=False, floatfmt=".3g") if hasattr(skill, "to_markdown")
             else skill.to_string(index=False))
    L.append("\nHow to read this table:\n"
             "- rho_vs_deformation_size: Spearman correlation of the raw improvement with the rms "
             "normal deformation. Strongly negative means the baseline is near a local optimum for that "
             "objective: 'how much you deform' matters more than 'where'. A linear sensitivity map "
             "is then the wrong model on raw data - use --control-magnitude.\n"
             "- q2_loo_size_only: skill of a model that knows ONLY the deformation size (a + b·m).\n"
             "- q2_loo_size_plus_shape: size model + linear surface functional, refitted in every "
             "fold. Shape/location information is only demonstrated if this clearly beats size_only.\n"
             "- q2_loo_nodal: skill of the linear surface model on the response actually mapped "
             "(see 'response').\n"
             "- Q² ≤ 0.2 → the maps for that objective are noise. grid_agreement_jaccard < 0.5 → "
             "'critical patches' depend on where the grid lines fall.\n")
    for o, res, pm, agree in report_obj:
        L.append(f"## {o.name} ({o.sense}), baseline = {o.base:.5g}\n")
        u = "" if o.absolute else " %"
        L.append(f"- improvement range {o.y.min():+.3g}{u} … {o.y.max():+.3g}{u}, "
                 f"{int((o.y[:n_d] > 0).sum())}/{n_d} designs better than baseline")
        L.append(f"- response mapped: {res['transform']}; ρ(improvement, deformation size) = "
                 f"{res['rho_mag']:+.2f}; Q² size-only {res['q2_mag']:.3f} vs size+shape "
                 f"{res['q2_comb']:.3f}")
        L.append(f"- nodal: Q²_LOO = {res['q2']:.3f}; maxT |r| threshold = {res['r_crit']:.3f}; "
                 f"max |r| = {np.nanmax(np.abs(res['r'])):.3f}")
        tab = pm["main"]["table"]
        top = tab[(tab["p_perm"] < args.patch_alpha)].head(8)
        if len(top):
            L.append("- patches with p < %.2f (main grid):\n" % args.patch_alpha)
            cols = ["patch", "iu", "iv", "spearman_rho", "p_perm", "q_BH", "p_fwer_maxT",
                    "ridge_coef_std", "ridge_sign_stability"]
            L.append(top[cols].to_markdown(index=False, floatfmt=".3g")
                     if hasattr(top, "to_markdown") else top[cols].to_string(index=False))
        else:
            L.append("- no patch reaches p < %.2f" % args.patch_alpha)
        L.append(f"- main vs shifted grid agreement (Jaccard of flagged area): {agree:.2f}\n")
    with open(os.path.join(args.outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# =============================================================================
# xlsx -> tidy csv for the DSI_Optimisation_Results layout
# =============================================================================
def xlsx2csv(args):
    """Parse the 'FP M1.x' sheet layout:
       block 1 rows start at 'CO 1 #', block 2 at 'CO 2 #', baseline under 'Baseline'.
       Columns B..O(or N) = main group, the column holding a text label in the id row
       (e.g. 'CO 2', 'CO cd') starts an extra group to its right."""
    import openpyxl
    import pandas as pd
    wb = openpyxl.load_workbook(args.xlsx, data_only=True)
    ws = wb[args.sheet]
    rows = [[c.value for c in r] for r in ws.iter_rows()]
    keymap = {"Pressure Recovery": "PR", "Normalised Drag": "CD", "Distortion": "DC", "Swirl": "SW"}

    out = []
    for r0, row in enumerate(rows):
        lab = row[0]
        if not (isinstance(lab, str) and lab.strip().startswith("CO") and lab.strip().endswith("#")):
            continue
        main_group = lab.replace("#", "").replace(" ", "").strip()     # CO1 / CO2
        opt_row = rows[r0 + 1]
        # collect the metric rows that follow (until the next blank label block)
        metrics = {}
        for rr in range(r0 + 2, min(r0 + 12, len(rows))):
            name = rows[rr][0]
            if isinstance(name, str) and name.strip() in keymap and keymap[name.strip()] not in metrics:
                metrics[keymap[name.strip()]] = rows[rr]
            if isinstance(name, str) and name.strip().startswith("CO") and name.strip().endswith("#"):
                break
        group = main_group
        n_in_group = 0
        for c in range(1, len(row)):
            v = row[c]
            if isinstance(v, str):                 # e.g. 'CO 2' / 'CO cd' -> new group
                group = v.replace(" ", "")
                n_in_group = 0
                continue
            if v is None:
                continue
            n_in_group += 1
            rec = dict(group=group, co=int(v), opt=opt_row[c], n=n_in_group)
            for k, mrow in metrics.items():
                rec[k] = mrow[c]
            if all(rec.get(k) is None for k in keymap.values()):
                continue
            out.append(rec)
    # baseline
    for r0, row in enumerate(rows):
        if any(isinstance(v, str) and v.strip() == "Baseline" for v in row[:3]):
            rec = dict(group="baseline", co=0, opt=rows[r0 + 1][1], n=0)
            for rr in range(r0 + 1, r0 + 10):
                name = rows[rr][0]
                if isinstance(name, str) and name.strip() in keymap:
                    rec[keymap[name.strip()]] = rows[rr][1]
            out.append(rec)
            break
    df = pd.DataFrame(out)
    df["is_baseline"] = (df["group"] == "baseline").astype(int)
    df["design_id"] = np.where(df["is_baseline"] == 1, "baseline",
                               df["group"] + "_" + df["co"].astype(str))
    fmt = lambda r: (args.baseline_fro if r["is_baseline"] else
                     args.fro_pattern.format(co=r["co"], opt=r["opt"], group=r["group"], n=r["n"]))
    df["fro_path"] = df.apply(fmt, axis=1)
    dup = df["fro_path"].duplicated(keep=False)
    if dup.any():
        print("WARNING: several rows map to the same file (CO numbers repeat across groups - "
              "include {group} in --fro-pattern):\n" + df.loc[dup, ["design_id", "fro_path"]].to_string())
    cols = ["design_id", "group", "co", "opt", "n", "is_baseline", "fro_path", "PR", "CD", "DC", "SW"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\n{len(df) - 1} designs + baseline -> {args.out}\n"
          "CHECK the fro_path column: formulas are evaluated from cached values, "
          "and the file-name mapping is an assumption.")


# =============================================================================
# DOE folder -> analysis table  (Aeropt2 layout: <root>/<n>/<name>.fro + morph_config_n_*.json)
# =============================================================================
RESULT_ALIASES = {
    "pressure recovery": "PR", "pr": "PR",
    "normalised drag": "CD", "normalized drag": "CD", "cd": "CD", "drag": "CD",
    "weighted combination": "J", "j": "J",
    "dc60": "DC", "distortion": "DC", "dc": "DC",
    "swirl": "SW", "sw": "SW",
}


def read_results_sheet(xlsx, sheet):
    """Design number in column A, metric names in the header row."""
    import openpyxl
    import pandas as pd
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    if sheet not in wb.sheetnames:
        raise SystemExit(f"sheet '{sheet}' not in {wb.sheetnames}")
    rows = [r for r in wb[sheet].iter_rows(values_only=True) if any(v is not None for v in r)]
    hdr_i = next(i for i, r in enumerate(rows)
                 if sum(isinstance(v, str) for v in r[1:]) >= 2)
    hdr = rows[hdr_i]
    cols = {}
    for j, h in enumerate(hdr[1:], start=1):
        if isinstance(h, str) and h.strip():
            cols[j] = RESULT_ALIASES.get(h.strip().lower(),
                                         "".join(c if c.isalnum() else "_" for c in h.strip()))
    recs = []
    for r in rows[hdr_i + 1:]:
        if r[0] is None:
            continue
        try:
            did = int(r[0])
        except (TypeError, ValueError):
            continue
        rec = {"design_id": did}
        rec.update({name: r[j] for j, name in cols.items()})
        recs.append(rec)
    return pd.DataFrame(recs), {hdr[j]: n for j, n in cols.items()}


def basis_check(root, tab):
    """Is control-node displacement a fixed linear map of the modal coefficients within each batch,
    and is it the SAME map across batches?  (Laplacian eigenvectors are only defined up to sign /
    rotation within degenerate eigenspaces, so rebuilding the basis can silently change it.)"""
    from scipy.linalg import subspace_angles
    C, Dv, grp = [], [], []
    for _, r in tab[tab["is_baseline"] == 0].iterrows():
        if not isinstance(r.get("morph_config"), str):
            continue
        with open(os.path.join(root, r["morph_config"])) as f:
            cfg = json.load(f)
        if not cfg.get("modal_coeffs") or not cfg.get("displacement_vector"):
            continue
        C.append(np.ravel(cfg["modal_coeffs"]).astype(float))
        Dv.append(np.ravel(cfg["displacement_vector"]).astype(float))
        grp.append(r["group"])
    if not C:
        return
    C, Dv, grp = np.array(C), np.array(Dv), np.array(grp)
    k = C.shape[1]
    lines = ["basis consistency check (control-node displacement vs modal coefficients)"]
    maps = {}
    for g in dict.fromkeys(grp):
        m = grp == g
        if m.sum() >= k + 1:
            A, *_ = np.linalg.lstsq(C[m], Dv[m], rcond=None)
            rel = np.linalg.norm(Dv[m] - C[m] @ A) / np.linalg.norm(Dv[m])
            maps[g] = A
            lines.append(f"  {g}: n={m.sum()}, relative residual of exact linear map = {rel:.2e}")
    names = list(maps)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ang = np.degrees(subspace_angles(maps[names[i]].T, maps[names[j]].T))
            rel = np.linalg.norm(maps[names[i]] - maps[names[j]]) / np.linalg.norm(maps[names[i]])
            lines.append(f"  {names[i]} vs {names[j]}: ||A1-A2||/||A1|| = {rel:.3f}; principal "
                         f"angles between spans (deg) = {np.round(ang, 2).tolist()}"
                         + ("  -> DIFFERENT bases: pool these batches only in surface space, "
                            "never in coefficient space" if rel > 1e-3 else "  -> same basis"))
    for g in dict.fromkeys(grp):
        if g in maps:
            continue
        m = grp == g
        errs = {n: np.linalg.norm(Dv[m] - C[m] @ A, axis=1) / np.linalg.norm(Dv[m], axis=1)
                for n, A in maps.items()}
        best = min(errs, key=lambda n: errs[n].mean())
        lines.append(f"  {g}: n={m.sum()} (too few to identify); closest basis {best}, "
                     f"relative mismatch {np.round(errs[best], 3).tolist()}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(root, "basis_check.txt"), "w") as f:
        f.write(txt + "\n")


def from_doe(args):
    import glob
    import pandas as pd
    root = os.path.abspath(args.doe_root)
    res, mapping = read_results_sheet(args.results, args.sheet)
    print(f"results sheet '{args.sheet}': {len(res)} rows, columns mapped {mapping}")
    recs = []
    for d in sorted((p for p in os.listdir(root) if p.isdigit() and os.path.isdir(os.path.join(root, p))),
                    key=int):
        full = os.path.join(root, d)
        fros = sorted(glob.glob(os.path.join(full, "*.fro")))
        cfgs = sorted(glob.glob(os.path.join(full, "morph_config*.json")))
        rec = {"design_id": int(d), "is_baseline": int(int(d) == args.baseline)}
        if len(fros) != 1:
            print(f"  WARNING folder {d}: {len(fros)} .fro files {fros} - skipped")
            continue
        rec["fro_path"] = os.path.relpath(fros[0], root)
        if cfgs:
            if len(cfgs) > 1:
                print(f"  WARNING folder {d}: several morph configs, using {cfgs[0]}")
            with open(cfgs[0]) as f:
                cfg = json.load(f)
            rec["morph_config"] = os.path.relpath(cfgs[0], root)
            rec["cfg_n"], rec["cfg_gen"] = cfg.get("n"), cfg.get("gen")
            rec["batch"] = os.path.basename(str(cfg.get("output_directory", "")).rstrip("/"))
            rec["group"] = f"{rec['batch']}_gen{rec['cfg_gen']}"
            rec["is_lhs"] = int(cfg.get("gen", 0) == 0)
            for k, c in enumerate(np.ravel(cfg.get("modal_coeffs") or []), start=1):
                rec[f"c{k}"] = float(c)
            m = os.path.basename(cfgs[0])
            tag = m.replace("morph_config_n_", "").replace(".json", "")
            if tag.isdigit() and int(tag) != int(d):
                print(f"  note folder {d}: config file is named {m} (content n={cfg.get('n')}, "
                      f"gen={cfg.get('gen')}) - using it")
        elif not rec["is_baseline"]:
            print(f"  WARNING folder {d}: no morph_config json")
        if rec["is_baseline"]:
            rec.update(group="baseline", is_lhs=1)
        recs.append(rec)
    tab = pd.DataFrame(recs).merge(res, on="design_id", how="left", validate="1:1")
    miss = tab.loc[tab[[c for c in mapping.values()]].isna().all(axis=1), "design_id"].tolist()
    if miss:
        print(f"  WARNING no results for designs {miss}")
    if int(tab["is_baseline"].sum()) != 1:
        raise SystemExit(f"baseline folder {args.baseline} not found")
    # consistency: weighted objective vs PR/CD
    if {"J", "PR", "CD"} <= set(tab.columns):
        jj = args.weights[0] * tab["CD"] - args.weights[1] * tab["PR"]
        err = np.nanmax(np.abs(jj - tab["J"]))
        print(f"  check: max |J - ({args.weights[0]}*CD - {args.weights[1]}*PR)| = {err:.2e}")
    basis_check(root, tab)
    out = args.out or os.path.join(root, "designs.csv")
    tab.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tab.drop(columns=[c for c in tab.columns if c.startswith("c") and c[1:].isdigit()])
              .to_string(index=False))
    print(f"\nwrote {out}  ({int((tab.is_baseline == 0).sum())} designs, "
          f"{int(((tab.is_baseline == 0) & (tab.is_lhs == 1)).sum())} from LHS)")


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyse", help="run the sensitivity analysis")
    a.add_argument("table", help="CSV: design_id, fro_path, is_baseline, objective columns")
    a.add_argument("--outdir", default="sens_out")
    a.add_argument("--objectives", default="PR:max,CD:min,J:min,DC:min",
                   help="comma list NAME:max|min[:abs]; ':abs' = absolute difference instead of %% "
                        "(use for quantities with a near-zero baseline, e.g. swirl). "
                        "J = w1*CD - w2*PR is built if absent")
    a.add_argument("--weights", nargs=2, type=float, default=[0.25, 0.75], help="w1 (CD) w2 (PR)")
    a.add_argument("--grid", nargs=2, type=int, default=[8, 4], help="patches along e1, e2")
    a.add_argument("--grid-axes", default="auto", help="'auto' (principal axes) or e.g. 'xy','xz'")
    a.add_argument("--surfaces", default="T",
                   help="'T' (default: t_surfaces from the morph configs), 'TU', 'all', "
                        "or explicit ids '14,15'")
    a.add_argument("--control-magnitude", action="store_true",
                   help="map the response after removing a + b*m (m = rms normal "
                        "deformation): isolates WHERE/which-sign effects from 'deformed more'")
    a.add_argument("--rank-y", action="store_true",
                   help="use normal scores of the ranks of the response (robust to outlier designs)")
    a.add_argument("--include-batch", nargs="*", default=None,
                   help="keep only designs whose 'batch' (morph output_directory name) is listed")
    a.add_argument("--exclude-batch", nargs="*", default=None,
                   help="drop designs whose 'batch' is listed (e.g. the old parameterisation)")
    a.add_argument("--lhs-only", action="store_true",
                   help="exclude designs with is_lhs = 0 (optimiser generations)")
    a.add_argument("--active-tol", type=float, default=1e-3,
                   help="node is active if max|d| > tol * global max|d|")
    a.add_argument("--min-patch-area", type=float, default=0.01,
                   help="drop patches with < this fraction of active area")
    a.add_argument("--flip-normals", action="store_true")
    a.add_argument("--normal-convention", choices=["fluid", "file"], default="fluid",
                   help="'fluid' (default): orient +n into the flow via a ray-parity test on the "
                        "closed .fro boundary; 'file': keep the stored orientation")
    a.add_argument("--no-baseline-sample", dest="include_baseline", action="store_false")
    a.add_argument("--n-perm-nodal", type=int, default=2000)
    a.add_argument("--n-perm-patch", type=int, default=5000)
    a.add_argument("--n-boot", type=int, default=500)
    a.add_argument("--stab-thr", type=float, default=0.9)
    a.add_argument("--patch-alpha", type=float, default=0.05)
    a.add_argument("--coef-prefix", default="c")
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=analyse)

    x = sub.add_parser("xlsx2csv", help="convert the results workbook to the analysis table")
    x.add_argument("xlsx")
    x.add_argument("--sheet", default="FP M1.6 WAT6")
    x.add_argument("--out", default="designs.csv")
    x.add_argument("--fro-pattern", default="surfaces/{group}_{co}.fro",
                   help="python format string; fields {co} {opt} {group} {n}")
    x.add_argument("--baseline-fro", default="surfaces/baseline.fro")
    x.set_defaults(func=xlsx2csv)

    d = sub.add_parser("from-doe", help="build the analysis table from an Aeropt2 DOE folder")
    d.add_argument("doe_root", help="folder containing 1/, 2/, ... (one .fro + morph_config each)")
    d.add_argument("--results", required=True, help="results workbook")
    d.add_argument("--sheet", default="Results")
    d.add_argument("--baseline", type=int, default=1, help="folder number of the baseline")
    d.add_argument("--weights", nargs=2, type=float, default=[0.25, 0.75])
    d.add_argument("--out", default=None, help="default: <doe_root>/designs.csv")
    d.set_defaults(func=from_doe)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
