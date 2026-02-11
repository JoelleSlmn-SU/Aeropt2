# MeshGeneration/pcaBasis.py
"""
PCA basis utilities for control-node displacement dimensionality reduction.

We compute PCA via SVD on a row-stacked data matrix:

    X ∈ R^{M x d}  where d = 3 * N_control_nodes

Each row is one flattened control-node displacement field (or any other
consistent representation). PCA is computed on centered data Xc = X - mean.

Outputs:
    mean  : (d,)
    V     : (d, k)   principal directions (orthonormal)
    sigma : (k,)     singular values (sqrt of variance * sqrt(M-1))

We store enough metadata to safely reuse / invalidate caches when the
control-node set or basis-generation settings change.
"""
from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_signature(
    control_nodes: np.ndarray,
    control_normals: Optional[np.ndarray],
    *,
    normal_project: bool,
    k_modes: int,
    knn: int,
    spectral_p: float,
    coeff_frac: float,
    amp_alpha: float,
    t_patch_scale: Optional[float],
    train_M: int,
    energy: float,
    k_red: Optional[int],
    version: str = "pcaBasis_v1",
) -> str:
    """
    Create a robust signature hash for a PCA cache.
    Any change in these inputs should force a rebuild.

    NOTE: we round floating arrays to reduce sensitivity to tiny float noise.
    """
    cn = np.asarray(control_nodes, float)
    cn_r = np.round(cn, decimals=10).tolist()

    if control_normals is None:
        nn_r = None
    else:
        nn = np.asarray(control_normals, float)
        nn_r = np.round(nn, decimals=10).tolist()

    payload = {
        "version": version,
        "control_nodes": cn_r,
        "control_normals": nn_r if normal_project else None,
        "normal_project": bool(normal_project),
        "k_modes": int(k_modes),
        "knn": int(knn),
        "spectral_p": float(spectral_p),
        "coeff_frac": float(coeff_frac),
        "amp_alpha": float(amp_alpha),
        "t_patch_scale": None if t_patch_scale is None else float(t_patch_scale),
        "train_M": int(train_M),
        "energy": float(energy),
        "k_red": None if k_red is None else int(k_red),
    }

    s = _stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


@dataclass
class PCABasis:
    mean: np.ndarray           # (d,)
    V: np.ndarray              # (d, k)
    sigma: np.ndarray          # (k,)
    explained: np.ndarray      # (k,) fraction
    signature: str
    meta: Dict[str, Any]


def build_pca_basis(
    X: np.ndarray,
    *,
    energy: float = 0.99,
    k_red: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA (via SVD) on X (M x d) and return (mean, V, sigma, explained).

    - energy: if k_red is None, keep smallest k such that cumulative explained >= energy
    - k_red: explicit number of retained components (overrides energy)
    """
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (Mxd), got shape {X.shape}")
    M, d = X.shape
    if M < 2:
        raise ValueError("Need at least 2 samples to build PCA basis")

    mean = X.mean(axis=0)
    Xc = X - mean

    # Economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Vt: (r, d), S: (r,)
    # Variance explained per component: S^2 / sum(S^2)
    var = S**2
    total = var.sum()
    explained = (var / total) if total > 0 else np.zeros_like(var)

    if k_red is None:
        energy = float(energy)
        energy = min(max(energy, 0.0), 1.0)
        cum = np.cumsum(explained)
        k = int(np.searchsorted(cum, energy) + 1) if cum.size else 1
    else:
        k = int(k_red)

    k = max(1, min(k, Vt.shape[0]))
    V = Vt[:k].T  # (d, k)
    sigma = S[:k]  # (k,)
    explained_k = explained[:k]

    return mean, V, sigma, explained_k


def save_pca_basis(
    path,
    mean,
    V,
    sigma,
    explained,
    signature=None,
    meta=None,
):
    import numpy as np

    payload = {
        "mean": mean,
        "V": V,
        "sigma": sigma,
        "explained": explained,
        "meta": meta or {},
    }

    if signature is not None:
        payload["meta"]["signature"] = signature

    np.savez(path, **payload)

def load_pca_basis(path: str) -> PCABasis:
    z = np.load(path, allow_pickle=False)
    mean = z["mean"]
    V = z["V"]
    sigma = z["sigma"]
    explained = z.get("explained", np.zeros_like(sigma))
    signature = str(z["signature"][0]) if "signature" in z else ""
    meta = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"][0]))
        except Exception:
            meta = {}
    return PCABasis(mean=mean, V=V, sigma=sigma, explained=explained, signature=signature, meta=meta)


def ensure_pca_cache(
    cache_path: str,
    *,
    signature: str,
    X_builder,
    energy: float = 0.99,
    k_red: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> PCABasis:
    """
    Ensure a PCA cache exists at cache_path with the given signature.
    If missing or signature mismatch, rebuild using X_builder() -> X (M×d).

    X_builder should be a callable with no args.
    """
    if os.path.exists(cache_path):
        try:
            p = load_pca_basis(cache_path)
            if p.signature == signature:
                return p
        except Exception:
            pass

    X = X_builder()
    mean, V, sigma, explained = build_pca_basis(X, energy=energy, k_red=k_red)
    p = PCABasis(
        mean=mean, V=V, sigma=sigma, explained=explained,
        signature=signature,
        meta=meta or {}
    )
    save_pca_basis(cache_path, p)
    return p


def reconstruct_disp_flat(mean: np.ndarray, V: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Reconstruct a flattened displacement vector:
        d = mean + V @ a
    """
    mean = np.asarray(mean, float).reshape(-1)
    V = np.asarray(V, float)
    a = np.asarray(a, float).reshape(-1)
    if V.shape[1] != a.size:
        raise ValueError(f"PCA coeff length {a.size} != V.shape[1] {V.shape[1]}")
    return mean + V @ a
