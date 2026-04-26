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


def build_pca_cache(
    *,
    output_dir: str,
    control_nodes: np.ndarray,
    normals: Optional[np.ndarray],
    get_displacements_fn,
    k_modes: int,
    M: int,
    energy: float = 0.99,
    k_red: Optional[int] = None,
    seed: int = 0,
    normal_project: bool = True,
    t_patch_scale: Optional[float] = None,
    amp_alpha: float = 0.02,
    vector_mode: str = "local_frame",
    frame_knn: int = 12,
    global_modes: bool = False,
    global_mode_config: Optional[list] = None,
    basis_axes: Optional[list] = None,
    use_local_modes: bool = True,
    global_only: bool = False,
    cache_name: str = "pca_basis.npz",
    cache_subdir: str = os.path.join("Control Nodes", "pca"),
    signature_version: str = "pcaBasis_v2",
) -> Dict[str, Any]:
    """
    High-level helper to build/reuse a PCA cache for control-node displacement fields.

    Parameters
    ----------
    get_displacements_fn : callable
        Usually MeshGeneration.controlNodeDisp.getDisplacements
    M : int
        Number of random training samples.
    """
    import os
    import json
    import hashlib
    import numpy as np

    cn = np.asarray(control_nodes, float).reshape((-1, 3))
    nn = None if normals is None else np.asarray(normals, float).reshape((-1, 3))

    if cn.ndim != 2 or cn.shape[1] != 3:
        raise ValueError(f"control_nodes must have shape (N,3), got {cn.shape}")
    if nn is not None and nn.shape != cn.shape:
        raise ValueError(f"normals shape {nn.shape} does not match control_nodes shape {cn.shape}")
    if int(M) < 2:
        raise ValueError("M must be at least 2 to build a PCA basis")

    cache_dir = os.path.join(output_dir, cache_subdir)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    # ------------------------------------------------------------------
    # robust signature: base signature + extra settings not currently
    # covered by make_signature()
    # ------------------------------------------------------------------
    base_sig = make_signature(
        control_nodes=cn,
        control_normals=nn,
        normal_project=bool(normal_project),
        k_modes=int(k_modes),
        knn=int(frame_knn),
        spectral_p=1.0,          # placeholder if random coeff generation is internal
        coeff_frac=1.0,          # placeholder if random coeff generation is internal
        amp_alpha=float(amp_alpha),
        t_patch_scale=t_patch_scale,
        train_M=int(M),
        energy=float(energy),
        k_red=k_red,
        version=signature_version,
    )

    extra_payload = {
        "vector_mode": vector_mode,
        "global_modes": bool(global_modes),
        "global_mode_config": global_mode_config or [],
        "basis_axes": basis_axes,
        "use_local_modes": bool(use_local_modes),
        "global_only": bool(global_only),
    }
    extra_sig = hashlib.sha256(
        json.dumps(extra_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    signature = hashlib.sha256((base_sig + "::" + extra_sig).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # build training matrix X
    # ------------------------------------------------------------------
    def X_builder():
        X_rows = []
        for i in range(int(M)):
            d = get_displacements_fn(
                output_dir=output_dir,
                seed=int(seed) + i,
                control_nodes=cn,
                normals=nn,
                coeffs=None,
                k_modes=int(k_modes),
                normal_project=bool(normal_project),
                t_patch_scale=t_patch_scale,
                amp_alpha=float(amp_alpha),
                vector_mode=vector_mode,
                frame_knn=int(frame_knn),
                global_modes=bool(global_modes),
                global_mode_config=global_mode_config or [],
                basis_axes=basis_axes,
                use_local_modes=bool(use_local_modes),
                global_only=bool(global_only),
            )
            d = np.asarray(d, float).reshape(-1)
            X_rows.append(d)

        X = np.vstack(X_rows)
        return X

    meta = {
        "signature": signature,
        "train_M": int(M),
        "energy": float(energy),
        "k_red_requested": None if k_red is None else int(k_red),
        "normal_project": bool(normal_project),
        "vector_mode": vector_mode,
        "k_modes": int(k_modes),
        "frame_knn": int(frame_knn),
        "amp_alpha": float(amp_alpha),
        "t_patch_scale": None if t_patch_scale is None else float(t_patch_scale),
        "global_modes": bool(global_modes),
        "global_mode_config": global_mode_config or [],
        "basis_axes": basis_axes,
        "use_local_modes": bool(use_local_modes),
        "global_only": bool(global_only),
        "n_control_nodes": int(cn.shape[0]),
    }

    p = ensure_pca_cache(
        cache_path,
        signature=signature,
        X_builder=X_builder,
        energy=float(energy),
        k_red=k_red,
        meta=meta,
    )

    return {
        "cache_path": cache_path,
        "signature": p.signature,
        "k_red": int(p.V.shape[1]),
        "mean_shape": tuple(p.mean.shape),
        "V_shape": tuple(p.V.shape),
        "sigma_shape": tuple(p.sigma.shape),
        "meta": p.meta,
    }

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
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    meta_dict = dict(meta or {})
    if signature is not None:
        meta_dict["signature"] = str(signature)

    payload = {
        "mean": np.asarray(mean, float),
        "V": np.asarray(V, float),
        "sigma": np.asarray(sigma, float),
        "explained": np.asarray(explained, float),
        "signature": np.asarray([str(signature or "")], dtype=object),
        "meta": np.asarray([json.dumps(meta_dict)], dtype=object),
    }

    np.savez(path, **payload)


def load_pca_basis(path: str) -> PCABasis:
    z = np.load(path, allow_pickle=False)
    mean = z["mean"]
    V = z["V"]
    sigma = z["sigma"]
    explained = z["explained"] if "explained" in z else np.zeros_like(sigma)

    signature = ""
    if "signature" in z:
        try:
            signature = str(z["signature"][0])
        except Exception:
            signature = ""

    meta = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"][0]))
        except Exception:
            meta = {}

    # fallback for older caches that stored signature only in meta
    if not signature:
        signature = str(meta.get("signature", ""))

    return PCABasis(
        mean=mean,
        V=V,
        sigma=sigma,
        explained=explained,
        signature=signature,
        meta=meta,
    )

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
