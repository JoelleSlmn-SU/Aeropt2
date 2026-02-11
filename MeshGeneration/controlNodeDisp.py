# MeshGeneration/controlNodeDisp.py
import os, sys
import numpy as np

sys.path.append(os.path.dirname("MeshGeneration"))
from MeshGeneration.modalBasis import build_laplacian_basis, expand_modal_coeffs, save_basis, load_basis, laplacian_smooth
from MeshGeneration.pcaBasis import load_pca_basis, reconstruct_disp_flat

def _surface_normals(points, knn=16):
    """PCA normals on the T-surface point cloud."""
    from sklearn.neighbors import NearestNeighbors
    P = np.asarray(points, float)
    N = len(P)
    k = min(max(3, knn), max(1, N-1))
    nn = NearestNeighbors(n_neighbors=k).fit(P)
    idx = nn.kneighbors(P, return_distance=False)
    normals = np.zeros((N, 3))
    for i in range(N):
        Q = P[idx[i]]
        Qc = Q - Q.mean(axis=0, keepdims=True)
        C = Qc.T @ Qc
        w, V = np.linalg.eigh(C)
        n = V[:, 0]
        n /= (np.linalg.norm(n) + 1e-12)
        normals[i] = n
    # orient consistently using surface centroid
    c = P.mean(axis=0)
    s = np.sign(((P - c) * normals).sum(axis=1))
    s[s == 0] = 1.0
    return normals * s[:, None]

def _map_normals_to_control(control_nodes, surf_pts, surf_normals, k=8):
    """Average k nearest surface normals for each control node."""
    from sklearn.neighbors import NearestNeighbors
    P = np.asarray(surf_pts, float)
    N = len(P)
    k = min(max(3, k), max(1, N-1))
    nn = NearestNeighbors(n_neighbors=k).fit(P)
    idx = nn.kneighbors(np.asarray(control_nodes, float), return_distance=False)
    out = []
    for row in idx:
        n = surf_normals[row].mean(axis=0)
        n /= (np.linalg.norm(n) + 1e-12)
        out.append(n)
    return np.asarray(out)

def _spectral_coeffs(num_modes, control_nodes, rng=None, p=2.0, frac=0.15):
    """
    Smooth modal coefficients with spectral decay and mesh-aware amplitude.
    frac ~ fraction of the mean 8th-neighbor spacing to target as RMS disp.
    """
    rng = np.random.default_rng(rng)
    j = np.arange(1, num_modes + 1, dtype=float)
    decay = 1.0 / (j ** p)

    # spacing estimate from 8-th neighbor
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(control_nodes, float)
    n_n = min(9, max(2, len(X)))  # include self → 8 neighbors
    nn = NearestNeighbors(n_neighbors=n_n).fit(X)
    dists, _ = nn.kneighbors(X)
    d8 = dists[:, -1].mean() if dists.shape[1] > 1 else 1.0

    amp = frac * d8
    c = rng.normal(0.0, 1.0, size=num_modes) * decay
    c *= amp / (np.linalg.norm(c) + 1e-12)
    return c

def estimate_normals(points: np.ndarray, knn: int = 12) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    P = np.asarray(points, float)
    N = len(P)

    n_nbrs = min(max(3, knn), max(1, N - 1))

    nn = NearestNeighbors(n_neighbors=n_nbrs).fit(P)
    idx = nn.kneighbors(P, return_distance=False)

    normals = np.zeros((N, 3))
    for i in range(N):
        nbrs = P[idx[i]]
        Q = nbrs - nbrs.mean(axis=0, keepdims=True)
        C = Q.T @ Q
        w, V = np.linalg.eigh(C)
        n = V[:, 0]
        n /= (np.linalg.norm(n) + 1e-12)
        normals[i] = n

    # orient roughly outward from centroid
    centroid = P.mean(axis=0)
    sign = np.sign(((P - centroid) * normals).sum(axis=1))
    sign[sign == 0] = 1.0
    return normals * sign[:, None]

def disp_rms(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    return float(np.sqrt(np.mean(np.sum(d*d, axis=1))))

def scale_to_target_rms(d: np.ndarray, target_rms: float, eps: float = 1e-12) -> np.ndarray:
    cur = disp_rms(d)
    if target_rms is None or target_rms <= 0:
        return d
    s = float(target_rms) / (cur + eps)
    return d * s


def getDisplacements(
    output_dir,
    seed=None,
    control_nodes=None,
    normals=None,
    coeffs=None,
    k_modes=16,
    normal_project=True,
    t_patch_scale=None,
    amp_alpha=0.0005,
    cache_name="modal_basis.npz",
    # ---- PCA reduced space ----
    use_pca: bool = False,
    pca_cache_path: str | None = None,
    pca_coeffs=None,
    pca_cache_name: str = "pca_basis.npz",
    # ---- optional: enforce same magnitude only for PCA comparisons ----
    target_disp_rms: float | None = None,
):
    """
    Returns a (N,3) displacement array for N control nodes.

    Scaling:
    - Uses ONE common scale for BOTH PCA and non-PCA:
        amp_scale = amp_alpha * len_scale
      where len_scale = t_patch_scale if provided else d_ref (local spacing from CNs)

    Optional comparison feature:
    - If use_pca=True and target_disp_rms is provided, the returned displacement field is
      additionally rescaled to have RMS magnitude = target_disp_rms (to match a baseline).
    """
    import os
    import numpy as np

    rng = np.random.default_rng(seed)

    # ---------------------------
    # helpers
    # ---------------------------
    def disp_rms(d: np.ndarray) -> float:
        d = np.asarray(d, float)
        return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))

    def scale_to_target_rms(d: np.ndarray, target: float, eps: float = 1e-12) -> np.ndarray:
        if target is None or target <= 0:
            return d
        cur = disp_rms(d)
        return d * (float(target) / (cur + eps))

    # ---------------------------
    # load CNs
    # ---------------------------
    if control_nodes is None:
        cn_path = os.path.join(output_dir, "Control Nodes", "control_nodes.npy")
        control_nodes = np.load(cn_path)

    control_nodes = np.asarray(control_nodes, float)
    Ncn = int(control_nodes.shape[0])

    # define a single length scale used by BOTH PCA and non-PCA
    # d_ref = typical local spacing (mean distance to k-th neighbor)
    from sklearn.neighbors import NearestNeighbors

    if Ncn >= 2:
        K_excl = min(8, max(1, Ncn - 1))
        n_query = min(K_excl + 1, max(1, Ncn - 1))  # include self but still < Ncn
        nn = NearestNeighbors(n_neighbors=n_query).fit(control_nodes)
        dists, _ = nn.kneighbors(control_nodes, return_distance=True)
        d_ref = float(dists[:, -1].mean()) if dists.shape[1] > 1 else 1.0
    else:
        d_ref = 1.0

    len_scale = float(t_patch_scale) if t_patch_scale is not None else float(d_ref)
    amp_scale = float(amp_alpha) * float(len_scale)


    # special case: single control node
    if Ncn == 1:
        normals = np.asarray(normals, float).reshape(1, 3)
        coeffs_ = np.asarray(pca_coeffs if (use_pca and pca_coeffs is not None) else coeffs, float).reshape(-1) \
                  if (pca_coeffs is not None or coeffs is not None) else np.zeros(1)
        x0 = float(coeffs_[0]) if coeffs_.size else 0.0

        # same scaling rule
        d_ctrl = (x0 * amp_scale) * normals

        if use_pca and target_disp_rms is not None:
            d_ctrl = scale_to_target_rms(d_ctrl, target_disp_rms)

        return d_ctrl

    
    # PCA branch
    if bool(use_pca):
        if not pca_cache_path:
            pca_cache_path = os.path.join(output_dir, "morph", pca_cache_name)
            if not os.path.exists(pca_cache_path):
                pca_cache_path = os.path.join(output_dir, "Control Nodes", "pca", pca_cache_name)

        if not os.path.exists(pca_cache_path):
            raise FileNotFoundError(f"PCA cache not found: {pca_cache_path}")

        # expects the existing helpers
        pca = load_pca_basis(pca_cache_path)

        # coefficients in reduced space
        a = np.asarray(pca_coeffs if pca_coeffs is not None else coeffs, dtype=float).reshape(-1)

        k = int(pca.V.shape[1])
        if a.size < k:
            a = np.pad(a, (0, k - a.size))
        elif a.size > k:
            a = a[:k]

        # apply the SAME physical scale used in non-PCA
        a_scaled = a * amp_scale

        d_flat = reconstruct_disp_flat(pca.mean, pca.V, a_scaled)
        d_ctrl = d_flat.reshape((-1, 3))

        # enforce identical RMS magnitude (useful for PCA vs non-PCA comparisons)
        if target_disp_rms is not None:
            d_ctrl = scale_to_target_rms(d_ctrl, target_disp_rms)

        return d_ctrl

    
    # Non-PCA: load/build Laplacian modal basis
    basis_path = os.path.join(output_dir, cache_name)

    need_rebuild = True
    phi = None

    if os.path.exists(basis_path):
        print(f"[DEBUG] Loading cached basis from {basis_path}")
        try:
            phi, _ = load_basis(basis_path)  # ignore cached normals
            print(f"[DEBUG] Loaded phi with shape {phi.shape}")
            need_rebuild = (phi is None) or (phi.shape[0] != int(control_nodes.shape[0]))
            if need_rebuild:
                print(
                    f"[WARN] Cached modal basis rows ({None if phi is None else phi.shape[0]}) "
                    f"!= current CN count ({control_nodes.shape[0]}). Rebuilding cache..."
                )
        except Exception as e:
            print(f"[WARN] Failed to load cached basis '{basis_path}': {e}. Rebuilding...")
            need_rebuild = True

    if need_rebuild:
        print(f"[DEBUG] Building new basis with k_modes={k_modes}")
        out = build_laplacian_basis(control_nodes, k_modes=k_modes, knn=6)
        _, phi = out if isinstance(out, tuple) else (None, out)
        print(f"[DEBUG] Built phi with shape {phi.shape}")
        save_basis(basis_path, phi, normals=None)

    N, k = phi.shape
    print(f"[DEBUG] Basis: N={N} nodes, k={k} modes")
    print(f"[DEBUG] len_scale={len_scale:.6f}, d_ref={d_ref:.6f}, amp_scale={amp_scale:.6e}")

    if normal_project:
        if normals is None:
            print("[WARN] normal_project=True but no normals provided, estimating from control_nodes")
            normals = estimate_normals(control_nodes, knn=12)

        normals = np.asarray(normals, float)

        if coeffs is None:
            print("[DEBUG] No coeffs provided, generating random coefficients with spectral decay")
            j = np.arange(1, k + 1, dtype=float)
            decay = 1.0 / (j ** 2.0)
            coeffs = rng.normal(0.0, 1.0, size=k) * decay

        coeffs = np.asarray(coeffs, float)
        if coeffs.size != k:
            print(f"[WARN] coeffs size {coeffs.size} != k={k}, padding/truncating")
            if coeffs.size < k:
                coeffs = np.pad(coeffs, (0, k - coeffs.size))
            else:
                coeffs = coeffs[:k]

        # apply the SAME physical scale as PCA
        coeffs_scaled = coeffs * amp_scale

        print(f"[DEBUG] Raw coeffs (first 5): {coeffs[:5].tolist()}")
        print(f"[DEBUG] Scaled coeffs (first 5): {coeffs_scaled[:5].tolist()}")
        print(f"[DEBUG] Scaled coeffs norm: {np.linalg.norm(coeffs_scaled):.6f}")

        d_ctrl = expand_modal_coeffs(phi, coeffs_scaled, normals=normals)

        # OPTIONAL - but it does reduce variance slightly
        d_ctrl = laplacian_smooth(control_nodes, d_ctrl, iters=1)

        d_norms = np.linalg.norm(d_ctrl, axis=1)
        cv = d_norms.std() / (d_norms.mean() + 1e-12)
        print(
            f"[DEBUG] Displacement norms: mean={d_norms.mean():.6f}, max={d_norms.max():.6f}, "
            f"std={d_norms.std():.6f}, CV={cv:.2%}"
        )

    else:
        if coeffs is None:
            coeffs = np.zeros(3 * k, dtype=float)
        else:
            coeffs = np.asarray(coeffs, float)
            if coeffs.size != 3 * k:
                print(f"[WARN] 3D basis expects {3*k} coeffs, got {coeffs.size}")
                if coeffs.size < 3 * k:
                    coeffs = np.pad(coeffs, (0, 3 * k - coeffs.size))
                else:
                    coeffs = coeffs[:3 * k]

        # apply the SAME physical scale as PCA (per-axis coefficients)
        coeffs_scaled = coeffs * amp_scale

        print(f"[DEBUG] 3D basis: using {coeffs_scaled.size} coefficients (scaled)")
        d_ctrl = expand_modal_coeffs(phi, coeffs_scaled, normals=None)

        d_norms = np.linalg.norm(d_ctrl, axis=1)
        print(
            f"[DEBUG] 3D basis displacement norms: mean={d_norms.mean():.6f}, "
            f"max={d_norms.max():.6f}, std={d_norms.std():.6f}"
        )

    return d_ctrl  # shape (N,3)
