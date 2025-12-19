# MeshGeneration/controlNodeDisp.py
import os, sys
import numpy as np

sys.path.append(os.path.dirname("MeshGeneration"))
from MeshGeneration.modalBasis import build_laplacian_basis, expand_modal_coeffs, save_basis, load_basis, laplacian_smooth

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

def getDisplacements(output_dir,
                     seed=None,
                     control_nodes=None,
                     normals=None,
                     coeffs=None,
                     k_modes=16,
                     normal_project=True,
                     t_patch_scale=None,
                     amp_alpha=0.02,
                     cache_name="modal_basis.npz"):
    """
    Returns a (N,3) displacement array for N control nodes using a Laplacian modal basis.
    
    CRITICAL: Each design MUST pass its own unique 'coeffs' array!
    
    - If coeffs is None, draws coefficients uniformly within bounds (for testing/DOE).
    - If normal_project is True, uses a scalar modal field projected along provided normals (one c per mode).
      Otherwise uses a 3-axis basis (three independent coefficient blocks).
    
    CACHING BEHAVIOR:
    - The modal basis (phi) is cached and shared across designs (correct - same mesh topology)
    - Normals are NOT cached - each design passes its own normals
    - Coefficients are NEVER cached - each design has unique coeffs
    """
    rng = np.random.default_rng(seed)
    if control_nodes is None:
        cn_path = os.path.join(output_dir, "Control Nodes", "control_nodes.npy")
        control_nodes = np.load(cn_path)

    if control_nodes.shape[0] == 1:
        normals = np.asarray(normals, float).reshape(1, 3)

        coeffs = np.asarray(coeffs, float).reshape(-1) if coeffs is not None else np.zeros(1)
        x0 = float(coeffs[0]) if coeffs.size else 5

        # scale design variable into physical displacement
        amp = x0
        if t_patch_scale is not None:
            amp = x0 * float(amp_alpha) * float(t_patch_scale)

        d_ctrl = amp * normals
        return d_ctrl

    control_nodes = np.asarray(control_nodes, float)
    
    # Load or build the modal basis (PHI ONLY - no normals cached!)
    basis_path = os.path.join(output_dir, cache_name)
    if os.path.exists(basis_path):
        print(f"[DEBUG] Loading cached basis from {basis_path}")
        phi, _ = load_basis(basis_path)  # Ignore any cached normals
        print(f"[DEBUG] Loaded phi with shape {phi.shape}")
    else:
        print(f"[DEBUG] Building new basis with k_modes={k_modes}")
        out = build_laplacian_basis(control_nodes, k_modes=k_modes, knn=6)
        _, phi = out if isinstance(out, tuple) else (None, out)
        print(f"[DEBUG] Built phi with shape {phi.shape}")
        # Save basis WITHOUT normals (pass None)
        save_basis(basis_path, phi, normals=None)

    N, k = phi.shape
    print(f"[DEBUG] Basis: N={N} nodes, k={k} modes")

    if normal_project:
        # === SCALAR MODAL FIELD (normal-projected) ===
        # MUST use the normals passed in (unique per design if needed)
        if normals is None:
            print("[WARN] normal_project=True but no normals provided, estimating from control_nodes")
            normals = estimate_normals(control_nodes, knn=12)
        
        normals = np.asarray(normals, float)
        print(f"[DEBUG] Using normals with shape {normals.shape}")
        
        # --- NEW: always derive a CAD-based target amplitude from local spacing ---
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(8, len(control_nodes))).fit(control_nodes)
        d8 = nn.kneighbors(return_distance=True)[0][:, -1].mean()
        frac = 0.15  # fraction of mean 8th-neighbour spacing to target as coeff RMS
        target_amp = frac * d8
        print(f"[DEBUG] Local spacing d8={d8:.6f}, target_amp={target_amp:.6f}")

        if coeffs is None:
            print("[DEBUG] No coeffs provided, generating random coefficients")
            # Spectral decay prior (shape only, amplitude handled by target_amp)
            j = np.arange(1, k+1, dtype=float)
            decay = 1.0 / (j**2.0)
            coeffs = rng.normal(0.0, 1.0, size=k) * decay

        # Ensure coeffs has length k
        coeffs = np.asarray(coeffs, float)
        if coeffs.size != k:
            print(f"[WARN] coeffs size {coeffs.size} != k={k}, padding/truncating")
            if coeffs.size < k:
                coeffs = np.pad(coeffs, (0, k - coeffs.size))
            else:
                coeffs = coeffs[:k]

        # --- NEW: normalise coeffs to CAD-based amplitude ---
        coeff_norm = np.linalg.norm(coeffs) + 1e-12
        scale = target_amp / coeff_norm
        coeffs *= scale
        print(
            f"[DEBUG] Normalising coeffs: norm_before={coeff_norm:.6f}, "
            f"scale={scale:.6f}, norm_after={np.linalg.norm(coeffs):.6f}"
        )

        # CRITICAL: Log coefficients to verify they're different per design
        print(f"[DEBUG] Using coeffs (first 5): {coeffs[:5].tolist()}")
        print(f"[DEBUG] Coeffs norm: {np.linalg.norm(coeffs):.6f}")

        # Expand: d_i = (phi @ coeffs)_i * normals_i
        d_ctrl = expand_modal_coeffs(phi, coeffs, normals=normals)
        d_ctrl = laplacian_smooth(control_nodes, d_ctrl, iters=3)
        
        # DEBUG: Verify displacements are non-zero and varied
        d_norms = np.linalg.norm(d_ctrl, axis=1)
        print(
            f"[DEBUG] Displacement norms: mean={d_norms.mean():.6f}, "
            f"max={d_norms.max():.6f}, std={d_norms.std():.6f}"
        )

    else:
        # === 3D PER-AXIS BASIS ===
        if coeffs is None:
            coeffs = np.zeros(3*k, dtype=float)
        else:
            coeffs = np.asarray(coeffs, float)
            if coeffs.size != 3*k:
                print(f"[WARN] 3D basis expects {3*k} coeffs, got {coeffs.size}")
                if coeffs.size < 3*k:
                    coeffs = np.pad(coeffs, (0, 3*k - coeffs.size))
                else:
                    coeffs = coeffs[:3*k]
        
        print(f"[DEBUG] 3D basis: using {coeffs.size} coefficients")
        d_ctrl = expand_modal_coeffs(phi, coeffs, normals=None)
        d_ctrl = laplacian_smooth(control_nodes, d_ctrl, iters=3)

    return d_ctrl  # shape (N,3)