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

    # <-- change this line:
    n_nbrs = min(max(3, knn), max(1, N - 1))  # ensure 1 <= n_nbrs < N

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
                     k_modes=16,
                     coeffs=None,
                     normal_project=True,
                     cache_name="modal_basis.npz"):
    """
    Returns a (N,3) displacement array for N control nodes using a Laplacian modal basis.
    - If coeffs is None, draws coefficients uniformly within bounds (for testing/DOE).
    - If normal_project is True, uses a scalar modal field projected along provided normals (one c per mode).
      Otherwise uses a 3-axis basis (three independent coefficient blocks).
    """
    rng = np.random.default_rng(seed)
    if control_nodes is None:
        # Expect caller to pass them; otherwise try to load from the run folder
        cn_path = os.path.join(output_dir, "Control Nodes", "control_nodes.npy")
        control_nodes = np.load(cn_path)

    basis_path = os.path.join(output_dir, cache_name)
    if os.path.exists(basis_path):
        phi, cached_normals = load_basis(basis_path)
        if normal_project and normals is None:
            normals = cached_normals
    else:
        out = build_laplacian_basis(np.asarray(control_nodes), k_modes=k_modes, knn=6)
        # build_laplacian_basis returns (evals, evecs); we only need evecs
        _, phi = out if isinstance(out, tuple) else (None, out)
        save_basis(basis_path, phi, normals)

    N, k = phi.shape

    # after phi is built; k = phi.shape[1]
    if normal_project:
        # spectral decay prior
        j = np.arange(1, k+1, dtype=float)
        decay = 1.0 / (j**2.0)        # p = 2; tune 1.5–3
        # target RMS amplitude as fraction of spacing
        # estimate spacing from control nodes
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=8).fit(control_nodes)
        d8 = nn.kneighbors(return_distance=True)[0][:, -1].mean()
        amp = 0.15 * d8               # ≤ 15% of local spacing
        # sample smooth coefficients
        rng = np.random.default_rng(seed)
        coeffs = rng.normal(0.0, 1.0, size=k) * decay
        coeffs *= (amp / (np.linalg.norm(coeffs) + 1e-12))
        d_ctrl = expand_modal_coeffs(phi, coeffs, normals=normals)
        d_ctrl = laplacian_smooth(control_nodes, d_ctrl, iters=3)

    # d_ctrl = np.clip(d_ctrl, lower_bound, upper_bound)

    return d_ctrl  # shape (N,3)
