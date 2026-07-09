# MeshGeneration/modal_basis.py
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import json
from scipy.sparse.csgraph import connected_components

def laplacian_smooth(points, disp, iters=2, knn=6):
    points = np.asarray(points, float)
    disp = np.asarray(disp, float)
    N = len(points)
    if N <= 2 or iters <= 0:
        return disp

    # Use at most N-1 neighbors (excluding self)
    k_excl = min(max(1, knn), N - 1)
    k_query = min(N, k_excl + 1)  # include self in query result

    tree = cKDTree(points)
    for _ in range(iters):
        _, idx = tree.query(points, k=k_query)
        # idx[:,0] is self → drop it
        nbrs = idx[:, 1:] if idx.ndim == 2 else idx[1:]
        disp = np.array([disp[row].mean(axis=0) for row in nbrs])
    return disp

def _mutual_knn_weighted_graph(points: np.ndarray, knn: int, sigma: float | None = None):
    """
    Build a symmetric mutual-kNN graph with Gaussian weights.

    Keeps edge (i,j) only if:
      j in kNN(i) AND i in kNN(j)

    Returns:
      W (csr_matrix): symmetric adjacency (zero diagonal)
      sigma (float): Gaussian length scale used
      knn_used (int): actual knn used (clamped)
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(points, float)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points to build a graph.")

    knn_used = int(min(max(1, knn), n - 1))
    n_neighbors = knn_used + 1  # include self

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    dists, idx = nbrs.kneighbors(X, return_distance=True)  # shapes (n, k+1)

    # Estimate sigma from mean 1st neighbor distance (excluding self)
    if sigma is None:
        nn1 = dists[:, 1] if dists.shape[1] > 1 else np.ones(n)
        sigma = float(np.mean(nn1))
        sigma = max(sigma, 1e-12)

    # membership sets for mutual check (exclude self at col 0)
    neigh_sets = [set(row[1:].tolist()) for row in idx]

    rows, cols, data = [], [], []
    for i in range(n):
        for pos in range(1, idx.shape[1]):
            j = int(idx[i, pos])
            if j == i:
                continue
            # mutual condition
            if i not in neigh_sets[j]:
                continue

            dist = float(dists[i, pos])
            w = float(np.exp(-(dist * dist) / (2.0 * sigma * sigma)))

            rows.append(i); cols.append(j); data.append(w)
            rows.append(j); cols.append(i); data.append(w)

    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W, sigma, knn_used

def _delaunay_weighted_graph(points: np.ndarray, edge_cutoff_factor: float = 2.5):
    """
    Build a weighted graph from 2D Delaunay triangulation of control nodes.

    Steps:
      1. Project 3D CNs to best-fit 2D PCA plane.
      2. Run scipy.spatial.Delaunay.
      3. Extract triangle edges.
      4. Remove unusually long edges.
      5. Weight remaining edges using Gaussian distance weights.
    """
    from scipy.spatial import Delaunay

    X = np.asarray(points, float)
    n = X.shape[0]

    if n < 3:
        raise ValueError("Need at least 3 control nodes for Delaunay graph.")

    # PCA projection to local 2D plane
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    uv = Xc @ Vt[:2].T

    tri = Delaunay(uv)

    edges = set()
    for simplex in tri.simplices:
        simplex = list(map(int, simplex))
        for a, b in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
            i, j = sorted((a, b))
            edges.add((i, j))

    if not edges:
        raise RuntimeError("Delaunay produced no graph edges.")

    edge_list = list(edges)
    lengths = np.array([np.linalg.norm(X[i] - X[j]) for i, j in edge_list], float)

    med = float(np.median(lengths))
    cutoff = float(edge_cutoff_factor) * med

    keep = lengths <= cutoff
    kept_edges = [e for e, k in zip(edge_list, keep) if k]
    kept_lengths = lengths[keep]

    if len(kept_edges) == 0:
        kept_edges = edge_list
        kept_lengths = lengths

    sigma = float(np.median(kept_lengths))
    sigma = max(sigma, 1e-12)

    rows, cols, data = [], [], []
    for (i, j), d in zip(kept_edges, kept_lengths):
        w = float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.setdiag(0.0)
    W.eliminate_zeros()

    return W, sigma, len(kept_edges)

def _save_graph_debug(debug_dir: str, prefix: str, points: np.ndarray, W: sp.spmatrix,
                     sigma: float, knn: int, mutual: bool):
    os.makedirs(debug_dir, exist_ok=True)

    # Save sparse adjacency
    npz_path = os.path.join(debug_dir, f"{prefix}.npz")
    sp.save_npz(npz_path, W.tocsr())

    # Summaries for quick inspection
    deg = np.asarray(W.sum(axis=1)).reshape(-1).tolist()
    n_comp, labels = connected_components(W, directed=False, return_labels=True)

    # Save json (edge list + metadata)
    Wcoo = W.tocoo()
    edges = list(zip(Wcoo.row.tolist(), Wcoo.col.tolist(), Wcoo.data.tolist()))

    js = {
        "n_points": int(points.shape[0]),
        "knn": int(knn),
        "sigma": float(sigma),
        "mutual": bool(mutual),
        "n_components": int(n_comp),
        "component_labels": labels.tolist(),
        "degree": deg,
        "num_edges_directed": int(len(edges)),
        "points": np.asarray(points, float).tolist(),
        "edges": edges,  # (i, j, w) list
    }
    json_path = os.path.join(debug_dir, f"{prefix}.json")
    with open(json_path, "w") as f:
        json.dump(js, f, indent=2)

    print(f"[DEBUG] Saved connectivity graph to:\n  {npz_path}\n  {json_path}")

def _knn_graph(X, k=6):
    # k-NN graph (undirected, unweighted)
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError("scikit-learn needed for modal basis (pip install scikit-learn)")
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    idx = nn.kneighbors(return_distance=False)
    N = len(X)
    rows, cols = [], []
    for i, nbrs in enumerate(idx):
        for j in nbrs:
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    W = W.tocsr()
    W.data[:] = 1.0
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W

def build_laplacian_basis(
    control_nodes,
    k_modes: int = 10,
    knn: int = 6,
    graph_method: str = "mutual_knn",   # "mutual_knn", "knn", or "delaunay"
    use_mutual_knn: bool = True,
    ensure_connected: bool = True,
    max_knn: int | None = None,
    delaunay_cutoff_factor: float = 2.5,
    debug: bool = True,
    debug_dir: str | None = None,
    debug_prefix: str = "connectivity_graph",
):
    """
    Build Laplacian eigenmodes from a (mutual) kNN graph.

    Key changes vs your current version:
    - Optionally uses MUTUAL kNN to stabilise connectivity.
    - Optionally increases knn until the graph becomes connected.
    - Optional debug dump of W (sparse) + JSON metadata.

    Returns:
      evals (k_modes,) , evecs (N, k_modes)  (skipping constant mode)
    """
    import numpy as np
    import scipy.sparse.linalg as spla

    points = np.asarray(control_nodes, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"control_nodes must be (N,3). Got {points.shape}")

    n = points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points to build a Laplacian basis.")

    # Clamp k_modes and knn
    k_modes = int(min(max(1, k_modes), n - 2))
    knn = int(min(max(1, knn), n - 1))
    if max_knn is None:
        max_knn = min(n - 1, max(knn, 12))

    # Remove exact duplicates (can break neighbour logic / sigma)
    uniq = np.unique(points, axis=0)
    if uniq.shape[0] != points.shape[0]:
        print(f"[WARN] Found duplicates: {points.shape[0] - uniq.shape[0]} removed.")
        points = uniq
        n = points.shape[0]
        k_modes = int(min(max(1, k_modes), n - 2))
        knn = int(min(max(1, knn), n - 1))
        max_knn = min(max_knn, n - 1)

    # --- build adjacency ---
    sigma = None
    W = None
    knn_used = knn

    graph_method = str(graph_method).lower().strip()
    if graph_method == "delaunay":
        W, sigma, n_edges = _delaunay_weighted_graph(
            points,
            edge_cutoff_factor=float(delaunay_cutoff_factor),
        )
        knn_used = -1
        n_comp = connected_components(W, directed=False, return_labels=False)

        if ensure_connected and n_comp != 1:
            print(f"[WARN] Delaunay graph has {n_comp} components. Falling back to mutual-kNN.")
            graph_method = "mutual_knn"
        else:
            print(
                f"[DEBUG] Delaunay graph: edges={n_edges}, "
                f"sigma={sigma:.6e}, components={n_comp}"
            )

    if graph_method == "mutual_knn":
        # try increasing knn until connected (optional)
        for k_try in range(knn, max_knn + 1):
            W_try, sigma_try, k_used = _mutual_knn_weighted_graph(points, k_try, sigma=None)
            n_comp = connected_components(W_try, directed=False, return_labels=False)
            if (not ensure_connected) or (n_comp == 1):
                W, sigma, knn_used = W_try, sigma_try, k_used
                if n_comp != 1:
                    print(f"[WARN] mutual-kNN graph not connected (components={n_comp}), proceeding anyway.")
                break

        if W is None:
            # fallback to union kNN (symmetric) if mutual cannot connect
            print("[WARN] Could not form connected mutual-kNN graph within max_knn. Falling back to symmetric union-kNN.")
            W, sigma, knn_used = _mutual_knn_weighted_graph(points, max_knn, sigma=None)  # still mutual, but max
    elif graph_method == "knn":
        # keep your old behaviour: symmetric union of kNN edges with Gaussian weights
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(knn + 1, n)).fit(points)
        dists, idx = nn.kneighbors(points, return_distance=True)
        nn1 = dists[:, 1] if dists.shape[1] > 1 else np.ones(n)
        sigma = float(np.mean(nn1))
        sigma = max(sigma, 1e-12)

        rows, cols, data = [], [], []
        for i in range(n):
            for pos in range(1, idx.shape[1]):
                j = int(idx[i, pos])
                if j == i:
                    continue
                dist = float(dists[i, pos])
                w = float(np.exp(-(dist * dist) / (2.0 * sigma * sigma)))
                rows += [i, j]
                cols += [j, i]
                data += [w, w]
        W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        W.setdiag(0.0)
        W.eliminate_zeros()
        knn_used = knn

    print(f"[DEBUG] kNN={knn_used} mutual={use_mutual_knn} sigma={sigma:.6e}")

    if debug:
        if debug_dir is None:
            debug_dir = os.getcwd()
        _save_graph_debug(
            debug_dir=debug_dir,
            prefix=debug_prefix,
            points=points,
            W=W,
            sigma=sigma,
            knn=knn_used,
            mutual=use_mutual_knn
        )

    # Laplacian
    degrees = np.asarray(W.sum(axis=1)).reshape(-1)

    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    I = sp.identity(n, format="csr")

    L = I - (D_inv_sqrt @ W @ D_inv_sqrt)
    L = L.tocsr()

    # small diagonal shift for numerical stability
    Ls = L + 1e-10 * sp.identity(n, format="csr")

    # eigen solve: compute k_modes+1 and drop the constant mode
    # use "SM" (smallest magnitude) since L is PSD; shift already avoids exact singularity
    evals, evecs = spla.eigsh(Ls, k=min(k_modes + 1, n - 1), which="SM")
    # sort ascending
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # drop first mode (≈ constant)
    evals = evals[1:k_modes + 1]
    evecs = evecs[:, 1:k_modes + 1]

    return evals, evecs

def expand_modal_coeffs(phi: np.ndarray,
                        coeffs: np.ndarray,
                        normals: np.ndarray | None = None):
    """
    Expand modal coefficients to 3D control-node displacements.
    If normals is None: produce 3D (per-axis) via Kron(phi, I3) * c
      - coeffs length must be 3*k
    If normals provided (N,3): produce normal-projected motion:
      d_i = (phi_i·c) * n_i  with c length = k
    Returns d_ctrl of shape (N,3).
    """
    N, k = phi.shape
    c = np.asarray(coeffs, float)

    if normals is None:
        # 3D per-axis basis: B = kron(I3, phi) ∈ R^{3N × 3k}
        if c.size != 3*k:
            raise ValueError(f"Expected {3*k} coefficients when normals=None, got {c.size}")
        phix, phiy, phiz = phi, phi, phi
        dx = phix @ c[0:k]
        dy = phiy @ c[k:2*k]
        dz = phiz @ c[2*k:3*k]
        return np.column_stack([dx, dy, dz])
    else:
        # Normal-projected modal field: scalar field s = phi @ c, then d_i = s_i * n_i
        if c.size != k:
            raise ValueError(f"Expected {k} coefficients for normal-projected basis, got {c.size}")
        s = phi @ c  # (N,)
        return (s[:, None]) * normals  # (N,3)

def save_basis(path, phi, normals):
    """
    Save basis functions with proper validation and error handling
    """
    import numpy as np
    import os
    
    # Validate phi
    if phi is None:
        print("[ERROR] phi is None")
        phi = np.array([])
    
    # Convert to numpy array and validate
    try:
        phi_array = np.asarray(phi)
        
        # Check if the array is valid
        if phi_array.size == 0:
            print("[WARN] phi is empty")
            phi_array = np.array([])
        elif phi_array.ndim == 1 and phi_array.dtype == object:
            # This means we have an array of arrays with different shapes
            print("[ERROR] phi contains arrays of different shapes")
            print(f"[DEBUG] phi contents: {[item.shape if hasattr(item, 'shape') else type(item) for item in phi_array]}")
            
            # Try to fix by finding common dimensions
            shapes = [item.shape for item in phi_array if hasattr(item, 'shape')]
            if shapes:                
                # If all have same number of rows but different columns, try to pad/truncate
                if len(set(shape[0] for shape in shapes)) == 1:
                    n_rows = shapes[0][0]
                    max_cols = max(shape[1] if len(shape) > 1 else 1 for shape in shapes)
                    
                    print(f"[FIX] Attempting to create {n_rows}x{max_cols} array")
                    fixed_array = np.zeros((n_rows, max_cols))
                    
                    for i, item in enumerate(phi_array):
                        if hasattr(item, 'shape') and len(item.shape) >= 1:
                            if len(item.shape) == 1:
                                fixed_array[:, i] = item[:n_rows]
                            else:
                                cols_to_copy = min(item.shape[1], max_cols - i)
                                if cols_to_copy > 0:
                                    fixed_array[:, i:i+cols_to_copy] = item[:n_rows, :cols_to_copy]
                    
                    phi_array = fixed_array
                    print(f"[FIX] Created fixed array with shape: {phi_array.shape}")
                else:
                    print("[ERROR] Cannot fix - incompatible dimensions")
                    phi_array = np.array([])
            else:
                print("[ERROR] No valid arrays found in phi")
                phi_array = np.array([])
                
    except Exception as e:
        print(f"[ERROR] Failed to process phi: {e}")
        phi_array = np.array([])
    
    # Validate normals
    if normals is None:
        normals_array = np.array([])
    else:
        try:
            normals_array = np.asarray(normals)
            print(f"[DEBUG] normals shape: {normals_array.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to process normals: {e}")
            normals_array = np.array([])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save with error handling
    try:
        np.savez(path, Phi=phi_array, normals=normals_array)
        print(f"[SUCCESS] Saved basis to {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save basis: {e}")
        # Save as pickle as fallback
        import pickle
        fallback_path = path.replace('.npz', '_fallback.pkl')
        with open(fallback_path, 'wb') as f:
            pickle.dump({'Phi': phi, 'normals': normals}, f)
        print(f"[FALLBACK] Saved as pickle to {fallback_path}")
        raise

def load_basis(path: str):
    z = np.load(path, allow_pickle=False)
    phi = z["Phi"]
    normals = z["normals"]
    if normals.size == 0:
        normals = None
    return phi, normals
