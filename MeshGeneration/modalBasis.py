# MeshGeneration/modal_basis.py
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree


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

def build_laplacian_basis(control_nodes, k_modes=10, knn=6):
    """
    Debug version with extensive validation
    """
    import numpy as np
    import scipy.sparse.linalg as spla
    
    # Convert to numpy array
    points = np.asarray(control_nodes)
    
    # Validate input
    if len(points) == 0:
        raise ValueError("No control nodes provided")
    
    if len(points) < k_modes + 2:
        k_modes = max(1, len(points) - 2)
    
    if len(points) <= knn:
        knn = max(1, len(points) - 1)
    
    # Check for duplicate points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) < len(points):
        points = unique_points
    
    # Check point distribution
    if len(points) > 1:
        distances = np.linalg.norm(points[1:] - points[0], axis=1)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if min_dist < 1e-10:
            print("[ERROR] Points are too close together (numerical precision issues)")
    
    # Build the graph Laplacian (this is the missing implementation)
    from sklearn.neighbors import NearestNeighbors
    
    # Find k-nearest neighbors
    N = len(points)
    n_neighbors = min(knn + 1, max(2, N - 1))  # must be < N
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Build adjacency matrix
    n = len(points)
    from scipy.sparse import csr_matrix
    
    # Create adjacency matrix with Gaussian weights
    row_indices = []
    col_indices = []
    data = []
    
    sigma = np.mean(distances[:, 1])  # Use mean nearest neighbor distance as sigma
    print(f"[DEBUG] Using sigma = {sigma:.6f} for Gaussian weights")
    
    for i in range(n):
        for j in range(1, len(indices[i])):  # Skip self (index 0)
            neighbor = indices[i][j]
            dist = distances[i][j]
            weight = np.exp(-dist**2 / (2 * sigma**2))
            
            row_indices.extend([i, neighbor])
            col_indices.extend([neighbor, i])
            data.extend([weight, weight])
    
    # Create sparse adjacency matrix
    W = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    
    # Compute degree matrix
    degrees = np.array(W.sum(axis=1)).flatten()
    D = csr_matrix((degrees, (range(n), range(n))), shape=(n, n))
    
    # Compute Laplacian: L = D - W
    L = D - W
    
    # Check if matrix is singular
    try:
        # Try a small shift to avoid singularity
        L_shifted = L + 1e-8 * csr_matrix(np.eye(n))
        
        # Compute eigenvalues
        evals, evecs = spla.eigsh(L_shifted, k=k_modes+1, sigma=0.0, which='LM')
        
        
        return evals[1:], evecs[:, 1:]  # Skip the first (constant) mode
        
    except RuntimeError as e:
        print(f"[ERROR] Eigenvalue computation failed: {e}")
        
        # Fallback: use PCA instead
        print("[FALLBACK] Using PCA instead of Laplacian eigenmodes")
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(k_modes, points.shape[1], len(points)-1))
        components = pca.fit_transform(points - np.mean(points, axis=0))
        
        # Create fake eigenvalues
        explained_var = pca.explained_variance_
        fake_evals = np.arange(1, len(explained_var) + 1)
        
        return fake_evals, components

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
