# MeshGeneration/controlNodeDisp.py
import os
import sys
import numpy as np

sys.path.append(os.path.dirname("MeshGeneration"))
from MeshGeneration.modalBasis import (
    build_laplacian_basis,
    expand_modal_coeffs,
    save_basis,
    load_basis,
    laplacian_smooth,
)


def _surface_normals(points, knn=16):
    """PCA normals on the T-surface point cloud."""
    from sklearn.neighbors import NearestNeighbors

    P = np.asarray(points, float)
    N = len(P)
    k = min(max(3, knn), max(1, N - 1))
    nn = NearestNeighbors(n_neighbors=k).fit(P)
    idx = nn.kneighbors(P, return_distance=False)
    normals = np.zeros((N, 3))
    for i in range(N):
        Q = P[idx[i]]
        Qc = Q - Q.mean(axis=0, keepdims=True)
        C = Qc.T @ Qc
        _, V = np.linalg.eigh(C)
        n = V[:, 0]
        n /= (np.linalg.norm(n) + 1e-12)
        normals[i] = n
    c = P.mean(axis=0)
    s = np.sign(((P - c) * normals).sum(axis=1))
    s[s == 0] = 1.0
    return normals * s[:, None]


def _map_normals_to_control(control_nodes, surf_pts, surf_normals, k=8):
    """Average k nearest surface normals for each control node."""
    from sklearn.neighbors import NearestNeighbors

    P = np.asarray(surf_pts, float)
    N = len(P)
    k = min(max(3, k), max(1, N - 1))
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
    decay = 1.0 / (j**p)

    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(control_nodes, float)
    n_n = min(9, max(2, len(X)))
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
        _, V = np.linalg.eigh(C)
        n = V[:, 0]
        n /= (np.linalg.norm(n) + 1e-12)
        normals[i] = n

    centroid = P.mean(axis=0)
    sign = np.sign(((P - centroid) * normals).sum(axis=1))
    sign[sign == 0] = 1.0
    return normals * sign[:, None]


def estimate_local_frame(points: np.ndarray, knn: int = 12):
    """
    Per-point orthonormal frame (t1, t2, n) from local PCA.
    n  : smallest eigenvector
    t1 : largest eigenvector
    t2 : completes right-handed basis
    """
    from sklearn.neighbors import NearestNeighbors

    P = np.asarray(points, float)
    N = len(P)
    k = min(max(4, knn), max(1, N - 1))

    nn = NearestNeighbors(n_neighbors=k).fit(P)
    idx = nn.kneighbors(P, return_distance=False)

    t1 = np.zeros((N, 3), float)
    t2 = np.zeros((N, 3), float)
    n = np.zeros((N, 3), float)

    for i in range(N):
        Q = P[idx[i]]
        Qc = Q - Q.mean(axis=0, keepdims=True)
        C = Qc.T @ Qc
        _, V = np.linalg.eigh(C)
        ni = V[:, 0]
        ti1 = V[:, 2]

        ni /= (np.linalg.norm(ni) + 1e-12)
        ti1 /= (np.linalg.norm(ti1) + 1e-12)

        ti1 = ti1 - ni * np.dot(ti1, ni)
        ti1 /= (np.linalg.norm(ti1) + 1e-12)

        ti2 = np.cross(ni, ti1)
        ti2 /= (np.linalg.norm(ti2) + 1e-12)

        n[i] = ni
        t1[i] = ti1
        t2[i] = ti2

    c = P.mean(axis=0)
    s = np.sign(((P - c) * n).sum(axis=1))
    s[s == 0] = 1.0
    n *= s[:, None]

    t2 = np.cross(n, t1)
    t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)
    return t1, t2, n


def _orthonormal_axes_from_points(points: np.ndarray):
    """
    Build a stable global body frame from the control-node cloud.

    For strongly anisotropic shapes, this follows PCA.
    For near-spherical shapes, PCA is ambiguous, so we fall back
    to the world frame to deliberately break symmetry.
    """
    P = np.asarray(points, float)
    Q = P - P.mean(axis=0, keepdims=True)
    C = Q.T @ Q
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    V = V[:, order]
    w = w[order]

    spread = np.sqrt(np.maximum(w, 1e-12))
    anis = spread[0] / max(spread[-1], 1e-12)

    if anis < 1.2:
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
    else:
        ex = V[:, 0]
        ez = V[:, 1]
        ey = V[:, 2]

        ex /= np.linalg.norm(ex) + 1e-12
        ey /= np.linalg.norm(ey) + 1e-12
        ez = np.cross(ex, ey)
        ez /= np.linalg.norm(ez) + 1e-12
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey) + 1e-12

    return ex, ey, ez


def build_global_modes(control_nodes, center=None, axes=None, mode_config=None):
    """
    Build user-selected global modes.

    Parameters
    ----------
    mode_config : list[dict]
        Example:
        [
            {"type": "stretch", "direction": "x"},
            {"type": "flatten", "direction": "z"},
            {"type": "camber", "direction": "y"},
        ]
    """
    P = np.asarray(control_nodes, float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"control_nodes must be (N,3). Got {P.shape}")

    if center is None:
        c = P.mean(axis=0)
    else:
        c = np.asarray(center, float)

    if axes is None:
        ex, ey, ez = _orthonormal_axes_from_points(P)
    else:
        ex = np.asarray(axes[0], float)
        ey = np.asarray(axes[1], float)
        ez = np.asarray(axes[2], float)
        ex /= np.linalg.norm(ex) + 1e-12
        ey /= np.linalg.norm(ey) + 1e-12
        ez = np.cross(ex, ey)
        ez /= np.linalg.norm(ez) + 1e-12
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey) + 1e-12

    axis_map = {"x": ex, "y": ey, "z": ez}

    Rv = P - c
    x = Rv @ ex
    y = Rv @ ey
    z = Rv @ ez

    Lx = max(np.max(np.abs(x)), 1e-12)
    Ly = max(np.max(np.abs(y)), 1e-12)
    Lz = max(np.max(np.abs(z)), 1e-12)

    xr = x / Lx
    yr = y / Ly
    zr = z / Lz

    abs_x = np.abs(xr)
    abs_y = np.abs(yr)
    abs_z = np.abs(zr)

    sign_x = np.sign(xr); sign_x[sign_x == 0.0] = 1.0
    sign_y = np.sign(yr); sign_y[sign_y == 0.0] = 1.0
    sign_z = np.sign(zr); sign_z[sign_z == 0.0] = 1.0

    def gauss(s, mu, sig):
        return np.exp(-0.5 * ((s - mu) / max(sig, 1e-12)) ** 2)

    mid_x = np.maximum(0.0, 1.0 - xr**2)
    mid_y = np.maximum(0.0, 1.0 - yr**2)
    mid_z = np.maximum(0.0, 1.0 - zr**2)

    body_core = mid_x * mid_z
    le = gauss(xr, -0.9, 0.22)
    te = gauss(xr, +0.9, 0.22)

    if not mode_config:
        mode_config = [
            {"type": "stretch", "direction": "x"},
            {"type": "flatten", "direction": "y"},
            {"type": "camber", "direction": "y"},
        ]

    modes = []
    names = []

    def add_mode(field, name):
        field = np.asarray(field, float)
        field -= field.mean(axis=0, keepdims=True)
        rms = np.sqrt(np.mean(np.sum(field**2, axis=1)))
        if rms < 1e-14:
            return
        modes.append(field / rms)
        names.append(name)

    for item in mode_config:
        mtype = str(item.get("type", "")).strip().lower()
        direction = str(item.get("direction", "x")).strip().lower()
        avec = axis_map.get(direction, ex)

        if mtype == "stretch":
            # Displaces points outward along direction proportional to their
            # signed coordinate - linearly scales the extent of the shape.
            coord = {"x": xr, "y": yr, "z": zr}[direction]
            field = coord[:, None] * avec[None, :]
            add_mode(field, f"stretch_{direction}")

        elif mtype == "flatten":
            # Pushes ALL points toward the midplane of direction, regardless
            # of which side they are on.  Uses -|coord| so both +z and -z
            # surfaces move inward.
            coord = {"x": xr, "y": yr, "z": zr}[direction]
            mag = -np.abs(coord)
            field = mag[:, None] * avec[None, :]
            add_mode(field, f"flatten_{direction}")

        elif mtype == "bulge":
            # Pushes the upper/lower surfaces outward to create aerofoil
            # thickness.  Peak is away from the midplane, tapering to zero at the tips.
            coord = {"x": xr, "y": yr, "z": zr}[direction]
            mag = np.abs(coord) * (1.0 - coord**2)
            field = mag[:, None] * avec[None, :]
            add_mode(field, f"bulge_{direction}")

        elif mtype == "camber":
            # Asymmetric Z-offset that varies along the chord (X) to arch the
            # mean camber line.  Envelope is the span bell-curve so the effect
            # tapers at the wingtips.
            mag = np.exp(-0.5 * ((xr - 0.0) / 0.45)**2)
            field = mag[:, None] * avec[None, :]
            add_mode(field, f"camber_{direction}")

        elif mtype == "twist":
            # Spanwise washout: chord-position (X) modulates displacement in
            # direction so sections rotate about the span axis.
            # direction sets which axis the twist deflects into.
            if direction == "x":
                mag = yr * zr          # span × thickness cross-term
            elif direction == "y":
                mag = yr * xr          # span ramps chord deflection in Y
            else:                      # "z" — standard washout
                mag = yr * xr          # span ramps chord deflection in Z
            field = mag[:, None] * avec[None, :]
            add_mode(field, f"twist_{direction}")

        elif mtype == "taper":
            # Reduces thickness toward the wingtips (spanwise taper).
            # Displacement in direction is gated by abs_z (thickness extent)
            # and reduced as |yr| grows toward the tip.
            mag = -(abs_y) * abs_z
            field = mag[:, None] * avec[None, :]
            add_mode(field, f"taper_{direction}")

        elif mtype == "bend":
            # Spanwise bending (dihedral / anhedral): displaces in direction
            # proportional to yr² so the wing bows up or down symmetrically.
            if direction == "x":
                mag = yr**2             # chord-axis bow driven by span position
            elif direction == "y":
                mag = yr**2             # spanwise bow in Y (dihedral)
            else:                       # "z"
                mag = yr**2             # spanwise bow in Z
            field = mag[:, None] * avec[None, :]
            add_mode(field, f"bend_{direction}")

        elif mtype == "leading_edge":
            coord = zr
            field = le[:, None] * np.sign(coord)[:, None] * avec[None, :]
            add_mode(field, f"leading_edge_{direction}")

        elif mtype == "trailing_edge":
            coord = -zr
            field = te[:, None] * np.sign(coord)[:, None] * avec[None, :]
            add_mode(field, f"trailing_edge_{direction}")

    if not modes:
        raise RuntimeError("Failed to construct global modes from mode_config.")

    G = np.stack(modes, axis=1)
    return G, names


def getDisplacements(
    output_dir,
    seed=None,
    control_nodes=None,
    normals=None,
    coeffs=None,
    k_modes=16,
    normal_project=True,
    t_patch_scale=None,
    amp_alpha=0.02,
    cache_name="modal_basis.npz",
    # non-normal local modal representation
    vector_mode="local_frame",   # "xyz" or "local_frame"
    frame_knn=12,
    # global modes
    global_modes=False,
    global_mode_config=None,
    global_amp_alpha=None,
    smooth_global=False,
    basis_axes=None,
    # new parameterisation controls
    parameterisation_method="modal",           # "modal" or "direct"
    direct_parameterisation_subtype=None,      # "xyz" or "normal"
    use_local_modes=True,
    global_only=False,
    # PCA-reduced displacement
    use_pca=False,
    pca_cache_path=None,
    pca_coeffs=None,
):
    """
    Returns a (N,3) displacement array for N control nodes.

    Supported parameterisation families
    -----------------------------------
    1) Direct parameterisation
       - parameterisation_method="direct"
       - direct_parameterisation_subtype="xyz"    -> coeffs are (N,3)
       - direct_parameterisation_subtype="normal" -> coeffs are (N,) along normals

    2) Modal parameterisation
       - local Laplacian modes
       - optional global modes
       - local-only / global-only / hybrid
       - optional PCA-reduced branch

    Modal coefficient layout
    ------------------------
    normal_project=True:
        local only        -> [ local(k) ]
        with globals      -> [ global(n_global), local(k) ]

    normal_project=False:
        vector_mode='xyz':
            local only    -> [x(k), y(k), z(k)]
            with globals  -> [global(n_global), x(k), y(k), z(k)]

        vector_mode='local_frame':
            local only    -> [n(k)] or [t1(k), n(k)] or [t1(k), t2(k), n(k)]
            with globals  -> [global(n_global), ...local coeffs...]

    PCA branch
    ----------
    If use_pca=True, coeffs are interpreted in PCA space and reconstructed
    directly to a flattened (N*3,) displacement vector, then reshaped to (N,3).
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # load control nodes if needed
    # ------------------------------------------------------------
    if control_nodes is None:
        cn_path = os.path.join(output_dir, "Control Nodes", "control_nodes.npy")
        control_nodes = np.load(cn_path)

    control_nodes = np.asarray(control_nodes, float)
    if control_nodes.ndim != 2 or control_nodes.shape[1] != 3:
        raise ValueError(f"control_nodes must be (N,3). Got {control_nodes.shape}")

    N = int(control_nodes.shape[0])

    # ------------------------------------------------------------
    # one-node special case
    # ------------------------------------------------------------
    if N == 1:
        if parameterisation_method == "direct":
            c = np.asarray(coeffs, float).reshape(-1) if coeffs is not None else np.zeros(3, dtype=float)
            if direct_parameterisation_subtype == "xyz":
                if c.size < 3:
                    c = np.pad(c, (0, 3 - c.size))
                else:
                    c = c[:3]
                d = c.reshape(1, 3)
            elif direct_parameterisation_subtype == "normal":
                if normals is None:
                    raise ValueError("Direct normal displacement requires normals.")
                nvec = np.asarray(normals, float).reshape(1, 3)
                a = float(c[0]) if c.size else 0.0
                d = a * nvec
            else:
                raise ValueError("For parameterisation_method='direct', subtype must be 'xyz' or 'normal'.")

            if t_patch_scale is not None:
                d *= float(amp_alpha) * float(t_patch_scale)
            return d

        if normals is None:
            normals = np.array([[0.0, 0.0, 1.0]], dtype=float)
        else:
            normals = np.asarray(normals, float).reshape(1, 3)

        c = np.asarray(coeffs, float).reshape(-1) if coeffs is not None else np.zeros(1, dtype=float)
        x0 = float(c[0]) if c.size else 0.0
        amp = x0
        if t_patch_scale is not None:
            amp = x0 * float(amp_alpha) * float(t_patch_scale)
        return amp * normals

    # ------------------------------------------------------------
    # common physical scales
    # ------------------------------------------------------------
    from sklearn.neighbors import NearestNeighbors

    n_query = min(4, len(control_nodes))
    nn = NearestNeighbors(n_neighbors=n_query).fit(control_nodes)
    dists, _ = nn.kneighbors(control_nodes, return_distance=True)
    d_ref = float(dists[:, -1].mean()) if dists.shape[1] > 1 else 1.0

    len_scale = float(t_patch_scale) if t_patch_scale is not None else d_ref
    amp_scale = float(amp_alpha) * float(len_scale)

    if global_amp_alpha is None:
        global_amp_alpha = 3.0 * float(amp_alpha)
    global_scale = float(global_amp_alpha) * float(len_scale)

    print(
        f"[DEBUG] len_scale={len_scale:.6f}, d_ref={d_ref:.6f}, "
        f"amp_scale={amp_scale:.6e}, global_scale={global_scale:.6e}"
    )

    # ------------------------------------------------------------
    # direct control-node parameterisation
    # ------------------------------------------------------------
    if str(parameterisation_method).lower() == "direct":
        subtype = str(direct_parameterisation_subtype or "").strip().lower()
        if subtype not in ("xyz", "normal"):
            raise ValueError(
                "For parameterisation_method='direct', "
                "direct_parameterisation_subtype must be 'xyz' or 'normal'."
            )

        if coeffs is None:
            if subtype == "xyz":
                coeffs = np.zeros(3 * N, dtype=float)
            else:
                coeffs = np.zeros(N, dtype=float)

        coeffs = np.asarray(coeffs, float).reshape(-1)

        if subtype == "xyz":
            expected = 3 * N
            if coeffs.size != expected:
                print(f"[WARN] direct xyz expects {expected} coeffs, got {coeffs.size}. Padding/truncating.")
                if coeffs.size < expected:
                    coeffs = np.pad(coeffs, (0, expected - coeffs.size))
                else:
                    coeffs = coeffs[:expected]
            d_ctrl = coeffs.reshape(N, 3) * amp_scale

        else:  # "normal"
            if normals is None:
                print("[WARN] direct normal requested but no normals provided, estimating from control_nodes")
                normals = estimate_normals(control_nodes, knn=12)
            normals = np.asarray(normals, float).reshape(N, 3)

            expected = N
            if coeffs.size != expected:
                print(f"[WARN] direct normal expects {expected} coeffs, got {coeffs.size}. Padding/truncating.")
                if coeffs.size < expected:
                    coeffs = np.pad(coeffs, (0, expected - coeffs.size))
                else:
                    coeffs = coeffs[:expected]

            d_ctrl = (coeffs[:, None] * normals) * amp_scale

        d_norms = np.linalg.norm(d_ctrl, axis=1)
        print(
            f"[DEBUG] Direct displacement norms: mean={d_norms.mean():.6f}, "
            f"max={d_norms.max():.6f}, std={d_norms.std():.6f}"
        )
        return d_ctrl

    # ------------------------------------------------------------
    # PCA-reduced branch
    # ------------------------------------------------------------
    if use_pca:
        if not pca_cache_path:
            raise ValueError("use_pca=True but no pca_cache_path provided.")
        if pca_coeffs is None:
            raise ValueError("use_pca=True but no pca_coeffs provided.")

        from MeshGeneration.pcaBasis import load_pca_basis

        p = load_pca_basis(pca_cache_path)
        z = np.asarray(pca_coeffs, float).reshape(-1)

        k_red = int(p.V.shape[1])
        if z.size != k_red:
            print(f"[WARN] PCA coeff size {z.size} != expected {k_red}. Padding/truncating.")
            if z.size < k_red:
                z = np.pad(z, (0, k_red - z.size))
            else:
                z = z[:k_red]

        flat = np.asarray(p.mean, float).reshape(-1) + (np.asarray(p.V, float) @ z)
        expected_flat = 3 * N
        if flat.size != expected_flat:
            raise ValueError(
                f"PCA reconstructed flat displacement has size {flat.size}, "
                f"expected {expected_flat} for N={N}."
            )

        d_ctrl = flat.reshape(N, 3)

        d_norms = np.linalg.norm(d_ctrl, axis=1)
        print(
            f"[DEBUG] PCA displacement norms: mean={d_norms.mean():.6f}, "
            f"max={d_norms.max():.6f}, std={d_norms.std():.6f}"
        )
        return d_ctrl

    # ------------------------------------------------------------
    # modal parameterisation
    # ------------------------------------------------------------
    parameterisation_method = "modal"

    if global_only:
        use_local_modes = False

    # ---- global basis (only if needed)
    G = None
    gnames = []
    n_global = 0
    if global_modes:
        G, gnames = build_global_modes(
            control_nodes,
            axes=basis_axes,
            mode_config=global_mode_config
        )
        n_global = int(G.shape[1])
        print(f"[DEBUG] Using {n_global} global modes: {gnames}")

    # ---- local Laplacian basis (only if needed)
    phi = None
    k = 0
    if use_local_modes:
        basis_path = os.path.join(output_dir, cache_name)
        need_rebuild = True

        if os.path.exists(basis_path):
            try:
                phi, _ = load_basis(basis_path)
                need_rebuild = (phi is None) or (phi.shape[0] != int(control_nodes.shape[0]))
                if need_rebuild:
                    print(
                        f"[WARN] Cached basis rows ({None if phi is None else phi.shape[0]}) != "
                        f"current CN count ({control_nodes.shape[0]}). Rebuilding."
                    )
            except Exception as e:
                print(f"[WARN] Failed to load cached basis '{basis_path}': {e}. Rebuilding.")
                need_rebuild = True

        if need_rebuild:
            print(f"[DEBUG] Building new basis with k_modes={k_modes}")
            out = build_laplacian_basis(control_nodes, k_modes=k_modes, knn=6)
            _, phi = out if isinstance(out, tuple) else (None, out)
            save_basis(basis_path, phi, normals=None)

        N_phi, k = phi.shape
        print(f"[DEBUG] Basis: N={N_phi} nodes, k={k} modes")

    # if neither local nor global are enabled, return zeros
    if (not use_local_modes) and (not global_modes):
        print("[WARN] Neither local modes nor global modes are enabled; returning zero displacement.")
        return np.zeros((N, 3), dtype=float)

    # ------------------------------------------------------------
    # normal-projected modal branch
    # ------------------------------------------------------------
    if use_local_modes and normal_project:
        if normals is None:
            print("[WARN] normal_project=True but no normals provided, estimating from control_nodes")
            normals = estimate_normals(control_nodes, knn=12)
        normals = np.asarray(normals, float)

        expected_len = n_global + k
        if coeffs is None:
            j = np.arange(1, k + 1, dtype=float)
            decay = 1.0 / (j**2.0)
            local = rng.normal(0.0, 1.0, size=k) * decay
            if global_modes:
                global_c = rng.normal(0.0, 1.0, size=n_global)
                coeffs = np.concatenate([global_c, local])
            else:
                coeffs = local

        coeffs = np.asarray(coeffs, float).reshape(-1)
        if coeffs.size != expected_len:
            print(f"[WARN] coeffs size {coeffs.size} != expected {expected_len}, padding/truncating")
            if coeffs.size < expected_len:
                coeffs = np.pad(coeffs, (0, expected_len - coeffs.size))
            else:
                coeffs = coeffs[:expected_len]

        if global_modes:
            cg = coeffs[:n_global] * global_scale
            cl = coeffs[n_global:n_global + k] * amp_scale
            d_global = np.sum(G * cg[None, :, None], axis=1)
        else:
            cl = coeffs * amp_scale
            d_global = np.zeros((N, 3), dtype=float)

        d_local = expand_modal_coeffs(phi, cl, normals=normals)
        d_ctrl = d_global + d_local

    # ------------------------------------------------------------
    # non-normal modal branch
    # ------------------------------------------------------------
    elif use_local_modes and (not normal_project):
        if vector_mode not in ("xyz", "local_frame"):
            raise ValueError("vector_mode must be 'xyz' or 'local_frame'")

        if coeffs is None:
            default_local_len = 3 * k
            coeffs = np.zeros(n_global + default_local_len, dtype=float)
        else:
            coeffs = np.asarray(coeffs, float).reshape(-1)

        if global_modes:
            if coeffs.size < n_global:
                coeffs = np.pad(coeffs, (0, n_global - coeffs.size))
            cg_raw = coeffs[:n_global]
            local_raw = coeffs[n_global:]
        else:
            cg_raw = None
            local_raw = coeffs

        if vector_mode == "xyz":
            expected_local = 3 * k
            if local_raw.size != expected_local:
                print(f"[WARN] xyz mode expects {expected_local} local coeffs, got {local_raw.size}. Padding/truncating.")
                if local_raw.size < expected_local:
                    local_raw = np.pad(local_raw, (0, expected_local - local_raw.size))
                else:
                    local_raw = local_raw[:expected_local]

            local_scaled = local_raw * amp_scale
            d_local = expand_modal_coeffs(phi, local_scaled, normals=None)

        else:  # local_frame
            if local_raw.size not in (k, 2 * k, 3 * k):
                print(f"[WARN] local_frame expects k, 2k or 3k local coeffs. Got {local_raw.size}. Padding to 3k.")
                if local_raw.size < 3 * k:
                    local_raw = np.pad(local_raw, (0, 3 * k - local_raw.size))
                else:
                    local_raw = local_raw[: 3 * k]

            local_scaled = local_raw * amp_scale
            if local_scaled.size == k:
                c1 = np.zeros(k)
                c2 = np.zeros(k)
                cn = local_scaled
            elif local_scaled.size == 2 * k:
                c1 = local_scaled[:k]
                c2 = np.zeros(k)
                cn = local_scaled[k:2 * k]
            else:
                c1 = local_scaled[:k]
                c2 = local_scaled[k:2 * k]
                cn = local_scaled[2 * k:3 * k]

            s1 = phi @ c1
            s2 = phi @ c2
            sn = phi @ cn
            t1, t2, nvec = estimate_local_frame(control_nodes, knn=frame_knn)
            d_local = (s1[:, None] * t1) + (s2[:, None] * t2) + (sn[:, None] * nvec)

        if global_modes:
            cg = cg_raw * global_scale
            d_global = np.sum(G * cg[None, :, None], axis=1)
        else:
            d_global = np.zeros_like(d_local)

        d_ctrl = d_global + d_local

    # ------------------------------------------------------------
    # global-only modal branch
    # ------------------------------------------------------------
    else:
        if not global_modes or G is None or n_global <= 0:
            print("[WARN] global_only/use_local_modes=False but no global modes are enabled; returning zeros.")
            d_ctrl = np.zeros((N, 3), dtype=float)
        else:
            if coeffs is None:
                coeffs = rng.normal(0.0, 1.0, size=n_global)
            coeffs = np.asarray(coeffs, float).reshape(-1)
            if coeffs.size != n_global:
                print(f"[WARN] global-only expects {n_global} coeffs, got {coeffs.size}. Padding/truncating.")
                if coeffs.size < n_global:
                    coeffs = np.pad(coeffs, (0, n_global - coeffs.size))
                else:
                    coeffs = coeffs[:n_global]

            cg = coeffs * global_scale
            d_ctrl = np.sum(G * cg[None, :, None], axis=1)

    # ------------------------------------------------------------
    # smoothing
    # ------------------------------------------------------------
    if (not global_modes) or smooth_global:
        d_ctrl = laplacian_smooth(control_nodes, d_ctrl, iters=1)

    d_norms = np.linalg.norm(d_ctrl, axis=1)
    print(
        f"[DEBUG] Displacement norms: mean={d_norms.mean():.6f}, "
        f"max={d_norms.max():.6f}, std={d_norms.std():.6f}"
    )
    return d_ctrl