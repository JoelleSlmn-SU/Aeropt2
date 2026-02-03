import numpy as np

def _bbox_diag(X: np.ndarray) -> float:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return float(np.linalg.norm(maxs - mins))

def _triangle_areas(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    # tri: (M,3) node ids
    a = P[tri[:, 0]]
    b = P[tri[:, 1]]
    c = P[tri[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

def _triangle_normals(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a = P[tri[:, 0]]
    b = P[tri[:, 1]]
    c = P[tri[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(nn, 1e-15)

def _edge_lengths(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a = P[tri[:, 0]]
    b = P[tri[:, 1]]
    c = P[tri[:, 2]]
    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)
    return np.stack([e0, e1, e2], axis=1)  # (M,3)

def _laplacian_energy(U: np.ndarray, node_connections: dict, gids: np.ndarray) -> float:
    # mean ||u_i - mean(u_neighbors)||^2 over gids
    # node_connections: {gid: [neighbor_gids...]}
    acc = 0.0
    cnt = 0
    for g in gids:
        nb = node_connections.get(int(g), [])
        if not nb:
            continue
        nb = np.asarray(nb, dtype=np.int64)
        mu = U[nb].mean(axis=0)
        du = U[g] - mu
        acc += float(du @ du)
        cnt += 1
    return acc / max(cnt, 1)

def compute_mesh_fail_features(ff0, ff1, morph_model=None):
    """
    ff0: original FroFile
    ff1: morphed FroFile (same node ids)
    morph_model: optional MorphModel (lets us compute T/U/C/D anchors cleanly)
    Returns: dict of scalar features
    """
    X0 = np.asarray(ff0.nodes, dtype=float)
    X1 = np.asarray(ff1.nodes, dtype=float)
    U  = X1 - X0

    # --- choose deforming region gids (best) ---
    if morph_model is not None:
        t = set(morph_model.get_t_node_gids(ff0))
        u = set(morph_model.get_u_node_gids(ff0))
        c = set(morph_model.get_c_node_gids(ff0))
        D = np.asarray(sorted(t | u), dtype=np.int64)
        C = set(c)

        # anchors: same logic as MorphMesh :contentReference[oaicite:6]{index=6}
        shared = sorted((t | u) & c)
        if len(shared) > 0:
            anchors = np.asarray(shared, dtype=np.int64)
        else:
            anchors = []
            for g in D:
                if any(nb in C for nb in ff0.node_connections.get(int(g), [])):
                    anchors.append(int(g))
            anchors = np.asarray(sorted(set(anchors)), dtype=np.int64)
    else:
        # fallback: use all boundary nodes if no model supplied
        D = np.arange(X0.shape[0], dtype=np.int64)
        anchors = np.empty((0,), dtype=np.int64)

    XD0 = X0[D]
    L = max(_bbox_diag(XD0), 1e-12)

    umag = np.linalg.norm(U[D], axis=1)
    feats = {
        "L": L,
        "u_max_L": float(np.max(umag) / L) if umag.size else 0.0,
        "u_mean_L": float(np.mean(umag) / L) if umag.size else 0.0,
        "u_p95_L": float(np.quantile(umag, 0.95) / L) if umag.size else 0.0,
        "u_std_L": float(np.std(umag) / L) if umag.size else 0.0,
        "u_rms_L": float(np.sqrt(np.mean(umag**2)) / L) if umag.size else 0.0,
        "lap_energy": float(_laplacian_energy(U, ff0.node_connections, D)),
    }

    # anchor-related
    if anchors.size:
        amag = np.linalg.norm(U[anchors], axis=1)
        feats["u_anchor_max_ratio"] = float(np.max(amag) / (np.max(umag) + 1e-15))
        feats["u_anchor_mean_ratio"] = float(np.mean(amag) / (np.mean(umag) + 1e-15))
        feats["anchor_frac"] = float(len(anchors) / max(len(D), 1))
    else:
        feats["u_anchor_max_ratio"] = 0.0
        feats["u_anchor_mean_ratio"] = 0.0
        feats["anchor_frac"] = 0.0

    # --- surface face features ---
    # NOTE: adjust these attribute names if your FroFile stores them differently
    tri = np.asarray(getattr(ff0, "boundary_triangles", []), dtype=np.int64)
    if tri.size:
        # common formats: (M,3) or (M,4) where last is surf id
        if tri.shape[1] >= 4:
            tri_nodes = tri[:, :3]
        else:
            tri_nodes = tri

        A0 = _triangle_areas(X0, tri_nodes)
        A1 = _triangle_areas(X1, tri_nodes)
        ar = A1 / np.maximum(A0, 1e-15)

        e0 = _edge_lengths(X0, tri_nodes)
        e1 = _edge_lengths(X1, tri_nodes)
        er = e1 / np.maximum(e0, 1e-15)  # (M,3)

        n0 = _triangle_normals(X0, tri_nodes)
        n1 = _triangle_normals(X1, tri_nodes)
        flipped = np.sum(np.einsum("ij,ij->i", n0, n1) < 0.0)

        feats.update({
            "area_ratio_min": float(np.min(ar)),
            "area_ratio_p01": float(np.quantile(ar, 0.01)),
            "area_ratio_p50": float(np.quantile(ar, 0.50)),
            "area_ratio_p99": float(np.quantile(ar, 0.99)),
            "edge_ratio_min": float(np.min(er)),
            "edge_ratio_p01": float(np.quantile(er, 0.01)),
            "edge_ratio_p99": float(np.quantile(er, 0.99)),
            "tri_flipped_frac": float(flipped / max(len(tri_nodes), 1)),
        })
    else:
        feats.update({
            "area_ratio_min": 1.0, "area_ratio_p01": 1.0, "area_ratio_p50": 1.0, "area_ratio_p99": 1.0,
            "edge_ratio_min": 1.0, "edge_ratio_p01": 1.0, "edge_ratio_p99": 1.0,
            "tri_flipped_frac": 0.0,
        })

    return feats
