"""
morphPropagation.py

Single source of truth for the *surface-propagation* stage of the production
morph (control-node displacements -> surface-mesh displacements).

Both of these call the functions below, so they cannot drift apart:
  * MeshGeneration.Morph.MorphMesh        (cluster / remoteMorph.py / local runs)
  * GUI.modal_explorer_gui                (interactive preview)

Stages
------
1. classify_regions(mesh_in, morph_model)
     D = T ∪ U (deforming), C (fixed), anchors = D nodes shared with / adjacent to C.
     Depends only on the baseline mesh + surface roles -> cache it.
2. rbf_parameters(control_nodes, d_verts, anchor_points)
     Adaptive Wendland support parameters exactly as MorphMesh uses them.
     Depends only on CN positions + D + anchors -> cache it.
3. deform_region(morph_model, regions, params)
     MorphModel.transformT on D with the production flags.
     This is the only step that depends on the displacement vector.
4. seam_report(regions, d_verts_m)
     Diagnostic: how far the D<->C seam nodes moved (should be ~0 for a
     watertight, unsheared seam; see note in seam_report).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Production constants. Change them HERE and both MorphMesh and the explorer
# pick them up.
# ---------------------------------------------------------------------------
RBF_K_NN = 3
RBF_BETA = 2.2            # < 1.0 = more local, > 1.0 = smoother
RBF_MIN_CLIP_FRAC = 0.01
RBF_MAX_CLIP_FRAC = 0.4
TRANSFORM_ANCHOR_TAPER = False
TRANSFORM_BOUNDARY_RECOVER = False


def compute_adaptive_rbf_params(control_nodes, d_verts, anchor_points=None,
                                k_nn=3,
                                beta=1.0,
                                min_clip_frac=0.01,
                                max_clip_frac=0.15):
    """
    Returns:
        min_R_frac, fallback_R_frac, R_scale

    Idea:
      - Use kNN spacing between control nodes as the main locality measure
      - Convert to fractions of the T/U patch scale L
      - Keep R_scale as the main global multiplier knob

    (Moved verbatim from Morph.py; Morph.py re-exports it.)
    """
    ctrl = np.asarray(control_nodes, dtype=float)
    T = np.asarray(d_verts, dtype=float)

    if ctrl.ndim != 2 or ctrl.shape[1] != 3:
        raise ValueError(f"control_nodes must be (N,3), got {ctrl.shape}")
    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError(f"d_verts must be (M,3), got {T.shape}")

    # Patch scale used by transformT internally as well
    mins = T.min(axis=0)
    maxs = T.max(axis=0)
    L = float(np.linalg.norm(maxs - mins))
    L = max(L, 1e-12)

    n_ctrl = ctrl.shape[0]

    # ---- one-control special case ----
    if n_ctrl == 1:
        if anchor_points is not None and len(anchor_points) > 0:
            A = np.asarray(anchor_points, dtype=float).reshape(-1, 3)
            da = np.linalg.norm(A - ctrl[0][None, :], axis=1)
            d_ref = float(np.median(da)) if da.size else 0.05 * L
        else:
            d_ref = 0.05 * L

        d_ref = float(np.clip(d_ref, min_clip_frac * L, max_clip_frac * L))
        min_R_frac = d_ref / L
        fallback_R_frac = d_ref / L
        R_scale = beta
        return min_R_frac, fallback_R_frac, R_scale

    # ---- control-node kNN spacing ----
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ctrl)
        k_eff = min(max(2, k_nn + 1), n_ctrl)   # +1 because self is included
        dists, _ = tree.query(ctrl, k=k_eff)
        # last column = distance to k_nn-th neighbour
        d_knn = dists[:, -1]
    except Exception:
        # fallback brute force
        diff = ctrl[:, None, :] - ctrl[None, :, :]
        D = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(D, np.inf)
        k_eff = min(max(1, k_nn), n_ctrl - 1)
        d_knn = np.partition(D, kth=k_eff - 1, axis=1)[:, k_eff - 1]

    d_knn = np.asarray(d_knn, float)
    d_knn = d_knn[np.isfinite(d_knn)]
    if d_knn.size == 0:
        d_knn = np.array([0.05 * L], dtype=float)

    # Robust spacing stats
    d_p10 = float(np.percentile(d_knn, 10))
    d_p50 = float(np.percentile(d_knn, 50))
    d_p90 = float(np.percentile(d_knn, 90))

    # Clip to sensible fractions of patch size
    d_min = float(np.clip(d_p10, min_clip_frac * L, max_clip_frac * L))
    d_typ = float(np.clip(d_p50, min_clip_frac * L, max_clip_frac * L))
    d_max = float(np.clip(d_p90, min_clip_frac * L, max_clip_frac * L))  # noqa: F841 (kept for parity)

    # Convert to transformT-style parameters
    min_R_frac = d_min / L
    fallback_R_frac = d_typ / L

    # beta is your main locality knob:
    #   smaller beta -> more local
    #   larger beta  -> smoother / more global
    R_scale = float(beta)

    print(
        "[ADAPT-RBF] "
        f"L={L:.6f}, "
        f"d10={d_p10:.6f}, d50={d_p50:.6f}, d90={d_p90:.6f}, "
        f"min_R_frac={min_R_frac:.6f}, "
        f"fallback_R_frac={fallback_R_frac:.6f}, "
        f"R_scale={R_scale:.6f}"
    )

    return min_R_frac, fallback_R_frac, R_scale


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------
@dataclass
class MorphRegions:
    t_gids: set
    u_gids: set
    c_gids: set
    t_gtl: Dict[int, int]
    t_verts: object
    u_gtl: Dict[int, int]
    u_verts: object
    d_gids: List[int]                 # sorted(T ∪ U)
    d_gtl: Dict[int, int]
    d_verts: np.ndarray               # (|D|, 3) baseline coordinates of D
    shared: List[int]                 # (T ∪ U) ∩ C
    anchor_gids: List[int]
    anchor_points: list               # list of (3,) baseline coordinates
    anchor_local: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # anchor_local: indices of anchor_gids inside d_gids (all anchors are D nodes)


def classify_regions(mesh_in, morph_model) -> MorphRegions:
    """
    D = T ∪ U, C fixed, anchors = D nodes shared with C (or, if none are shared,
    D nodes with a C neighbour). Identical logic to the original MorphMesh STEP 2.

    NOTE: uses mesh_in.node_connections for the fallback anchor search.
    MorphMesh calls mesh_in.copy() first, which guarantees it exists; we only
    build it here if it is genuinely missing.
    """
    t_gids = set(morph_model.get_t_node_gids(mesh_in))
    u_gids = set(morph_model.get_u_node_gids(mesh_in))
    c_gids = set(morph_model.get_c_node_gids(mesh_in))

    # mappings and coords for T and U (needed later to split)
    t_gtl, t_verts = morph_model.get_t_node_vertices(mesh_in)  # {gid: local}, coords
    u_gtl, u_verts = ({}, [])
    if len(u_gids) > 0:
        u_gtl, u_verts = morph_model.get_u_node_vertices(mesh_in)

    # build D (union)
    d_gids = sorted(t_gids | u_gids)
    d_gtl, d_verts = mesh_in.convert_node_ids_to_coordinates(d_gids)

    # Get shared points with C subset
    shared = sorted((t_gids | u_gids) & c_gids)
    if len(shared) > 0:
        anchor_gids = shared
    else:
        if not getattr(mesh_in, "node_connections", None):
            mesh_in.set_node_connections()
        anchor_gids = [
            g for g in d_gids
            if any(nb in c_gids for nb in mesh_in.node_connections.get(g, []))
        ]
    anchor_points = [mesh_in.nodes[g] for g in anchor_gids]
    anchor_local = np.asarray([d_gtl[g] for g in anchor_gids], dtype=np.int64)

    return MorphRegions(
        t_gids=t_gids, u_gids=u_gids, c_gids=c_gids,
        t_gtl=t_gtl, t_verts=t_verts, u_gtl=u_gtl, u_verts=u_verts,
        d_gids=d_gids, d_gtl=d_gtl, d_verts=np.asarray(d_verts, float),
        shared=shared, anchor_gids=anchor_gids, anchor_points=anchor_points,
        anchor_local=anchor_local,
    )


# ---------------------------------------------------------------------------
# RBF parameters
# ---------------------------------------------------------------------------
def rbf_parameters(control_nodes, d_verts, anchor_points) -> dict:
    """Adaptive transformT parameters, exactly as MorphMesh sets them."""
    min_R_frac, fallback_R_frac, R_scale = compute_adaptive_rbf_params(
        control_nodes=control_nodes,
        d_verts=d_verts,
        anchor_points=anchor_points,
        k_nn=RBF_K_NN,
        beta=RBF_BETA,
        min_clip_frac=RBF_MIN_CLIP_FRAC,
        max_clip_frac=RBF_MAX_CLIP_FRAC,
    )

    # optional: tighten seam correction a bit too
    corr_R_frac = max(0.5 * min_R_frac, 0.008)
    corr_band_frac = max(0.75 * fallback_R_frac, 0.02)

    if len(control_nodes) == 1:
        min_R_frac = 0.5
        fallback_R_frac = 0.75
        R_scale = 2.0

        corr_R_frac = 0.01
        corr_band_frac = 0.10

    print(
        "[ADAPT-RBF] "
        f"corr_R_frac={corr_R_frac:.6f}, "
        f"corr_band_frac={corr_band_frac:.6f}"
    )

    return dict(
        min_R_frac=min_R_frac,
        fallback_R_frac=fallback_R_frac,
        R_scale=R_scale,
        corr_R_frac=corr_R_frac,
        corr_band_frac=corr_band_frac,
    )


# ---------------------------------------------------------------------------
# Deformation of D
# ---------------------------------------------------------------------------
def deform_region(morph_model, regions: MorphRegions, params: dict):
    """
    Deform D directly with MorphModel.transformT and the production flags.
    Returns what transformT returns (list of [x, y, z]).

    morph_model.control_nodes / .displacement_vector must be *lists*
    (transformT does `if not self.control_nodes`).
    """
    return morph_model.transformT(
        regions.d_verts,
        anchor_points=regions.anchor_points,
        min_R_frac=params["min_R_frac"],
        fallback_R_frac=params["fallback_R_frac"],
        R_scale=params["R_scale"],
        anchor_taper=TRANSFORM_ANCHOR_TAPER,
        boundary_recover=TRANSFORM_BOUNDARY_RECOVER,
        corr_R_frac=params["corr_R_frac"],
        corr_band_frac=params["corr_band_frac"],
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def seam_report(regions: MorphRegions, d_verts_m) -> dict:
    """
    Displacement statistics on D and on the D<->C seam (anchor nodes).

    Interpretation: C nodes that are NOT in D never move. Anchor nodes are in D,
    so with anchor_taper=False and boundary_recover=False nothing forces them to
    zero. A non-zero seam displacement means either
      - shared nodes (T∩C) move, dragging the edge of the C faces with them, or
      - D nodes next to C move while their C neighbours stay put -> sheared cells.
    `seam_ratio` = max seam |u| / max D |u|. ~0 is what you want.
    """
    Dm = np.asarray(d_verts_m, float)
    U = Dm - regions.d_verts
    umag = np.linalg.norm(U, axis=1) if U.size else np.zeros(0)
    d_max = float(umag.max()) if umag.size else 0.0
    if regions.anchor_local.size:
        a = umag[regions.anchor_local]
        seam_max = float(a.max())
        seam_rms = float(np.sqrt(np.mean(a ** 2)))
    else:
        seam_max = seam_rms = 0.0
    return dict(
        n_D=int(len(regions.d_gids)),
        n_anchor=int(len(regions.anchor_gids)),
        n_shared=int(len(regions.shared)),
        d_max=d_max,
        d_rms=float(np.sqrt(np.mean(umag ** 2))) if umag.size else 0.0,
        seam_max=seam_max,
        seam_rms=seam_rms,
        seam_ratio=(seam_max / d_max) if d_max > 0 else 0.0,
    )
