import os, sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname("MeshGeneration"))
sys.path.append(os.path.dirname("ConvertFileType"))
sys.path.append(os.path.dirname("Utilities"))
from MeshGeneration.controlNodeDisp import _surface_normals, _map_normals_to_control, _spectral_coeffs, estimate_normals, getDisplacements
from MeshGeneration.MorphModel import *
from MeshGeneration.Rbf import *
from MeshGeneration.BasisFunctions import get_bf
from MeshGeneration.Morph import MorphMesh
from ConvertFileType.convertVtmtoFro import *
from Utilities.PointClouds import *

# 6) Optional bump window (for DSI): Gaussian windowing around a footprint
def _apply_bump_window(viewer, cn, cn_normals, d_ctrl, logger=None):
    """
    Apply a Gaussian bump window to control-node displacements.
    - Safe guards for missing/invalid inputs
    - Numerically stable for small/large radii
    - Optional one-sided clipping removes only the negative *normal component*
    """
    # Feature toggle
    if not getattr(viewer, "bump_enable", False):
        return d_ctrl

    c0 = getattr(viewer, "bump_center", None)
    r  = getattr(viewer, "bump_radius", None)

    # Validate inputs
    if c0 is None or r is None:
        if logger: logger.log("[BUMP] Skipped: center/radius not set.")
        return d_ctrl
    try:
        c0 = np.asarray(c0, dtype=float).reshape(3)
        r  = float(r)
    except Exception:
        if logger: logger.log("[BUMP] Skipped: could not parse center/radius.")
        return d_ctrl

    if not np.isfinite(r) or r <= 0.0:
        if logger: logger.log(f"[BUMP] Skipped: invalid radius r={r}.")
        return d_ctrl

    # Compute squared distances to center
    dx = cn - c0[None, :]
    d2 = np.einsum("ij,ij->i", dx, dx)

    # Gaussian weights: w = exp(-0.5 * (dist / r)^2)
    # (interpret r as Gaussian sigma)
    # To avoid underflow, clamp very large d2/r^2 (optional but harmless)
    s2 = r * r
    z  = -0.5 * d2 / s2
    z  = np.maximum(z, -50.0)       # exp(-50) ~ 1.9e-22
    w  = np.exp(z)                  # shape (N,)

    # Apply radial window
    d_ctrl = d_ctrl * w[:, None]

    # One-sided: remove only negative *normal* components (keep tangential part)
    if getattr(viewer, "bump_one_sided", False):
        # Ensure normals are unit-length to interpret dot product as signed magnitude
        n = cn_normals
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n_safe = np.divide(n, np.clip(n_norm, 1e-12, None))
        a_n = np.einsum("ij,ij->i", d_ctrl, n_safe)          # signed normal component
        neg = a_n < 0.0
        if np.any(neg):
            # Remove only the negative normal component: d <- d - a_n * n
            d_ctrl[neg] = d_ctrl[neg] - (a_n[neg, None] * n_safe[neg])
    return d_ctrl

def runSurfMorph(viewer, n=0, debug=True, surf_dir=os.getcwd()):
    logger = viewer.logger if hasattr(viewer, "logger") else None
    base_name = os.path.basename(viewer.input_filepath).rsplit(".", 1)[0]
    # Load .fro file from .vtm/.vtk or directly
    if n == 0 or n == 1:
        if viewer.input_filepath.endswith(('.vtm', '.vtk')):
            # Get input base name
            fro_name = base_name + ".fro"  # Replace .vtm/.vtk with .fro
            local_fro_path = os.path.join(viewer.rbf_original, fro_name)

            # Convert and save locally
            logger.log("[LOCAL] Converting to .fro")
            mesh = vtm_to_fro(viewer.input_filepath, local_fro_path)
            logger.log(f"Saved .fro locally as: {local_fro_path}")
        else:
            mesh = FroFile(viewer.input_filepath)


    logger.log(f"[CHECK] T={viewer.TSurfaces}, U={viewer.USurfaces}, C={viewer.CSurfaces}")

    t_ids = [int(s) if str(s).isdigit() else int(viewer.mesh_obj.get_surface_id(s)) for s in viewer.TSurfaces]
    u_ids = [int(s) if str(s).isdigit() else int(viewer.mesh_obj.get_surface_id(s)) for s in viewer.USurfaces]
    c_ids = [int(s) if str(s).isdigit() else int(viewer.mesh_obj.get_surface_id(s)) for s in viewer.CSurfaces]

    tmp_morpher = MorphModel(f_name=None, con=None, pb=[], T=[], U=[], cn=[])
    tmp_morpher.t_surfaces = t_ids
    # get T-surface vertices for normals
    t_gtl, t_verts = tmp_morpher.get_t_node_vertices(mesh)  # (gid→local, points)

    # 3) Normals: from T-surface → averaged onto control nodes
    cn = np.asarray(viewer.control_nodes, float)
    if len(t_verts) >= 4:
        surf_normals = _surface_normals(t_verts, knn=16)
        cn_normals = _map_normals_to_control(cn, t_verts, surf_normals, k=8)
    else:
        # Fallback to control-node-only estimate if T is tiny
        if logger: logger.log("[WARN] Too few T vertices for surface normals; falling back to control-node PCA normals.")
        cn_normals = estimate_normals(cn, knn=12)

    # 4) Smooth modal sampling (spectral decay + mesh-scale amplitude)
    k_modes = getattr(viewer, "k_modes", 10)
    rng_seed = getattr(viewer, "seed", 0)
    p = getattr(viewer, "spectral_p", 2.0)
    frac = getattr(viewer, "coeff_frac", 0.15)

    coeffs = getattr(viewer, "modal_coeffs", None)
    if coeffs is None:
        coeffs = _spectral_coeffs(k_modes, cn, rng=rng_seed, p=p, frac=frac)

    d_ctrl = getDisplacements(
        viewer.output_dir,
        seed=rng_seed,
        control_nodes=cn,
        normals=cn_normals,
        coeffs=coeffs,                 
        k_modes=k_modes,
        normal_project=getattr(viewer, "normal_project", True)
    )
    
    # Debugging rigid translation
    vbar   = d_ctrl.mean(axis=0)
    spread = np.linalg.norm(d_ctrl - vbar, axis=1).max()
    logger.log(f"[DEBUG] mean disp = {vbar}, max dev from mean = {spread:.3e}")

    # Optional: remove rigid component (helps avoid 'everything slides together')
    remove_rigid = getattr(viewer, "remove_rigid_component", True)
    if remove_rigid:
        d_ctrl = d_ctrl - vbar
        logger.log("[DEBUG] Removed rigid (mean) component from control-node displacements.")

    # Apply optional bump windowing (if enabled in the UI)
    d_ctrl = _apply_bump_window(viewer, cn, cn_normals, d_ctrl, logger=logger)
    d_ctrl = d_ctrl*10

    # --- make results available to the GUI for immediate preview
    viewer.cn_points        = cn
    viewer.cn_displacements = d_ctrl
    viewer.cn_targets       = cn + d_ctrl

    # enqueue on GUI thread safely
    try:
        viewer.requestPlotCNs.emit()
    except Exception as e:
        if logger: logger.log(f"[WARN] Could not enqueue CN preview: {e}")

    if logger:
        max_mag = float(np.linalg.norm(d_ctrl, axis=1).max())
        if 'coeffs' in locals():
            logger.log(f"[MODAL] k={k_modes}, |d|_max={max_mag:.3e}, coeff_norm={np.linalg.norm(coeffs):.3e}")
        else:
            logger.log(f"[MODAL] k={k_modes}, |d|_max={max_mag:.3e}")
    
    # Add 0 disp anchors on U & C surfaces
    def _sample_surface_vertices(surface_ids, max_pts_per_surf=500):
        pts = []
        for sid in surface_ids:
            # You'll have similar helper in MorphModel like get_u/c_node_vertices; fall back to mesh_obj:
            surf = viewer.mesh_obj.get_surface_mesh(viewer.mesh_obj.get_surface_name(sid))
            if surf is None:
                continue
            P = np.asarray(surf.points, float)
            if len(P) == 0: 
                continue
            if len(P) > max_pts_per_surf:
                # uniform sampling
                idx = np.linspace(0, len(P)-1, max_pts_per_surf, dtype=int)
                P = P[idx]
            pts.append(P)
        return np.vstack(pts) if pts else np.zeros((0,3), float)
    
    u_anchor = _sample_surface_vertices(u_ids, max_pts_per_surf=1000)
    c_anchor = _sample_surface_vertices(c_ids, max_pts_per_surf=1000)
    anchors  = np.vstack([u_anchor, c_anchor]) if (len(u_anchor)+len(c_anchor))>0 else np.zeros((0,3), float)

    # Build augmented control set with zeros on anchors
    if anchors.shape[0] > 0:
        cn_aug   = np.vstack([cn, anchors])
        d_aug    = np.vstack([d_ctrl, np.zeros((anchors.shape[0], 3), float)])
    else:
        cn_aug, d_aug = cn, d_ctrl
    
    logger.log(f"[INFO] Control Nodes = {cn.tolist()}")
    logger.log(f"[INFO] Displacements = {d_ctrl.tolist()}")
    
    # 7) Hand data to MorphModel and run morph
    morpher = MorphModel(f_name=None, con=None, pb=[], T=[], U=[], cn=[])
    morpher.t_surfaces = t_ids
    morpher.u_surfaces = u_ids
    morpher.c_surfaces = c_ids
    morpher.control_nodes = cn_aug.tolist()
    morpher.displacement_vector = d_aug

    m_m = MorphMesh(mesh, base_name, morpher, viewer, viewer.output_dir, debug=debug, rb="original")
    return m_m
    

def saveMorphedMesh(viewer, points):
    logger = viewer.logger if hasattr(viewer, "logger") else None
    blocks = pv.MultiBlock()

    kdtree = cKDTree(points)

    for name in viewer.mesh_obj.get_surface_names():
        surf = viewer.mesh_obj.get_surface_mesh(name).copy()
        surf_pts = surf.points

        try:
            _, idx = kdtree.query(surf_pts)
            surf.points = points[idx]
            blocks[name] = surf
        except Exception as e:
            logger.log(f"Failed to update surface '{name}': {e}")
            continue

    save_path = os.path.join(viewer.output_dir, f"crm.vtm")
    blocks.save(save_path)
    logger.log(f"Morphed mesh saved to: {save_path}")
    return blocks

