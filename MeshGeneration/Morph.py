"""
    This file contains the main morph function implementated by Ben Smith.
"""
import sys, os
import numpy as np
import pyvista as pv
from pyvista import Color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from scipy.spatial.distance import pdist

sys.path.append(os.path.dirname("MeshGeneration"))
sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("Utilities"))
sys.path.append(os.path.dirname("ConvertFileType"))
from FileRW.FroFile import *
from MeshGeneration.MorphModel import *
from MeshGeneration.Rbf import *
from MeshGeneration.ShapeParameterSelection import *
from MeshGeneration.BasisFunctions import get_bf
from Utilities.PointClouds import *
from ConvertFileType.convertFrotoEnsight import convert_fro_to_ensight
from MeshFailureClassifier.sweep.classifier_inputs import *

colors = [
    Color("red").hex_rgb,
    Color("blue").hex_rgb,
    Color("green").hex_rgb,
    Color("orange").hex_rgb,
    Color("purple").hex_rgb,
    Color("teal").hex_rgb,
    Color("gold").hex_rgb,
]

def maintain_cp(f, *args):
    """
        Decorator 
    """
    def wrapper(*args):
        verts, phi = args

        N = len(verts)
        cp_0 = center_point_of_vertices(verts)
        translated_verts = f(*args)
        cp_t = center_point_of_vertices(translated_verts)

        correction = [a-b for a,b in zip(cp_t, cp_0)]
        
        for i in range(0,N):
            translated_verts[i] = [a-b for a,b in zip(translated_verts[i], correction)]
        
        return translated_verts
    return wrapper

def plot_debug_mesh(ff, title="Mesh Debug", exclude_ids=[1, 2]):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    all_points = ff.nodes

    colors = plt.cm.get_cmap('tab20', len(ff.get_surface_ids()))

    for i, sid in enumerate(ff.get_surface_ids()):
        if sid in exclude_ids:
            continue

        _, lc_ids = ff.get_surface_nodes(sid)
        surf_points = ff.nodes[np.array(lc_ids, dtype=int)]

        if len(surf_points) == 0:
            print(f"[DEBUG] Surface {sid} has 0 nodes — skipping")
            continue

        x, y, z = surf_points[:, 0], surf_points[:, 1], surf_points[:, 2]
        ax.scatter(x, y, z, s=5, label=f"Surface {sid}", color=colors(i))

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(10)
    
def deduplicate_points(S, F, tol=1e-8):
    """
    Remove duplicate points from S (source) and associated displacements F.
    Uses a tolerance to avoid floating-point issues.
    """
    unique = {}
    for s, f in zip(S, F):
        key = tuple([round(x / tol) for x in s])  # Key based on rounded coords
        if key not in unique:
            unique[key] = f
    S_dedup = [np.array(key) * tol for key in unique.keys()]
    F_dedup = list(unique.values())
    return S_dedup, F_dedup

def farthest_point_sampling(X, n_keep, seed=0):
    """Simple FPS to thin control points to n_keep."""
    X = np.asarray(X, float)
    N = len(X)
    if N <= n_keep: 
        return np.arange(N)
    rng = np.random.default_rng(seed)
    idxs = [rng.integers(N)]
    d2 = np.full(N, np.inf)
    for _ in range(n_keep - 1):
        last = idxs[-1]
        # squared distances to the new center
        dist2 = np.sum((X - X[last])**2, axis=1)
        d2 = np.minimum(d2, dist2)
        idxs.append(int(np.argmax(d2)))
    return np.array(idxs, dtype=int)

def dedup_sf(S, F, tol=1e-6):
    """Deduplicate S and align F, using a tolerance that’s sane for your units."""
    S = np.asarray(S, float)
    F = np.asarray(F, float)
    # round-based key
    key = np.round(S / tol).astype(np.int64)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return S[uniq_idx], F[uniq_idx]

class DummyLogger:
    def log(self, msg):
        print(msg)

def MorphMesh(mesh_in: FroFile, base_name, morph_model, viewer, output_dir,
             debug=True, rb="original", n=0, gen=0, train_model=False):
    """
    Single-region morph:
      D = T ∪ U  (deforming region)
      C          (fixed region)
      anchors    = D nodes connected to C
    Deform D directly, with decay-to-zero approaching anchors.
    """
    import os
    import numpy as np

    logger = viewer.logger if viewer is not None else DummyLogger()

    mesh_out = mesh_in.copy()

    # --- convenience sets ---
    t_gids = set(morph_model.get_t_node_gids(mesh_in))
    u_gids = set(morph_model.get_u_node_gids(mesh_in))
    c_gids = set(morph_model.get_c_node_gids(mesh_in))

    logger.log(f"[CHECK] T surfaces: {morph_model.t_surfaces}")
    logger.log(f"[CHECK] U surfaces: {morph_model.u_surfaces}")
    logger.log(f"[CHECK] count(T_gids)={len(t_gids)}, count(U_gids)={len(u_gids)}, count(C_gids)={len(c_gids)}")

    # STEP 2 - Deform D = T & U
    logger.log("STEP 2 - Translating (D = T ∪ U)")

    # mappings and coords for T and U (needed later to split)
    t_gtl, t_verts = morph_model.get_t_node_vertices(mesh_in)  # {gid: local}, list of coords
    u_gtl, u_verts = ({}, [])
    if len(u_gids) > 0:
        u_gtl, u_verts = morph_model.get_u_node_vertices(mesh_in)

    # build D (union)
    d_gids = sorted(t_gids | u_gids)
    d_gtl, d_verts = mesh_in.convert_node_ids_to_coordinates(d_gids)  # {gid: local}, list coords

    # Get shared points with C subset
    shared = sorted((t_gids | u_gids) & c_gids)
    if len(shared) > 0:
        anchor_gids = shared
    else:
        anchor_gids = [
            g for g in d_gids
            if any(nb in c_gids for nb in mesh_in.node_connections.get(g, []))
        ]
    anchor_points = [mesh_in.nodes[g] for g in anchor_gids]
    logger.log(f"[STEP2] |D|={len(d_gids)} anchors(D↔C)={len(anchor_gids)} (shared={len(shared)})")

    if len(anchor_gids) == 0:
        logger.log("[WARNING] No D–C anchor nodes found. D may move rigidly relative to C.")
    if len(anchor_gids) == len(d_gids):
        logger.log("[WARNING] All D nodes are anchored to C. Nothing can move.")

    # deform D directly
    d_verts_m = morph_model.transformT(
        d_verts,
        anchor_points=anchor_points,
        anchor_taper=False,        # <-- this restores variance
        boundary_recover=True,     # <-- seam fixed via pass-2 correction
    )

    # split D back into T and U using the gid→local maps
    t_verts_m = [None] * len(t_verts)
    for gid, tl in t_gtl.items():
        t_verts_m[tl] = d_verts_m[d_gtl[gid]]

    u_verts_m = []
    if u_gtl:
        u_verts_m = [None] * len(u_verts)
        for gid, ul in u_gtl.items():
            u_verts_m[ul] = d_verts_m[d_gtl[gid]]

    # Optional debug plot after Step 2
    if debug and viewer is not None and hasattr(viewer, "debug_plot_requested"):
        dbg = mesh_in.copy()
        # apply T
        for gid, tl in t_gtl.items():
            dbg.nodes[gid] = t_verts_m[tl]
        # apply U
        if u_gtl:
            for gid, ul in u_gtl.items():
                dbg.nodes[gid] = u_verts_m[ul]
        viewer.debug_plot_requested.emit(dbg, "[STEP2] After Deforming D = T∪U")

    
    b_c = morph_model.GetIndepBoundaries(mesh_in)  # {boundary_id: [gids]}
    bt = {k: np.array([0.0, 0.0, 0.0]) for k in b_c.keys()}

    
    # STEP 5 - Update mesh nodes
    logger.log("STEP 5 - Update the vertice positions in the mesh")

    # Vectorised T update
    if t_gtl:
        t_g = np.fromiter(t_gtl.keys(), dtype=np.int64)
        t_l = np.fromiter(t_gtl.values(), dtype=np.int64)
        t_arr = np.asarray(t_verts_m, dtype=mesh_out.nodes.dtype)
        mesh_out.nodes[t_g] = t_arr[t_l]

    # Vectorised U update
    if u_gtl and len(u_verts_m) > 0:
        u_g = np.fromiter(u_gtl.keys(), dtype=np.int64)
        u_l = np.fromiter(u_gtl.values(), dtype=np.int64)
        u_arr = np.asarray(u_verts_m, dtype=mesh_out.nodes.dtype)
        mesh_out.nodes[u_g] = u_arr[u_l]

    # Optional rigid boundary translation (kept safe)
    if getattr(morph_model, "rigid_boundary_translation", False):
        logger.log("[Morph] rigid_boundary_translation = True → applying rigid C translation")
        b_idx, b_xyz = morph_model.recover_boundaries_array(mesh_out, bt, rb=rb)
        if getattr(b_idx, "size", 0):
            mesh_out.nodes[b_idx] = b_xyz
    else:
        logger.log("[Morph] rigid_boundary_translation = False → leaving C nodes fixed")

    if debug and viewer is not None and hasattr(viewer, "debug_plot_requested"):
        viewer.debug_plot_requested.emit(mesh_out, "[STEP5] Final Morphed Mesh")

    logger.log("Morphing Function Complete")

    filename = f"{base_name}_{n}.fro"

    if train_model:
        compute_mesh_fail_features(mesh_in, mesh_out, morph_model=morph_model)

    # Save output (HPC / headless mode)
    if viewer is None:
        remote_dir = output_dir
        remote_subdir = os.path.join(remote_dir, "surfaces", f"n_{gen}")
        os.makedirs(remote_subdir, exist_ok=True)
        out_path = os.path.join(remote_subdir, filename)
        mesh_out.write_file(out_path)
        logger.log(f"[HPC] Morphed file saved to: {out_path}")

    return mesh_out

def MorphVolume(m_0, morph_model, debug=False):
    pass

def MorphCAD(cad_path, ffd_lattice, ffd_ctrl_disp, face_roles, out_dir, debug=False):
    """
    Placeholder CAD morph function.

    Intended future behaviour:
      - load CAD from cad_path,
      - use ffd_lattice + ffd_ctrl_disp and face_roles (T/U/C)
        to deform the model,
      - write morphed CAD into out_dir and return its path.

    For now this is a no-op: it just copies the original CAD into out_dir.
    """
    import os, shutil

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(cad_path)
    out_step = os.path.join(out_dir, base)
    shutil.copyfile(cad_path, out_step)

    print(f"[MorphCAD] WARNING: CAD morph not implemented; copied '{cad_path}' to '{out_step}'")
    return out_step


if __name__ == "__main__":
    print("hello?")
