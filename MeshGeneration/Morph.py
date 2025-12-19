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

def MorphMesh(m_0:FroFile, base_name, morph_model, viewer, output_dir, debug=True, rb="original", n=0, gen = 0):
    """
        m_0   : FroFile 
        m     : MorphModel
        debug : bool
    """
    logger = viewer.logger if viewer is not None else DummyLogger()
    m_m = m_0.copy()
    
    
    ## Step 3
    step_3_bf = get_bf(morph_model.bf_t)
    step_3_c  = lambda cents, kvs, bfi: morph_model.c_mod_t * get_shape(morph_model.c_t)(cents, kvs, bfi)
    ## Step 4
    step_4_bf = get_bf(morph_model.bf_u)
    step_4_c  = lambda cents, kvs, bfi: morph_model.c_mod_u * get_shape(morph_model.c_u)(cents, kvs, bfi)

    # Step 1 - Define target nodes/unconstrained nodes
    # Step 2 - Perform transformation f on t - f(t)
    # Step 3 - recover all boundarys (not with u).
    # Step 4 - if u - propogate into u, fixing boundarys and limits? 
    # Step 5 - update m_0 to m_m

    logger.log(f"[CHECK] T surfaces: {morph_model.t_surfaces}")
    logger.log(f"[CHECK] U surfaces: {morph_model.u_surfaces}")
    logger.log(f"[CHECK] count(T_gids)={len(morph_model.get_t_node_gids(m_0))}, "
            f"count(U_gids)={len(morph_model.get_u_node_gids(m_0))}")

    # Step 2 - f(t)
    logger.log(f"STEP 2 - Translating ")
    t_gtl, t_verts = morph_model.get_t_node_vertices(m_0)
    b_ct = morph_model.GetTIndepBoundaries(m_0)  # {boundary_id: [gids]}

    # Get ONLY the T nodes that actually touch C (the real boundary)
    anchor_gids = morph_model.get_tc_boundary_only(m_0)
    anchor_points = [m_0.nodes[g] for g in anchor_gids]
    logger.log(f"[STEP2] anchors (T∩C boundary) = {len(anchor_gids)} / |T|={len(t_gtl)}")

    # Debug: check if we have proper anchors
    if len(anchor_gids) == 0:
        logger.log(f"[WARNING] No T-C boundary nodes found! All T nodes will be free to move.")
    elif len(anchor_gids) == len(t_gtl):
        logger.log(f"[WARNING] All T nodes are on boundary! No interior nodes to displace.")
        
    t_set = set(morph_model.get_t_node_gids(m_0))
    c_set = set(morph_model.get_c_node_gids(m_0))
    anchor_gids = [g for g in t_set 
                if any(n in c_set for n in m_0.node_connections.get(g, []))]
    anchor_points = [m_0.nodes[g] for g in anchor_gids]
    logger.log(f"[STEP2] anchors (T boundary) = {len(anchor_gids)} / |T|={len(t_gtl)}")

    t_verts_t = morph_model.transformT(t_verts, anchor_points=anchor_points)
    cp_before = np.array(center_point_of_vertices(t_verts))
    cp_after  = np.array(center_point_of_vertices(t_verts_t))
    cp_diff   = cp_after-cp_before
    logger.log(f"CP Before: {cp_before}")
    logger.log(f"CP After : {cp_after}")
    logger.log(f"Diff     : {cp_diff}")

    if debug and hasattr(viewer, "debug_plot_requested"):
        m_m = m_0.copy()
        for g_id, lc_id in t_gtl.items():
            m_m.nodes[g_id] = t_verts_t[lc_id]
        viewer.debug_plot_requested.emit(m_m, "[STEP2] After RBF training")
        
   # Step 3 - recover all boundarys (not with u).
    t = set(morph_model.get_t_node_gids(m_0))
    u = set(morph_model.get_u_node_gids(m_0))
    c = set(morph_model.get_c_node_gids(m_0))
    touches = sum(any(n in c for n in m_0.node_connections.get(i, [])) for i in (t|u))
    logger.log(f"[CHECK] |T|={len(t)} |U|={len(u)} |C|={len(c)} ; T∪U touching C = {touches}")

    logger.log(f"STEP 3 - Re-Morphing  to recover boundaries")
    b_c  = morph_model.GetIndepBoundaries(m_0)
    logger.log(f"STEP 3 - b_c keys = {list(b_c.keys())}")
    fixed_boundaries_points = [] # VIS
    b_c_verts_t = [] # VIS
    
    S = []
    F = []
    bt = {}

    c_0_s = [] # vis
    c_t_s = [] # vis

    # work out corrected boundarys
    logger.log(f"Boundarys between t,u and c:")
    for x in b_c.keys():
        logger.log(f"Boundary: {x}")
        fixed_boundaries_points = fixed_boundaries_points + b_c[x] # store a list of all boundary nodes that are recovered for visualisation
        v     = []
        v_t   = []
        v_uc  = []
        # loop through all nodes in the boundary and store their original/translated positions in lists
        for g_id in b_c[x]:
            # check if this point is a part of the target surfaces. b_c includes boundaries to be considered between T&U with C
            # here we are only interested in recovering the nodes in T as these are the only ones that have been moved so far
            if g_id in t_gtl:
                v.append(t_verts[t_gtl[g_id]])
                v_t.append(t_verts_t[t_gtl[g_id]])
            else:
                v_uc.append(g_id)
        logger.log(f"len of v = {len(v)} len of not v (uc) = {len(v_uc)}")
        # check that lists have been populated. If they have not then this boundary is a boundary between U and C NOT T
        if len(v) > 0 and len(v_t) > 0:
            v_t_f = []
            for g_id in b_c[x]:
                if g_id in t_gtl:
                    node_loc_tltd = np.array(t_verts_t[t_gtl[g_id]]) # s = source node from translated nodes 
                    node_loc      = np.array(t_verts[t_gtl[g_id]]) # source node from untranslated nodes
                    # f = original_position - translated_position + boundary translation
                    # corrects the boundary to its original shape and shifts the shape to 
                    # its correct position relative to the surrounding geometrys new shape
                    if len(v_uc) > 0:
                        b_t = cp_diff
                    else:
                        c_0 = center_point_of_vertices(v)
                        c_t = center_point_of_vertices(v_t)
                        c_0_s.append(c_0) # VIS
                        c_t_s.append(c_t) # VIS
                        b_t = np.array([a - b for a,b in zip(c_t, c_0)])
                    #log.debg(f"New center point calculation: {b_t}")
                    bt[x] = b_t 
                    fk = [b-a+c for a,b,c in zip(node_loc_tltd, node_loc, b_t)] # f = known translation for point
                    v_t_f.append([a+b for a,b in zip(node_loc_tltd, fk)]) # vtf = known final position of point (ie the source node + displacement) VIS only
                    S = S + [node_loc_tltd] # S = list of all sources
                    F = F + [fk] # F = iist of all known translations 
            b_c_verts_t = b_c_verts_t + v_t_f  # b c verts t = list of all  VIS only
        else:
            logger.log(f"{x} Doesnt need correcting")
            bt[x] = np.array([0.0, 0.0, 0.0])
    
    # perform interpolation to recover boundary
    if len(F) > 0:
        logger.log(f"Morph to fix the boundarys based on known translations F: {len(F)}")
        # morph based on known translations
        counter = 0
        bfs= []
        cs = []
        dims = ["x","y","z"]
        for fk in np.array(F).T:
            #bfhp, chp, condshp, _ = rbf_hyperparameter_optimise_condition(S, f)
            bf = step_3_bf
            c  = step_3_c(S, fk, bf)
            bfs.append(bf)
            cs.append(c)
            #log.info(f"RBF: By using hyperparameter optimisation for {dims[counter]}:")
            #log.debg(f"BF: {bfhp.__name__}, C: {chp}, RMSE: {rmsehp}, Spear: {spearhp}")
            #log.debg(f"matrix condition vals = {condshp}")
            counter += 1
        #bf = BasisFunctions.wendland_c_0
        #cs  = [4.5*step_3_c(S, f, bf) for f in np.array(F).T]
        logger.log(f"Basis Fn for RBF are - {[x.__name__ for x in bfs]}")
        logger.log(f"C Values for RBF are - {cs}")
        ls, bs, conds = train_3d(S, F, bfs, cs)
        ds            = predict_3d(ls, bs, S, t_verts_t, bfs, cs)
        t_verts_m = [(np.array(a)+np.array(b)).tolist() for a,b in zip(t_verts_t, ds)]
        for ii in range(3):
            eval_func   = lambda x: predict(ls[ii], bs[ii], S, [x], bf=bfs[ii], c=cs[ii])
            avg_error_abssp, avg_error_pcsp, sd_error_pcsp, spearsp, kendalsp, rmsesp = rbf_stats(S, np.array(F).T[ii], eval_func)
            logger.log(f"RBF: Using manually selected parameters for {dims[ii]}:")
            logger.log(f"BF: {bfs[ii].__name__}, C: {cs[ii]}, RMSE: {rmsesp}, Spear: {spearsp}")
            logger.log(f"matrix condition vals = {conds[ii]}")
    else:
        t_verts_m = t_verts_t
    
    
    if debug and hasattr(viewer, "debug_plot_requested"):
        ff_copy = m_0.copy()
        for g_id, lc_id in t_gtl.items():
            ff_copy.nodes[g_id] = t_verts_m[lc_id]
        viewer.debug_plot_requested.emit(ff_copy, "[STEP3] After Recovering Boundary Nodes")
    
    # Step 4 - if u - propogate into u, fixing boundarys and limits?
    u_gtl = {}          # ensure defined even if there are no U nodes
    u_verts_m = []
    logger.log(f"STEP 4 - Propagate into u.")
    if len(morph_model.get_u_node_gids(m_0)) > 0:
        u_gtl, u_verts = morph_model.get_u_node_vertices(m_0)
        u_verts = np.asarray(u_verts, float)
        u_verts_m = u_verts.copy()

        # --- Build training set from T displacement field ---
        t_verts_np   = np.asarray(t_verts, float)
        t_verts_m_np = np.asarray(t_verts_m, float)
        T_disp = (t_verts_m_np - t_verts_np)   # (Nt,3)

        Nt = len(t_verts_np)
        if Nt == 0:
            logger.log("[STEP4][WARN] No T vertices available; skipping U propagation.")
        else:
            # If T is large, thin it to keep training stable/fast (FPS gives good coverage)
            n_keep = int(min(2500, Nt))   # 1000–3000 is a good general range
            if Nt > n_keep:
                idx = farthest_point_sampling(t_verts_np, n_keep=n_keep, seed=0)
                S = t_verts_np[idx]
                F = T_disp[idx]
            else:
                S = t_verts_np
                F = T_disp

            # Deduplicate to avoid tiny-distance issues
            S, F = dedup_sf(S, F, tol=1e-6)

            # --- Scale in a geometry-robust way (use combined T+U region) ---
            SU = np.vstack([S, u_verts])
            mins = SU.min(axis=0)
            maxs = SU.max(axis=0)
            L = float(np.linalg.norm(maxs - mins))
            L = max(L, 1e-9)
            center = SU.mean(axis=0)

            S_s = (S - center) / L
            U_s = (u_verts - center) / L

            # --- Choose a NON-tiny support radius (this is the big fix) ---
            # For compact support bf, c ≈ 1/R where R is support radius in *scaled* space.
            # R=0.35 means a point influences ~35% of region size → smooth propagation.
            R = float(getattr(morph_model, "u_support_radius", 0.35))   # tune 0.25–0.60
            R = np.clip(R, 0.10, 1.50)
            c_val = 1.0 / max(R, 1e-6)
            c_val *= float(getattr(morph_model, "c_mod_u", 1.0))

            step_4_bf = get_bf(morph_model.bf_u)
            bfs = [step_4_bf] * 3
            cs  = [c_val]    * 3

            # Light regularisation helps avoid ripples / overshoot
            reg = float(getattr(morph_model, "rbf_lambda_u", 1e-8))

            logger.log(f"[STEP4] Training sources: {len(S_s)} (from T)")
            logger.log(f"[STEP4] BF={step_4_bf.__name__}, R(scaled)={R:.3f}, c={c_val:.3e}, lambda={reg:.1e}")

            # Train/predict
            try:
                ls, bs, conds = train_3d(S_s.tolist(), F.tolist(), bfs, cs, reg_lambda=reg)
            except TypeError:
                ls, bs, conds = train_3d(S_s.tolist(), F.tolist(), bfs, cs)

            dU = np.asarray(predict_3d(ls, bs, S_s.tolist(), U_s.tolist(), bfs, cs), float)

            # Optional: smooth the predicted displacement on U to remove any residual “crease”
            smooth_iters = int(getattr(morph_model, "u_smooth_iters", 2))
            smooth_k     = int(getattr(morph_model, "u_smooth_k", 12))
            smooth_lam   = float(getattr(morph_model, "u_smooth_strength", 0.20))
            if smooth_iters > 0 and smooth_lam > 0:
                try:
                    from scipy.spatial import cKDTree
                    kdt = cKDTree(u_verts)
                    _, nn = kdt.query(u_verts, k=min(smooth_k + 1, len(u_verts)))
                    nn = nn[:, 1:]
                    lam = np.clip(smooth_lam, 0.0, 1.0)
                    for _ in range(smooth_iters):
                        dU_mean = dU[nn].mean(axis=1)
                        dU = (1.0 - lam) * dU + lam * dU_mean
                except Exception:
                    pass

            u_verts_m = (u_verts + dU)

            # Convert back to list-of-lists expected downstream
            u_verts_m = u_verts_m.tolist()


    if debug and hasattr(viewer, "debug_plot_requested"):
        ff_copy = m_0.copy()
        for g_id, lc_id in t_gtl.items():
            ff_copy.nodes[g_id] = t_verts_m[lc_id]
        for g_id, lc_id in u_gtl.items():
            ff_copy.nodes[g_id] = u_verts_m[lc_id]
        viewer.debug_plot_requested.emit(ff_copy, "[STEP4] After Propagating to U Nodes")
        
    # Step 5 - update m_0 to m_m
    logger.log(f"STEP 5 - Update the vertice positions in the mesh")

    # Vectorised T update
    if t_gtl:
        t_g = np.fromiter(t_gtl.keys(), dtype=np.int64)
        t_l = np.fromiter(t_gtl.values(), dtype=np.int64)
        t_arr = np.asarray(t_verts_m, dtype=m_m.nodes.dtype)
        m_m.nodes[t_g] = t_arr[t_l]

    # Vectorised U update
    if u_gtl and len(u_verts_m) > 0:
        u_g = np.fromiter(u_gtl.keys(), dtype=np.int64)
        u_l = np.fromiter(u_gtl.values(), dtype=np.int64)
        u_arr = np.asarray(u_verts_m, dtype=m_m.nodes.dtype)
        m_m.nodes[u_g] = u_arr[u_l]

    
    print("Check 1")
    if getattr(morph_model, "rigid_boundary_translation", False):
        # Recover boundary motion: rigidly move C components with their attached boundaries
        logger.log("[Morph] rigid_boundary_translation = True → applying rigid C translation")
        b_idx, b_xyz = morph_model.recover_boundaries_array(m_m, bt, rb=rb)
        print("Check 2")
        if b_idx.size:
            m_m.nodes[b_idx] = b_xyz
    else:
        logger.log("[Morph] rigid_boundary_translation = False → leaving C nodes fixed")
        print("Check 2 (skipped rigid C translation)")

        
    
    if debug and hasattr(viewer, "debug_plot_requested"):
        viewer.debug_plot_requested.emit(m_m, "[STEP5] Final Morphed Mesh")
        
    logger.log("Morphing Function Complete")
    
    filename = f"{base_name}_{n}.fro"
    
    if viewer is None:
        remote_dir = output_dir  # now guaranteed to be correct, even if viewer is None
        remote_subdir = os.path.join(remote_dir, "surfaces", f"n_{gen}")
        os.makedirs(remote_subdir, exist_ok=True)

        m_m.write_file(os.path.join(remote_subdir, filename))
        logger.log(f"[HPC] Morphed file saved to: {os.path.join(remote_subdir, filename)}")

    return m_m

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
