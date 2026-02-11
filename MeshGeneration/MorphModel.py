from time import time
import os
import sys
import numpy as np

import os
import sys

from collections import deque
import numpy as np

sys.path.append(os.path.dirname("MeshGeneration"))
sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("Utilities"))
from MeshGeneration.BasisFunctions import get_bf
from MeshGeneration.Rbf import train_3d, predict_3d
from MeshGeneration.ShapeParameterSelection import *
from MeshGeneration.BasisFunctions import get_bf
from Utilities.PointClouds import *

class MorphModelBase:
    """Base class definition of all parameters required to define a morph"""
    def __init__(self, f_name=None, con=False, pb=None, path=None, f=None, T=None, U=None, C=None, cn=[], dispVector=[]):
        self.f_name      = f_name
        self.path        = path
        self.constraint  = con
        self.phi_bounds  = pb if pb is not None else []
        self.bf_t        = "wendland_c_2"
        self.c_t         = "radius"
        self.c_mod_t     = 1.0
        self.bf_u        = "wendland_c_0"
        self.c_u         = "one"
        self.c_mod_u     = 0.7
        self.rigid_boundary_translation  = False
        self._f          = f
        self.const_prop  = 0

        self.t_sections  = []
        self.t_node_ids  = {}
        self.t_face_ids  = {}
        self.t_surfaces  = T if T is not None else []

        self.u_sections  = []
        self.u_node_ids  = {}
        self.u_face_ids  = {}
        self.u_surfaces  = U if U is not None else []
        
        self.c_surfaces = []
        self.c_surfaces = C if C is not None else []
        
        self.control_nodes = [] if cn is None else cn
        self.displacement_vector = [] if dispVector is None else dispVector      
        
        self.alpha_support = 0.3  # fraction of T-surface size used as support radius (tune 0.1–0.5)
        self.rbf_lambda = 1e-10   
        
        
class MorphModel(MorphModelBase):
    def __init__(self, f_name=None, con=False, pb=[], path=None, T=[], U=[], cn=[], displacement_vector=[]):
        super().__init__(f_name, con, pb, path, T, U, cn)
        
        # Ensure proper initialization if parameters were passed
        if cn:
            self.control_nodes = cn
            print(f"  Set control_nodes from cn parameter: {self.control_nodes}")
        
        if displacement_vector:
            self.displacement_vector = displacement_vector
            print(f"  Set displacement_vector from parameter: {self.displacement_vector}")

    def set_control_data(self, control_nodes, displacement_vector):
        """Explicitly set control nodes and displacement vector"""
        self.control_nodes = control_nodes
        self.displacement_vector = displacement_vector
        print(f"[DEBUG] Control data set: {len(control_nodes)} nodes, {len(displacement_vector)} displacements")
        
    def validate_control_data(self):
        """Validate that control data is properly set"""
        if not self.control_nodes:
            print("[ERROR] control_nodes is empty or None")
            return False
        if not self.displacement_vector:
            print("[ERROR] displacement_vector is empty or None")
            return False
        if len(self.control_nodes) != len(self.displacement_vector):
            print(f"[ERROR] Mismatch: {len(self.control_nodes)} control nodes vs {len(self.displacement_vector)} displacements")
            return False
        print(f"[OK] Control data valid: {len(self.control_nodes)} control points")
        return True
    
    def _compute_C_components(self, ff, C_list):
        """
        Label connected components on the subgraph induced by C nodes.
        Returns:
        isC:   (N,) bool mask for C nodes
        comp:  (N,) int32, -1 for non-C, else component id
        comps: list[list[int]] nodes of each component
        """
        N = int(getattr(ff, "node_count", len(ff.nodes)))
        isC = np.zeros(N, dtype=bool)
        C = np.asarray(C_list, dtype=np.int64)
        if C.size:
            isC[C] = True

        comp = np.full(N, -1, dtype=np.int32)
        comps = []
        for start in C:
            if comp[start] != -1:
                continue
            q = deque([start])
            cid = len(comps)
            comp[start] = cid
            nodes = []
            while q:
                u = q.popleft()
                nodes.append(u)
                for v in ff.node_connections.get(u, ()):
                    if isC[v] and comp[v] == -1:
                        comp[v] = cid
                        q.append(v)
            comps.append(nodes)
        return isC, comp, comps

    def __str__(self):
        retval = ""
        retval += f"Function      : {self.f_name}\n"
        retval += f"Constraint    : {self.constraint}\n"
        retval += f"Bounds for Phi: {self.phi_bounds}\n"
        retval += f"T Basis Func  : {self.bf_t}\n"
        retval += f"T Shape Meth  : {self.c_t}\n"
        retval += f"T Shape Mod   : {self.c_mod_t}\n"
        retval += f"U Basis Func  : {self.bf_u}\n"
        retval += f"U Shape Meth  : {self.c_u}\n"
        retval += f"U Shape Mod   : {self.c_mod_u}\n"
        retval += f"T: Num Sections = {len(self.t_sections)}\n"
        retval += f"T: Num Nodes    = {len(self.t_node_ids)}\n"
        retval += f"T: Num Faces    = {len(self.t_face_ids)}\n"
        retval += f"T: Num Surfaces = {len(self.t_surfaces)}\n"
        retval += f"U: Num Sections = {len(self.u_sections)}\n"
        retval += f"U: Num Nodes    = {len(self.u_node_ids)}\n"
        retval += f"U: Num Faces    = {len(self.u_face_ids)}\n"
        retval += f"U: Num Surfaces = {len(self.u_surfaces)}\n"
        retval += f"CN: Control Nodes = {len(self.control_nodes)}\n"
        retval += f"DV: Displacement Vector = {len(self.displacement_vector)}\n"
        return retval
    
    def get_tc_anchor_gids(self, ff):
        t = set(self.get_t_node_gids(ff))
        c = set(self.get_c_node_gids(ff))
        anchors = sorted(t & c)
        return anchors
    
    def get_tc_anchor_gids(self, ff):
        t = set(self.get_t_node_gids(ff))
        c = set(self.get_c_node_gids(ff))
        anchors = sorted(t & c)
        return anchors

    def get_tc_boundary_only(self, ff):
        """Get only the nodes that are BOTH in T AND in C (the actual boundary)"""
        t_set = set(self.get_t_node_gids(ff))
        c_set = set(self.get_c_node_gids(ff))
        
        print("len(t_set) =", len(t_set))
        print("len(c_set) =", len(c_set))
        print("overlap T∩C =", len(t_set & c_set))
        
        # Only nodes that are in T AND have at least one C neighbor
        boundary = []
        for t_node in t_set:
            neighbors = ff.node_connections.get(t_node, [])
            if any(n in c_set for n in neighbors):
                boundary.append(t_node)
        
        return boundary
    
    def transformT(
        self,
        t_verts,
        anchor_points=None,
        tol=1e-9,
        # ---- control influence radius (PASS 1) ----
        min_R_frac=0.1,
        fallback_R_frac=0.85,
        R_scale=2.75,

        # ---- legacy taper (DISABLE for new method) ----
        anchor_taper=False,       # default OFF
        dc_ramp_frac=0.2,

        # ---- control mapping ----
        snap_tol_frac=1e-10,

        # ---- Adaptive seam safeguard (optional; preserves variance) ----
        seam_adapt=True,
        seam_k=10,
        seam_band_frac=0.06,
        seam_tau0=3.0,
        seam_tau1=8.0,
        seam_kmax=3.0,

        # ---- NEW: boundary correction (PASS 2) ----
        boundary_recover=True,
        corr_R_frac=0.03,
        corr_R_scale=1.0,
        corr_lambda=1e-6,         # bump default; 1e-8 is often too small for stability
        corr_band_frac=0.06,      # only used for weighting (application band), not constraints
        max_corr_sources=1200,
        corr_chunk=5000,
        seed=0,
    ):
        """
        Two-pass morph:

        PASS 1:
            U1(x) = sum_i phi(||x-ci||/Ri) * disp_i

        PASS 2 (seam correction):
            Constrain ONLY seam points (anchors mapped onto TU mesh):
                K(S,S) W = -U1(S)
            Apply correction locally with a distance-to-seam weight w(d):
                U(x) = U1(x) + w(d(x)) * Uc(x)

        Key: Do NOT include the whole seam band as hard constraints (causes ringing/blotches).
        """
        import numpy as np

        if not self.control_nodes or self.displacement_vector is None:
            return np.asarray(t_verts, float).tolist()

        T = np.asarray(t_verts, dtype=float)
        if T.size == 0:
            return []

        ctrl = np.asarray(self.control_nodes, dtype=float)
        disp = np.asarray(self.displacement_vector, dtype=float)
        if ctrl.ndim == 1:
            ctrl = ctrl.reshape(1, -1)
        if disp.ndim == 1:
            disp = disp.reshape(1, -1)

        if ctrl.shape[0] != disp.shape[0]:
            raise ValueError(f"Control-node count ({ctrl.shape[0]}) != displacement count ({disp.shape[0]})")

        # patch scale
        mins = T.min(axis=0)
        maxs = T.max(axis=0)
        L = float(np.linalg.norm(maxs - mins))
        L = max(L, 1e-12)

        # --- helpers ---
        def nearest_dist(points, query_points):
            """Return (dists, idx) from each query point to nearest point in points."""
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(points)
                d, idx = tree.query(query_points, k=1)
                return np.asarray(d, float), np.asarray(idx, int)
            except Exception:
                diff = query_points[:, None, :] - points[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                idx = np.argmin(d2, axis=1).astype(int)
                d = np.sqrt(d2[np.arange(d2.shape[0]), idx])
                return d.astype(float), idx

        # snap controls to actual mesh vertices
        dmin, idx0 = nearest_dist(T, ctrl)
        snap_tol = max(float(snap_tol_frac) * L, 1e-12)
        ctrl_snap = np.array(ctrl, copy=True)
        need_snap = dmin > snap_tol
        if np.any(need_snap):
            ctrl_snap[need_snap] = T[idx0[need_snap]]

        n_ctrl = int(ctrl_snap.shape[0])

        # Wendland C4 kernel
        def phi(s):
            s = np.asarray(s, dtype=float)
            out = np.zeros_like(s)
            m = s < 1.0
            sm = s[m]
            out[m] = (1.0 - sm) ** 6 * (35.0 * sm**2 + 18.0 * sm + 3.0)
            return out

        # ---------- PASS 1 ----------
        A = None
        if anchor_points is not None and len(anchor_points) > 0:
            A = np.asarray(anchor_points, dtype=float)
            if A.ndim == 1:
                A = A.reshape(1, -1)

            Ri, _ = nearest_dist(A, ctrl_snap)
            R_min = max(float(min_R_frac) * L, 1e-9)
            Ri = np.maximum(Ri, R_min) * float(R_scale)
        else:
            Ri = np.full((n_ctrl,), float(R_scale) * max(float(fallback_R_frac) * L, 1e-9), dtype=float)

        N = T.shape[0]
        U1 = np.zeros((N, 3), dtype=float)
        for i in range(n_ctrl):
            r = np.linalg.norm(T - ctrl_snap[i][None, :], axis=1)
            wi = phi(r / Ri[i])
            U1 += wi[:, None] * disp[i][None, :]

        # Precompute distance-to-anchor for metric if anchors exist
        db = None
        if A is not None:
            db, _ = nearest_dist(A, T)

        # ---------- Adaptive seam safeguard (adjust PASS-2 params only) ----------
        corr_R_frac_eff = float(corr_R_frac)
        corr_band_frac_eff = float(corr_band_frac)
        corr_R_scale_eff = float(corr_R_scale)

        if seam_adapt and boundary_recover and (A is not None) and (db is not None):
            band = db < max(float(seam_band_frac) * L, 1e-12)
            if np.count_nonzero(band) > max(50, int(seam_k) + 2):
                ridge_ratio = 0.0
                try:
                    from scipy.spatial import cKDTree
                    treeT_all = cKDTree(T)
                    dnn, inn = treeT_all.query(T[band], k=int(seam_k) + 1)
                    dnn = np.asarray(dnn, float)[:, 1:]
                    inn = np.asarray(inn, int)[:, 1:]
                    U_band = U1[band]
                    eps = 1e-12
                    dU = np.linalg.norm(U_band[:, None, :] - U1[inn], axis=2)
                    g_local = np.max(dU / (dnn + eps), axis=1)
                    g = float(np.percentile(g_local, 95.0))
                    Umag = np.linalg.norm(U1, axis=1)
                    u_ref = float(np.percentile(Umag, 95.0))
                    g_ref = (u_ref / L) + 1e-12
                    ridge_ratio = g / g_ref
                except Exception:
                    ridge_ratio = 0.0

                if ridge_ratio > float(seam_tau0):
                    s = (ridge_ratio - float(seam_tau0)) / (float(seam_tau1) - float(seam_tau0) + 1e-12)
                    s = float(np.clip(s, 0.0, 1.0))
                    mult = 1.0 + s * (float(seam_kmax) - 1.0)
                    corr_band_frac_eff = float(corr_band_frac) * mult
                    corr_R_frac_eff = float(corr_R_frac) * (1.0 + 0.35 * (mult - 1.0))
                    corr_R_scale_eff = float(corr_R_scale)

        # ---------- Legacy anchor taper (ONLY if PASS-2 disabled) ----------
        if anchor_taper and (not boundary_recover) and (A is not None) and (db is not None):
            ramp = max(float(dc_ramp_frac) * L, 1e-12)
            t = np.clip(db / ramp, 0.0, 1.0)
            wA = t**3 * (t * (t * 6.0 - 15.0) + 10.0)
            wA[db <= tol] = 0.0
            U1 *= wA[:, None]

        # ---------- PASS 2: seam correction (seam-only constraints; band-only weighting) ----------
        if boundary_recover and (A is not None):
            # Map anchors to T indices (seam indices)
            dB, idxB = nearest_dist(T, A)  # query=A onto points=T
            B_tol = max(1e-10 * L, 1e-12)
            keep = dB <= B_tol
            idxB = np.asarray(idxB[keep], int)

            if idxB.size > 0:
                idxB = np.unique(idxB)

                # Use ONLY seam indices as constraints
                S = T[idxB]
                F = -U1[idxB]

                # Thin if too many (works on seam only)
                if S.shape[0] > int(max_corr_sources):
                    rng = np.random.default_rng(seed)
                    Ns = S.shape[0]
                    pick = rng.choice(Ns, size=int(max_corr_sources), replace=False)
                    S = S[pick]
                    F = F[pick]

                Rc = max(float(corr_R_frac_eff) * L, 1e-9) * float(corr_R_scale_eff)
                Rband = max(float(corr_band_frac_eff) * L, 1e-12)

                # Build K and solve (regularised)
                dSS = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2)
                K = phi(dSS / Rc)
                K.flat[:: K.shape[0] + 1] += float(corr_lambda)

                W = np.linalg.solve(K, F)

                # Weighting band uses distance to seam (S)
                d_to_seam, _ = nearest_dist(S, T)
                t = np.clip(d_to_seam / Rband, 0.0, 1.0)
                w = 1.0 - (t**3 * (t * (t * 6.0 - 15.0) + 10.0))

                # Evaluate correction field
                Uc = np.zeros_like(U1)
                for i0 in range(0, N, int(corr_chunk)):
                    i1 = min(N, i0 + int(corr_chunk))
                    Tc = T[i0:i1]
                    dTS = np.linalg.norm(Tc[:, None, :] - S[None, :, :], axis=2)
                    Kts = phi(dTS / Rc)
                    Uc[i0:i1] = Kts @ W

                U = U1 + (w[:, None] * Uc)
                return (T + U).tolist()

        return (T + U1).tolist()


    def get_nodes(self, sections, surfaces, faces, nodes, ff):
        g_ids = []

        # Skip sections; only use surfaces directly
        for s in surfaces:
            try:
                _, ids = ff.get_surface_nodes(int(s))
                g_ids.extend(ids)
            except Exception as e:
                print(f"[WARN] Could not get nodes for surface {s}: {e}")

        for s, fs in faces.items():
            try:
                _, f_ids = ff.get_surface_tria3(int(s))
                for f in fs:
                    if f < len(f_ids):
                        g_ids.extend(ff.get_tria3_nodes(f_ids[f]))
            except:
                continue

        for s, ns in nodes.items():
            try:
                _, s_ids = ff.get_surface_nodes(int(s))
                for n in ns:
                    if n < len(s_ids):
                        g_ids.append(s_ids[n])
            except:
                continue
        
        return list(set(g_ids))

    def get_t_node_gids(self, ff):
        return self.get_nodes(self.t_sections, self.t_surfaces, self.t_face_ids, self.t_node_ids, ff)
    
    def get_u_node_gids(self, ff):
        u = self.get_nodes(self.u_sections, self.u_surfaces, self.u_face_ids, self.u_node_ids, ff)
        t = self.get_t_node_gids(ff)
        U = [x for x in u if x not in t]
        return U

    def get_c_node_gids(self, ff):
        # Prefer explicit user-selected C surfaces if provided
        if hasattr(self, "c_surfaces") and self.c_surfaces:
            C = []
            for s in self.c_surfaces:
                _, ids = ff.get_surface_nodes(int(s))
                C.extend(ids)
            return list(set(C))

        # Fallback: complement of T ∪ U, excluding FARFIELD (legacy behaviour)
        t = set(self.get_t_node_gids(ff))
        u = set(self.get_u_node_gids(ff))
        c = set(range(ff.node_count)) - (t | u)
        f = []
        for s in getattr(ff, "farfield_ids", []):
            _, ids = ff.get_surface_nodes(int(s))
            f.extend(ids)
        return list(c - set(f))

    def get_t_node_vertices(self,ff):
        return ff.convert_node_ids_to_coordinates(self.get_t_node_gids(ff))

    def get_u_node_vertices(self,ff):
        return ff.convert_node_ids_to_coordinates(self.get_u_node_gids(ff))

    def get_c_node_vertices(self,ff):
        return ff.convert_node_ids_to_coordinates(self.get_c_node_gids(ff))
    
    def GetIndepBoundaries(self, ff, m=None, e=None):
        # Defaults
        if m == None:
            t = self.get_t_node_gids(ff)
            u = self.get_u_node_gids(ff)
            m = list(set(t) | set(u))
        if e == None:
            c = self.get_c_node_gids(ff)
            e = c
        m = [int(i) for i in m]
        e = set(int(i) for i in e)
        # find all boundary nodes (unordered)
        b = []
        for g_id in m:
            connected_nodes = ff.node_connections.get(g_id, [])
            for cn in connected_nodes:
                if cn in e:
                    b.append(g_id)
                    break
        indep_boundary = {}
        boundary_id  = 0
        for g_id in b:
            # loop through each node in the boundary
            connected_nodes = ff.node_connections[g_id]
            in_boundarys = []
            for cn in connected_nodes:
                for k, v in indep_boundary.items():
                    # if one of the nodes its connected to is already in the boundary, add
                    # it to that chain.
                    if (cn in v) and (g_id not in indep_boundary[k]):
                        indep_boundary[k].append(g_id)
                        in_boundarys.append(k)
            
            if len(in_boundarys) == 0:
                # if g_id is not in any boundarys, then we need to create a new chain for it. 
                indep_boundary[boundary_id] = [g_id]
                boundary_id += 1
            elif len(in_boundarys) > 1:
                # if it is attatched to multiple chains then these chains are actually linked. 
                # amalgamate them all together. 
                # add the new boundary
                new_bound = set([])
                for k in in_boundarys:
                    new_bound = new_bound | set(indep_boundary[k])
                indep_boundary[boundary_id] = list(new_bound)
                boundary_id += 1
                # remove the old boundarys
                for k in in_boundarys:
                    del indep_boundary[k]
                new_indep_boundary = {}
                # renumber boundarys in the list
                for inew, iold in enumerate(list(indep_boundary.keys())):
                    new_indep_boundary[inew] = indep_boundary[iold]
                indep_boundary.clear()
                for k,v in new_indep_boundary.items():
                    indep_boundary[k] = v
                new_indep_boundary.clear()
        
        return indep_boundary


    def GetTIndepBoundaries(self, ff):
        t = self.get_t_node_gids(ff)
        u = self.get_u_node_gids(ff)
        c = self.get_c_node_gids(ff)
        return self.GetIndepBoundaries(ff, m=t, e=list(set(c) | set(u)))

    def get_indep_boundaries(self, ff):
        pass

    def GetRigidBodies_original(self, ff, bts):
        """
        For each moved boundary b, return the list of C nodes rigidly translated with it.
        Uses one-time component labelling on C; does NOT modify GetIndepBoundaries.
        """
        bounds = self.GetIndepBoundaries(ff)       # {b_id: [seed nodes]}
        C_list = self.get_c_node_gids(ff)          # list of C node ids

        isC, comp, comps = self._compute_C_components(ff, C_list)

        rigid_bodies = {}
        for b, seeds in bounds.items():
            bt = bts.get(b, (0.0, 0.0, 0.0))
            # skip if zero translation or no seeds
            if not seeds or (bt[0] == 0.0 and bt[1] == 0.0 and bt[2] == 0.0):
                continue

            # Which C-components touch this boundary? (C neighbors of seeds)
            touched = set()
            for s in seeds:
                for nb in ff.node_connections.get(s, ()):
                    if isC[nb]:
                        cid = comp[nb]
                        if cid >= 0:
                            touched.add(cid)

            # Union all touched C-components
            if touched:
                body = []
                for cid in touched:
                    body.extend(comps[cid])   # list of ints
                rigid_bodies[b] = body
            else:
                rigid_bodies[b] = []          # no C connected; keep semantics
        return rigid_bodies


    def GetRigidBodies_chains(self, ff, bts):
        rigid_bodies = {}
        boundary_id  = 0
        start_time = time()
        ccount = 0
        C = self.get_c_node_gids(ff)
        to_explore = set(C)
        bounds = self.GetIndepBoundaries(ff)
        for k,v in bounds.items():
            to_explore = to_explore | set(v)
        maxrate = 0
        for g_id in to_explore:
            ccount += 1
            # loop through each node in the boundary
            connected_nodes = ff.node_connections[g_id]
            in_boundarys = set([])
            for cn in connected_nodes:
                for k, v in rigid_bodies.items():
                    # if one of the nodes its connected to is already in the boundary, add
                    # it to that chain.
                    if cn in v:
                        #if g_id not in v:
                        v.add(g_id)
                        in_boundarys.add(k)
            #rem = len(C) - ccount
            #rate = ccount/max(1,(time()-start_time))
            #maxrate = max(rate,maxrate)
            #eta_s  = rem/max(rate,1)
            #print(f"Rate: {rate} checked/s (eta - {eta_s}s) - max rate = {maxrate}")
            if len(in_boundarys) == 0:
                # if g_id is not in any boundarys, then we need to create a new chain for it. 
                rigid_bodies[boundary_id] = set([g_id])
                boundary_id += 1
            elif len(in_boundarys) > 1:
                # if it is attatched to multiple chains then these chains are actually linked. 
                # amalgamate them all together. 
                # add the new boundary
                new_bound = set([])
                for k in in_boundarys:
                    new_bound = new_bound | set(rigid_bodies[k])
                rigid_bodies[boundary_id] = set(new_bound)
                boundary_id += 1
                # remove the old boundarys
                for k in in_boundarys:
                    del rigid_bodies[k]
                new_indep_boundary = {}
                # renumber boundarys in the list
                for inew, iold in enumerate(list(rigid_bodies.keys())):
                    new_indep_boundary[inew] = rigid_bodies[iold]
                rigid_bodies.clear()
                for k,v in new_indep_boundary.items():
                    rigid_bodies[k] = set(v)
                new_indep_boundary.clear()
        end_time = time()
        print(f"get rigid bodies finished in {end_time-start_time} s")
        # need to correct boundary ids
        rigid_bodies_corrected = {}
        for key, bound in self.GetIndepBoundaries(ff).items():
            node = bound[0]
            for rb in rigid_bodies.values():
                if node in rb:
                    rigid_bodies_corrected[key] = list(rb)
                    break
        return rigid_bodies_corrected

    def recover_boundaries_array(self, ff, bts, rb="original", exclude_T=True, dedup="last"):
        """
        Returns:
            b_idx  : (M,) int64  global node IDs to update
            b_xyz  : (M,3) float new coordinates for those nodes
        """
        import numpy as np
        # Pick rigid-body grouping method
        rigid_body_method = self.GetRigidBodies_original if rb == "original" else self.GetRigidBodies_chains
        rb_dict = rigid_body_method(ff, bts)  # {key: iterable of global ids}
        
        N = int(getattr(ff, "node_count", len(ff.nodes)))
        nodes = ff.nodes  # (N,3) array

        # Build a fast T mask once
        T = np.asarray(list(self.get_t_node_gids(ff)), dtype=np.int64)
        Tmask = np.zeros(N, dtype=bool)
        if T.size:
            Tmask[T] = True

        idx_chunks = []
        delta_chunks = []

        for key, boundary in rb_dict.items():
            # boundary ids -> numpy
            g = np.asarray(boundary, dtype=np.int64)
            if g.size == 0:
                continue

            # drop T nodes if requested
            if exclude_T:
                g = g[~Tmask[g]]
                if g.size == 0:
                    continue

            # broadcast this boundary's rigid-body translation
            bt = np.asarray(bts[key], dtype=nodes.dtype).reshape(1, 3)
            idx_chunks.append(g)
            # repeat bt for every node in this boundary
            delta_chunks.append(np.repeat(bt, g.size, axis=0))
        
        if not idx_chunks:
            return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=nodes.dtype)

        b_idx = np.concatenate(idx_chunks)
        b_dlt = np.vstack(delta_chunks)
        b_xyz = nodes[b_idx] + b_dlt  # vectorised

        # Optional: deduplicate if nodes appear in multiple groups
        if b_idx.size and dedup:
            if dedup == "last":
                # keep the last occurrence per node id
                rev_idx = b_idx[::-1]
                keep_rev = np.zeros_like(rev_idx, dtype=bool)
                _, first_pos = np.unique(rev_idx, return_index=True)
                keep_rev[first_pos] = True
                keep = keep_rev[::-1]
                b_idx = b_idx[keep]
                b_xyz = b_xyz[keep]
            elif dedup == "first":
                _, keep = np.unique(b_idx, return_index=True)
                b_idx = b_idx[keep]
                b_xyz = b_xyz[keep]
            # else: leave duplicates as-is
        
        return b_idx, b_xyz

    def recover_boundaries(self, ff, bts, rb="original"):
        b_idx, b_xyz = self.recover_boundaries_array(ff, bts, rb=rb)
        # Dict creation is slower; only do this if callers truly require a dict
        return dict(zip(b_idx.tolist(), b_xyz.tolist()))

class MorphModelCAD(MorphModelBase):
    """
    CAD-oriented morph model.

    Same basic parameters as MorphModel but without any FroFile-specific
    helpers. This will be used alongside a CAD-FFD layer that knows how
    to interpret the control-node displacements on the CAD surfaces.
    """
    def __init__(self, f_name=None, con=False, pb=None, path=None,
                 T=None, U=None, cn=None, displacement_vector=None):
        super().__init__(
            f_name=f_name,
            con=con,
            pb=pb if pb is not None else [],
            path=path,
            T=T if T is not None else [],
            U=U if U is not None else [],
            cn=cn if cn is not None else [],
            dispVector=displacement_vector if displacement_vector is not None else [],
        )
        self.face_roles = {}   # e.g. {face_id: FaceRole.T/C/U}
        self.logger = None

    def set_control_data(self, control_nodes, displacement_vector):
        self.control_nodes = control_nodes or []
        self.displacement_vector = displacement_vector or []

    def validate_control_data(self):
        if not self.control_nodes or not self.displacement_vector:
            return False
        return len(self.control_nodes) == len(self.displacement_vector)
