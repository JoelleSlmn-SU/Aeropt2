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
        self.bf_t        = "wendland_c_0"
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
        
        self.alpha_support = 0.25  # fraction of T-surface size used as support radius (tune 0.1–0.5)
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
        # Anchor “strength” controls (robust across geometries)
        anchor_subsample=12,     # keep every Nth anchor (3 is a good default)
        anchor_soft=0,       # 0 = hard zero anchors, 0.1 = allow 10% motion at anchors
        anchor_strength=1,      # DO NOT duplicate anchors by default (over-damps easily)
        # Optional displacement smoothing (helps residual ridges without clamping)
        smooth_iters=0,         # 0 disables
        smooth_k=12,
        smooth_strength=0.20,   # 0..1
    ):
        """
        Constrained/soft-anchored RBF morph for T surface.

        Key idea:
        - Control nodes impose the desired displacement
        - Anchor points impose a *soft* (not necessarily zero) displacement, so they don't over-damp
        - No post-mask ramp multiplication (avoids “ramp ridge” artefacts)

        Defaults chosen to work well with 1–O(10) control nodes and O(100–1000) anchors.

        Parameters:
        anchor_subsample : keep every Nth anchor (reduces over-constraint)
        anchor_soft      : fraction of mean control displacement used at anchors (0..1)
        anchor_strength  : repeats anchor constraints (weighting). Keep at 1 unless you *need* stiffer seam.
        """
        import numpy as np

        if not self.control_nodes or self.displacement_vector is None:
            return np.asarray(t_verts, float).tolist()

        # ----- control sources -----
        source = np.asarray(self.control_nodes, dtype=float)
        disp   = np.asarray(self.displacement_vector, dtype=float)

        if source.ndim == 1:
            source = source.reshape(1, -1)
        if disp.ndim == 1:
            disp = disp.reshape(1, -1)

        if source.shape[0] != disp.shape[0]:
            raise ValueError(f"Control-node count ({source.shape[0]}) != displacement count ({disp.shape[0]})")

        T_raw = np.asarray(t_verts, float)
        if T_raw.size == 0:
            return []

        # ----- anchor constraints (soft displacement) -----
        source_aug = source
        disp_aug   = disp

        if anchor_points is not None and len(anchor_points) > 0:
            B = np.asarray(anchor_points, float)
            if B.ndim == 1:
                B = B.reshape(1, -1)

            # Subsample anchors to avoid over-constraining (VERY important when few control nodes)
            if anchor_subsample and anchor_subsample > 1:
                B = B[::int(anchor_subsample), :]

            # Remove anchors that coincide with a control node to avoid contradictory constraints
            try:
                from scipy.spatial import cKDTree
                treeS = cKDTree(source)
                dmin, _ = treeS.query(B, k=1)
                B = B[dmin > (10 * tol)]
            except Exception:
                pass

            if len(B) > 0:
                # Soft anchor displacement = anchor_soft * mean(control displacement)
                # (keeps seam “mostly fixed” but prevents global over-damping)
                anchor_soft = float(np.clip(anchor_soft, 0.0, 1.0))
                disp_mean = np.mean(disp, axis=0, keepdims=True)   # (1,3)
                Z = np.repeat(disp_mean, len(B), axis=0) * anchor_soft

                # Optional extra weighting via duplication (keep = 1 by default)
                if anchor_strength and anchor_strength > 1:
                    rep = int(anchor_strength)
                    B = np.repeat(B, rep, axis=0)
                    Z = np.repeat(Z, rep, axis=0)

                source_aug = np.vstack([source, B])
                disp_aug   = np.vstack([disp,   Z])

        # ---------------- Surface-dependent scaling ----------------
        mins = T_raw.min(axis=0)
        maxs = T_raw.max(axis=0)
        L = float(np.linalg.norm(maxs - mins))
        L = max(L, 1e-9)

        center = T_raw.mean(axis=0)
        S = (source_aug - center) / L
        T = (T_raw     - center) / L

        bf = get_bf(self.bf_t)

        alpha = float(getattr(self, "alpha_support", 0.25))
        c = 1.0 / max(alpha, 1e-6)
        c *= float(getattr(self, "c_mod_t", 1.0))

        bfs = [bf] * 3
        cs  = [c]  * 3

        S_list = S.tolist()
        D_list = disp_aug.tolist()

        try:
            ls, bs, conds = train_3d(
                S_list, D_list, bfs, cs,
                reg_lambda=float(getattr(self, "rbf_lambda", 0.0))
            )
        except TypeError:
            ls, bs, conds = train_3d(S_list, D_list, bfs, cs)

        pred = predict_3d(ls, bs, S_list, T.tolist(), bfs, cs)
        pred = np.asarray(pred, float)

        # ---------------- Optional: smooth the displacement field ----------------
        if smooth_iters and smooth_iters > 0 and smooth_strength and smooth_strength > 0:
            try:
                from scipy.spatial import cKDTree
                kdt = cKDTree(T_raw)
                _, idx = kdt.query(T_raw, k=min(int(smooth_k) + 1, len(T_raw)))
                idx = idx[:, 1:]  # drop self
                lam = float(np.clip(smooth_strength, 0.0, 1.0))
                for _ in range(int(smooth_iters)):
                    nbr_mean = pred[idx].mean(axis=1)
                    pred = (1.0 - lam) * pred + lam * nbr_mean
            except Exception:
                pass

        # ---------------- Optional: “nearly fixed” anchors (project to nearest T nodes) ----------------
        # If you want anchors closer to the seam to be more fixed than soft anchors allow, set anchor_soft smaller
        # or uncomment below to softly clamp a small subset.
        # if anchor_points is not None and len(anchor_points) > 0:
        #     try:
        #         from scipy.spatial import cKDTree
        #         treeT = cKDTree(T_raw)
        #         B0 = np.asarray(anchor_points, float)
        #         if B0.ndim == 1:
        #             B0 = B0.reshape(1, -1)
        #         d, j = treeT.query(B0, k=1)
        #         # Hard clamp only very-close points
        #         pred[j[d <= (10*tol)]] *= 0.0
        #     except Exception:
        #         pass

        return (T_raw + pred).tolist()


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
