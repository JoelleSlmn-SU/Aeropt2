from time import time
import os
import sys
import numpy as np

import os
import sys
sys.path.append(os.path.dirname('mfirst'))
from mfirst.MeshModification.CreateMorph import MorphModelBase
from mfirst.Utilities.Logger import Logger
from mfirst.Utilities.Paths import load_functions


class MorphModelCRM(MorphModelBase):
    def __init__(self, f_name=None, con=False, pb=[], path=None, f=None, T=[], U=[], phi=0.0):
        super().__init__(f_name, con, pb, path, f, T, U, phi)
        self.log = Logger()
    
    def f(self, verts, phi):#, verts, phi):
        if self._f is not None:
            return self._f(verts, phi)
        elif self.f_name is not None:
            funcs = load_functions(f"{self.path}functions.py")
            func = getattr(funcs, f"{self.f_name}")
            return func(verts, phi)
        else:
            self.log.warn(f"Function name undefined. _f: {self._f} | f_name: {self.f_name}")
            raise RuntimeError()
        
    def f2(self, verts, phi):#, verts, phi):
        if self._f is not None:
            return self._f(verts, phi)
        elif self.f_name is not None:
            funcs = load_functions(f"{self.path}functions.py")
            func = getattr(funcs, f"{self.f_name}")
            return func(verts, phi)
        else:
            self.log.warn(f"Function name undefined. _f: {self._f} | f_name: {self.f_name}")
            raise RuntimeError()

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
        return retval
    
    def get_nodes(self, sections, surfaces, faces, nodes, ff):
        g_ids = []
        # get node ids of sections. Sections = list of surfaces
        for s in sections:
            s_ids = ff.sections[s]
            for s in s_ids:
                [g_ids.append(x) for x in ff.get_surface_nodes(s)[1]]

        # get node ids of surfaces
        for s in surfaces:
            [g_ids.append(x) for x in ff.get_surface_nodes(s)[1]]

        # get node ids of faces
        for s,fs in faces.items():
            # paraview uses local ids to identify faces/nodes so need 
            # to convert from surfaces local space to global space
            _, f_ids = ff.get_surface_tria3(s)
            for f in fs:
                if f < len(f_ids):
                    [g_ids.append(x) for x in ff.get_tria3_nodes(f_ids[f])]

        # just append node ids
        for s, ns in nodes.items():
            # paraview uses local ids to identify faces/nodes so need 
            # to convert from surfaces local space to global space
            _, s_ids = ff.get_surface_nodes(s)
            for n in ns:
                if n < len(s_ids):
                    g_ids.append(s_ids[n])
        # remove duplicates
        g_ids = list(set(g_ids))
        return g_ids

    def get_tnodes_crm(self, sections, surfaces, faces, nodes, ff):
        g_ids = []
        for node in ff.nodes:
            if round(node[1],2) == sections and -10 <= node[2] <= 10:
                temp = np.array(node)
                g_ids.append(float(np.where((ff.nodes == temp).all(axis=1))[0]))
        g_ids = list(set(g_ids))
        return g_ids
    
    def get_unodes_crm(self, sections, surfaces, faces, nodes, ff):
        g_ids = []
        for node in ff.nodes:
            if sections[0] < round(node[1],2) < sections[1] and -10 <= node[2] <= 10:
                temp = np.array(node)
                g_ids.append(float(np.where((ff.nodes == temp).all(axis=1))[0]))
        g_ids = list(set(g_ids))
        return g_ids

    def get_t_node_gids(self, ff):
        if type(self.t_sections)==str:
            return self.get_nodes(self.t_sections, self.t_surfaces, self.t_face_ids, self.t_node_ids, ff)
        elif type(self.t_sections)==float:
            return self.get_tnodes_crm(self.t_sections, self.t_surfaces, self.t_face_ids, self.t_node_ids, ff)
    
    def get_u_node_gids(self, ff):
        u_gids = self.get_unodes_crm(self.u_sections, self.u_surfaces, self.u_face_ids, self.u_node_ids, ff)
        t_gids = self.get_t_node_gids(ff)
        U = [x for x in u_gids if x not in t_gids]
        return U

    def get_c_node_gids(self, ff):
        t = self.get_t_node_gids(ff)
        t = set(t)
        u = set(self.get_u_node_gids(ff))
        c = set([i for i in range(ff.node_count)])
        C = c - (t | u) 
        f = []
        for s in ff.farfield_ids:
            [f.append(x) for x in ff.get_surface_nodes(s)[1]]
        C = C - set(f)
        return C

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
        # find all boundary nodes (unordered)
        b = []
        for g_id in m:
            connected_nodes = ff.node_connections[g_id]
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

    def plot_TUC(self, ff, title="title"):
        from mfirst.Utilities.plotter import plot_point_clouds
        T  = self.get_t_node_gids(ff)
        U  = self.get_u_node_gids(ff)
        C  = self.get_c_node_gids(ff)
        iB = self.GetIndepBoundaries(ff)
        B = set([])
        for k, v in iB.items():
            B = B | set(v)
        B = list(B)
        _, t = ff.convert_node_ids_to_coordinates(T)
        _, u = ff.convert_node_ids_to_coordinates(U)
        _, c = ff.convert_node_ids_to_coordinates(C)
        _, b = ff.convert_node_ids_to_coordinates(B)
        cv = {}
        cv["red"]    = t
        cv["blue"]   = u
        cv["yellow"] = c
        #cv["yellow"] = b
        plot_point_clouds(cv, title)

    def get_indep_boundaries(self, ff):
        pass

    def GetRigidBodies_original(self, ff, bts):
        # all nodes in c connected to u or c need to be linearly translated. 
        rigid_bodies = {}
        bounds = self.GetIndepBoundaries(ff)
        #print(f"bts keys    = {list(bts.keys())}")
        #print(f"bounds keys = {list(bounds.keys())}")
        #for b,t in bts.items():
        #    print(f"{b} - {t}")
        C      = self.get_c_node_gids(ff)
        T      = self.get_t_node_gids(ff)
        target_nodes = {}
        start_time = time()
        for b in bounds.keys():
            self.log.debg(f"checking bound: {b}")
            bt = bts[b]
            #print(f"bt = {bt}")
            if all(v == 0.0 for v in bt):
                continue
            g_ids = bounds[b]
            #print(f"num g_ids = {len(g_ids)}")
            #print(f"calculating for {b} - num nodes = {len(g_ids)}")
            to_check = g_ids
            checked  = []
            while len(to_check) > 0:
                #input()
                next_check = to_check.pop()
                connected = ff.node_connections[next_check]
                #print(f"tc = {len(to_check)}, cd  = {len(checked)}, tot C = {len(C)}, Node: {next_check} - {connected}")
                for c in connected:
                    if (c not in checked) and (c in C) and (c not in to_check):
                        to_check.append(c)
                #to_check = list(set(to_check) | set(connected))
                checked.append(next_check)
            ## checked now equals the set of all nodes in C connected to 
            # the moved boundary. set the translation accordingly?
            rigid_bodies[b] = checked
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

    def recover_boundaries(self, ff, bts, rb="original"):
        if rb == "original":
            rigid_body_method = self.GetRigidBodies_original
        elif rb == "chains":
            rigid_body_method = self.GetRigidBodies_chains
        # all nodes in c connected to u or c need to be linearly translated. 
        #bounds = self.GetIndepBoundaries(ff)
        #print(f"bts keys    = {list(bts.keys())}")
        #print(f"bounds keys = {list(bounds.keys())}")
        #for b,t in bts.items():
        #    print(f"{b} - {t}")
        #C      = self.get_c_node_gids(ff)
        T      = self.get_t_node_gids(ff)
        target_nodes = {}
        #start_time = time()
        rb_dict = rigid_body_method(ff, bts)
        for key, boundary in rb_dict.items():
            self.log.debg(f"checking bound: {key}")
            bt = bts[key]
            
            for g_id in boundary:
                if g_id not in T:
                    target_nodes[g_id] = ff.nodes[g_id] + bt
        #print("boundaries corrected")
        return target_nodes

if __name__ == "__main__":
    from mfirst.FileIO.FroFile import FroFile
    from mfirst.Enumerations.LogLevel import LogLevel
    from mfirst.Config.RemoteConfig import RemoteConfig
    #fn  = f"{RemoteConfig().mfirst_client}{os.sep}dev{os.sep}mesh_morphing_cases{os.sep}squares.fro"
    #ff = FroFile().defaultSquares(f"{fn}", log_level=LogLevel.NONE)
    #mm = MorphModel()
    #mm.t_surfaces = [2]
    #mm.u_surfaces = [3]
    #mm.plot_TUC(ff)
    fn  = f"{RemoteConfig().mfirst_client}{os.sep}dev{os.sep}mesh_morphing_cases{os.sep}Skylon.fro"
    ff = FroFile().defaultSkylon(f"{fn}", log_level=LogLevel.NONE)
    mm = MorphModel()
    #mm.t_sections = ['mid_panel_port']
    #mm.u_sections = ['fuselage', 'fuselage_front']
    mm.t_sections = ['center_mid_panel_port']
    mm.u_sections = ['mid_panel_port']
    #mm.t_sections = ['fuselage_front']
    mm.plot_TUC(ff)
