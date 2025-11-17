import numpy as np


class Mesh():
    def __init__(self):
        self.node_count               = 0
        self.tet_count                = 0
        self.pyramid_count            = 0
        self.prism_count              = 0
        self.hex_count                = 0
        self.boundary_triangle_count  = 0
        self.boundary_quad_count      = 0
        self.boundary_edge_count      = 0
        self.number_of_edges          = 1
        self.surface_count            = 0

        self.nodes                    = [] # Nx3
        self.tets                     = [] # Nx4?                        # tetra4
        self.pyramids                 = [] # Nx5?                        # pyramid5
        self.prism                    = [] # Nx6?                        # penta6
        self.hex                      = [] # Nx8?                        # hexa8
        self.boundary_triangles       = [] # Nx4 - p1, p2, p3, surf      # tria3
        self.boundary_quads           = [] # Nx5 - p1, p2, p3, p4, surf  # quad4
        self.boundary_edges           = [] # Nx3 - p1, p2, surf          # unsure - just to help support 2d meshes

        self.boundary_surface_ids_n_3 = {}
        self.boundary_surface_ids_n_4 = {}

    def get_surface_nodes(self, surf_id):
        """
            Gets all of the node ids for a particular surface. note this does not return their coordinate values (x,y,z)
            Arguments
            ----------
                surf_id : int
                    Surface ID. View file in paraview/ensight to confirm id's.
            Returns
            ----------
                gl_to_lc : dict
                    key : int
                        global id (indeces) of the node in self.nodes.
                    val : int
                        local id of that node in this array.
                g_ids    : list
                    list of all global ids for the nodes in surface <surf_id>
        """
        g_ids = []
        gl_to_lc = {}
        _, f3 = self.get_surface_tria3(surf_id)
        _, f4 = self.get_surface_quad4(surf_id)
        for fid in f3:
            [g_ids.append(x) for x in self.get_tria3_nodes(fid)]
        for fid in f4:
            [g_ids.append(x) for x in self.get_quad4_nodes(fid)]
            
        g_ids = list(set(g_ids))
        for lc_id, g_id in enumerate(g_ids):
            gl_to_lc[g_id] = lc_id
        return gl_to_lc, g_ids

    def get_surface_tria3(self, surf_id):
        """
            Gets all of the face ids for a particular surface
            Arguments
            ----------
                surf_id : int
                    Surface ID. View file in paraview/ensight to confirm id's.
            Returns
            ----------
                gl_to_lc : dict
                    key : int
                        global id (indeces) of the face in self.boundary_triangles.
                    val : int
                        local id of that face in this array. 
                f_ids    : list
                    list of all face ids for the faces in surface <surf_id>
        """
        f_ids = []
        gl_to_lc = {}
        for i, bt in enumerate(self.boundary_triangles):
            if bt[3] == surf_id:
                g_id = i
                lc_id = len(f_ids)
                gl_to_lc[g_id]  = lc_id
                f_ids.append(g_id)
        return gl_to_lc, f_ids
    
    def get_surface_quad4(self, surf_id):
        """
            Gets all of the quad ids for a particular surface
            Arguments
            ----------
                surf_id : int
                    Surface ID. View file in paraview/ensight to confirm id's.
            Returns
            ----------
                gl_to_lc : dict
                    key : int
                        global id (indeces) of the face in self.boundary_quads.
                    val : int
                        local id of that face in this array. 
                f_ids    : list
                    list of all face ids for the faces in surface <surf_id>
        """
        f_ids = []
        gl_to_lc = {}
        for i, bt in enumerate(self.boundary_quads):
            if bt[4] == surf_id:
                g_id = i
                lc_id = len(f_ids)
                gl_to_lc[g_id]  = lc_id
                f_ids.append(g_id)
        return gl_to_lc, f_ids

    def get_tria3_nodes(self, face_id):
        """
            Gets all of the node ids for a particular face. note this does not return their coordinate values (x,y,z)
            Arguments
            ----------
                face_id : int
                    Face ID - Index of self.boundary_triangles
            Returns
            ----------
                node_ids : list
                    list of all global ids for the nodes in surface <surf_id>
        """
        node_ids = self.boundary_triangles[face_id][:-1]
        return node_ids

    def get_quad4_nodes(self, face_id):
        """
            Gets all of the node ids for a particular face. note this does not return their coordinate values (x,y,z)
            Arguments
            ----------
                face_id : int
                    Face ID - Index of self.boundary_quads
            Returns
            ----------
                node_ids : list
                    list of all global ids for the nodes in surface <surf_id>
        """
        node_ids = self.boundary_quads[face_id][:-1]
        return node_ids

    def convert_node_ids_to_coordinates(self, node_ids):
        """
            Converts a list of node ids (global ids) to coordinate values.
            Arguments
            ----------
                node_ids : list
                    list of global ids of nodes. 
            Returns
            ----------
                gl_to_lc : dict
                    key : int
                        global id (indeces) of the node in self.nodes.
                    val : int
                        local id of that node in coordinates.
                coordinates : list
                    list of all global ids for the nodes in surface <surf_id>
        """
        N = len(node_ids)
        coordinates = np.empty((N,3))
        gl_to_lc = {}
        for lc_id, g_id in enumerate(node_ids):
            lc_id, g_id = int(lc_id), int(g_id)
            coordinates[lc_id] = np.array(self.nodes[g_id])
            gl_to_lc[g_id] = lc_id
        return gl_to_lc, coordinates

    def get_element_by_ensight_id(self, ens_id):
        if ens_id == "tetra4":
            return self.tets
        elif ens_id == "pyramid5":
            return self.pyramids
        elif ens_id == "penta6":
            return self.prism
        elif ens_id == "hexa8":
            return self.hex
        elif ens_id == "tria3":
            return self.boundary_triangles #[:,:3] # remove the surface id column
        elif ens_id == "quad4":
            return self.boundary_quads #[:,:4]     # remove the surface id column

    def flip_along_axis(self, axis_norm=np.array([0,1,0])):
        """flips and merges a half geometry to recreate full geometry."""
        # take all nodes and flip them
        # eliminate duplicates
        flipped_nodes = []
        
        for x in self.nodes:
            flipped_nodes.append(x[:].tolist())
        
        unflipped_to_flipped = {}
        for i,x in enumerate(self.nodes):
            if x[axis_norm.value] == 0.0:
                unflipped_to_flipped[i] = i
                continue
            x[axis_norm.value] *= -1.0
            unflipped_to_flipped[i] = len(flipped_nodes)
            flipped_nodes.append(x[:].tolist())

        flipped_boundary_triangles = []

        for bt in self.boundary_triangles:
            flipped_boundary_triangles.append(bt)

        for bt in self.boundary_triangles:
            nbt = []
            nbt.append(unflipped_to_flipped[bt[0]])
            nbt.append(unflipped_to_flipped[bt[1]])
            nbt.append(unflipped_to_flipped[bt[2]])
            nbt.append(bt[3]+self.surface_count)
            flipped_boundary_triangles.append(nbt)

        flipped_boundary_quads = []

        for bq in self.boundary_quads:
            flipped_boundary_quads.append(bq)

        for bq in self.boundary_quads:
            nbq = []
            nbq.append(unflipped_to_flipped[bq[0]])
            nbq.append(unflipped_to_flipped[bq[1]])
            nbq.append(unflipped_to_flipped[bq[2]])
            nbq.append(unflipped_to_flipped[bq[3]])
            nbq.append(bq[4]+self.surface_count)
            flipped_boundary_quads.append(nbq)
        
        # update shit
        self.nodes = flipped_nodes
        self.node_count = len(flipped_nodes)
        self.boundary_triangles = flipped_boundary_triangles
        self.boundary_triangle_count = len(flipped_boundary_triangles)
        self.boundary_quads = flipped_boundary_quads
        self.boundary_quad_count = len(flipped_boundary_quads)
        self.boundary_surface_ids_n_3 = {}
        for i, bt in enumerate(flipped_boundary_triangles):
            self.boundary_surface_ids_n_3[i] = bt[3]
        self.boundary_surface_ids_n_4 = {}
        for i, bq in enumerate(flipped_boundary_quads):
            self.boundary_surface_ids_n_4[i] = bq[4]

    def remove_surfaces(self, surf_ids):
        """
            removes all nodes and faces that make up the farfield and surface planes.
        """

        # determine which nodes and elements need to be removed etc.
        n3_to_remove = []
        n3_to_keep   = []
        n4_to_remove = []
        n4_to_keep   = []
        nodes_to_remove = []
        nodes_to_keep   = []

        for i,x in enumerate(self.boundary_triangles):
            p1,p2,p3,s_id = x
            if s_id in surf_ids:
                n3_to_remove.append(i)
                nodes_to_remove.append(p1)
                nodes_to_remove.append(p2)
                nodes_to_remove.append(p3)
            else:
                n3_to_keep.append(i)
                nodes_to_keep.append(p1)
                nodes_to_keep.append(p2)
                nodes_to_keep.append(p3)

        for i,x in enumerate(self.boundary_quads):
            p1,p2,p3,p4,s_id = x
            if s_id in surf_ids:
                n4_to_remove.append(i)
                nodes_to_remove.append(p1)
                nodes_to_remove.append(p2)
                nodes_to_remove.append(p3)
                nodes_to_remove.append(p4)
            else:
                n4_to_keep.append(i)
                nodes_to_keep.append(p1)
                nodes_to_keep.append(p2)
                nodes_to_keep.append(p3)
                nodes_to_keep.append(p4)
        
        # remove nodes in the remove list that intersect with keep and so should not be dropped.
        nodes_to_remove = set(nodes_to_remove)
        nodes_to_keep   = set(nodes_to_keep)
        nodes_to_remove = nodes_to_remove - nodes_to_keep
        nodes_to_remove = list(nodes_to_remove)
        nodes_to_keep   = list(nodes_to_keep)
        
        # reindex new lists
        all_surfaces = [x+1 for x in range(self.surface_count)]
        sub_list_s_ids, gl_to_lc_s_ids = self.reindex_subset(all_surfaces, [x-1 for x in surf_ids]) # subtract 1 to get index of surface not id
        sub_list_nodes, gl_to_lc_nodes = self.reindex_subset(self.nodes, nodes_to_remove)
        sub_list_trias, gl_to_lc_trias = self.reindex_subset(self.boundary_triangles, n3_to_remove)
        sub_list_quads, gl_to_lc_quads = self.reindex_subset(self.boundary_quads, n4_to_remove)
        sub_list_trias = [[gl_to_lc_nodes[g_id] for g_id in x[:-1]] + [gl_to_lc_s_ids[x[-1]-1]+1] for x in sub_list_trias]
        sub_list_quads = [[gl_to_lc_nodes[g_id] for g_id in x[:-1]] + [gl_to_lc_s_ids[x[-1]-1]+1] for x in sub_list_quads]

        # update
        self.nodes                   = sub_list_nodes
        self.node_count              = len(sub_list_nodes)
        self.boundary_triangles      = sub_list_trias
        self.boundary_triangle_count = len(sub_list_trias)
        self.boundary_quads          = sub_list_quads
        
        self.boundary_quad_count     = len(sub_list_quads)
        # update boundary dictionarys too
        boundary_surface_ids_n_3 = {}
        for k,v in self.boundary_surface_ids_n_3.items():
            if v not in surf_ids:
                boundary_surface_ids_n_3[gl_to_lc_trias[k]] = gl_to_lc_s_ids[v-1]+1
        self.boundary_surface_ids_n_3 = boundary_surface_ids_n_3
        boundary_surface_ids_n_4 = {}
        for k,v in self.boundary_surface_ids_n_4.items():
            if v not in surf_ids:
                boundary_surface_ids_n_4[gl_to_lc_quads[k]] = gl_to_lc_s_ids[v-1]+1
        self.boundary_surface_ids_n_4 = boundary_surface_ids_n_4

    def reindex_subset(self, full_list, rem_list):
        """
            reindexes a set of inputs. removes rem_list from full_list and 
            returns the new list along with a gl_to_lc converter to convert 
            from something that uses full list to needing to use sub_list.
        """
        gl_to_lc = {}
        sub_list = []
        for i, x in enumerate(full_list):
            if i in rem_list:
                continue
            gl_to_lc[i] = len(sub_list)
            sub_list.append(x)
        return sub_list, gl_to_lc

    @property
    def total_element_count(self) -> int:
        return self.tet_count + self.pyramid_count + self.prism_count + self.hex_count

    @property
    def boundary_total_count(self) -> int:
        return self.boundary_triangle_count + self.boundary_quad_count

    @property
    def get_surface_ids(self):
        ids_tri = set(self.boundary_surface_ids_n_3.tolist()) if getattr(self, "boundary_surface_ids_n_3", None) is not None else set()
        ids_quad = set(self.boundary_surface_ids_n_4.tolist()) if getattr(self, "boundary_surface_ids_n_4", None) is not None else set()
        all_ids = ids_tri.union(ids_quad)
        if not all_ids:
            print("[WARN] No surface IDs found in FroFile.")
        return sorted(all_ids)

    def __str__(self) -> str:
        return (
            f"Nodes:      {len(self.nodes):>12,}/{self.node_count:>12,}\n"
            f"Bound Tri:  {len(self.boundary_triangles):>12,}/{self.boundary_triangle_count:>12,}\n"
            f"Bound Quad: {len(self.boundary_quads):>12,}/{self.boundary_quad_count:>12,}\n"
            f"Tetrahedra: {len(self.tets):>12,}/{self.tet_count:>12,}\n"
            f"Pyramids:   {len(self.pyramids):>12,}/{self.pyramid_count:>12,}\n"
            f"Prisms:     {len(self.prism):>12,}/{self.prism_count:>12,}\n"
            f"Hexahedra:  {len(self.hex):>12,}/{self.hex_count:>12,}\n"
            f"Total Elem: {len(self.tets) + len(self.pyramids) + len(self.prism) + len(self.hex):>12,}/{self.total_element_count:>12,}\n"
            f"Surface Ids {self.num_surface_ids:>12,}"
        )
    
    def read_file(self, filename : str) -> int:
        pass

    def write_file(self, filename : str, mesh=None) -> int:
        pass

    @classmethod
    def from_pyvista(cls, pv_mesh):
        import numpy as np
        import pyvista as pv
        try:
            import vtk
        except Exception:
            vtk = None

        # ---------- helpers ----------
        def _is_multiblock(obj):
            if obj is None:
                return False
            if 'MultiBlock' in type(obj).__name__:
                return True
            return hasattr(obj, 'get_block_name') or hasattr(obj, 'GetNumberOfBlocks') or hasattr(obj, '__len__')

        def _mb_len(mb):
            try:
                return int(len(mb))
            except Exception:
                if hasattr(mb, 'GetNumberOfBlocks'):
                    return int(mb.GetNumberOfBlocks())
                return 0

        def _mb_get(mb, i):
            try:
                return mb[i]
            except Exception:
                if hasattr(mb, 'GetBlock'):
                    return mb.GetBlock(i)
                return None

        def _mb_name(mb, i):
            if hasattr(mb, 'get_block_name'):
                try:
                    n = mb.get_block_name(i)
                    if n:
                        return str(n)
                except Exception:
                    pass
            if vtk is not None and hasattr(mb, 'GetMetaData'):
                try:
                    md = mb.GetMetaData(i)
                    if md and md.Has(vtk.vtkCompositeDataSet.NAME()):
                        return md.Get(vtk.vtkCompositeDataSet.NAME())
                except Exception:
                    pass
            return f"block_{i}"

        def _iter_leaves_any(obj, prefix=""):
            if _is_multiblock(obj) and not isinstance(obj, pv.DataSet):
                n = _mb_len(obj)
                for i in range(n):
                    child = _mb_get(obj, i)
                    key = _mb_name(obj, i)
                    part = str(key).strip() if key is not None else f"block_{i}"
                    new_prefix = f"{prefix}{part}/" if prefix else f"{part}/"
                    yield from _iter_leaves_any(child, new_prefix)
            else:
                if isinstance(obj, pv.DataSet):
                    name = prefix[:-1] if prefix.endswith("/") else (prefix or "block")
                    yield name, obj
                else:
                    try:
                        wrapped = pv.wrap(obj)
                        if isinstance(wrapped, pv.DataSet):
                            name = prefix[:-1] if prefix.endswith("/") else (prefix or "block")
                            yield name, wrapped
                    except Exception:
                        return

        def _get_sid_array(block, n_cells, fallback):
            if n_cells <= 0:
                return np.empty((0,), dtype=int)
            for key in ("SurfaceID", "surface_id", "surfaceId", "SURFACE_ID"):
                if key in block.cell_data:
                    arr = np.asarray(block.cell_data[key]).ravel()
                    if arr.size == n_cells:
                        return arr.astype(int) + 1
            if len(block.cell_data) == 1:
                arr = np.asarray(list(block.cell_data.values())[0]).ravel()
                if arr.size == n_cells:
                    return arr.astype(int) + 1
            return np.full(n_cells, int(fallback), dtype=int)

        # ---------- main ----------
        self = cls()
        self.boundary_triangles = []
        self.boundary_quads = []
        self.boundary_surface_ids_n_3 = {}
        self.boundary_surface_ids_n_4 = {}

        all_nodes = []
        node_offset = 0
        surface_ids = set()

        if _is_multiblock(pv_mesh) and not isinstance(pv_mesh, pv.DataSet):
            leaves = list(_iter_leaves_any(pv_mesh))
            leaves = [(nm, leaf) for nm, leaf in leaves if isinstance(leaf, pv.DataSet)]
        else:
            leaves = [("block_0", pv_mesh if isinstance(pv_mesh, pv.DataSet) else pv.wrap(pv_mesh))]

        for leaf_idx, (leaf_name, block) in enumerate(leaves, start=1):
            if block is None or not hasattr(block, "points"):
                continue

            try:
                points = np.asarray(block.points)
            except Exception:
                block = pv.wrap(block)
                points = np.asarray(block.points)

            n_points = int(getattr(block, "n_points", 0) or 0)
            n_cells  = int(getattr(block, "n_cells", 0) or 0)

            if n_points > 0:
                all_nodes.append(points)

            # --- cell handling ---
            if n_cells > 0:
                if hasattr(block, "cells_dict") and block.cells_dict:
                    # Standard path
                    cells = block.cells_dict
                    print(f"[INFO] Block '{leaf_name}' cell types: {list(cells.keys())}")

                    n_tri  = len(cells.get(5, []))
                    n_quad = len(cells.get(9, []))
                    n_poly = len(cells.get(7, []))
                    sids_all = _get_sid_array(block, n_cells, fallback=leaf_idx)

                    sids_tri  = sids_all[:n_tri] if n_tri else np.empty((0,), dtype=int)
                    sids_quad = sids_all[n_tri:n_tri + n_quad] if n_quad else np.empty((0,), dtype=int)
                    sids_poly = sids_all[n_tri + n_quad : n_tri + n_quad + n_poly] if n_poly else np.empty((0,), dtype=int)

                    if sids_tri.size != n_tri:  sids_tri = np.full(n_tri, leaf_idx, dtype=int)
                    if sids_quad.size != n_quad: sids_quad = np.full(n_quad, leaf_idx, dtype=int)
                    if sids_poly.size != n_poly: sids_poly = np.full(n_poly, leaf_idx, dtype=int)

                    if 5 in cells and n_tri:
                        for i, t in enumerate(cells[5]):
                            tri = [int(t[0])+node_offset, int(t[1])+node_offset, int(t[2])+node_offset, int(sids_tri[i])]
                            self.boundary_triangles.append(tri)
                            self.boundary_surface_ids_n_3[len(self.boundary_triangles)-1] = int(sids_tri[i])

                    if 9 in cells and n_quad:
                        for i, q in enumerate(cells[9]):
                            quad = [int(q[0])+node_offset, int(q[1])+node_offset, int(q[2])+node_offset, int(q[3])+node_offset, int(sids_quad[i])]
                            self.boundary_quads.append(quad)
                            self.boundary_surface_ids_n_4[len(self.boundary_quads)-1] = int(sids_quad[i])

                    if 7 in cells and n_poly:
                        for pi, poly in enumerate(cells[7]):
                            ids = [int(x)+node_offset for x in poly]
                            if len(ids) >= 3:
                                for k in range(1, len(ids)-1):
                                    tri = [ids[0], ids[k], ids[k+1], int(sids_poly[pi])]
                                    self.boundary_triangles.append(tri)
                                    self.boundary_surface_ids_n_3[len(self.boundary_triangles)-1] = int(sids_poly[pi])

                    surface_ids.update(sids_tri.tolist())
                    surface_ids.update(sids_quad.tolist())
                    surface_ids.update(sids_poly.tolist())

                elif isinstance(block, pv.PolyData) and block.faces.size > 0:
                    # --- fallback: parse faces directly ---
                    faces = block.faces.reshape((-1, 4))  # each row: [nverts, id0, id1, id2...]
                    print(f"[INFO] Block '{leaf_name}' using PolyData.faces fallback, {len(faces)} faces")
                    for f in faces:
                        nverts = f[0]
                        ids = (f[1:1+nverts] + node_offset).tolist()
                        if nverts == 3:
                            tri = [ids[0], ids[1], ids[2], leaf_idx]
                            self.boundary_triangles.append(tri)
                            self.boundary_surface_ids_n_3[len(self.boundary_triangles)-1] = leaf_idx
                        elif nverts == 4:
                            quad = [ids[0], ids[1], ids[2], ids[3], leaf_idx]
                            self.boundary_quads.append(quad)
                            self.boundary_surface_ids_n_4[len(self.boundary_quads)-1] = leaf_idx
                        elif nverts > 4:
                            # triangulate n-gon
                            for k in range(1, nverts-1):
                                tri = [ids[0], ids[k], ids[k+1], leaf_idx]
                                self.boundary_triangles.append(tri)
                                self.boundary_surface_ids_n_3[len(self.boundary_triangles)-1] = leaf_idx
                    surface_ids.add(leaf_idx)

            node_offset += n_points

        if not all_nodes:
            raise RuntimeError("[Mesh.from_pyvista] No points found in any dataset.")

        self.nodes = np.vstack(all_nodes)
        self.node_count = len(self.nodes)
        self.boundary_triangle_count = len(self.boundary_triangles)
        self.boundary_quad_count = len(self.boundary_quads)
        self.surface_count = len(surface_ids) if surface_ids else 0

        print(f"[✔] Mesh.from_pyvista: nodes={self.node_count}, "
            f"triangles={self.boundary_triangle_count}, quads={self.boundary_quad_count}, "
            f"surfaces={self.surface_count}")
        return self
