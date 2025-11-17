from datetime import datetime
from math import cos, sin
import json 
import re

# add mfirst path guard?
import os
import sys
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname("FileRW"))
from FileRW.Mesh import Mesh

import numpy as np

class GeometryData():
    def __init__(self):
        self.volume = 0.0
        self.surface_area = 0.0
        self.center_mass = [] #: List[float] = field(default_factory=list, compare=False)
        self.centroid = [] # : List[float] = field(default_factory=list, compare=False)

        self.x_planform_area = 0.0
        self.y_planform_area = 0.0
        self.z_planform_area = 0.0

        self.min_x = 0.0
        self.max_x = 0.0
        self.min_y = 0.0
        self.max_y = 0.0
        self.min_z = 0.0
        self.max_z = 0.0

        self.is_watertight = False #: bool = False

    @property
    def bounds(self):
        return (self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z)

    '''def GetProperty(self, prop: GeometryDataProperty):
        return getattr(self, prop.name)'''
    
    @property
    def Length(self):
        return self.max_x - self.min_x
    
    @property
    def Width(self):
        return self.max_y - self.min_y
    
    @property
    def Height(self):
        return self.max_z - self.min_z

    def __str__(self):
        retVal = ""
        retVal += "Volume:       {}\n".format(self.volume)
        retVal += "Surface Area: {}\n".format(self.surface_area)
        retVal += "Center:       {}\n".format(self.center_mass)
        retVal += "Centroid:     {}\n".format(self.centroid)
        retVal += "X Plan:       {}\n".format(self.x_planform_area)
        retVal += "Y Plan:       {}\n".format(self.y_planform_area)
        retVal += "Z Plan:       {}\n".format(self.z_planform_area)
        retVal += "Length:       {}\n".format(self.Length)
        retVal += "Width:        {}\n".format(self.Width)
        retVal += "Height:       {}\n".format(self.Height)
        return retVal

class FroFile(Mesh):

    def __init__(self, filepath=None, sections={}, farfield=None):
        super().__init__()

        # initialise some basic support variables for the class.
        self.everything_else    = [] # until the parser can successfully parse the mangled fro files OR writes them out more neatly, we only need to concern ourselfs with the coord and triangle data
        self.farfield_ids       = []
        self.sections           = {}
        self.section_faces_n_3  = {}
        self.section_faces_n_4  = {}
        self.section_vertices   = {}
        self.section_nodes      = {}
        self.section_bounds     = {}

        self.num_curves    = 0
        self.num_segments  = 0

        if filepath is not None:
            self.read_file(filepath)

        if farfield is not None:
            self.setupFarfield(farfield)
        #print(sections)
        for k,v in sections.items():
            #k,v = list(s.items())[0]
            self.add_section(k, v)
    
    @property
    def num_surface_ids(self):
        if hasattr(self, "boundary_surface_ids_n_3") and isinstance(self.boundary_surface_ids_n_3, dict):
            tri_ids = set(self.boundary_surface_ids_n_3.values())
        else:
            tri_ids = set()

        if hasattr(self, "boundary_surface_ids_n_4") and isinstance(self.boundary_surface_ids_n_4, dict):
            quad_ids = set(self.boundary_surface_ids_n_4.values())
        else:
            quad_ids = set()

        return max(tri_ids.union(quad_ids)) if tri_ids or quad_ids else 0
    
    ## Mesh Interface
    def read_file(self, filepath):
        with open(filepath) as f:
            lines = f.read().splitlines()

        l1 = lines[0].split()
        if len(l1) == 8:
            self.boundary_quad_count     = 0
            self.boundary_triangle_count = int(l1[0])
            self.node_count              = int(l1[1])
            num_curves                   = int(l1[4])
            num_segments                 = int(l1[5])
        else:
            self.boundary_quad_count     = int(l1[0])
            self.boundary_triangle_count = int(l1[1])
            self.node_count              = int(l1[2])
            num_curves                   = int(l1[5])
            num_segments                 = int(l1[6])

        surface_count = 0

        # ✅ numeric arrays with correct shapes/dtypes
        self.nodes              = np.zeros((self.node_count,              3), dtype=float)
        self.boundary_triangles = np.zeros((self.boundary_triangle_count, 4), dtype=int)
        self.boundary_quads     = np.zeros((self.boundary_quad_count,     5), dtype=int)

        # initialise id containers (dicts are fine; property handles both)
        self.boundary_surface_ids_n_3 = {}
        self.boundary_surface_ids_n_4 = {}

        # --- vertices ---
        i0 = 1
        i1 = i0 + self.node_count
        for line in range(i0, i1):
            idx = line - i0
            c = lines[line].split()
            # columns: id x y z ...
            self.nodes[idx, 0] = float(c[1])
            self.nodes[idx, 1] = float(c[2])
            self.nodes[idx, 2] = float(c[3])

        # --- quads ---
        i0 = i1
        i1 = i0 + self.boundary_quad_count
        for line in range(i0, i1):
            idx = line - i0
            q = lines[line].split()
            v1 = int(q[1]) - 1
            v2 = int(q[2]) - 1
            v3 = int(q[3]) - 1
            v4 = int(q[4]) - 1
            sid = int(q[5])
            self.boundary_quads[idx, :] = [v1, v2, v3, v4, sid]
            self.boundary_surface_ids_n_4[idx] = sid
            surface_count = max(surface_count, sid)

        # --- triangles ---
        i0 = i1
        i1 = i0 + self.boundary_triangle_count
        for line in range(i0, i1):
            idx = line - i0
            t = lines[line].split()
            v1 = int(t[1]) - 1
            v2 = int(t[2]) - 1
            v3 = int(t[3]) - 1
            sid = int(t[4])
            self.boundary_triangles[idx, :] = [v1, v2, v3, sid]
            self.boundary_surface_ids_n_3[idx] = sid
            surface_count = max(surface_count, sid)

        self.surface_count = surface_count

        # stash the tail
        self.everything_else = lines[i1:]

        # --- node connectivity (using flat rows now) ---
        self.node_connections = {i: [] for i in range(self.node_count)}

        # triangles
        for row in self.boundary_triangles:
            p1, p2, p3, _sid = row
            for a, b in [(p1, p2), (p2, p3), (p3, p1)]:
                if b not in self.node_connections[a]:
                    self.node_connections[a].append(int(b))
                if a not in self.node_connections[b]:
                    self.node_connections[b].append(int(a))

        # quads
        for row in self.boundary_quads:
            p1, p2, p3, p4, _sid = row
            pts = [p1, p2, p3, p4]
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = pts[i], pts[j]
                    if b not in self.node_connections[a]:
                        self.node_connections[a].append(int(b))
                    if a not in self.node_connections[b]:
                        self.node_connections[b].append(int(a))
        
    def set_node_connections(self, tol=None):
        """
        Build node-to-node adjacency from boundary faces and glue co-located nodes
        across surfaces using a tolerance.
        Produces: self.node_connections: dict[int, list[int]]
        """
        import numpy as np
        from collections import defaultdict

        N = int(getattr(self, "node_count", 0) or (len(self.nodes) if hasattr(self, "nodes") else 0))
        if N == 0 or not hasattr(self, "nodes"):
            self.node_connections = {}
            return

        # Ensure arrays exist with right shapes/dtypes
        tris = np.asarray(getattr(self, "boundary_triangles", np.zeros((0, 4), dtype=int)))
        quads = np.asarray(getattr(self, "boundary_quads", np.zeros((0, 5), dtype=int)))
        if tris.size and tris.dtype != np.int64 and tris.dtype != np.int32:
            tris = tris.astype(int, copy=False)
        if quads.size and quads.dtype != np.int64 and quads.dtype != np.int32:
            quads = quads.astype(int, copy=False)

        # Assemble with sets (fast, dedup naturally)
        adj = {i: set() for i in range(N)}

        # Triangles: cols 0..2 are node ids (last col is sid)
        for row in tris:
            if row.shape[0] < 3:  # guard malformed
                continue
            p1, p2, p3 = int(row[0]), int(row[1]), int(row[2])
            for a, b in ((p1, p2), (p2, p3), (p3, p1)):
                if 0 <= a < N and 0 <= b < N:
                    adj[a].add(b); adj[b].add(a)

        # Quads: cols 0..3 are node ids (last col is sid)
        for row in quads:
            if row.shape[0] < 4:
                continue
            p1, p2, p3, p4 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            pts = (p1, p2, p3, p4)
            for i in range(4):
                a = pts[i]
                for j in range(i + 1, 4):
                    b = pts[j]
                    if 0 <= a < N and 0 <= b < N:
                        adj[a].add(b); adj[b].add(a)

        # Adaptive tolerance for co-located nodes: use a fraction of median edge length
        if tol is None:
            edge_lengths = []
            nodes = np.asarray(self.nodes, float)
            # sample a subset of edges to estimate scale
            for row in tris[:min(len(tris), 2000)]:
                p1, p2, p3 = int(row[0]), int(row[1]), int(row[2])
                edge_lengths += [
                    np.linalg.norm(nodes[p1] - nodes[p2]),
                    np.linalg.norm(nodes[p2] - nodes[p3]),
                    np.linalg.norm(nodes[p3] - nodes[p1]),
                ]
            for row in quads[:min(len(quads), 2000)]:
                p1, p2, p3, p4 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                edge_lengths += [
                    np.linalg.norm(nodes[p1] - nodes[p2]),
                    np.linalg.norm(nodes[p2] - nodes[p3]),
                    np.linalg.norm(nodes[p3] - nodes[p4]),
                    np.linalg.norm(nodes[p4] - nodes[p1]),
                ]
            if edge_lengths:
                med = np.median(edge_lengths)
                tol = max(1e-9, 1e-3 * med)   # 0.1% of median edge length
            else:
                # Fallback: scale from bbox
                mins = nodes.min(axis=0); maxs = nodes.max(axis=0)
                diag = np.linalg.norm(maxs - mins)
                tol = max(1e-9, 1e-6 * diag)

        # Glue co-located nodes across surfaces using a hashing grid
        inv = 1.0 / tol
        nodes = np.asarray(self.nodes, float)
        coord_hash = defaultdict(list)
        for i in range(N):
            key = tuple(np.round(nodes[i] * inv).astype(int))
            coord_hash[key].append(i)

        for group in coord_hash.values():
            if len(group) < 2:
                continue
            # fully connect the co-located group
            for i in group:
                gi = int(i)
                if 0 <= gi < N:
                    adj_i = adj[gi]
                    for j in group:
                        gj = int(j)
                        if gi != gj and 0 <= gj < N:
                            adj_i.add(gj)

        # Convert to sorted lists for downstream code
        self.node_connections = {i: sorted(lst) for i, lst in adj.items()}

        
    def write_file(self, filename=None):
        """
            filename should include fileextension (.fro) when passed in
            converts from 0 indexed python ids to 1 indexed fortran indexes
            filename can also include a path to save to a specific location
        """
        if filename is None:
            now = datetime.now()
            t = now.strftime("%d_%m_%Y_%H_%M_%S")
            filename = f"new_{t}.fro"
        retval = f"    {self.boundary_triangle_count}    {self.node_count}    1    0  {self.num_surface_ids}   {10}    0    0\n"

        def fff(f_in):
            # fortran float formatting - formats a float value into the same format 
            # modified from https://stackoverflow.com/questions/13031636/string-format-start-scientific-notation-with-0-for-positive-number-with-fo
            foo = re.sub(r"([1-9])\.", r"0.\1", "{: .11E}".format(f_in))    # shift decimal place by 1
            new_exp = "E{:+03}".format(int(re.search(r"E([+-]\d+)", foo).group(1))+1) # get the old exponent and increment its integer value by 1
            foo = re.sub(r"E([+-]\d+)", new_exp, foo) # reprint the exponent
            # For Formatting info see https://docs.python.org/3.3/library/string.html#formatspec 
            # Format {:>9} - id, col length 9, prepend with spaces : search 'Aligning the text and specifying a width'
            # Format {: .12E} - vertices, include a space if positive, but not if negative, column width consistently 12, display exponent form but big E not little : search 'Replacing %+f, %-f, and % f and specifying a sign'
            # format re 
            #   re.sub(r"([1-9])\.", r"0.\1", "{: .11E}".format(testnum))
            #   re.sub(r"E(\+\-)", r"E\_", "{: .11E}".format(testnum))
            # - \d = [0-9]
            # - () match everything in this bracket
            # - \. . is a special character so needs to be escaped with \. 
            # - [] is a set of values - special characters lose speciality in this set
            # - + after a set (as returned by \d) will search for any number of occurances
            return foo
        
        for i, vert in enumerate(self.nodes):
            retval += "{:>9} {} {} {} {}\n".format(i+1, fff(vert[0]), fff(vert[1]), fff(vert[2]), fff(0.0), fff(0.0)) 

        for i, face in enumerate(self.boundary_quads):
            retval += "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}  \n".format(i+1, int(face[0]+1), int(face[1]+1), int(face[2]+1), int(face[3]+1), int(face[4])) 

        for i, face in enumerate(self.boundary_triangles):
            retval += "{:>10}{:>10}{:>10}{:>10}{:>10}  \n".format(i+1, int(face[0]+1), int(face[1]+1), int(face[2]+1), int(face[3])) 
            

        for ee in self.everything_else:
            retval += ee + "\n"

        of = open(filename, "w")
        of.write(retval)
        of.close()

    def clean(self, tol=1e-8, remove_unreferenced=True):
        """
        Remove duplicate nodes (within `tol`) and optionally remove unreferenced nodes.
        Updates connectivity and node_connections.
        """
        import numpy as np

        coords = np.asarray(self.nodes, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 3:
            raise ValueError("self.nodes must be an (N,3) array")

        # --- Step 1: detect duplicates using rounding ---
        keys = [tuple(np.round(pt / tol).astype(int)) for pt in coords]

        unique_map = {}
        old_to_new = {}
        new_nodes = []
        for old_id, key in enumerate(keys):
            if key in unique_map:
                old_to_new[old_id] = unique_map[key]  # duplicate
            else:
                new_id = len(new_nodes)
                unique_map[key] = new_id
                old_to_new[old_id] = new_id
                new_nodes.append(coords[old_id])

        # --- Step 2: update faces with new IDs ---
        self.boundary_triangles = np.array([
            [old_to_new[int(f[0])], old_to_new[int(f[1])],
            old_to_new[int(f[2])], int(f[3])]
            for f in self.boundary_triangles
        ], dtype=int)

        self.boundary_quads = np.array([
            [old_to_new[int(f[0])], old_to_new[int(f[1])],
            old_to_new[int(f[2])], old_to_new[int(f[3])], int(f[4])]
            for f in self.boundary_quads
        ], dtype=int)

        # --- Step 3: remove unreferenced nodes if requested ---
        if remove_unreferenced:
            used_ids = set()
            for f in self.boundary_triangles:
                used_ids.update([int(f[0]), int(f[1]), int(f[2])])
            for f in self.boundary_quads:
                used_ids.update([int(f[0]), int(f[1]), int(f[2]), int(f[3])])

            used_ids = sorted(list(used_ids))

            # build mapping old → compact new
            remap = {old: new for new, old in enumerate(used_ids)}
            compact_nodes = [new_nodes[old] for old in used_ids]

            self.boundary_triangles = np.array([
                [remap[int(f[0])], remap[int(f[1])],
                remap[int(f[2])], int(f[3])]
                for f in self.boundary_triangles
            ], dtype=int)

            self.boundary_quads = np.array([
                [remap[int(f[0])], remap[int(f[1])],
                remap[int(f[2])], remap[int(f[3])], int(f[4])]
                for f in self.boundary_quads
            ], dtype=int)

            self.nodes = np.array(compact_nodes, dtype=float)
            self.node_count = len(self.nodes)
        else:
            self.nodes = np.array(new_nodes, dtype=float)
            self.node_count = len(self.nodes)

        # --- Step 4: rebuild node_connections ---
        self.node_connections = {i: [] for i in range(self.node_count)}
        for t in self.boundary_triangles:
            a, b, c = t[:3]
            for u, v in [(a, b), (b, c), (c, a)]:
                if v not in self.node_connections[u]:
                    self.node_connections[u].append(v)
                if u not in self.node_connections[v]:
                    self.node_connections[v].append(u)
        for q in self.boundary_quads:
            a, b, c, d = q[:4]
            for u, v in [(a, b), (b, c), (c, d), (d, a), (a, c), (b, d)]:
                if v not in self.node_connections[u]:
                    self.node_connections[u].append(v)
                if u not in self.node_connections[v]:
                    self.node_connections[v].append(u)

        print(f"[CLEAN] Reduced to {self.node_count} nodes "
                    f"(duplicates removed, {'unused removed' if remove_unreferenced else 'unused kept'}).")


    ## Initialisation Functions
    @classmethod
    def fromMesh(cls, mesh, ff_range=None):
        import numpy as np
        from collections import defaultdict

        ff = cls()

        ff_nodes = []
        ff_bt_n3 = []
        ff_bq_n4 = []

        # 1) Collect all unique surface node IDs
        g_ids = set()
        for bt in mesh.boundary_triangles:
            g_ids.update([bt[0], bt[1], bt[2]])
        for bq in mesh.boundary_quads:
            g_ids.update([bq[0], bq[1], bq[2], bq[3]])
        g_ids = sorted(g_ids)  # Ensure consistent ordering

        # 2) Map global IDs to new local surface node indices
        gl_to_lc = {}
        for g_id in g_ids:
            gl_to_lc[g_id] = len(ff_nodes)
            ff_nodes.append(mesh.nodes[g_id])
        ff.node_count = len(ff_nodes)

        # 3) Remap triangles and quads to local surface node space
        for bt in mesh.boundary_triangles:
            bt_new = [gl_to_lc[bt[0]], gl_to_lc[bt[1]], gl_to_lc[bt[2]], bt[3]]
            ff_bt_n3.append(bt_new)
        ff.boundary_triangle_count = len(ff_bt_n3)

        for bq in mesh.boundary_quads:
            bq_new = [gl_to_lc[bq[0]], gl_to_lc[bq[1]], gl_to_lc[bq[2]], gl_to_lc[bq[3]], bq[4]]
            ff_bq_n4.append(bq_new)
        ff.boundary_quad_count = len(ff_bq_n4)

        # 4) Assign data arrays
        ff.nodes = np.zeros((ff.node_count, 3))
        ff.boundary_triangles = np.zeros((ff.boundary_triangle_count, 4), dtype=int)
        ff.boundary_quads = np.zeros((ff.boundary_quad_count, 5), dtype=int)

        ff.boundary_surface_ids_n_3 = np.zeros(ff.boundary_triangle_count, dtype=int)
        ff.boundary_surface_ids_n_4 = np.zeros(ff.boundary_quad_count, dtype=int)

        for i, n in enumerate(ff_nodes):
            ff.nodes[i] = n

        for i, bt in enumerate(ff_bt_n3):
            ff.boundary_triangles[i, :3] = bt[:3]
            ff.boundary_triangles[i, 3] = bt[3]
            ff.boundary_surface_ids_n_3[i] = int(bt[3])

        for i, bq in enumerate(ff_bq_n4):
            ff.boundary_quads[i, :4] = bq[:4]
            ff.boundary_quads[i, 4] = bq[4]
            ff.boundary_surface_ids_n_4[i] = int(bq[4])

        # 5) Original face-based node connections
        ff.node_connections = {i: [] for i in range(ff.node_count)}
        for bt in ff.boundary_triangles:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = bt[i], bt[j]
                    if b not in ff.node_connections[a]:
                        ff.node_connections[a].append(b)
                    if a not in ff.node_connections[b]:
                        ff.node_connections[b].append(a)
        for bq in ff.boundary_quads:
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = bq[i], bq[j]
                    if b not in ff.node_connections[a]:
                        ff.node_connections[a].append(b)
                    if a not in ff.node_connections[b]:
                        ff.node_connections[b].append(a)

        # 6) Add spatially-aware cross-surface connections
        ff.set_node_connections()

        print(f"[✔] FroFile constructed: nodes={ff.node_count}, triangles={ff.boundary_triangle_count}, quads={ff.boundary_quad_count}")
        return ff

    @classmethod
    def fromFile(cls, filepath):
        ff = cls()
        ff.read_file(filepath)
        return ff
    
    @classmethod
    def fromFolder(cls, folderpath):
        """
            given a path to an MFIRST morphing folder, loads a FroFile and any section definitions defined.

            folder path MUST
        """
        folderpath = os.path.normpath(folderpath.rstrip("/\\"))
        folder = folderpath
        filename = os.path.split(folderpath)[-1]
        
        ff = cls.fromFile(f"{folder}{os.sep}{filename}.fro")

        remaining = []
        allocated = []

        for fn in os.listdir(folder):
            if fn == "MorphInfo.json":
                with open(f"{folder}{os.sep}{fn}", "r") as f:
                    d = json.load(f)
                    farfield_surface_ids = d["farfield"]
                    ff.setupFarfield(farfield_surface_ids)
                    for section in d["sections"]:
                        section_name, section_surface_ids = list(section.keys())[0], list(section.values())[0]
                        ff.add_section(section_name, section_surface_ids)
                        allocated.extend(section_surface_ids)
        
        for s in range(1,ff.surface_count+1):
            if s not in allocated:
                remaining.append(s)

        ff.add_section("undefined", remaining)

        return ff

    '''def remap_duplicate_node_ids(self, tol=1e-8):
        """
        Remaps spatially duplicate nodes to share a common global ID.
        This preserves the full node list but ensures consistent connectivity across surfaces.
        """
        import numpy as np
        from collections import defaultdict
        from scipy.spatial import KDTree
        import numpy as np

        nodes = np.array(self.nodes)
        kdtree = KDTree(nodes)
        canonical_ids = list(range(len(nodes)))

        for i in range(len(nodes)):
            if canonical_ids[i] != i:
                continue  # Already remapped
            idxs = kdtree.query_ball_point(nodes[i], tol)
            for j in idxs:
                if j != i:
                    canonical_ids[j] = i

        # Remap triangle and quad references
        for i in range(self.boundary_triangle_count):
            for j in range(3):
                self.boundary_triangles[i][j] = canonical_ids[int(self.boundary_triangles[i][j])]

        for i in range(self.boundary_quad_count):
            for j in range(4):
                self.boundary_quads[i][j] = canonical_ids[int(self.boundary_quads[i][j])]

        # Expand node array so that g_id references are valid
        expanded_nodes = np.zeros((len(canonical_ids), 3))
        for old_id, new_id in enumerate(canonical_ids):
            expanded_nodes[old_id] = self.nodes[new_id]
        self.nodes = expanded_nodes
        self.node_count = len(expanded_nodes)

        # Rebuild connections
        self.node_connections = {i: [] for i in range(self.node_count)}
        for t in self.boundary_triangles:
            p1, p2, p3 = t[:-1]
            for a, b in [(p1, p2), (p2, p3), (p3, p1)]:
                if b not in self.node_connections[a]:
                    self.node_connections[a].append(b)
                if a not in self.node_connections[b]:
                    self.node_connections[b].append(a)
        for q in self.boundary_quads:
            p = q[:-1]
            for i in range(4):
                for j in range(i+1, 4):
                    a, b = p[i], p[j]
                    if b not in self.node_connections[a]:
                        self.node_connections[a].append(b)
                    if a not in self.node_connections[b]:
                        self.node_connections[b].append(a)

        print(f"[✔] Remapped duplicate node IDs using KDTree. Total canonical nodes: {len(set(canonical_ids))}")'''
    
    ## Visualisation
    def plot_conditions(self, mm, title, out_folder="", camera_angle=None, save=True):
        """mm : MorphModel"""
        cv = {}
        
        _, verts, f3, f4 = self.get_section_vertices_and_faces("main_body")
        if len(f3) > 0 and len(f4) > 0:
            t_faces = np.concatenate([np.hstack(f3), np.hstack(f4)])
        elif len(f3) > 0:
            t_faces = f3
        elif len(f4) > 0:
            t_faces = f4
        else:
            t_faces = f3
        if len(verts) > 0:
            cv["lightgrey"]   = (verts, t_faces)
        t_gids = mm.get_t_node_gids(self)
        u_gids = mm.get_u_node_gids(self)
        c_gids = mm.get_c_node_gids(self)
        
        # remove any farfield nodes
        if "farfield" in self.sections:
            _, ff_nodes = self.get_surface_nodes("farfield")
            t_gids = list(set(t_gids)-set(ff_nodes))
            u_gids = list(set(u_gids)-set(ff_nodes))
            c_gids = list(set(c_gids)-set(ff_nodes))
            
        
        _, u_verts = self.convert_node_ids_to_coordinates(u_gids)
        if len(u_verts) > 0:
            cv["blue"]  = u_verts
        _, t_verts = self.convert_node_ids_to_coordinates(t_gids)
        if len(t_verts) > 0:
            cv["green"]  = t_verts
        _, c_verts = self.convert_node_ids_to_coordinates(c_gids)
        if len(c_verts) > 0:
            cv["red"]  = c_verts        

    ## Main Functions
    def setupFarfield(self, farfield_ids, radius=None):
        def mag(p1: np.array) -> float:
            return np.sqrt(sum([x**2 for x in p1]))

        def dist_between_points(p1, p2):
            return mag(p2-p1)
        
        # Set farfields
        # we want to add 2 generic sections, one for the farfield and one for the main geometry body
        non_farfield_ids = []
        
        if radius is not None:  # need to work out the farfield, change ids and then setup
            print(f"creating far field surfaces from radius: {radius}") 
            ff_id = self.num_surface_ids+1
            print(f"Num of surfaces = {self.num_surface_ids} so the farfield id is {ff_id}")
            ff_faces = []
            for i, f in enumerate(self.boundary_triangles):
                for n in f[:-1]:
                    r = dist_between_points(self.nodes[n], [0.0,0.0,0.0])
                    if r > radius:
                        ff_faces.append(i)
            print(f"{len(ff_faces)} faces in farfield")
            for i in ff_faces:
                self.boundary_triangles[i][-1] = ff_id
            print(f"Num of surfaces is now = {self.num_surface_ids} with farfield id {ff_id}")
            farfield_ids = [ff_id]
        
        self.farfield_ids = farfield_ids

        if len(farfield_ids) > 0: # farfield ids are already defined
            self.add_section("farfield", farfield_ids)
        
        for x in range(1, self.num_surface_ids+1):
            if x not in farfield_ids:
                non_farfield_ids.append(x)
        self.add_section("main_body", non_farfield_ids)
    
    def copy(self):
        """return a deepcopy of the frofile"""
        # If original has no connections, build them (adaptive tol)
        if not hasattr(self, "node_connections") or not self.node_connections:
            self.set_node_connections(self)

        # Build the copy from the mesh state
        other = FroFile.fromMesh(self)

        # Always rebuild adjacency on the copy with adaptive tolerance
        other.set_node_connections()      # <-- crucial

        # copy metadata / sections
        other.everything_else   = self.everything_else[:]
        other.sections          = self.sections.copy()
        other.section_faces_n_3 = self.section_faces_n_3.copy()
        other.section_faces_n_4 = self.section_faces_n_4.copy()
        other.section_vertices  = self.section_vertices.copy()
        other.section_nodes     = self.section_nodes.copy()
        other.section_bounds    = self.section_bounds.copy()
        return other

    
    def get_surface_ids(self):
        import numpy as np
        ids = set()

        # Prefer reading SIDs directly from faces (works regardless of the aux containers' types)
        bt = getattr(self, "boundary_triangles", None)
        if bt is not None:
            try:
                bt_arr = np.asarray(bt)
                if bt_arr.size and bt_arr.shape[1] >= 4:
                    ids.update(bt_arr[:, 3].astype(int).tolist())
            except Exception:
                # list-of-lists fallback
                for row in bt or []:
                    if len(row) >= 4:
                        ids.add(int(row[3]))

        bq = getattr(self, "boundary_quads", None)
        if bq is not None:
            try:
                bq_arr = np.asarray(bq)
                if bq_arr.size and bq_arr.shape[1] >= 5:
                    ids.update(bq_arr[:, 4].astype(int).tolist())
            except Exception:
                for row in bq or []:
                    if len(row) >= 5:
                        ids.add(int(row[4]))

        # Fallback to the auxiliary containers if face arrays were empty
        if not ids:
            v3 = getattr(self, "boundary_surface_ids_n_3", None)
            if v3 is not None:
                if isinstance(v3, dict):
                    ids.update(int(x) for x in v3.values())
                else:
                    ids.update(np.asarray(v3).astype(int).tolist())

            v4 = getattr(self, "boundary_surface_ids_n_4", None)
            if v4 is not None:
                if isinstance(v4, dict):
                    ids.update(int(x) for x in v4.values())
                else:
                    ids.update(np.asarray(v4).astype(int).tolist())

        return sorted(ids)

    
    def update_debug_plot(self, ff, exclude_ids=[1, 2], title="Debug Mesh"):
        self.plot_ax.clear()
        colors = plt.cm.get_cmap('tab20', len(ff.get_surface_ids()))

        for i, sid in enumerate(ff.get_surface_ids()):
            if sid in exclude_ids:
                continue

            g_ids, _ = ff.get_surface_nodes(sid)
            points = ff.nodes[np.array(g_ids, dtype=int)]
            if len(points) == 0:
                continue

            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            self.plot_ax.scatter(x, y, z, s=2, label=f"Surface {sid}", color=colors(i))

        self.plot_ax.set_title(title)
        self.plot_ax.set_xlabel("X")
        self.plot_ax.set_ylabel("Y")
        self.plot_ax.set_zlabel("Z")
        self.plot_ax.legend(loc="upper right", fontsize='x-small')
        self.plot_ax.grid(True)

        self.canvas.draw()