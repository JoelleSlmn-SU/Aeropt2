import struct
import numpy as np
import os, sys

sys.path.append(os.path.dirname("FileRW"))
from FileRW.Mesh import Mesh
from FileRW.FortranBinaryFile import FortranBinaryFile
from Utilities.DataType import DataType
from FileRW.FliteFile import FliteFile
from enum import Enum
class PltFileType(Enum):
    TET = 0
    HYBRID = 1
    UNDETERMINED = 2

class Plt(Mesh):

    ## Mesh Interface
    def write_file(self, filename: str, mesh=None) -> int:
        if not mesh:
            mesh = self

        with FortranBinaryFile(filename) as f:
            # Write Out Meta Data as a single record (bytecount, data, bytecount)
            f.write_int(40                           )
            f.write_int(mesh.total_element_count     )
            f.write_int(mesh.node_count              )
            f.write_int(mesh.boundary_total_count    )
            f.write_int(mesh.hex_count               )
            f.write_int(mesh.prism_count             )
            f.write_int(mesh.pyramid_count           )
            f.write_int(mesh.tet_count               )
            f.write_int(mesh.boundary_quad_count     )
            f.write_int(mesh.boundary_triangle_count )
            f.write_int(mesh.number_of_edges         )
            f.write_int(40                           )
            
            # Write out volume element connectivity matrices
            f.write_record(mesh.hex,      DataType.INTEGER, python_indexed=True)
            f.write_record(mesh.prism,    DataType.INTEGER, python_indexed=True)
            f.write_record(mesh.pyramids, DataType.INTEGER, python_indexed=True)
            f.write_record(mesh.tets,     DataType.INTEGER, python_indexed=True)

            # Write out nodes
            f.write_record(mesh.nodes,    DataType.DOUBLE)

            # Write out boundary elements with surface ids
            # surface ids for Mesh are stored seperately from element array,
            # but plt files store them together, so need to include them.
            boundary_tris = []
            # flite stores both tri and quad boundary faces in an nx5 array, so need
            # to combine both into one array here and write them out.
            for i, tri in enumerate(mesh.boundary_triangles):
                tri.append(1) # dummy variable at index 4 for tris - unused by preprocessor.
                tri.append(mesh.boundary_surface_ids[i]-1)
                boundary_tris.append(tri)

            boundary_quads = []
            for i, quad in enumerate(mesh.boundary_quads):
                i += mesh.boundary_triangle_count
                quad.append(mesh.boundary_surface_ids[i]-1)
                boundary_quads.append(quad)

            self.write_record(boundary_quads, DataType.DOUBLE)
            self.write_record(boundary_tris,  DataType.DOUBLE)

        return 0

    def read_file(self, filename: str, mesh=None):
        print(f"plt file: {filename}")
        if mesh == None:
            mesh = self 
        plt = FortranBinaryFile(filename)
        #plt.open()
        #plt.f.seek(0)

        #header = [x[0] for x in plt.read_record(0, (10,1), 4)]
        header = plt.read_record_auto([10,1], 4)
        _, self.node_count, _, self.hex_count, self.prism_count, self.pyramid_count, self.tet_count, self.boundary_quad_count, self.boundary_triangle_count, self.num_of_edges = header
        print(self)
        print("Reading hex")
        self.hex                = plt.read_record(hex_offset,      (8, self.hex_count),               step = 4).T
        print("Reading prism")
        self.prism              = plt.read_record(prism_offset,    (6, self.prism_count),             step = 4).T
        print("Reading pyramids")
        self.pyramids           = plt.read_record(pyramid_offset,  (5, self.pyramid_count),           step = 4).T
        print("Reading tets")
        self.tets               = plt.read_record(tet_offset,      (4, self.tet_count),               step = 4).T
        print("Reading nodes")
        self.nodes              = plt.read_record(node_offset,     (3, self.node_count),              step = 8).T
        print("Reading boundary_quads")
        boundary_quads          = plt.read_record(quad_offset,     (5, self.boundary_quad_count),     step = 4).T
        print("Reading boundary_triangles")
        boundary_triangles      = plt.read_record(triangle_offset, (5, self.boundary_triangle_count), step = 4).T
        
        surf_num = 0
        for i in range(len(boundary_quads)):
            #boundary_quads[i][0] -= 1 # id
            p1 = boundary_quads[i][0] - 1 # p1
            p2 = boundary_quads[i][1] - 1 # p2
            p3 = boundary_quads[i][2] - 1 # p3
            p4 = boundary_quads[i][3] - 1 # p4
            surf = boundary_quads[i][4]
            surf_num = max([surf_num, surf])
            self.boundary_surface_ids_n_4[i] = surf
            self.boundary_quads.append([p1, p2, p3, p4, surf])
        for i in range(len(boundary_triangles)):
            #print(boundary_triangles[i])
            #boundary_triangles[i][0] -= 1 # id
            p1 = boundary_triangles[i][0] - 1 # p1
            p2 = boundary_triangles[i][1] - 1 # p2
            p3 = boundary_triangles[i][2] - 1 # p3
            surf = boundary_triangles[i][4] # surf?
            surf_num = max([surf_num, surf])
            self.boundary_surface_ids_n_3[i] = surf
            self.boundary_triangles.append([p1, p2, p3, surf])
        self.surface_count = surf_num
        print(self)

    def determine_filetype(self, header):
        return PltFileType.UNDETERMINED
    
    def unpack_tet_header(self, header):
        pass
    
    def unpack_hyb_header(self, header):
        """
            description

            Parameters
            ----------
            header : str
                description

            Returns
            -------

        """
        pass

    def section_offsets(self):
        """
            Calculates all the section offsets from the start of the
            file so we know where to start reading a section from

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        self.hex_offset      = 48
        self.prism_offset    = FortranBinaryFile.calc_offset(self.hex_offset     , self.hex_count           , 8, 4)
        self.pyramid_offset  = FortranBinaryFile.calc_offset(self.prism_offset   , self.prism_count         , 6, 4)
        self.tet_offset      = FortranBinaryFile.calc_offset(self.pyramid_offset , self.pyramid_count       , 5, 4)
        self.node_offset     = FortranBinaryFile.calc_offset(self.tet_offset     , self.tet_count           , 4, 4)
        self.quad_offset     = FortranBinaryFile.calc_offset(self.node_offset    , self.node_count          , 3, 8)
        self.triangle_offset = FortranBinaryFile.calc_offset(self.quad_offset    , self.boundary_quad_count , 5, 4)

    ## Main Functions
    def lazy_read(self, filename: str, mesh=None):
        if mesh == None:
            mesh = self 
        plt = FortranBinaryFile(filename)# open(filename, 'rb')
        plt.open()
        plt.f.seek(0)

        header = [x[0] for x in plt.read_record(0, (10,1), 4)]
        tot_vol_elem_count, self.node_count, tot_surf_elem_count, self.hex_count, self.prism_count, self.pyramid_count, self.tet_count, self.boundary_quad_count, self.boundary_triangle_count, self.num_of_edges = header

    def extract_fro_file(self, out_filename, mesh=None):
        from FileRW.FroFile import FroFile
        
        if mesh == None:
            mesh = self
        ff = FroFile.fromMesh(mesh)
        return ff
        node_ids = []
        gl_to_lc = {}
        lc_to_gl = {}
        vertices = []
        bq_new   = [] # self.boundary_quads.copy()
        bt_new   = [] # self.boundary_triangles.copy()

        # get list of node ids and vertices array
        for bq in self.boundary_quads:
            for p in bq[:-1]:
                node_ids.append(int(p))
        for bt in self.boundary_triangles:
            for p in bt[:-2]:
                node_ids.append(int(p))
        
        node_ids = list(set(node_ids))

        for i, n in enumerate(node_ids):
            gl_to_lc[n] = i
            lc_to_gl[i] = n
            v = self.nodes[n]
            vertices.append(np.array([v[0], v[1], v[2]]))
        
        num_surfaces = 0
        # fix quads
        for i in range(len(self.boundary_quads)):
            s_id = self.boundary_quads[i][4]
            p1 = gl_to_lc[self.boundary_quads[i][0]]
            p2 = gl_to_lc[self.boundary_quads[i][1]]
            p3 = gl_to_lc[self.boundary_quads[i][2]]
            p4 = gl_to_lc[self.boundary_quads[i][3]]
            bq_new.append(np.array([p1, p2, p3, p4, s_id]))
            num_surfaces = max(num_surfaces, s_id)

        # fix tris
        for i in range(len(self.boundary_triangles)):
            s_id = self.boundary_triangles[i][3]
            p1 = gl_to_lc[self.boundary_triangles[i][0]]
            p2 = gl_to_lc[self.boundary_triangles[i][1]]
            p3 = gl_to_lc[self.boundary_triangles[i][2]]
            bt_new.append(np.array([p1, p2, p3, s_id]))
            num_surfaces = max(num_surfaces, s_id)

        ff = FroFile()
        ff.nodes              = np.array(vertices) # Nx3
        ff.boundary_triangles = np.array(bt_new)   # Nx4 - p1, p2, p3, surf      # tria3
        ff.boundary_quads     = np.array(bq_new)   # Nx5 - p1, p2, p3, p4, surf  # quad4
        
        #ff.num_surfaces            = num_surfaces
        ff.boundary_triangle_count = len(bt_new)
        ff.boundary_quad_count     = len(bq_new)
        ff.node_count              = len(vertices)
        ff.num_segments            = num_surfaces

        ff.write_file(out_filename)
        return ff

    def validate(self, filename):
        def read_record(self, init_offset):
            pos, data = init_offset, 0
            self.f.seek(pos)
            prefix = struct.unpack( '<i', self.f.read(4))[0]
            if prefix < 0:
                suffix = 1
                data = 0
                while suffix > 0:
                    data += abs(prefix) # replace with array data read
                    pos += 4+abs(prefix)
                    self.f.seek(pos)
                    suffix = struct.unpack( '<i', self.f.read(4))[0]
                    pos += 4
                    if suffix > 0:
                        prefix = struct.unpack( '<i', self.f.read(4))[0]
            else:
                pos += 4+prefix
                self.f.seek(pos)
                data = prefix # replace with array data read
                suffix = struct.unpack( '<i', self.f.read(4))[0]
                pos += 4
            return data, pos
        plt = FortranBinaryFile(filename)
        plt.f.seek(0)

        header = [x[0] for x in plt.read_record(0, 48, (10,1), 4)]
        _, self.node_count, _, self.hex_count, self.prism_count, self.pyramid_count, self.tet_count, self.boundary_quad_count, self.boundary_triangle_count, self.num_of_edges = header

        pos = 0
        plt.f.seek(pos)

        data_len, pos = read_record(plt, pos)
        print(f"Header - {data_len}/40")
        data_len, pos = read_record(plt, pos)
        print(f"Hexa   - {data_len}/0")
        data_len, pos = read_record(plt, pos)
        print(f"Prism  - {data_len}/{6*4*self.prism_count}")
        data_len, pos = read_record(plt, pos)
        print(f"Pyramd - {data_len}/{5*4*self.pyramid_count}")
        data_len, pos = read_record(plt, pos)
        print(f"Tetra  - {data_len}/{4*4*self.tet_count}")
        data_len, pos = read_record(plt, pos)
        print(f"Node   - {data_len}/{3*8*self.node_count}")
        data_len, pos = read_record(plt, pos)
        print(f"Quads  - {data_len}/{5*4*self.boundary_quad_count}")
        data_len, pos = read_record(plt, pos)
        print(f"Tris   - {data_len}/{5*4*self.boundary_triangle_count}")
        if len(plt.f.read(4)) == 0:
            print(f"End of file reached - final pos, ie len, = {pos}/3299275280 (err={pos-3299275280})")

if __name__ == "__main__":
    fn = FliteFile().getFileExtOptions("plt", exc_p=["_v1","_v2"])
    #fn = "/Users/bensmith/MFIRST/dev/Data/perfect.plt"
    plt = Plt()
    plt.lazy_read(fn)
    print(plt)
    #plt.read_file(fn)
    #print(plt)
    #plt.tet_count = 134217748
    #plt.write_record2(plt.tet_count,     plt.tets,     4, None)