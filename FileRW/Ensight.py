import os
import sys
import numpy as np

sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("Utilities"))
from FileRW.Mesh import Mesh
from FileRW.FortranBinaryFile import FortranBinaryFile, DataType
from Utilities.Paths import concat_fn_to_fp

class EnsightCase:
    """
        Attributes
        ----------
        case_name : str
            case name. appended to all file names and used to identify this case against any other locally stored ensight files
        file_path : str
            directory to save to. 
        case_file : CaseFile
            case file object that writes out the .case file that stores which variables make up the case.
        geo_file  : GeoFile
            geo file object that writes the geometry structure to the .geo file. contains information about all nodes, surfaces, boundarys etc.
        scalars_per_node : dict
            dictionary used to store the scalar values for each scalar variable for each node
            key : str  
                name of the scalar value eg density, spacing etc
            val : list 
                list of the scalar values at each node. list index == g_id of the node
        scalars_per_elem : dict
            dictionary used to store the scalar values for each scalar variable for each type of element
            key : str  
                name of the scalar value eg density, spacing etc
            val : dict 
                    key : str
                        name of the ensight 'element type' eg tetra4, tria3 etc
                    val : list
                        list of the scalar values at each of these elements. list index == g_id of the element

        vectors : dict 
            dictionary used to store the vector values for each vector variable for each node
            ngl. idk
    """
    def __init__(self, case_name, file_path, mesh:Mesh=None):
        """
            Constructs an ensight case

            Parameters
            ----------
            case_name : str
                case name. appended to all file names and used to identify this case against any other locally stored ensight files
            file_path : str
                file path to store everything in
            mesh : Mesh
                contains the actual mesh. can come from Fro, Plt, or Ugrid. anything that stores its data in the Mesh interface format. 

            Returns
            -------
            None
        """
        self.case_name = case_name
        self.file_path = file_path
        #if not os.path.exists(file_path):
        #    os.makedirs(file_path)
        self.case_file = CaseFile(case_name, file_path)
        self.geo_file  = GeoFile(case_name,  file_path, mesh)
        
        self.case_file.geometry_file = self.geo_file.filename
        self.scalars_per_node = {}
        self.scalars_per_elem = {}
        self.vectors_per_node = {}

    def write(self):
        from time import time
        start = time()
        print("[Ensight] Starting GEO write...")

        self.open()
        self.write_string("Fortran Binary", 80)
        self.write_string("ENSIGHT", 80)
        self.write_string("GEO FILE", 80)
        self.write_string("node id assign", 80)
        self.write_string("element id assign", 80)

        print(f"[Ensight] Wrote header in {time() - start:.2f} s")

        pc = 1

        # External Flow Field
        if self.mesh.total_element_count > 0:
            t0 = time()
            self.write_string("part", 80)
            self.write_record([pc], dt=DataType.INTEGER)
            self.write_string("External Flowfield", 80)
            self.write_string("coordinates", 80)
            self.write_record([self.mesh.node_count], dt=DataType.INTEGER)
            nodes = np.array(self.mesh.nodes).T
            self.write_record(nodes[0], dt=DataType.FLOAT)
            self.write_record(nodes[1], dt=DataType.FLOAT)
            self.write_record(nodes[2], dt=DataType.FLOAT)

            print(f"[Ensight] Wrote external field (nodes: {self.mesh.node_count}) in {time() - t0:.2f} s")

            if self.mesh.hex_count > 0:
                self.write_string("hexa8", 80)
                self.write_record([self.mesh.hex_count], dt=DataType.INTEGER)
                self.write_record(self.mesh.hex, python_indexed=False)

            if self.mesh.prism_count > 0:
                self.write_string("penta6", 80)
                self.write_record([self.mesh.prism_count], dt=DataType.INTEGER)
                self.write_record(self.mesh.prism, python_indexed=False)

            if self.mesh.pyramid_count > 0:
                self.write_string("pyramid5", 80)
                self.write_record([self.mesh.pyramid_count], dt=DataType.INTEGER)
                self.write_record(self.mesh.pyramids, python_indexed=False)

            if self.mesh.tet_count > 0:
                self.write_string("tetra4", 80)
                self.write_record([self.mesh.tet_count], dt=DataType.INTEGER)
                self.write_record(self.mesh.tets, python_indexed=False)
            pc += 1

        # Surfaces
        num_surfaces = self.mesh.num_surface_ids
        print(f"[Ensight] Preparing to write {num_surfaces} surfaces...")

        for isurf in range(1, num_surfaces + 1):
            t1 = time()
            ng_lc, nodes = self.mesh.get_surface_nodes(isurf)
            _, surf_tris = self.mesh.get_surface_tria3(isurf)
            _, surf_quads = self.mesh.get_surface_quad4(isurf)
            _, coordinates = self.mesh.convert_node_ids_to_coordinates(nodes)

            self.write_string("part", 80)
            self.write_record([pc], dt=DataType.INTEGER)
            self.write_string(f"Surface {str(isurf): >4}", 80)
            self.write_string("coordinates", 80)
            self.write_record([len(nodes)], dt=DataType.INTEGER)
            self.write_record(coordinates.T[0], dt=DataType.FLOAT)
            self.write_record(coordinates.T[1], dt=DataType.FLOAT)
            self.write_record(coordinates.T[2], dt=DataType.FLOAT)

            if len(surf_tris) > 0:
                self.write_string("tria3", 80)
                self.write_record([len(surf_tris)], dt=DataType.INTEGER)
                tris = [[ng_lc[g_id] for g_id in self.mesh.get_tria3_nodes(t)] for t in surf_tris]
                self.write_record(tris, dt=DataType.INTEGER, python_indexed=True)

            if len(surf_quads) > 0:
                self.write_string("quad4", 80)
                self.write_record([len(surf_quads)], dt=DataType.INTEGER)
                quads = [[ng_lc[g_id] for g_id in self.mesh.get_quad4_nodes(q)] for q in surf_quads]
                self.write_record(quads, dt=DataType.INTEGER, python_indexed=True)

            print(f"[Ensight] Wrote surface {isurf} with {len(nodes)} nodes in {time() - t1:.2f} s")
            pc += 1

        self.close()
        print(f"[Ensight] GEO write complete in {time() - start:.2f} s")


    def read(self):
        pass

class CaseFile:
    def __init__(self, case_name, file_path=None):
        self.case_name = case_name
        self.file_path = file_path

        self.geometry_file = None
        self.scalar_fields_per_node = []
        self.scalar_fields_per_elem = []
        self.vector_fields = []
    
    @property
    def full_pathname(self) -> str:
        if self.file_path:
            filename = concat_fn_to_fp(self.file_path, f"ENSIGHT{self.case_name}.case")
        else:
            filename = f"ENSIGHT{self.case_name}.case"
        return filename

    def write(self):
        case_dir = os.path.dirname(self.full_pathname)
        os.makedirs(case_dir, exist_ok=True)

        # Now write
        with open(self.full_pathname, "w") as case_file:
            case_file.write(str(self))

    def read(self):
        case_file          = open(self.full_pathname, "r")
        lines              = case_file.read().split("\n")
        self.geometry_file = lines[4].split(":")[1].strip()

        if len(lines) > 7:
            assert lines[6].strip("\n") == "VARIABLE"
            c = 0
            for l in lines[7:]:
                l = l.split(':')
                c += 1
                if len(l) > 1:
                    t  = l[0]
                    ft = l[1].split(".")[1]
                    if t == "scalar per node":
                        self.scalar_fields_per_node.append(ft)
                    elif t == "scalar per element":
                        self.scalar_fields_per_elem.append(ft)
                    elif t == "vector per node":
                        self.vector_fields.append(ft)
    
    def __str__(self) -> str:
        retval = ""
        retval += f"FORMAT\n"
        retval += f"type:  ensight gold\n"
        retval += f"\n"
        retval += f"GEOMETRY\n"
        retval += f"model: {self.geometry_file}\n"
        if self.scalar_fields_per_node or self.scalar_fields_per_elem or self.vector_fields:
            retval += f"VARIABLE\n"
            for sf in self.scalar_fields_per_node:
                retval += f"scalar per node:  {sf}   ENSIGHT{self.case_name}.{sf}\n"
            for sf in self.scalar_fields_per_elem:
                retval += f"scalar per element:  {sf}   ENSIGHT{self.case_name}.{sf}\n"
            for sf in self.vector_fields:
                retval += f"vector per node:  {sf}   ENSIGHT{self.case_name}.{sf}\n"
        return retval

class GeoFile(FortranBinaryFile):
    def __init__(self, case_name, file_path=None, mesh=None):
        super().__init__(f"ENSIGHT{case_name}.geo", file_path, overwrite=True)
        self.mesh = mesh

    @property
    def full_pathname(self) -> str:
        if self.filepath:
            filename = concat_fn_to_fp(self.filepath, self.filename)
        else:
            filename = f"{self.filename}"
        return filename

    def write(self):
        self.open()
        self.write_string("Fortran Binary",                  80)
        self.write_string("ENSIGHT",                         80)
        self.write_string("GEO FILE",                        80)
        self.write_string("node id assign",                  80)
        self.write_string("element id assign",               80)

        pc = 1
        # External Flow Field
        if self.mesh.total_element_count > 0:
            self.write_string("part",                               80)
            self.write_record([pc], dt=DataType.INTEGER)
            self.write_string("External Flowfield",                 80)
            self.write_string("coordinates",                        80)
            self.write_record([self.mesh.node_count], dt=DataType.INTEGER)
            nodes = np.array(self.mesh.nodes).T
            self.write_record(nodes[0], dt=DataType.FLOAT)
            self.write_record(nodes[1], dt=DataType.FLOAT)
            self.write_record(nodes[2], dt=DataType.FLOAT)
            
            if self.mesh.hex_count > 0:
                self.write_string("hexa8",                          80)
                self.write_record([self.mesh.hex_count], dt=DataType.INTEGER)
                self.write_record(self.mesh.hex, python_indexed=False)

            if self.mesh.prism_count > 0:
                self.write_string("penta6",                          80)
                self.write_record([self.mesh.prism_count], dt=DataType.INTEGER)
                self.write_record(self.mesh.prism, python_indexed=False)

            if self.mesh.pyramid_count > 0:
                self.write_string("pyramid5",                        80)
                self.write_record([self.mesh.pyramid_count], dt=DataType.INTEGER)
                self.write_record(self.mesh.pyramids, python_indexed=False)

            if self.mesh.tet_count > 0:
                self.write_string("tetra4",                          80)
                self.write_record([self.mesh.tet_count], dt=DataType.INTEGER)
                self.write_record(self.mesh.tets, python_indexed=False)
            pc += 1

        # Surfaces
        for isurf in range(1, self.mesh.num_surface_ids+1):
            ng_lc, nodes   = self.mesh.get_surface_nodes(isurf)
            _, surf_tris   = self.mesh.get_surface_tria3(isurf)
            _, surf_quads  = self.mesh.get_surface_quad4(isurf)
            _, coordinates = self.mesh.convert_node_ids_to_coordinates(nodes)
            self.write_string("part",                    80)
            self.write_record([pc], dt=DataType.INTEGER)
            self.write_string(f"Surface {str(isurf): >4}",80)
            self.write_string("coordinates",             80)
            self.write_record([len(nodes)], dt=DataType.INTEGER)
            self.write_record(coordinates.T[0], dt=DataType.FLOAT)
            self.write_record(coordinates.T[1], dt=DataType.FLOAT)
            self.write_record(coordinates.T[2], dt=DataType.FLOAT)
            if len(surf_tris) > 0:
                self.write_string("tria3",                   80)
                self.write_record([len(surf_tris)], dt=DataType.INTEGER)
                tris = []
                for t in surf_tris:
                    tris.append([ng_lc[g_id] for g_id in self.mesh.get_tria3_nodes(t)])
                self.write_record(tris, dt=DataType.INTEGER, python_indexed=True)

            if len(surf_quads) > 0:
                self.write_string("quad4",                   80)
                self.write_record([len(surf_quads)], dt=DataType.INTEGER)
                quads = []
                for q in surf_quads:
                    quads.append([ng_lc[g_id] for g_id in self.mesh.get_quad4_nodes(q)])
                self.write_record(quads, dt=DataType.INTEGER, python_indexed=True)

            pc += 1

    #def read

class ScalarFilePerNode(FortranBinaryFile):
    def __init__(self, field_name, case_name, file_path=None, vals=None, gf=None, overwrite=False):
        """
            Attributes
            -------
            field_name : str
                name of the field type, eg density, spacing etc. also used to define file type.
            case_name : str
                name of the ensight case, ie the filename
            file_path : str
                path to save the ensight files in
            vals : list
                list of values for each node. list index should match global id
            gf : GeoFile
                the underlying geo file containing all the mesh data including surfaces etc.

        """
        super().__init__(f"ENSIGHT{case_name}.{field_name}", file_path, overwrite=overwrite)
        self.field_name = field_name
        self.case_name  = case_name
        self.header = f"{field_name} Variable File"
        self.vals = vals
        self.sections = None
        self.gf = gf

    @property
    def full_pathname(self) -> str:
        if self.filepath:
            filename = concat_fn_to_fp(self.filepath, self.filename)
        else:
            filename = f"{self.filename}"
        return filename

    def write(self, overwrite=False):
        ovr = self.overwrite
        self.overwrite = overwrite
        if self.sections is None:
            self.write_from_geo_file()
        else:
            self.write_from_sections()
        self.overwrite = ovr
    
    def write_from_sections(self):
        self.open()
        self.write_string(self.header, 80)
        for part_id, sf in self.sections.items():
            self.write_string("part",        80)
            self.write_record([part_id], dt=DataType.INTEGER)
            part_id += 1
            self.write_string("coordinates", 80)
            self.write_record(sf, dt=DataType.FLOAT)
        self.close()

    def write_from_geo_file(self):
        self.open()
        part_id = 1
        self.write_string(self.header, 80)
        if self.gf.mesh.total_element_count > 0:
            self.write_string("part",        80)
            self.write_record([part_id], dt=DataType.INTEGER)
            part_id += 1
            self.write_string("coordinates", 80)
            self.write_record(self.vals, dt=DataType.FLOAT)
        # loop through each surface and output 
        for isurf in range(1, self.gf.mesh.num_surface_ids+1):
            g_ids = []

            for triangle in self.gf.mesh.boundary_triangles:
                if triangle[3] == isurf:
                    for g_id in triangle[0:3]:
                        g_ids.append(g_id)

            for quad in self.gf.mesh.boundary_quads:
                if quad[4] == isurf:
                    for g_id in quad[0:4]:
                        g_ids.append(g_id)

            g_ids = list(set(g_ids))
            sf = []
            for g_id in g_ids:
                try:
                    sf.append(self.vals[g_id])
                except IndexError:
                    print(f"gid = {g_id}, vals = {len(self.vals)}")
                    raise
            self.write_string("part",        80)
            self.write_record([part_id], dt=DataType.INTEGER)
            part_id += 1
            self.write_string("coordinates", 80)
            self.write_record(sf, dt=DataType.FLOAT)
        self.close()

    def read(self):
        self.open()
        self.header = self.read_string()[0]
        #print(self.header)
        parts = {}
        while not self.eof:
            part = self.read_string()[0]
            part_num = self.read_record_auto(1)[0]
            #print(f"reading {part.strip()} {part_num}")
            coord = self.read_string()[0]
            #print(coord)
            part_coords = self.read_record_auto((-1), DataType.FLOAT)
            #print(len(part_coords))
            parts[part_num] = part_coords
        self.sections = parts
        self.close()
        

class ScalarFilePerElement(FortranBinaryFile):
    def __init__(self, field_name, case_name, file_path=None, vals=None, gf=None, overwrite=True):

        """
            Attributes
            -------
            field_name : str
                name of the field type, eg density, spacing etc. also used to define file type.
            case_name : str
                name of the ensight case, ie the filename
            file_path : str
                path to save the ensight files in
            vals : dict
                key: element type (tria3, quad4, tetra4, hex8, penta6 etc)
                val: scalar value at the centroid of that element
            gf : GeoFile
                the underlying geo file containing all the mesh data including surfaces etc.

        """
        super().__init__(f"ENSIGHT{case_name}.{field_name}", file_path, overwrite=overwrite)
        self.field_name = field_name
        self.case_name  = case_name
        self.header = f"{field_name} Variable File"
        self.vals = vals
        self.gf = gf

    @property
    def full_pathname(self) -> str:
        if self.filepath:
            filename = concat_fn_to_fp(self.filepath, self.filename)
        else:
            filename = f"{self.filename}"
        return filename

    def write(self):
        self.open()
        part_id = 1
        self.write_string(self.header, 80)
        if self.gf.mesh.total_element_count > 0:
            self.write_string("part",        80)
            self.write_record([part_id], dt=DataType.INTEGER)
            part_id += 1
            for k,v in self.vals.items():
                self.write_string(f"{k}", 80)
                self.write_record(v, dt=DataType.FLOAT)

        # loop through each surface and output 
        for isurf in range(1, self.gf.mesh.num_surface_ids+1):
            temp = {}
            for k, v in self.vals.items():
                g_ids = []

                for i, face in enumerate(self.gf.mesh.get_element_by_ensight_id(k)):
                    if face[-1] == isurf:
                        g_ids.append(i) # TODO - check for duplicates
                sf = []
                for g_id in g_ids:
                    try:
                        sf.append(v[g_id])
                    except IndexError:
                        raise
                if len(sf) > 0:
                    temp[k] = sf
                
            # output the part
            if len(temp.keys()) > 0:
                self.write_string("part",        80)
                self.write_record([isurf], dt=DataType.INTEGER)
                part_id += 1
                for k, sf in temp.items():
                    if len(sf) > 0:
                        self.write_string(f"{k}", 80)
                        self.write_record(sf, dt=DataType.FLOAT)

class VectorFilePerNode(FortranBinaryFile):
    def __init__(self, field_name, case_name, file_path=None, vals=None, gf=None, overwrite=False):
        """
            Attributes
            -------
            field_name : str
                name of the field type, eg density, spacing etc. also used to define file type.
            case_name : str
                name of the ensight case, ie the filename
            file_path : str
                path to save the ensight files in
            vals : list
                list of values for each node. list index should match global id
            gf : GeoFile
                the underlying geo file containing all the mesh data including surfaces etc.

        """
        super().__init__(f"ENSIGHT{case_name}.{field_name}", file_path, overwrite=overwrite)
        self.field_name = field_name
        self.case_name  = case_name
        self.header = f"{field_name} Variable File"
        self.vals = vals
        self.gf = gf

    @property
    def full_pathname(self) -> str:
        if self.filepath:
            filename = concat_fn_to_fp(self.filepath, self.filename)
        else:
            filename = f"{self.filename}"
        return filename

    def write(self):
        self.open()
        part_id = 1
        self.write_string(self.header, 80)
        if self.gf.mesh.total_element_count > 0:
            self.write_string("part",        80)
            self.write_record([part_id], dt=DataType.INTEGER)
            part_id += 1
            self.write_string("coordinates", 80)
            self.write_record(self.vals, dt=DataType.FLOAT)
        # loop through each surface and output 
        for isurf in range(1, self.gf.mesh.num_surface_ids+1):
            g_ids = []
            surf_tris = []
            lc_id = 0

            for triangle in self.gf.mesh.boundary_triangles:
                if triangle[3] == isurf:
                    surf_tris.append([x for x in triangle[0:3]])
                    for g_id in triangle[0:3]:
                        if g_id not in g_ids:
                            g_ids.append(g_id)
                            #gl_to_lc[g_id] = lc_id
            sf = []
            for g_id in g_ids:
                try:
                    sf.append(self.vals[g_id])
                except IndexError:
                    print(f"gid = {g_id}, vals = {len(self.vals)}")
                    raise
            self.write_string("part",        80)
            self.write_record([part_id], dt=DataType.INTEGER)
            part_id += 1
            self.write_string("coordinates", 80)
            self.write_record(sf, dt=DataType.FLOAT)

    def read(self):
        self.open()
        self.header = self.read_string()[0]
        #print(self.header)
        parts = {}
        while not self.eof:
            part = self.read_string()[0]
            part_num = self.read_record_auto(1)[0]
            #print(f"reading {part.strip()} {part_num}")
            coord = self.read_string()[0]
            #print(coord)
            part_coordsx = self.read_record_auto((-1), DataType.FLOAT)
            part_coordsy = self.read_record_auto((-1), DataType.FLOAT)
            part_coordsz = self.read_record_auto((-1), DataType.FLOAT)
            part_coords = np.array([part_coordsx, part_coordsy, part_coordsz]).T
            #print(len(part_coords))
            #print(len(part_coords[0]))
            parts[part_num] = part_coords
        self.sections = parts
        self.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfn = sys.argv[1]
        cf = CaseFile(cfn)
        cf.read()
    else:
        print("Please specify filetype.")