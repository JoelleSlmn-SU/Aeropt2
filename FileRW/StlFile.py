import pyvista as pv
import os, sys
sys.path.append(os.path.dirname("FileRW"))
from FileRW.Mesh import Mesh

class Stl(Mesh):
    
    # Mesh Interface
    def read_file(self, filename: str, mesh=None):
        print(f"stl file: {filename}")
        if mesh == None:
            mesh = self 
        
        # Load the STL file
        pvmesh = pv.read(filename)
        self.from_pyvista(pvmesh)