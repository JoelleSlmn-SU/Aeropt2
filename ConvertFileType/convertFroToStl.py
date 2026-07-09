import numpy as np
import pyvista as pv
import os, sys

sys.path.append(os.path.dirname("FileRW"))
from FileRW.FroFile import FroFile

def convert_fro_to_stl(fro_path, stl_path=None):
    ff = FroFile.fromFile(fro_path)

    points = np.asarray(ff.nodes, float)
    faces = []

    for tri in ff.boundary_triangles:
        faces.extend([3, int(tri[0]), int(tri[1]), int(tri[2])])

    for quad in ff.boundary_quads:
        faces.extend([4, int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3])])

    mesh = pv.PolyData(points, np.asarray(faces))
    mesh = mesh.extract_surface().triangulate()

    if stl_path is None:
        stl_path = fro_path.rsplit(".", 1)[0] + ".stl"

    mesh.save(stl_path)
    return stl_path

fro_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\sphere_1.fro"
stl_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\sphere_1.stl"

convert_fro_to_stl(fro_path, stl_path)