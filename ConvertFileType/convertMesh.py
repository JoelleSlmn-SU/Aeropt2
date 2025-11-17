import os, sys
import pyvista as pv

sys.path.append(os.path.dirname('ConvertFileType'))
sys.path.append(os.path.dirname('ShapeParameterization'))
sys.path.append(os.path.dirname('MeshGeneration'))
from ConvertFileType.convertToStep import *
from ShapeParameterization.surfaceFitting import *
from MeshGeneration.RBFTransform import *
from MeshGeneration.controlNodeDisp import *
from MeshGeneration.meshFile import *

filename = os.path.join(os.getcwd(), "Outputs", "demo", f"morphed.vtm")
outfile = os.path.join(os.getcwd(), "Outputs", "demo", f"morphed_surfaces.step")
#plotNurbsSurface(nurbs_surf, points)
#
#save_path = os.path.join(os.getcwd(), "Outputs", "d10", f"morphed_geom.step")
export_vtm_to_step(filename, outfile)