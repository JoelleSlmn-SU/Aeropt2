import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname('ShapeParameterization'))
from ShapeParameterization.surfaceFitting import *

def loadMeshFile(filename):
    filename = os.path.join("Outputs/Mesh Data/", filename)
    mesh = pv.read(filename)
    return mesh

def readDatFile(filename):
    
    #filename = os.path.join(os.getcwd, filename)
    with open(filename, 'r') as file:
        lines = file.readlines()

    curves = {}
    i = 2
    while i < len(lines):
        line = lines[i].strip()

        if line and len(line.split()) == 2 and line.split()[0].isdigit():
            curve_id = int(line.split()[0])
            num_points = int(lines[i + 1].strip())
            coords = []

            for j in range(num_points):
                coord_line = lines[i + 2 + j].strip()
                x, y, z = map(float, coord_line.split())
                coords.append([x, y, z])

            curves[curve_id] = np.array(coords)
            i += 2 + num_points
        elif line == "Surfaces":
            break
        else:
            i += 1  # Skip header or unrelated lines

    surfaces = {}
    while i < len(lines):
        line = lines[i].strip()

        if line and len(line.split()) == 2 and line.split()[0].isdigit():
            surface_id = int(line.split()[0])
            num_p2 = num_points**2
            coords = []

            for j in range(num_p2):
                coord_line = lines[i + 2 + j].strip()
                x, y, z = map(float, coord_line.split())
                coords.append([x, y, z])

            surfaces[surface_id] = np.array(coords)
            i += 2 + num_points
        elif line == "Mesh Generation":
            break
        else:
            i += 1  # Skip header or unrelated lines
            
    return curves, surfaces

def plotCurves3D(curvesDict, ax=None, show_labels=False):
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

    for curve_id, points in curvesDict.items():
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax.plot(x, y, z, label=f"Curve {curve_id}")

        if show_labels:
            mid_idx = len(points) // 2
            ax.text(x[mid_idx], y[mid_idx], z[mid_idx], f"{curve_id}", fontsize=9)

    return ax
    
def plotSurfaces3D(surfacesDict, ax=None, show_labels=False):
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

    for surface_id, points in surfacesDict.items():
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax.scatter(x, y, z, label=f"Surface {surface_id}")

        if show_labels:
            mid_idx = len(points) // 2
            ax.text(x[mid_idx], y[mid_idx], z[mid_idx], f"{surface_id}", fontsize=9)

    return ax

cwd = os.getcwd()
filename = os.path.join(cwd, r"ShapeParameterization\fort.dat")
curves, surfaces = readDatFile(filename)

'''newSurf = {}
for key, value in surfaces.items():
    if 2 < key < 4:
        newSurf[key] = surfaces.get(key)'''

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot curves and surfaces on the same axes
plotCurves3D(curvesDict=curves, ax=ax, show_labels=False)
plotSurfaces3D(surfacesDict=surfaces, ax=ax, show_labels=True)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_aspect("equal")
ax.set_title("Curves and Surfaces from .dat File")
#ax.legend()
plt.tight_layout()
plt.show()

surfID = 2

filename = f"selected_surface_{surfID}.vtk"
mesh = loadMeshFile(filename)
points = mesh.points

surf = fitSurface3(surfaces, pts=points, surfID=surfID)
plotNurbsSurface(surf)