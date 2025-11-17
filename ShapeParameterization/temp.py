'''# Surface Fitting Skeleton with Enhanced Strategy for Complex Geometry

import os
import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
from geomdl import fitting

# === Step 1: Load and Preprocess ===
def load_and_clean_surface(filepath):
    reader = pv.get_reader(filepath)
    multiblock = reader.read()
    mesh = multiblock.merge() if isinstance(multiblock, pv.MultiBlock) else multiblock
    mesh = mesh.clean()
    return mesh

# === Step 2: Segment Surface by Curvature/Clustering ===
def segment_surface(mesh, method="dbscan"):
    points = mesh.points
    if method == "dbscan":
        db = DBSCAN(eps=2.0, min_samples=100).fit(points)
        labels = db.labels_
        return [mesh.extract_points(points[labels == i], include_cells=True) for i in np.unique(labels) if i != -1]
    else:
        raise NotImplementedError("Only DBSCAN is supported currently.")

# === Step 3: PCA Alignment ===
def pca_align(points):
    pca = PCA(n_components=3)
    aligned = pca.fit_transform(points)
    return aligned, pca

def inverse_pca_transform(points, pca):
    return pca.inverse_transform(points)

# === Step 4: Grid Sampling and Interpolation ===
def create_structured_grid(points, resolution=(30, 30)):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xi = np.linspace(x.min(), x.max(), resolution[0])
    yi = np.linspace(y.min(), y.max(), resolution[1])
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi[np.isnan(zi)] = np.nanmean(zi)  # fill NaNs
    return xi, yi, zi


# === Step 6: Visualize (Placeholder) ===
def visualize_surface_fit(mesh, surf, original_points):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    evalpts = np.array(surf.evalpts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(original_points[:,0], original_points[:,1], original_points[:,2], color='black', s=5, label='Original')
    ax.plot_trisurf(evalpts[:,0], evalpts[:,1], evalpts[:,2], cmap='viridis', alpha=0.7)
    ax.legend()
    plt.show()

# === Main Orchestrator ===
def process_surface_file(filepath):
    mesh = load_and_clean_surface(filepath)
    segments = segment_surface(mesh)
    for seg in segments:
        aligned_pts, pca = pca_align(seg.points)
        xi, yi, zi = create_structured_grid(aligned_pts)
        surf = fit_nurbs_surface(xi, yi, zi)
        original_evalpts = inverse_pca_transform(np.array(surf.evalpts), pca)
        visualize_surface_fit(seg, surf, seg.points)

# Example usage:
# process_surface_file("path/to/surface.vtk")

#!/usr/bin/env python'''
# -*- coding: utf-8 -*-

"""
    Examples for the NURBS-Python Package
    Released under MIT License
    Developed by Onur Rauf Bingol (c) 2018
"""

import os
import numpy as np
np.float = float
np.int = int
from geomdl import BSpline
from geomdl import exchange
from geomdl.visualization import VisVTK as vis


# Fix file path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Create a BSpline surface instance
surf = BSpline.Surface()

# Set degrees
surf.degree_u = 3
surf.degree_v = 3

# Set control points
surf.set_ctrlpts(*exchange.import_txt("ex_surface01.cpt", two_dimensional=True))

# Set knot vectors
surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]

# Set evaluation delta
surf.delta = 0.025

# Evaluate surface points
surf.evaluate()

# Import and use Matplotlib's colormaps
from matplotlib import cm

# Plot the control point grid and the evaluated surface
vis_comp = vis.VisSurface()
surf.vis = vis_comp
surf.render(colormap=cm.cool)

# Good to have something here to put a breakpoint
pass