import os
import sys
import numpy as np
import pyvista as pv

# Project imports
sys.path.append(os.path.dirname('MeshGeneration'))
sys.path.append(os.path.dirname('ConvertFileType'))
from MeshGeneration.meshFile import load_mesh

# Wing NURBS tools
try:
    from wing_nurbs import WingNURBSGenerator, process_wing_surface  # process_wing_surface may not exist in your file
except Exception:
    from wing_nurbs import WingNURBSGenerator
    process_wing_surface = None

# ---------- Evaluation helpers ----------
def _eval_pt(surf, u, v):
    if hasattr(surf, 'evaluate_single'):
        p = surf.evaluate_single((float(u), float(v)))
    elif hasattr(surf, 'surface_point'):
        p = surf.surface_point(float(u), float(v))
    else:
        raise AttributeError("Surface has neither evaluate_single nor surface_point")
    p = np.asarray(p, float)
    if p.size >= 3:
        return p[:3]
    if p.size == 2:
        return np.array([p[0], p[1], 0.0], float)
    return np.zeros(3, float)


def evaluate_surface_grid(surf, nu=120, nv=80):
    # avoid exact 0/1 to side-step open knot issues
    us = np.linspace(1e-9, 1 - 1e-9, nu)
    vs = np.linspace(1e-9, 1 - 1e-9, nv)
    G = np.empty((nv, nu, 3), float)
    for j, vv in enumerate(vs):
        for i, uu in enumerate(us):
            G[j, i] = _eval_pt(surf, uu, vv)
    return G


def grid_to_polydata(grid_xyz: np.ndarray) -> pv.PolyData:
    nv, nu, _ = grid_xyz.shape
    pts = grid_xyz.reshape(-1, 3)
    faces = []
    def idx(i, j):
        return j * nu + i
    for j in range(nv - 1):
        for i in range(nu - 1):
            faces += [4, idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
    faces = np.asarray(faces, np.int64)
    return pv.PolyData(pts, faces)


def build_surface_from_points(points: np.ndarray, method: str, n_span: int, n_chord: int):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 8:
        raise ValueError("points must be (N,3) with N>=8")

    method = (method or 'fit').lower()

    if method in {"robust_grid", "rbf_grid"} and process_wing_surface is not None:
        surf, _ = process_wing_surface(points, normals=None, method=method)
        return surf

    gen = WingNURBSGenerator(points, n_span=n_span, n_chord=n_chord)

    if method == 'direct':
        grid = gen.create_structured_grid_robust(n_chord=n_chord, n_span=n_span)
        return gen.build_surface(grid)

    if hasattr(gen, 'generate'):
        return gen.generate()

    grid = gen.create_structured_grid_robust(n_chord=n_chord, n_span=n_span)
    return gen.build_surface(grid)


# ---------- Visualization ----------
def visualize(points: np.ndarray, surface, nu: int, nv: int):
    dense = evaluate_surface_grid(surface, nu=max(60, nu), nv=max(40, nv))
    poly = grid_to_polydata(dense)

    plotter = pv.Plotter()
    plotter.add_points(points, render_points_as_spheres=True, point_size=5.0, opacity=0.25)
    plotter.add_mesh(poly, show_edges=True, smooth_shading=True, opacity=0.95)
    plotter.add_axes(); plotter.show_grid()
    plotter.show()


# ---------- CLI entry ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fit + visualize a NURBS surface from a selected mesh surface')
    parser.add_argument('--mesh_path', nargs='?', default=os.path.join(os.getcwd(), 'Inputs', 'MeshData', 'crm.vtm'))
    parser.add_argument('--surface', help='Surface index or name (optional)')
    parser.add_argument('--method', choices=['fit', 'direct', 'robust_grid', 'rbf_grid'], default='fit')
    parser.add_argument('--n_span', type=int, default=25, help='spanwise samples (v)')
    parser.add_argument('--n_chord', type=int, default=15, help='chordwise samples (u)')
    args = parser.parse_args()

    #python ShapeParameterization/nurbs_pipeline.py --mesh_path Inputs/Mesh Data/crm.vtm --surface "9" --method robust_grid --n_span 30 --n_chord 20
    
    # Load mesh and pick points
    mesh = load_mesh(args.mesh_path)
    pts = None
    if args.surface is None:
        pts = mesh.get_all_points()
    else:
        # try index, then name
        surf_sel = args.surface
        names = list(mesh.get_surface_names())
        try:
            idx = int(surf_sel)
            if idx < 0 or idx >= len(names):
                raise IndexError
            key = names[idx]
        except (ValueError, IndexError):
            key = surf_sel
            if key not in names:
                raise KeyError(f"Surface '{surf_sel}' not found. Available: {names}")
        pts = mesh.get_surface_points(key)

    if pts is None or len(pts) < 8:
        raise ValueError("Selected surface has too few points to fit a NURBS surface.")

    # Build surface (visualize-only path; no export)
    nurbs_surface = build_surface_from_points(pts, method=args.method, n_span=args.n_span, n_chord=args.n_chord)

    # Visualize
    visualize(np.asarray(pts, float), nurbs_surface, nu=args.n_chord * 3, nv=args.n_span * 2)
