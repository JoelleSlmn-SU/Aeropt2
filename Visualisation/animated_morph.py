import numpy as np
import pyvista as pv

def animate_vtk_morph(mesh0_path, mesh1_path, out_path="morph.mp4",
                  n_frames=60, fps=30, show_edges=False, opacity=1.0):
    m0 = pv.read(mesh0_path)
    m1 = pv.read(mesh1_path)

    # Sanity checks: topology must match (same number/order of points)
    if m0.n_points != m1.n_points:
        raise ValueError(f"Point count differs: {m0.n_points} vs {m1.n_points}. "
                         "Need same mesh topology for point interpolation.")
    if m0.n_cells != m1.n_cells:
        raise ValueError(f"Cell count differs: {m0.n_cells} vs {m1.n_cells}.")

    p0 = m0.points.copy()
    p1 = m1.points.copy()

    # We'll update a copy in-place for rendering
    m = m0.copy()

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(m, show_edges=show_edges, opacity=opacity)
    pl.camera_position = pl.camera_position  # set once; optionally overwrite with a known camera

    # Choose writer by extension
    if out_path.lower().endswith(".gif"):
        pl.open_gif(out_path, fps=fps)
    else:
        pl.open_movie(out_path, fps=fps)

    for t in np.linspace(0.0, 1.0, n_frames):
        m.points = (1 - t) * p0 + t * p1
        pl.write_frame()

    pl.close()
    return out_path

# Example:
# animate_morph("orig.vtk", "morphed.vtk", out_path="morph.mp4", n_frames=80, fps=25, opacity=0.6)

import numpy as np
import pyvista as pv
import os, sys

sys.path.append(os.path.dirname("FileRW"))
from FileRW.FroFile import FroFile  # adjust import path to your project

import numpy as np
import pyvista as pv
from FileRW.FroFile import FroFile  # adjust to your project import

def fro_surface_polydata(fro: FroFile, surface_id: int) -> pv.PolyData:
    pts = np.asarray(fro.nodes, dtype=float)

    faces = []

    tris = np.asarray(getattr(fro, "boundary_triangles", np.zeros((0, 4), dtype=int)))
    if tris.size:
        sel = tris[tris[:, 3] == int(surface_id)]
        for a, b, c, _sid in sel:
            faces.extend([3, int(a), int(b), int(c)])

    quads = np.asarray(getattr(fro, "boundary_quads", np.zeros((0, 5), dtype=int)))
    if quads.size:
        sel = quads[quads[:, 4] == int(surface_id)]
        for a, b, c, d, _sid in sel:
            faces.extend([4, int(a), int(b), int(c), int(d)])

    if not faces:
        raise ValueError(f"No boundary faces found for surface_id={surface_id}")

    return pv.PolyData(pts, np.array(faces, dtype=np.int64))

import numpy as np

def set_camera_from_paraview(pl):
    pos   = (1283.1, 8264.98, 18682.9)
    focal = (5785.03,  1245.4, 1273.31)
    up    = (0.3489, -0.834, 0.4266)

    pl.camera_position = (pos, focal, up)

    # View angle (ParaView calls it View Angle; VTK/PyVista uses view_angle)
    try:
        pl.camera.view_angle = 3.72
    except Exception:
        pass
    
def interactive_pick_camera(mesh):
    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=True)
    pl.show()
    print("CAMERA:", pl.camera_position)

def frame_camera_on_mesh(plotter: pv.Plotter, mesh: pv.PolyData, view="isometric", padding=1.15):
    # bounds: (xmin,xmax, ymin,ymax, zmin,zmax)
    b = np.array(mesh.bounds, dtype=float)
    center = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2], dtype=float)
    diag = np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
    dist = max(diag * padding, 1e-6)

    plotter.set_focus(center)

    if view == "xy":
        plotter.camera_position = (center + np.array([0, 0, dist]), center, (0, 1, 0))
    elif view == "xz":
        plotter.camera_position = (center + np.array([0, dist, 0]), center, (0, 0, 1))
    elif view == "yz":
        plotter.camera_position = (center + np.array([dist, 0, 0]), center, (0, 0, 1))
    else:  # isometric-ish
        plotter.camera_position = (center + np.array([dist, dist, dist]), center, (0, 0, 1))

    plotter.reset_camera()
    
def preview_surface_only(orig_fro_path, morphed_fro_path, surface_id: int):
    f0 = FroFile.fromFile(orig_fro_path)
    f1 = FroFile.fromFile(morphed_fro_path)

    s0 = fro_surface_polydata(f0, surface_id)
    s1 = fro_surface_polydata(f1, surface_id)

    pl = pv.Plotter()
    pl.add_mesh(s0, opacity=0.25, show_edges=False)   # baseline faint
    pl.add_mesh(s1, opacity=1.0,  show_edges=True)    # morphed strong

    frame_camera_on_mesh(pl, s1, view="isometric")
    pl.show()
    
def fro_to_polydata(fro: FroFile) -> pv.PolyData:
    """
    Build a single PolyData from triangles+quads in the FroFile.
    Assumes fro.boundary_triangles has shape (nTri,4) = [i,j,k,sid]
            fro.boundary_quads     has shape (nQuad,5)= [i,j,k,l,sid]
    """
    pts = np.asarray(fro.nodes, dtype=float)

    faces = []

    tris = np.asarray(getattr(fro, "boundary_triangles", np.zeros((0, 4), dtype=int)))
    if tris.size:
        for a, b, c, _sid in tris:
            faces.extend([3, int(a), int(b), int(c)])

    quads = np.asarray(getattr(fro, "boundary_quads", np.zeros((0, 5), dtype=int)))
    if quads.size:
        for a, b, c, d, _sid in quads:
            faces.extend([4, int(a), int(b), int(c), int(d)])

    poly = pv.PolyData(pts, np.array(faces, dtype=np.int64))
    return poly


def animate_fro_morph_surface(
    fro0_path, fro1_path, surface_id: int,
    out_path="morph.mp4",
    n_frames=80, fps=25,
    base_opacity=0.12,
    morph_opacity=1.0,
    show_edges=True,
    az_deg=5, el_deg=5,
    color_by_displacement=True,
):
    f0 = FroFile.fromFile(fro0_path)
    f1 = FroFile.fromFile(fro1_path)

    # checks (same as yours)
    if f0.node_count != f1.node_count:
        raise ValueError("node_count differs; cannot interpolate.")
    if np.any(np.asarray(f0.boundary_triangles) != np.asarray(f1.boundary_triangles)):
        raise ValueError("Triangle connectivity differs.")
    if np.any(np.asarray(f0.boundary_quads) != np.asarray(f1.boundary_quads)):
        raise ValueError("Quad connectivity differs.")

    p0 = np.asarray(f0.nodes, float)
    p1 = np.asarray(f1.nodes, float)

    # Build ONLY the morphed patch surface
    base_mesh  = fro_surface_polydata(f0, surface_id).copy()
    morph_mesh = fro_surface_polydata(f0, surface_id).copy()

    # Precompute displacement magnitude for the patch nodes (for coloring)
    # Note: PolyData uses ALL points, but your faces index into global node list.
    # So we can color by per-point magnitude using global displacements.
    disp = (p1 - p0)
    disp_mag = np.linalg.norm(disp, axis=1)

    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")

    # baseline ghost
    pl.add_mesh(base_mesh, style="wireframe", line_width=1.0, opacity=1.0)

    deform_scale = 5.0  # try 5–20 for presentations 
    # Morphed (animated) - color by displacement magnitude to make change pop
    morph_mesh["disp_mag"] = disp_mag  # per-point scalar
    pl.add_mesh(morph_mesh, scalars="disp_mag", opacity=1.0, show_edges=True)
    pl.add_scalar_bar(title="|Δx|", n_labels=4)
       
    pl.add_text(f"Baseline = wireframe | Morphed = solid (x{deform_scale:.0f} deformation)",
            position="upper_left", font_size=12, color="black")

    # camera framing + chosen angle
    set_camera_from_paraview(pl)

    # writer
    if out_path.lower().endswith(".gif"):
        pl.open_gif(out_path, fps=fps)
    else:
        pl.open_movie(out_path)  # <-- do NOT pass fps (your earlier fix)

    for t in np.linspace(0.0, 1.0, n_frames):
        morph_mesh.points = p0 + deform_scale * t * (p1 - p0)
        pl.write_frame()

    pl.close()
    return out_path

def animate_split_screen_fro_surface(
    fro0_path, fro1_path, surface_id: int,
    out_path="split_morph.mp4",
    n_frames=80, fps=25,
    deform_scale=10.0,          # presentation exaggeration
    base_surface_opacity=0.15,
):
    f0 = FroFile.fromFile(fro0_path)
    f1 = FroFile.fromFile(fro1_path)

    # Must match topology for direct interpolation
    if f0.node_count != f1.node_count:
        raise ValueError("node_count differs; cannot interpolate.")
    if np.any(np.asarray(f0.boundary_triangles) != np.asarray(f1.boundary_triangles)):
        raise ValueError("Triangle connectivity differs.")
    if np.any(np.asarray(f0.boundary_quads) != np.asarray(f1.boundary_quads)):
        raise ValueError("Quad connectivity differs.")

    p0 = np.asarray(f0.nodes, float)
    p1 = np.asarray(f1.nodes, float)
    disp = (p1 - p0)
    disp_mag = np.linalg.norm(disp, axis=1)

    base_mesh  = fro_surface_polydata(f0, surface_id).copy()
    morph_mesh = fro_surface_polydata(f0, surface_id).copy()
    morph_mesh["disp_mag"] = disp_mag

    # 1 row, 2 columns
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 800))
    pl.set_background("white")

    # ---------------- LEFT: BASELINE (static) ----------------
    pl.subplot(0, 0)
    pl.add_text("Baseline", font_size=14, color="black")
    pl.add_mesh(base_mesh, style="wireframe", line_width=2.0, opacity=1.0)
    pl.add_mesh(base_mesh, opacity=base_surface_opacity, show_edges=False)

    # ---------------- RIGHT: MORPHED (animated) ----------------
    pl.subplot(0, 1)
    pl.add_text(f"Morphed (x{deform_scale:.0f} deformation shown)", font_size=14, color="black")
    morph_mesh["disp_mag"] = disp_mag  # per-point scalar
    pl.add_mesh(morph_mesh, scalars="disp_mag", opacity=1.0, show_edges=False)
    pl.add_scalar_bar(title="|Δx|", n_labels=4)

    # Set camera on BOTH subplots, then link them
    pl.subplot(0, 0); set_camera_from_paraview(pl)
    cam = pl.camera_position  # capture
    pl.subplot(0, 1); pl.camera_position = cam

    # Keep cameras locked together
    try:
        pl.link_views()
    except Exception:
        pass

    # Writer (IMPORTANT: don't pass fps to open_movie due to imageio mismatch)
    if out_path.lower().endswith(".gif"):
        pl.open_gif(out_path, fps=fps)
    else:
        pl.open_movie(out_path)

    # Animate only the RIGHT mesh points
    for t in np.linspace(0.0, 1.0, n_frames):
        morph_mesh.points = p0 + deform_scale * t * disp
        pl.write_frame()

    pl.close()
    return out_path

# Example:
orig_dir = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\temp\corner.fro"
x = 1
morph_dir = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\temp\corner_3.fro"
out_dir = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\temp\corner3_morph.mp4"
out_dir2 = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\temp\corner3_morph_split.mp4"

animate_fro_morph_surface(orig_dir, morph_dir, surface_id=14, out_path=out_dir)
animate_split_screen_fro_surface(
    orig_dir, morph_dir,
    surface_id=14,
    out_path=out_dir2,
    n_frames=90,
    fps=25,
    deform_scale=1.0
)
