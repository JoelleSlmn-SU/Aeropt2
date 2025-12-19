import json
import os
import numpy as np
import pyvista as pv

def load_morph_case(json_path):
    """Load morph config and extract mesh path, CNs, and displacements."""
    with open(json_path, "r") as f:
        data = json.load(f)

    out_dir = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\Corner Bump Surface Coarse Optimisation"
    n = 0

    # Path to the morphed T-surface (adjust name as needed)
    # Typical structure in AerOpt:
    #     output_directory/surfaces/n_1/output.vtk
    surface_path = os.path.join(out_dir, "surfaces", f"n_{n}", "output.vtk")

    if not os.path.exists(surface_path):
        # try alternative naming
        alt = os.path.join(out_dir, "surfaces", f"n_{n}", "T_surface.vtm")
        if os.path.exists(alt):
            surface_path = alt
        else:
            raise FileNotFoundError(f"Could not find surface mesh for n={n}")

    control_nodes = np.array(data["control_nodes"], dtype=float)
    displacements = np.array(data["displacement_vector"], dtype=float)

    return surface_path, control_nodes, displacements

def visualise_morph(
    mesh_path,
    control_nodes,
    displacements,
    scale=None,              # if None, auto-scale like mesh_gui
    screenshot_path=None,
    window_size=(1600, 1200),
):
    """
    Visualise one morph case, using a style similar to MeshViewer.plot_control_displacements:
      - semi-transparent T-surface
      - black spheres for original CNs
      - red spheres for displaced CNs
      - red tube segments between them
      - colour bar for |displacement| on the segments
    """

    control_nodes = np.asarray(control_nodes, dtype=float)
    displacements = np.asarray(displacements, dtype=float)

    # Check if data is valid
    if control_nodes.size == 0 or displacements.size == 0:
        print("WARNING: Empty arrays!")
        return

    # Reshape if needed (flat array to (N, 3))
    if control_nodes.ndim == 1:
        control_nodes = control_nodes.reshape(-1, 3)
    if displacements.ndim == 1:
        displacements = displacements.reshape(-1, 3)

    assert control_nodes.shape == displacements.shape
    N = control_nodes.shape[0]

    # --- Load mesh and reduce to a single PolyData surface ---
    mesh = pv.read(mesh_path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine().extract_surface()
    else:
        mesh = mesh.extract_surface()

    # --- Geometric scales (mimic mesh_gui.plot_control_displacements) ---
    # Domain length scale L from mesh bounds (fallback: from CN bbox)
    if mesh.n_points > 0:
        b = np.array(mesh.bounds, float)
        ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        L = float(np.linalg.norm(ext)) or 1.0
        lmin = float(max(ext.min(), 1e-12))
    else:
        P = control_nodes
        bmin, bmax = P.min(axis=0), P.max(axis=0)
        ext = bmax - bmin
        L = float(np.linalg.norm(ext)) or 1.0
        lmin = float(max(ext.min(), 1e-12))

    # Displacements and auto-scale
    mags = np.linalg.norm(displacements, axis=1)
    dmax = float(mags.max() or 1.0)

    if scale is None:
        # Same spirit as in mesh_gui: small deflections are amplified
        if dmax < 0.05 * L:
            auto_scale = 0.02 * L / dmax
        else:
            auto_scale = 1.0
    else:
        auto_scale = float(scale)

    disp_scaled = auto_scale * displacements
    targets = control_nodes + disp_scaled

    # Lift CNs a tiny bit off the surface to reduce z-fighting
    lift = 1e-4 * L
    cnP = control_nodes.copy()
    tgtP = targets.copy()
    cnP[:, 2] += lift
    tgtP[:, 2] += lift

    # --- Set up plotter ---
    plotter = pv.Plotter(off_screen=bool(screenshot_path))
    plotter.window_size = window_size

    # T-surface (background)
    plotter.add_mesh(
        mesh,
        color=(0.85, 0.85, 0.9),
        opacity=0.8,
        show_edges=True,
        label="T surface",
    )

    # --- Glyphs for CNs (black & red spheres, like mesh_gui) ---
    r = 0.012 * lmin

    sph = pv.Sphere(radius=r)

    cn_poly = pv.PolyData(cnP[0])
    tgt_poly = pv.PolyData(tgtP[0])

    cn_glyphs = cn_poly.glyph(geom=sph, scale=False)
    tgt_glyphs = tgt_poly.glyph(geom=sph, scale=False)

    act_cn = plotter.add_mesh(
        cn_glyphs,
        color="black",
        lighting=False,
        label="Control nodes (orig)",
    )
    act_tgt = plotter.add_mesh(
        tgt_glyphs,
        color="red",
        lighting=False,
        label="Control nodes (displaced)",
    )

    # --- Tube segments between orig and displaced CNs ---
    pts = np.vstack([cnP, tgtP])
    lines = np.hstack([[2, i, i + N] for i in range(N)]).astype(np.int64)
    segs = pv.PolyData(pts, lines=lines)

    # Attach displacement magnitude as a cell scalar, so we can show a color bar
    segs.cell_data["disp_mag"] = mags  # one value per segment

    act_segs = plotter.add_mesh(
        segs,
        scalars="disp_mag",
        cmap="viridis",
        line_width=3,
        render_lines_as_tubes=True,
        opacity=0.9,
        scalar_bar_args=dict(
            title="|Δx|",
            vertical=True,
            fmt="%.2e",
        ),
        label="Displacement vectors",
    )

    # Axes & legend & info text
    plotter.add_axes()
    plotter.add_legend()
    plotter.add_text(
        f"N={N}, max|Δx|={dmax:.3e}, L={L:.3e}, scale×{auto_scale:.2f}",
        position="upper_left",
        font_size=10,
    )

    # Camera
    plotter.camera_position = [
        (-6997.334, 5895.07, 26083.2),   # camera position
        (5561.85, 966.818, 1248.69),     # focus point
        (0.537, -0.733, 0.417),     # view-up vector
    ]
    plotter.camera.zoom(7)

    if screenshot_path:
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        print("Saving screenshot to:", screenshot_path)
        plotter.show(screenshot=screenshot_path, auto_close=True)
    else:
        plotter.show()

# ----------------- Example driver for many morphs ----------------- #
if __name__ == "__main__":
    out_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\Corner Bump Surface\surfaces\n_0"
    for i in range(5):
        fpath = f"morph_config_n_{i+1}.json"
        json_path = os.path.join(out_path, fpath)

        mesh_path, cn, disp = load_morph_case(json_path)

        print("Loaded mesh:", mesh_path)
        print("Control nodes:", cn.shape)
        print("Displacements:", disp.shape)

        # Save figure to file
        spath = f"morph_n1_{i+1}_vis.png"
        screenshot = os.path.join(out_path, spath)

        visualise_morph(
            mesh_path,
            cn[0],
            disp[0],
            scale=1.0,                 # can increase to exaggerate arrow visibility
            screenshot_path=screenshot,
            window_size=(1600, 1200),
        )

        print(f"Saved visualisation → {screenshot}")