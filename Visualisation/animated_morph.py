import os
import sys
import json
import numpy as np
import pyvista as pv

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def get_surfaces_from_morph_config(json_path, mode="TU", unique=True, sort=True):
    """Load a morph_config JSON file and return surface IDs based on mode.

    mode ∈ { "T","U","C","TU","TC","UC","TUC" }.
    """
    with open(json_path, "r") as f:
        cfg = json.load(f)

    T = list(cfg.get("t_surfaces", []))
    U = list(cfg.get("u_surfaces", []))
    C = list(cfg.get("c_surfaces", []))

    mode = mode.upper()
    if mode == "T":
        out = T
    elif mode == "U":
        out = U
    elif mode == "C":
        out = C
    elif mode == "TU":
        out = T + U
    elif mode == "TC":
        out = T + C
    elif mode == "UC":
        out = U + C
    elif mode == "TUC":
        out = T + U + C
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use one of: T, U, C, TU, TC, UC, TUC")

    out = [int(s) for s in out]
    if unique:
        out = list(set(out))
    if sort:
        out.sort()
    return out


def add_slider_safe(plotter: pv.Plotter, **kwargs):
    """Call add_slider_widget but ignore kwargs unsupported by your PyVista build."""
    import inspect
    sig = inspect.signature(plotter.add_slider_widget)
    safe_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return plotter.add_slider_widget(**safe_kwargs)


def set_camera_from_paraview(pl: pv.Plotter):
    """Hard-coded camera copied from ParaView (edit as needed)."""
    pos   = (1283.1, 8264.98, 18682.9)
    focal = (5785.03,  1245.4, 1273.31)
    up    = (0.3489, -0.834, 0.4266)

    pl.camera_position = (pos, focal, up)
    try:
        pl.camera.view_angle = 3.72
    except Exception:
        pass


def frame_camera_on_mesh(plotter: pv.Plotter, mesh: pv.PolyData, view="isometric", padding=1.15):
    b = np.array(mesh.bounds, dtype=float)  # (xmin,xmax, ymin,ymax, zmin,zmax)
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


# ----------------------------------------------------------------------
# FRO helpers
# ----------------------------------------------------------------------

# Make sure your project root is on sys.path so FroFile import works.
# If this file lives in Aeropt2/Visualisation/, the following generally works:
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from FileRW.FroFile import FroFile
except Exception as e:
    raise ImportError(
        "Could not import FileRW.FroFile. Ensure Aeropt2 is on your PYTHONPATH "
        "or adjust the sys.path injection near the top of this script."
    ) from e


def fro_surface_polydata(fro: "FroFile", surface_id) -> pv.PolyData:
    """Return a PolyData for *only* the selected boundary surface(s),
    while keeping the FULL global point array (fro.nodes).

    This is the original behaviour that keeps indexing consistent for:
      - p0/p1/disp arrays (global)
      - face connectivity indices (global)
      - interactive slider updates (mesh.points can be set with global arrays)
    """
    pts = np.asarray(fro.nodes, dtype=float)

    if isinstance(surface_id, (list, tuple, set, np.ndarray)):
        sids = set(int(s) for s in surface_id)
    else:
        sids = {int(surface_id)}

    faces = []

    tris = np.asarray(getattr(fro, "boundary_triangles", np.zeros((0, 4), dtype=int)))
    if tris.size:
        sel = tris[np.isin(tris[:, 3], list(sids))]
        for a, b, c, _sid in sel:
            faces.extend([3, int(a), int(b), int(c)])

    quads = np.asarray(getattr(fro, "boundary_quads", np.zeros((0, 5), dtype=int)))
    if quads.size:
        sel = quads[np.isin(quads[:, 4], list(sids))]
        for a, b, c, d, _sid in sel:
            faces.extend([4, int(a), int(b), int(c), int(d)])

    if not faces:
        raise ValueError(f"No boundary faces found for surface_id(s)={sorted(sids)}")

    poly = pv.PolyData(pts, faces=np.asarray(faces, dtype=np.int64))
    poly.point_data["gid"] = np.arange(pts.shape[0], dtype=np.int64)  # trivial global ids
    return poly


def fro_to_polydata(fro: "FroFile") -> pv.PolyData:
    """Build a single PolyData of all boundary triangles+quads."""
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

    return pv.PolyData(pts, np.asarray(faces, dtype=np.int64))


def preview_surface_only(orig_fro_path, morphed_fro_path, surface_id: int):
    """Quick sanity check: overlay baseline vs morphed for one surface."""
    f0 = FroFile.fromFile(orig_fro_path)
    f1 = FroFile.fromFile(morphed_fro_path)

    s0 = fro_surface_polydata(f0, surface_id)
    s1 = fro_surface_polydata(f1, surface_id)

    pl = pv.Plotter()
    pl.add_mesh(s0, opacity=0.25, show_edges=False)
    pl.add_mesh(s1, opacity=1.0, show_edges=True)

    frame_camera_on_mesh(pl, s1, view="isometric")
    pl.show()


# ----------------------------------------------------------------------
# Animations
# ----------------------------------------------------------------------

def animate_vtk_morph(mesh0_path, mesh1_path, out_path="morph.mp4",
                      n_frames=60, fps=30, show_edges=False, opacity=1.0):
    """Interpolate between two VTK/VTP/VTM-compatible meshes with identical topology."""
    m0 = pv.read(mesh0_path)
    m1 = pv.read(mesh1_path)

    if m0.n_points != m1.n_points:
        raise ValueError(f"Point count differs: {m0.n_points} vs {m1.n_points}. Need same topology.")
    if m0.n_cells != m1.n_cells:
        raise ValueError(f"Cell count differs: {m0.n_cells} vs {m1.n_cells}.")

    p0 = m0.points.copy()
    p1 = m1.points.copy()
    m = m0.copy()

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(m, show_edges=show_edges, opacity=opacity)

    if out_path.lower().endswith(".gif"):
        pl.open_gif(out_path, fps=fps)
    else:
        pl.open_movie(out_path)

    for t in np.linspace(0.0, 1.0, n_frames):
        m.points = (1 - t) * p0 + t * p1
        pl.write_frame()

    pl.close()
    return out_path


def animate_fro_morph_surface(
    fro0_path, fro1_path, surface_id,
    out_path="morph.mp4",
    n_frames=80, fps=25,
    base_opacity=0.12,
    morph_opacity=1.0,
    show_edges=True,
    deform_scale=2.0,
    color_by_displacement=True,
):
    f0 = FroFile.fromFile(fro0_path)
    f1 = FroFile.fromFile(fro1_path)

    # topology checks
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

    base_mesh  = fro_surface_polydata(f0, surface_id).copy(deep=True)
    morph_mesh = fro_surface_polydata(f0, surface_id).copy(deep=True)

    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")

    pl.add_mesh(base_mesh, style="wireframe", line_width=0.2, opacity=0.4, show_scalar_bar=False)
    #pl.add_mesh(base_mesh, opacity=base_opacity, show_edges=False, show_scalar_bar=False)

    if color_by_displacement:
        morph_mesh["disp_mag"] = disp_mag  # global-sized; matches global point array
        pl.add_mesh(morph_mesh, scalars="disp_mag", opacity=morph_opacity, show_edges=False)
    else:
        pl.add_mesh(morph_mesh, opacity=morph_opacity, show_edges=show_edges)

    pl.add_text(f"Morph (x{deform_scale:.0f} deformation shown)",
                position="upper_left", font_size=12, color="black")

    set_camera_from_paraview(pl)
    pl.reset_camera()

    if out_path.lower().endswith(".gif"):
        pl.open_gif(out_path, fps=fps)
    else:
        pl.open_movie(out_path)

    for t in np.linspace(0.0, 1.0, n_frames):
        morph_mesh.points = p0 + (deform_scale * float(t)) * disp
        pl.write_frame()

    pl.close()
    return out_path


def animate_split_screen_fro_surface(
    fro0_path, fro1_path, surface_id,
    out_path="split_morph.mp4",
    n_frames=80, fps=25,
    deform_scale=10.0,
    base_surface_opacity=0.15,
    show_edges=False,
):
    f0 = FroFile.fromFile(fro0_path)
    f1 = FroFile.fromFile(fro1_path)

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

    base_mesh  = fro_surface_polydata(f0, surface_id).copy(deep=True)
    morph_mesh = fro_surface_polydata(f0, surface_id).copy(deep=True)
    morph_mesh["disp_mag"] = disp_mag

    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 800))
    pl.set_background("white")

    pl.subplot(0, 0)
    pl.add_text("Baseline", font_size=14, color="black")
    pl.add_mesh(base_mesh, style="wireframe", line_width=2.0, opacity=1.0)
    pl.add_mesh(base_mesh, opacity=base_surface_opacity, show_edges=False)

    pl.subplot(0, 1)
    pl.add_text(f"Morphed (x{deform_scale:.0f} shown)", font_size=14, color="black")
    pl.add_mesh(morph_mesh, scalars="disp_mag", opacity=1.0, show_edges=show_edges)

    pl.subplot(0, 1)
    set_camera_from_paraview(pl)
    pl.reset_camera()
    cam = pl.camera_position

    pl.subplot(0, 0)
    pl.camera_position = cam
    try:
        pl.link_views()
    except Exception:
        pass

    if out_path.lower().endswith(".gif"):
        pl.open_gif(out_path, fps=fps)
    else:
        pl.open_movie(out_path)

    for t in np.linspace(0.0, 1.0, n_frames):
        morph_mesh.points = p0 + (deform_scale * float(t)) * disp
        pl.write_frame()

    pl.close()
    return out_path

import os

def _ensure_ext(path: str, ext: str) -> str:
    """Return path with the given extension (ext includes dot)."""
    root, _ = os.path.splitext(path)
    return root + ext

def _save_both(callable_anim, out_path: str, save_mp4=True, save_gif=True, **kwargs):
    """
    Generic helper: calls an existing animator twice (mp4 + gif),
    reusing the same kwargs so the animations match.
    """
    outputs = {}

    if save_mp4:
        mp4_path = out_path if out_path.lower().endswith(".mp4") else _ensure_ext(out_path, ".mp4")
        try:
            outputs["mp4"] = callable_anim(out_path=mp4_path, **kwargs)
            print(f"[ANIM] Saved MP4: {mp4_path}")
        except Exception as e:
            # MP4 requires ffmpeg (PyVista open_movie). If missing, GIF can still succeed.
            print(f"[ANIM][WARN] MP4 failed ({e}). GIF may still succeed.")

    if save_gif:
        gif_path = out_path if out_path.lower().endswith(".gif") else _ensure_ext(out_path, ".gif")
        try:
            outputs["gif"] = callable_anim(out_path=gif_path, **kwargs)
            print(f"[ANIM] Saved GIF: {gif_path}")
        except Exception as e:
            print(f"[ANIM][ERROR] GIF failed ({e}).")

    return outputs


def animate_vtk_morph_both(
    mesh0_path, mesh1_path, out_path="morph",
    n_frames=60, fps=30, show_edges=False, opacity=1.0,
    save_mp4=True, save_gif=True,
):
    return _save_both(
        callable_anim=lambda out_path, **kw: animate_vtk_morph(
            mesh0_path, mesh1_path, out_path=out_path, **kw
        ),
        out_path=out_path,
        save_mp4=save_mp4,
        save_gif=save_gif,
        n_frames=n_frames,
        fps=fps,
        show_edges=show_edges,
        opacity=opacity,
    )


def animate_fro_morph_surface_both(
    fro0_path, fro1_path, surface_id,
    out_path="morph",
    n_frames=80, fps=25,
    base_opacity=0.12,
    morph_opacity=1.0,
    show_edges=True,
    deform_scale=2.0,
    color_by_displacement=True,
    save_mp4=True, save_gif=True,
):
    return _save_both(
        callable_anim=lambda out_path, **kw: animate_fro_morph_surface(
            fro0_path, fro1_path, surface_id, out_path=out_path, **kw
        ),
        out_path=out_path,
        save_mp4=save_mp4,
        save_gif=save_gif,
        n_frames=n_frames,
        fps=fps,
        base_opacity=base_opacity,
        morph_opacity=morph_opacity,
        show_edges=show_edges,
        deform_scale=deform_scale,
        color_by_displacement=color_by_displacement,
    )


def animate_split_screen_fro_surface_both(
    fro0_path, fro1_path, surface_id,
    out_path="split_morph",
    n_frames=80, fps=25,
    deform_scale=10.0,
    base_surface_opacity=0.15,
    show_edges=False,
    save_mp4=True, save_gif=True,
):
    return _save_both(
        callable_anim=lambda out_path, **kw: animate_split_screen_fro_surface(
            fro0_path, fro1_path, surface_id, out_path=out_path, **kw
        ),
        out_path=out_path,
        save_mp4=save_mp4,
        save_gif=save_gif,
        n_frames=n_frames,
        fps=fps,
        deform_scale=deform_scale,
        base_surface_opacity=base_surface_opacity,
        show_edges=show_edges,
    )


# ----------------------------------------------------------------------
# Option A: ParaView interactive export (time series)
# ----------------------------------------------------------------------

def export_morph_series_multiblock(
    fro0_path,
    fro1_path,
    surface_ids,
    out_dir,
    base_name="morph",
    n_frames=60,
    deform_scale=1.0,
    file_ext="vtm",
):
    """Export a ParaView-friendly time-series as MULTIBLOCK files (.vtm).

    Output:
      out_dir/base_name_0000.vtm
      out_dir/base_name_0001.vtm
      ...

    Each multiblock contains one block per surface_id, so you can toggle
    surfaces in ParaView's Pipeline Browser.
    """
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(surface_ids, (list, tuple, set, np.ndarray)):
        sids = [int(s) for s in surface_ids]
    else:
        sids = [int(surface_ids)]

    f0 = FroFile.fromFile(fro0_path)
    f1 = FroFile.fromFile(fro1_path)

    # topology checks
    if f0.node_count != f1.node_count:
        raise ValueError("node_count differs; cannot interpolate.")
    if np.any(np.asarray(f0.boundary_triangles) != np.asarray(f1.boundary_triangles)):
        raise ValueError("Triangle connectivity differs.")
    if np.any(np.asarray(f0.boundary_quads) != np.asarray(f1.boundary_quads)):
        raise ValueError("Quad connectivity differs.")

    p0 = np.asarray(f0.nodes, float)
    p1 = np.asarray(f1.nodes, float)
    disp = (p1 - p0) * float(deform_scale)
    disp_mag = np.linalg.norm(disp, axis=1)

    # Build per-surface meshes once
    meshes0 = []
    for sid in sids:
        m = fro_surface_polydata(f0, sid).copy(deep=True)
        m.point_data["disp_mag"] = disp_mag
        m.point_data["sid"] = np.full(m.n_points, sid, dtype=np.int32)
        meshes0.append(m)

    for k, t in enumerate(np.linspace(0.0, 1.0, int(n_frames))):
        mb = pv.MultiBlock()
        for i, sid in enumerate(sids):
            m = meshes0[i].copy(deep=True)
            m.points = p0 + float(t) * disp
            mb[f"surface_{sid}"] = m

        out_path = os.path.join(out_dir, f"{base_name}_{k:04d}.{file_ext}")
        mb.save(out_path)

    readme = os.path.join(out_dir, "HOW_TO_OPEN_IN_PARAVIEW.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "ParaView open instructions:\n"
            f"1) File -> Open -> select {base_name}_0000.{file_ext}\n"
            "2) Tick 'Open file series'\n"
            "3) Apply\n"
            "4) Use time controls (play / slider)\n"
            "5) Toggle blocks (surfaces) in the Pipeline Browser\n"
            "6) File -> Save State... to create a .pvsm you can reopen later\n"
        )

    return out_dir


# ----------------------------------------------------------------------
# PyVista interactive (debugging)
# ----------------------------------------------------------------------

def interactive_fro_morph_surface(
    fro0_path,
    fro1_path,
    surface_id,
    deform_scale=1.0,
    show_edges=True,
    color_by_displacement=True,
    show=True,
):
    """Local PyVista interactive morph with a small slider bottom-left."""
    f0 = FroFile.fromFile(fro0_path)
    f1 = FroFile.fromFile(fro1_path)

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

    mesh = fro_surface_polydata(f0, surface_id).copy(deep=True)
    if color_by_displacement:
        mesh["disp_mag"] = disp_mag

    pl = pv.Plotter()
    pl.set_background("white")

    if color_by_displacement:
        pl.add_mesh(
            mesh,
            scalars="disp_mag",
            cmap="viridis",
            show_edges=False,
            smooth_shading=True,
        )

        # --- Edge overlay (wireframe, transparent surface)
        pl.add_mesh(
            mesh,
            style="wireframe",
            color="grey",
            line_width=0.3,
            opacity=0.4,
        )
    else:
        pl.add_mesh(mesh, show_edges=show_edges)

    pl.add_text("Interactive morph: drag slider, move camera freely", font_size=12)

    def _update(t):
        mesh.points = p0 + (deform_scale * float(t)) * disp
        mesh.modified()

    add_slider_safe(
        pl,
        callback=_update,
        rng=[0.0, 1.0],
        value=0.0,
        title="t",
        pointa=(0.02, 0.08),
        pointb=(0.28, 0.08),
        slider_width=0.02,
        tube_width=0.008,
        fmt="%.2f",
    )

    pl.reset_camera()
    if show:
        pl.show()
    return pl


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    
    folder = [
        "CB Morph 12CN"
    ]

    #x_case = 7
    gen = 0
    
    for x_case in range(1,11):
        base = Path(r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples")
        case_root = base / folder[0]

        orig_fro = str(case_root / "surfaces" / f"PCA" / f"corner.fro")
        morp_fro = str(case_root / "surfaces" / f"PCA" / f"corner_{str(x_case)}.fro")
        cfg_json = str(case_root / "surfaces" / f"PCA" / f"morph_config_n_1.json")

        # 1) MP4 animations
        surfaces = get_surfaces_from_morph_config(cfg_json, mode="TU")
        out_1 = str(case_root / "surfaces" / f"PCA" / f"n0_{str(x_case)}_corner_morph_TU")
        out_2 = str(case_root / "surfaces" / f"PCA" / f"n0_{str(x_case)}_corner_morph_split_TU")
        
        animate_fro_morph_surface_both(
            orig_fro, morp_fro, surface_id=surfaces,
            out_path=out_1,
            n_frames=90, fps=25, deform_scale=1.0,
            save_mp4=False, save_gif=True,
        )

        animate_split_screen_fro_surface_both(
            orig_fro, morp_fro, surface_id=surfaces,
            out_path=out_2,
            n_frames=90, fps=25, deform_scale=1.0,
            save_mp4=False, save_gif=True,
        )

        # 2) Interactive mesh
        export = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Morph\surfaces\n_0"
        
        #interactive_fro_morph_surface(orig_fro, morp_fro, surface_id=surfaces, deform_scale=1.0)
        #export_morph_series_multiblock(orig_fro, morp_fro, surface_ids=surfaces, out_dir=export, base_name="corner_morph", n_frames=90, deform_scale=1.0)
    pass