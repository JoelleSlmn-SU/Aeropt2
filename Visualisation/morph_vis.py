import json
import os
import sys
import numpy as np
import pyvista as pv

# ---------------------------------------------------------------------------
# Make the repo root importable regardless of where this script is launched
# from, so we can reuse MeshGeneration.meshFile.load_mesh() instead of
# re-deriving MultiBlock/name-resolution logic here. Same upward-search
# pattern already used in Remote/remoteMorph.py.
# ---------------------------------------------------------------------------
def _add_repo_root_to_path(start_dir):
    d = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(d, "MeshGeneration")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent

_add_repo_root_to_path(os.path.dirname(os.path.abspath(__file__)))

try:
    from MeshGeneration.meshFile import load_mesh as _load_mesh_native
    _HAVE_MESHFILE = True
except Exception as _e:
    print(f"[morph_vis][WARN] Could not import MeshGeneration.meshFile ({_e}); "
          f"falling back to a minimal built-in .vtm reader.")
    _HAVE_MESHFILE = False


class _SimpleVtmFallback:
    """
    Minimal stand-in for MeshGeneration.meshFile.VtmMesh, used only if that
    module isn't importable (e.g. this script is copied out of the repo).
    Provides just get_surface_names() / get_surface_mesh() / get_surface_id(),
    which is all this script needs. Mirrors the leaf-flattening pattern
    already used in FileRW/Mesh.py, so nested MultiBlocks are handled the
    same way as elsewhere in the codebase.
    """

    def __init__(self, filepath):
        root = pv.read(filepath)
        self.blocks = []  # list[(name, pv.DataSet)]

        def _iter_leaves(obj, prefix=""):
            if isinstance(obj, pv.MultiBlock):
                for i in range(len(obj)):
                    child = obj[i]
                    key = obj.get_block_name(i) or f"block_{i}"
                    new_prefix = f"{prefix}{key}/" if prefix else f"{key}/"
                    yield from _iter_leaves(child, new_prefix)
            elif isinstance(obj, pv.DataSet):
                name = prefix[:-1] if prefix.endswith("/") else (prefix or "block")
                yield name, obj

        if isinstance(root, pv.MultiBlock):
            leaves = [(n, b) for n, b in _iter_leaves(root)
                      if b is not None and getattr(b, "n_points", 0) > 0]
        else:
            leaves = [("block_0", root)]

        self.blocks = leaves
        self._by_name = {n: b for n, b in self.blocks}
        self._by_id = {i: b for i, (_, b) in enumerate(self.blocks)}
        self._id_by_name = {n: i for i, (n, _) in enumerate(self.blocks)}

    def get_surface_names(self):
        return [n for n, _ in self.blocks]

    def get_surface_id(self, name_or_id):
        if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and str(name_or_id).isdigit()):
            return int(name_or_id)
        if name_or_id in self._id_by_name:
            return self._id_by_name[name_or_id]
        raise KeyError(f"Surface '{name_or_id}' not found.")

    def get_surface_mesh(self, name_or_id):
        if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and str(name_or_id).isdigit()):
            sid = int(name_or_id)
            if sid in self._by_id:
                return self._by_id[sid]
            raise KeyError(f"Surface id {sid} not found (0..{len(self.blocks) - 1}).")
        if name_or_id in self._by_name:
            return self._by_name[name_or_id]
        raise KeyError(f"Surface '{name_or_id}' not found. Available: {list(self._by_name)}")

    def get_surface_name(self, name_or_id):
        if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and str(name_or_id).isdigit()):
            sid = int(name_or_id)
            for n, i in self._id_by_name.items():
                if i == sid:
                    return n
            raise KeyError(f"Surface id {sid} not found.")
        if name_or_id in self._by_name:
            return name_or_id
        raise KeyError(f"Surface '{name_or_id}' not found.")


def _load_vtm(vtm_path):
    """Load a .vtm using the repo's own VtmMesh if available, else the fallback above."""
    if _HAVE_MESHFILE:
        return _load_mesh_native(vtm_path)
    return _SimpleVtmFallback(vtm_path)


def list_surfaces(vtm_path):
    """
    Quick utility: print the surface names/ids available in a .vtm file, so
    you can decide what to pass as `surface_selectors`. Run this once from a
    REPL (or add a call at the top of your driver) before picking names.
    """
    mesh_obj = _load_vtm(vtm_path)
    names = mesh_obj.get_surface_names()
    print(f"[morph_vis] {vtm_path}: {len(names)} surfaces")
    for nm in names:
        try:
            sid = mesh_obj.get_surface_id(nm)
        except Exception:
            sid = "?"
        print(f"    id={sid!s:>4}  name={nm}")
    return names


def _resolve_selected_blocks(mesh_obj, selectors, t_ids=None, u_ids=None, c_ids=None):
    """
    Turn a list of selectors into a list of leaf pv.DataSet blocks.

    Each selector may be:
      - an int / numeric string  -> surface id
      - an exact block name (as printed by list_surfaces)
      - "T" / "U" / "C"          -> expands to the t_surfaces / u_surfaces /
                                     c_surfaces id groups from the morph
                                     config (same T/U/C convention used in
                                     Visualisation/surface_rep.py)
      - "all"                    -> every block in the file

    `selectors=None` (or omitted) defaults to "all", i.e. the same
    full-surface behaviour the original script had.
    """
    all_names = mesh_obj.get_surface_names()

    if not selectors or (len(selectors) == 1 and str(selectors[0]).strip().lower() == "all"):
        chosen = list(all_names)
    else:
        chosen = []
        for s in selectors:
            key = str(s).strip().upper()
            if key == "T":
                if not t_ids:
                    print("[morph_vis][WARN] Selector 'T' given but no t_surfaces ids "
                          "were found in the morph config; skipping.")
                chosen.extend(t_ids or [])
            elif key == "U":
                if not u_ids:
                    print("[morph_vis][WARN] Selector 'U' given but no u_surfaces ids "
                          "were found in the morph config; skipping.")
                chosen.extend(u_ids or [])
            elif key == "C":
                if not c_ids:
                    print("[morph_vis][WARN] Selector 'C' given but no c_surfaces ids "
                          "were found in the morph config; skipping.")
                chosen.extend(c_ids or [])
            else:
                chosen.append(s)

    blocks, resolved_names, seen = [], [], set()
    for sel in chosen:
        try:
            blk = mesh_obj.get_surface_mesh(sel)
        except (KeyError, ValueError) as e:
            print(f"[morph_vis][WARN] Surface selector '{sel}' not found, skipping ({e}).")
            continue
        if blk is None or getattr(blk, "n_points", 0) == 0:
            continue
        if id(blk) in seen:
            continue
        seen.add(id(blk))
        blocks.append(blk)
        if hasattr(mesh_obj, "get_surface_name"):
            try:
                resolved_names.append(mesh_obj.get_surface_name(sel))
                continue
            except Exception:
                pass
        resolved_names.append(str(sel))

    if not blocks:
        raise RuntimeError(
            f"Surface selection {selectors} resolved to zero blocks.\n"
            f"Available surfaces in this .vtm ({len(all_names)}): {all_names}"
        )

    return blocks, resolved_names


def _load_display_surface(mesh_path, surface_selectors=None,
                           t_surfaces=None, u_surfaces=None, c_surfaces=None,
                           verbose=True):
    """
    Load the background surface to display, from either a legacy single
    .vtk (old behaviour, kept for backward compatibility) or a .vtm
    multiblock file with an explicit surface selection.

    Returns (combined_polydata, resolved_surface_names, all_available_names).
    """
    ext = os.path.splitext(mesh_path)[1].lower()

    if ext == ".vtk":
        mesh = pv.read(mesh_path)
        mesh = mesh.combine().extract_surface() if isinstance(mesh, pv.MultiBlock) else mesh.extract_surface()
        return mesh, ["<single .vtk surface>"], ["<single .vtk surface>"]

    if ext != ".vtm":
        raise ValueError(f"Unsupported mesh extension '{ext}' for {mesh_path} (expected .vtk or .vtm).")

    mesh_obj = _load_vtm(mesh_path)
    all_names = mesh_obj.get_surface_names()

    if verbose:
        print(f"[morph_vis] {mesh_path}: {len(all_names)} surfaces available:")
        for nm in all_names:
            print(f"    - {nm}")

    blocks, resolved_names = _resolve_selected_blocks(
        mesh_obj, surface_selectors, t_surfaces, u_surfaces, c_surfaces
    )
    if verbose:
        print(f"[morph_vis] Displaying {len(blocks)} surface(s): {resolved_names}")

    combined = pv.MultiBlock(blocks).combine().extract_surface()
    return combined, resolved_names, all_names


def load_morph_case(json_path, out_dir=None, n=0, vtm_filename=None):
    """
    Load morph config and locate the corresponding .vtm multiblock mesh,
    plus CNs, displacements, and the T/U/C surface-id groups.

    Parameters
    ----------
    json_path : str
        Path to morph_config_n_i.json (as written by pipeline_cluster.py /
        runSimRemote.py).
    out_dir : str, optional
        LOCAL root of the surfaces/ tree. The 'output_directory' field
        inside the json is the *remote* HPC path and is not valid here, so
        (exactly as the original script did) we override it with a local
        path instead of trusting the json blindly.
    n : int
        Generation index, kept for parity with the original signature.
        Cross-checked against cfg['gen'] when present.
    vtm_filename : str, optional
        Explicit .vtm filename to use instead of guessing it from vtk_name.
        Use this if your morph step doesn't name the deformed .vtm the same
        as the baseline mesh.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if out_dir is None:
        out_dir = (r"C:\Users\joell\OneDrive - Swansea University\Desktop"
                   r"\PhD Documents\01-Codes\Aeropt2\examples\crm_prelim")

    gen = data.get("gen", n)
    surf_dir = os.path.join(out_dir, "surfaces", f"n_{gen}")

    base = os.path.splitext(os.path.basename(data.get("vtk_name", "")))[0]
    candidates = []
    if vtm_filename:
        candidates.append(os.path.join(surf_dir, vtm_filename))
    if base:
        candidates.append(os.path.join(out_dir, f"{base}.vtm"))
    candidates.append(os.path.join(out_dir, "crm.vtm"))
    try:
        for fn in os.listdir(surf_dir):
            if fn.lower().endswith(".vtm") and os.path.join(surf_dir, fn) not in candidates:
                candidates.append(os.path.join(surf_dir, fn))
    except OSError:
        pass

    surface_path = next((p for p in candidates if os.path.exists(p)), None)
    if surface_path is None:
        raise FileNotFoundError(
            f"Could not find a .vtm surface mesh for n={n} (gen={gen}) in {surf_dir}.\n"
            f"Tried:\n  " + "\n  ".join(candidates) +
            "\n(NOTE: if your morph step only writes a .fro on the deformed mesh and never "
            "exports a per-generation .vtm, this file genuinely won't exist yet -- you'd "
            "need to add that export step, or point vtm_filename/out_dir at wherever the "
            "deformed multiblock actually gets written.)"
        )

    control_nodes = np.array(data["control_nodes"], dtype=float)
    displacements = np.array(data["displacement_vector"], dtype=float)
    t_surfaces = data.get("t_surfaces", [])
    u_surfaces = data.get("u_surfaces", [])
    c_surfaces = data.get("c_surfaces", [])

    return surface_path, control_nodes, displacements, t_surfaces, u_surfaces, c_surfaces


def visualise_morph(
    mesh_path,
    control_nodes,
    displacements,
    surface_selectors=None,   # e.g. ["T"], ["T", "U"], ["all"], or explicit names/ids
    t_surfaces=None,          # surface-id group from the morph config (for the "T" shortcut)
    u_surfaces=None,
    c_surfaces=None,
    scale=None,              # if None, auto-scale like mesh_gui
    screenshot_path=None,
    window_size=(1600, 1200),
    transparent_background=True,
):
    """
    Visualise one morph case, using a style similar to MeshViewer.plot_control_displacements:
      - semi-transparent background surface (one or more selected blocks from the .vtm)
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

    # --- Load mesh and reduce the selected surface(s) to a single PolyData ---
    mesh, shown_names, available_names = _load_display_surface(
        mesh_path,
        surface_selectors=surface_selectors,
        t_surfaces=t_surfaces,
        u_surfaces=u_surfaces,
        c_surfaces=c_surfaces,
    )

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
            auto_scale = 3.0 * L / dmax
        else:
            auto_scale = 3.0
    else:
        auto_scale = float(scale)

    disp_scaled = auto_scale * displacements
    targets = control_nodes + disp_scaled

    # Lift CNs a tiny bit off the surface to reduce z-fighting
    lift = 1e-3 * L
    cnP = control_nodes.copy()
    tgtP = targets.copy()
    cnP[:, 2] += lift
    tgtP[:, 2] += lift

    # --- Set up plotter ---
    plotter = pv.Plotter(off_screen=bool(screenshot_path))
    plotter.window_size = window_size

    # Background surface(s)
    plotter.add_mesh(
        mesh,
        color="#3246a8",
        opacity=0.7,
        label="+".join(shown_names) if len(shown_names) <= 3 else f"{len(shown_names)} surfaces",
    )
    plotter.add_mesh(
        mesh,
        style="wireframe",
        color="#3246a8",
        opacity=0.3,
    )

    # --- Glyphs for CNs (black & red spheres, like mesh_gui) ---
    r1 = 0.3
    r2 = 0.3 * lmin


    sph = pv.Sphere(radius=r1)
    sph2 = pv.Sphere(radius=r2)

    cn_poly = pv.PolyData(cnP)
    tgt_poly = pv.PolyData(tgtP)

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
        label="Displacement vectors",
    )

    # Axes & legend & info text
    plotter.add_axes()
    plotter.add_legend()
    plotter.add_text(
        f"N={N}, max|Δx|={dmax:.3e}, L={L:.3e}, scale×{auto_scale:.2f}\n"
        f"surfaces: {', '.join(shown_names)}",
        position="upper_left",
        font_size=10,
    )

    # Camera
    plotter.camera_position = [
        (-6997.334, 5895.07, 26083.2),   # camera position
        (5561.85, 966.818, 1248.69),     # focus point
        (0.537, -0.733, 0.417),     # view-up vector
    ]
    plotter.camera_position = [
        (51.9, 162.54, 82.247),   # camera position
        (59.2, 56.596, -7.136),     # focus point
        (0.0, -0.645, 0.7643),     # view-up vector
    ]
    #plotter.camera.zoom(7)

    if screenshot_path:
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        print("Saving screenshot to:", screenshot_path)
        # plotter.show(screenshot=...) has no transparent_background option, and
        # since this plotter is already off_screen (created with
        # off_screen=bool(screenshot_path) above), plotter.screenshot() alone
        # renders and captures - no need to call show() first.
        plotter.screenshot(screenshot_path, transparent_background=transparent_background)
        plotter.close()
    else:
        plotter.show()

# ----------------- Example driver for many morphs ----------------- #
if __name__ == "__main__":
    out_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\crm_prelim\surfaces\n_0"

    # Which surface(s) to draw as the background. "T" reuses the t_surfaces
    # ids stored in each morph_config_n_i.json (same T-surface as before,
    # just resolved from the .vtm instead of a separate output.vtk).
    # Once you've seen the printed surface list for your file, you can swap
    # this for explicit names/ids, e.g. ["T", "U"] or ["intake_wall_0003"].
    SURFACE_SELECTION = ["T", "U"]

    for i in range(0, 2):
        fpath = f"morph_config_n_{i+1}.json"
        json_path = os.path.join(out_path, fpath)

        mesh_path, cn, disp, t_ids, u_ids, c_ids = load_morph_case(json_path)

        print("Loaded mesh:", mesh_path)
        print("Control nodes:", cn.shape)
        print("Displacements:", disp.shape)

        # Save figure to file
        spath = f"morph_n1_{i+1}_vis.png"
        screenshot = os.path.join(out_path, spath)

        visualise_morph(
            mesh_path,
            cn,
            disp,
            surface_selectors=SURFACE_SELECTION,
            t_surfaces=t_ids,
            u_surfaces=u_ids,
            c_surfaces=c_ids,
            scale=2.0,                 # can increase to exaggerate arrow visibility
            screenshot_path=screenshot,
            window_size=(1600, 1200),
        )

        print(f"Saved visualisation → {screenshot}")