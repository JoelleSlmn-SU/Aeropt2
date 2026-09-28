"""
storyboard_figures.py

Consolidated, save-to-disk figure generation for the CRM-wing parameterisation
storyboard (UK Fluids 2026 talk).

This merges three prototype scripts -- surface_rep.py, morph_vis.py, vis.py --
into one file, and fixes the thing all three had in common: none of them
reliably wrote a PNG. surface_rep.py and vis.py only ever called .show() /
plt.show() (interactive, nothing saved); morph_vis.py *could* save via
screenshot_path but only did so if you remembered to pass it. Every function
below always takes an explicit out_png and always writes it.

Layout:
    0. repo import path          (unchanged from morph_vis.py / surface_rep.py)
    1. INPUT PATHS                <- edit this block for a new case
    2. OUTPUT PATHS                <- edit this block to change where PNGs land
    3. shared camera / colour     <- one camera + palette for every panel, so
                                      the storyboard doesn't re-orient slide to
                                      slide (this was the main "aesthetic"
                                      inconsistency across the old scripts --
                                      three different hard-coded camera tuples
                                      across morph_vis.py/animated_morph.py)
    4. mesh loading                (from morph_vis.py, incl. its .vtm fallback
                                      reader and T/U/C selector resolution)
    5. plot_surface_classification (from surface_rep.py)
    6. plot_control_node_displacement (from morph_vis.py)
    7. schematic_* plots           (from vis.py -- illustrative only, built from
                                      a standalone control_nodes.npy + Delaunay
                                      connectivity, NOT the real CRM mesh)
    8. generate_all_panels()        <- driver, run this file directly

Two source scripts (animated_morph.py, modal_rep.py) were left out of the
merge on purpose -- they're a different kind of figure (MP4/GIF animation,
and Laplacian/PCA mode-shape quivers) rather than a static storyboard panel,
and they already save to disk correctly. Worth folding in the same way later
if we want the mode-shape panel to come from this one file too.
"""

import os
import sys
import json
import platform
from pathlib import Path

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

# ============================================================================
# 0. REPO IMPORT PATH
# ============================================================================
# Walk upward from this file until a folder containing MeshGeneration/ is
# found, and put it on sys.path. Same upward-search pattern already used in
# Remote/remoteMorph.py, morph_vis.py and surface_rep.py.


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
except Exception as e:
    print(f"[storyboard][WARN] Could not import MeshGeneration.meshFile ({e}); "
          f"falling back to a minimal built-in .vtm reader.")
    _HAVE_MESHFILE = False


# ============================================================================
# 1. INPUT PATHS -- edit this block for whichever case you're rendering
# ============================================================================
CASE_ROOT = Path(
    r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents"
    r"\01-Codes\Aeropt2\examples\crm_prelim"
)

INPUTS = {
    # Baseline multiblock surface mesh (all boundary surfaces, undeformed)
    "vtm_baseline": CASE_ROOT / "crm.vtm",

    # morph_config_n_i.json: holds control_nodes, displacement_vector, and
    # t_surfaces / u_surfaces / c_surfaces IDs for one morph case
    "morph_config": CASE_ROOT / "surfaces" / "n_0" / "morph_config_n_1.json",

    # Surface IDs to always exclude from the classification plot
    # (e.g. symmetry plane, farfield -- same list as surface_rep.py's example)
    "exclude_surfaces": [15, 16],

    # Standalone control-node arrays used ONLY by the schematic_* panels below
    # (these are illustrative, not read from the real CRM mesh -- see vis.py)
    "control_nodes_npy": Path(
        r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents"
        r"\01-Codes\Aeropt2\examples\crm_prelim\Control Nodes\control_nodes.npy"
    ),
    "control_normals_npy": Path(
        r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents"
        r"\01-Codes\Aeropt2\examples\crm_prelim\Control Nodes\control_normals.npy"
    ),
}

# ============================================================================
# 2. OUTPUT PATHS -- every function writes here unless you pass out_png=...
# ============================================================================
OUTPUT_DIR = CASE_ROOT / "figs_storyboard"

OUTPUTS = {
    "surface_classification": OUTPUT_DIR / "01_surface_classification.png",
    "control_node_displacement": OUTPUT_DIR / "02_control_node_displacement.png",
    "schematic_control_nodes": OUTPUT_DIR / "00a_schematic_control_nodes.png",
    "schematic_connectivity": OUTPUT_DIR / "00b_schematic_connectivity.png",
    "schematic_morph_xyz": OUTPUT_DIR / "00c_schematic_morph_xyz.png",
    "schematic_morph_normal": OUTPUT_DIR / "00d_schematic_morph_normal.png",
}


# ============================================================================
# 3. SHARED CAMERA / COLOUR -- reused by every PyVista panel so the storyboard
#    doesn't jump viewpoint slide to slide (this is the fix for "photo
#    positioning" -- the three source scripts each had their own copy of a
#    camera tuple, and animated_morph.py even had two, with the first
#    silently overwritten by the second).
# ============================================================================
CAMERA_POSITION = [
    (51.9, 162.54, 82.247),   # camera position
    (59.2, 56.596, -7.136),   # focal point
    (0.0, -0.645, 0.7643),    # view-up
]
CAMERA_ZOOM = 9

COLORS = {
    "T": "#3246a8",       # target surfaces        -- blue
    "U": "#2fad35",       # unconstrained surfaces  -- green
    "C": "#a83238",       # constrained surfaces    -- red
    "other": "#c9c9c9",
    "background": "white",
}

WINDOW_SIZE = (1600, 1200)


def _apply_camera(plotter, position=CAMERA_POSITION, zoom=CAMERA_ZOOM):
    plotter.camera_position = position
    plotter.camera.zoom(zoom)


def _ensure_headless_display():
    """On a headless HPC/cluster node, off_screen rendering needs either a
    working OSMesa/EGL VTK build or a virtual X server. Call this once at the
    top of your driver script if screenshots come back blank on the cluster.
    Not called automatically -- harmless on a normal desktop, but importing
    pyvista's Xvfb helper on a machine that already has a display is
    unnecessary, so this is opt-in."""
    if platform.system() == "Linux":
        pv.start_xvfb()


# ============================================================================
# 4. MESH LOADING (shared by plot_surface_classification & plot_control_node_displacement)
# ============================================================================
class _SimpleVtmFallback:
    """Minimal stand-in for MeshGeneration.meshFile.VtmMesh, used only if that
    module isn't importable (e.g. this script is copied out of the repo).
    Provides get_surface_names() / get_surface_mesh() / get_surface_id() /
    get_surface_name() plus a `.blocks` list, which is everything the plot
    functions below need. Mirrors the leaf-flattening pattern already used in
    FileRW/Mesh.py, so nested MultiBlocks are handled the same way as
    elsewhere in the codebase."""

    def __init__(self, filepath):
        root = pv.read(str(filepath))

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
        return _load_mesh_native(str(vtm_path))
    return _SimpleVtmFallback(vtm_path)


def _find_vtm_from_config(cfg_path, cfg):
    """Best-effort .vtm discovery from a morph_config, for when INPUTS["vtm_baseline"]
    isn't set explicitly. Same search order as the original surface_rep.py /
    morph_vis.py had independently -- consolidated here so it only exists once."""
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    base = os.path.splitext(os.path.basename(cfg.get("vtk_name", "")))[0]

    candidates = []
    if base:
        candidates.append(os.path.join(cfg_dir, f"{base}.vtm"))
        out_dir = cfg.get("output_directory", "")
        gen = cfg.get("gen", 0)
        candidates.append(os.path.join(out_dir, "surfaces", f"n_{gen}", f"{base}.vtm"))
        candidates.append(os.path.join(out_dir, f"{base}.vtm"))
    try:
        for fn in os.listdir(cfg_dir):
            if fn.lower().endswith(".vtm"):
                candidates.append(os.path.join(cfg_dir, fn))
    except OSError:
        pass

    for p in candidates:
        if p and os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Could not locate a .vtm for {cfg_path}. Tried:\n  " + "\n  ".join(candidates)
    )


def _resolve_selected_blocks(mesh_obj, selectors, t_ids=None, u_ids=None, c_ids=None):
    """Turn a list of selectors into a list of leaf pv.DataSet blocks.

    Each selector may be:
      - an int / numeric string  -> surface id
      - an exact block name (as printed by list_surfaces)
      - "T" / "U" / "C"          -> expands to the t_surfaces / u_surfaces /
                                     c_surfaces id groups from the morph config
      - "all"                    -> every block in the file
    """
    all_names = mesh_obj.get_surface_names()

    if not selectors or (len(selectors) == 1 and str(selectors[0]).strip().lower() == "all"):
        chosen = list(all_names)
    else:
        chosen = []
        for s in selectors:
            key = str(s).strip().upper()
            if key == "T":
                chosen.extend(t_ids or [])
            elif key == "U":
                chosen.extend(u_ids or [])
            elif key == "C":
                chosen.extend(c_ids or [])
            else:
                chosen.append(s)

    blocks, resolved_names, seen = [], [], set()
    for sel in chosen:
        try:
            blk = mesh_obj.get_surface_mesh(sel)
        except (KeyError, ValueError) as e:
            print(f"[storyboard][WARN] Surface selector '{sel}' not found, skipping ({e}).")
            continue
        if blk is None or getattr(blk, "n_points", 0) == 0:
            continue
        if id(blk) in seen:
            continue
        seen.add(id(blk))
        blocks.append(blk)
        try:
            resolved_names.append(mesh_obj.get_surface_name(sel))
        except Exception:
            resolved_names.append(str(sel))

    if not blocks:
        raise RuntimeError(
            f"Surface selection {selectors} resolved to zero blocks.\n"
            f"Available surfaces ({len(all_names)}): {all_names}"
        )
    return blocks, resolved_names


def list_surfaces(vtm_path=None):
    """Utility: print the surface names/ids available in a .vtm, so you can
    decide what to pass as surface_selectors / exclude_surfaces."""
    vtm_path = Path(vtm_path or INPUTS["vtm_baseline"])
    mesh_obj = _load_vtm(vtm_path)
    names = mesh_obj.get_surface_names()
    print(f"[storyboard] {vtm_path}: {len(names)} surfaces")
    for nm in names:
        try:
            sid = mesh_obj.get_surface_id(nm)
        except Exception:
            sid = "?"
        print(f"    id={sid!s:>4}  name={nm}")
    return names


# ============================================================================
# 5. PLOT 1 -- surface classification (T/U/C colour-coded), from surface_rep.py
# ============================================================================
def plot_surface_classification(
    vtm_path=None,
    morph_config_path=None,
    exclude_surfaces=None,
    out_png=None,
    show_edges=False,
    opacity=1.0,
    off_screen=True,
):
    """Colour every boundary surface by its T/U/C membership.
    Storyboard panel: "surface setup / classification"."""
    morph_config_path = Path(morph_config_path or INPUTS["morph_config"])
    out_png = Path(out_png or OUTPUTS["surface_classification"])
    if exclude_surfaces is None:
        exclude_surfaces = INPUTS.get("exclude_surfaces", [])

    with open(morph_config_path, "r") as f:
        cfg = json.load(f)

    T = set(map(int, cfg.get("t_surfaces", [])))
    U = set(map(int, cfg.get("u_surfaces", [])))
    C = set(map(int, cfg.get("c_surfaces", [])))
    exclude = set(map(int, exclude_surfaces))

    if vtm_path is None:
        vtm_path = INPUTS.get("vtm_baseline")
        if vtm_path is None or not Path(vtm_path).exists():
            vtm_path = _find_vtm_from_config(morph_config_path, cfg)
    mesh_obj = _load_vtm(vtm_path)

    pl = pv.Plotter(off_screen=off_screen, window_size=WINDOW_SIZE)
    pl.set_background(COLORS["background"])

    counts = {"T": 0, "U": 0, "C": 0, "other": 0, "excluded": 0}
    for name, blk in getattr(mesh_obj, "blocks", []):
        sid = int(mesh_obj.get_surface_id(name))
        if sid in exclude:
            counts["excluded"] += 1
            continue
        if sid in T:
            color, key = COLORS["T"], "T"
        elif sid in U:
            color, key = COLORS["U"], "U"
        elif sid in C:
            color, key = COLORS["C"], "C"
        else:
            color, key = COLORS["other"], "other"
        counts[key] += 1

        pl.add_mesh(blk, color=color, opacity=opacity, pickable=False)
        if show_edges:
            pl.add_mesh(blk, style="wireframe", color="black", line_width=0.05, opacity=0.05)

    _apply_camera(pl)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pl.show(screenshot=str(out_png), auto_close=True)
    print(
        f"[storyboard] surface_classification -> {out_png}  "
        f"(T={counts['T']} U={counts['U']} C={counts['C']} "
        f"other={counts['other']} excluded={counts['excluded']})"
    )
    return out_png


# ============================================================================
# 6. PLOT 2 -- control-node displacement glyphs, from morph_vis.py
# ============================================================================
def plot_control_node_displacement(
    vtm_path=None,
    morph_config_path=None,
    surface_selectors=("T", "U"),
    scale=None,
    node_radius=0.3,
    out_png=None,
    off_screen=True,
):
    """Background T/U surfaces (semi-transparent) + original (black) vs
    displaced (red) control nodes joined by tubes coloured by |displacement|.
    Storyboard panel: "control-node graph, before -> after"."""
    morph_config_path = Path(morph_config_path or INPUTS["morph_config"])
    out_png = Path(out_png or OUTPUTS["control_node_displacement"])

    with open(morph_config_path, "r") as f:
        cfg = json.load(f)

    control_nodes = np.array(cfg["control_nodes"], dtype=float)
    displacements = np.array(cfg["displacement_vector"], dtype=float)
    t_ids = cfg.get("t_surfaces", [])
    u_ids = cfg.get("u_surfaces", [])
    c_ids = cfg.get("c_surfaces", [])

    if control_nodes.ndim == 1:
        control_nodes = control_nodes.reshape(-1, 3)
    if displacements.ndim == 1:
        displacements = displacements.reshape(-1, 3)
    if control_nodes.shape != displacements.shape:
        raise ValueError(
            f"control_nodes {control_nodes.shape} and displacement_vector "
            f"{displacements.shape} disagree."
        )
    N = control_nodes.shape[0]

    if vtm_path is None:
        vtm_path = INPUTS.get("vtm_baseline")
        if vtm_path is None or not Path(vtm_path).exists():
            vtm_path = _find_vtm_from_config(morph_config_path, cfg)
    mesh_obj = _load_vtm(vtm_path)
    blocks, resolved_names = _resolve_selected_blocks(
        mesh_obj, list(surface_selectors), t_ids, u_ids, c_ids
    )
    mesh = pv.MultiBlock(blocks).combine().extract_surface()

    b = np.array(mesh.bounds, float)
    ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    L = float(np.linalg.norm(ext)) or 1.0
    lmin = float(max(ext.min(), 1e-12))

    mags = np.linalg.norm(displacements, axis=1)
    dmax = float(mags.max() or 1.0)
    if scale is not None:
        auto_scale = float(scale)
    elif dmax < 0.05 * L:
        auto_scale = 0.02 * L / dmax
    else:
        auto_scale = 1.0

    disp_scaled = auto_scale * displacements
    targets = control_nodes + disp_scaled

    lift = 1e-3 * L
    cnP, tgtP = control_nodes.copy(), targets.copy()
    cnP[:, 2] += lift
    tgtP[:, 2] += lift

    pl = pv.Plotter(off_screen=off_screen, window_size=WINDOW_SIZE)
    pl.set_background(COLORS["background"])

    pl.add_mesh(mesh, color=COLORS["T"], opacity=0.7,
                label="+".join(resolved_names) if len(resolved_names) <= 3 else f"{len(resolved_names)} surfaces")
    pl.add_mesh(mesh, style="wireframe", color=COLORS["T"], opacity=0.3)

    # NOTE: node_radius is an absolute size in mesh units, same as the fixed
    # camera position above -- both are calibrated to your CRM/DSI case's
    # coordinate scale, not auto-derived from `lmin`/`L`. (The original
    # morph_vis.py computed a second radius `r2 = 0.3 * lmin` here but never
    # actually used it -- dead code, dropped.) If the spheres come out too
    # large/small for a different case, pass node_radius explicitly rather
    # than expecting it to auto-scale.
    sph = pv.Sphere(radius=node_radius)
    cn_glyphs = pv.PolyData(cnP).glyph(geom=sph, scale=False)
    tgt_glyphs = pv.PolyData(tgtP).glyph(geom=sph, scale=False)
    pl.add_mesh(cn_glyphs, color="black", lighting=False, label="Control nodes (orig)")
    pl.add_mesh(tgt_glyphs, color="red", lighting=False, label="Control nodes (displaced)")

    pts = np.vstack([cnP, tgtP])
    lines = np.hstack([[2, i, i + N] for i in range(N)]).astype(np.int64)
    segs = pv.PolyData(pts, lines=lines)
    segs.cell_data["disp_mag"] = mags
    pl.add_mesh(segs, scalars="disp_mag", cmap="viridis", line_width=3,
                render_lines_as_tubes=True, opacity=0.9, show_scalar_bar=True,
                label="Displacement vectors")

    _apply_camera(pl)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pl.show(screenshot=str(out_png), auto_close=True)
    print(
        f"[storyboard] control_node_displacement -> {out_png}  "
        f"(N={N}, max|disp|={dmax:.3e}, L={L:.3e}, scale x{auto_scale:.2f})"
    )
    return out_png


# ============================================================================
# 7. SCHEMATIC PLOTS -- illustrative, NOT tied to the real CRM mesh, from vis.py
#    Use these only for conceptual panels (e.g. "what a control-node graph
#    is", "XYZ vs normal-projected displacement"). They build their own
#    Delaunay connectivity on a flattened point cloud -- do not present this
#    as the real surface mesh; use plot_surface_classification for that.
# ============================================================================
def _load_schematic_nodes(control_nodes_path=None, control_normals_path=None):
    control_nodes_path = Path(control_nodes_path or INPUTS["control_nodes_npy"])
    control_normals_path = Path(control_normals_path or INPUTS["control_normals_npy"])
    cn = np.load(control_nodes_path).astype(float)
    normals = np.load(control_normals_path).astype(float)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    return cn, normals


def _build_flat_surface_connectivity(points):
    """PCA-flatten an (approximately planar) point cloud to 2D, Delaunay
    triangulate, and return (faces, edges, uv) indexed into the ORIGINAL points."""
    P = np.asarray(points, dtype=float)
    centroid = P.mean(axis=0)
    Q = P - centroid
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    uv = np.column_stack((Q @ e1, Q @ e2))
    faces = Delaunay(uv).simplices
    edges = set()
    for i, j, k in faces:
        edges.update({tuple(sorted((i, j))), tuple(sorted((j, k))), tuple(sorted((k, i)))})
    return faces, sorted(edges), uv


def _set_equal_3d(ax, points):
    mins, maxs = points.min(axis=0), points.max(axis=0)
    centre = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def _clean_axis(ax, title):
    ax.set_title(title)
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_schematic_control_nodes(control_nodes_path=None, out_png=None):
    """Bare control-node point cloud, no connectivity yet."""
    cn, _ = _load_schematic_nodes(control_nodes_path)
    out_png = Path(out_png or OUTPUTS["schematic_control_nodes"])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cn[:, 0], cn[:, 1], cn[:, 2], s=45, color=COLORS["T"])
    _clean_axis(ax, "Control nodes")
    _set_equal_3d(ax, cn)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[storyboard] schematic_control_nodes -> {out_png}")
    return out_png


def plot_schematic_connectivity(control_nodes_path=None, out_png=None):
    """Control nodes + generated Delaunay graph G=(V,E). Illustrates the
    *concept* of a control-node graph -- not the true CRM surface mesh."""
    cn, _ = _load_schematic_nodes(control_nodes_path)
    faces, edges, _ = _build_flat_surface_connectivity(cn)
    out_png = Path(out_png or OUTPUTS["schematic_connectivity"])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(cn[faces], alpha=0.18, facecolor=COLORS["T"]))
    ax.scatter(cn[:, 0], cn[:, 1], cn[:, 2], s=35, color=COLORS["T"])
    for i, j in edges:
        p = cn[[i, j]]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=0.8, color="black")
    _clean_axis(ax, "Control-node graph  G = (V, E)")
    _set_equal_3d(ax, cn)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[storyboard] schematic_connectivity -> {out_png}")
    return out_png


def plot_schematic_morph(
    mode="normal",
    control_nodes_path=None,
    control_normals_path=None,
    amp=20.0,
    seed=7,
    out_png=None,
):
    """Illustrative before/after: random XYZ displacement (mode='xyz') vs
    random normal-projected displacement (mode='normal'). Motivates why
    normal-projected deformation is preferred (ties to the
    Constructive/Deformative/Hybrid framing)."""
    if mode not in ("xyz", "normal"):
        raise ValueError("mode must be 'xyz' or 'normal'")

    cn, normals = _load_schematic_nodes(control_nodes_path, control_normals_path)
    faces, _, _ = _build_flat_surface_connectivity(cn)
    rng = np.random.default_rng(seed)

    if mode == "xyz":
        d = rng.normal(0.0, 1.5, size=cn.shape) * amp
        title = "Morphed surface: random XYZ displacement"
        default_out = OUTPUTS["schematic_morph_xyz"]
    else:
        scalar = rng.normal(0.0, 1.5, size=(len(cn), 1))
        d = amp * scalar * normals
        title = "Morphed surface: random normal displacement"
        default_out = OUTPUTS["schematic_morph_normal"]

    morphed = cn + d
    out_png = Path(out_png or default_out)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(cn[faces], alpha=0.12, facecolor=COLORS["T"]))
    ax.add_collection3d(Poly3DCollection(morphed[faces], alpha=0.35, facecolor=COLORS["C"]))
    ax.scatter(cn[:, 0], cn[:, 1], cn[:, 2], s=20, alpha=0.35, color=COLORS["T"])
    ax.scatter(morphed[:, 0], morphed[:, 1], morphed[:, 2], s=35, color=COLORS["C"])
    for p0, p1 in zip(cn, morphed):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                linestyle="--", linewidth=0.7, alpha=0.7, color="black")
    _clean_axis(ax, title)
    _set_equal_3d(ax, np.vstack([cn, morphed]))
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[storyboard] schematic_morph_{mode} -> {out_png}")
    return out_png


# ============================================================================
# 8. DRIVER -- generate every storyboard panel in one call
# ============================================================================
def generate_all_panels(skip_real_mesh=False, skip_schematic=False):
    """Run every plot function with the paths in INPUTS/OUTPUTS above.
    Each group is wrapped so a missing file in one group (e.g. you haven't
    exported control_nodes.npy yet) doesn't stop the other group from running."""
    made = {}

    if not skip_real_mesh:
        try:
            made["surface_classification"] = plot_surface_classification()
        except Exception as e:
            print(f"[storyboard][SKIP] surface_classification: {e}")
        try:
            made["control_node_displacement"] = plot_control_node_displacement()
        except Exception as e:
            print(f"[storyboard][SKIP] control_node_displacement: {e}")

    if not skip_schematic:
        try:
            made["schematic_control_nodes"] = plot_schematic_control_nodes()
            made["schematic_connectivity"] = plot_schematic_connectivity()
            made["schematic_morph_xyz"] = plot_schematic_morph(mode="xyz")
            made["schematic_morph_normal"] = plot_schematic_morph(mode="normal")
        except Exception as e:
            print(f"[storyboard][SKIP] schematic panels: {e}")

    print("\n[storyboard] Done. Panels written:")
    for k, v in made.items():
        print(f"  {k}: {v}")
    return made


if __name__ == "__main__":
    # On a headless HPC node with no working OSMesa/EGL VTK build, uncomment:
    # _ensure_headless_display()
    generate_all_panels()