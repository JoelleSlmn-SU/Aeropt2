# convertVtmtoFro.py
import pyvista as pv
import numpy as np
import os, sys

sys.path.append(os.path.dirname("FileRW"))
from FileRW.FroFile import FroFile
from FileRW.Mesh import Mesh

# ---- helpers ----
def _iter_leaves(obj, prefix=""):
    """
    Yield leaf datasets from a (possibly nested) MultiBlock using index-based API.
    Returns tuples of (hier_name, dataset).
    """
    import pyvista as pv
    if isinstance(obj, pv.MultiBlock):
        for i in range(len(obj)):
            child = obj[i]
            key = obj.get_block_name(i) or f"block_{i}"
            part = str(key).strip()
            new_prefix = f"{prefix}{part}/" if prefix else f"{part}/"
            # Recurse
            yield from _iter_leaves(child, new_prefix)
    elif isinstance(obj, pv.DataSet):
        # Trim the trailing slash
        name = prefix[:-1] if prefix.endswith("/") else (prefix or "block")
        yield name, obj
    else:
        return

def _get_surface_ids(block, n_cells, fallback_id):
    """Case-insensitive lookup for surface id-like arrays; else fallback."""
    if n_cells == 0:
        return np.array([], dtype=int)
    # try common keys
    for key in ("SurfaceID", "surface_id", "surfaceId", "SURFACE_ID"):
        if key in block.cell_data:
            arr = np.asarray(block.cell_data[key]).ravel()
            if arr.size == n_cells:
                # optional +1 to reserve 0 for 'unknown'
                return arr.astype(int) + 1
    # if block has exactly one cell array, you might want to use it:
    if len(block.cell_data) == 1:
        arr = np.asarray(list(block.cell_data.values())[0]).ravel()
        if arr.size == n_cells:
            return arr.astype(int) + 1
    # fallback: unique per-leaf id
    return np.full(n_cells, int(fallback_id), dtype=int)

# ---- single VTK -> Fro ----
def vtk_to_fro(vtk_path, outpath=None, write_file=True):
    import pyvista as pv
    from FileRW.FroFile import FroFile
    from FileRW.Mesh import Mesh

    ds = pv.read(vtk_path)
    print(ds)
    
    if not isinstance(ds, pv.PolyData):
        ds = ds.extract_surface()

    # Force triangulation of ANY polygons
    ds = ds.triangulate()

    # 🔧 Assign a dummy surface_id if none exists
    if "surface_id" not in ds.cell_data:
        ds.cell_data["surface_id"] = np.zeros(ds.n_cells, dtype=np.int32)

    print("n_cells:", ds.n_cells, "n_points:", ds.n_points)
    
    mesh_obj = Mesh.from_pyvista(ds)
    fro_obj = FroFile.fromMesh(mesh_obj)

    if write_file and outpath:
        fro_obj.write_file(outpath)
        print("File Written to ", f"{outpath}")

    return fro_obj


# ---- VTM -> Fro ----
def vtm_to_fro(vtm_path, outpath=None, write_file=True, tol=1e-6):
    """
    Convert a .vtm multiblock mesh to a .fro file with deduplicated nodes
    and consistent surface IDs, preserving the original structure.
    """
    import pyvista as pv
    import numpy as np
    from FileRW.FroFile import FroFile
    from FileRW.Mesh import Mesh

    root = pv.read(vtm_path)
    leaves = list(_iter_leaves(root))
    if not leaves:
        raise RuntimeError("[VTM→FRO] No valid leaf datasets found in file.")

    all_pts = []
    all_tris, all_quads = [], []
    coord_map = {}   # key: rounded coord tuple → global id
    next_gid = 0

    def get_gid(pt):
        nonlocal next_gid
        key = tuple(np.round(pt, 6))
        if key not in coord_map:
            coord_map[key] = next_gid
            all_pts.append(pt)
            next_gid += 1
        return coord_map[key]

    surface_id_counter = 1

    for leaf_name, leaf in leaves:
        if leaf is None or not hasattr(leaf, "points") or leaf.n_points == 0:
            continue

        points = np.asarray(leaf.points)
        cells_dict = getattr(leaf, "cells_dict", {})
        n_cells = int(getattr(leaf, "n_cells", 0) or 0)
        if n_cells == 0:
            continue

        # --- get surface IDs properly ---
        if "surface_id" in leaf.cell_data:
            surf_ids = np.asarray(leaf.cell_data["surface_id"]).astype(int)
        elif "SurfaceID" in leaf.cell_data:
            surf_ids = np.asarray(leaf.cell_data["SurfaceID"]).astype(int)
        else:
            surf_ids = np.full(n_cells, surface_id_counter, int)
            surface_id_counter += 1

        # --- triangles ---
        if 5 in cells_dict:
            tris = cells_dict[5]
            for i, tri in enumerate(tris):
                gids = [get_gid(points[idx]) for idx in tri]
                sid = int(surf_ids[i % len(surf_ids)])
                all_tris.append([gids[0]+1, gids[1]+1, gids[2]+1, sid])  # 1-based indexing

        # --- quads ---
        if 9 in cells_dict:
            quads = cells_dict[9]
            for i, quad in enumerate(quads):
                gids = [get_gid(points[idx]) for idx in quad]
                sid = int(surf_ids[i % len(surf_ids)])
                all_quads.append([gids[0]+1, gids[1]+1, gids[2]+1, gids[3]+1, sid])

        max_index = 0
        if all_tris:
            max_index = max(max_index, np.max(np.array(all_tris)[:,:-1]))
        if all_quads:
            max_index = max(max_index, np.max(np.array(all_quads)[:,:-1]))
        assert max_index < len(all_pts), (
            f"[VTM→FRO] Out-of-bounds connectivity: max node index {max_index} >= {len(all_pts)} nodes"
        )
        
        # --- poly fallback ---
        # --- triangles (0-based internally) ---
        if 5 in cells_dict:
            tris = cells_dict[5]
            for i, tri in enumerate(tris):
                gids = [get_gid(points[idx]) for idx in tri]
                sid = int(surf_ids[i % len(surf_ids)])
                all_tris.append([gids[0], gids[1], gids[2], sid])

        # --- quads (0-based internally) ---
        if 9 in cells_dict:
            quads = cells_dict[9]
            for i, quad in enumerate(quads):
                gids = [get_gid(points[idx]) for idx in quad]
                sid = int(surf_ids[i % len(surf_ids)])
                all_quads.append([gids[0], gids[1], gids[2], gids[3], sid])
                
    max_index = 0
    if all_tris:
        max_index = max(max_index, np.max(np.array(all_tris)[:, :-1]))
    if all_quads:
        max_index = max(max_index, np.max(np.array(all_quads)[:, :-1]))

    if max_index >= len(all_pts):
        raise RuntimeError(
            f"[VTM→FRO] Connectivity out of range: max node index {max_index} for {len(all_pts)} nodes"
        )

    # --- build final Mesh / FroFile objects ---
    m = Mesh()
    m.nodes = np.array(all_pts, dtype=float)
    m.node_count = len(all_pts)
    m.boundary_triangles = np.array(all_tris, dtype=int)
    m.boundary_triangle_count = len(all_tris)
    m.boundary_quads = np.array(all_quads, dtype=int)
    m.boundary_quad_count = len(all_quads)
    m.surface_count = len(set([t[-1] for t in all_tris] + [q[-1] for q in all_quads]))

    # shift to 1-based indexing only when writing
    for arr in (m.boundary_triangles, m.boundary_quads):
        if arr.size:
            arr[:, :-1] += 1

    fro_obj = FroFile.fromMesh(m)

    if write_file and outpath:
        fro_obj.write_file(outpath)
        print(f"[✔] Wrote FRO: {outpath}")
        print(f"Nodes: {m.node_count:,}, Triangles: {m.boundary_triangle_count:,}, Quads: {m.boundary_quad_count:,}, Surfaces: {m.surface_count:,}")

    return fro_obj



def check_fro_bounds(fro_path):
    pts, faces = [], []
    with open(fro_path) as f:
        for line in f:
            parts = line.split()
            if len(parts)==3:
                try: float(parts[0])
                except: continue
                pts.append(parts)
            elif len(parts) in (4,5) and all(p.isdigit() for p in parts):
                faces.append([int(p) for p in parts[:-1]])
    pts = np.array(pts,float)
    faces = np.array(faces,int)
    if len(faces)==0:
        print("No faces found.")
        return
    max_idx = np.max(faces)
    print(f"Nodes: {len(pts)}, Max index: {max_idx}")
    if max_idx>len(pts):
        print("❌ Out-of-bounds indices detected!")
    else:
        print("✅ All connectivity indices within range.")
