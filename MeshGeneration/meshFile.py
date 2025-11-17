from abc import ABC, abstractmethod
import os, sys
import pyvista as pv
import numpy as np
sys.path.append(os.path.dirname('ConvertFileType'))
sys.path.append(os.path.dirname('ShapeParameterization'))
from ConvertFileType.convertToStep import export_nurbs_surface_to_step
from ShapeParameterization.surfaceFitting import fitSurfaceNURBs

import pyvista as pv
import re
import fnmatch

def _normalize_name(name: str) -> str:
    # Make matching robust across exporters
    s = name.strip()
    s = s.replace("\\", "/")
    if s.startswith("Base/"):
        s = s[5:]
    if s.endswith("-dom"):
        s = s[:-4]
    return s

def _build_name_index(blocks):
    """
    blocks: List[(name, pv.DataSet)]
    Returns:
        norm_name -> original_name
        original_name -> norm_name
        available_norm_names: set
    """
    norm2orig = {}
    orig2norm = {}
    for orig, _ in blocks:
        norm = _normalize_name(orig)
        # keep the first original seen for that norm key
        norm2orig.setdefault(norm, orig)
        orig2norm[orig] = norm
    return norm2orig, orig2norm, set(norm2orig.keys())

def _match_specs_to_names(specs, available_names):
    """
    specs can include:
      - exact normalized names: 'Forebody-3'
      - globs: 'Forebody-*', '*Nacelle*'
      - regex: '/^FIXED-.*Wing.*/'
      - integer IDs as strings: '12' (treated as integer surface_id)
    Returns a set of normalized names that match.
    """
    matched = set()
    if specs is None:
        return matched

    for spec in specs:
        if spec is None:
            continue

        # integer ID support like "12" (rarely used if names are strings)
        if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
            # Caller can map these to cell_data['surface_id'] if needed.
            # Here we skip ID->name translation because names are authoritative.
            # Keep this branch if you also support integer-based datasets.
            continue

        if isinstance(spec, str):
            s = spec.strip()

            # regex form: '/.../'
            if len(s) >= 2 and s[0] == "/" and s[-1] == "/":
                pattern = s[1:-1]
                rx = re.compile(pattern)
                for nm in available_names:
                    if rx.search(nm):
                        matched.add(nm)
                continue

            # glob form: contains wildcards
            if "*" in s or "?" in s or "[" in s:
                for nm in available_names:
                    if fnmatch.fnmatch(nm, s):
                        matched.add(nm)
                continue

            # exact name: normalize and match
            matched_name = _normalize_name(s)
            if matched_name in available_names:
                matched.add(matched_name)
            else:
                # allow partial contains for convenience (optional)
                for nm in available_names:
                    if matched_name == nm or matched_name in nm:
                        matched.add(nm)
    return matched

def _safe_concat(chunks, axis=0, what="arrays"):
    chunks = [c for c in (chunks or []) if c is not None and np.size(c) > 0]
    if not chunks:
        raise RuntimeError(f"Cannot concatenate: no {what} found (empty).")
    return np.concatenate(chunks, axis=axis)

def _flatten_multiblock(obj):
    """Yield leaf datasets from a (possibly nested) MultiBlock tree."""
    if isinstance(obj, pv.MultiBlock):
        for _, child in obj.items():
            yield from _flatten_multiblock(child)
    elif isinstance(obj, pv.DataSet):
        yield obj
    else:
        # Unknown/None nodes — skip
        return
    
# --- fix this helper ---
def _iter_leaves(obj, prefix=""):
    """Yield (name, dataset) for all leaf DataSets in a (possibly nested) MultiBlock."""
    if isinstance(obj, pv.MultiBlock):
        # .items() yields (key, child). key may be '' or None.
        for idx, (key, child) in enumerate(obj._items()):
            part = str(key).strip() if key not in (None, "") else f"block_{idx}"
            new_prefix = f"{prefix}{part}/" if prefix else f"{part}/"
            yield from _iter_leaves(child, new_prefix)
    elif isinstance(obj, pv.DataSet):
        name = prefix[:-1] if prefix.endswith("/") else (prefix or "block")
        yield name, obj
    else:
        return

class Mesh(ABC):
    def __init__(self, filepath):
        self.filepath = filepath
        self.surfaces = {}
        self.global_ids = {}
        self.local_ids = {}

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_surface_points(self, name):
        pass

    @abstractmethod
    def get_surface_mesh(self, name):
        """Return a renderable PyVista mesh for the named surface."""
        pass

    @abstractmethod
    def get_all_points(self):
        pass

    @abstractmethod
    def get_surface_names(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def export_step(self, path, surface_name=None):
        pass
    
    @abstractmethod
    def _collect_surface_nodes(self, surface_list):
        pass


class VtkMesh(Mesh):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.mesh = None
        self.points = None

    def load(self):
        self.mesh = pv.read(self.filepath)
        self.points = self.mesh.points
        n_pts = self.points.shape[0]
        
        self.surfaces["full"] = np.arange(n_pts)
        self.global_ids["full"] = np.arange(n_pts)
        self.local_ids["full"] = np.arange(n_pts)

    def get_surface_points(self, name="full"):
        return self.points[self.surfaces[name]]

    def get_surface_mesh(self, name="full"):
        return self.mesh

    def get_all_points(self):
        return self.points

    def get_surface_names(self):
        return list(self.surfaces.keys())

    def save(self, path):
        self.mesh.save(path)
        print(f"Saved mesh to {path}")

    def export_step(self, path, surface_name="full"):
        pts = self.get_surface_points(surface_name)
        surf = fitSurfaceNURBs(pts)
        export_nurbs_surface_to_step(surf, path, format='step')
        print(f"Exported {surface_name} to {path}")
        
    def _collect_surface_nodes(self, surface_list):
        nodes = []
        for name in surface_list:
            try:
                pts = self.get_surface_points(name)
                nodes.append(pts)
            except Exception as e:
                print(f"Warning: could not get surface '{name}': {e}")
        if nodes:
            return np.vstack(nodes)
        else:
            return np.zeros((0, 3))


class VtmMesh(Mesh):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.data = None
        self.blocks = []

    def _norm(self, s: str) -> str:
        return str(s).strip().lower().replace(" ", "_")

    def _resolve_name_or_id(self, name_or_id):
        # 1) numeric ID (int or '123')
        if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and name_or_id.isdigit()):
            sid = int(name_or_id)
            if sid in self.id_to_surface:           # map: id -> surface object/data
                return sid
            # Some code bases store label per id:
            if hasattr(self, "id_to_label") and sid in self.id_to_label:
                return sid
            raise ValueError(f"Surface id {sid} not found.")

        # 2) exact label
        if hasattr(self, "label_to_id") and name_or_id in self.label_to_id:
            return self.label_to_id[name_or_id]

        # 3) normalized label
        key = self._norm(name_or_id)
        if hasattr(self, "norm_label_to_id") and key in self.norm_label_to_id:
            return self.norm_label_to_id[key]

        raise ValueError(f"Surface '{name_or_id}' not found (by id, original label, or normalized label).")

    def load(self):
        reader = pv.get_reader(self.filepath)
        root = reader.read()
        
        self.blocks = []
        self.tagged_blocks = []
        self.global_ids = {}
        self.local_ids  = {}
        self.surfaces   = {}
        self._global_id_map = []

        if isinstance(root, pv.DataSet) and not isinstance(root, pv.MultiBlock):
            leaves = [("block_0", root)]
        else:
            leaves = list(_iter_leaves(root))
        
        # Ensure unique names even if exporter duplicated keys
        seen = {}
        def _unique(n):
            k = n; cnt = 1
            while k in seen:
                cnt += 1
                k = f"{n}__{cnt}"
            seen[k] = True
            return k

        global_counter = 0
        surface_id_counter = 0

        for raw_name, ds in leaves:
            if ds is None:
                continue

            n_cells  = int(getattr(ds, "n_cells", 0) or 0)
            n_points = int(getattr(ds, "n_points", 0) or 0)
            # keep datasets that have either cells OR points
            if (n_cells + n_points) == 0:
                continue

            blk = ds.copy()
            # Tag each leaf with a unique surface id
            if n_cells > 0:
                blk.cell_data["surface_id"] = np.full(n_cells, surface_id_counter, dtype=np.int32)
            surface_id_counter += 1

            name = _unique(raw_name)
            
            self.surface_id_map = getattr(self, "surface_id_map", {})
            self.surface_name_map = getattr(self, "surface_name_map", {})
            self.surface_id_map[name] = int(surface_id_counter)
            self.surface_name_map[raw_name] = int(surface_id_counter)
            # if you also build normalized names, map those too:
            if hasattr(self, "orig2norm"):
                norm = self.orig2norm.get(name)
                if norm:
                    self.surface_id_map[norm] = int(surface_id_counter)
            
            self.friendly_names = {}
            for bname, sid in self.surface_name_map.items():
                # use normalized key for consistency
                self.friendly_names[bname] = sid
                        
            num_pts = n_points
            self.local_ids[name]  = np.arange(num_pts, dtype=np.int64)
            self.global_ids[name] = np.arange(global_counter, global_counter + num_pts, dtype=np.int64)
            self.surfaces[name]   = np.arange(num_pts, dtype=np.int64)
            global_counter += num_pts

            self.blocks.append((name, blk))
            self.tagged_blocks.append(blk)

        # ---- build normalized name indices *after* blocks & surfaces are complete ----
        self.norm2orig, self.orig2norm, self.available_norm_names = _build_name_index(self.blocks)
        self.surfaces_norm = {}
        for orig_name, idxs in self.surfaces.items():
            # guard against any weird names
            if orig_name in self.orig2norm:
                self.surfaces_norm[self.orig2norm[orig_name]] = idxs

        # ---- merge to unstructured grid (optional) ----
        if self.tagged_blocks:
            combined = None
            for _, blk in self.blocks:
                ug = blk.cast_to_unstructured_grid()
                combined = ug if combined is None else combined.merge(ug, merge_points=False)
            self.combined_mesh = combined
        else:
            self.combined_mesh = None

        # ---- build ID/label lookup maps ----
        self.id_to_surface = {}
        self.label_to_id = {}
        self.norm_label_to_id = {}
        self.id_to_label = {}

        for bname, blk in self.blocks:
            sid = self.surface_id_map[bname]
            self.id_to_surface[sid] = blk
            self.label_to_id[bname] = sid
            norm = _normalize_name(bname)
            self.norm_label_to_id[norm] = sid
            self.id_to_label[sid] = bname

    # resolve T/U/C names whether user passes original or normalized
    def _resolve_name(self, name: str) -> str:
        # direct hit on original name?
        if any(name == bname for bname, _ in self.blocks):
            return name
        # try normalized mapping
        nm = _normalize_name(name)
        if hasattr(self, "norm2orig") and nm in self.norm2orig:
            return self.norm2orig[nm]
        raise ValueError(f"Surface '{name}' not found (original or normalized).")

    def get_surface_points(self, name_or_id):
        sid = self._resolve_name_or_id(name_or_id)
        # proceed using sid to fetch points
        surf = self.id_to_surface[sid]
        return surf.points
    
    def get_surface_id(self, name_or_id):
        """Accepts original name, normalized name, or an int/str(int). Returns integer surface ID."""
        # Already an int
        if isinstance(name_or_id, int):
            return name_or_id
        # Numeric string
        s = str(name_or_id)
        if s.isdigit():
            return int(s)
        # Try direct map
        if hasattr(self, "surface_id_map") and s in self.surface_id_map:
            return int(self.surface_id_map[s])
        # Try normalized
        if hasattr(self, "orig2norm"):
            norm = self.orig2norm.get(s) or s
            if hasattr(self, "surface_id_map") and norm in self.surface_id_map:
                return int(self.surface_id_map[norm])
        # Fallback: inspect the block’s cell_data if present
        for bname, blk in getattr(self, "blocks", []):
            if bname == s:
                if "surface_id" in blk.cell_data and blk.n_cells > 0:
                    return int(np.asarray(blk.cell_data["surface_id"])[0])
                # if you stored it earlier:
                if hasattr(self, "surface_id_map") and bname in self.surface_id_map:
                    return int(self.surface_id_map[bname])
        raise KeyError(f"Surface ID not found for '{name_or_id}'")

    def get_surface_mesh(self, name):
        resolved = self._resolve_name(name)
        for blk_name, blk in self.blocks:
            if blk_name == resolved:
                return blk
        raise ValueError(f"Surface '{name}' not found.")

    def resolve_group_indices(self, specs, *, group_label="T"):
        """
        specs: list of strings (names, globs, regex '/.../') or ints (ignored unless you implement ID mapping)
        Returns: np.ndarray of unique point indices for that group across matched blocks.
        """
        matched_norm = _match_specs_to_names(specs, self.available_norm_names)

        if not matched_norm:
            raise RuntimeError(
                f"[Morph] Surface group '{group_label}' resolved to EMPTY.\n"
                f"  Specs: {specs}\n"
                f"  Available: {sorted(self.available_norm_names)}"
            )

        idx_chunks = []
        for nm in matched_norm:
            if nm not in self.surfaces_norm:
                # no points registered for this name; skip loudly
                print(f"[WARN] Group {group_label}: matched '{nm}' but no points recorded.")
                continue
            idx_chunks.append(self.surfaces_norm[nm])

        return _safe_concat(idx_chunks, axis=0, what=f"{group_label} point-indices")

    def get_all_points(self):
        return np.vstack([blk.points for _, blk in self.blocks])

    def get_surface_names(self):
        return list(self.surfaces.keys())
    
    def get_surface_name(self, name_or_id):
        """
        Return the original block name for a given surface ID or name.
        - If given an int (or numeric string), map ID -> name.
        - If given a name, return the resolved original name.
        """
        # numeric id
        if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and name_or_id.isdigit()):
            sid = int(name_or_id)
            if hasattr(self, "id_to_label") and sid in self.id_to_label:
                return self.id_to_label[sid]
            # fallback: scan blocks (shouldn't be needed if id_to_label is built)
            for bname, blk in getattr(self, "blocks", []):
                if "surface_id" in blk.cell_data and blk.n_cells > 0:
                    if int(np.asarray(blk.cell_data["surface_id"])[0]) == sid:
                        return bname
            raise KeyError(f"Surface name not found for id={sid}")
        # already a name: resolve and return original
        return self._resolve_name(name_or_id)

    def get_surface_mesh(self, name_or_id):
        """
        Accepts either a surface name (original or normalized) or an integer ID.
        Returns the leaf dataset (PyVista) for that surface.
        """
        # allow integer id
        if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and str(name_or_id).isdigit()):
            sid = int(name_or_id)
            if hasattr(self, "id_to_surface") and sid in self.id_to_surface:
                return self.id_to_surface[sid]
            raise KeyError(f"Surface dataset not found for id={sid}")

        # name path (existing behavior)
        resolved = self._resolve_name(name_or_id)
        for blk_name, blk in self.blocks:
            if blk_name == resolved:
                return blk
        raise ValueError(f"Surface '{name_or_id}' not found.")

    def save(self, path):
        multiblock = pv.MultiBlock([blk for _, blk in self.blocks])
        multiblock.save(path)
        print(f"Saved multi-block mesh to {path}")

    def export_step(self, path, surface_name):
        pts = self.get_surface_points(surface_name)
        surf = fitSurfaceNURBs(pts)
        export_nurbs_surface_to_step(surf, path, format='step')
        print(f"Exported {surface_name} to {path}")
    
    def _collect_surface_nodes(self, surface_list):
        nodes = []
        for name in surface_list:
            try:
                pts = self.get_surface_points(name)
                nodes.append(pts)
            except Exception as e:
                print(f"Warning: could not get surface '{name}': {e}")
        if nodes:
            return np.vstack(nodes)
        else:
            return np.zeros((0, 3))

    def assign_global_ids_by_coordinates(self):
        """
        Assigns consistent global node IDs across all surfaces by matching rounded coordinates.
        Shared coordinates get the same global ID.
        """
        coord_to_gid = {}
        next_gid = 0
        self.global_ids = {}
        self.local_ids = {}

        for name, blk in self.blocks:
            if blk is None or blk.n_points == 0:
                self.global_ids[name] = np.array([], dtype=int)
                self.local_ids[name] = np.array([], dtype=int)
                continue

            points = blk.points
            gids = []
            lids = []

            for i, pt in enumerate(points):
                key = tuple(np.round(pt, decimals=6))
                if key not in coord_to_gid:
                    coord_to_gid[key] = next_gid
                    next_gid += 1
                gids.append(coord_to_gid[key])
                lids.append(i)

            self.global_ids[name] = np.array(gids)
            self.local_ids[name] = np.array(lids)

# Factory function
def load_mesh(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".vtm", ".case"):          # ✅ treat EnSight case like a MultiBlock
        mesh = VtmMesh(filepath)
    elif ext == ".vtk":
        mesh = VtkMesh(filepath)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    mesh.load()
    mesh.assign_global_ids_by_coordinates()
    return mesh