import argparse, os, csv
import json
import numpy as np
import vtk
from paraview.simple import *
from paraview.servermanager import Fetch
from vtk.util import numpy_support as ns
try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

# ---- constants (keep aligned with your main script) ----
GAMMA = 1.4
NAME_RHO = "density"
NAME_ENERGY = "energy"
NAME_U = "velocity"

AIP_CENTER = (11.3, 0.4, 0.21)
AIP_RADIUS = 0.37
AIP_SELECTORS = [
    "/Root/Surfaces/Surface 111",
    "/Root/Surface 111",
    "/Root/Blocks/Surface 111",
    "/Root/Block_111",
]
cx, cy, cz = AIP_CENTER

def q(name):
    return f'"{name}"'

# ---------------- helpers ----------------
def load_monitor_config(path):
    if not path or not os.path.isfile(path):
        return {"interval": 50, "enabled": True, "monitors": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def _first_dataset_with_pointdata(ds):
    for leaf in _iter_leaf_datasets(ds):
        if hasattr(leaf, "GetPointData") and leaf.GetPointData() is not None:
            return leaf
    return None

def _get_point_array_numpy(proxy, name):
    ds = Fetch(proxy)
    leaf = _first_dataset_with_pointdata(ds)
    if leaf is None:
        raise RuntimeError(f"No point-data dataset found while looking for '{name}'.")

    arr = leaf.GetPointData().GetArray(name)
    if arr is None:
        raise RuntimeError(f"Point array '{name}' not found.")
    return ns.vtk_to_numpy(arr)

def _iter_leaf_datasets(ds):
    """Yield leaf datasets from either a single dataset or a multiblock dataset."""
    if ds is None:
        return

    if hasattr(ds, "GetNumberOfBlocks"):
        n = ds.GetNumberOfBlocks()
        for i in range(n):
            b = ds.GetBlock(i)
            if b is None:
                continue
            # recurse into nested multiblocks
            if hasattr(b, "GetNumberOfBlocks"):
                for bb in _iter_leaf_datasets(b):
                    yield bb
            else:
                yield b
    else:
        yield ds
        
def compute_projected_area_sum(polydata, direction="x"):
    """
    Fallback projected area:
    sum of projected triangle areas = sum(A * |n_dir|)

    This can overcount overlaps, but does not require shapely.
    """
    if polydata is None or polydata.GetNumberOfCells() == 0:
        return 0.0

    pts = ns.vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
    idir = {"x": 0, "y": 1, "z": 2}[direction.lower()]

    proj_area = 0.0

    for ic in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(ic)
        ids = cell.GetPointIds()
        if ids.GetNumberOfIds() != 3:
            continue

        i1, i2, i3 = ids.GetId(0), ids.GetId(1), ids.GetId(2)

        p1 = pts[i1]
        p2 = pts[i2]
        p3 = pts[i3]

        veca = p3 - p1
        vecb = p2 - p1
        normal_vec = np.cross(vecb, veca)

        mag = np.linalg.norm(normal_vec)
        if mag <= 0.0:
            continue

        area = 0.5 * mag
        normal = normal_vec / mag

        proj_area += area * abs(normal[idir])

    return float(proj_area)
        
def compute_projected_union_area(polydata, direction="x"):
    """
    Best projected area available:
    - use shapely union of projected triangles if available
    - otherwise fall back to summed projected facet area
    """
    if polydata is None or polydata.GetNumberOfCells() == 0:
        return 0.0

    if not HAS_SHAPELY:
        print("[MON][WARN] shapely not available; using projected-area sum fallback.", flush=True)
        return compute_projected_area_sum(polydata, direction=direction)

    pts = ns.vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)

    if direction.lower() == "x":
        keep = (1, 2)   # yz
    elif direction.lower() == "y":
        keep = (0, 2)   # xz
    elif direction.lower() == "z":
        keep = (0, 1)   # xy
    else:
        raise ValueError(f"Unsupported direction '{direction}'")

    polys = []

    for ic in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(ic)
        ids = cell.GetPointIds()
        if ids.GetNumberOfIds() != 3:
            continue

        i1, i2, i3 = ids.GetId(0), ids.GetId(1), ids.GetId(2)

        p1 = pts[i1]
        p2 = pts[i2]
        p3 = pts[i3]

        tri2d = [
            (float(p1[keep[0]]), float(p1[keep[1]])),
            (float(p2[keep[0]]), float(p2[keep[1]])),
            (float(p3[keep[0]]), float(p3[keep[1]])),
        ]

        poly = Polygon(tri2d)
        if poly.is_valid and poly.area > 0.0:
            polys.append(poly)

    if not polys:
        return 0.0

    union_poly = unary_union(polys)
    return float(union_poly.area)
        
def compute_freestream_properties(root):
    box = Box()
    box.XLength = 0.5
    box.YLength = 5.0
    box.ZLength = 5.0
    box.Center = [-8.0, 0.0, 0.0]
    UpdatePipeline(proxy=box)

    sampled = ResampleWithDataset(SourceDataArrays=root, DestinationMesh=box)
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    sampled.CellLocator = 'Static Cell Locator'
    UpdatePipeline(proxy=sampled)

    rho = _get_point_array_numpy(sampled, NAME_RHO).astype(float)
    energy = _get_point_array_numpy(sampled, NAME_ENERGY).astype(float)
    vel = _get_point_array_numpy(sampled, NAME_U).astype(float)

    vmag2 = np.sum(vel**2, axis=1)
    p = (energy - 0.5 * rho * vmag2) * (GAMMA - 1.0)

    good = np.isfinite(rho) & np.isfinite(p) & np.all(np.isfinite(vel), axis=1) & (rho > 0.0) & (p > 0.0)
    if not np.any(good):
        raise RuntimeError("No valid freestream samples.")

    rho_inf = float(np.median(rho[good]))
    p_inf   = float(np.median(p[good]))
    u_inf   = np.median(vel[good], axis=0).astype(float)
    q_inf   = 0.5 * rho_inf * float(np.dot(u_inf, u_inf))

    a_inf = np.sqrt(GAMMA * p_inf / rho_inf)
    mach_inf = float(np.linalg.norm(u_inf) / a_inf) if a_inf > 0 else float("nan")

    return {
        "rho_inf": rho_inf,
        "p_inf": p_inf,
        "u_inf": u_inf,
        "q_inf": q_inf,
        "mach_inf": mach_inf,
    }

def compute_surface_pressure_drag(root, surface_ids, direction="x", symmetry_factor=1, gamma=1.4):
    fs = compute_freestream_properties(root)
    p_inf = fs["p_inf"]
    q_inf = fs["q_inf"]

    surf_poly = isolate_surface_ids(root, surface_ids)

    # projected reference area from union of projected triangles
    proj_area = compute_projected_union_area(surf_poly, direction=direction)

    # Resample solution fields from root onto selected surface
    surf_src = TrivialProducer()
    surf_src.GetClientSideObject().SetOutput(surf_poly)
    UpdatePipeline(proxy=surf_src)

    sampled = ResampleWithDataset(SourceDataArrays=root, DestinationMesh=surf_src)
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    sampled.CellLocator = 'Static Cell Locator'
    UpdatePipeline(proxy=sampled)

    surf_pd, rho_nm = ensure_pointdata(sampled, NAME_RHO)
    surf_pd, e_nm   = ensure_pointdata(surf_pd, NAME_ENERGY)
    surf_pd, u_nm   = ensure_pointdata(surf_pd, NAME_U)
    UpdatePipeline(proxy=surf_pd)

    ds = Fetch(surf_pd)
    leaf = _first_dataset_with_pointdata(ds)
    if leaf is None:
        raise RuntimeError("No dataset found for drag computation.")

    pts = ns.vtk_to_numpy(leaf.GetPoints().GetData()).astype(float)
    rho = ns.vtk_to_numpy(leaf.GetPointData().GetArray(rho_nm)).astype(float)
    energy = ns.vtk_to_numpy(leaf.GetPointData().GetArray(e_nm)).astype(float)
    vel = ns.vtk_to_numpy(leaf.GetPointData().GetArray(u_nm)).astype(float)

    idir = {"x": 0, "y": 1, "z": 2}[direction.lower()]
    drag_force = 0.0

    for ic in range(leaf.GetNumberOfCells()):
        cell = leaf.GetCell(ic)
        ids = cell.GetPointIds()
        if ids.GetNumberOfIds() != 3:
            continue

        i1, i2, i3 = ids.GetId(0), ids.GetId(1), ids.GetId(2)

        p1 = pts[i1]
        p2 = pts[i2]
        p3 = pts[i3]

        veca = p3 - p1
        vecb = p2 - p1
        normal_vec = np.cross(vecb, veca)

        mag = np.linalg.norm(normal_vec)
        if mag <= 0.0:
            continue

        area = 0.5 * mag
        normal = normal_vec / mag

        press1 = (gamma - 1.0) * rho[i1] * (energy[i1] - 0.5 * np.dot(vel[i1], vel[i1]))
        press2 = (gamma - 1.0) * rho[i2] * (energy[i2] - 0.5 * np.dot(vel[i2], vel[i2]))
        press3 = (gamma - 1.0) * rho[i3] * (energy[i3] - 0.5 * np.dot(vel[i3], vel[i3]))

        p_face = (press1 + press2 + press3) / 3.0
        dp = p_face - p_inf

        drag_force += -dp * normal[idir] * area

    drag_force *= symmetry_factor

    drag_over_q = drag_force / q_inf if q_inf > 0.0 else float("nan")
    cd_proj = drag_force / (q_inf * proj_area) if (q_inf > 0.0 and proj_area > 0.0) else float("nan")

    return {
        "drag_force": float(drag_force),
        "q_inf": float(q_inf),
        "drag_over_q": float(drag_over_q),
        "proj_area": float(proj_area),
        "cd_proj": float(cd_proj),
        "p_inf": float(p_inf),
        "rho_inf": float(fs["rho_inf"]),
    }

def list_arrays(proxy):
    ds = Fetch(proxy)
    out = {}

    for leaf in _iter_leaf_datasets(ds):
        pd = leaf.GetPointData() if hasattr(leaf, "GetPointData") else None
        cd = leaf.GetCellData() if hasattr(leaf, "GetCellData") else None

        if pd:
            for i in range(pd.GetNumberOfArrays()):
                a = pd.GetArray(i)
                if a is not None and a.GetName():
                    out[a.GetName()] = (a.GetNumberOfComponents(), "POINTS")

        if cd:
            for i in range(cd.GetNumberOfArrays()):
                a = cd.GetArray(i)
                if a is not None and a.GetName():
                    out[a.GetName()] = (a.GetNumberOfComponents(), "CELLS")

    return out

def ensure_pointdata(proxy, name):
    """Return (proxy_with_point_array, array_name). Converts Cell->Point if needed."""
    arrs = list_arrays(proxy)
    if name in arrs and arrs[name][1] == "POINTS":
        return proxy, name
    if name in arrs and arrs[name][1] == "CELLS":
        try:
            c2p = CellDatatoPointData(Input=proxy)
        except:
            c2p = CellDataToPointData(Input=proxy)
        c2p.ProcessAllArrays = 1
        UpdatePipeline(proxy=c2p)
        return c2p, name
    raise RuntimeError(f"Array '{name}' not found on dataset.")

def _threshold_set(thr, loc, arr_name, lo, hi):
    try:
        thr.SelectInputScalars = [loc, arr_name]
    except:
        try:
            thr.Scalars = [loc, arr_name]
        except:
            pass
    try:
        thr.ThresholdMethod = 'Between'
        thr.LowerThreshold = float(lo)
        thr.UpperThreshold = float(hi)
    except:
        thr.ThresholdRange = (float(lo), float(hi))

def _threshold_surface_id(root, target_sid=111):
    arrs = list_arrays(root)
    for nm in ("surface_id", "SurfaceId", "SURFACE_ID", "SurfaceID", "ElementBlockIds"):
        if nm in arrs:
            ncomp, loc = arrs[nm]
            thr = Threshold(Input=root)
            _threshold_set(thr, loc, nm, target_sid - 0.5, target_sid + 0.5)
            UpdatePipeline(proxy=thr)
            ds = Fetch(thr)
            if ds and ds.GetNumberOfPoints() > 0:
                poly = ExtractSurface(Input=thr)
                UpdatePipeline(proxy=poly)
                return poly
    return None

def print_block_tree(ds, prefix="/Root", indent=0):
    if ds is None:
        return

    if hasattr(ds, "GetNumberOfBlocks"):
        n = ds.GetNumberOfBlocks()
        for i in range(n):
            block = ds.GetBlock(i)
            if block is None:
                continue

            name = None
            try:
                md = ds.GetMetaData(i)
                if md is not None and md.Has(vtk.vtkCompositeDataSet.NAME()):
                    name = md.Get(vtk.vtkCompositeDataSet.NAME())
            except Exception:
                pass

            if not name:
                name = f"Block_{i}"

            path = f"{prefix}/{name}"
            print("  " * indent + path, flush=True)

            if hasattr(block, "GetNumberOfBlocks"):
                print_block_tree(block, prefix=path, indent=indent + 1)

def _extract_block(root, selectors):
    for sel in selectors:
        try:
            eb = ExtractBlock(Input=root)
            eb.Selectors = [sel]
            UpdatePipeline(proxy=eb)
            ds = Fetch(eb)
            if ds and ds.GetNumberOfPoints() > 0:
                try:
                    mb = MergeBlocks(Input=eb)
                    UpdatePipeline(proxy=mb)
                    poly = ExtractSurface(Input=mb)
                except:
                    poly = ExtractSurface(Input=eb)
                UpdatePipeline(proxy=poly)
                return poly
        except:
            pass
    return None

def _pick_aip_geometric(root, cx, cy, cz, r, dx=0.02, cos_tol=0.90):
    surf = ExtractSurface(Input=root)
    UpdatePipeline(proxy=surf)

    try:
        norms = SurfaceNormals(Input=surf)
    except:
        norms = GenerateSurfaceNormals(Input=surf)
    norms.FeatureAngle = 180.0
    norms.ComputeCellNormals = 1
    UpdatePipeline(proxy=norms)

    c_absx = Calculator(Input=norms)
    c_absx.ResultArrayName = "abs_x_dist"
    c_absx.Function = f"abs(coordsX - {cx})"
    UpdatePipeline(proxy=c_absx)

    t_x = Threshold(Input=c_absx)
    _threshold_set(t_x, "POINTS", "abs_x_dist", 0.0, float(dx))
    UpdatePipeline(proxy=t_x)

    c_rad = Calculator(Input=t_x)
    c_rad.ResultArrayName = "r_from_center"
    c_rad.Function = f"sqrt((coordsY-{cy})^2 + (coordsZ-{cz})^2)"
    UpdatePipeline(proxy=c_rad)

    t_r = Threshold(Input=c_rad)
    _threshold_set(t_r, "POINTS", "r_from_center", 0.0, float(r))
    UpdatePipeline(proxy=t_r)

    arrs = list_arrays(t_r)
    if "Normals" in arrs and arrs["Normals"][1] == "CELLS":
        try:
            n2p = CellDatatoPointData(Input=t_r)
        except:
            n2p = CellDataToPointData(Input=t_r)
        n2p.ProcessAllArrays = 1
        UpdatePipeline(proxy=n2p)
        base_for_norm = n2p
    else:
        base_for_norm = t_r

    c_ax = Calculator(Input=base_for_norm)
    c_ax.ResultArrayName = "ax_align"
    c_ax.Function = "abs(Normals_X)"
    UpdatePipeline(proxy=c_ax)

    t_ax = Threshold(Input=c_ax)
    _threshold_set(t_ax, "POINTS", "ax_align", float(cos_tol), 1.0)
    UpdatePipeline(proxy=t_ax)

    ds_ax = Fetch(t_ax)
    out = ExtractSurface(Input=t_ax) if (ds_ax and ds_ax.GetNumberOfPoints() > 0) else ExtractSurface(Input=t_r)
    UpdatePipeline(proxy=out)

    cln = Clean(Input=out)
    UpdatePipeline(proxy=cln)
    return cln

def isolate_surface_ids(root, target_sids):
    ds = Fetch(root)
    if ds is None or not hasattr(ds, "GetNumberOfBlocks"):
        raise RuntimeError("Root dataset is not a multiblock dataset.")

    pieces = []

    print(f"[MON] Extracting blocks directly for surfaces: {target_sids}", flush=True)

    for sid in target_sids:
        # assume Surface N corresponds to Block_N
        if sid < 0 or sid >= ds.GetNumberOfBlocks():
            print(f"[MON][WARN] surface {sid} out of block range", flush=True)
            continue

        block = ds.GetBlock(sid)
        if block is None:
            print(f"[MON][WARN] block {sid} is None", flush=True)
            continue

        pieces.append(block)
        print(f"[MON] grabbed block {sid}", flush=True)

    if not pieces:
        raise RuntimeError(f"Could not isolate requested surfaces: {target_sids}")

    if len(pieces) == 1:
        merged_ds = pieces[0]
    else:
        append = vtk.vtkAppendFilter()
        for blk in pieces:
            append.AddInputData(blk)
        append.Update()
        merged_ds = append.GetOutput()

    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(merged_ds)
    geom.Update()
    poly = geom.GetOutput()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(clean.GetOutput())
    tri.Update()

    return tri.GetOutput()

def isolate_aip_surface(root, cx, cy, cz, r):
    surf = _extract_block(root, AIP_SELECTORS)
    if surf:
        return surf
    surf = _threshold_surface_id(root, 111)
    if surf:
        return surf
    surf = _pick_aip_geometric(root, cx, cy, cz, r)
    if surf:
        return surf
    raise RuntimeError("AIP isolation failed.")

def get_pressure_recovery_surface(root, monitor):
    sids = list(monitor.get("surface_ids", []))
    if sids:
        return isolate_surface_ids(root, sids)
    return isolate_aip_surface(root, cx, cy, cz, AIP_RADIUS)

def _ensure_pressure_on(proxy):
    px_rho, rho_nm = ensure_pointdata(proxy, NAME_RHO)
    px_e,   e_nm   = ensure_pointdata(px_rho, NAME_ENERGY)
    px_u,   u_nm   = ensure_pointdata(px_e,   NAME_U)

    calc = Calculator(Input=px_u)
    calc.ResultArrayName = "p_calc"
    calc.Function = f"({e_nm} - 0.5*{rho_nm}*mag({u_nm})^2) * ({GAMMA}-1)"
    UpdatePipeline(proxy=calc)
    return calc, "p_calc"

def compute_freestream_total_pressure(root):
    box = Box()
    box.XLength = 0.5
    box.YLength = 5.0
    box.ZLength = 5.0
    box.Center = [-8.0, 0.0, 0.0]
    UpdatePipeline(proxy=box)

    sampled = ResampleWithDataset(SourceDataArrays=root, DestinationMesh=box)
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    sampled.CellLocator = 'Static Cell Locator'
    UpdatePipeline(proxy=sampled)

    rho = _get_point_array_numpy(sampled, NAME_RHO).astype(float)
    energy = _get_point_array_numpy(sampled, NAME_ENERGY).astype(float)
    vel = _get_point_array_numpy(sampled, NAME_U).astype(float)

    vmag2 = np.sum(vel**2, axis=1)
    p = (energy - 0.5 * rho * vmag2) * (GAMMA - 1.0)

    good = np.isfinite(rho) & np.isfinite(p) & (rho > 0.0) & (p > 0.0)
    if not np.any(good):
        raise RuntimeError("No valid freestream samples for P0 computation.")

    a = np.sqrt(GAMMA * p[good] / rho[good])
    M = np.sqrt(vmag2[good]) / a
    P0 = p[good] * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) ** (GAMMA / (GAMMA - 1.0))

    return float(np.median(P0))

def compute_pressure_recovery(sampled_on_aip, P0_inf):
    rho = _get_point_array_numpy(sampled_on_aip, NAME_RHO).astype(float)
    energy = _get_point_array_numpy(sampled_on_aip, NAME_ENERGY).astype(float)
    vel = _get_point_array_numpy(sampled_on_aip, NAME_U).astype(float)

    vmag2 = np.sum(vel**2, axis=1)
    p = (energy - 0.5 * rho * vmag2) * (GAMMA - 1.0)

    good = np.isfinite(rho) & np.isfinite(p) & (rho > 0.0) & (p > 0.0)
    if not np.any(good):
        raise RuntimeError("No valid AIP samples for pressure-recovery computation.")

    a = np.sqrt(GAMMA * p[good] / rho[good])
    M = np.sqrt(vmag2[good]) / a
    P0_AIP = 0.9 * p[good] * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) ** (GAMMA / (GAMMA - 1.0))

    P0_mean = float(np.mean(P0_AIP))
    recovery = P0_mean / P0_inf if P0_inf != 0.0 else float("nan")
    return P0_mean, recovery

def compute_surface_force_coefficient(root, surface_ids, mach, direction="x", symmetry_factor=1, gamma=1.4):
    surf_poly = isolate_surface_ids(root, surface_ids)

    # Resample solution fields from root onto selected surface
    surf_src = TrivialProducer()
    surf_src.GetClientSideObject().SetOutput(surf_poly)
    UpdatePipeline(proxy=surf_src)

    sampled = ResampleWithDataset(SourceDataArrays=root, DestinationMesh=surf_src)
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    sampled.CellLocator = 'Static Cell Locator'
    UpdatePipeline(proxy=sampled)

    surf_pd, rho_nm = ensure_pointdata(sampled, NAME_RHO)
    surf_pd, e_nm   = ensure_pointdata(surf_pd, NAME_ENERGY)
    surf_pd, u_nm   = ensure_pointdata(surf_pd, NAME_U)
    UpdatePipeline(proxy=surf_pd)

    ds = Fetch(surf_pd)
    leaf = _first_dataset_with_pointdata(ds)
    if leaf is None:
        raise RuntimeError("No dataset found for force computation.")

    pts = ns.vtk_to_numpy(leaf.GetPoints().GetData()).astype(float)
    rho = ns.vtk_to_numpy(leaf.GetPointData().GetArray(rho_nm)).astype(float)
    energy = ns.vtk_to_numpy(leaf.GetPointData().GetArray(e_nm)).astype(float)
    vel = ns.vtk_to_numpy(leaf.GetPointData().GetArray(u_nm)).astype(float)

    idir = {"x": 0, "y": 1, "z": 2}[direction.lower()]
    cp_inf = 1.0 / (gamma * mach * mach)

    total_force_coeff = 0.0

    for ic in range(leaf.GetNumberOfCells()):
        cell = leaf.GetCell(ic)
        ids = cell.GetPointIds()
        if ids.GetNumberOfIds() != 3:
            continue

        i1, i2, i3 = ids.GetId(0), ids.GetId(1), ids.GetId(2)

        p1 = pts[i1]
        p2 = pts[i2]
        p3 = pts[i3]

        veca = p3 - p1
        vecb = p2 - p1
        normal_vec = np.cross(vecb, veca)

        mag = np.linalg.norm(normal_vec)
        if mag <= 0.0:
            continue

        area = 0.5 * mag
        normal = normal_vec / mag

        press1 = (gamma - 1.0) * rho[i1] * (energy[i1] - 0.5 * np.dot(vel[i1], vel[i1]))
        press2 = (gamma - 1.0) * rho[i2] * (energy[i2] - 0.5 * np.dot(vel[i2], vel[i2]))
        press3 = (gamma - 1.0) * rho[i3] * (energy[i3] - 0.5 * np.dot(vel[i3], vel[i3]))

        p_face = (press1 + press2 + press3) / 3.0
        cp_face = p_face - cp_inf

        total_force_coeff += -area * cp_face * normal[idir]

    return float(symmetry_factor * total_force_coeff)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--mach", type=float, default=1.0)
    ap.add_argument("--monitors", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    r = EnSightReader(CaseFileName=args.case)
    UpdatePipeline(proxy=r)
    
    cfg = load_monitor_config(args.monitors)

    print("Available arrays on root:", list_arrays(r), flush=True)
    results = {}
    
    for mon in cfg.get("monitors", []):
        if not mon.get("enabled", True):
            continue

        mtype = str(mon.get("type", "")).strip().lower()
        name = str(mon.get("name", mtype)).strip() or mtype

        try:
            if mtype == "pressure_recovery":
                pr_geom = get_pressure_recovery_surface(r, mon)
                UpdatePipeline(proxy=pr_geom)

                sampled = ResampleWithDataset(SourceDataArrays=r, DestinationMesh=pr_geom)
                sampled.PassPointArrays = 1
                sampled.PassCellArrays = 1
                sampled.CellLocator = 'Static Cell Locator'
                UpdatePipeline(proxy=sampled)

                P0_inf = compute_freestream_total_pressure(r)
                P0_mean, pr = compute_pressure_recovery(sampled, P0_inf)

                results["P0_inf_median_Pa"] = P0_inf
                results["P0_mean_AIP_Pa"] = P0_mean
                results[name] = pr

            elif mtype == "drag":
                drag_res = compute_surface_pressure_drag(
                    root=r,
                    surface_ids=mon.get("surface_ids", []),
                    direction=mon.get("direction", "x"),
                    symmetry_factor=int(mon.get("symmetry_factor", 1)),
                    gamma=GAMMA,
                )

                results[name + "_force"] = drag_res["drag_force"]
                results[name + "_over_q"] = drag_res["drag_over_q"]
                results[name + "_proj_area"] = drag_res["proj_area"]
                results[name + "_cd_proj"] = drag_res["cd_proj"]

            else:
                print(f"[MON][WARN] Unsupported monitor type: {mtype}", flush=True)

        except Exception as e:
            print(f"[MON][WARN] Failed to compute {name}: {e}", flush=True)
            results[name] = float("nan")
            
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_header = not (args.append and os.path.isfile(args.out))

    fieldnames = ["iter"] + list(results.keys())

    with open(args.out, "a" if args.append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        row = {"iter": args.iter}
        row.update(results)
        w.writerow(row)

if __name__ == "__main__":
    main()
