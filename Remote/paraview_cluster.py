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
    p_inf   = fs["p_inf"]
    q_inf   = fs["q_inf"]
    rho_inf = fs["rho_inf"]
    u_inf   = fs["u_inf"]

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
    
    vmag2 = np.sum(vel**2, axis=1)

    # assuming conservative total energy
    p_A = (gamma - 1.0) * (energy - 0.5 * rho * vmag2)

    # assuming specific total energy
    p_B = (gamma - 1.0) * rho * (energy - 0.5 * vmag2)

    print("---- PRESSURE DEBUG ----")
    print("p_A min/med/max:",
        np.nanmin(p_A),
        np.nanmedian(p_A),
        np.nanmax(p_A))

    print("p_B min/med/max:",
        np.nanmin(p_B),
        np.nanmedian(p_B),
        np.nanmax(p_B))

    print("p_inf:", p_inf)
    print("------------------------")

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

        press1 = (gamma - 1.0) * (energy[i1] - 0.5 * rho[i1] * np.dot(vel[i1], vel[i1]))
        press2 = (gamma - 1.0) * (energy[i2] - 0.5 * rho[i2] * np.dot(vel[i2], vel[i2]))
        press3 = (gamma - 1.0) * (energy[i3] - 0.5 * rho[i3] * np.dot(vel[i3], vel[i3]))

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

def _plane_indices(axis="x"):
    axis = str(axis).lower()
    if axis == "x":
        return 0, 1, 2   # normal=x, plane=(y,z)
    if axis == "y":
        return 1, 0, 2   # normal=y, plane=(x,z)
    if axis == "z":
        return 2, 0, 1   # normal=z, plane=(x,y)
    raise ValueError(f"Unsupported AIP axis: {axis}")


def _total_pressure_from_arrays(rho, energy, vel, gamma=GAMMA):
    rho = np.asarray(rho, float)
    energy = np.asarray(energy, float)
    vel = np.asarray(vel, float)

    vmag2 = np.sum(vel**2, axis=1)

    # Same convention as the existing pressure recovery code
    p = (energy - 0.5 * rho * vmag2) * (gamma - 1.0)

    good = (
        np.isfinite(rho)
        & np.isfinite(energy)
        & np.isfinite(p)
        & np.all(np.isfinite(vel), axis=1)
        & (rho > 0.0)
        & (p > 0.0)
    )

    P0 = np.full_like(p, np.nan, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mach = np.full_like(p, np.nan, dtype=float)

    a = np.sqrt(gamma * p[good] / rho[good])
    V = np.sqrt(vmag2[good])
    mach[good] = V / a
    q[good] = 0.5 * rho[good] * vmag2[good]

    P0[good] = p[good] * (
        1.0 + 0.5 * (gamma - 1.0) * mach[good] ** 2
    ) ** (gamma / (gamma - 1.0))

    return P0, p, q, mach, good


def generate_aip_rake_points(center, radius, axis="x", n_angles=5, n_radial=8,
                             r_inner=0.0, include_endpoint=False):
    """
    Generate rake points on an AIP plane.

    For axis='x', the AIP plane is y-z and x is fixed.
    """
    center = np.asarray(center, float).reshape(3)
    radius = float(radius)
    r_inner = float(r_inner)

    if radius <= 0:
        raise ValueError("AIP radius must be positive.")
    if n_angles < 1 or n_radial < 1:
        raise ValueError("n_angles and n_radial must be >= 1.")
    if r_inner < 0 or r_inner >= radius:
        raise ValueError("r_inner must satisfy 0 <= r_inner < radius.")

    normal_i, a_i, b_i = _plane_indices(axis)

    if include_endpoint:
        theta = np.linspace(0.0, 2.0 * np.pi, int(n_angles))
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)

    dr = (radius - r_inner) / float(n_radial)
    radii = r_inner + (np.arange(int(n_radial), dtype=float) + 0.5) * dr

    pts = []
    angle_ids = []
    radial_ids = []

    for ia, th in enumerate(theta):
        for ir, rr in enumerate(radii):
            p = center.copy()
            p[a_i] = center[a_i] + rr * np.cos(th)
            p[b_i] = center[b_i] + rr * np.sin(th)
            p[normal_i] = center[normal_i]
            pts.append(p)
            angle_ids.append(ia)
            radial_ids.append(ir)

    return np.asarray(pts, float), theta, radii, np.asarray(angle_ids), np.asarray(radial_ids)


def _circular_delta(a, b):
    """
    Smallest signed angular difference a-b in radians.
    """
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _extract_sampled_surface_arrays(root, aip_poly):
    """
    Resample solution fields from root onto the selected AIP surface.
    Returns points, rho, energy, velocity.
    """
    aip_src = TrivialProducer()
    aip_src.GetClientSideObject().SetOutput(aip_poly)
    UpdatePipeline(proxy=aip_src)

    sampled = ResampleWithDataset(SourceDataArrays=root, DestinationMesh=aip_src)
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    sampled.CellLocator = "Static Cell Locator"
    UpdatePipeline(proxy=sampled)

    sampled, rho_nm = ensure_pointdata(sampled, NAME_RHO)
    sampled, e_nm = ensure_pointdata(sampled, NAME_ENERGY)
    sampled, u_nm = ensure_pointdata(sampled, NAME_U)
    UpdatePipeline(proxy=sampled)

    ds = Fetch(sampled)
    leaf = _first_dataset_with_pointdata(ds)
    if leaf is None:
        raise RuntimeError("No sampled AIP point-data dataset found for DC60.")

    pts = ns.vtk_to_numpy(leaf.GetPoints().GetData()).astype(float)
    rho = ns.vtk_to_numpy(leaf.GetPointData().GetArray(rho_nm)).astype(float)
    energy = ns.vtk_to_numpy(leaf.GetPointData().GetArray(e_nm)).astype(float)
    vel = ns.vtk_to_numpy(leaf.GetPointData().GetArray(u_nm)).astype(float)

    if pts.size == 0:
        raise RuntimeError("Selected AIP surface contains no points.")

    return leaf, pts, rho, energy, vel


def compute_dc60_distortion(root, monitor):
    """
    Compute DC60-style circumferential total-pressure distortion from a selected AIP surface.

    Method:
      1. isolate selected AIP surface
      2. resample solution fields onto that surface
      3. generate rake points
      4. snap each rake point to closest AIP mesh node
      5. compute total pressure at sampled nodes
      6. DC60 = (mean(P0) - min_60deg_sector_mean(P0)) / mean(q)

    Config keys:
      surface_ids, center, radius, axis, n_angles, n_radial,
      r_inner, sector_deg, denominator
    """
    center = monitor.get("center", AIP_CENTER)
    radius = float(monitor.get("radius", AIP_RADIUS))
    axis = monitor.get("axis", "x")

    n_angles = int(monitor.get("n_angles", 5))
    n_radial = int(monitor.get("n_radial", 8))
    r_inner = float(monitor.get("r_inner", 0.0))
    sector_deg = float(monitor.get("sector_deg", 60.0))
    denominator_mode = str(monitor.get("denominator", "q_mean")).lower()

    sids = list(monitor.get("surface_ids", []))
    if sids:
        aip_poly = isolate_surface_ids(root, sids)
    else:
        aip_poly = isolate_aip_surface(root, center[0], center[1], center[2], radius)

    leaf, surf_pts, rho, energy, vel = _extract_sampled_surface_arrays(root, aip_poly)

    rake_pts, rake_theta, rake_radii, angle_ids, radial_ids = generate_aip_rake_points(
        center=center,
        radius=radius,
        axis=axis,
        n_angles=n_angles,
        n_radial=n_radial,
        r_inner=r_inner,
        include_endpoint=False,
    )

    # VTK closest-node lookup on the sampled AIP surface
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(leaf)
    locator.BuildLocator()

    closest_ids = np.array(
        [int(locator.FindClosestPoint(p.tolist())) for p in rake_pts],
        dtype=int,
    )

    sampled_pts = surf_pts[closest_ids]
    sampled_rho = rho[closest_ids]
    sampled_energy = energy[closest_ids]
    sampled_vel = vel[closest_ids]

    P0, p_static, q, mach, good = _total_pressure_from_arrays(
        sampled_rho, sampled_energy, sampled_vel, gamma=GAMMA
    )

    if not np.any(good):
        raise RuntimeError("No valid rake samples for DC60 computation.")

    P0_valid = P0[good]
    q_valid = q[good]

    P0_mean = float(np.nanmean(P0_valid))
    q_mean = float(np.nanmean(q_valid))

    # Compute worst sector from rake angular positions
    theta_all = rake_theta[angle_ids]
    theta_valid = theta_all[good]

    half_sector = np.deg2rad(sector_deg) / 2.0
    sector_means = []

    for th0 in rake_theta:
        in_sector = np.abs(_circular_delta(theta_valid, th0)) <= half_sector

        # With 5 rakes and a 60 degree sector, this may capture one rake.
        # That is okay; it becomes equivalent to worst rake mean.
        if not np.any(in_sector):
            nearest = np.argmin(np.abs(_circular_delta(theta_valid, th0)))
            in_sector = np.zeros_like(theta_valid, dtype=bool)
            in_sector[nearest] = True

        sector_means.append(float(np.nanmean(P0_valid[in_sector])))

    sector_means = np.asarray(sector_means, float)
    P0_sector_min = float(np.nanmin(sector_means))

    if denominator_mode == "p0_mean":
        denom = P0_mean
    elif denominator_mode == "constant":
        denom = float(monitor.get("denominator_value", 1.0))
    else:
        denom = q_mean

    DC60 = (P0_mean - P0_sector_min) / denom if denom > 0.0 else float("nan")

    # useful diagnostics
    unique_nodes = int(len(np.unique(closest_ids)))
    max_snap_dist = float(np.max(np.linalg.norm(sampled_pts - rake_pts, axis=1)))
    mean_snap_dist = float(np.mean(np.linalg.norm(sampled_pts - rake_pts, axis=1)))

    return {
        "DC60": float(DC60),
        "P0_mean": P0_mean,
        "P0_sector_min": P0_sector_min,
        "q_mean": q_mean,
        "denom": float(denom),
        "n_rake": int(len(rake_pts)),
        "n_unique_nodes": unique_nodes,
        "max_snap_dist": max_snap_dist,
        "mean_snap_dist": mean_snap_dist,
    }

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
    P0_AIP = p[good] * (1.0 + 0.5 * (GAMMA - 1.0) * M * M) ** (GAMMA / (GAMMA - 1.0))

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



# -------------------------------------------------------------------------
# Generic monitor helpers: line probes, plane/surface integrals, separation
# -------------------------------------------------------------------------
def _as_vec3(value, default=None, name="vector"):
    if default is None:
        default = [0.0, 0.0, 0.0]
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 3:
            raise ValueError
        return arr.astype(float)
    except Exception:
        arr = np.asarray(default, dtype=float).reshape(3)
        print(f"[MON][WARN] Invalid {name}; using default {arr.tolist()}", flush=True)
        return arr


def _normalise(v, default=None):
    v = _as_vec3(v, default if default is not None else [1.0, 0.0, 0.0])
    n = float(np.linalg.norm(v))
    if n <= 1e-14:
        return _as_vec3(default if default is not None else [1.0, 0.0, 0.0])
    return v / n


def _static_pressure(rho, energy, vel, gamma=GAMMA):
    rho = np.asarray(rho, float)
    energy = np.asarray(energy, float)
    vel = np.asarray(vel, float)
    vmag2 = np.sum(vel * vel, axis=1)
    return (energy - 0.5 * rho * vmag2) * (gamma - 1.0)


def _derived_flow_arrays(rho, energy, vel, gamma=GAMMA):
    rho = np.asarray(rho, float)
    energy = np.asarray(energy, float)
    vel = np.asarray(vel, float)
    vmag = np.sqrt(np.sum(vel * vel, axis=1))
    p = _static_pressure(rho, energy, vel, gamma=gamma)
    good = np.isfinite(rho) & np.isfinite(p) & np.all(np.isfinite(vel), axis=1) & (rho > 0.0) & (p > 0.0)
    mach = np.full_like(p, np.nan, dtype=float)
    qdyn = np.full_like(p, np.nan, dtype=float)
    P0 = np.full_like(p, np.nan, dtype=float)
    if np.any(good):
        a = np.sqrt(gamma * p[good] / rho[good])
        mach[good] = vmag[good] / np.maximum(a, 1e-30)
        qdyn[good] = 0.5 * rho[good] * vmag[good] ** 2
        P0[good] = p[good] * (1.0 + 0.5 * (gamma - 1.0) * mach[good] ** 2) ** (gamma / (gamma - 1.0))
    return {
        "rho": rho,
        "energy": energy,
        "velocity": vel,
        "velocity_magnitude": vmag,
        "pressure": p,
        "static_pressure": p,
        "mach": mach,
        "q": qdyn,
        "dynamic_pressure": qdyn,
        "total_pressure": P0,
        "P0": P0,
        "good": good,
    }


def _fetch_point_arrays(proxy):
    """Return first leaf point coordinates plus rho/energy/velocity and derived arrays."""
    proxy, rho_nm = ensure_pointdata(proxy, NAME_RHO)
    proxy, e_nm = ensure_pointdata(proxy, NAME_ENERGY)
    proxy, u_nm = ensure_pointdata(proxy, NAME_U)
    UpdatePipeline(proxy=proxy)
    ds = Fetch(proxy)
    leaf = _first_dataset_with_pointdata(ds)
    if leaf is None or leaf.GetPoints() is None:
        raise RuntimeError("No point-data leaf found.")
    pts = ns.vtk_to_numpy(leaf.GetPoints().GetData()).astype(float)
    rho = ns.vtk_to_numpy(leaf.GetPointData().GetArray(rho_nm)).astype(float)
    energy = ns.vtk_to_numpy(leaf.GetPointData().GetArray(e_nm)).astype(float)
    vel = ns.vtk_to_numpy(leaf.GetPointData().GetArray(u_nm)).astype(float)
    data = _derived_flow_arrays(rho, energy, vel)
    data["points"] = pts
    data["leaf"] = leaf
    return data


def _make_polydata_points(points):
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(ns.numpy_to_vtk(points, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    verts = vtk.vtkCellArray()
    for i in range(points.shape[0]):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    poly.SetVerts(verts)
    return poly


def _resample_root_to_polydata(root, polydata):
    src = TrivialProducer()
    src.GetClientSideObject().SetOutput(polydata)
    UpdatePipeline(proxy=src)
    sampled = ResampleWithDataset(SourceDataArrays=root, DestinationMesh=src)
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    sampled.CellLocator = "Static Cell Locator"
    UpdatePipeline(proxy=sampled)
    return sampled


def _get_array_by_alias(data, name):
    key = str(name or "").strip()
    aliases = {
        "p": "pressure", "pressure": "pressure", "static_pressure": "pressure",
        "mach": "mach", "m": "mach",
        "rho": "rho", "density": "rho",
        "velocity": "velocity_magnitude", "vel": "velocity_magnitude", "vmag": "velocity_magnitude",
        "velocity_magnitude": "velocity_magnitude", "speed": "velocity_magnitude",
        "q": "q", "dynamic_pressure": "q",
        "p0": "total_pressure", "P0": "total_pressure", "total_pressure": "total_pressure",
        "energy": "energy",
    }
    k = aliases.get(key, aliases.get(key.lower(), key))
    if k not in data:
        raise KeyError(f"Variable '{name}' not available. Available={list(data.keys())}")
    return data[k]


def compute_line_probe(root, monitor, out_dir=None, iteration=None):
    """
    Sample flow along a user-defined line. Also detects a shock proxy using
    max(dp/ds), min(dMach/ds), or a combined pressure/Mach-gradient indicator.
    """
    p0 = _as_vec3(monitor.get("point_a", monitor.get("p1", [0, 0, 0])), name="point_a")
    p1 = _as_vec3(monitor.get("point_b", monitor.get("p2", [1, 0, 0])), default=[1, 0, 0], name="point_b")
    n_samples = int(monitor.get("n_samples", monitor.get("samples", 200)))
    n_samples = max(2, n_samples)

    # Use explicit point cloud instead of ParaView PlotOverLine because it is
    # more stable across ParaView versions and easier to save as a profile.
    t = np.linspace(0.0, 1.0, n_samples)
    pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    sampled = _resample_root_to_polydata(root, _make_polydata_points(pts))
    data = _fetch_point_arrays(sampled)

    s = np.linalg.norm(pts - p0[None, :], axis=1)
    p = data["pressure"]
    M = data["mach"]
    vmag = data["velocity_magnitude"]
    rho = data["rho"]
    P0 = data["total_pressure"]

    good = data["good"] & np.isfinite(s)
    if np.count_nonzero(good) < 3:
        raise RuntimeError("Line probe has fewer than 3 valid samples.")

    # Fill isolated invalid values by interpolation so gradients are not ruined.
    def interp_good(y):
        y = np.asarray(y, float)
        ok = good & np.isfinite(y)
        if np.count_nonzero(ok) < 2:
            return y
        return np.interp(s, s[ok], y[ok])

    p_i = interp_good(p)
    M_i = interp_good(M)
    dpds = np.gradient(p_i, s, edge_order=1)
    dMds = np.gradient(M_i, s, edge_order=1)

    method = str(monitor.get("shock_method", monitor.get("method", "combined_pressure_mach_gradient"))).lower()
    if method in ("pressure", "max_dpds", "dpds"):
        score = dpds
    elif method in ("mach", "min_dmds", "dmds"):
        score = -dMds
    else:
        # Normalised combined compression indicator: pressure jumps up and Mach drops.
        sp = dpds / (np.nanmax(np.abs(dpds[good])) + 1e-30)
        sm = (-dMds) / (np.nanmax(np.abs(dMds[good])) + 1e-30)
        score = sp + sm

    valid_idx = np.where(good & np.isfinite(score))[0]
    ishock = int(valid_idx[np.nanargmax(score[valid_idx])])

    result = {
        "shock_s": float(s[ishock]),
        "shock_x": float(pts[ishock, 0]),
        "shock_y": float(pts[ishock, 1]),
        "shock_z": float(pts[ishock, 2]),
        "shock_score": float(score[ishock]),
        #"shock_dpds": float(dpds[ishock]),
        "shock_dMds": float(dMds[ishock]),
        #"p_min": float(np.nanmin(p[good])),
        #"p_max": float(np.nanmax(p[good])),
        "mach_min": float(np.nanmin(M[good])),
        "mach_max": float(np.nanmax(M[good])),
    }

    # Optional detailed line-profile CSV for debugging / plots.
    if out_dir:
        try:
            prof_dir = os.path.join(out_dir, "profiles")
            os.makedirs(prof_dir, exist_ok=True)
            safe_name = str(monitor.get("name", "line_probe")).replace("/", "_").replace(" ", "_")
            it = "unknown" if iteration is None else str(iteration)
            path = os.path.join(prof_dir, f"{safe_name}_iter_{it}.csv")
            with open(path, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["s", "x", "y", "z", "pressure", "mach", "velocity_magnitude", "density", "total_pressure", "dpds", "dMds", "shock_score"])
                for i in range(n_samples):
                    wr.writerow([s[i], pts[i,0], pts[i,1], pts[i,2], p[i], M[i], vmag[i], rho[i], P0[i], dpds[i], dMds[i], score[i]])
        except Exception as e:
            print(f"[MON][WARN] Failed to write line profile CSV: {e}", flush=True)

    return result


def _polydata_cell_area_and_normal(pts, cell):
    ids = cell.GetPointIds()
    if ids.GetNumberOfIds() < 3:
        return 0.0, np.array([np.nan, np.nan, np.nan]), []
    id_list = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
    p0 = pts[id_list[0]]
    area_vec = np.zeros(3, dtype=float)
    for j in range(1, len(id_list) - 1):
        a = pts[id_list[j]] - p0
        b = pts[id_list[j + 1]] - p0
        area_vec += 0.5 * np.cross(a, b)
    area = float(np.linalg.norm(area_vec))
    if area <= 1e-30:
        return 0.0, np.array([np.nan, np.nan, np.nan]), id_list
    return area, area_vec / area, id_list


def _integrate_on_polydata_with_point_arrays(polydata, data, normal=None, variable="mass_flow"):
    pts = data["points"]
    rho = data["rho"]
    vel = data["velocity"]
    p = data["pressure"]
    M = data["mach"]
    P0 = data["total_pressure"]
    qdyn = data["q"]

    user_n = None if normal is None else _normalise(normal)
    area_total = 0.0
    mdot = 0.0
    flux_abs = 0.0
    sums = {"pressure": 0.0, "mach": 0.0, "total_pressure": 0.0, "q": 0.0, "velocity_magnitude": 0.0}

    for ic in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(ic)
        area, ncell, ids = _polydata_cell_area_and_normal(pts, cell)
        if area <= 0.0 or not ids:
            continue
        nvec = user_n if user_n is not None else ncell
        ids = np.asarray(ids, dtype=int)
        rho_f = float(np.nanmean(rho[ids]))
        vel_f = np.nanmean(vel[ids], axis=0)
        vn = float(np.dot(vel_f, nvec))
        face_mdot = rho_f * vn * area
        mdot += face_mdot
        flux_abs += abs(face_mdot)
        area_total += area
        sums["pressure"] += float(np.nanmean(p[ids])) * area
        sums["mach"] += float(np.nanmean(M[ids])) * area
        sums["total_pressure"] += float(np.nanmean(P0[ids])) * area
        sums["q"] += float(np.nanmean(qdyn[ids])) * area
        sums["velocity_magnitude"] += float(np.nanmean(np.linalg.norm(vel[ids], axis=1))) * area

    out = {"area": float(area_total), "mdot": float(mdot), "mass_flow": float(mdot), "abs_mdot": float(flux_abs)}
    if area_total > 0.0:
        for k, v in sums.items():
            out[f"mean_{k}"] = float(v / area_total)
    else:
        for k in sums:
            out[f"mean_{k}"] = float("nan")
    return out


def compute_plane_integral(root, monitor):
    """
    Integrate quantities over either a user-defined slice plane or existing surface IDs.
    Primary mass-flow result is mdot = integral rho * (V dot n) dA.
    """
    mode = str(monitor.get("mode", "slice")).lower()
    normal = _normalise(monitor.get("normal", monitor.get("plane_normal", [1, 0, 0])))

    if mode == "surface" or monitor.get("surface_ids"):
        sids = [int(x) for x in monitor.get("surface_ids", [])]
        if not sids:
            raise RuntimeError("plane_integral surface mode requires surface_ids.")
        poly = isolate_surface_ids(root, sids)
        sampled = _resample_root_to_polydata(root, poly)
        data = _fetch_point_arrays(sampled)
        return _integrate_on_polydata_with_point_arrays(poly, data, normal=normal, variable=monitor.get("variable", "mass_flow"))

    origin = _as_vec3(monitor.get("origin", monitor.get("plane_origin", [0, 0, 0])), name="plane_origin")
    sl = Slice(Input=root)
    try:
        sl.SliceType = "Plane"
    except Exception:
        pass
    sl.SliceType.Origin = origin.tolist()
    sl.SliceType.Normal = normal.tolist()
    UpdatePipeline(proxy=sl)

    surf = ExtractSurface(Input=sl)
    UpdatePipeline(proxy=surf)
    tri = Triangulate(Input=surf)
    UpdatePipeline(proxy=tri)
    data = _fetch_point_arrays(tri)
    ds = Fetch(tri)
    leaf = _first_dataset_with_pointdata(ds)
    if leaf is None:
        raise RuntimeError("Slice produced no usable dataset.")
    return _integrate_on_polydata_with_point_arrays(leaf, data, normal=normal, variable=monitor.get("variable", "mass_flow"))


def _surface_with_normals(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.Update()
    return normals.GetOutput()


def compute_separation(root, monitor):
    """
    Separation proxy on selected wall surfaces.

    Always computes near-wall reversed-flow fraction using V dot e_stream < 0.
    If method='separation_thickness', it also samples both sides of the wall
    normal and estimates the maximum reversed-flow thickness.
    """
    sids = [int(x) for x in monitor.get("surface_ids", [])]
    if not sids:
        raise RuntimeError("separation monitor requires surface_ids.")

    e_stream = _normalise(monitor.get("streamwise_direction", monitor.get("direction", [1, 0, 0])))
    max_wall_distance = float(monitor.get("max_wall_distance", monitor.get("wall_distance", 0.02)))
    n_normal = int(monitor.get("normal_samples", monitor.get("n_normal", 12)))
    n_normal = max(2, n_normal)
    method = str(monitor.get("method", "reversed_flow_fraction")).lower()
    max_points = int(monitor.get("max_surface_points", 2000))

    poly = isolate_surface_ids(root, sids)
    poly_n = _surface_with_normals(poly)

    sampled = _resample_root_to_polydata(root, poly_n)
    data = _fetch_point_arrays(sampled)
    pts = data["points"]
    vel = data["velocity"]
    good = data["good"]
    u_stream = vel @ e_stream

    good_sep = good & np.isfinite(u_stream)
    if not np.any(good_sep):
        raise RuntimeError("No valid surface samples for separation monitor.")

    reversed_mask = good_sep & (u_stream < 0.0)
    reversed_fraction = float(np.count_nonzero(reversed_mask) / max(1, np.count_nonzero(good_sep)))
    min_u = float(np.nanmin(u_stream[good_sep]))

    result = {
        "sep_reversed_fraction": reversed_fraction,
        "sep_area_fraction_proxy": reversed_fraction,
        "sep_min_streamwise_velocity": min_u,
        "sep_max_reverse_velocity": float(max(0.0, -min_u)),
    }

    if np.any(reversed_mask):
        result["sep_centroid_x"] = float(np.nanmean(pts[reversed_mask, 0]))
        result["sep_centroid_y"] = float(np.nanmean(pts[reversed_mask, 1]))
        result["sep_centroid_z"] = float(np.nanmean(pts[reversed_mask, 2]))
    else:
        result["sep_centroid_x"] = result["sep_centroid_y"] = result["sep_centroid_z"] = float("nan")

    if method not in ("separation_thickness", "thickness", "max_thickness"):
        return result

    # Thickness estimate: sample along +/- normal for reversed wall points, because
    # imported surface normals may point into the solid. Use a capped subset for cost.
    normals_arr = poly_n.GetPointData().GetArray("Normals")
    if normals_arr is None:
        return result
    normals_np = ns.vtk_to_numpy(normals_arr).astype(float)

    idx = np.where(reversed_mask)[0]
    if idx.size > max_points:
        idx = idx[np.linspace(0, idx.size - 1, max_points).astype(int)]

    dvals = np.linspace(0.0, max_wall_distance, n_normal)
    thicknesses = []

    def sample_side(sign):
        sample_pts = []
        owner = []
        dist = []
        for ii in idx:
            n = normals_np[ii]
            nn = np.linalg.norm(n)
            if nn <= 1e-14:
                continue
            n = sign * n / nn
            for d in dvals:
                sample_pts.append(pts[ii] + d * n)
                owner.append(ii)
                dist.append(d)
        if not sample_pts:
            return None, None, None
        sampled2 = _resample_root_to_polydata(root, _make_polydata_points(np.asarray(sample_pts)))
        d2 = _fetch_point_arrays(sampled2)
        return np.asarray(owner), np.asarray(dist), d2["velocity"] @ e_stream

    best_thick = []
    for sign in (+1.0, -1.0):
        owner, dist, us = sample_side(sign)
        if owner is None:
            continue
        for ii in idx:
            m = owner == ii
            if not np.any(m):
                continue
            dd = dist[m]
            uu = us[m]
            order = np.argsort(dd)
            dd = dd[order]
            uu = uu[order]
            neg = np.isfinite(uu) & (uu < 0.0)
            if not np.any(neg):
                thick = 0.0
            elif np.all(neg):
                thick = float(dd[-1])
            else:
                # first index where flow recovers positive after initial reversed region
                pos_after = np.where(~neg)[0]
                thick = float(dd[pos_after[0]]) if pos_after.size else float(dd[-1])
            best_thick.append(thick)

    if best_thick:
        arr = np.asarray(best_thick, float)
        result["sep_max_thickness"] = float(np.nanmax(arr))
        result["sep_mean_thickness"] = float(np.nanmean(arr))
    else:
        result["sep_max_thickness"] = float("nan")
        result["sep_mean_thickness"] = float("nan")

    return result

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

                pr_src = TrivialProducer()
                pr_src.GetClientSideObject().SetOutput(pr_geom)
                UpdatePipeline(proxy=pr_src)

                sampled = ResampleWithDataset(SourceDataArrays=r, DestinationMesh=pr_src)
                sampled.PassPointArrays = 1
                sampled.PassCellArrays = 1
                sampled.CellLocator = 'Static Cell Locator'
                UpdatePipeline(proxy=sampled)

                P0_inf = compute_freestream_total_pressure(r)
                P0_mean, rec = compute_pressure_recovery(sampled, P0_inf)

                #results[f"{name}_P0_inf"] = P0_inf
                #results[f"{name}_P0_mean"] = P0_mean
                results[f"{name}_pressure_recovery"] = rec
                
            elif mtype in ("dc60", "distortion", "distortion_dc60"):
                dc = compute_dc60_distortion(r, mon)

                results[name] = dc["DC60"]

            elif mtype == "drag":
                drag_res = compute_surface_pressure_drag(
                    root=r,
                    surface_ids=mon.get("surface_ids", []),
                    direction=mon.get("direction", "x"),
                    symmetry_factor=int(mon.get("symmetry_factor", 1)),
                    gamma=GAMMA,
                )

                results[name + "_over_q"] = drag_res["drag_over_q"]
                #results[name + "_force"] = drag_res.get("drag_force", float("nan"))
                #results[name + "_cd_proj"] = drag_res.get("cd_proj", float("nan"))

            elif mtype in ("line_probe", "shock_location", "shock"):
                out_dir = os.path.dirname(os.path.abspath(args.out))
                lp = compute_line_probe(r, mon, out_dir=out_dir, iteration=args.iter)
                for k, v in lp.items():
                    results[f"{name}_{k}"] = v

            elif mtype in ("plane_integral", "mass_flow", "massflow"):
                pi = compute_plane_integral(r, mon)
                for k, v in pi.items():
                    results[f"{name}_{k}"] = v

            elif mtype in ("separation", "flow_separation"):
                sep = compute_separation(r, mon)
                for k, v in sep.items():
                    results[f"{name}_{k}"] = v

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
