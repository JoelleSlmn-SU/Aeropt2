import argparse, os, csv
import numpy as np
from paraview.simple import *
from paraview.servermanager import Fetch
from vtk.util import numpy_support as ns

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
    "/Root/111",
]
cx, cy, cz = AIP_CENTER

# ---------------- helpers ----------------
def list_arrays(proxy):
    ds = Fetch(proxy)
    out = {}
    pd = ds.GetPointData()
    cd = ds.GetCellData()
    if pd:
        for i in range(pd.GetNumberOfArrays()):
            a = pd.GetArray(i)
            out[a.GetName()] = (a.GetNumberOfComponents(), "POINTS")
    if cd:
        for i in range(cd.GetNumberOfArrays()):
            a = cd.GetArray(i)
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

def isolate_aip_surface(root, cx, cy, cz, r):
    surf = _threshold_surface_id(root, 111)
    if surf: return surf
    surf = _extract_block(root, AIP_SELECTORS)
    if surf: return surf
    surf = _pick_aip_geometric(root, cx, cy, cz, r)
    if surf: return surf
    raise RuntimeError("AIP isolation failed.")

def _ensure_pressure_on(proxy):
    # p = (energy - 0.5*rho*|U|^2)*(gamma-1)  [energy per unit volume]
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
    UpdatePipeline(proxy=sampled)

    p_src, p = _ensure_pressure_on(sampled)
    rho_src, rho = ensure_pointdata(p_src, NAME_RHO)
    u_src, u = ensure_pointdata(rho_src, NAME_U)

    calc_M = Calculator(Input=u_src)
    calc_M.ResultArrayName = "M_inf"
    calc_M.Function = f"mag({u})/sqrt({GAMMA}*{p}/{rho})"
    UpdatePipeline(proxy=calc_M)

    calc_P0 = Calculator(Input=calc_M)
    calc_P0.ResultArrayName = "P0_total"
    calc_P0.Function = f"{p} * pow(1 + 0.5*({GAMMA}-1)*M_inf*M_inf, {GAMMA}/({GAMMA}-1))"
    UpdatePipeline(proxy=calc_P0)

    vtk_data = Fetch(calc_P0)
    arr = vtk_data.GetPointData().GetArray("P0_total")
    p0_np = ns.vtk_to_numpy(arr)
    return float(np.median(p0_np))

def compute_pressure_recovery(sampled_on_aip, P0_inf):
    p_src, p_arr = _ensure_pressure_on(sampled_on_aip)
    rho_src, rho = ensure_pointdata(p_src, NAME_RHO)
    u_src, u = ensure_pointdata(rho_src, NAME_U)

    calc_M = Calculator(Input=u_src)
    calc_M.ResultArrayName = "M_local"
    calc_M.Function = f"mag({u})/sqrt({GAMMA}*{p_arr}/{rho})"
    UpdatePipeline(proxy=calc_M)

    calc_P0 = Calculator(Input=calc_M)
    calc_P0.ResultArrayName = "P0_AIP"
    calc_P0.Function = f"0.9*{p_arr} * pow(1 + 0.5*({GAMMA}-1)*M_local*M_local, {GAMMA}/({GAMMA}-1))"
    UpdatePipeline(proxy=calc_P0)

    vtk_data = Fetch(calc_P0)
    p0_arr = vtk_data.GetPointData().GetArray("P0_AIP")
    p0_vals = ns.vtk_to_numpy(p0_arr)

    P0_mean = float(np.mean(p0_vals))
    recovery = P0_mean / P0_inf if P0_inf != 0.0 else float("nan")
    return P0_mean, recovery

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    r = EnSightReader(CaseFileName=args.case)
    UpdatePipeline(proxy=r)

    aip_geom = isolate_aip_surface(r, cx, cy, cz, AIP_RADIUS)
    UpdatePipeline(proxy=aip_geom)

    sampled = ResampleWithDataset(SourceDataArrays=r, DestinationMesh=aip_geom)
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    sampled.CellLocator = 'Static Cell Locator'
    UpdatePipeline(proxy=sampled)

    P0_inf = compute_freestream_total_pressure(r)
    P0_mean_aip, pr = compute_pressure_recovery(sampled, P0_inf)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_header = not (args.append and os.path.isfile(args.out))
    with open(args.out, "a" if args.append else "w", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["iter", "P0_inf_median_Pa", "P0_mean_AIP_Pa", "pressure_recovery"])
        w.writerow([args.iter, P0_inf, P0_mean_aip, pr])

if __name__ == "__main__":
    main()
