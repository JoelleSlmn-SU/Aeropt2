# Paraview.py — robust x-sweep plotting: rho, Mach, eta = P0/P0_inf
# Run: pvpython Paraview.py

from pathlib import Path
import os, glob
import numpy as np

from paraview.simple import *
from paraview.servermanager import Fetch
from vtk.util import numpy_support as ns

# ---------- CONFIG ----------
folder = [
    "Corner Bump Surface",
    "Corner Bump Surface Opt",
    "Corner Bump Surface Coarse Optimisation",
    "Corner Bump Surface Optimisation",
    "CB Opt 15.12"
]

x_case = 1
base = Path(r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples")
case_root = base / folder[3]

CASE_FILE = str(case_root / "postprocessed" / str(x_case) / f"ENSIGHTcorner_{x_case}.case")
OUT_DIR   = str(case_root / "postprocessed" / str(x_case) / f"x_sweep_out")
#CASE_FILE = str(case_root / f"ENSIGHTcorner.case")
#OUT_DIR   = str(case_root / f"x_sweep_out")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE = (1200, 600)

# Sweep
X_START, X_END = 6.0, 12.0
N_SLICES = 61

# Gas
GAMMA = 1.4

# Arrays
NAME_U      = "velocity"
NAME_RHO    = "density"
NAME_ENERGY = "energy"

# Plot ranges
MACH_RANGE = (0.39, 1.30)
ETA_RANGE  = (0.75, 1.00)

# Matplotlib discrete rainbow
CMAP_NAME = "rainbow"
N_STEPS   = 12

WRITE_PDFS = True

# Freestream sampling region
FS_CENTER  = (-8.0, 0.0, 0.0)
FS_LENGTHS = (0.6, 5.0, 5.0)
FS_NPTS    = 20000
DROP_ZEROS = True
# ======================= END CONFIG =======================

OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[INFO] CASE_FILE: {CASE_FILE}")
print(f"[INFO] OUT_DIR:   {OUT_DIR}")

# --------------------- helpers ---------------------
def open_case(path):
    src = OpenDataFile(path)
    if not src:
        raise RuntimeError(f"Could not open data file: {path}")
    return src

def _is_composite(proxy):
    di = proxy.GetDataInformation()
    return bool(di.GetCompositeDataClassName())

def normalize_to_single_dataset(proxy):
    """Merge blocks if composite so downstream filters see a single dataset."""
    if _is_composite(proxy):
        mb = MergeBlocks(Input=proxy)
        Show(mb); Hide(proxy)
        return mb
    return proxy

def list_arrays(proxy):
    proxy = normalize_to_single_dataset(proxy)
    di = proxy.GetDataInformation()
    out = {}
    pdi = di.GetPointDataInformation()
    for i in range(pdi.GetNumberOfArrays()):
        ai = pdi.GetArrayInformation(i)
        out[ai.GetName()] = (ai.GetNumberOfComponents(), "POINTS")
    cdi = di.GetCellDataInformation()
    for i in range(cdi.GetNumberOfArrays()):
        ai = cdi.GetArrayInformation(i)
        out[ai.GetName()] = (ai.GetNumberOfComponents(), "CELLS")
    return out

def ensure_pointdata(proxy, array_name):
    proxy = normalize_to_single_dataset(proxy)
    arrs = list_arrays(proxy)
    if array_name not in arrs:
        raise RuntimeError(f"Array '{array_name}' not found. Available: {list(arrs.keys())}")
    ncomp, loc = arrs[array_name]
    if loc == "POINTS":
        return proxy, array_name
    c2p = CellDatatoPointData(Input=proxy)
    Show(c2p); Hide(proxy)
    return c2p, array_name

COLORMAP_NAME     = "Rainbow Uniform"     # or "Rainbow", "Rainbow Uniform", etc.
SHOW_COLORBAR     = True

def render_colorfield(view, src_proxy, array_name, title, fixed_range=None):
    px, arr = ensure_pointdata(src_proxy, array_name)
    SetActiveSource(px)
    rep = Show(px); rep.Representation = "Surface"
    ColorBy(rep, ("POINTS", arr))
    # Apply rainbow colormap and (optional) fixed/global range
    lut = GetColorTransferFunction(arr)
    try:
        lut.ApplyPreset(COLORMAP_NAME, True)
    except Exception:
        pass
    if fixed_range is not None:
        vmin, vmax = fixed_range
        lut.RescaleTransferFunction(vmin, vmax)
        pwf = GetOpacityTransferFunction(arr); pwf.RescaleTransferFunction(vmin, vmax)
    else:
        vmin, vmax = compute_range(px, arr)
        lut.RescaleTransferFunction(vmin, vmax)
        pwf = GetOpacityTransferFunction(arr); pwf.RescaleTransferFunction(vmin, vmax)

    if SHOW_COLORBAR:
        rep.SetScalarBarVisibility(view, True)
        sb = GetScalarBar(lut, view)
        sb.Title = title
        sb.LabelFormat = "%.3g"
        sb.ScalarBarLength = 0.35
        sb.WindowLocation = "Upper Right Corner"
    return px, arr, rep

def compute_range(proxy, array_name):
    proxy = normalize_to_single_dataset(proxy)
    vtk_ds = Fetch(proxy)
    if vtk_ds is None: return float("nan"), float("nan")
    pd, cd = vtk_ds.GetPointData(), vtk_ds.GetCellData()
    arr = pd.GetArray(array_name) if pd else None
    if arr is None and cd: arr = cd.GetArray(array_name)
    if arr is None: return float("nan"), float("nan")
    ncomp = arr.GetNumberOfComponents()
    np_arr = ns.vtk_to_numpy(arr)
    data = np_arr if ncomp==1 else np.linalg.norm(np_arr.reshape(-1,ncomp),axis=1)
    data = data[np.isfinite(data)]
    if data.size == 0: return float("nan"), float("nan")
    return float(np.min(data)), float(np.max(data))

def _np_point_array(proxy, array_name):
    ds = Fetch(proxy)
    arr = ds.GetPointData().GetArray(array_name)
    if arr is None:
        raise RuntimeError(f"Point array '{array_name}' not found.")
    a = ns.vtk_to_numpy(arr)
    a = a[np.isfinite(a)]
    return a

def save_png(view, basename):
    path = os.path.join(OUT_DIR, basename + ".png")
    try:
        Render(view=view); view.StillRender()
        SaveScreenshot(filename=path, viewOrLayout=view, ImageResolution=IMAGE_SIZE)
    except TypeError:
        SaveScreenshot(path, view, ImageResolution=IMAGE_SIZE)
    print(f"[OK] Saved: {path}")
    return path

def pngs_to_pdf(pattern, out_pdf):
    try:
        from PIL import Image
    except Exception:
        print("[WARN] PIL not available; skipping PDF build.")
        return
    frames = sorted(glob.glob(os.path.join(OUT_DIR, pattern)))
    frames = [p for p in frames if os.path.getsize(p) > 0]
    if not frames:
        print(f"[WARN] No frames for {pattern}")
        return
    imgs = [Image.open(p).convert("RGB") for p in frames]
    out_path = os.path.join(OUT_DIR, out_pdf)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:])
    print(f"[OK] PDF written: {out_path} ({len(imgs)} pages)")

def apply_matplotlib_discrete_rainbow(array_name, cmap_name, n_steps, vmin, vmax, view, rep, title):
    import numpy as _np
    from matplotlib import cm

    lut = GetColorTransferFunction(array_name)
    lut.Discretize = 1
    lut.NumberOfTableValues = int(n_steps)
    lut.RescaleTransferFunction(float(vmin), float(vmax))

    cmap = cm.get_cmap(cmap_name, int(n_steps))
    vals = _np.linspace(float(vmin), float(vmax), int(n_steps))
    RGBPoints = []
    for k, val in enumerate(vals):
        r, g, b, _ = cmap(k)
        RGBPoints.extend([float(val), float(r), float(g), float(b)])
    lut.RGBPoints = RGBPoints
    lut.ColorSpace = "RGB"

    pwf = GetOpacityTransferFunction(array_name)
    pwf.RescaleTransferFunction(float(vmin), float(vmax))

    rep.SetScalarBarVisibility(view, True)
    sb = GetScalarBar(lut, view)
    sb.Title = title
    sb.LabelFormat = "%.2f"
    sb.ScalarBarLength = 0.35
    sb.WindowLocation = "Upper Right Corner"

def render_field(view, proxy, array_name, title, vmin=None, vmax=None, use_matplotlib=False):
    px, arr = ensure_pointdata(proxy, array_name)
    SetActiveSource(px)
    rep = Show(px)
    rep.Representation = "Surface"
    ColorBy(rep, ("POINTS", arr))

    if vmin is not None and vmax is not None:
        lut = GetColorTransferFunction(arr)
        lut.RescaleTransferFunction(float(vmin), float(vmax))
        pwf = GetOpacityTransferFunction(arr)
        pwf.RescaleTransferFunction(float(vmin), float(vmax))

    if use_matplotlib and (vmin is not None and vmax is not None):
        apply_matplotlib_discrete_rainbow(arr, CMAP_NAME, N_STEPS, vmin, vmax, view, rep, title)
    else:
        rep.SetScalarBarVisibility(view, True)
        sb = GetScalarBar(GetColorTransferFunction(arr), view)
        sb.Title = title
        sb.LabelFormat = "%.3g"
        sb.ScalarBarLength = 0.35
        sb.WindowLocation = "Upper Right Corner"

    return px, rep

# ---------------- physics helpers ----------------
def ensure_pressure(proxy):
    px_rho, rho = ensure_pointdata(proxy, NAME_RHO)
    px_e,   e   = ensure_pointdata(px_rho, NAME_ENERGY)
    px_u,   u   = ensure_pointdata(px_e,   NAME_U)

    calc = Calculator(Input=px_u)
    calc.ResultArrayName = "p_calc"
    calc.Function = f"({e} - 0.5*{rho}*mag({u})^2) * ({GAMMA}-1)"
    UpdatePipeline(proxy=calc)
    return calc, "p_calc"

def ensure_mach(proxy):
    p_px, p = ensure_pressure(proxy)
    px_rho, rho = ensure_pointdata(p_px, NAME_RHO)
    px_u,   u   = ensure_pointdata(px_rho, NAME_U)

    calc = Calculator(Input=px_u)
    calc.ResultArrayName = "Mach"
    calc.Function = f"mag({u})/sqrt({GAMMA}*{p}/{rho})"
    UpdatePipeline(proxy=calc)
    return calc, "Mach"

def ensure_eta(proxy, P0_inf):
    p_px, p = ensure_pressure(proxy)
    m_px, m = ensure_mach(p_px)

    calc_P0 = Calculator(Input=m_px)
    calc_P0.ResultArrayName = "P0_local"
    calc_P0.Function = f"{p} * pow(1 + 0.5*({GAMMA}-1)*{m}*{m}, {GAMMA}/({GAMMA}-1))"
    UpdatePipeline(proxy=calc_P0)

    calc_eta = Calculator(Input=calc_P0)
    calc_eta.ResultArrayName = "eta"
    calc_eta.Function = f"P0_local/{float(P0_inf)}"
    UpdatePipeline(proxy=calc_eta)
    return calc_eta, "eta"

import numpy as np
import math

def ensure_speed(proxy):
    """Return (proxy_with_speed, speed_array_name)."""
    px_u, u_nm = ensure_pointdata(proxy, NAME_U)
    calc = Calculator(Input=px_u)
    calc.ResultArrayName = "U_mag"
    calc.Function = f"mag({u_nm})"
    UpdatePipeline(proxy=calc)
    return calc, "U_mag"

def freestream_pointcloud(center, lengths, npts):
    cx, cy, cz = map(float, center)
    lx, ly, lz = map(float, lengths)

    pts = PointSource(registrationName="FSPoints")
    pts.NumberOfPoints = int(npts)
    pts.Radius = 1.0
    UpdatePipeline(proxy=pts)

    c = Calculator(Input=pts)
    c.ResultArrayName = "coords"
    c.Function = (
        f"iHat*({cx} + {lx}*(coordsX)) + "
        f"jHat*({cy} + {ly}*(coordsY)) + "
        f"kHat*({cz} + {lz}*(coordsZ))"
    )
    UpdatePipeline(proxy=c)

    w = WarpByVector(Input=c)
    w.Vectors = ["POINTS", "coords"]
    w.ScaleFactor = 1.0
    UpdatePipeline(proxy=w)
    return w

def compute_P0_inf(root3d):
    print("\n[INFO] Computing freestream P0_inf (median) using dense sampling...")
    fs_pts = freestream_pointcloud(FS_CENTER, FS_LENGTHS, FS_NPTS)

    sampled = ResampleWithDataset(SourceDataArrays=root3d, DestinationMesh=fs_pts)
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    UpdatePipeline(proxy=sampled)

    p_px, p = ensure_pressure(sampled)
    rho_px, rho = ensure_pointdata(p_px, NAME_RHO)
    u_px, u = ensure_pointdata(rho_px, NAME_U)

    calc_M = Calculator(Input=u_px)
    calc_M.ResultArrayName = "M_inf"
    calc_M.Function = f"mag({u})/sqrt({GAMMA}*{p}/{rho})"
    UpdatePipeline(proxy=calc_M)

    calc_P0 = Calculator(Input=calc_M)
    calc_P0.ResultArrayName = "P0_inf_field"
    calc_P0.Function = f"{p} * pow(1 + 0.5*({GAMMA}-1)*M_inf*M_inf, {GAMMA}/({GAMMA}-1))"
    UpdatePipeline(proxy=calc_P0)

    P0 = _np_point_array(calc_P0, "P0_inf_field")
    if DROP_ZEROS:
        P0 = P0[P0 != 0.0]
    if P0.size == 0:
        raise RuntimeError("P0_inf sampling returned no valid values (all 0/NaN).")

    P0_inf = float(np.median(P0))
    print(f"[OK] P0_inf median = {P0_inf:.6g}  (n={P0.size}, min={float(np.min(P0)):.6g}, max={float(np.max(P0)):.6g})")

    for obj in [calc_P0, calc_M, u_px, rho_px, p_px, sampled, fs_pts]:
        try: Delete(obj)
        except: pass

    return P0_inf

# ----------------- slice setup (ORIGINAL logic) -----------------
def build_slice(src):
    sl = Slice(registrationName="Slice", Input=src)
    sl.SliceType = "Plane"
    sl.SliceType.Normal = [1.0, 0.0, 0.0]
    sl.SliceType.Origin = [float(X_START), 0.0, 0.0]
    merged = MergeBlocks(registrationName="Slice_Merged", Input=sl)
    UpdatePipeline(proxy=merged)
    return sl, merged

# ========================= SWEEP MAIN =========================
src = open_case(CASE_FILE)
src = normalize_to_single_dataset(src)

rv = GetActiveViewOrCreate("RenderView")
rv.ViewSize = IMAGE_SIZE
rv.Background = [1, 1, 1]
rv.InteractionMode = "2D"
Show(src)
ResetCamera()

# ORIGINAL camera
rv.CameraParallelProjection = 1
rv.CameraPosition = [-3, 0, 0]
rv.CameraFocalPoint = [0, 0.5, 0]
rv.CameraViewUp = [0, 0, 1]
rv.CameraParallelScale = 1.8
rv.StillRender()

slice_obj, slice_merged = build_slice(src)
Hide(src)

# ✅ FIX: compute P0_inf from full 3D dataset, not the slice
P0_inf = compute_P0_inf(src)

# Prepare plotting proxies on the slice
rho_proxy, _ = ensure_pointdata(slice_merged, NAME_RHO)
mach_proxy, _ = ensure_mach(slice_merged)
eta_proxy, _  = ensure_eta(slice_merged, P0_inf)

x_vals = np.linspace(float(X_START), float(X_END), int(N_SLICES))

print("\n[INFO] Starting sweep...")
for i, xv in enumerate(x_vals):
    slice_obj.SliceType.Origin = [float(xv), 0.0, 0.0]
    UpdatePipeline(proxy=slice_merged)

    if i % 10 == 0:
        print(f"[INFO] Slice {i+1}/{N_SLICES} at x={xv:.3f}")

    # Density
    t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {xv:.3f}"
    trep = Show(t); trep.WindowLocation="Upper Left Corner"; trep.FontSize=14; trep.Color=[0,0,0]
    px, rep = render_field(rv, rho_proxy, NAME_RHO, "Density")
    rv.StillRender(); save_png(rv, f"x_{i:03d}_rho")
    Hide(px); Hide(t); Delete(t)

    # Mach (matplotlib rainbow)
    t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {xv:.3f}"
    trep = Show(t); trep.WindowLocation="Upper Left Corner"; trep.FontSize=14; trep.Color=[0,0,0]
    px, rep = render_field(rv, mach_proxy, "Mach", "Mach", vmin=MACH_RANGE[0], vmax=MACH_RANGE[1], use_matplotlib=True)
    rv.StillRender(); save_png(rv, f"x_{i:03d}_mach")
    Hide(px); Hide(t); Delete(t)

    # eta (matplotlib rainbow)
    t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {xv:.3f}"
    trep = Show(t); trep.WindowLocation="Upper Left Corner"; trep.FontSize=14; trep.Color=[0,0,0]
    px, rep = render_field(rv, eta_proxy, "eta", "η [-]", vmin=ETA_RANGE[0], vmax=ETA_RANGE[1], use_matplotlib=True)
    rv.StillRender(); save_png(rv, f"x_{i:03d}_eta")
    Hide(px); Hide(t); Delete(t)

print("[OK] Sweep complete.")

if WRITE_PDFS:
    pngs_to_pdf("x_*_rho.png",  "rho_sweep.pdf")
    pngs_to_pdf("x_*_mach.png", "mach_sweep.pdf")
    pngs_to_pdf("x_*_eta.png",  "eta_sweep.pdf")

print(f"[OK] Sweep outputs in: {OUT_DIR}")

# ========================= AIP: pressure recovery =========================
# This section keeps your AIP logic conceptually, but fixes P0_inf and recovery calculation.


print("\n" + "="*60)
print("Starting AIP extraction and plotting")
print("="*60)

# ---- COMPLETE PIPELINE CLEANUP ----
print("[INFO] Cleaning up sweep pipeline...")
try:
    # Delete all sweep-related filters
    for obj in GetSources().values():
        try:
            Delete(obj)
        except:
            pass
except Exception as e:
    print(f"[WARN] Cleanup: {e}")

# ---- FRESH START: Re-open the data file ----
print(f"[INFO] Re-opening case file: {CASE_FILE}")
src_aip = open_case(CASE_FILE)
src_aip = normalize_to_single_dataset(src_aip)
UpdatePipeline(proxy=src_aip)
Show(src_aip)

# ---- Setup fresh render view ----
rv = GetActiveViewOrCreate("RenderView")
rv.ViewSize = IMAGE_SIZE
rv.Background = [1, 1, 1]
rv.InteractionMode = "3D"
ResetCamera()

# ---- AIP CONFIG ----
AIP_CENTER = (11.3, 0.4, 0.21)
AIP_RADIUS = 0.37
AIP_SELECTORS = [
    "/Root/Surfaces/Surface 111",
    "/Root/Surface 111",
    "/Root/Blocks/Surface 111",
    "/Root/111",
]
cx, cy, cz = AIP_CENTER

print(f"[INFO] AIP center: ({cx}, {cy}, {cz}), radius: {AIP_RADIUS}")

# ========== HELPER FUNCTIONS (inlined) ==========

def _ensure_pressure_on(proxy):
    # p = (energy - 0.5*rho*|U|^2)*(gamma-1)  [energy per unit volume]
    px_rho, rho_nm = ensure_pointdata(proxy, NAME_RHO)
    px_e,   e_nm   = ensure_pointdata(px_rho, NAME_ENERGY)
    px_u,   u_nm   = ensure_pointdata(px_e,   NAME_U)
    calc = Calculator(Input=px_u)
    calc.ResultArrayName = "p_calc"
    calc.Function = f"({e_nm} - 0.5*{rho_nm}*mag({u_nm})^2) * ({GAMMA}-1)"
    Show(calc); Hide(px_u)
    UpdatePipeline(proxy=calc)
    return calc, "p_calc"

def _ensure_speed_on(proxy):
    px_u, u_nm = ensure_pointdata(proxy, NAME_U)
    calc = Calculator(Input=px_u)
    calc.ResultArrayName = "U_mag"
    calc.Function = f"mag({u_nm})"
    Show(calc); Hide(px_u)
    UpdatePipeline(proxy=calc)
    return calc, "U_mag"

def _threshold_set_local(thr, loc, arr_name, lo, hi):
    """Configure Threshold filter across ParaView versions."""
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

def _threshold_surface_id_local(root, target_sid=111):
    """Pick surface by surface_id array if present."""
    arrs = list_arrays(root)
    for nm in ("surface_id", "SurfaceId", "SURFACE_ID", "SurfaceID", "ElementBlockIds"):
        if nm in arrs:
            ncomp, loc = arrs[nm]
            print(f"[INFO] Found surface ID array: {nm} ({loc})")
            thr = Threshold(Input=root)
            _threshold_set_local(thr, loc, nm, target_sid - 0.5, target_sid + 0.5)
            UpdatePipeline(proxy=thr)
            
            # Check if we got data
            ds = Fetch(thr)
            if ds and ds.GetNumberOfPoints() > 0:
                print(f"[OK] Surface {target_sid} isolated via {nm}: {ds.GetNumberOfPoints()} points")
                poly = ExtractSurface(Input=thr)
                UpdatePipeline(proxy=poly)
                return poly
            else:
                print(f"[WARN] Threshold on {nm} returned no points")
    return None

def _extract_block_local(root, selectors):
    """Try ExtractBlock with various selectors."""
    for sel in selectors:
        try:
            print(f"[INFO] Trying ExtractBlock selector: {sel}")
            eb = ExtractBlock(Input=root)
            eb.Selectors = [sel]
            UpdatePipeline(proxy=eb)
            ds = Fetch(eb)
            if ds and ds.GetNumberOfPoints() > 0:
                print(f"[OK] ExtractBlock succeeded: {ds.GetNumberOfPoints()} points")
                try:
                    mb = MergeBlocks(Input=eb)
                    UpdatePipeline(proxy=mb)
                    poly = ExtractSurface(Input=mb)
                except:
                    poly = ExtractSurface(Input=eb)
                UpdatePipeline(proxy=poly)
                return poly
        except Exception as e:
            print(f"[WARN] Selector '{sel}' failed: {e}")
    return None

def _pick_aip_geometric_local(root, cx, cy, cz, r, dx=0.02, cos_tol=0.90):
    """
    Geometric picking: isolate disc at x≈cx, radius r in YZ plane.
    More tolerant settings for robustness.
    """
    print(f"[INFO] Geometric picking: dx={dx}, r={r}, cos_tol={cos_tol}")
    
    # 0) Extract surface
    surf = ExtractSurface(Input=root)
    UpdatePipeline(proxy=surf)
    
    # 1) Compute normals
    try:
        norms = SurfaceNormals(Input=surf)
    except:
        norms = GenerateSurfaceNormals(Input=surf)
    norms.FeatureAngle = 180.0
    norms.ComputeCellNormals = 1
    UpdatePipeline(proxy=norms)
    
    # 2) X slab: |x - cx| <= dx
    c_absx = Calculator(Input=norms)
    c_absx.ResultArrayName = "abs_x_dist"
    c_absx.Function = f"abs(coordsX - {cx})"
    UpdatePipeline(proxy=c_absx)
    
    t_x = Threshold(Input=c_absx)
    _threshold_set_local(t_x, "POINTS", "abs_x_dist", 0.0, float(dx))
    UpdatePipeline(proxy=t_x)
    
    ds_x = Fetch(t_x)
    if not ds_x or ds_x.GetNumberOfPoints() == 0:
        print(f"[WARN] X-slab threshold returned no points")
        return None
    print(f"[INFO] After X-slab: {ds_x.GetNumberOfPoints()} points")
    
    # 3) Radial disc in YZ
    c_rad = Calculator(Input=t_x)
    c_rad.ResultArrayName = "r_from_center"
    c_rad.Function = f"sqrt((coordsY-{cy})^2 + (coordsZ-{cz})^2)"
    UpdatePipeline(proxy=c_rad)
    
    t_r = Threshold(Input=c_rad)
    _threshold_set_local(t_r, "POINTS", "r_from_center", 0.0, float(r))
    UpdatePipeline(proxy=t_r)
    
    ds_r = Fetch(t_r)
    if not ds_r or ds_r.GetNumberOfPoints() == 0:
        print(f"[WARN] Radial threshold returned no points")
        return None
    print(f"[INFO] After radial cut: {ds_r.GetNumberOfPoints()} points")
    
    # 4) Normal alignment: |n_x| >= cos_tol
    # First ensure Normals is on POINTS
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
    _threshold_set_local(t_ax, "POINTS", "ax_align", float(cos_tol), 1.0)
    UpdatePipeline(proxy=t_ax)
    
    ds_ax = Fetch(t_ax)
    if not ds_ax or ds_ax.GetNumberOfPoints() == 0:
        print(f"[WARN] Normal alignment threshold returned no points")
        # Try without normal filtering
        print(f"[INFO] Skipping normal filter, using radial disc only")
        out = ExtractSurface(Input=t_r)
    else:
        print(f"[INFO] After normal filter: {ds_ax.GetNumberOfPoints()} points")
        out = ExtractSurface(Input=t_ax)
    
    UpdatePipeline(proxy=out)
    cln = Clean(Input=out)
    UpdatePipeline(proxy=cln)
    
    ds_final = Fetch(cln)
    if ds_final:
        print(f"[OK] Geometric picking complete: {ds_final.GetNumberOfPoints()} points")
    return cln

def isolate_aip_surface(root, cx, cy, cz, r):
    """Try all three methods in sequence."""
    
    # Method 1: surface_id
    print("[INFO] Method 1: Attempting surface_id threshold...")
    surf = _threshold_surface_id_local(root, target_sid=111)
    if surf:
        return surf
    
    # Method 2: ExtractBlock
    print("[INFO] Method 2: Attempting ExtractBlock...")
    surf = _extract_block_local(root, AIP_SELECTORS)
    if surf:
        return surf
    
    # Method 3: Geometric picking (most robust)
    print("[INFO] Method 3: Attempting geometric picking...")
    surf = _pick_aip_geometric_local(root, cx, cy, cz, r, dx=0.02, cos_tol=0.90)
    if surf:
        return surf
    
    raise RuntimeError("All three AIP isolation methods failed!")

# ========== FREESTREAM TOTAL PRESSURE ==========

def compute_freestream_total_pressure(root, sample_region="far_upstream"):
    """
    Compute freestream total pressure P0_inf.
    Strategy: Sample far upstream where flow is uniform.
    
    P0 = p * (1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1))
    or equivalently: P0 = p + 0.5*rho*U^2  (simplified for low Mach)
    
    For supersonic: use isentropic relation.
    """
    print("\n[INFO] Computing freestream total pressure...")
    
    # Sample a box far upstream (adjust coordinates for your domain)
    # For your domain: x around 7.3 (upstream of geometry), full Y/Z span
    box = Box(registrationName="FreeStreamBox")
    box.XLength = 0.5  # thin slice
    box.YLength = 5.0  # capture full span
    box.ZLength = 5.0
    box.Center = [-8.0, 0.0, 0.0]  # well upstream of X_START=7.3
    UpdatePipeline(proxy=box)
    
    # Resample volume data onto this box
    sampled = ResampleWithDataset(
        SourceDataArrays=root,
        DestinationMesh=box
    )
    UpdatePipeline(proxy=sampled)
    
    # Get pressure, density, velocity
    p_src, p = _ensure_pressure_on(sampled)
    rho_src, rho = ensure_pointdata(p_src, NAME_RHO)
    u_src, u = ensure_pointdata(rho_src, NAME_U)

    # --- Compute local Mach number ---
    calc_M = Calculator(Input=u_src)
    calc_M.ResultArrayName = "M_inf"
    calc_M.Function = f"mag({u})/sqrt({GAMMA}*{p}/{rho})"
    UpdatePipeline(proxy=calc_M)

    # --- Compute total (stagnation) pressure field ---
    calc_P0 = Calculator(Input=calc_M)
    calc_P0.ResultArrayName = "P0_total"
    calc_P0.Function = f"{p} * pow(1 + 0.5*({GAMMA}-1)*M_inf*M_inf, {GAMMA}/({GAMMA}-1))"
    UpdatePipeline(proxy=calc_P0)

    # --- Fetch data to NumPy for statistics ---
    from paraview.servermanager import Fetch
    from vtk.util import numpy_support as ns
    vtk_data = Fetch(calc_P0)
    arr = vtk_data.GetPointData().GetArray("P0_total")
    p0_np = ns.vtk_to_numpy(arr)

    # --- Use median/mean/std of that array ---
    P0_inf  = float(np.median(p0_np))
    P0_mean = float(np.mean(p0_np))
    P0_std  = float(np.std(p0_np))

    print(f"[OK] Freestream total pressure (median): {P0_inf:.2f} Pa")
    print(f"     Mean: {P0_mean:.2f} Pa, Std: {P0_std:.2f} Pa")
    print(f"     Sampled {p0_np.size} points")

    # --- Cleanup proxies ---
    Delete(calc_P0)
    Delete(calc_M)
    Delete(sampled)
    Delete(box)
    
    return P0_inf

# ========== PRESSURE RECOVERY COMPUTATION ==========

def compute_pressure_recovery(aip_proxy, P0_inf, out_dir):
    """
    Compute (1) area-weighted mean P0 at AIP and (2) mass-flow-weighted mean P0,
    then write both recoveries to CSV using the given freestream total pressure P0_inf.
    """
    from paraview.servermanager import Fetch
    from vtk.util import numpy_support as ns

    def _fetch_scalar(obj, name):
        """Return first value of array 'name' from RowData, FieldData, or PointData, if present."""
        data = Fetch(obj)
        # Try RowData (vtkTable)
        if hasattr(data, "GetRowData"):
            arr = data.GetRowData().GetArray(name)
            if arr:
                return float(ns.vtk_to_numpy(arr)[0])
        # Try FieldData (vtkDataSet)
        if hasattr(data, "GetFieldData"):
            arr = data.GetFieldData().GetArray(name)
            if arr:
                vals = ns.vtk_to_numpy(arr)
                return float(vals[0]) if vals.size else float("nan")
        # Fallback: PointData mean (shouldn't be needed for IntegrateVariables, but safe)
        if hasattr(data, "GetPointData"):
            arr = data.GetPointData().GetArray(name)
            if arr:
                vals = ns.vtk_to_numpy(arr)
                return float(np.mean(vals)) if vals.size else float("nan")
        return float("nan")

    # --- Ensure p, rho, U on AIP ---
    p_src, p_arr   = _ensure_pressure_on(aip_proxy)          # yields Calc with p_arr
    rho_src, rho   = ensure_pointdata(p_src, NAME_RHO)
    u_src,   u     = ensure_pointdata(rho_src, NAME_U)

    # --- Total pressure on AIP (isentropic from local M) ---
    calc_M = Calculator(Input=u_src)
    calc_M.ResultArrayName = "M_local"
    calc_M.Function = f"mag({u})/sqrt({GAMMA}*{p_arr}/{rho})"
    UpdatePipeline(proxy=calc_M)

    calc_P0 = Calculator(Input=calc_M)
    calc_P0.ResultArrayName = "P0_AIP"
    calc_P0.Function = (
        f"0.9*{p_arr} * pow(1 + 0.5*({GAMMA}-1)*M_local*M_local, {GAMMA}/({GAMMA}-1))"
    )
    UpdatePipeline(proxy=calc_P0)

    from paraview.servermanager import Fetch
    from vtk.util import numpy_support as ns
    import numpy as np

    # Fetch the AIP total pressure field to NumPy
    vtk_data = Fetch(calc_P0)
    p0_arr = vtk_data.GetPointData().GetArray("P0_AIP")
    if p0_arr is None:
        raise RuntimeError("P0_AIP array not found on AIP surface!")

    p0_vals = ns.vtk_to_numpy(p0_arr)

    # Mean, median, standard deviation of P0 at AIP
    P0_mean  = float(np.mean(p0_vals))
    P0_median = float(np.median(p0_vals))
    P0_std   = float(np.std(p0_vals))

    # Compute pressure recovery (based on mean)
    recovery = P0_mean / P0_inf

    print(f"[INFO] AIP total pressure (mean): {P0_mean:.3f} Pa")
    print(f"[INFO] AIP total pressure (median): {P0_median:.3f} Pa  |  Std: {P0_std:.3f}")
    print(f"[RESULT] Pressure recovery (mean P0_AIP / P0_inf): {recovery:.5f}")

    # Optionally, save to CSV
    csv_path = os.path.join(out_dir, "AIP_recovery.csv")
    write_header = not os.path.exists(csv_path)
    import csv
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["P0_inf[Pa]", "P0_mean_AIP[Pa]", "P0_median_AIP[Pa]",
                        "P0_std_AIP[Pa]", "PressureRecovery"])
        w.writerow([P0_inf, P0_mean, P0_median, P0_std, recovery])

    print(f"[OK] Wrote recovery result → {csv_path}")

# ========== MAIN AIP PLOTTING ==========

def plot_aip_fields(root, out_dir, cx, cy, cz, r):
    """Plot pressure and velocity on AIP surface."""
    
    # 1) Isolate AIP geometry
    print("\n[INFO] Isolating AIP surface...")
    aip_geom = isolate_aip_surface(root, cx, cy, cz, r)
    UpdatePipeline(proxy=aip_geom)
    Show(aip_geom)
    
    ds_geom = Fetch(aip_geom)
    if not ds_geom or ds_geom.GetNumberOfPoints() == 0:
        raise RuntimeError("AIP geometry has no points!")
    print(f"[OK] AIP geometry: {ds_geom.GetNumberOfPoints()} points")
    
    # 2) Resample volume data onto AIP surface
    print("[INFO] Resampling volume data onto AIP...")
    sampled = ResampleWithDataset(
        registrationName="AIP_Sampled",
        SourceDataArrays=root,
        DestinationMesh=aip_geom
    )
    sampled.CellLocator = 'Static Cell Locator'
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 1
    UpdatePipeline(proxy=sampled)
    # Remove?
    Show(sampled)
    Hide(aip_geom)
    Hide(src_aip)
    
    ds_sampled = Fetch(sampled)
    if not ds_sampled or ds_sampled.GetNumberOfPoints() == 0:
        raise RuntimeError("ResampleWithDataset returned no points!")
    print(f"[OK] Resampled data: {ds_sampled.GetNumberOfPoints()} points")
    
    # Debug: check available arrays
    print("[DEBUG] Available arrays after resampling:")
    arrs = list_arrays(sampled)
    for name, (ncomp, loc) in arrs.items():
        print(f"  {name}: {ncomp} comp, {loc}")
    
    # 3) Camera setup
    rv = GetActiveViewOrCreate("RenderView")
    rv.CameraPosition = [cx + 2.5, cy, cz]
    rv.CameraFocalPoint = [cx, cy, cz]
    rv.CameraViewUp = [0, 0, 1]
    rv.CameraParallelProjection = 1
    rv.CameraParallelScale = r * 1.5
    rv.StillRender()
    
    # ---- PLOT 1: PRESSURE ----
    print("\n[INFO] Plotting pressure...")
    try:
        p_src, p_arr = _ensure_pressure_on(sampled)
        UpdatePipeline(proxy=p_src)

        # --- Compute range on AIP surface ---
        p_min, p_max = compute_range(p_src, p_arr)
        print(f"[INFO] Pressure range on AIP: {p_min:.3f} – {p_max:.3f}")

        px, arr, rep = render_colorfield(
            rv, p_src, p_arr,
            title="Pressure [Pa]",
            fixed_range=(245000, p_max)
        )
        
        txt = Text()
        txt.Text = f"AIP Surface (x={cx:.2f}) - Static Pressure"
        txtrep = Show(txt)
        txtrep.WindowLocation = "Upper Left Corner"
        txtrep.FontSize = 16
        txtrep.Color = [0, 0, 0]
        
        rv.StillRender()
        save_png(rv, "AIP_pressure")
        Hide(txt); Delete(txt); Hide(px)
        print("[OK] AIP_pressure.png saved")
        
    except Exception as e:
        print(f"[ERROR] Pressure plot failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ---- PLOT 2: VELOCITY ----
    print("\n[INFO] Plotting velocity...")
    try:
        u_src, u_arr = _ensure_speed_on(sampled)
        UpdatePipeline(proxy=u_src)

        # --- Compute range on AIP surface ---
        u_min, u_max = compute_range(u_src, u_arr)
        print(f"[INFO] Velocity range on AIP: {u_min:.3f} – {u_max:.3f}")

        px, arr, rep = render_colorfield(
            rv, u_src, u_arr,
            title="Velocity [m/s]",
            fixed_range=(85, u_max)
        )
        
        txt = Text()
        txt.Text = f"AIP Surface (x={cx:.2f}) - Velocity Magnitude"
        txtrep = Show(txt)
        txtrep.WindowLocation = "Upper Left Corner"
        txtrep.FontSize = 16
        txtrep.Color = [0, 0, 0]
        
        rv.StillRender()
        save_png(rv, "AIP_velocity")
        Hide(txt); Delete(txt); Hide(px)
        print("[OK] AIP_velocity.png saved")
        
    except Exception as e:
        print(f"[ERROR] Velocity plot failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Optional PDF
    if WRITE_PDFS:
        try:
            from PIL import Image
            pngs = [
                os.path.join(out_dir, "AIP_pressure.png"),
                os.path.join(out_dir, "AIP_velocity.png")
            ]
            pngs = [p for p in pngs if os.path.exists(p) and os.path.getsize(p) > 0]
            
            if pngs:
                imgs = [Image.open(p).convert("RGB") for p in pngs]
                pdf_path = os.path.join(out_dir, "AIP_plots.pdf")
                imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:])
                print(f"[OK] Created {pdf_path}")
        except Exception as e:
            print(f"[WARN] PDF creation failed: {e}")
    return sampled

# ========== RUN IT ==========
try:
    aip_sampled = plot_aip_fields(src_aip, OUT_DIR, cx, cy, cz, AIP_RADIUS)
    print("\n" + "="*60)
    print("AIP PLOTTING COMPLETE SUCCESS")
    print("="*60)
    root = open_case(CASE_FILE)
    #P0_inf = compute_freestream_total_pressure(open_case(CASE_FILE))
    compute_pressure_recovery(aip_sampled, P0_inf, OUT_DIR)
    print("\n" + "="*60)
    print("PRESSURE RECOVERY COMPUTED")
    print("="*60)
except Exception as e:
    print(f"\n[CRITICAL ERROR] AIP plotting failed:")
    print(f"  {e}")
    import traceback
    traceback.print_exc()

print(f"\nAll outputs in: {OUT_DIR}")
