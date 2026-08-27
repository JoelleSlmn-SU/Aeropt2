# Paraview.py — unified post-processing: x-sweep contours (rho/Mach/eta) plus
# AIP pressure recovery, distortion and swirl, all gated by boolean toggles.
# Run: pvpython Paraview.py
#
# ---------------------------------------------------------------------------
# WHAT CHANGED vs the previous two-script version (read once, then ignore):
#  - Everything lives in one file again; the duplicate helper functions that
#    existed in both Paraview.py and Paraview_AIP_Distortion_Swirl.py have
#    been merged into single canonical versions (ensure_pressure/ensure_mach/
#    ensure_speed/ensure_total_pressure). The AIP-only "_ensure_pressure_on"
#    /"_ensure_speed_on"/"_ensure_P0AIP_on" variants are gone.
#  - P0_inf (freestream total pressure) is now computed AT MOST ONCE per run
#    (previously it was computed twice: once for the eta sweep, once again
#    for the AIP section) and only if something actually needs it
#    (PLOT_ETA or COMPUTE_PR).
#  - The AIP pipeline is only rebuilt from scratch (full cleanup + re-open)
#    if the x-sweep actually ran; if you only want AIP/CSV output (e.g.
#    batch runs from an optimisation loop with the sweep plots off), the
#    case is opened once and reused directly.
#  - `isolate_surface_ids` (defined but never called anywhere in the
#    original script) has been dropped as dead code.
#  - Distortion and swirl are new: see the toggles and the method notes
#    below the CONFIG block.
# ---------------------------------------------------------------------------
# METHOD NOTES for distortion/swirl (read before trusting the numbers):
#  - AIP axis assumed = global +X through AIP_CENTER (same convention as the
#    x-sweep slice normal [1,0,0]). Swirl is decomposed in the local
#    cylindrical frame (x,r,theta) about that axis, theta=atan2(z-cz,y-cy).
#    If the duct axis isn't exactly global X at the AIP station, this is
#    wrong and needs a rotated frame — check first if swirl looks unphysical.
#  - COMPUTE_SWIRL and COMPUTE_PR both use the actual isolated AIP surface
#    mesh, triangle-area-weighted (not a plain point average — a plain mean
#    over an unstructured resampled point cloud is biased toward wherever
#    mesh points happen to be dense).
#  - COMPUTE_DISTORTION uses a separate synthetic ring/rake polar probe grid
#    (N_RINGS x N_RAKES, equal-area rings) resampled straight from the 3D
#    volume field — this is the standard SAE ARP1420-style instrumentation
#    pattern for a circumferential distortion coefficient DC(theta), and
#    doesn't depend on the AIP surface-isolation fallbacks succeeding.
#    DC_WINDOW_DEG defaults to 60 deg; N_RAKES=12 makes that exact (a 2-rake
#    window). q_bar = mean(P0-p) over valid probes (SAE/compressible
#    convention) — set QBAR_DEFINITION="0.5rhoU2" for the incompressible one.
#  - Swirl reported here is BULK only (area-weighted mean signed, mean |.|,
#    RMS, max). Full SAE AIR5686 paired (co-/counter-rotating) swirl
#    detection is NOT implemented.
#  - Clock convention for the distortion polar plot: "looking downstream",
#    theta=0 at 12 o'clock, clockwise by default (POLAR_CLOCKWISE) — check
#    this matches how you report clock positions elsewhere.
# ---------------------------------------------------------------------------

from pathlib import Path
import os, glob, csv, math
from datetime import datetime
import numpy as np
import vtk

from paraview.simple import *
from paraview.servermanager import Fetch
from vtk.util import numpy_support as ns

# ============================== CONFIG (case) ================================
folder = [
    "corner_optimisation_2"
]

x_case = 15
gen = 0
base = Path(r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples")
case_root = base / folder[0]

CASE_FILE = str(case_root / "postprocessed" / f"n_{gen}" / f"{str(x_case)}" / f"ENSIGHTcorner_{str(x_case)}.case")
OUT_DIR   = str(case_root / "postprocessed" / f"n_{gen}" / f"{str(x_case)}" / f"x_sweep_out")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.abspath(OUT_DIR)

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

# Matplotlib discrete rainbow (sweep plots)
CMAP_NAME = "rainbow"
N_STEPS   = 24
# Diverging colormap (AIP swirl/distortion plots)
CMAP_DIVERGING = "RdBu_r"

WRITE_PDFS = True

# Freestream sampling region
FS_CENTER  = (-8.0, 0.0, 0.0)
FS_LENGTHS = (0.6, 5.0, 5.0)
FS_NPTS    = 20000
DROP_ZEROS = True

# ========================= WHAT TO RUN (toggles) =============================
PLOT_DENSITY = False
PLOT_MACH    = False
PLOT_ETA     = False

PLOT_AIP           = True   # generate AIP contour PNGs (velocity + whichever of PR/distortion/swirl are on)
COMPUTE_PR         = True   # area-weighted pressure recovery on the AIP -> CSV
COMPUTE_DISTORTION = True   # SAE-style DC(theta) circumferential distortion -> CSV
COMPUTE_SWIRL      = True   # bulk swirl angle stats on the AIP -> CSV
# ===============================================================================

# AIP definition
AIP_CENTER = (11.3, 0.4, 0.21)
AIP_RADIUS = 0.37
AIP_SELECTORS = [
    "/Root/Surfaces/Surface 111",
    "/Root/Surface 111",
    "/Root/Blocks/Surface 111",
    "/Root/Block_111",
    "/Root/111",
]

# Distortion ring/rake grid
N_RINGS = 5
N_RAKES = 12               # 30 deg spacing; a 2-rake window = an EXACT 60 deg DC window
RING_SPACING = "area"      # "area" (equal-area rings, SAE-style) or "radius"
THETA_OFFSET_DEG = 0.0
DC_WINDOW_DEG = 60.0
QBAR_DEFINITION = "P0-p"   # "P0-p" (SAE/compressible) or "0.5rhoU2"
POLAR_CLOCKWISE = True

# Swirl / distortion plot ranges (None = auto per-run)
SWIRL_RANGE = (-15.0, 15.0)   # deg

RUN_TAG = f"{folder[0]}/n{gen}/x{x_case}"

RUN_SWEEP       = PLOT_DENSITY or PLOT_MACH or PLOT_ETA
RUN_AIP_SURFACE = PLOT_AIP or COMPUTE_PR or COMPUTE_SWIRL   # needs the real isolated AIP surface
RUN_AIP_PROBES  = COMPUTE_DISTORTION                         # needs the ring/rake probe grid
RUN_AIP         = RUN_AIP_SURFACE or RUN_AIP_PROBES
NEED_P0_INF     = PLOT_ETA or COMPUTE_PR

print(f"[INFO] CASE_FILE: {CASE_FILE}")
print(f"[INFO] OUT_DIR:   {OUT_DIR}")
print(f"[INFO] Sweep: density={PLOT_DENSITY} mach={PLOT_MACH} eta={PLOT_ETA}  ->  run_sweep={RUN_SWEEP}")
print(f"[INFO] AIP:   plot={PLOT_AIP} PR={COMPUTE_PR} distortion={COMPUTE_DISTORTION} swirl={COMPUTE_SWIRL}  ->  run_aip={RUN_AIP}")

# ================================ helpers =====================================

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

COLORMAP_NAME     = "Rainbow Uniform"
SHOW_COLORBAR     = True

def render_colorfield(view, src_proxy, array_name, title, fixed_range=None):
    px, arr = ensure_pointdata(src_proxy, array_name)
    SetActiveSource(px)
    rep = Show(px, view)
    rep.Representation = "Surface"
    ColorBy(rep, ("POINTS", arr))
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

def apply_matplotlib_discrete_cmap(array_name, cmap_name, n_steps, vmin, vmax, view, rep, title):
    from matplotlib import cm
    lut = GetColorTransferFunction(array_name)
    lut.Discretize = 1
    lut.NumberOfTableValues = int(n_steps)
    lut.RescaleTransferFunction(float(vmin), float(vmax))

    cmap = cm.get_cmap(cmap_name, int(n_steps))
    vals = np.linspace(float(vmin), float(vmax), int(n_steps))
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

def render_field(view, proxy, array_name, title, vmin=None, vmax=None, use_matplotlib=False, cmap_name=None):
    px, arr = ensure_pointdata(proxy, array_name)
    SetActiveSource(px)
    rep = Show(px, view)
    rep.Representation = "Surface"
    ColorBy(rep, ("POINTS", arr))

    if vmin is not None and vmax is not None:
        lut = GetColorTransferFunction(arr)
        lut.RescaleTransferFunction(float(vmin), float(vmax))
        pwf = GetOpacityTransferFunction(arr)
        pwf.RescaleTransferFunction(float(vmin), float(vmax))

    if use_matplotlib and (vmin is not None and vmax is not None):
        apply_matplotlib_discrete_cmap(arr, cmap_name or CMAP_NAME, N_STEPS, vmin, vmax, view, rep, title)
    else:
        rep.SetScalarBarVisibility(view, True)
        sb = GetScalarBar(GetColorTransferFunction(arr), view)
        sb.Title = title
        sb.LabelFormat = "%.3g"
        sb.ScalarBarLength = 0.35
        sb.WindowLocation = "Upper Right Corner"

    return px, rep

# ---------------- physics helpers (canonical, shared by sweep + AIP) ----------------
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

def ensure_total_pressure(proxy, result_name="P0"):
    """P0 = p*(1+0.5*(gamma-1)*M^2)^(gamma/(gamma-1)). Shared by ensure_eta
    (freestream/local eta) and the AIP pressure-recovery contour."""
    m_px, m = ensure_mach(proxy)
    p_px, p = ensure_pointdata(m_px, "p_calc")   # p_calc already computed inside ensure_mach->ensure_pressure
    calc = Calculator(Input=p_px)
    calc.ResultArrayName = result_name
    calc.Function = f"{p} * pow(1 + 0.5*({GAMMA}-1)*{m}*{m}, {GAMMA}/({GAMMA}-1))"
    UpdatePipeline(proxy=calc)
    return calc, result_name

def ensure_eta(proxy, P0_inf):
    p0_px, p0 = ensure_total_pressure(proxy, "P0_local")
    calc_eta = Calculator(Input=p0_px)
    calc_eta.ResultArrayName = "eta"
    calc_eta.Function = f"{p0}/{float(P0_inf)}"
    UpdatePipeline(proxy=calc_eta)
    return calc_eta, "eta"

def ensure_speed(proxy):
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

# ----------------- slice setup (sweep) -----------------
def build_slice(src):
    sl = Slice(registrationName="Slice", Input=src)
    sl.SliceType = "Plane"
    sl.SliceType.Normal = [1.0, 0.0, 0.0]
    sl.SliceType.Origin = [float(X_START), 0.0, 0.0]
    merged = MergeBlocks(registrationName="Slice_Merged", Input=sl)
    UpdatePipeline(proxy=merged)
    return sl, merged

# ---- AIP isolation chain ----
def _vtk_polydata_to_proxy(polydata, name="IsolatedSurface"):
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        return None
    prod = TrivialProducer(registrationName=name)
    prod.GetClientSideObject().SetOutput(polydata)
    UpdatePipeline(proxy=prod)
    return prod

def _direct_extract_surface_blocks_local(root, target_sids):
    """
    In the EnSight multiblock tree, `Surface N` corresponds to block index N,
    so this avoids fragile selector strings. Returns a ParaView proxy
    containing triangulated, cleaned surface polydata.
    """
    ds = Fetch(root)
    if ds is None or not hasattr(ds, "GetNumberOfBlocks"):
        print("[WARN] Direct block extraction skipped: root is not a multiblock dataset.")
        return None

    pieces = []
    nblocks = int(ds.GetNumberOfBlocks())
    print(f"[INFO] Direct block extraction for surfaces {target_sids}; root has {nblocks} blocks")

    for sid in target_sids:
        sid = int(sid)
        if sid < 0 or sid >= nblocks:
            print(f"[WARN] Surface {sid} is outside multiblock range 0..{nblocks-1}")
            continue
        block = ds.GetBlock(sid)
        if block is None:
            print(f"[WARN] Surface/block {sid} is None")
            continue
        pieces.append(block)
        try:
            npts = block.GetNumberOfPoints()
        except Exception:
            npts = -1
        print(f"[OK] Grabbed block {sid} directly ({npts} points)")

    if not pieces:
        return None

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

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(geom.GetOutput())
    clean.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(clean.GetOutput())
    tri.Update()

    out = tri.GetOutput()
    print(f"[OK] Direct surface extraction complete: {out.GetNumberOfPoints()} points, {out.GetNumberOfCells()} cells")
    return _vtk_polydata_to_proxy(out, name="DirectSurfaceBlockExtract")

def _threshold_set_local(thr, loc, arr_name, lo, hi):
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
    arrs = list_arrays(root)
    for nm in ("surface_id", "SurfaceId", "SURFACE_ID", "SurfaceID", "ElementBlockIds"):
        if nm in arrs:
            ncomp, loc = arrs[nm]
            print(f"[INFO] Found surface ID array: {nm} ({loc})")
            thr = Threshold(Input=root)
            _threshold_set_local(thr, loc, nm, target_sid - 0.5, target_sid + 0.5)
            UpdatePipeline(proxy=thr)
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
    print(f"[INFO] Geometric picking: dx={dx}, r={r}, cos_tol={cos_tol}")
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
    _threshold_set_local(t_x, "POINTS", "abs_x_dist", 0.0, float(dx))
    UpdatePipeline(proxy=t_x)

    ds_x = Fetch(t_x)
    if not ds_x or ds_x.GetNumberOfPoints() == 0:
        print(f"[WARN] X-slab threshold returned no points")
        return None
    print(f"[INFO] After X-slab: {ds_x.GetNumberOfPoints()} points")

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
    """Try the proven direct block extraction first, then robust fallbacks."""
    print("[INFO] Method 1: Direct multiblock extraction of AIP surface ID 111...")
    surf = _direct_extract_surface_blocks_local(root, [111])
    if surf:
        return surf
    print("[INFO] Method 2: Attempting ExtractBlock selectors...")
    surf = _extract_block_local(root, AIP_SELECTORS)
    if surf:
        return surf
    print("[INFO] Method 3: Attempting surface_id threshold...")
    surf = _threshold_surface_id_local(root, target_sid=111)
    if surf:
        return surf
    print("[INFO] Method 4: Attempting geometric picking...")
    surf = _pick_aip_geometric_local(root, cx, cy, cz, r, dx=0.01, cos_tol=0.90)
    if surf:
        return surf
    raise RuntimeError("All AIP isolation methods failed!")

# -------------------- distortion / swirl helpers --------------------

def fetch_point_arrays(proxy, names, need_points=False):
    ds = Fetch(proxy)
    if ds is None:
        raise RuntimeError("Fetch returned None dataset.")
    pd = ds.GetPointData()
    out = {}
    for nm in names:
        arr = pd.GetArray(nm) if pd else None
        out[nm] = ns.vtk_to_numpy(arr) if arr is not None else None
    if need_points:
        pts = ds.GetPoints()
        out["_points"] = ns.vtk_to_numpy(pts.GetData()) if pts is not None else None
    return ds, out

def compute_physics(rho, u, p):
    """Elementwise (M, P0, valid_mask) from rho[N], u[N,3], p[N]."""
    rho = np.asarray(rho, dtype=float)
    p = np.asarray(p, dtype=float)
    u = np.asarray(u, dtype=float)
    speed = np.linalg.norm(u, axis=1)
    valid = np.isfinite(rho) & np.isfinite(p) & (rho > 0) & (p > 0)
    M = np.full_like(p, np.nan)
    P0 = np.full_like(p, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        M[valid] = speed[valid] / np.sqrt(GAMMA * p[valid] / rho[valid])
        P0[valid] = p[valid] * np.power(1.0 + 0.5 * (GAMMA - 1.0) * M[valid] ** 2, GAMMA / (GAMMA - 1.0))
    return M, P0, valid

def cylindrical_decompose(u, theta):
    """theta: local angle (rad) about +X axis, theta=atan2(z-cz, y-cy)."""
    Ux, Uy, Uz = u[:, 0], u[:, 1], u[:, 2]
    ct, st = np.cos(theta), np.sin(theta)
    Ur = Uy * ct + Uz * st
    Ut = -Uy * st + Uz * ct
    swirl_deg = np.degrees(np.arctan2(Ut, Ux))
    return Ux, Ur, Ut, swirl_deg

def circular_mean_deg(angles_deg):
    a = np.radians(angles_deg[np.isfinite(angles_deg)])
    if a.size == 0:
        return float("nan")
    v = np.mean(np.exp(1j * a))
    return float(np.degrees(np.angle(v)))

def triangle_areas_and_centroid_values(ds, point_values):
    """Area-weighted mean of per-point scalar fields over a triangulated
    surface `ds`. point_values: dict name -> np.ndarray[N] aligned with ds
    points. Returns (means dict, total_area, triangulated_dataset)."""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(ds)
    tri.Update()
    tds = tri.GetOutput()

    pts = ns.vtk_to_numpy(tds.GetPoints().GetData())
    ncells = tds.GetNumberOfCells()
    areas = np.zeros(ncells)
    cell_vals = {k: np.zeros(ncells) for k in point_values}

    same_topology = (tds.GetNumberOfPoints() == ds.GetNumberOfPoints())
    if same_topology:
        remapped = point_values
    else:
        loc = vtk.vtkPointLocator()
        loc.SetDataSet(ds)
        loc.BuildLocator()
        idx = np.array([loc.FindClosestPoint(pts[i]) for i in range(pts.shape[0])])
        remapped = {k: v[idx] for k, v in point_values.items()}

    all_nan_counts = {k: 0 for k in point_values}
    id_list = vtk.vtkIdList()
    for c in range(ncells):
        tds.GetCellPoints(c, id_list)
        if id_list.GetNumberOfIds() != 3:
            continue
        i0, i1, i2 = id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        areas[c] = area
        for k, v in remapped.items():
            triplet = (v[i0], v[i1], v[i2])
            # A triangle whose 3 vertices are ALL NaN (e.g. all 3 landed on
            # resample points that failed compute_physics()'s validity mask)
            # would make np.nanmean emit a noisy "Mean of empty slice"
            # RuntimeWarning and return NaN anyway -- skip straight to NaN
            # instead, and count it so a large fraction is visible rather
            # than silently swallowed.
            if not np.any(np.isfinite(triplet)):
                cell_vals[k][c] = np.nan
                all_nan_counts[k] += 1
            else:
                cell_vals[k][c] = np.nanmean(triplet)

    for k, n_bad in all_nan_counts.items():
        if n_bad > 0:
            frac = n_bad / ncells
            level = "WARN" if frac > 0.02 else "INFO"
            print(f"[{level}] triangle_areas_and_centroid_values: {n_bad}/{ncells} "
                  f"({frac*100:.1f}%) cells had all-NaN '{k}' vertices (excluded from "
                  f"the area-weighted mean). A handful near the disc rim is expected "
                  f"resampling edge-effect; >2% may mean part of the AIP surface is "
                  f"landing outside the flow domain -- check AIP_CENTER/AIP_RADIUS.")

    total_area = float(np.sum(areas))
    means = {}
    for k, v in cell_vals.items():
        good = np.isfinite(v)
        w = np.where(good, areas, 0.0)
        denom = np.sum(w)
        means[k] = float(np.sum(w * np.where(good, v, 0.0)) / denom) if denom > 0 else float("nan")
    return means, total_area, tds

def build_polar_probe_cloud(cx, cy, cz, R, n_rings, n_rakes, ring_spacing="area", theta_offset_deg=0.0):
    """Structured ring/rake point cloud in the plane x=cx, centred at (cy,cz)."""
    if ring_spacing == "area":
        r_edges = R * np.sqrt(np.arange(n_rings + 1) / float(n_rings))
    elif ring_spacing == "radius":
        r_edges = R * (np.arange(n_rings + 1) / float(n_rings))
    else:
        raise ValueError("RING_SPACING must be 'area' or 'radius'")
    r_bar = np.sqrt(0.5 * (r_edges[:-1] ** 2 + r_edges[1:] ** 2))

    dtheta = 2.0 * np.pi / n_rakes
    theta0 = np.radians(theta_offset_deg)
    theta_centers = theta0 + dtheta * np.arange(n_rakes)

    ring_idx = np.repeat(np.arange(n_rings), n_rakes)
    rake_idx = np.tile(np.arange(n_rakes), n_rings)
    r = r_bar[ring_idx]
    theta = theta_centers[rake_idx]

    X = np.full_like(r, cx)
    Y = cy + r * np.cos(theta)
    Z = cz + r * np.sin(theta)
    coords = np.column_stack([X, Y, Z]).astype(np.float64)

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(ns.numpy_to_vtk(coords, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    verts = vtk.vtkCellArray()
    for i in range(coords.shape[0]):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    poly.SetVerts(verts)

    proxy = _vtk_polydata_to_proxy(poly, name="AIP_PolarProbes")
    return proxy, ring_idx, rake_idx, r, theta, r_edges, r_bar

def plot_polar_field(field_2d, r_edges, n_rakes, theta_offset_deg, clockwise,
                      title, out_path, vmin=None, vmax=None, cmap="RdBu_r"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtheta = 2.0 * np.pi / n_rakes
    theta0 = np.radians(theta_offset_deg)
    theta_edges = theta0 + dtheta * (np.arange(n_rakes + 1) - 0.5)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1 if clockwise else 1)

    if vmin is None or vmax is None:
        finite = field_2d[np.isfinite(field_2d)]
        vmax_auto = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmin, vmax = -vmax_auto, vmax_auto

    mesh = ax.pcolormesh(theta_edges, r_edges, field_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_rlabel_position(135)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, pad=0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

def clock_position(theta_deg, clockwise=True):
    """Cosmetic conversion of a probe angle to a 12-o'clock-referenced clock
    label, consistent with plot_polar_field's orientation."""
    deg = theta_deg % 360.0
    hours = (deg / 30.0) if clockwise else ((360.0 - deg) / 30.0)
    hours = hours % 12.0
    if hours == 0:
        hours = 12.0
    return hours

# ================================= MAIN ========================================

src_raw = open_case(CASE_FILE)               # raw multiblock reader (needed for AIP block extraction)
src = normalize_to_single_dataset(src_raw)   # merged single dataset (used for all resampling)

P0_inf = None
if NEED_P0_INF:
    P0_inf = compute_P0_inf(src)
else:
    print("[INFO] Skipping freestream P0_inf (PLOT_ETA and COMPUTE_PR are both off).")

results = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "run_tag": RUN_TAG,
    "case_file": CASE_FILE,
    "P0_inf_Pa": P0_inf if P0_inf is not None else "",
}

# ================================= X-SWEEP =====================================
if RUN_SWEEP:
    rv = GetActiveViewOrCreate("RenderView")
    rv.ViewSize = IMAGE_SIZE
    rv.Background = [1, 1, 1]
    rv.InteractionMode = "2D"
    Show(src)
    ResetCamera()

    rv.CameraParallelProjection = 1
    rv.CameraPosition = [-3, 0, 0]
    rv.CameraFocalPoint = [0, 0.5, 0]
    rv.CameraViewUp = [0, 0, 1]
    rv.CameraParallelScale = 1.8
    rv.StillRender()

    slice_obj, slice_merged = build_slice(src)
    Hide(src)

    rho_proxy = mach_proxy = eta_proxy = None
    if PLOT_DENSITY:
        rho_proxy, _ = ensure_pointdata(slice_merged, NAME_RHO)
    if PLOT_MACH:
        mach_proxy, _ = ensure_mach(slice_merged)
    if PLOT_ETA:
        eta_proxy, _ = ensure_eta(slice_merged, P0_inf)

    x_vals = np.linspace(float(X_START), float(X_END), int(N_SLICES))

    print("\n[INFO] Starting sweep...")
    for i, xv in enumerate(x_vals):
        slice_obj.SliceType.Origin = [float(xv), 0.0, 0.0]
        UpdatePipeline(proxy=slice_merged)

        if i % 10 == 0:
            print(f"[INFO] Slice {i+1}/{N_SLICES} at x={xv:.3f}")

        if PLOT_DENSITY:
            t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {xv:.3f}"
            trep = Show(t); trep.WindowLocation="Upper Left Corner"; trep.FontSize=14; trep.Color=[0,0,0]
            px, rep = render_field(rv, rho_proxy, NAME_RHO, "Density")
            rv.StillRender(); save_png(rv, f"x_{i:03d}_rho")
            Hide(px); Hide(t); Delete(t)

        if PLOT_MACH:
            t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {xv:.3f}"
            trep = Show(t); trep.WindowLocation="Upper Left Corner"; trep.FontSize=14; trep.Color=[0,0,0]
            px, rep = render_field(rv, mach_proxy, "Mach", "Mach", vmin=MACH_RANGE[0], vmax=MACH_RANGE[1], use_matplotlib=True)
            rv.StillRender(); save_png(rv, f"x_{i:03d}_mach")
            Hide(px); Hide(t); Delete(t)

        if PLOT_ETA:
            t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {xv:.3f}"
            trep = Show(t); trep.WindowLocation="Upper Left Corner"; trep.FontSize=14; trep.Color=[0,0,0]
            px, rep = render_field(rv, eta_proxy, "eta", "\u03b7 [-]", vmin=ETA_RANGE[0], vmax=ETA_RANGE[1], use_matplotlib=True)
            rv.StillRender(); save_png(rv, f"x_{i:03d}_eta")
            Hide(px); Hide(t); Delete(t)

    print("[OK] Sweep complete.")

    if WRITE_PDFS:
        if PLOT_DENSITY: pngs_to_pdf("x_*_rho.png",  "rho_sweep.pdf")
        if PLOT_MACH:    pngs_to_pdf("x_*_mach.png", "mach_sweep.pdf")
        if PLOT_ETA:     pngs_to_pdf("x_*_eta.png",  "eta_sweep.pdf")

    print(f"[OK] Sweep outputs in: {OUT_DIR}")
else:
    print("[INFO] Skipping x-sweep entirely (PLOT_DENSITY, PLOT_MACH, PLOT_ETA are all off).")

# ================================== AIP ========================================
if RUN_AIP:
    print("\n" + "="*60)
    print("Starting AIP extraction / distortion / swirl")
    print("="*60)

    if RUN_SWEEP:
        print("[INFO] Cleaning up sweep pipeline...")
        try:
            for obj in list(GetSources().values()):
                try: Delete(obj)
                except: pass
        except Exception as e:
            print(f"[WARN] Cleanup: {e}")
        print(f"[INFO] Re-opening case file for AIP stage: {CASE_FILE}")
        src_raw = open_case(CASE_FILE)
        src = normalize_to_single_dataset(src_raw)

    cx, cy, cz = AIP_CENTER
    print(f"[INFO] AIP center: ({cx}, {cy}, {cz}), radius: {AIP_RADIUS}")

    rv = None

    if PLOT_AIP:
        rv = GetActiveViewOrCreate("RenderView")
        rv.ViewSize = IMAGE_SIZE
        rv.Background = [1, 1, 1]
        rv.InteractionMode = "3D"

        # ---------------------------------------------------------
        # AIP-ONLY VIEW
        # Hide every existing ParaView source before plotting AIP.
        # ---------------------------------------------------------
        for proxy in list(GetSources().values()):
            try:
                Hide(proxy, rv)
            except Exception:
                pass

        # Look directly downstream at the AIP.
        # The AIP plane normal is +X, so the camera sits upstream/downstream
        # on the X axis and looks directly at the Y-Z plane.
        rv.CameraPosition = [cx + 2.5, cy, cz]
        rv.CameraFocalPoint = [cx, cy, cz]
        rv.CameraViewUp = [0, 0, 1]

        # Orthographic view prevents perspective distortion of the disc.
        rv.CameraParallelProjection = 1

        # Frame tightly around the AIP.
        rv.CameraParallelScale = AIP_RADIUS * 1.15

        rv.StillRender()

    else:
        print(
            "[INFO] PLOT_AIP is off — skipping RenderView/screenshot "
            "setup, computing metrics only."
        )

    # ---------- PATH 1: real AIP surface (PR + swirl, area-weighted) ----------
    P0_bar_area = p_bar_area = q_bar_area = None
    swirl_mean_signed = swirl_mean_abs = swirl_rms = swirl_max_abs = None

    if RUN_AIP_SURFACE:
        try:
            print("\n[INFO] Isolating AIP surface...")
            aip_surf = isolate_aip_surface(src_raw, cx, cy, cz, AIP_RADIUS)
            UpdatePipeline(proxy=aip_surf)
            
            print("\n[DEBUG] Arrays directly on extracted Surface 111:")
            print(list_arrays(aip_surf))

            print("[INFO] Using flow arrays already present on extracted Surface 111...")
            sampled_surf = aip_surf
            UpdatePipeline(proxy=sampled_surf)
            
            ds_check = Fetch(sampled_surf)

            mask = ds_check.GetPointData().GetArray("vtkValidPointMask")

            if mask is not None:
                mask_np = ns.vtk_to_numpy(mask).astype(bool)
                print(
                    f"[DEBUG] AIP Resample validity: "
                    f"{np.count_nonzero(mask_np)}/{len(mask_np)} "
                    f"({100*np.mean(mask_np):.2f}%)"
                )

            p_src, p_arr = ensure_pressure(sampled_surf)
            ds_surf, arrs = fetch_point_arrays(p_src, [NAME_RHO, NAME_U, p_arr], need_points=True)
            rho = arrs[NAME_RHO]; u = arrs[NAME_U]; p = arrs[p_arr]; pts = arrs["_points"]
            if rho is None or u is None or p is None or pts is None:
                raise RuntimeError("Missing expected arrays on the resampled AIP surface.")

            M, P0, valid = compute_physics(rho, u, p)
            theta = np.arctan2(pts[:, 2] - cz, pts[:, 1] - cy)
            Ux, Ur, Ut, swirl_deg = cylindrical_decompose(u, theta)

            P0m = np.where(valid, P0, np.nan)
            pm = np.where(valid, p, np.nan)
            swirlm = np.where(valid, swirl_deg, np.nan)

            means, total_area, tds = triangle_areas_and_centroid_values(
                ds_surf, {"P0": P0m, "p": pm, "abs_swirl": np.abs(swirlm), "sq_swirl": swirlm ** 2}
            )
            P0_bar_area = means["P0"]; p_bar_area = means["p"]
            q_bar_area = (P0_bar_area - p_bar_area) if (not math.isnan(P0_bar_area) and not math.isnan(p_bar_area)) else float("nan")
            swirl_mean_abs = means["abs_swirl"]
            swirl_rms = math.sqrt(means["sq_swirl"]) if not math.isnan(means["sq_swirl"]) else float("nan")
            swirl_mean_signed = circular_mean_deg(swirlm)
            swirl_max_abs = float(np.nanmax(np.abs(swirlm))) if np.any(np.isfinite(swirlm)) else float("nan")

            print(f"[OK] AIP surface: {ds_surf.GetNumberOfPoints()} points, area = {total_area:.5g} m^2")
            if COMPUTE_PR and P0_inf:
                print(f"[RESULT] Pressure recovery (area-weighted) = {P0_bar_area/P0_inf:.5f}  (P0_bar={P0_bar_area:.6g} Pa)")
            if COMPUTE_SWIRL:
                print(f"[RESULT] Swirl (area-weighted): mean(signed)={swirl_mean_signed:.2f} deg, "
                      f"mean(|.|)={swirl_mean_abs:.2f} deg, RMS={swirl_rms:.2f} deg, max(|.|)={swirl_max_abs:.2f} deg")

            if PLOT_AIP:
                print("\n[INFO] Plotting velocity...")
                u_src, u_arr = ensure_speed(sampled_surf)
                u_min, u_max = compute_range(u_src, u_arr)
                px, arr, rep = render_colorfield(rv, u_src, u_arr, title="Velocity [m/s]", fixed_range=(u_min, 180))
                txt = Text(); txt.Text = f"AIP Surface (x={cx:.2f}) - Velocity Magnitude"
                trep = Show(txt); trep.WindowLocation = "Upper Left Corner"; trep.FontSize = 16; trep.Color = [0, 0, 0]
                rv.StillRender(); save_png(rv, "AIP_velocity")
                Hide(txt); Delete(txt); Hide(px)
                print("[OK] AIP_velocity.png saved")

                if COMPUTE_PR:
                    print("\n[INFO] Plotting pressure recovery (P0/P0_inf)...")
                    p0_src, p0_arr = ensure_total_pressure(sampled_surf, "P0_AIP")
                    rec = Calculator(Input=p0_src)
                    rec.ResultArrayName = "PR_AIP"
                    rec.Function = f"{p0_arr} / {float(P0_inf)}"
                    UpdatePipeline(proxy=rec)
                    px, arr, rep = render_colorfield(rv, rec, "PR_AIP", title="Pressure Recovery (P0/P0_inf) [-]",
                                                       fixed_range=(0.9, 0.98))
                    txt = Text(); txt.Text = f"AIP Surface (x={cx:.2f}) - Pressure Recovery"
                    trep = Show(txt); trep.WindowLocation = "Upper Left Corner"; trep.FontSize = 16; trep.Color = [0, 0, 0]
                    rv.StillRender(); save_png(rv, "AIP_pressure_recovery")
                    Hide(txt); Delete(txt); Hide(px)
                    print("[OK] AIP_pressure_recovery.png saved")

                if COMPUTE_SWIRL:
                    print("\n[INFO] Plotting swirl angle...")
                    ds_plot = vtk.vtkPolyData()
                    ds_plot.DeepCopy(ds_surf)
                    swirl_vtk = ns.numpy_to_vtk(np.nan_to_num(swirlm, nan=0.0), deep=True)
                    swirl_vtk.SetName("SwirlAngleDeg")
                    ds_plot.GetPointData().AddArray(swirl_vtk)
                    swirl_plot_proxy = _vtk_polydata_to_proxy(ds_plot, name="AIP_Swirl_Plot")

                    if SWIRL_RANGE is not None:
                        svmin, svmax = SWIRL_RANGE
                    else:
                        finite = swirlm[np.isfinite(swirlm)]
                        svmax = float(np.nanmax(np.abs(finite))) if finite.size else 10.0
                        svmin, svmax = -svmax, svmax

                    px, rep = render_field(rv, swirl_plot_proxy, "SwirlAngleDeg", "Swirl angle [deg]",
                                            vmin=svmin, vmax=svmax, use_matplotlib=True, cmap_name=CMAP_DIVERGING)
                    txt = Text(); txt.Text = f"AIP Surface (x={cx:.2f}) - Swirl Angle"
                    trep = Show(txt); trep.WindowLocation = "Upper Left Corner"; trep.FontSize = 16; trep.Color = [0, 0, 0]
                    rv.StillRender(); save_png(rv, "AIP_swirl_angle")
                    Hide(txt); Delete(txt); Hide(px)
                    print("[OK] AIP_swirl_angle.png saved")

        except Exception as e:
            print(f"[ERROR] AIP surface path (PR/swirl) failed: {e}")
            import traceback; traceback.print_exc()

    # ---------- PATH 2: ring/rake probe grid (distortion) ----------
    DC_value = radial_distortion = q_bar_probes = P0_bar_probes = None
    worst_ring = DC_worst_clock = effective_window_deg = None

    if RUN_AIP_PROBES:
        try:
            print("\n[INFO] Building ring/rake probe grid and resampling volume field onto it...")
            probe_proxy, ring_idx, rake_idx, r_probe, theta_probe, r_edges, r_bar = build_polar_probe_cloud(
                cx, cy, cz, AIP_RADIUS, N_RINGS, N_RAKES, RING_SPACING, THETA_OFFSET_DEG
            )
            sampled_probes = ResampleWithDataset(registrationName="AIP_Probes_Sampled",
                                                  SourceDataArrays=src, DestinationMesh=probe_proxy)
            sampled_probes.PassPointArrays = 1
            sampled_probes.PassCellArrays = 1
            UpdatePipeline(proxy=sampled_probes)

            p_src2, p_arr2 = ensure_pressure(sampled_probes)
            _, arrs2 = fetch_point_arrays(p_src2, [NAME_RHO, NAME_U, p_arr2, "vtkValidPointMask"])
            rho2 = arrs2[NAME_RHO]; u2 = arrs2[NAME_U]; p2 = arrs2[p_arr2]; mask2 = arrs2["vtkValidPointMask"]
            if rho2 is None or u2 is None or p2 is None:
                raise RuntimeError("Probe-grid resampling did not produce expected arrays.")

            M2, P0_2, phys_valid = compute_physics(rho2, u2, p2)
            probe_mask = mask2.astype(bool) if mask2 is not None else np.ones_like(p2, dtype=bool)
            if mask2 is None:
                print("[WARN] 'vtkValidPointMask' not found; assuming all probes are valid.")
            valid2 = probe_mask & phys_valid

            valid_fraction = float(np.mean(valid2))
            if valid_fraction < 0.98:
                print(f"[WARN] Only {valid_fraction*100:.1f}% of ring/rake probes landed inside the flow domain "
                      f"— check AIP_CENTER/AIP_RADIUS and the assumed +X axial direction if this is low.")

            P0_valid = np.where(valid2, P0_2, np.nan)
            p_valid = np.where(valid2, p2, np.nan)
            P0_grid = P0_valid.reshape(N_RINGS, N_RAKES)

            P0_bar_probes = float(np.nanmean(P0_valid))
            p_bar_probes = float(np.nanmean(p_valid))
            if QBAR_DEFINITION == "P0-p":
                q_bar_probes = P0_bar_probes - p_bar_probes
            elif QBAR_DEFINITION == "0.5rhoU2":
                speed2 = np.linalg.norm(u2, axis=1)
                q_bar_probes = float(np.nanmean(np.where(valid2, 0.5 * rho2 * speed2 ** 2, np.nan)))
            else:
                raise ValueError("QBAR_DEFINITION must be 'P0-p' or '0.5rhoU2'")

            ring_avgs = np.nanmean(P0_grid, axis=1)

            rake_spacing_deg = 360.0 / N_RAKES
            n_window = max(1, int(round(DC_WINDOW_DEG / rake_spacing_deg)))
            effective_window_deg = n_window * rake_spacing_deg
            if abs(effective_window_deg - DC_WINDOW_DEG) > 1e-6:
                print(f"[WARN] Requested DC window {DC_WINDOW_DEG:.1f} deg isn't a multiple of the rake spacing "
                      f"({rake_spacing_deg:.2f} deg/rake, N_RAKES={N_RAKES}). Using {effective_window_deg:.2f} deg "
                      f"instead (adjust N_RAKES for an exact window).")

            DC_ring = np.full(N_RINGS, np.nan)
            DC_ring_theta = np.full(N_RINGS, np.nan)
            theta_grid = theta_probe.reshape(N_RINGS, N_RAKES)
            for k in range(N_RINGS):
                best_avg = np.inf
                best_start = 0
                for start in range(N_RAKES):
                    idx = [(start + i) % N_RAKES for i in range(n_window)]
                    vals = P0_grid[k, idx]
                    if np.all(np.isnan(vals)):
                        continue
                    avg = np.nanmean(vals)
                    if avg < best_avg:
                        best_avg = avg
                        best_start = start
                if np.isfinite(best_avg) and np.isfinite(q_bar_probes) and q_bar_probes != 0:
                    DC_ring[k] = (ring_avgs[k] - best_avg) / q_bar_probes
                    DC_ring_theta[k] = np.degrees(theta_grid[k, best_start])

            worst_ring = int(np.nanargmax(DC_ring)) if np.any(np.isfinite(DC_ring)) else None
            DC_value = float(DC_ring[worst_ring]) if worst_ring is not None else float("nan")
            DC_worst_clock = clock_position(DC_ring_theta[worst_ring], POLAR_CLOCKWISE) if worst_ring is not None else float("nan")
            radial_distortion = (float((np.nanmax(ring_avgs) - np.nanmin(ring_avgs)) / P0_bar_probes)
                                  if np.isfinite(P0_bar_probes) and P0_bar_probes != 0 else float("nan"))

            print(f"[OK] Probe-grid P0_bar = {P0_bar_probes:.6g} Pa, q_bar ({QBAR_DEFINITION}) = {q_bar_probes:.6g} Pa")
            print(f"[RESULT] DC({effective_window_deg:.0f}) = {DC_value:.4f} "
                  f"(worst ring {worst_ring+1 if worst_ring is not None else '?'}/{N_RINGS}, "
                  f"~{DC_worst_clock:.1f} o'clock)")
            print(f"[RESULT] Radial pressure distortion (ring-to-ring) = {radial_distortion:.4f}")

            if P0_bar_area is not None and P0_bar_probes:
                pct_diff = 100.0 * abs(P0_bar_area - P0_bar_probes) / P0_bar_probes
                print(f"[CHECK] Area-weighted P0_bar vs probe-grid P0_bar differ by {pct_diff:.2f}% "
                      f"(large disagreement -> increase N_RINGS/N_RAKES or re-check AIP_CENTER/AIP_RADIUS).")

            if PLOT_AIP:
                print("\n[INFO] Plotting distortion polar carpet...")
                plot_polar_field(
                    P0_grid / (P0_bar_probes if P0_bar_probes else 1.0) - 1.0,
                    r_edges, N_RAKES, THETA_OFFSET_DEG, POLAR_CLOCKWISE,
                    "AIP P0 deficit (P0/P0bar - 1)",
                    os.path.join(OUT_DIR, "AIP_distortion_polar.png"),
                    cmap=CMAP_DIVERGING,
                )

        except Exception as e:
            print(f"[ERROR] AIP probe-grid (distortion) path failed: {e}")
            import traceback; traceback.print_exc()

    # ---------- CSV ----------
    results.update({
        "P0_bar_area_Pa": P0_bar_area if P0_bar_area is not None else "",
        "recovery_area": (P0_bar_area / P0_inf) if (COMPUTE_PR and P0_bar_area is not None and P0_inf) else "",
        "P0_bar_probes_Pa": P0_bar_probes if P0_bar_probes is not None else "",
        "recovery_probes": (P0_bar_probes / P0_inf) if (COMPUTE_PR and P0_bar_probes is not None and P0_inf) else "",
        "q_bar_Pa": q_bar_probes if q_bar_probes is not None else "",
        "qbar_definition": QBAR_DEFINITION if COMPUTE_DISTORTION else "",
        "DC_window_deg_requested": DC_WINDOW_DEG if COMPUTE_DISTORTION else "",
        "DC_window_deg_effective": effective_window_deg if effective_window_deg is not None else "",
        "DC_value": DC_value if DC_value is not None else "",
        "DC_worst_ring": (worst_ring + 1) if worst_ring is not None else "",
        "DC_worst_clock_oclock": DC_worst_clock if DC_worst_clock is not None else "",
        "radial_pressure_distortion": radial_distortion if radial_distortion is not None else "",
        "swirl_mean_signed_deg": swirl_mean_signed if swirl_mean_signed is not None else "",
        "swirl_mean_abs_deg": swirl_mean_abs if swirl_mean_abs is not None else "",
        "swirl_rms_deg": swirl_rms if swirl_rms is not None else "",
        "swirl_max_abs_deg": swirl_max_abs if swirl_max_abs is not None else "",
        "N_RINGS": N_RINGS if COMPUTE_DISTORTION else "",
        "N_RAKES": N_RAKES if COMPUTE_DISTORTION else "",
        "AIP_center": str(AIP_CENTER),
        "AIP_radius_m": AIP_RADIUS,
        "PLOT_DENSITY": PLOT_DENSITY, "PLOT_MACH": PLOT_MACH, "PLOT_ETA": PLOT_ETA, "PLOT_AIP": PLOT_AIP,
        "COMPUTE_PR": COMPUTE_PR, "COMPUTE_DISTORTION": COMPUTE_DISTORTION, "COMPUTE_SWIRL": COMPUTE_SWIRL,
    })

    csv_path = os.path.join(OUT_DIR, "AIP_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        if write_header:
            w.writeheader()
        w.writerow(results)
    print(f"\n[OK] Wrote AIP results (PR / distortion / swirl) -> {csv_path}")

else:
    print("\n[INFO] Skipping AIP stage entirely (PLOT_AIP, COMPUTE_PR, COMPUTE_DISTORTION, COMPUTE_SWIRL are all off).")

print(f"\nAll outputs in: {OUT_DIR}")