# sweep_x_rho_mach.py — ParaView batch script for ρ & Mach x-sweep contours
# Usage: pvpython sweep_x_rho_mach.py

# ========================= CONFIG =========================
CASE_FILE   = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\temp\smaller mf\ENSIGHTcorner.case"   # .case/.foam/.pvtu/.vtu/etc.
OUT_DIR     = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\temp\contours\smaller mf\x_sweep_out"              # output directory for images + HTML
IMAGE_SIZE  = (1200, 600)                   # pixels (w, h)

# Sweep extents and resolution
X_START, X_END = 7.3, 13.5
N_SLICES       = 63

# Gas properties (used only if we must compute Mach or rho)
GAMMA  = 1.4
R_SPEC = 287.0   # J/(kg·K) for air (set to 1.0 if your data are non-dimensional)
PATM = 101000.0
PNORM_RANGE = (0.0, 0.9)

# Field array names as they appear in your dataset
NAME_U        = "velocity"    # vector (u,v,w)
NAME_USCALED  = "velscaled"    # vector (u,v,w)
NAME_RHO      = "density"     # scalar
NAME_ENERGY   = "energy"

WALLDIST_CANDIDATES = ["wallDistance", "yPlus", "y_plus", "Yplus", "yplus", "y_plus_avg", "distance_to_wall"]

COLORMAP_NAME     = "Rainbow Uniform"     # or "Rainbow", "Rainbow Uniform", etc.
SHOW_COLORBAR     = True
USE_GLOBAL_RANGE  = True                      # keep same scale across slices
MACH_DISCRETE_STEPS = 12
MACH_RANGE          = (0.39, 1.3)
# If you have a vibrant preset JSON exported from ParaView/EnSight, set these:
MACH_PRESET_JSON    = None  
MACH_PRESET_NAME    = None

DELTA99_LEVEL     = 0.99                      # U/Ue level
UE_PERCENTILE     = 95.0

WRITE_PDFS        = True
WRITE_DELTA99_CSV = True

# Which variables to output (fixed to rho & Mach per your request)
VARS_TO_PLOT = ("rho", "Mach")
# ======================= END CONFIG =======================

import os, math, glob, csv
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[INFO] Writing images to: {OUT_DIR}")
print(f"[INFO] CWD: {os.getcwd()}")

from paraview.simple import *
from paraview.servermanager import Fetch
from vtk.util import numpy_support as ns
import numpy as np

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

def get_array_numpy(proxy, array_name):
    """Return 1D numpy array for scalar or magnitude for vector (POINTS preferred)."""
    proxy = normalize_to_single_dataset(proxy)
    vtk_ds = Fetch(proxy)
    if vtk_ds is None: return None
    pd, cd = vtk_ds.GetPointData(), vtk_ds.GetCellData()
    arr = pd.GetArray(array_name) if pd else None
    if arr is None and cd: arr = cd.GetArray(array_name)
    if arr is None: return None
    ncomp = arr.GetNumberOfComponents()
    np_arr = ns.vtk_to_numpy(arr)
    if ncomp == 1: return np_arr
    return np.linalg.norm(np_arr.reshape(-1,ncomp), axis=1)

def save_png(view, basename):
    path = os.path.join(OUT_DIR, basename + ".png")
    try: view.EnableRayTracing = 0
    except Exception: pass
    try:
        Render(view=view); view.StillRender()
        SaveScreenshot(filename=path, viewOrLayout=view, ImageResolution=IMAGE_SIZE)
    except TypeError:
        SaveScreenshot(path, view, ImageResolution=IMAGE_SIZE)
    except Exception as e:
        print(f"[ERROR] SaveScreenshot failed: {e}")
        try:
            WriteImage(path, view=view)
        except Exception as e2:
            print(f"[ERROR] WriteImage failed: {e2}")
    try:
        sz = os.path.getsize(path); print(f"[OK] Saved: {path} ({sz} bytes)")
    except Exception:
        print(f"[WARN] Save reported success but file not found: {path}")
    return path

def ensure_density(slice_proxy):
    SetActiveSource(slice_proxy)
    return ensure_pointdata(slice_proxy, NAME_RHO)

def ensure_pressure(slice_proxy):
    px_rho, rho_nm = ensure_pointdata(slice_proxy, NAME_RHO)
    px_e,   e_nm   = ensure_pointdata(px_rho, NAME_ENERGY)
    px_u, u_nm = ensure_pointdata(slice_proxy, NAME_U)
    calc = Calculator(Input=px_e)
    calc.ResultArrayName = "p_calc"
    calc.Function = f"({e_nm}-0.5*{rho_nm}*mag({u_nm})^2)*({GAMMA}-1)"
    Show(calc); Hide(px_e); SetActiveSource(calc)
    return calc, "p_calc"

def ensure_speed(slice_proxy):
    px_u, u_nm = ensure_pointdata(slice_proxy, NAME_U)
    calc = Calculator(Input=px_u)
    calc.ResultArrayName = "U_mag"
    calc.Function = f"mag({u_nm})"
    Show(calc); Hide(px_u); SetActiveSource(calc)
    return calc, "U_mag"

def ensure_mach(slice_proxy):
    """Build Mach = |U| / sqrt(GAMMA * p / rho) with p from energy."""
    # 1) ensure p
    p_px, p_nm = ensure_pressure(slice_proxy)
    # 2) ensure rho & U are point-data
    px_rho, rho_nm = ensure_pointdata(p_px, NAME_RHO)
    px_u,   u_nm   = ensure_pointdata(px_rho, NAME_U)
    # 3) calculator for Mach
    calc = Calculator(Input=px_u)
    calc.ResultArrayName = "Mach"
    calc.Function = f"mag({u_nm})/sqrt({GAMMA}*{p_nm}/{rho_nm})"
    Show(calc); Hide(px_u); SetActiveSource(calc)
    return calc, "Mach"

def find_walldist_name(slice_proxy):
    arrs = list_arrays(slice_proxy)
    for name in WALLDIST_CANDIDATES:
        if name in arrs:
            return name
    return None

def _apply_matplotlib_colormap(array_name, cmap_name='rainbow', n_steps=12, vmin=0.39, vmax=1.3, view=None, rep=None):
    """
    Use a vibrant Matplotlib colormap for the given scalar array in ParaView.
    Works for headless pvpython too (no GUI preset import needed).
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np

    # Get the LUT object
    lut = GetColorTransferFunction(array_name)
    lut.Discretize = 1
    lut.NumberOfTableValues = n_steps
    lut.RescaleTransferFunction(vmin, vmax)

    # Get RGB samples from the chosen colormap
    cmap = cm.get_cmap(cmap_name, n_steps)
    rgb_list = np.linspace(0, 1, n_steps)
    colors = [cmap(x)[:3] for x in rgb_list]  # RGB only (ignore alpha)

    # Convert to ParaView RGBPoints array: [x1, R1, G1, B1, x2, R2, G2, B2, ...]
    vals = np.linspace(vmin, vmax, n_steps)
    RGBPoints = []
    for val, (r, g, b) in zip(vals, colors):
        RGBPoints.extend([val, r, g, b])
    lut.RGBPoints = RGBPoints
    lut.ColorSpace = 'RGB'
    lut.InterpretValuesAsCategories = 0

    # Optional: show scalar bar
    if rep and view:
        rep.SetScalarBarVisibility(view, True)
        sb = GetScalarBar(lut, view)
        sb.Title = array_name
        sb.LabelFormat = "%.2f"
        sb.ScalarBarLength = 0.35
        sb.WindowLocation = "Upper Right Corner"

    print(f"[OK] Applied matplotlib colormap '{cmap_name}' ({n_steps} steps, {vmin}–{vmax}) to {array_name}")
    return lut

def _apply_mach_colormap(rep, array_name, view, vmin, vmax):
    """Make Mach discrete with MACH_DISCRETE_STEPS and (optionally) apply an imported preset."""
    lut = GetColorTransferFunction(array_name)
    # Optional preset import
    if MACH_PRESET_JSON and MACH_PRESET_NAME:
        try:
            ImportPresets(filename=MACH_PRESET_JSON)
            ApplyPreset(lut, MACH_PRESET_NAME, True)
        except Exception as e:
            print(f"[WARN] Could not apply preset '{MACH_PRESET_NAME}': {e}")
    # Discretize
    lut.Discretize = 1
    lut.NumberOfTableValues = int(MACH_DISCRETE_STEPS)
    # Fixed range
    lut.RescaleTransferFunction(float(vmin), float(vmax))
    pwf = GetOpacityTransferFunction(array_name)
    pwf.RescaleTransferFunction(float(vmin), float(vmax))
    # Show colorbar if requested
    if SHOW_COLORBAR:
        rep.SetScalarBarVisibility(view, True)
        sb = GetScalarBar(lut, view)
        sb.Title = "Mach"
        sb.LabelFormat = "%.2f"
        sb.ScalarBarLength = 0.35
        sb.WindowLocation = "Upper Right Corner"

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

# ----------------- build pipeline -----------------
def build_slice(src):
    sl = Slice(registrationName="Slice", Input=src)
    sl.SliceType = "Plane"
    sl.SliceType.Normal = [1.0, 0.0, 0.0]   # x-normal
    sl.SliceType.Origin = [X_START, 0.0, 0.0]
    merged = MergeBlocks(registrationName="Slice_Merged", Input=sl)
    Show(merged)
    return sl, merged

# ------------------- main -------------------
src = open_case(CASE_FILE)
rv = GetActiveViewOrCreate("RenderView")
rv.ViewSize = IMAGE_SIZE
rv.Background = [1,1,1]
rv.InteractionMode = "2D"
Show(src)
ResetCamera()

# Build a slice we will move along x
slice_obj, slice_merged = build_slice(src)
Hide(src)

# --- Camera setup: look along -X with +Y up ---
b = slice_merged.GetDataInformation().GetBounds()   # (xmin,xmax, ymin,ymax, zmin,zmax)
y0 = 0.5*(b[2]+b[3]);  z0 = 0.5*(b[4]+b[5])
rv.CameraParallelProjection = 1          # keep orthographic view
rv.CameraPosition = [-3, 0, 0]          # camera on +X side
rv.CameraFocalPoint = [0, 0.5, 0]          # look toward origin (negative X)
rv.CameraViewUp = [0, 0, 1]              # z is up
rv.CameraParallelScale = 1.8
rv.StillRender()

# Precompute helpers for rho and Mach (on the slice)
rho_proxy, rho_name   = ensure_density(slice_merged)
mach_proxy, mach_name = ensure_mach(slice_merged)
spd_proxy,  spd_name  = ensure_speed(slice_merged)  # for BL Ue & U_norm
walldist_name = find_walldist_name(slice_merged)
if walldist_name:
    print(f"[INFO] Using wall-distance field for δ99: {walldist_name}")
else:
    print("[INFO] No wall-distance field found; will plot δ99 contour only (no numeric thickness).")


# Prepare sweep
x_vals = np.linspace(X_START, X_END, N_SLICES)
if USE_GLOBAL_RANGE:
    rho_min, rho_max = +np.inf, -np.inf
    for x in x_vals:
        slice_obj.SliceType.Origin = [float(x), 0.0, 0.0]
        UpdatePipeline(proxy=slice_merged)
        vmin,vmax = compute_range(rho_proxy, rho_name)
        if np.isfinite(vmin) and np.isfinite(vmax): rho_min, rho_max = min(rho_min,vmin), max(rho_max,vmax)
    if not np.isfinite(rho_min): rho_min, rho_max = 0.0, 1.0
else:
    rho_min=rho_max=None
    
if WRITE_DELTA99_CSV:
    csv_path = os.path.join(OUT_DIR, "delta99_by_slice.csv")
    csvf = open(csv_path, "w", newline=""); csvw = csv.writer(csvf)
    csvw.writerow(["i", "x", "Ue(95pct)", "delta99_mean", "delta99_min", "delta99_max", "n_samples"])

# Sweep and render
for i, x in enumerate(x_vals):
    slice_obj.SliceType.Origin = [float(x), 0.0, 0.0]
    UpdatePipeline(proxy=slice_merged)

    # ----- DENSITY (rainbow) -----
    try:
        px, arr, rep = render_colorfield(rv, rho_proxy, rho_name, title="Density",
                                         fixed_range=(rho_min, rho_max) if USE_GLOBAL_RANGE else None)
        # label
        t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {x:.3f}"
        trep = Show(t); trep.WindowLocation = "Upper Left Corner"; trep.FontSize = 14; trep.Color=[0,0,0]
        rv.StillRender(); save_png(rv, f"x_{i:03d}_rho")
        Hide(t); Delete(t); Hide(px)
    except Exception as e:
        print(f"[WARN] Density at x={x:.4f}: {e}")

    try:
        # Render once (we’ll override LUT to discrete & fixed range)
        px, arr, rep = render_colorfield(rv, mach_proxy, mach_name, title="Mach",
                                        fixed_range=MACH_RANGE)
        _apply_matplotlib_colormap('Mach', cmap_name='rainbow', n_steps=12,
                                  vmin=0.39, vmax=1.3, view=rv, rep=rep)

        t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {x:.3f}"
        trep = Show(t); trep.WindowLocation = "Upper Left Corner"; trep.FontSize = 14; trep.Color=[0,0,0]
        rv.StillRender(); save_png(rv, f"x_{i:03d}_mach")
        Hide(t); Delete(t); Hide(px)
    except Exception as e:
        print(f"[WARN] Mach at x={x:.4f}: {e}")


    # ----- δ99 overlay & annotation -----
    try:
        # 1) robust Ue from the 95th percentile of |U| on the slice (you already had this)
        umag = get_array_numpy(spd_proxy, spd_name)
        if umag is None or umag.size == 0:
            raise RuntimeError("No speed data on slice.")
        Ue = float(np.nanpercentile(umag[np.isfinite(umag)], UE_PERCENTILE))
        if not np.isfinite(Ue) or Ue == 0.0:
            Ue = 1.0  # avoid divide-by-zero

        # 2) U_norm = |U| / Ue (point data)
        calcU = Calculator(Input=spd_proxy)
        calcU.ResultArrayName = "U_norm"
        calcU.Function = f"{spd_name}/{Ue}"
        Show(calcU); Hide(spd_proxy)

        # 3) δ99 contour: U_norm = 0.99
        cont = Contour(Input=calcU)
        cont.ContourBy = ["POINTS", "U_norm"]
        cont.Isosurfaces = [float(DELTA99_LEVEL)]  # 0.99 by default

        # Draw δ99 as a plain line (no scalar colors)
        crep = Show(cont)
        crep.Representation = "Surface"
        ColorBy(crep, None)

        # 4) Color the background by U_norm with fixed range [0, 0.99]
        #    (this makes the BL visualization consistent across slices)
        uview_src, uarr, urep = render_colorfield(
            rv, calcU, "U_norm",
            title="U / Ue",
            fixed_range=(0.0, 0.99)
        )

        # 5) If wall-distance array exists, sample it on δ99 to get thickness stats
        if walldist_name:
            rs = ResampleWithDataset(Source=cont, Input=slice_merged,
                                    PassPointArrays=1, PassCellArrays=0, PassFieldArrays=0)
            rrep = Show(rs); Hide(rrep)
            vtk_rs = Fetch(rs)
            pd = vtk_rs.GetPointData() if vtk_rs else None
            da = pd.GetArray(walldist_name) if pd else None
            if da:
                dvals = ns.vtk_to_numpy(da)
                dvals = dvals[np.isfinite(dvals)]
                if dvals.size > 0 and WRITE_DELTA99_CSV:
                    csvw.writerow([i, x, Ue, float(np.mean(dvals)),
                                float(np.min(dvals)), float(np.max(dvals)), int(dvals.size)])

        # 6) annotate + save
        tt = Text()
        tt.Text = f"U/Ue in [0, 0.99] — δ99 at {DELTA99_LEVEL*100:.1f}%   x={x:.3f}"
        ttrep = Show(tt); ttrep.WindowLocation = "Upper Left Corner"; ttrep.FontSize = 14; ttrep.Color=[0,0,0]
        rv.StillRender()
        save_png(rv, f"x_{i:03d}_UoverUe_delta99")

        # cleanup
        Hide(tt); Delete(tt)
        Hide(cont); Delete(cont)
        Hide(calcU); Delete(calcU)
        Hide(uview_src)

    except Exception as e:
        print(f"[WARN] U/Ue + δ99 at x={x:.4f}: {e}")
        
    try:
        p_px, p_nm = ensure_pressure(slice_merged)  # reuse existing helper (p from energy)
        calc_pn = Calculator(Input=p_px)
        calc_pn.ResultArrayName = "p_over_patm"
        calc_pn.Function = f"{p_nm}/{PATM}"
        Show(calc_pn); Hide(p_px)

        px, arr, rep = render_colorfield(
            rv, calc_pn, "p_over_patm",
            title="p / p_atm",
            fixed_range=PNORM_RANGE  # or None to auto-scale
        )
        t = Text(); t.Text = f"Mach 1.3, AoA 3.0, AoS 0.0, WAT 6.0, x = {x:.3f}"
        trep = Show(t); trep.WindowLocation = "Upper Left Corner"; trep.FontSize = 14; trep.Color=[0,0,0]
        rv.StillRender(); save_png(rv, f"x_{i:03d}_p_over_patm")
        Hide(t); Delete(t); Hide(px); Delete(calc_pn)
    except Exception as e:
        print(f"[WARN] p/p_atm at x={x:.4f}: {e}")

if WRITE_DELTA99_CSV:
    try: csvf.close(); print(f"[OK] Wrote: {csv_path}")
    except Exception: pass

# ----------------- build PDFs -----------------
if WRITE_PDFS:
    try:
        from PIL import Image
        def pngs_to_pdf(pattern, out_pdf):
            frames = sorted(glob.glob(os.path.join(OUT_DIR, pattern)))
            frames = [p for p in frames if os.path.getsize(p) > 0]
            if not frames:
                print(f"[WARN] No frames for {pattern}"); return
            imgs = [Image.open(p).convert("RGB") for p in frames]
            imgs[0].save(os.path.join(OUT_DIR, out_pdf), save_all=True, append_images=imgs[1:])
            print(f"[OK] PDF written: {os.path.join(OUT_DIR, out_pdf)} ({len(imgs)} pages)")
        pngs_to_pdf("x_*_rho.png",     "rho_sweep.pdf")
        pngs_to_pdf("x_*_mach.png",    "mach_sweep.pdf")
        pngs_to_pdf("x_*_p_over_patm.png",    "normalised_pressure_sweep.pdf")
        pngs_to_pdf("x_*_delta99.png", "delta99_sweep.pdf")
    except Exception as e:
        print(f"[WARN] Could not assemble PDFs: {e}")

print("\nDone.")
print(f"Output folder: {OUT_DIR}")

# ========================= AIP Mapping -> PNGs + PDF =========================
# Extracts AIP surface and plots P0 deficit and swirl angle
# Output files:
#   <OUT_DIR>\AIP_P0deficit.png
#   <OUT_DIR>\AIP_Swirl.png
#   <OUT_DIR>\AIP_plots.pdf
# ============================================================================

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
    box.Center = [-4.0, 0.0, 0.0]  # well upstream of X_START=7.3
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
    calc_P0.Function = f"101000 * pow(1 + 0.5*({GAMMA}-1)*1.3*1.3, {GAMMA}/({GAMMA}-1))"
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
        f"{p_arr} * pow(1 + 0.5*({GAMMA}-1)*M_local*M_local, {GAMMA}/({GAMMA}-1))"
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
    P0_inf = compute_freestream_total_pressure(open_case(CASE_FILE))
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
