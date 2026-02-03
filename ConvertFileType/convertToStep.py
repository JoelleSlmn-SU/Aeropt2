"""
convertToStep.py — STEP↔FRO linking + morphed STEP export (robust trimming)

What this version fixes vs your current state:
- Trimming reconstruction no longer assumes the NEW BSpline surface shares the original CAD (u,v) domain.
  We map original CAD (u,v) -> NEW surface parameter domain before evaluating occ_surf_new.Value().
- Trimming is *best-effort*: if an outer/inner wire can't be rebuilt, we fall back to an untrimmed face
  (so the export completes instead of crashing).
- Per-surface export is guarded: a single problematic surface won't stop the whole STEP file.
- Additional diagnostics for large projection errors and trim failures.

Notes:
- This produces a morphed STEP as a *compound of faces* (surface set). Making a watertight solid
  requires sewing/healing and is a separate step.
"""

import os, sys
import numpy as np

# If your project uses FileRW as a package, prefer proper imports.
# Keep this for your current Aeropt2 layout.
sys.path.append(os.path.dirname("FileRW"))

# =====================
# Robust OCP helpers
# =====================

def _is_valid_shape(shape) -> bool:
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        return bool(BRepCheck_Analyzer(shape, True).IsValid())
    except Exception:
        return True  # if checker unavailable, don't block export


def _fix_face(face):
    """
    Try to repair a face so it becomes STEP-writable.
    Returns (fixed_face, was_fixed_flag).
    """
    try:
        from OCP.ShapeFix import ShapeFix_Face
        from OCP.ShapeAnalysis import ShapeAnalysis_WireOrder
        from OCP.BRepLib import BRepLib

        # build missing 3D curves from pcurves, etc.
        try:
            BRepLib.BuildCurves3d_s(face)
        except Exception:
            pass

        sff = ShapeFix_Face(face)
        sff.SetPrecision(1e-6)
        sff.SetMaxTolerance(1e-3)
        sff.FixOrientation()
        sff.FixAddNaturalBound()
        sff.FixMissingSeam()
        sff.FixWireTool().FixReorder(True)
        sff.Perform()
        fixed = sff.Face()
        return fixed, True
    except Exception:
        return face, False


def _fix_shape(shape):
    """
    Broader repair; sometimes helps compounds.
    """
    try:
        from OCP.ShapeFix import ShapeFix_Shape
        sfs = ShapeFix_Shape(shape)
        sfs.SetPrecision(1e-6)
        sfs.SetMaxTolerance(1e-3)
        sfs.Perform()
        return sfs.Shape(), True
    except Exception:
        return shape, False


def _as_face(shape_obj):
    from OCP.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS
    if isinstance(shape_obj, TopoDS_Face):
        return shape_obj
    if not isinstance(shape_obj, TopoDS_Shape):
        return None
    try:
        face = TopoDS.Face_s(shape_obj)
        if hasattr(face, "IsNull") and face.IsNull():
            return None
        return face
    except Exception:
        return None



def _as_wire(shape_obj):
    """Best-effort cast to TopoDS_Wire (handles cases where explorer returns TopoDS_Shape)."""
    from OCP.TopoDS import TopoDS_Shape, TopoDS_Wire, TopoDS
    if isinstance(shape_obj, TopoDS_Wire):
        return shape_obj
    if not isinstance(shape_obj, TopoDS_Shape):
        return None
    try:
        w = TopoDS.Wire_s(shape_obj)
        if hasattr(w, "IsNull") and w.IsNull():
            return None
        return w
    except Exception:
        return None

def _uv_bounds(face):
    from OCP.BRepTools import BRepTools
    if hasattr(BRepTools, "UVBounds_s"):
        return BRepTools.UVBounds_s(face)
    return BRepTools.UVBounds(face)

def _outer_wire(face):
    from OCP.BRepTools import BRepTools
    if hasattr(BRepTools, "OuterWire_s"):
        return BRepTools.OuterWire_s(face)
    return BRepTools.OuterWire(face)

def _curve_on_surface(edge, face):
    """Return (Geom2d_Curve, first, last) for trimming p-curve on face."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    c2d_ad = BRepAdaptor_Curve2d(edge, face)
    first = float(c2d_ad.FirstParameter())
    last  = float(c2d_ad.LastParameter())
    c2d = c2d_ad.Curve()
    if hasattr(c2d, "GetObject"):
        c2d = c2d.GetObject()
    return c2d, first, last

def _surf_bounds(occ_surf):
    """Return (u0,u1,v0,v1) bounds for a Geom_Surface-like object."""
    try:
        return (float(occ_surf.FirstUParameter()),
                float(occ_surf.LastUParameter()),
                float(occ_surf.FirstVParameter()),
                float(occ_surf.LastVParameter()))
    except Exception:
        return (0.0, 1.0, 0.0, 1.0)

def _map_uv_to_surf_params(u, v, u_min, u_max, v_min, v_max, occ_surf):
    """Map original (u,v) into the parameter bounds of occ_surf (typically normalised)."""
    su0, su1, sv0, sv1 = _surf_bounds(occ_surf)
    du = (u_max - u_min) if (u_max - u_min) != 0.0 else 1.0
    dv = (v_max - v_min) if (v_max - v_min) != 0.0 else 1.0
    uu = (u - u_min) / du
    vv = (v - v_min) / dv
    uu = min(1.0, max(0.0, float(uu)))
    vv = min(1.0, max(0.0, float(vv)))
    U = su0 + uu * (su1 - su0)
    V = sv0 + vv * (sv1 - sv0)
    return U, V

# =====================
# STEP loading
# =====================

def _load_step_shape_and_faces(step_path):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = _as_face(exp.Current())
        if f is not None:
            faces.append(f)
        exp.Next()
    return shape, faces

# =====================
# Projection utilities
# =====================

def _project_points_to_face_uv(face, points_xyz, require_in_bounds=True):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.gp import gp_Pnt

    surf_ad = BRepAdaptor_Surface(face, True)
    geom_surf = surf_ad.Surface().Surface()

    umin, umax, vmin, vmax = _uv_bounds(face)

    uv = np.zeros((len(points_xyz), 2), float)
    proj = np.zeros((len(points_xyz), 3), float)
    dist = np.zeros((len(points_xyz),), float)
    ok = np.ones((len(points_xyz),), dtype=bool)

    for i, p in enumerate(points_xyz):
        P = gp_Pnt(float(p[0]), float(p[1]), float(p[2]))
        pr = GeomAPI_ProjectPointOnSurf(P, geom_surf)
        if pr.NbPoints() < 1:
            ok[i] = False
            dist[i] = np.inf
            continue
        u, v = pr.LowerDistanceParameters()
        if require_in_bounds:
            if not (umin <= u <= umax and vmin <= v <= vmax):
                ok[i] = False
        Q = pr.NearestPoint()
        uv[i] = (u, v)
        proj[i] = (Q.X(), Q.Y(), Q.Z())
        dist[i] = pr.LowerDistance()
    return uv, proj, dist, ok

# =====================
# Interpolation helpers
# =====================

def _interpolate_scattered_points(uv_samples, xyz_samples, u_grid, v_grid):
    """Interpolate scattered (u,v)->xyz onto a regular grid."""
    from scipy.interpolate import Rbf

    u_samples = uv_samples[:, 0]
    v_samples = uv_samples[:, 1]

    x_samples = xyz_samples[:, 0]
    y_samples = xyz_samples[:, 1]
    z_samples = xyz_samples[:, 2]

    try:
        rbf_x = Rbf(u_samples, v_samples, x_samples, function='thin_plate', smooth=0.001)
        rbf_y = Rbf(u_samples, v_samples, y_samples, function='thin_plate', smooth=0.001)
        rbf_z = Rbf(u_samples, v_samples, z_samples, function='thin_plate', smooth=0.001)

        xg = rbf_x(u_grid, v_grid)
        yg = rbf_y(u_grid, v_grid)
        zg = rbf_z(u_grid, v_grid)
        return np.stack([xg, yg, zg], axis=-1)
    except Exception as e:
        print(f"  [interp] RBF failed ({e}); falling back to linear griddata")
        from scipy.interpolate import griddata
        pts = uv_samples
        grid_pts = np.column_stack([u_grid.ravel(), v_grid.ravel()])
        xg = griddata(pts, x_samples, grid_pts, method='linear').reshape(u_grid.shape)
        yg = griddata(pts, y_samples, grid_pts, method='linear').reshape(u_grid.shape)
        zg = griddata(pts, z_samples, grid_pts, method='linear').reshape(u_grid.shape)
        return np.stack([xg, yg, zg], axis=-1)

# =====================
# Trimming reconstruction
# =====================

def _bspline_bounds(surf):
    # Geom_BSplineSurface provides bounds via Bounds(...)
    u1 = float(surf.FirstUParameter())
    u2 = float(surf.LastUParameter())
    v1 = float(surf.FirstVParameter())
    v2 = float(surf.LastVParameter())
    return u1, u2, v1, v2


def _map_uv_linear(u, v, cad_uv_bounds, new_uv_bounds):
    cu1, cu2, cv1, cv2 = cad_uv_bounds
    nu1, nu2, nv1, nv2 = new_uv_bounds

    # protect against degenerate bounds
    du = (cu2 - cu1) if abs(cu2 - cu1) > 1e-14 else 1.0
    dv = (cv2 - cv1) if abs(cv2 - cv1) > 1e-14 else 1.0

    U = nu1 + (u - cu1) * (nu2 - nu1) / du
    V = nv1 + (v - cv1) * (nv2 - nv1) / dv
    return float(U), float(V)

def _fit_bspline_curve_through_points(pts_xyz):
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.gp import gp_Pnt
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    arr = TColgp_Array1OfPnt(1, len(pts_xyz))
    for i, p in enumerate(pts_xyz, start=1):
        arr.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    return GeomAPI_PointsToBSpline(arr).Curve()

def _build_trimmed_face_from_original_uv(face_orig, occ_surf_new, uv_domain, n_edge_samples=60, tol=1e-6):
    """
    Best-effort trimming reconstruction.

    IMPORTANT:
      occ_surf_new (made from a point grid) does NOT preserve the original CAD (u,v) domain.
      Therefore, we map original p-curve (u,v) values into the new surface parameter domain using uv_domain.

    uv_domain = (u_min,u_max,v_min,v_max) used when building the interpolation grid.
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace,
    )
    from OCP.gp import gp_Pnt

    u_min, u_max, v_min, v_max = map(float, uv_domain)
    outer = _as_wire(_outer_wire(face_orig))
    if outer is None:
        # No recoverable outer wire; return bounded untrimmed face
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        nu1, nu2, nv1, nv2 = _bspline_bounds(occ_surf_new)
        return BRepBuilderAPI_MakeFace(occ_surf_new, nu1, nu2, nv1, nv2, tol).Face()


    def rebuild_wire(wire):
        wexp = BRepTools_WireExplorer(wire, face_orig)
        mk_wire = BRepBuilderAPI_MakeWire()

        first_start = None
        prev_end = None
        built = 0

        while wexp.More():
            edge = wexp.Current()

            try:
                c2d, first, last = _curve_on_surface(edge, face_orig)
                ts = np.linspace(float(first), float(last), int(n_edge_samples))
            except Exception as ex:
                print(f"    [trim] no p-curve: {ex}")
                return None

            pts = []
            try:
                for t in ts:
                    p2d = c2d.Value(float(t))
                    u, v = float(p2d.X()), float(p2d.Y())
                    cad_bounds = (u_min, u_max, v_min, v_max)
                    new_bounds = _bspline_bounds(occ_surf_new)
                    U, V = _map_uv_linear(u, v, cad_bounds, new_bounds)
                    P = occ_surf_new.Value(U, V)
                    pts.append((P.X(), P.Y(), P.Z()))
            except Exception as ex:
                print(f"    [trim] surf.Value failed: {ex}")
                return None

            if len(pts) < 2:
                return None

            p_start = np.asarray(pts[0], float)
            p_end   = np.asarray(pts[-1], float)

            # chain continuity
            if prev_end is not None:
                p_start = prev_end
            if first_start is None:
                first_start = p_start.copy()

            crv3d = _fit_bspline_curve_through_points(pts)

            E = BRepBuilderAPI_MakeEdge(
                crv3d,
                gp_Pnt(float(p_start[0]), float(p_start[1]), float(p_start[2])),
                gp_Pnt(float(p_end[0]),   float(p_end[1]),   float(p_end[2])),
            )
            if not E.IsDone():
                return None

            mk_wire.Add(E.Edge())
            built += 1
            prev_end = p_end.copy()
            wexp.Next()

        if built == 0:
            return None

        # close loop if needed
        if first_start is not None and prev_end is not None:
            gap = float(np.linalg.norm(prev_end - first_start))
            if gap > 1e-6:
                close_edge = BRepBuilderAPI_MakeEdge(
                    gp_Pnt(float(prev_end[0]),    float(prev_end[1]),    float(prev_end[2])),
                    gp_Pnt(float(first_start[0]), float(first_start[1]), float(first_start[2])),
                )
                if close_edge.IsDone():
                    mk_wire.Add(close_edge.Edge())

        if not mk_wire.IsDone():
            return None

        try:
            return mk_wire.Wire()
        except Exception:
            return None

    outer_new = rebuild_wire(outer)
    if outer_new is None:
        # fallback: untrimmed face
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        return BRepBuilderAPI_MakeFace(occ_surf_new, *_bspline_bounds(occ_surf_new), tol).Face()

    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    mk_face = BRepBuilderAPI_MakeFace(occ_surf_new, outer_new, True)

    # inner wires
    expw = TopExp_Explorer(face_orig, TopAbs_WIRE)
    while expw.More():
        w = _as_wire(expw.Current())
        if w is None:
            expw.Next()
            continue
        try:
            if w.IsSame(outer):
                expw.Next()
                continue
        except Exception:
            pass

        inner_new = rebuild_wire(w)
        if inner_new is not None:
            mk_face.Add(inner_new)
        expw.Next()

    if not mk_face.IsDone():
        return BRepBuilderAPI_MakeFace(occ_surf_new, *_bspline_bounds(occ_surf_new), tol).Face()

    return mk_face.Face()

# =====================
# Public API
# =====================

def build_step_fro_link(step_path, fro_path, link_npz_path, surface_to_face_index=None, require_in_bounds=True):
    """Build STEP↔FRO link storing (sid)->(gids, face_index, uv, proj0, dist0)."""
    from FileRW.FroFile import FroFile

    ff = FroFile.fromFile(fro_path)
    _shape, faces = _load_step_shape_and_faces(step_path)

    sids = list(ff.get_surface_ids())
    print(f"[link] FRO surfaces: {len(sids)}  |  STEP faces: {len(faces)}")

    link = {}
    for sid in sids:
        _, gids = ff.get_surface_nodes(sid)
        gids = np.asarray(gids, dtype=np.int64)
        xyz = ff.nodes[gids]

        if surface_to_face_index and sid in surface_to_face_index:
            cand = [int(surface_to_face_index[sid])]
        else:
            cand = list(range(len(faces)))

        best = None
        for fi in cand:
            uv, proj, dist, ok = _project_points_to_face_uv(faces[fi], xyz, require_in_bounds=require_in_bounds)
            if not ok.any():
                continue
            score = float(dist[ok].mean())
            if best is None or score < best["score"]:
                best = dict(face_index=int(fi), uv=uv, proj0=proj, dist0=dist, score=score)

        if best is None:
            raise RuntimeError(f"No suitable CAD face for surface {sid}")

        mean_dist = float(best["score"])
        finite = best["dist0"][np.isfinite(best["dist0"])]
        max_dist = float(finite.max()) if finite.size else float("inf")

        if max_dist > 1e-2:
            print(f"[CAD-LINK][WARNING] Surface {sid:4d}: large proj error max={max_dist:.3e} mean={mean_dist:.3e}")

        link[int(sid)] = dict(
            gids=gids,
            face_index=int(best["face_index"]),
            uv=np.asarray(best["uv"], float),
            proj0=np.asarray(best["proj0"], float),
            dist0=np.asarray(best["dist0"], float),
        )
        print(f"[CAD-LINK] Surface {sid:4d}: points={len(gids):5d} | face={best['face_index']:3d} | dist mean={mean_dist:.3e}")

    np.savez_compressed(link_npz_path, link=link)
    print(f"[CAD-LINK] Saved link to {link_npz_path}")

def export_morphed_step_from_link(
    step_path,
    link_npz_path,
    morphed_fro_path,
    out_step_path,
    grid_density=30,
    skip_if_mean_dist_gt=None,
    # NEW safety knobs
    max_grid_pts=3600,       # cap nu*nv (e.g., 60x60)
    max_samples=8000,        # cap UV->XYZ training samples per surface
    k_idw=12,                # nearest neighbors for IDW
    trim_edge_samples=30,    # reduce from 60; big speedup, fewer failures
    trim_fallback_untrimmed=True,
):
    """
    Export morphed STEP as a compound of faces, with bounded memory/time per surface.

    Key changes vs your current version:
    - Caps grid size and training samples per surface
    - Uses local IDW interpolation in (u,v) via KDTree (no global RBF / dense solve)
    - Fills the OCC point grid on-the-fly (no xyz_grid allocation)
    - Trimming is best-effort; falls back to untrimmed face if wire rebuild fails
    """
    import numpy as np
    import gc
    import os
    import time
    import tempfile

    from FileRW.FroFile import FroFile
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.Interface import Interface_Static

    # --- fast local interpolator (IDW) ---
    try:
        from scipy.spatial import cKDTree as _KDTree
    except Exception:
        _KDTree = None

    def _downsample_uv_xyz(uv, xyz, max_n):
        n = uv.shape[0]
        if n <= max_n:
            return uv, xyz
        idx = np.random.choice(n, size=max_n, replace=False)
        return uv[idx], xyz[idx]

    def _build_idw_predictor(uv_train, xyz_train, k=12):
        """
        Returns callable f(u,v)->(x,y,z) using k-NN IDW.
        """
        uv_train = np.asarray(uv_train, float)
        xyz_train = np.asarray(xyz_train, float)

        if _KDTree is not None:
            tree = _KDTree(uv_train)

            def predict(u, v, eps=1e-12):
                d, ii = tree.query([u, v], k=min(k, uv_train.shape[0]))
                d = np.atleast_1d(d)
                ii = np.atleast_1d(ii)
                # exact hit
                if d.size > 0 and d.min() < 1e-14:
                    return xyz_train[ii[d.argmin()]]
                w = 1.0 / (d + eps)
                w /= w.sum()
                return (xyz_train[ii] * w[:, None]).sum(axis=0)

            return predict

        # fallback (brute) – only safe if heavily downsampled
        def predict(u, v, eps=1e-12):
            du = uv_train[:, 0] - u
            dv = uv_train[:, 1] - v
            d2 = du * du + dv * dv
            j = int(np.argmin(d2))
            if d2[j] < 1e-28:
                return xyz_train[j]
            # take k smallest
            kk = min(k, uv_train.shape[0])
            ii = np.argpartition(d2, kk - 1)[:kk]
            d = np.sqrt(d2[ii])
            w = 1.0 / (d + eps)
            w /= w.sum()
            return (xyz_train[ii] * w[:, None]).sum(axis=0)

        return predict

    # --- load inputs ---
    ffm = FroFile.fromFile(morphed_fro_path)
    _shape, faces = _load_step_shape_and_faces(step_path)
    data = np.load(link_npz_path, allow_pickle=True)["link"].item()

    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    added = 0
    print(f"[CAD-EXPORT] Surfaces in link: {len(data)}")

    # precompute target grid size cap (square-ish)
    cap_side = int(np.sqrt(int(max_grid_pts)))
    cap_side = max(cap_side, 8)

    for sid, rec in data.items():
        try:
            face = faces[int(rec["face_index"])]
            uv = np.asarray(rec["uv"], float)
            gids = np.asarray(rec["gids"], dtype=np.int64)
            Pm = np.asarray(ffm.nodes[gids], float)

            # ----- skip bad links (prevents pathological surfaces) -----
            if skip_if_mean_dist_gt is not None:
                dist0 = np.asarray(rec.get("dist0", []), float)
                finite = dist0[np.isfinite(dist0)]
                mean_dist = float(finite.mean()) if finite.size else float("inf")
                if mean_dist > float(skip_if_mean_dist_gt):
                    print(f"[CAD-EXPORT][SKIP] sid={sid}: mean link dist {mean_dist:.3e} > {skip_if_mean_dist_gt}")
                    continue

            n_points = uv.shape[0]
            if n_points < 10:
                print(f"[CAD-EXPORT][SKIP] sid={sid}: too few points ({n_points})")
                continue

                        # ----- build displacement field in XYZ (projected CAD surface -> morphed mesh) -----
            proj0 = np.asarray(rec.get("proj0", None), float)
            if proj0 is None or proj0.size == 0:
                # fall back to using mesh points directly as sources
                proj0 = np.asarray(ffm.nodes[gids], float)

            disp0 = np.asarray(Pm, float) - np.asarray(proj0, float)

            # Downsample sources/disp (KDTree build + queries stay bounded)
            if proj0.shape[0] > max_samples:
                idx = np.random.choice(proj0.shape[0], size=int(max_samples), replace=False)
                src = proj0[idx]
                val = disp0[idx]
            else:
                src = proj0
                val = disp0

            # If KDTree unavailable, downsample more aggressively (brute fallback)
            if _KDTree is None and src.shape[0] > 2000:
                idx = np.random.choice(src.shape[0], size=2000, replace=False)
                src = src[idx]
                val = val[idx]

            # Build k-NN IDW in 3D
            if _KDTree is not None:
                tree3 = _KDTree(src)

                def disp_predict(P, eps=1e-12):
                    d, ii = tree3.query(P, k=min(k_idw, src.shape[0]))
                    d = np.atleast_1d(d); ii = np.atleast_1d(ii)
                    if d.size and d.min() < 1e-14:
                        return val[ii[d.argmin()]]
                    w = 1.0 / (d + eps)
                    w /= w.sum()
                    return (val[ii] * w[:, None]).sum(axis=0)
            else:
                def disp_predict(P, eps=1e-12):
                    d2 = np.sum((src - P[None, :])**2, axis=1)
                    j = int(np.argmin(d2))
                    if d2[j] < 1e-28:
                        return val[j]
                    kk = min(k_idw, src.shape[0])
                    ii = np.argpartition(d2, kk - 1)[:kk]
                    d = np.sqrt(d2[ii])
                    w = 1.0 / (d + eps)
                    w /= w.sum()
                    return (val[ii] * w[:, None]).sum(axis=0)

            # ----- choose UV grid from linked UV range clamped to face UV bounds -----
            try:
                from OCP.BRepTools import BRepTools
                # Different OCP builds expose UVBounds_s / UVBounds
                if hasattr(BRepTools, "UVBounds_s"):
                    uFmin, uFmax, vFmin, vFmax = BRepTools.UVBounds_s(face)
                else:
                    uFmin, uFmax, vFmin, vFmax = BRepTools.UVBounds(face)
                uFmin, uFmax, vFmin, vFmax = float(uFmin), float(uFmax), float(vFmin), float(vFmax)
            except Exception:
                # fallback: use linked uv span
                uFmin, uFmax = float(np.nanmin(uv[:, 0])), float(np.nanmax(uv[:, 0]))
                vFmin, vFmax = float(np.nanmin(uv[:, 1])), float(np.nanmax(uv[:, 1]))

            u_min, u_max = float(np.nanmin(uv[:, 0])), float(np.nanmax(uv[:, 0]))
            v_min, v_max = float(np.nanmin(uv[:, 1])), float(np.nanmax(uv[:, 1]))

            # expand slightly, then clamp
            du = (u_max - u_min) if abs(u_max - u_min) > 1e-14 else 1.0
            dv = (v_max - v_min) if abs(v_max - v_min) > 1e-14 else 1.0
            pad_u = 0.02 * du
            pad_v = 0.02 * dv
            u_min = max(uFmin, u_min - pad_u)
            u_max = min(uFmax, u_max + pad_u)
            v_min = max(vFmin, v_min - pad_v)
            v_max = min(vFmax, v_max + pad_v)

            # precompute target grid size cap (square-ish)
            cap_side = int(np.sqrt(int(max_grid_pts)))
            cap_side = max(cap_side, 8)

            nu = min(max(int(grid_density), 8), cap_side)
            nv = min(max(int(grid_density), 8), cap_side)
            if n_points < 400:
                nu = min(nu, 20); nv = min(nv, 20)

            u_vals = np.linspace(u_min, u_max, nu)
            v_vals = np.linspace(v_min, v_max, nv)

            # ----- sample original CAD surface then apply displacement field -----
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            surf_ad = BRepAdaptor_Surface(face, True)

            arr = TColgp_Array2OfPnt(1, nu, 1, nv)

            # fallback point if anything goes wrong
            fallback = np.asarray(Pm[0], float)

            for i in range(nu):
                ui = float(u_vals[i])
                for j in range(nv):
                    vj = float(v_vals[j])
                    P0 = surf_ad.Value(ui, vj)
                    p0 = np.array([P0.X(), P0.Y(), P0.Z()], float)
                    dP = disp_predict(p0)
                    p = p0 + dP
                    if not np.isfinite(p).all():
                        p = fallback
                    arr.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

            occ_surf = GeomAPI_PointsToBSplineSurface(arr, 3, 8).Surface()

            # ----- trimming (best-effort) -----
            try:
                new_face = _build_trimmed_face_from_original_uv(
                    face,
                    occ_surf,
                    (u_min, u_max, v_min, v_max),
                    n_edge_samples=int(trim_edge_samples),
                )
                if not _is_valid_shape(new_face):
                    fixed_face, _ = _fix_face(new_face)
                    if _is_valid_shape(fixed_face):
                        new_face = fixed_face
                    else:
                        # fallback: untrimmed (often valid), else skip
                        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
                        fallback = BRepBuilderAPI_MakeFace(occ_surf, 1e-6).Face()
                        if _is_valid_shape(fallback):
                            new_face = fallback
                            print(f"[CAD-EXPORT][FACE-FALLBACK] sid={sid}: used untrimmed face")
                        else:
                            print(f"[CAD-EXPORT][SKIP] sid={sid}: face invalid even after fix")
                            continue

                builder.Add(comp, new_face)
                
            except Exception as tex:
                if trim_fallback_untrimmed:
                    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
                    print(f"[CAD-EXPORT][TRIM-FALLBACK] sid={sid}: {tex}")
                    new_face = BRepBuilderAPI_MakeFace(occ_surf, 1e-6).Face()
                else:
                    raise

            builder.Add(comp, new_face)
            added += 1
            print(f"[CAD-EXPORT] sid={sid}: OK (face={rec['face_index']}, npts={n_points}, grid={nu}x{nv}, train={src.shape[0]})")

            # ----- force cleanup to prevent creeping RAM -----
            try:
                del arr, occ_surf, new_face
            except Exception:
                pass
            try:
                del src, val, surf_ad, tree3
            except Exception:
                pass
            gc.collect()

        except Exception as ex:
            print(f"[CAD-EXPORT][WARNING] sid={sid} failed: {ex}")
            # try to free anything big from this iteration
            gc.collect()
            continue

    if added == 0:
        raise RuntimeError("No faces were added to the compound; refusing to write an empty STEP.")

    # --- Optional: validity check (cheap) ---
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        ok = BRepCheck_Analyzer(comp, True).IsValid()
        print(f"[CAD-EXPORT] Compound validity: {ok}")
    except Exception as e:
        print(f"[CAD-EXPORT] Validity check skipped: {e}")

    # --- Optional: sew faces into a shell (often helps STEP writer a lot) ---
    # If sewing is too slow, comment this block out.
    comp_to_write = comp
    if _is_valid_shape(comp):
        try:
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
            sew = BRepBuilderAPI_Sewing(1e-4)
            sew.Add(comp)
            sew.Perform()
            sewn = sew.SewedShape()
            if sewn is not None and not sewn.IsNull() and _is_valid_shape(sewn):
                comp_to_write = sewn
                print("[CAD-EXPORT] Sewing: used sewed shape")
            else:
                print("[CAD-EXPORT] Sewing: result invalid; writing compound")
        except Exception as e:
            print(f"[CAD-EXPORT] Sewing skipped: {e}")
    else:
        print("[CAD-EXPORT] Sewing skipped (compound invalid)")

    # --- IMPORTANT: write to local temp FIRST (avoid OneDrive locks/thrash) ---
    out_step_path = os.fspath(out_step_path)
    out_dir = os.path.dirname(out_step_path)
    os.makedirs(out_dir, exist_ok=True)

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"morphed_{int(time.time())}.step")

    print(f"[CAD-EXPORT] Writing STEP with {added} faces -> TEMP: {tmp_path}")

    writer = STEPControl_Writer()
    from OCP.Interface import Interface_Static
    Interface_Static.SetIVal_s("write.step.nonmanifold", 1)   # allow non-manifold shells
    Interface_Static.SetCVal_s("write.step.schema", "AP203")

    # NOTE: Transfer may succeed even if Write later struggles, so keep them separate
    writer.Transfer(comp_to_write, STEPControl_AsIs)
    status = writer.Write(tmp_path)

    # OCP usually returns 1 on success, but be defensive:
    if status != 1 or (not os.path.exists(tmp_path)) or os.path.getsize(tmp_path) < 1024:
        raise RuntimeError(f"STEP export failed or produced empty file. status={status}, tmp_exists={os.path.exists(tmp_path)}")

    # move into final location
    print(f"[CAD-EXPORT] Moving TEMP -> {out_step_path}")
    try:
        os.replace(tmp_path, out_step_path)
    except PermissionError:
        # On Windows/OneDrive sometimes replace fails; fall back to copy+remove
        import shutil
        shutil.copy2(tmp_path, out_step_path)
        os.remove(tmp_path)

    print(f"[CAD-EXPORT] ✓ Wrote STEP: {out_step_path} ({os.path.getsize(out_step_path)/1e6:.2f} MB)")

# =====================

surface_to_face_index = {i : i-1 for i in range(1, 129)}

# Step 1: Build the link (do this once)
#build_step_fro_link(
#    step_path=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\corner.stp",
#    fro_path=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\corner.fro",
#    link_npz_path=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\orig_step_mesh_link.npz",
#    surface_to_face_index=surface_to_face_index,
#)

# Step 2: Export morphed geometry
#export_morphed_step_from_link(
#    step_path=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\corner.stp",
#    link_npz_path=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\orig_step_mesh_link.npz",
#    morphed_fro_path=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\corner_1.fro",
#    out_step_path=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\corner_1.step",
#    max_grid_pts=2500,
#    max_samples=5000,
#    k_idw=12,
#    trim_edge_samples=20,
#)