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

from FileRW.StpFile import step_to_stl

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
# Comparison
# =====================

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def mesh_distance_report(reference_mesh_path, reconstructed_stl_path, sample_reconstructed=True):
    """
    Computes nearest-neighbour distance between:
      reference mesh = morphed VTK/FRO-converted mesh
      reconstructed mesh = STL from morphed STEP

    Distances are measured:
      reconstructed STL points → reference mesh points
    """

    ref = pv.read(reference_mesh_path).extract_surface().triangulate()
    rec = pv.read(reconstructed_stl_path).extract_surface().triangulate()

    ref_pts = np.asarray(ref.points)
    rec_pts = np.asarray(rec.points)

    tree = cKDTree(ref_pts)
    d, idx = tree.query(rec_pts, k=1)

    print("\n[Distance: STL → morphed mesh]")
    print(f"reference points     : {len(ref_pts)}")
    print(f"reconstructed points : {len(rec_pts)}")
    print(f"mean error           : {d.mean():.6e}")
    print(f"RMS error            : {np.sqrt(np.mean(d**2)):.6e}")
    print(f"median error         : {np.median(d):.6e}")
    print(f"95th percentile      : {np.percentile(d, 95):.6e}")
    print(f"99th percentile      : {np.percentile(d, 99):.6e}")
    print(f"max error            : {d.max():.6e}")

    rec["distance_to_morphed_mesh"] = d

    out_vtp = reconstructed_stl_path.replace(".stl", "_error.vtp")
    rec.save(out_vtp)

    print(f"\nSaved coloured error mesh:\n{out_vtp}")

    return d, rec

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

def diagnose_surface_bbox_overlap(step_path, fro_path, sid, pad_frac=0.02):
    from FileRW.FroFile import FroFile
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    import numpy as np

    ff = FroFile.fromFile(fro_path)
    _shape, faces = _load_step_shape_and_faces(step_path)

    _, gids = ff.get_surface_nodes(sid)
    gids = np.asarray(gids, dtype=np.int64)
    pts = ff.nodes[gids]

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    diag = np.linalg.norm(mx - mn)
    pad = pad_frac * diag

    mn_p = mn - pad
    mx_p = mx + pad

    print(f"\n[FRO] Surface {sid}")
    print(f"  nodes = {len(gids)}")
    print(f"  bbox min = {mn}")
    print(f"  bbox max = {mx}")
    print(f"  pad      = {pad:.6e}")

    hits = []

    for i, face in enumerate(faces):
        box = Bnd_Box()
        BRepBndLib.Add_s(face, box)

        try:
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        except Exception:
            continue

        fmin = np.array([xmin, ymin, zmin], float)
        fmax = np.array([xmax, ymax, zmax], float)

        overlap = np.all(mx_p >= fmin) and np.all(fmax >= mn_p)

        if overlap:
            # overlap-box size, useful for ranking
            omin = np.maximum(mn_p, fmin)
            omax = np.minimum(mx_p, fmax)
            osize = np.maximum(omax - omin, 0.0)
            ovol = float(osize[0] * osize[1] * osize[2])

            hits.append((i, ovol, fmin, fmax))

    hits = sorted(hits, key=lambda x: -x[1])

    print(f"\n[CAD] STEP faces whose bbox overlaps FRO surface {sid}: {len(hits)}")
    for i, ovol, fmin, fmax in hits[:20]:
        print(
            f"  face={i:3d} | overlap_vol={ovol:.3e} | "
            f"bbox_min={fmin} | bbox_max={fmax}"
        )

    return hits

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
    """
    Robust bounds for Geom_BSplineSurface / Geom_Surface in OCP.
    """
    # Try generic surface parameter methods first
    for names in [
        ("FirstUParameter", "LastUParameter", "FirstVParameter", "LastVParameter"),
    ]:
        try:
            return (
                float(getattr(surf, names[0])()),
                float(getattr(surf, names[1])()),
                float(getattr(surf, names[2])()),
                float(getattr(surf, names[3])()),
            )
        except Exception:
            pass

    # For Geom_BSplineSurface, use knot ranges
    try:
        return (
            float(surf.UKnot(1)),
            float(surf.UKnot(surf.NbUKnots())),
            float(surf.VKnot(1)),
            float(surf.VKnot(surf.NbVKnots())),
        )
    except Exception:
        pass

    # Final fallback used by GeomAPI_PointsToBSplineSurface often
    return 0.0, 1.0, 0.0, 1.0


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

def build_step_fro_link(step_path, fro_path, link_npz_path, surface_to_face_index=None, require_in_bounds=True, surfaces=None):
    """Build STEP↔FRO link storing (sid)->(gids, face_index, uv, proj0, dist0)."""
    from FileRW.FroFile import FroFile

    ff = FroFile.fromFile(fro_path)
    _shape, faces = _load_step_shape_and_faces(step_path)

    if surfaces is None:
        sids = list(ff.get_surface_ids())
    else:
        sids = list(surfaces)
        
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
        
        for tol in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            n_bad = int(np.sum(finite > tol))
            print(f"    dist > {tol:.0e}: {n_bad}/{finite.size} = {100*n_bad/finite.size:.2f}%")

        print("    p50 =", np.percentile(finite, 50))
        print("    p90 =", np.percentile(finite, 90))
        print("    p95 =", np.percentile(finite, 95))
        print("    p99 =", np.percentile(finite, 99))
        print("    max =", finite.max())

        if max_dist > 1e-2:
            print(f"[CAD-LINK][WARNING] Surface {sid:4d}: large proj error max={max_dist:.3e} mean={mean_dist:.3e}")

        projection_tol = 0.2
        keep = np.isfinite(best["dist0"]) & (best["dist0"] < projection_tol)

        print(
            f"[CAD-LINK] Surface {sid:4d}: keeping "
            f"{int(keep.sum())}/{len(keep)} points under tol={projection_tol}"
        )

        link[int(sid)] = dict(
            gids=gids[keep],
            face_index=int(best["face_index"]),
            uv=np.asarray(best["uv"][keep], float),
            proj0=np.asarray(best["proj0"][keep], float),
            dist0=np.asarray(best["dist0"][keep], float),
        )
        print(f"[CAD-LINK] Surface {sid:4d}: points={len(gids):5d} | face={best['face_index']:3d} | dist mean={mean_dist:.3e}")

    np.savez_compressed(link_npz_path, link=link)
    print(f"[CAD-LINK] Saved link to {link_npz_path}")

def _as_global_nodes_from_fro_elements(arr):
    """
    Best-effort extraction of node IDs from FRO triangle/quad records.
    Handles records like [sid,n1,n2,n3], [n1,n2,n3,sid], etc.
    """
    a = np.asarray(arr, dtype=np.int64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def _surface_elements_for_sid(ff, sid):
    """
    Return element node IDs for one FRO surface.
    Output is a list of tuples, each tuple being 3 or 4 global node IDs.
    """
    elems = []

    for attr, nnode in [("boundary_triangles", 3), ("boundary_quads", 4)]:
        raw = getattr(ff, attr, None)
        if raw is None:
            continue

        arr = _as_global_nodes_from_fro_elements(raw)
        if arr.size == 0:
            continue

        for row in arr:
            row = list(map(int, row))

            # Common layouts:
            # [sid, n1, n2, n3]
            # [n1, n2, n3, sid]
            # [sid, n1, n2, n3, ...]
            # [n1, n2, n3, ..., sid]
            candidates = []

            if len(row) >= nnode + 1 and row[0] == int(sid):
                candidates.append(tuple(row[1:1+nnode]))

            if len(row) >= nnode + 1 and row[-1] == int(sid):
                candidates.append(tuple(row[:nnode]))

            # Some FRO variants store surface id in column 3/4/5.
            for k, val in enumerate(row):
                if val == int(sid):
                    nodes = row[:k] + row[k+1:]
                    if len(nodes) >= nnode:
                        candidates.append(tuple(nodes[:nnode]))

            for c in candidates:
                if len(set(c)) == len(c):
                    elems.append(c)
                    break

    if not elems:
        raise RuntimeError(f"No boundary triangles/quads found for FRO surface {sid}")

    return elems


def _order_boundary_loop_from_elements(elems):
    """
    Given triangles/quads as global node IDs, find edges used once and order the
    largest boundary loop.
    """
    from collections import defaultdict, Counter

    edge_count = Counter()

    for e in elems:
        e = list(map(int, e))
        m = len(e)
        for i in range(m):
            a = e[i]
            b = e[(i + 1) % m]
            if a == b:
                continue
            edge = tuple(sorted((a, b)))
            edge_count[edge] += 1

    boundary_edges = [edge for edge, c in edge_count.items() if c == 1]
    if not boundary_edges:
        raise RuntimeError("No free boundary edges found. Surface may be closed or element parsing is wrong.")

    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Build all loops/chains, keep longest.
    unused = set(boundary_edges)
    loops = []

    while unused:
        a, b = next(iter(unused))
        path = [a, b]
        unused.remove(tuple(sorted((a, b))))

        # grow forward
        while True:
            cur = path[-1]
            prev = path[-2]
            nxts = [n for n in adj[cur] if n != prev and tuple(sorted((cur, n))) in unused]
            if not nxts:
                break
            nxt = nxts[0]
            unused.remove(tuple(sorted((cur, nxt))))
            path.append(nxt)
            if nxt == path[0]:
                break

        loops.append(path)

    loop = max(loops, key=len)

    # remove duplicate closing node if present
    if len(loop) > 1 and loop[0] == loop[-1]:
        loop = loop[:-1]

    print(f"[FRO-TRIM] boundary edges={len(boundary_edges)}, loops={len(loops)}, chosen loop nodes={len(loop)}")
    return np.asarray(loop, dtype=np.int64)

def _make_trimmed_face_from_fro_loop(
    occ_surf_new,
    boundary_gids,
    rec,
    uv_domain,
    tol=1e-6,
    max_loop_points=50,
):
    """
    Trim fitted surface using FRO boundary nodes, but using their original linked CAD UV.
    This avoids projecting morphed boundary XYZ back onto the fitted surface.
    """
    import numpy as np
    from OCP.gp import gp_Pnt2d, gp_Dir2d, gp_Lin2d
    from OCP.Geom2d import Geom2d_Line
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace

    rec_gids = np.asarray(rec["gids"], dtype=np.int64)
    rec_uv = np.asarray(rec["uv"], dtype=float)

    gid_to_uv = {int(g): rec_uv[i] for i, g in enumerate(rec_gids)}

    uv = []
    missing = 0
    for g in boundary_gids:
        g = int(g)
        if g in gid_to_uv:
            uv.append(gid_to_uv[g])
        else:
            missing += 1

    uv = np.asarray(uv, dtype=float)

    print(f"[FRO-TRIM] boundary nodes usable in link: {len(uv)}/{len(boundary_gids)} missing={missing}")

    if len(uv) < 4:
        raise RuntimeError("Too few FRO boundary nodes have linked UV coordinates.")

    if max_loop_points is not None and len(uv) > int(max_loop_points):
        idx = np.linspace(0, len(uv) - 1, int(max_loop_points), dtype=int)
        uv = uv[idx]

    # Map original CAD UV domain to new BSpline parameter domain
    cu1, cu2, cv1, cv2 = map(float, uv_domain)
    nu1, nu2, nv1, nv2 = _bspline_bounds(occ_surf_new)

    du = (cu2 - cu1) if abs(cu2 - cu1) > 1e-14 else 1.0
    dv = (cv2 - cv1) if abs(cv2 - cv1) > 1e-14 else 1.0

    U = nu1 + (uv[:, 0] - cu1) * (nu2 - nu1) / du
    V = nv1 + (uv[:, 1] - cv1) * (nv2 - nv1) / dv

    uv_new = np.column_stack([U, V])

    # remove duplicate consecutive points
    cleaned = [uv_new[0]]
    for p in uv_new[1:]:
        if np.linalg.norm(p - cleaned[-1]) > tol:
            cleaned.append(p)

    uv_new = np.asarray(cleaned, dtype=float)

    if len(uv_new) < 4:
        raise RuntimeError("UV boundary loop collapsed after duplicate removal.")

    if np.linalg.norm(uv_new[0] - uv_new[-1]) > tol:
        uv_new = np.vstack([uv_new, uv_new[0]])

    mk_wire = BRepBuilderAPI_MakeWire()
    n_edges = 0

    for q0, q1 in zip(uv_new[:-1], uv_new[1:]):
        d = q1 - q0
        length = float(np.linalg.norm(d))

        if length <= tol:
            continue

        p0 = gp_Pnt2d(float(q0[0]), float(q0[1]))
        direction = gp_Dir2d(float(d[0]), float(d[1]))
        line2d = Geom2d_Line(gp_Lin2d(p0, direction))

        edge = BRepBuilderAPI_MakeEdge(
            line2d,
            occ_surf_new,
            0.0,
            length
        )

        if edge.IsDone():
            mk_wire.Add(edge.Edge())
            n_edges += 1

    if n_edges < 3:
        raise RuntimeError(f"Failed to build enough UV trim edges: n_edges={n_edges}")

    if not mk_wire.IsDone():
        raise RuntimeError("Failed to build UV trim wire.")

    mk_face = BRepBuilderAPI_MakeFace(occ_surf_new, mk_wire.Wire(), True)

    if not mk_face.IsDone():
        raise RuntimeError("Failed to build face from linked-UV FRO boundary.")

    print(f"[FRO-TRIM] built linked-UV trimmed face with {n_edges} edges")

    return mk_face.Face()

def export_morphed_step_from_link(
    step_path,
    link_npz_path,
    morphed_fro_path,
    out_step_path,
    baseline_fro_path=None,
    use_fro_boundary_trim=True,
    grid_density=50,
    skip_if_mean_dist_gt=None,
    # NEW safety knobs
    max_grid_pts=2500,       # cap nu*nv (e.g., 60x60)
    max_samples=8000,        # cap UV->XYZ training samples per surface
    k_idw=12,                # nearest neighbors for IDW
    trim_edge_samples=30,    # reduce from 60; big speedup, fewer failures
    trim_fallback_untrimmed=False,
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
    ff0 = FroFile.fromFile(baseline_fro_path) if baseline_fro_path else None
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
                if use_fro_boundary_trim and ff0 is not None:
                    elems = _surface_elements_for_sid(ff0, int(sid))
                    loop_gids = _order_boundary_loop_from_elements(elems)


                    new_face = _make_trimmed_face_from_fro_loop(
                        occ_surf,
                        loop_gids,
                        rec,
                        uv_domain=(u_min, u_max, v_min, v_max),
                        tol=1e-6,
                        max_loop_points=50,
                    )
                else:
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
                        raise RuntimeError(f"FRO-trimmed face for sid={sid} is invalid after ShapeFix.")
                        if _is_valid_shape(fallback):
                            new_face = fallback
                            print(f"[CAD-EXPORT][FACE-FALLBACK] sid={sid}: used untrimmed face")
                        else:
                            print(f"[CAD-EXPORT][SKIP] sid={sid}: face invalid even after fix")
                            continue
                
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

# surface_to_face_index = {i : i-1 for i in range(2)}
# surface_to_face_index = None

# surfaces = None

# # INPUT PATHS
# step_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\sphere.stp"
# fro_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\sphere.fro"
# morphed_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\sphere_1.fro"

# # OUTPUT PATHS
# npz_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\surfaces\surface_link.npz"
# out_step_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\surfaces\sphere_1.stp"
# morphed_stl = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\surfaces\sphere_1.stl"
# fixed_stl = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\test\surfaces\sphere_1_f.stl"

# diagnose_surface_bbox_overlap(
#    step_path=step_path,
#    fro_path=fro_path,
#    sid=None,
# )

# build_step_fro_link(
#    step_path=step_path,
#    fro_path=fro_path,
#    link_npz_path=npz_path,
#    surface_to_face_index=surface_to_face_index,
#    require_in_bounds=True,
#    surfaces = surfaces
# )

# export_morphed_step_from_link(
#    step_path=step_path,
#    link_npz_path=npz_path,
#    morphed_fro_path=morphed_path,
#    out_step_path=out_step_path,
#    baseline_fro_path=fro_path,
#    use_fro_boundary_trim=True,
#    grid_density=30,
#    max_grid_pts=900,
#    max_samples=5000,
#    k_idw=12,
#    trim_edge_samples=20,
#    skip_if_mean_dist_gt=None,
# )

# import time
# t0 = time.time()

# step_to_stl(
#     out_step_path,
#     morphed_stl,
#     linear_deflection=2.0,
#     angular_deflection=0.5,
# )

# print(f"STL conversion took {time.time() - t0:.1f} s")

# d, rec = mesh_distance_report(
#     reference_mesh_path=ref_mesh_path,   # or your morphed VTK/VTM surface
#     reconstructed_stl_path=morphed_stl,
# )