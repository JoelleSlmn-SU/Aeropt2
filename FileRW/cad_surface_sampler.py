import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------- OCC / OCP imports with fallback ----------

try:
    # cadquery-ocp style
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    import OCP.TopoDS as TDS
    _BACKEND = "OCP"
except Exception:
    # pythonocc-core style
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.IGESControl import IGESControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    import OCC.Core.TopoDS as TDS
    _BACKEND = "OCC"


def _as_face(shape_obj):
    """
    Try very hard to turn a TopoDS_Shape into a TopoDS_Face.

    Strategy:
    - Look for a callable in TDS whose name contains 'topods_face' (OCP helper)
    - Fallback to pythonOCC topods_Face / TopoDS_Face.DownCast if available
    - If nothing works, return None and let caller skip this face
    """
    # 0. Already some kind of Face subclass?
    try:
        from TDS import TopoDS_Face  # this may or may not exist
        if isinstance(shape_obj, TopoDS_Face):
            return shape_obj
    except Exception:
        pass

    # 1. Try to find a helper that looks like topods_Face in this module
    for name in dir(TDS):
        if "topods_face" in name.lower():
            helper = getattr(TDS, name)
            if callable(helper):
                try:
                    face = helper(shape_obj)
                    # Check for IsNull if present
                    if hasattr(face, "IsNull") and face.IsNull():
                        continue
                    return face
                except Exception:
                    continue

    # 2. pythonOCC-style helpers (if that backend is present)
    try:
        from OCC.Core.TopoDS import topods_Face as _topods_Face  # type: ignore
        face = _topods_Face(shape_obj)
        if hasattr(face, "IsNull") and face.IsNull():
            face = None
        if face is not None:
            return face
    except Exception:
        pass

    try:
        from OCC.Core.TopoDS import TopoDS_Face as _TF  # type: ignore[attr-defined]
        face = _TF.DownCast(shape_obj)
        if hasattr(face, "IsNull") and face.IsNull():
            face = None
        if face is not None:
            return face
    except Exception:
        pass

    # 3. Give up
    print(f"[WARN] _as_face: could not cast object of type {type(shape_obj)} to Face")
    return None


def load_cad_shape(path: str):
    """
    Load a STEP or IGES file and return a TopoDS_Shape.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in (".stp", ".step", ".igs", ".iges"):
        raise ValueError(f"Unsupported CAD extension: {ext}")

    if ext in (".stp", ".step"):
        reader = STEPControl_Reader()
        status = reader.ReadFile(p.as_posix())
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP read failed for {p}")
        reader.TransferRoots()
        shape = reader.OneShape()
    else:
        reader = IGESControl_Reader()
        status = reader.ReadFile(p.as_posix())
        if status != IFSelect_RetDone:
            raise RuntimeError(f"IGES read failed for {p}")
        reader.TransferRoots()
        shape = reader.OneShape()

    return shape


def sample_cad_surfaces(path: str, uv_samples: int = 7):
    """
    Load a STEP/IGES file and sample all CAD faces on a small UV grid.

    Returns:
        surfaces: dict[int, dict]
            sid -> {
                "sid": int,
                "uv_bounds": (umin, umax, vmin, vmax),
                "uv_grid": (us, vs),
                "xyz_grid": P,
                "xyz_flat": P.reshape(-1, 3),
                "face": face,
            }
    """
    shape = load_cad_shape(path)

    exp = TopExp_Explorer(shape, TopAbs_FACE)

    nu = max(2, int(uv_samples))
    nv = max(2, int(uv_samples))
    surfaces: dict[int, dict] = {}
    sid = 0

    while exp.More():
        raw_face = exp.Current()
        exp.Next()

        print("Invoked with:", raw_face)

        face = _as_face(raw_face)
        if face is None:
            print("[WARN] Skipping a shape that cannot be cast to Face")
            continue

        try:
            surf = BRepAdaptor_Surface(face)  # <-- now only ever called with Face
        except Exception as e:
            print(f"[WARN] BRepAdaptor_Surface failed for face: {e}")
            continue

        umin = float(surf.FirstUParameter())
        umax = float(surf.LastUParameter())
        vmin = float(surf.FirstVParameter())
        vmax = float(surf.LastVParameter())

        if not np.isfinite([umin, umax, vmin, vmax]).all():
            print(f"[WARN] Non-finite UV bounds: {(umin, umax, vmin, vmax)}")
            continue
        if umax == umin or vmax == vmin:
            print(f"[WARN] Degenerate UV interval: U=({umin},{umax}) V=({vmin},{vmax})")
            continue

        us = np.linspace(umin, umax, nu)
        vs = np.linspace(vmin, vmax, nv)

        P = np.zeros((nu, nv, 3), dtype=float)

        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                try:
                    gp = surf.Value(u, v)
                except Exception:
                    continue
                P[i, j, 0] = gp.X()
                P[i, j, 1] = gp.Y()
                P[i, j, 2] = gp.Z()

        if not np.any(P):
            print("[WARN] No points sampled on this face; skipping")
            continue

        sid += 1
        surfaces[sid] = {
            "sid": sid,
            "uv_bounds": (umin, umax, vmin, vmax),
            "uv_grid": (us, vs),
            "xyz_grid": P,
            "xyz_flat": P.reshape(-1, 3),
            "face": face,
        }

    return surfaces


def plot_surface_samples(surfaces: dict[int, dict], face_ids=None):
    """
    Quick 3D scatter plot of sampled points per face.
    """
    if not surfaces:
        print("[INFO] No surfaces were sampled; nothing to plot.")
        return

    if face_ids is None:
        face_ids = sorted(surfaces.keys())
    else:
        face_ids = [int(fid) for fid in face_ids]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colours = plt.cm.tab20(np.linspace(0, 1, max(1, len(face_ids))))

    any_points = False

    for idx, sid in enumerate(face_ids):
        surf = surfaces.get(sid)
        if surf is None:
            continue
        pts = surf["xyz_flat"]
        if pts.size == 0:
            continue
        any_points = True
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=5, alpha=0.7,
            label=f"Face {sid}",
            color=colours[idx % len(colours)],
        )

    if not any_points:
        print("[INFO] All selected faces had zero points; nothing to plot.")
        return

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Sampled CAD surface points")

    # only add legend if there are labelled artists
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", fontsize="small")

    # equal-ish aspect ratio
    xyz_all = np.vstack([surfaces[s]["xyz_flat"] for s in face_ids if s in surfaces and surfaces[s]["xyz_flat"].size])
    xmid = 0.5 * (xyz_all[:, 0].min() + xyz_all[:, 0].max())
    ymid = 0.5 * (xyz_all[:, 1].min() + xyz_all[:, 1].max())
    zmid = 0.5 * (xyz_all[:, 2].min() + xyz_all[:, 2].max())
    r = max(
        xyz_all[:, 0].max() - xyz_all[:, 0].min(),
        xyz_all[:, 1].max() - xyz_all[:, 1].min(),
        xyz_all[:, 2].max() - xyz_all[:, 2].min(),
    ) * 0.5
    ax.set_xlim(xmid - r, xmid + r)
    ax.set_ylim(ymid - r, ymid + r)
    ax.set_zlim(zmid - r, zmid + r)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cad_path = os.path.join(os.getcwd(), "examples\Corner Bump CAD\Plattenmodell_CornerBump_2025-07-03_cf.stp")

    if not os.path.exists(cad_path):
        raise SystemExit(f"Set cad_path to a valid file (now: {cad_path})")

    surfaces = sample_cad_surfaces(cad_path, uv_samples=9)
    print(f"Found {len(surfaces)} surfaces")

    plot_surface_samples(surfaces)
