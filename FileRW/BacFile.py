import os, sys
import numpy as np

sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("Utilities"))
from FileRW.FliteFile import FliteFile
from FileRW.FroFile import FroFile
from FileRW.DatFile import DatFile
from Utilities.Vectors import vec_sub
from Utilities.PointClouds import bounds
from Utilities.PointClouds import dist_between_points
from Utilities.PointClouds import center_point_of_vertices
from Utilities.Axis import Axis

def auto_source_params(bounds_or_len, slider_val=50):
    """
    Compute element spacing and radii for a source region based on geometry size
    and a user-controlled slider.

    Parameters
    ----------
    bounds_or_len : tuple | float
        Either a (xmin,xmax,ymin,ymax,zmin,zmax) tuple or a single float length.
    slider_val : int
        Slider value between 0 (coarsest) and 100 (finest).

    Returns
    -------
    spacing, r1, r2 : floats
        Rounded spacing and radii.
    """
    import numpy as np

    # Extract characteristic length
    if isinstance(bounds_or_len, (tuple, list)) and len(bounds_or_len) == 6:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds_or_len
        L = max(xmax - xmin, ymax - ymin, zmax - zmin)
    else:
        L = float(bounds_or_len)

    # Normalize slider to 0–1
    t = np.clip(slider_val / 100.0, 0.0, 1.0)

    # Interpolate spacing scale: coarse = 2*L, fine = 0.5*L
    spacing = (2.0 - 1.5 * t) * L / 50.0  # denominator controls base density

    # Radii also scale with L
    r1 = (0.5 - 0.3 * t) * L
    r2 = (2.0 - 1.0 * t) * L

    # Round to nearest 10
    def round10(x):
        return max(10.0, round(x / 10.0) * 10.0)

    spacing = round10(spacing)
    r1 = round10(r1)
    r2 = round10(r2)

    return spacing, r1, r2

class BacFile(FliteFile):
    def __init__(self, name="default"):
        self.filename       = f"{name}.bac"
        self.stretch_points = []
        self.surface_things = []
        self.point_sources  = []
        self.line_sources   = []
        self.planar_sources = []

    @classmethod
    def defaultCRMFineMesh(cls, name="crm"):
        bf = cls(name)
        bf.stretch_points.append(StretchPoint(1, 1000.0, -1000.0, -1000.0, 200.0, 200.0, 200.0))
        bf.stretch_points.append(StretchPoint(2, 1000.0, 1000.0, -1000.0, 200.0, 200.0, 200.0))
        bf.stretch_points.append(StretchPoint(3, -1000.0, 1000.0, -1000.0, 200.0, 200.0, 200.0))
        bf.stretch_points.append(StretchPoint(4, -1000.0, -1000.0, -1000.0, 200.0, 200.0, 200.0))
        bf.stretch_points.append(StretchPoint(5, 1000.0, -1000.0, 1000.0, 200.0, 200.0, 200.0))
        bf.stretch_points.append(StretchPoint(6, 1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0))
        bf.stretch_points.append(StretchPoint(7, -1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0))
        bf.stretch_points.append(StretchPoint(8, -1000.0, -1000.0, 1000.0, 200.0, 200.0, 200.0))

        bf.surface_things.append(SurfaceThing(1, 1, 2, 4, 8))
        bf.surface_things.append(SurfaceThing(2, 1, 2, 8, 6))
        bf.surface_things.append(SurfaceThing(3, 1, 6, 8, 5))
        bf.surface_things.append(SurfaceThing(4, 2, 3, 4, 7))
        bf.surface_things.append(SurfaceThing(5, 2, 7, 4, 8))
        bf.surface_things.append(SurfaceThing(6, 2, 7, 8, 6))
        
        #                                                             x1,   y1,    z1,  s1,  r11,  r21,    x12   y2,    z2,   s2, r12,  r22, 
        bf.line_sources.append(LineSource("Wing Section 1",           0.0,  0.0,   0.0, 0.1, 1.0,  3.0,   72.56, 95.46, 8.1,  0.1, 1.0, 3, Axis.X))
        bf.line_sources.append(LineSource("Wing Section 2",        42.9,  0.0,  0.0,  0.1, 1.0, 3.0, 48.0,  28.0,   0.0,  0.1, 1.0, 3, Axis.X))
        bf.line_sources.append(LineSource("Wing Section 3",         82.0, 95.0,  8.0,  0.1, 0.75, 2.5 , 48.0, 28.0,  0.0,  0.1, 0.75, 2.5, Axis.X))
        
        #                                                             x1,   y1,    z1, s1,  r11,  r21, x2,  y2,  z2, s2,  r11,  r21,   x3   y3,    z3, s3,  r11,  r21
        bf.planar_sources.append(PlanarSource("Wing Section 1",           0.0, 0.0, 0.0, 0.5, 2.5, 5, 72.56, 95.46, 8.1, 0.5, 2.5, 5, 42.9, 0, 0.0, 0.5, 2.5, 5.0, Axis.X))
        bf.planar_sources.append(PlanarSource("Wing Section 2",        72.56, 95.46, 8.1, 0.5, 2.5, 5, 42.9, 0.0, 0.0, 0.5, 2.5, 5, 48.0, 28.0, 0.0, 0.5, 2.5, 5.0, Axis.X))
        bf.planar_sources.append(PlanarSource("Wing Section 3",         72.56, 95.46,  8.1,  0.5, 2.5, 5 , 48.0, 28.0,  0.0,  0.5, 2.5, 5, 82.0, 95.0, 8.0, 0.5, 2.5, 5, Axis.X))
        
        return bf
    
    @classmethod
    def fromGeometry(cls, name="auto", geometry=None, path=None, slider_val=50,
                    focus_surfaces=None, mode="parametric",
                    farfield_pad=0.35, uv_samples=5,
                    list_faces=False, fallback_to_mesh=True, gmsh_target_frac=0.002):
        """
        Build a BAC from CAD geometry.
        - quality: int [0–100], coarse→fine
        - focus_surfaces: list of FACE ids to refine
        """

        if mode.lower() != "parametric":
            raise NotImplementedError("fromGeometry currently supports mode='parametric' only.")

        fs = sorted(set(int(s) for s in (focus_surfaces or [])))

        # ---------- presets (aligned with fromMesh) ----------
        
        pq = slider_val if slider_val else 50
        fs = sorted(set(int(s) for s in (focus_surfaces or [])))

        # ---------- minimal OCC/OCP imports ----------
        try:
            # Prefer OCP (pip: cadquery-ocp)
            from OCP.STEPControl import STEPControl_Reader
            from OCP.IFSelect import IFSelect_RetDone
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            _BACKEND = "OCP"
        except Exception as e:
            raise RuntimeError(
                "fromGeometry(parametric) needs either cadquery-ocp (pip) or pythonocc-core (conda)."
            ) from e

        import numpy as np
        from pathlib import Path

        # ---------- helpers ----------
        def _as_face(shape_obj):
            """Cast generic TopoDS_Shape -> TopoDS_Face (OCP first; pythonOCC fallback)."""
            try:
                from OCP.TopoDS import topods_Face as _topods_Face
                return _topods_Face(shape_obj)
            except Exception:
                pass
            try:
                from OCC.Core.TopoDS import topods_Face as _topods_Face
                return _topods_Face(shape_obj)
            except Exception:
                pass
            try:
                from OCC.Core.TopoDS import TopoDS_Face as _TF
                return _TF.DownCast(shape_obj)
            except Exception:
                return None

        def _shape_inventory(shape):
            """Return counts of subshapes (quick debug of assemblies/wireframes)."""
            try:
                from OCP.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE,
                                        TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND)
                from OCP.TopExp import TopExp_Explorer
            except Exception:
                raise RuntimeError(
                    "fromGeometry(parametric) needs either cadquery-ocp (pip) or pythonocc-core (conda)."
                ) from e
                
            out = {}
            for kind, name in [(TopAbs_VERTEX,"VERTEX"), (TopAbs_EDGE,"EDGE"), (TopAbs_WIRE,"WIRE"),
                            (TopAbs_FACE,"FACE"), (TopAbs_SHELL,"SHELL"), (TopAbs_SOLID,"SOLID"),
                            (TopAbs_COMPOUND,"COMPOUND")]:
                n = 0
                ex = TopExp_Explorer(shape, kind)
                while ex.More():
                    n += 1; ex.Next()
                out[name] = n
            return out

        def _load_shape_from_step_any(step_path):
            """
            Load STEP and return a TopoDS_Shape.
            Tries: STEPControl_Reader → STEPCAF (assemblies) → cadquery importer.
            Returns a shape that *has faces*, else raises.
            """
            pth = Path(step_path)
            if not pth.exists():
                raise FileNotFoundError(f"STEP not found: {pth}")
            if pth.stat().st_size < 20:
                raise RuntimeError(f"STEP looks empty: {pth}")

            # Plain reader
            try:
                rdr = STEPControl_Reader()
                status = rdr.ReadFile(pth.as_posix())
                if status == IFSelect_RetDone:
                    rdr.TransferRoots()
                    shp = rdr.OneShape()
                    inv = _shape_inventory(shp)
                    if inv.get("FACE", 0) > 0:
                        return shp
            except Exception:
                pass

            # STEPCAF to flatten assemblies
            try:
                try:
                    from OCP.STEPCAFControl import STEPCAFControl_Reader
                    from OCP.XCAFApp import XCAFApp_Application
                    from OCP.TDocStd import TDocStd_Document
                    from OCP.XCAFDoc import XCAFDoc_DocumentTool_ShapeTool
                    from OCP.TDF import TDF_LabelSequence
                    from OCP.BRep import BRep_Builder
                    from OCP.TopoDS import TopoDS_Compound
                except Exception as e:
                    raise RuntimeError(
                        "fromGeometry(parametric) needs either cadquery-ocp (pip) or pythonocc-core (conda)."
                    ) from e

                app = XCAFApp_Application.GetApplication()
                doc = TDocStd_Document("xcaf"); app.NewDocument("MDTV-XCAF", doc)
                caf = STEPCAFControl_Reader()
                if caf.ReadFile(pth.as_posix()):
                    caf.Transfer(doc)
                    shape_tool = XCAFDoc_DocumentTool_ShapeTool(doc.Main())
                    labels = TDF_LabelSequence(); shape_tool.GetFreeShapes(labels)
                    bb = BRep_Builder(); comp = TopoDS_Compound(); bb.MakeCompound(comp)
                    for i in range(labels.Length()):
                        s = shape_tool.GetShape(labels.Value(i+1)); bb.Add(comp, s)
                    inv = _shape_inventory(comp)
                    if inv.get("FACE", 0) > 0:
                        return comp
            except Exception:
                pass

            # CadQuery fallback
            try:
                import cadquery as cq
                asm = cq.importers.importStep(pth.as_posix())
                shp = asm.val().wrapped
                inv = _shape_inventory(shp)
                if inv.get("FACE", 0) > 0:
                    return shp
            except Exception:
                pass

            raise RuntimeError("Could not load any B-Rep faces from STEP (part may be wireframe/tessellated).")

        # ---------- load shape ----------
        if geometry is not None:
            shape = geometry
        elif isinstance(path, str) and path.lower().endswith((".step", ".stp")):
            shape = _load_shape_from_step_any(path)
        else:
            raise TypeError("fromGeometry: provide a STEP path (.step/.stp) or a TopoDS_Shape 'geometry'.")

        # ---------- traverse faces parametrically ----------
        bf = cls(name)

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_id = 0

        all_pts = []           # for farfield bounds
        face_rect = {}         # sid -> ((umin,umax),(vmin,vmax))
        face_centroid = {}     # sid -> (x,y,z)
        face_lenU, face_lenV = {}, {}
        face_obj = {}          # sid -> TopoDS_Face

        nu = max(2, int(uv_samples))
        nv = max(2, int(uv_samples))

        while explorer.More():
            face_id += 1
            raw = explorer.Current(); explorer.Next()
            face = _as_face(raw)
            # In OCP, topods_Face returns an object even if null; bail if cannot adapt
            try:
                surf = BRepAdaptor_Surface(face)
            except Exception:
                continue

            umin = float(surf.FirstUParameter()); umax = float(surf.LastUParameter())
            vmin = float(surf.FirstVParameter()); vmax = float(surf.LastVParameter())
            if not np.isfinite([umin,umax,vmin,vmax]).all() or umax == umin or vmax == vmin:
                continue  # degenerate or infinite param range we don't want to sample

            # sample small UV grid for robust centroid/bounds & lengths
            us = np.linspace(umin, umax, nu)
            vs = np.linspace(vmin, vmax, nv)
            pts = []
            for u in us:
                for v in vs:
                    P = surf.Value(u, v)
                    pts.append((P.X(), P.Y(), P.Z()))
            if not pts:
                continue
            P = np.asarray(pts, float)
            c = P.mean(axis=0)

            # lengths along param directions (at mid param)
            um, vm = 0.5*(umin+umax), 0.5*(vmin+vmax)
            Pu0 = surf.Value(umin, vm); Pu1 = surf.Value(umax, vm)
            Pv0 = surf.Value(um, vmin); Pv1 = surf.Value(um, vmax)
            lenU = np.linalg.norm([Pu1.X()-Pu0.X(), Pu1.Y()-Pu0.Y(), Pu1.Z()-Pu0.Z()])
            lenV = np.linalg.norm([Pv1.X()-Pv0.X(), Pv1.Y()-Pv0.Y(), Pv1.Z()-Pv0.Z()])

            sid = len(face_rect) + 1  # compact ids for the valid/evaluable faces
            face_rect[sid] = ((umin, umax), (vmin, vmax))
            face_centroid[sid] = (float(c[0]), float(c[1]), float(c[2]))
            face_lenU[sid] = float(lenU)
            face_lenV[sid] = float(lenV)
            face_obj[sid] = face
            all_pts.extend(P.tolist())

        if list_faces:
            inv = _shape_inventory(shape)
            print(f"[fromGeometry] shape inventory: {inv}")
            if face_rect:
                print("[fromGeometry] faces:")
                for sid in sorted(face_rect):
                    (umin,umax),(vmin,vmax)=face_rect[sid]
                    print(f"  id={sid:3d}  lenU={face_lenU[sid]:.3f}  lenV={face_lenV[sid]:.3f}  centroid={face_centroid[sid]}")

        # ---------- if no faces: optional Gmsh fallback ----------
        if not all_pts:
            if fallback_to_mesh and isinstance(path, str) and path.lower().endswith((".step",".stp")):
                try:
                    import gmsh
                    # build an adapter (nodes + per-surface node lists) from Gmsh
                    class _MeshAdapter:
                        def __init__(self, nodes, surf_map):
                            self.nodes = np.asarray(nodes, float)
                            self._surfaces = {int(k): sorted(set(v)) for k, v in surf_map.items()}
                        def get_surface_nodes(self, sid: int):
                            sid = int(sid); gids = self._surfaces.get(sid, [])
                            return (gids, gids)

                    gmsh.initialize()
                    gmsh.open(path)
                    ents2 = gmsh.model.getEntities(2)
                    if not ents2:
                        gmsh.finalize()
                        raise RuntimeError("Gmsh: no surface entities in STEP.")
                    # bbox for size
                    bbox = [ np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf ]
                    for dim, tag in ents2:
                        bx = gmsh.model.getBoundingBox(dim, tag)
                        bbox[0] = min(bbox[0], bx[0]); bbox[1] = min(bbox[1], bx[1]); bbox[2] = min(bbox[2], bx[2])
                        bbox[3] = max(bbox[3], bx[3]); bbox[4] = max(bbox[4], bx[4]); bbox[5] = max(bbox[5], bx[5])
                    dx,dy,dz = bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]
                    diag = float(np.linalg.norm([dx,dy,dz]))
                    h = max(1e-5, gmsh_target_frac * diag)
                    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
                    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
                    gmsh.model.mesh.generate(2)

                    ents2_sorted = sorted(ents2, key=lambda e: e[1])
                    tag2idx = {}
                    nodes = []
                    surf_map = {}
                    next_sid = 1
                    for dim, stag in ents2_sorted:
                        nodeTags, coords, _ = gmsh.model.mesh.getNodes(dim, stag, includeBoundary=False)
                        idxs = []
                        for i, t in enumerate(nodeTags):
                            if t not in tag2idx:
                                j = len(nodes)
                                tag2idx[t] = j
                                nodes.append((coords[3*i], coords[3*i+1], coords[3*i+2]))
                            idxs.append(tag2idx[t])
                        surf_map[next_sid] = idxs
                        next_sid += 1
                    gmsh.finalize()

                    adapter = _MeshAdapter(nodes, surf_map)
                    valid_fs = [s for s in fs if 1 <= s <= len(surf_map)]
                    return cls.fromMesh(name=name, mesh=adapter, quality=slider_val, focus_surfaces=valid_fs)

                except Exception as e:
                    try:
                        gmsh.finalize()
                    except Exception:
                        pass
                    raise RuntimeError(
                        "fromGeometry: no faces in STEP and Gmsh fallback failed. "
                        f"Details: {e}"
                    )

            # no fallback or fallback disabled
            raise RuntimeError("fromGeometry(parametric): STEP contains no evaluable faces.")

        # ---------- build farfield box from bounds(all_pts) ----------
        all_pts = np.asarray(all_pts, float)
        xmin, ymin, zmin = all_pts.min(axis=0)
        xmax, ymax, zmax = all_pts.max(axis=0)
        Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)
        
        spacing, r1, r2 = auto_source_params((xmin, xmax, ymin, ymax, zmin, zmax), slider_val)

        fxmin = xmin - farfield_pad*Lx
        fxmax = xmax + farfield_pad*Lx
        fymin = ymin - farfield_pad*Ly
        fymax = ymax + farfield_pad*Ly
        fzmin = zmin - farfield_pad*Lz
        fzmax = zmax + farfield_pad*Lz

        sx = 1000.00
        sy = 1000.00
        sz = 1000.00

        corners = [
            (fxmax, fymin, fzmin),
            (fxmax, fymax, fzmin),
            (fxmin, fymax, fzmin),
            (fxmin, fymin, fzmin),
            (fxmax, fymin, fzmax),
            (fxmax, fymax, fzmax),
            (fxmin, fymax, fzmax),
            (fxmin, fymin, fzmax),
        ]
        for i, (x,y,z) in enumerate(corners, start=1):
            cls_point = StretchPoint(i, x, y, z, sx, sy, sz)
            bf.stretch_points.append(cls_point)


        # 6 faces of the farfield box (keep original ordering)
        bf.surface_things.append(SurfaceThing(1, 1, 2, 4, 8))  # z-
        bf.surface_things.append(SurfaceThing(2, 1, 2, 8, 6))  # x+
        bf.surface_things.append(SurfaceThing(3, 1, 6, 8, 5))  # z+
        bf.surface_things.append(SurfaceThing(4, 2, 3, 4, 7))  # y+
        bf.surface_things.append(SurfaceThing(5, 2, 7, 4, 8))  # x-
        bf.surface_things.append(SurfaceThing(6, 2, 7, 8, 6))  # y-

        # optional global mid-planes
        cx, cy, cz = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)
        spacing, r1, r2 = auto_source_params((xmin, xmax, ymin, ymax, zmin, zmax), slider_val)
        bf.planar_sources.append(PlanarSource(
            "Global XY midplane",
            fxmin, fymin, cz, spacing, r1, r2,
            fxmax, fymin, cz, spacing, r1, r2,
            fxmin, fymax, cz, spacing, r1, r2, Axis.Z))
        bf.planar_sources.append(PlanarSource(
            "Global YZ midplane",
            cx, fymin, fzmin, spacing, r1, r2,
            cx, fymin, fzmax, spacing, r1, r2,
            cx, fymax, fzmin, spacing, r1, r2, Axis.X))
        bf.planar_sources.append(PlanarSource(
            "Global XZ midplane",
            fxmin, cy, fzmin, spacing, r1, r2,
            fxmax, cy, fzmin, spacing, r1, r2,
            fxmin, cy, fzmax, spacing, r1, r2, Axis.Y))

        # ---------- per-face refinement ----------
        for sid in fs:
            if sid not in face_rect or sid not in face_obj:
                continue
            (umin, umax), (vmin, vmax) = face_rect[sid]
            um, vm = 0.5*(umin+umax), 0.5*(vmin+vmax)
            cx_s, cy_s, cz_s = face_centroid[sid]

            # point
            bf.point_sources.append(
                PointSource(f"SoI {sid} centroid", cx_s, cy_s, cz_s, spacing, r1, r2, center_on_cp=True)
            )

            # line along dominant param direction
            spacing, r1, r2 = auto_source_params(max(face_lenU[sid], face_lenV[sid]), slider_val)
            surf = BRepAdaptor_Surface(face_obj[sid])
            if face_lenU[sid] >= face_lenV[sid]:
                p0 = surf.Value(umin, vm);  p1 = surf.Value(umax, vm)
            else:
                p0 = surf.Value(um, vmin);  p1 = surf.Value(um, vmax)
            
            pA = surf.Value(umin, vmin)
            pB = surf.Value(umax, vmin)
            pC = surf.Value(umin, vmax)
            
            bf.line_sources.append(
                LineSource(f"SoI {sid} backbone",
                    p0.X(), p0.Y(), p0.Z(), spacing, r1, r2,
                    p1.X(), p1.Y(), p1.Z(), spacing, r1, r2, Axis.NONE)
            )

            bf.planar_sources.append(
                PlanarSource(f"SoI {sid} sheet",
                    pA.X(), pA.Y(), pA.Z(), spacing, r1, r2,
                    pB.X(), pB.Y(), pB.Z(), spacing, r1, r2,
                    pC.X(), pC.Y(), pC.Z(), spacing, r1, r2, Axis.NONE)
            )

        return bf
    
    @classmethod
    def fromDat(cls, dat_path: str, name="auto", quality=50, focus_surfaces=None):
            """
            Build a BAC file directly from a DAT file.
            - dat_path: path to .dat geometry file
            - quality: 0–100, coarse→fine mesh
            - focus_surfaces: optional list of surface IDs to refine
            """
            df = DatFile(dat_path)

            # gather all points from Surfaces1 section for bounds
            pts = []
            i = df.idx["Surfaces1"] + 1
            while i < df.idx["MeshGen"]:
                header = df._parse_ints(df.lines[i]); i += 1
                dims   = df._parse_ints(df.lines[i]); i += 1
                nx, ny = dims[0], dims[1]
                for _ in range(nx * ny):
                    pts.append(list(map(float, df.lines[i].split())))
                    i += 1

            pts = np.array(pts, float)
            xmin, ymin, zmin = pts.min(axis=0)
            xmax, ymax, zmax = pts.max(axis=0)

            # create standard BAC with farfield + refinement
            bf = cls.fromMesh(
                name=name,
                mesh=type("Tmp", (), {"nodes": pts, "get_surface_nodes": lambda self, sid: ([], [])})(),
                quality=quality,
                focus_surfaces=focus_surfaces or [],
            )
            return bf
    
    @classmethod
    def fromMesh(cls, name="auto", mesh=None, mesh_path=None,  quality=50, focus_surfaces=None,
        global_spacing=None, focus_spacing=None, growth=None, farfield_pad=0.35):
        # --- NORMALISE INPUTS ---
        try:
            from FileRW.FroFile import FroFile
        except Exception as e:
            FroFile = None

        # Allow either `mesh` as FroFile or str, or `mesh_path` as str
        if mesh is None and mesh_path is None:
            raise ValueError("fromMesh: provide `mesh` (FroFile or path) or `mesh_path`.")

        if isinstance(mesh, str):
            if FroFile is None:
                raise RuntimeError("fromMesh: cannot import FroFile to load mesh from path.")
            mesh = FroFile(mesh)
        elif mesh is None and isinstance(mesh_path, str):
            if FroFile is None:
                raise RuntimeError("fromMesh: cannot import FroFile to load mesh_path.")
            mesh = FroFile(mesh_path)

        # Final sanity
        if not hasattr(mesh, "nodes"):
            raise TypeError("fromMesh: `mesh` must be a FroFile or a path to a .fro file.")

        bf = cls(name)

        p = quality
        stretch_frac = 0.2

        focus_surfaces = list(focus_surfaces) if focus_surfaces else []

        # ----------------------------
        # 2) Global AABB + farfield
        # ----------------------------
        # bounds(mesh) returns (xmin,xmax,ymin,ymax,zmin,zmax)
        xmin,xmax,ymin,ymax,zmin,zmax = bounds(mesh.nodes)
        Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)
        Lmax = max(Lx, Ly, Lz)

        # Inflate the farfield box
        pad_x = Lx * farfield_pad
        pad_y = Ly * farfield_pad
        pad_z = Lz * farfield_pad

        fxmin = xmin - pad_x
        fxmax = xmax + pad_x
        fymin = ymin - pad_y
        fymax = ymax + pad_y
        fzmin = zmin - pad_z
        fzmax = zmax + pad_z

        # Corner sizes proportional to domain size
        sx = 1000.00
        sy = 1000.00
        sz = 1000.00

        # 8 StretchPoints (id = 1..8) in a consistent ordering
        corners = [
            (fxmax, fymin, fzmin),
            (fxmax, fymax, fzmin),
            (fxmin, fymax, fzmin),
            (fxmin, fymin, fzmin),
            (fxmax, fymin, fzmax),
            (fxmax, fymax, fzmax),
            (fxmin, fymax, fzmax),
            (fxmin, fymin, fzmax),
        ]
        for i, (x,y,z) in enumerate(corners, start=1):
            bf.stretch_points.append(StretchPoint(i, x, y, z, sx, sy, sz))

        # 6 SurfaceThing faces (using the above corner indices)
        # Faces: 1: z-, 2: x+, 3: z+, 4: y+, 5: x-, 6: y-
        bf.surface_things.append(SurfaceThing(1, 1, 2, 4, 8))  # bottom (z-)
        bf.surface_things.append(SurfaceThing(2, 1, 2, 8, 6))  # x+ side
        bf.surface_things.append(SurfaceThing(3, 1, 6, 8, 5))  # top (z+)
        bf.surface_things.append(SurfaceThing(4, 2, 3, 4, 7))  # y+ side
        bf.surface_things.append(SurfaceThing(5, 2, 7, 4, 8))  # x- side
        bf.surface_things.append(SurfaceThing(6, 2, 7, 8, 6))  # y- side

        # ----------------------------------------------------
        # 3) Add a gentle global "blanket" planar source set
        # ----------------------------------------------------
        # One planar per principal plane (mid-planes), using global spacing
        cx, cy, cz = (0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax))

        # XY mid-plane
        spacing, r1, r2 = auto_source_params((xmin, xmax, ymin, ymax, zmin, zmax), quality)
        bf.planar_sources.append(
            PlanarSource(
                "Global XY midplane",
                fxmin, fymin, cz, spacing, r1, r2,
                fxmax, fymin, cz, spacing, r1, r2,
                fxmin, fymax, cz, spacing, r1, r2,
                Axis.Z
            )
        )
        # YZ mid-plane
        bf.planar_sources.append(
            PlanarSource(
                "Global YZ midplane",
                cx, fymin, fzmin, spacing, r1, r2,
                cx, fymin, fzmax, spacing, r1, r2,
                cx, fymax, fzmin, spacing, r1, r2,
                Axis.X
            )
        )
        # XZ mid-plane
        bf.planar_sources.append(
            PlanarSource(
                "Global XZ midplane",
                fxmin, cy, fzmin, spacing, r1, r2,
                fxmax, cy, fzmin, spacing, r1, r2,
                fxmin, cy, fzmax, spacing, r1, r2,
                Axis.Y
            )
        )

        # ----------------------------------------------------
        # 4) Surfaces of interest refinements
        # ----------------------------------------------------
        # Helpers: local AABB, centroid, and principal axis pick
        def _surface_bounds_and_centroid(ff, sid: int):
            _, local_ids = ff.get_surface_nodes(int(sid))
            pts = ff.nodes[local_ids] if local_ids is not None and len(local_ids) else None
            if pts is None or len(pts) == 0:
                return None
            (sxmin,sxmax, symin,symax, szmin,szmax) = bounds(pts)
            c = center_point_of_vertices(pts)
            return (sxmin,sxmax, symin,symax, szmin,szmax), c, pts

        def _longest_axis(sbounds):
            sxmin,sxmax, symin,symax, szmin,szmax = sbounds
            ex = (sxmax - sxmin); ey = (symax - symin); ez = (szmax - szmin)
            if ex >= ey and ex >= ez:
                return "x"
            if ey >= ex and ey >= ez:
                return "y"
            return "z"

        for sid in focus_surfaces:
            info = _surface_bounds_and_centroid(mesh, sid)
            if info is None:
                continue
            (sxmin,sxmax, symin,symax, szmin,szmax), c, pts = info
            cx_s, cy_s, cz_s = c

            # Point source at centroid
            spacing, r1, r2 = auto_source_params((sxmin, sxmax, symin, symax, szmin, szmax), quality)
            bf.point_sources.append(
                PointSource(f"SoI {sid} centroid", cx_s, cy_s, cz_s, spacing, r1, r2, center_on_cp=True)
            )

            # Line source along longest local axis
            long_ax = _longest_axis((sxmin,sxmax, symin,symax, szmin,szmax))
            if long_ax == "x":
                x1,y1,z1 = sxmin, cy_s, cz_s
                x2,y2,z2 = sxmax, cy_s, cz_s
                axis_tag = Axis.X
            elif long_ax == "y":
                x1,y1,z1 = cx_s, symin, cz_s
                x2,y2,z2 = cx_s, symax, cz_s
                axis_tag = Axis.Y
            else:
                x1,y1,z1 = cx_s, cy_s, szmin
                x2,y2,z2 = cx_s, cy_s, szmax
                axis_tag = Axis.Z

            bf.line_sources.append(
                LineSource(
                    f"SoI {sid} backbone",
                    x1,y1,z1, spacing, r1, r2,
                    x2,y2,z2, spacing, r1, r2,
                    axis_tag
                )
            )

            # Planar source covering the local AABB (three corners)
            # Choose plane by the *smallest* thickness (assume surface approx. planar)
            ex = (sxmax - sxmin); ey = (symax - symin); ez = (szmax - szmin)
            # If thickness along z is smallest -> use XY plane, etc.
            if ez <= ex and ez <= ey:
                # XY-aligned plane at mid z
                zc = 0.5 * (szmin + szmax)
                x1,y1,z1 = sxmin, symin, zc
                x2,y2,z2 = sxmax, symin, zc
                x3,y3,z3 = sxmin, symax, zc
                axis_tag = Axis.Z
            elif ex <= ey and ex <= ez:
                # YZ-aligned plane at mid x
                xc = 0.5 * (sxmin + sxmax)
                x1,y1,z1 = xc, symin, szmin
                x2,y2,z2 = xc, symin, szmax
                x3,y3,z3 = xc, symax, szmin
                axis_tag = Axis.X
            else:
                # XZ-aligned plane at mid y
                yc = 0.5 * (symin + symax)
                x1,y1,z1 = sxmin, yc, szmin
                x2,y2,z2 = sxmax, yc, szmin
                x3,y3,z3 = sxmin, yc, szmax
                axis_tag = Axis.Y

            bf.planar_sources.append(
                PlanarSource(
                    f"SoI {sid} sheet",
                    x1,y1,z1, spacing, r1, r2,
                    x2,y2,z2, spacing, r1, r2,
                    x3,y3,z3, spacing, r1, r2,
                    axis_tag
                )
            )

        return bf
    
    @classmethod
    def fromFile(cls, filepath: str):
        # sometimes the bac file needs to be initialsed from a filepath 
        name = filepath.split("/")[-1][:-4]
        file = open(filepath).readlines()
        return cls.fromLines(name, file)

    @classmethod
    def fromLines(cls, name:str, file: list):
        bf = cls(name)

        section = ""
        line = 0
        
        section = file[line].strip("*").strip().strip(".").strip()
        line += 1
        # section should now = background mesh
        bm_data = file[line].split()
        line += 1
        for i in range(line, line + int(bm_data[0])):
            line1 = file[line].split()
            id = int(line1[0])
            x = float(line1[1])
            y = float(line1[2])
            z = float(line1[3])
            line += 1
            line2 = file[line].split()
            x_size = float(line2[3])
            line += 1
            line3 = file[line].split()
            y_size = float(line3[3])
            line += 1
            line4 = file[line].split()
            z_size = float(line4[3])
            line += 1
            sp = StretchPoint(id, x, y, z, x_size, y_size, z_size)
            bf.stretch_points.append(sp)
        
        for i in range(line, line + int(bm_data[1])):
            params = file[line].split()
            line += 1
            st = SurfaceThing(int(params[0]), int(params[1]), int(params[2]), int(params[3]), int(params[4]))
            bf.surface_things.append(st)
        
        # point sources
        section = file[line].strip("*").strip().strip(".").strip()
        line += 1
        for i in range(line, line + int(bm_data[2])):
            name = file[line]
            line += 1
            points = file[line].split()
            line += 1
            center_on_cp = False
            if name == "Central point source 1" or name == "Central point source 2":
                center_on_cp = True
            ps = PointSource(name, float(points[0]), float(points[1]), float(points[2]), float(points[3]), float(points[4]), float(points[5]), center_on_cp)
            bf.point_sources.append(ps)

        # line sources
        section = file[line].strip("*").strip().strip(".").strip()
        line += 1
        for i in range(line, line + int(bm_data[3])):
            name = file[line]
            line += 1
            points1 = file[line].split()
            line += 1
            points2 = file[line].split()
            line += 1
            axis = Axis.NONE
            if name == "Fuselage axis":
                axis = Axis.X
            ls = LineSource(name, 
                float(points1[0]), float(points1[1]), float(points1[2]), float(points1[3]), float(points1[4]), float(points1[5]),
                float(points2[0]), float(points2[1]), float(points2[2]), float(points2[3]), float(points2[4]), float(points2[5]),
                axis
                )
            bf.line_sources.append(ls)

        return bf

    @classmethod
    def fromString(cls, name:str, data: str):
        # data coming in from the server is sometimes gathered using cat <filename> and so is handed over to BacFile as a string
        # this class method splits that data into the inidiviual lines used to initialise the class
        file = data.split("\n")
        return cls.fromLines(name, file)

    @classmethod
    def local(cls):
        filename = cls.getFileExtension("bac")
        if filename is None:
            return None
        return cls.fromFile(filename) 

    def __str__(self):
        retval = ""  
        retval += "* background mesh ...\n"
        retval += f"{len(self.stretch_points)}\t{len(self.surface_things)}\t{len(self.point_sources)}\t{len(self.line_sources)}\t0\n"
        for sp in self.stretch_points:
            retval += f"{sp}\n"
        for st in self.surface_things:
            retval += f"{st}\n"
        retval += "* points\n"
        for ps in self.point_sources:
            retval += f"{ps}\n"
        retval += "* lines\n"
        for ls in self.line_sources:
            retval += f"{ls}\n"
        retval += "* triangles\n"
        for ls in self.planar_sources:
            retval += f"{ls}\n"
        return retval

    def UpdateSources(self, original_points, translated_points):
        for i in range(0, len(self.point_sources)):
            p = self.point_sources[i]
            print(f"Point Source = {i+1}/{len(self.point_sources)} p = {p.name}")
            if not p.center_on_cp:
                closest_point = -1
                closest_dist = 1000000000
                for j in range(0, len(original_points)):
                    op = original_points[j]
                    dist = dist_between_points(p.vertex, op)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = j

                closest_point_old_location = original_points[closest_point]
                closest_point_new_location = translated_points[closest_point]

                translation = vec_sub(closest_point_new_location, closest_point_old_location)
                self.point_sources[i].translate(translation)
            elif p.center_on_cp:
                old_point = center_point_of_vertices(original_points)
                new_point = center_point_of_vertices(translated_points)
                translation = vec_sub(new_point, old_point)
                
                _,_,ominy,omaxy,_,_ = bounds(original_points)
                _,_,tminy,tmaxy,_,_ = bounds(translated_points)
                _or = max(abs(ominy), abs(omaxy))
                _tr = max(abs(tminy), abs(tmaxy))
                ri = _tr/_or
                r1 = self.point_sources[i].r1 * ri
                r2 = self.point_sources[i].r2 * ri

                self.point_sources[i].translate(translation)
                self.point_sources[i].r1 = r1
                self.point_sources[i].r2 = r2
            print(p)

        for i in range(0, len(self.line_sources)):
            l = self.line_sources[i]
            print("Line Source = {}/{} l = {}".format(i+1, len(self.line_sources), l.name))
            if l.axis == Axis.NONE:
                p1_closest_point = -1
                p1_closest_dist = 1000000000
                p2_closest_point = -1
                p2_closest_dist = 1000000000
                for j in range(0, len(original_points)):
                    op = original_points[j]
                    dist1 = dist_between_points(l.vertex1, op)
                    dist2 = dist_between_points(l.vertex2, op)
                    if dist1 < p1_closest_dist:
                        p1_closest_dist = dist1
                        p1_closest_point = j
                    if dist2 < p2_closest_dist:
                        p2_closest_dist = dist2
                        p2_closest_point = j

                closest_point_old_location1 = original_points[p1_closest_point]
                closest_point_new_location1 = translated_points[p1_closest_point]
                closest_point_old_location2 = original_points[p2_closest_point]
                closest_point_new_location2 = translated_points[p2_closest_point]

                translation1 = vec_sub(closest_point_new_location1, closest_point_old_location1)
                translation2 = vec_sub(closest_point_new_location2, closest_point_old_location2)
                self.line_sources[i].translate_p1(translation1)
                self.line_sources[i].translate_p2(translation2)

            else:
                ominx, omaxx, ominy, omaxy, ominz, omaxz = bounds(translated_points)
                tminx, tmaxx, tminy, tmaxy, tminz, tmaxz = bounds(translated_points)
                if l.axis == Axis.X:
                    self.line_sources[i].x1 = tminx 
                    self.line_sources[i].x2 = tmaxx
                    _or = max(max(abs(ominy), abs(omaxy)), max(abs(ominz), abs(omaxz)))
                    _tr = max(max(abs(tminy), abs(tmaxy)), max(abs(tminz), abs(tmaxz)))
                    ri = _tr/_or
                    self.line_sources[i].r11 = self.line_sources[i].r11 * ri
                    self.line_sources[i].r21 = self.line_sources[i].r21 * ri * 1.5
                    self.line_sources[i].r12 = self.line_sources[i].r12 * ri
                    self.line_sources[i].r22 = self.line_sources[i].r22 * ri * 1.5
                else:
                    print("worry about this one later.... [TODO]")
            
            print(l.to_string())
            
        for i in range(0, len(self.planar_sources)):
            p = self.planar_sources[i]
            print("Plane Source = {}/{} p = {}".format(i+1, len(self.planar_sources), p.name))
            if p.axis == Axis.NONE:
                p1_closest_point = -1
                p1_closest_dist = 1000000000
                p2_closest_point = -1
                p2_closest_dist = 1000000000
                p3_closest_point = -1
                p3_closest_dist = 1000000000
                for j in range(0, len(original_points)):
                    op = original_points[j]
                    dist1 = dist_between_points(p.vertex1, op)
                    dist2 = dist_between_points(p.vertex2, op)
                    dist3 = dist_between_points(p.vertex3, op)
                    if dist1 < p1_closest_dist:
                        p1_closest_dist = dist1
                        p1_closest_point = j
                    if dist2 < p2_closest_dist:
                        p2_closest_dist = dist2
                        p2_closest_point = j
                    if dist3 < p3_closest_dist:
                        p3_closest_dist = dist3
                        p3_closest_point = j

                closest_point_old_location1 = original_points[p1_closest_point]
                closest_point_new_location1 = translated_points[p1_closest_point]
                closest_point_old_location2 = original_points[p2_closest_point]
                closest_point_new_location2 = translated_points[p2_closest_point]
                closest_point_old_location3 = original_points[p3_closest_point]
                closest_point_new_location3 = translated_points[p3_closest_point]

                translation1 = vec_sub(closest_point_new_location1, closest_point_old_location1)
                translation2 = vec_sub(closest_point_new_location2, closest_point_old_location2)
                translation3 = vec_sub(closest_point_new_location3, closest_point_old_location3)
                self.planar_sources[i].translate_p1(translation1)
                self.planar_sources[i].translate_p2(translation2)
                self.planar_sources[i].translate_p3(translation3)

            else:
                ominx, omaxx, ominy, omaxy, ominz, omaxz = bounds(translated_points)
                tminx, tmaxx, tminy, tmaxy, tminz, tmaxz = bounds(translated_points)
                if p.axis == Axis.X:
                    self.planar_sources[i].x1 = tminx 
                    self.planar_sources[i].x2 = tmaxx
                    _or = max(max(abs(ominy), abs(omaxy)), max(abs(ominz), abs(omaxz)))
                    _tr = max(max(abs(tminy), abs(tmaxy)), max(abs(tminz), abs(tmaxz)))
                else:
                    print("worry about this one later.... [TODO]")

class StretchPoint():
    def __init__(self, sp_id, x, y, z, x_size, y_size, z_size):
        self.sp_id = sp_id
        self.x = x
        self.y = y
        self.z = z
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.x1_stretch = 1.0
        self.y1_stretch = 0.0
        self.z1_stretch = 0.0
        self.x2_stretch = 0.0
        self.y2_stretch = 1.0
        self.z2_stretch = 0.0
        self.x3_stretch = 0.0
        self.y3_stretch = 0.0
        self.z3_stretch = 1.0

    def __str__(self):
        retval  = ""
        retval += f"{self.sp_id}\t{self.x}\t{self.y}\t{self.z}\n"
        retval += f"{self.x1_stretch}\t{self.x2_stretch}\t{self.x3_stretch}\t{self.x_size}\n"
        retval += f"{self.y1_stretch}\t{self.y2_stretch}\t{self.y3_stretch}\t{self.y_size}\n"
        retval += f"{self.z1_stretch}\t{self.z2_stretch}\t{self.z3_stretch}\t{self.z_size}"
        return retval

class SurfaceThing():
    def __init__(self, st_id, surf1, surf2, surf3, surf4):
        self.st_id = st_id
        self.surf1 = surf1
        self.surf2 = surf2
        self.surf3 = surf3
        self.surf4 = surf4
    
    def __str__(self):
        return f"\t{self.st_id}\t{self.surf1}\t{self.surf2}\t{self.surf3}\t{self.surf4}"

class PointSource():
    def __init__(self, name, x, y, z, spacing, r1, r2, center_on_cp=False):
        self.name         = name.strip().strip("\n").strip()
        self.x            = x
        self.y            = y
        self.z            = z
        self.spacing      = spacing
        self.r1           = r1
        self.r2           = r2
        self.center_on_cp = center_on_cp

    @property
    def vertex(self):
        return (self.x, self.y, self.z)

    def translate(self, translation):
        self.x += translation[0]
        self.y += translation[1]
        self.z += translation[2]
    
    def __str__(self):
        retval = ""
        retval += f"{self.name}\n"
        retval += f"{self.x} {self.y} {self.z} {self.spacing} {self.r1} {self.r2}"
        return retval

class LineSource():
    def __init__(self, name, x1, y1, z1, spacing1, r11, r21, x2, y2, z2, spacing2, r12, r22, axis=Axis.NONE):
        self.name     = name.strip().strip("\n").strip()
        self.x1       = x1
        self.y1       = y1 
        self.z1       = z1
        self.spacing1 = spacing1
        self.r11      = r11
        self.r21      = r21
        self.x2       = x2 
        self.y2       = y2 
        self.z2       = z2 
        self.spacing2 = spacing2
        self.r12      = r12
        self.r22      = r22
        self.axis     = axis

    @property
    def vertex1(self):
        return (self.x1, self.y1, self.z1)

    @property
    def vertex2(self):
        return (self.x2, self.y2, self.z2)

    def translate_p1(self, translation):
        self.x1 += translation[0]
        self.y1 += translation[1]
        self.z1 += translation[2]
    
    def translate_p2(self, translation):
        self.x2 += translation[0]
        self.y2 += translation[1]
        self.z2 += translation[2]

    def __str__(self):
        retval = ""
        retval += f"{self.name}\n"
        retval += f"{self.x1} {self.y1} {self.z1} {self.spacing1} {self.r11} {self.r21}\n"
        retval += f"{self.x2} {self.y2} {self.z2} {self.spacing2} {self.r12} {self.r22}  "
        return retval
    
class PlanarSource():
    def __init__(self, name, x1, y1, z1, s1, r11, r12, x2, y2, z2, s2, r21, r22, x3, y3, z3, s3, r31, r32, axis=Axis.NONE):
        self.name     = name.strip().strip("\n").strip()
        self.x1       = x1
        self.y1       = y1 
        self.z1       = z1
        self.spacing1 = s1
        self.r11      = r11
        self.r12      = r12
        self.x2       = x2 
        self.y2       = y2 
        self.z2       = z2
        self.spacing2 = s2
        self.r21      = r21
        self.r22      = r22
        self.x3      = x3
        self.y3       = y3 
        self.z3       = z3
        self.spacing3 = s3
        self.r31      = r31
        self.r32      = r32
        self.axis     = axis

    @property
    def vertex1(self):
        return (self.x1, self.y1, self.z1)

    @property
    def vertex2(self):
        return (self.x2, self.y2, self.z2)
    
    @property
    def vertex3(self):
        return (self.x3, self.y3, self.z3)

    def translate_p1(self, translation):
        self.x1 += translation[0]
        self.y1 += translation[1]
        self.z1 += translation[2]
    
    def translate_p2(self, translation):
        self.x2 += translation[0]
        self.y2 += translation[1]
        self.z2 += translation[2]
        
    def translate_p3(self, translation):
        self.x3 += translation[0]
        self.y3 += translation[1]
        self.z3 += translation[2]

    def __str__(self):
        retval = ""
        retval += f"{self.name}\n"
        retval += f"{self.x1} {self.y1} {self.z1} {self.spacing1} {self.r11} {self.r12}\n"
        retval += f"{self.x2} {self.y2} {self.z2} {self.spacing2} {self.r21} {self.r22}\n"
        retval += f"{self.x3} {self.y3} {self.z3} {self.spacing3} {self.r31} {self.r32}"
        return retval


if __name__ == "__main__":
    bac = BacFile.local()
    if bac is None:
        print("No BAC file found in local directory")
    else:
        os.system(f"echo '{bac}' | less -f /dev/stdin")