# cad_viewer_occ.py
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtWidgets import QWidget
import ctypes
import numpy as np
import math

from OCP.Aspect import Aspect_DisplayConnection
from OCP.OpenGl import OpenGl_GraphicDriver
from OCP.V3d import V3d_Viewer
from OCP.AIS import AIS_InteractiveContext, AIS_Shape
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_MATRAGRAY
from OCP.STEPControl import STEPControl_Reader
from OCP.IGESControl import IGESControl_Reader
from OCP.TopoDS import TopoDS_Shape, TopoDS_Face
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.Graphic3d import Graphic3d_RenderingParams
from OCP.WNT import WNT_Window
from OCP.BRep import BRep_Tool
from OCP.BRepTools import BRepTools
from OCP.Geom import Geom_BSplineSurface
from OCP.GeomAdaptor import GeomAdaptor_Surface

def _capsule_from_hwnd(hwnd: int, name: bytes = b"HWND"):
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return PyCapsule_New(ctypes.c_void_p(hwnd), name, None)

def face_to_bspline(face_like):
    """
    Given a TopoDS_Face-like object, return a Geom_BSplineSurface,
    or None if the underlying surface is not a BSpline.

    Handles both:
      - h_surf being a handle with .GetObject()
      - h_surf being a concrete Geom_Surface / Geom_BSplineSurface
    """
    face = _as_face(face_like)
    if face is None:
        return None

    # Get the geometric surface from the face
    h_surf = BRep_Tool.Surface_s(face)
    if h_surf is None:
        return None

    # Newer bindings often return the surface directly; older return a handle.
    surf = h_surf.GetObject() if hasattr(h_surf, "GetObject") else h_surf

    # Try to downcast to Geom_BSplineSurface
    try:
        bs = Geom_BSplineSurface.DownCast(surf)
    except Exception:
        bs = None

    # Some bindings let DownCast return None even when surf is already a BSpline
    if bs is None and isinstance(surf, Geom_BSplineSurface):
        bs = surf

    if bs is None:
        # Not a BSpline surface
        return None

    return bs

def _as_face(shape_obj):
    """
    Cast a TopoDS_Shape (face-like) into a TopoDS_Face.
    
    Returns:
        TopoDS_Face instance or None if cast fails.
    """
    # Import at function level to catch any issues
    try:
        from OCP.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS
    except ImportError as e:
        print(f"[CAD] _as_face: failed to import OCP.TopoDS: {e}")
        return None

    # Already a face?
    if isinstance(shape_obj, TopoDS_Face):
        return shape_obj

    # Must at least be a shape
    if not isinstance(shape_obj, TopoDS_Shape):
        print(f"[CAD] _as_face: object is not a TopoDS_Shape: {type(shape_obj)}")
        return None

    # Use the TopoDS static method (capitalized, not lowercase 'topods')
    try:
        face = TopoDS.Face_s(shape_obj)  # Note: Face_s (static method)
    except Exception as e:
        print(f"[CAD] TopoDS.Face_s cast failed: {e} (type={type(shape_obj)})")
        return None

    # Check for null face
    try:
        if hasattr(face, "IsNull") and face.IsNull():
            print("[CAD] _as_face: TopoDS.Face_s returned a null face")
            return None
    except Exception:
        pass

    return face

def _bspline_poles_from_face(face_like):
    """
    Given a TopoDS_Face, return an (N, 3) array of BSpline poles,
    or None if the supporting surface is not a BSpline.
    """
    face = _as_face(face_like)
    if face is None:
        return None

    h_surf = BRep_Tool.Surface_s(face)  # <-- note _s
    if h_surf is None:
        return None

    surf = h_surf.GetObject() if hasattr(h_surf, "GetObject") else h_surf

    try:
        bs = Geom_BSplineSurface.DownCast(surf)
    except Exception:
        bs = None

    if bs is None and isinstance(surf, Geom_BSplineSurface):
        bs = surf

    nu = bs.NbUPoles()
    nv = bs.NbVPoles()

    pts = np.zeros((nu * nv, 3), dtype=float)
    k = 0
    for i in range(1, nu + 1):       # OCCT indices are 1-based
        for j in range(1, nv + 1):
            p = bs.Pole(i, j)
            pts[k, 0] = p.X()
            pts[k, 1] = p.Y()
            pts[k, 2] = p.Z()
            k += 1

    return pts

from OCP.BRepAdaptor import BRepAdaptor_Surface  # already imported at top

def _control_points_from_face(face_like, nu_samples=25, nv_samples=25):
    """
    Given a TopoDS_Face-like object, return an (N, 3) array of control points.

    Implementation mirrors the robust face sampling used in BacFile.fromGeometry:
    - We always go via BRepAdaptor_Surface(face)
    - We use its finite U/V parameter bounds
    - We then sample a small (nu_samples x nv_samples) UV grid on the trimmed face
    """
    face = _as_face(face_like)
    if face is None:
        print("[CAD] _control_points_from_face: _as_face returned None")
        return None

    try:
        surf = BRepAdaptor_Surface(face)
    except Exception as e:
        print(f"[CAD] BRepAdaptor_Surface failed: {e}")
        return None

    # Finite param range for this trimmed face
    try:
        umin = float(surf.FirstUParameter())
        umax = float(surf.LastUParameter())
        vmin = float(surf.FirstVParameter())
        vmax = float(surf.LastVParameter())
    except Exception as e:
        print(f"[CAD] Failed to query UV bounds from BRepAdaptor_Surface: {e}")
        return None

    # Guard against degenerate / infinite ranges
    if not np.isfinite([umin, umax, vmin, vmax]).all():
        print(f"[CAD] Non-finite UV bounds: {(umin, umax, vmin, vmax)}")
        return None
    if umax == umin or vmax == vmin:
        print(f"[CAD] Degenerate UV interval for face: "
              f"U=({umin},{umax}), V=({vmin},{vmax})")
        return None

    nu = max(2, int(nu_samples))
    nv = max(2, int(nv_samples))

    us = np.linspace(umin, umax, nu)
    vs = np.linspace(vmin, vmax, nv)

    pts = np.zeros((nu * nv, 3), dtype=float)
    k = 0

    for u in us:
        for v in vs:
            try:
                P = surf.Value(u, v)
            except Exception:
                # skip any dodgy param pairs rather than killing the whole face
                continue
            pts[k, 0] = P.X()
            pts[k, 1] = P.Y()
            pts[k, 2] = P.Z()
            k += 1

    if k == 0:
        print("[CAD] _control_points_from_face: no points sampled on face")
        return None

    return pts[:k, :]

class OCCViewer(QWidget):
    """Interactive OpenCascade (OCP/AIS) viewer embedded in PyQt5."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # critical for OCC: own a native handle and paint directly
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setAttribute(Qt.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True) 

        # OCC graphics
        self._display = Aspect_DisplayConnection()
        self._driver  = OpenGl_GraphicDriver(self._display)
        self.viewer   = V3d_Viewer(self._driver)
        self.viewer.SetDefaultLights(); self.viewer.SetLightOn()
        self.view     = self.viewer.CreateView()
        self.ctx      = AIS_InteractiveContext(self.viewer)
        self._configure_rendering()

        # OCC window bound to this widget (created on demand)
        self._wnt = None

        # state
        self.shape: TopoDS_Shape | None = None
        self.faces: list[TopoDS_Face]   = []
        self.ais_faces: list[AIS_Shape]  = []
        self.ais_highlight: AIS_Shape | None = None
        self.hidden = set()

        # colors
        self._col_neutral = Quantity_Color(0.82, 0.82, 0.85, Quantity_TOC_RGB)
        self._col_high    = Quantity_Color(1.00, 0.93, 0.25, Quantity_TOC_RGB)

        # interaction
        self._mouse_pressed = False
        self._btn = None
        self._last = QPoint()

        try:
            self.view.SetBackgroundColor(Quantity_Color(0.10, 0.10, 0.12, Quantity_TOC_RGB))
        except Exception:
            pass
        try:
            self.ctx.SetTrihedronOn()
        except Exception:
            pass

        self.setMouseTracking(True)  # to get move events without dragging

    # ---------- Qt painting ----------
    def sizeHint(self) -> QSize:
        return QSize(900, 600)

    def paintEngine(self):
        # tell Qt we paint externally (OCC) so it won't try to use QPaintEngine
        return None

    def paintEvent(self, ev):
        # draw the OCC view whenever Qt asks us to paint
        self._ensure_window()
        try:
            self.view.Redraw()
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)
        self._ensure_window()
        try:
            self.view.MustBeResized()
            self.view.Redraw()
        except Exception:
            pass

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._ensure_window()
        try:
            self.view.MustBeResized()
            self.view.Redraw()
        except Exception:
            pass

    def _ensure_window(self):
        if self._wnt is not None:
            return
        wid  = self.effectiveWinId() if hasattr(self, "effectiveWinId") else self.winId()
        hwnd = int(wid)
        self._wnt = WNT_Window(_capsule_from_hwnd(hwnd), Quantity_NOC_MATRAGRAY)
        self.view.SetWindow(self._wnt)
        if not self._wnt.IsMapped():
            self._wnt.Map()
        self.view.MustBeResized()
        self.view.Redraw()

    def _configure_rendering(self):
        try:
            rp: Graphic3d_RenderingParams = self.view.ChangeRenderingParams()
            if hasattr(rp, "ToEnableDepthPeeling"): rp.ToEnableDepthPeeling = True
            elif hasattr(rp, "UseDepthPeeling"):    rp.UseDepthPeeling = True
            if hasattr(rp, "NbMsaaSamples"):        rp.NbMsaaSamples = 8
        except Exception:
            pass

    # ---------- CAD I/O ----------
    # ---------- CAD control nodes (for parameterisation) ----------

    def get_face_control_net(self, face_index: int):
        """
        Return an (N, 3) array of control points for a given face index.

        - If face is BSpline → returns its poles.
        - Otherwise → returns sampled (u,v) points on the face.
        """
        print(f"[DEBUG] get_face_control_net called with index {face_index}")
        
        if self.shape is None or not self.faces:
            print(f"[DEBUG] No shape or faces: shape={self.shape}, faces={len(self.faces) if self.faces else 0}")
            return None
        
        if not (0 <= face_index < len(self.faces)):
            print(f"[DEBUG] Face index {face_index} out of range [0, {len(self.faces)})")
            return None

        face = self.faces[face_index]
        print(f"[DEBUG] Got face object: {face}")
        
        try:
            pts = _control_points_from_face(face)
            print(f"[DEBUG] _control_points_from_face returned: {pts.shape if pts is not None else None}")
            return pts
        except Exception as e:
            print(f"[DEBUG] Exception in _control_points_from_face: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_control_nodes_from_faces(self, face_indices=None):
        """
        Collect CAD-level control nodes from the requested faces.
        """
        print(f"[DEBUG] get_control_nodes_from_faces called with indices: {face_indices}")
        
        if self.shape is None or not self.faces:
            print(f"[DEBUG] No shape or faces available")
            return None, None, {}

        if face_indices is None:
            face_indices = list(range(len(self.faces)))
            print(f"[DEBUG] Using all {len(face_indices)} faces")

        all_pts = []
        all_ids = []
        face_slices = {}
        offset = 0

        for idx in face_indices:
            print(f"[DEBUG] Processing face index {idx}")
            
            if not (0 <= idx < len(self.faces)):
                print(f"[DEBUG] Face {idx} out of range, skipping")
                continue

            pts = self.get_face_control_net(idx)
            if pts is None or len(pts) == 0:
                print(f"[DEBUG] Face {idx} returned no control points")
                continue

            n_pts = pts.shape[0]
            print(f"[DEBUG] Face {idx} contributed {n_pts} points")
            all_pts.append(pts)
            all_ids.append(np.full(n_pts, idx, dtype=int))
            face_slices[idx] = slice(offset, offset + n_pts)
            offset += n_pts

        if not all_pts:
            print(f"[DEBUG] No points collected from any face")
            return None, None, {}

        control_nodes = np.vstack(all_pts)
        face_ids = np.concatenate(all_ids)
        print(f"[DEBUG] Total control nodes collected: {control_nodes.shape[0]}")
        return control_nodes, face_ids, face_slices

    def _read_shape(self, path: str) -> TopoDS_Shape:
        p = path.lower()
        if p.endswith((".step", ".stp")):
            r = STEPControl_Reader(); st = r.ReadFile(path)
            if st != 1 or r.TransferRoots() == 0: raise RuntimeError("STEP import failed")
            return r.OneShape()
        if p.endswith((".iges", ".igs")):
            r = IGESControl_Reader(); st = r.ReadFile(path)
            if st != 1 or r.TransferRoots() == 0: raise RuntimeError("IGES import failed")
            return r.OneShape()
        raise ValueError(f"Unsupported CAD extension: {path}")

    # ---------- Display & control ----------
    def display_cad(self, path: str):
        """Load and display all faces immediately (shaded)."""
        self._ensure_window()
        self.clear(redraw=False)

        self.shape = self._read_shape(path)
        if self.shape is None or self.shape.IsNull():
            raise RuntimeError("Loaded shape is null")

        # collect faces
        self.faces.clear()
        exp = TopExp_Explorer(self.shape, TopAbs_FACE)
        while exp.More():
            raw_face = exp.Current()
            face = _as_face(raw_face)
            if face is None:
                print(f"[CAD] Skipping explorer shape that could not be cast to Face: {raw_face}")
            else:
                self.faces.append(face)
            exp.Next()

        print(f"[CAD] Collected {len(self.faces)} faces: {self.faces}")
        
        # create & display each face as its own AIS actor
        self.ais_faces = []
        self.hidden.clear()
        for f in self.faces:
            a = AIS_Shape(f)
            try:
                a.SetColor(self._col_neutral)
            except Exception:
                pass
            self.ctx.Display(a, False)              # batch
            self.ctx.SetDisplayMode(a, 1, False)    # shaded
            self.ais_faces.append(a)

        # finalize & show
        self.ctx.UpdateCurrentViewer()
        self.view.Redraw()
        self.fit_all()
        self.update()

    def highlight_face(self, idx: int):
        if idx < 0 or idx >= len(self.ais_faces): return
        # unhighlight previous (restore neutral color)
        if self.ais_highlight is not None:
            try:
                self.ais_highlight.UnsetColor()
                self.ais_highlight.SetColor(self._col_neutral)
            except Exception:
                pass
            self.ais_highlight = None

        a = self.ais_faces[idx]
        try:
            a.SetColor(self._col_high)
        except Exception:
            pass
        self.ais_highlight = a
        self.ctx.UpdateCurrentViewer()
        try: self.view.Redraw()
        except Exception: pass
        self.update()

    def hide_faces(self, indices):
        for i in indices:
            if 0 <= i < len(self.ais_faces):
                try:
                    self.ctx.Erase(self.ais_faces[i], False)
                    self.hidden.add(i)
                except Exception:
                    pass
        self.ctx.UpdateCurrentViewer()
        try: self.view.Redraw()
        except Exception: pass
        self.update()

    def show_all_faces(self):
        for i, a in enumerate(self.ais_faces):
            try:
                self.ctx.Display(a, False)
                a.UnsetColor()
                a.SetColor(self._col_neutral)
            except Exception:
                pass
        self.hidden.clear()
        self.ais_highlight = None
        self.ctx.UpdateCurrentViewer()
        try: self.view.Redraw()
        except Exception: pass
        self.update()

    def reset_camera(self):
        self.fit_all()

    def clear(self, redraw=True):
        try:
            for a in self.ais_faces:
                self.ctx.Erase(a, False)
            if self.ais_highlight is not None:
                self.ctx.Erase(self.ais_highlight, False)
        except Exception:
            pass
        self.ais_faces = []
        self.faces = []
        self.hidden.clear()
        self.shape = None
        if redraw:
            try: self.view.Redraw()
            except Exception: pass
        self.update()

    def fit_all(self):
        try:
            self._ensure_window()
            self.view.MustBeResized()
            self.view.FitAll()
            self.view.ZFitAll()
            self.view.Redraw()
        except Exception as e:
            print(f"[WARN] fit_all failed: {e}")
        self.update()

    # ---------- Mouse interaction ----------
    def mousePressEvent(self, ev):
        self._ensure_window()
        self._mouse_pressed = True
        self._btn = ev.button()
        self._last = ev.pos()
        if self._btn == Qt.LeftButton:
            # start OCC rotation at this point
            self.view.StartRotation(ev.x(), ev.y())
        ev.accept()

    def mouseMoveEvent(self, ev):
        if not self._mouse_pressed:
            return
        dx = ev.x() - self._last.x()
        dy = ev.y() - self._last.y()

        if self._btn == Qt.LeftButton:
            # rotate with OCC's built-in
            self.view.Rotation(ev.x(), ev.y())
        elif self._btn == Qt.MiddleButton or (self._btn == Qt.LeftButton and ev.modifiers() & Qt.ShiftModifier):
            # pan
            try:
                self.view.Pan(dx, -dy)  # Y inverted
            except Exception:
                pass
        elif self._btn == Qt.RightButton or (self._btn == Qt.LeftButton and ev.modifiers() & Qt.ControlModifier):
            # zoom (fallback if wheel not used)
            try:
                # robust zoom: try factor, else 2-point zoom
                if hasattr(self.view, "Zoom"):
                    try:
                        # many builds expose Zoom(factor: float)
                        self.view.Zoom(1.0 - dy * 0.01)
                    except TypeError:
                        # fallback: Zoom(x1,y1,x2,y2)
                        self.view.Zoom(self._last.x(), self._last.y(), ev.x(), ev.y())
            except Exception:
                pass

        self._last = ev.pos()
        self.update()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self._mouse_pressed = False
        self._btn = None
        ev.accept()

    def wheelEvent(self, ev):
        # zoom with the wheel
        delta = ev.angleDelta().y()
        try:
            if hasattr(self.view, "Zoom"):
                try:
                    # prefer Zoom(factor)
                    self.view.Zoom(0.9 if delta > 0 else 1.1)
                except TypeError:
                    # fallback: 2-point zoom using cursor as focus
                    x = ev.x(); y = ev.y()
                    dz = -delta * 0.25
                    self.view.Zoom(x, y, x, y + dz)
        except Exception:
            pass
        self.update()
        ev.accept()
        