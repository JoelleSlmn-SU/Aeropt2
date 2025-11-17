# cad_viewer_occ.py
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtWidgets import QWidget
import ctypes

from OCP.Aspect import Aspect_DisplayConnection
from OCP.OpenGl import OpenGl_GraphicDriver
from OCP.V3d import V3d_Viewer
from OCP.AIS import AIS_InteractiveContext, AIS_Shape
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_MATRAGRAY
from OCP.STEPControl import STEPControl_Reader
from OCP.IGESControl import IGESControl_Reader
from OCP.TopoDS import TopoDS_Shape
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.Graphic3d import Graphic3d_RenderingParams
from OCP.WNT import WNT_Window

def _capsule_from_hwnd(hwnd: int, name: bytes = b"HWND"):
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return PyCapsule_New(ctypes.c_void_p(hwnd), name, None)

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
        self.faces: list[TopoDS_Shape]   = []
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
            self.faces.append(exp.Current())
            exp.Next()

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
