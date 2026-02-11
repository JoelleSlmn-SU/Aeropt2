import os, sys
import numpy as np
import matplotlib.cm as cm
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, pyqtSlot, QMetaObject, Q_ARG, QObject, QEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy, QDialog, QComboBox, QSpinBox,
    QFileDialog, QLineEdit, QHBoxLayout, QInputDialog, QCheckBox, QListWidget, QDialogButtonBox
)
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

for dir in ["FileRW", "ShapeParameterization", "MeshGeneration", "ConvertFileType", "Remote", "Local", "GUI"]:
    sys.path.append(os.path.dirname(dir))
from ShapeParameterization.surfaceFitting import selectControlNodes
from MeshGeneration.meshFile import load_mesh
from MeshGeneration.controlNodeDisp import _surface_normals, _map_normals_to_control
from Local.runSimLocal import *
from ConvertFileType.convertToStep import *
from GUI.workers import MorphWorker
import pickle
from MeshGeneration.pcaBasis import (
    make_signature, load_pca_basis, save_pca_basis, build_pca_basis
)

def _dedup_preserve_order(seq):
        """Deduplicate while preserving order."""
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    
class VtkRightClickFilter(QObject):
    def __init__(self, parent, on_click):
        super().__init__(parent)
        self._on_click = on_click

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            # Qt gives x,y in widget coords (origin top-left)
            x = event.pos().x()
            y = event.pos().y()
            self._on_click(x, y)
            return True  # consume
        return False

class MeshViewer(QWidget):
    control_ready = pyqtSignal()
    log_signal = pyqtSignal(str)
    
    debug_plot_requested = pyqtSignal(object, str)
    requestPlotCNs = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__()
        self.main_window = parent
        self.main_layout = QVBoxLayout()
        self.plotter = QtInteractor(self)
        self.plotter.setVisible(False)
        self.main_layout.addWidget(self.plotter)

        self.placeholder = QLabel("No mesh loaded")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.placeholder)
        self.setLayout(self.main_layout)
        
        self.rbf_original = getattr(self.main_window, "rbf_original", None)
        self.rbf_current  = getattr(self.main_window, "rbf_current",  None)

        self.figure = None
        self.canvas = None
        self.plot_ax = None
        self._dbg_scatter = None
        self._dbg_order = None        # np.ndarray of global node IDs (point order)
        self._dbg_pending = False
        self._dbg_last_title = "Debug"
        self.debug_plot_requested.connect(self._update_debug_plot_mpl)
        
        self.rigid_boundary_translation = False

        self.mesh_obj = None
        self.surface_actors = {}
        self.TSurfaces = []
        self.CSurfaces = []
        self.USurfaces = []
        self.hidden_surfaces = set()
        
        self.debug_mode = True
        
        if hasattr(parent, "logger"):
            self.logger = parent.logger
        else:
            self.logger = None
            
        self.log_signal.connect(self._append_log)
        
        self.requestPlotCNs.connect(self.plot_control_displacements, type=Qt.QueuedConnection)
        self._cn_preview_busy = False
        self._cn_preview_actors = []
        
        self.open_debug_btn = QPushButton("Open Interactive Debug")
        self.open_debug_btn.clicked.connect(self._open_last_debug_html)
        self.main_layout.addWidget(self.open_debug_btn)
        self.open_debug_btn.setVisible(False)

    def set_logger(self, logger):
        self.logger = logger
        
    def _append_log(self, msg): 
        if self.logger:
            self.logger.log(msg)
        
    def log(self, msg):
        self.log_signal.emit(msg)
                
    def set_remote_output_directory(self, path):
        self.remote_output_dir = path
        
    def set_output_directory(self, path):
        self.output_dir = path
        if self.output_dir == os.path.join(os.getcwd(), "Outputs"):
            for sub in ["preprocessed", "solutions", "volumes", "surfaces"]:
                os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

    def set_input_filepath(self, path):
        self.input_filepath = path

    def load_mesh_file(self, filename):
        self.rbf_original = getattr(self.main_window, "rbf_original", None)
        self.rbf_current  = getattr(self.main_window, "rbf_current", None)
        
        if hasattr(self, 'placeholder'):
            self.main_layout.removeWidget(self.placeholder)
            self.placeholder.deleteLater()

        self.plotter.clear()
        self.mesh_obj = load_mesh(filename)
        self.surface_actors.clear()
        self.hidden_surfaces.clear()

        self._setup_controls()
        self._add_mesh_to_plotter()
        
        self._show_surface_summary()

    def reset_viewer(self):
        """Hard reset of MeshViewer UI + state."""
        # stop any morph thread
        try:
            if hasattr(self, "_morph_thread") and self._morph_thread:
                if self._morph_thread.isRunning():
                    self._morph_thread.requestInterruption()
                    self._morph_thread.quit()
                    self._morph_thread.wait(2000)
        except Exception:
            pass

        # remove summary dock
        if hasattr(self, "summary_dock") and self.summary_dock:
            try:
                self.main_window.removeDockWidget(self.summary_dock)
            except Exception:
                pass
            self.summary_dock = None
            self.summary_table = None

        # kill highlight overlay
        if hasattr(self, "_highlight_actor") and self._highlight_actor is not None:
            try:
                self.plotter.remove_actor(self._highlight_actor)
            except Exception:
                pass
            self._highlight_actor = None

        # tear down plotter safely (avoid wglMakeCurrent spam)
        try:
            if self.plotter:
                try: self.plotter.disable_picking()
                except Exception: pass
                try:
                    rw = getattr(self.plotter, "ren_win", None)
                    if rw is not None:
                        rw.Finalize()
                    iren = getattr(self.plotter, "interactor", None)
                    if iren is not None:
                        iren.TerminateApp()
                except Exception:
                    pass
                try: self.plotter.clear()
                except Exception: pass
                try: self.plotter.close()
                except Exception: pass
        except Exception:
            pass

        # clear dynamic widgets created during sessions
        for name in [
            "reset_btn","cam_btn","hide_btn","tc_btn","export_btn",
            "T_btn","C_btn","U_btn","edit_btn","finish_btn",
            "form_widget","cn_btn","back_btn","debug_checkbox","open_debug_btn"
        ]:
            w = getattr(self, name, None)
            if w is not None:
                try: w.setParent(None)
                except Exception: pass
                setattr(self, name, None)

        # rebuild a fresh, hidden plotter and a clean placeholder
        try:
            self.plotter = QtInteractor(self)
            self.plotter.setVisible(False)
            self.main_layout.insertWidget(0, self.plotter)  # keep it first
        except Exception:
            pass

        # placeholder label
        if not hasattr(self, "placeholder") or self.placeholder is None:
            from PyQt5.QtWidgets import QLabel
            self.placeholder = QLabel("No mesh loaded")
            self.placeholder.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(self.placeholder)
        else:
            self.placeholder.setText("No mesh loaded")
            self.placeholder.setVisible(True)

        # reset internal state
        self.mesh_obj = None
        self.surface_actors = {}
        self.hidden_surfaces = set()
        self.TSurfaces, self.USurfaces, self.CSurfaces = [], [], []
        self.T_names, self.U_names, self.C_names = [], [], []
        self.control_nodes = None
        self._dbg_scatter = None
        self._dbg_order = None
        self._dbg_pending = False
        self._dbg_last_title = "Debug"
        self.figure = None
        self.canvas = None
        self.plot_ax = None

    def _show_surface_summary(self):
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QDockWidget, QAbstractItemView

        table = QTableWidget()
        names = self.mesh_obj.get_surface_names()
        table.setRowCount(len(names))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Surface", "Points", "Bounds"])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for i, name in enumerate(names):
            pts = self.mesh_obj.get_surface_points(name)
            pts = np.array(pts)
            bounds = np.ptp(pts,axis=0) if pts is not None and len(pts) else [0,0,0]
            table.setItem(i, 0, QTableWidgetItem(str(name)))
            table.setItem(i, 1, QTableWidgetItem(str(len(pts))))
            table.setItem(i, 2, QTableWidgetItem(f"{bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}"))

        dock = QDockWidget("Surface Summary", self)
        dock.setWidget(table)

        # keep refs so they don't get GC'd
        self.summary_dock = dock
        self.summary_table = table

        # react to selection
        table.itemSelectionChanged.connect(self._on_surface_row_selected)

        self.main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.log("[INFO] Surface summary panel updated.")
        
    def _on_surface_row_selected(self):
        # clear previous overlay
        if hasattr(self, "_highlight_actor") and self._highlight_actor is not None:
            try:
                self.plotter.remove_actor(self._highlight_actor)
            except Exception:
                pass
            self._highlight_actor = None

        sel = self.summary_table.selectedItems()
        if not sel:
            self.plotter.render()
            return

        # first column of the selected row holds the name
        row = self.summary_table.currentRow()
        name_item = self.summary_table.item(row, 0)
        if not name_item:
            return

        surface_name = name_item.text()

        try:
            # keep everything visible; just add a wireframe overlay on the chosen surface
            mesh = self.mesh_obj.get_surface_mesh(surface_name)
            self._highlight_actor = self.plotter.add_mesh(
                mesh,
                style="wireframe",
                color="yellow",
                line_width=3,
                opacity=1.0,
                pickable=False
            )
            try:
                self.plotter.fly_to(mesh.center)
            except Exception:
                pass

            self.plotter.render()
            self.log(f"[INFO] Highlighted surface: {surface_name}")

        except Exception as e:
            self.log(f"[WARN] Could not highlight '{surface_name}': {e}")

    def _setup_controls(self):
        self.plotter = QtInteractor(self)
        self.main_layout.addWidget(self.plotter)
        self.plotter.reset_camera()
        
        self.reset_btn = QPushButton("Reset Surfaces")
        self.reset_btn.clicked.connect(self.reset_surfaces)
        self.main_layout.addWidget(self.reset_btn)

        self.cam_btn = QPushButton("Reset Camera")
        self.cam_btn.clicked.connect(self.reset_camera)
        self.main_layout.addWidget(self.cam_btn)
        
        self.hide_btn = QPushButton("Hide Surface")
        self.hide_btn.clicked.connect(self.toggle_hide_mode)
        self.main_layout.addWidget(self.hide_btn)
        self.hide_mode_enabled = False

        self.tc_btn = QPushButton("Select T/U/C Surfaces")
        self.tc_btn.clicked.connect(self.tc_surfaces)
        self.main_layout.addWidget(self.tc_btn)

        self.export_btn = QPushButton("Export Visible Mesh")
        self.export_btn.clicked.connect(self.export_visible_mesh)
        self.main_layout.addWidget(self.export_btn)
        
        self.debug_checkbox = QCheckBox("Enable morphing debug")
        self.debug_checkbox.setChecked(True)
        self.debug_checkbox.stateChanged.connect(lambda: setattr(self, 'debug_mode', self.debug_checkbox.isChecked()))
        self.main_layout.addWidget(self.debug_checkbox)

    def _add_mesh_to_plotter(self):
        self.plotter.clear()
        surface_names = self.mesh_obj.get_surface_names()
        cmap = cm.get_cmap("tab20")
        self.colors = [tuple(cmap(i)[:3]) for i in range(len(surface_names))]
        sargs = dict(interactive=True)
        
        for i, name in enumerate(surface_names):
            color = self.colors[i]
            try:
                mesh = self.mesh_obj.get_surface_mesh(name)
                actor = self.plotter.add_mesh(mesh, color=color, show_edges=True, pickable=True,
                                            show_scalar_bar=True, scalar_bar_args=sargs)
                self.surface_actors[name] = actor
            except Exception as e:
                self.log(f"Failed to plot surface '{name}': {e}")

        self.plotter.setVisible(True)
        self.plotter.render()
        self.plotter.enable_anti_aliasing()

    def _ensure_mpl_canvas(self):
        if self.figure is None:
            self.figure = Figure(figsize=(5, 5))
            self.canvas = FigureCanvas(self.figure)
            self.plot_ax = self.figure.add_subplot(111, projection='3d')
            self.plot_ax.set_xlabel("X"); self.plot_ax.set_ylabel("Y"); self.plot_ax.set_zlabel("Z")
            self.main_layout.addWidget(self.canvas)

    def _build_point_order_and_colors(self, fro_obj):
        """Compute a fixed point order (global IDs) and a color array per point (by surface)."""
        exclude = {50, 98}  # same exclusions you used before
        order, colors = [], []
        sids = fro_obj.get_surface_ids()

        # Color map (tab20) stable by surface index
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('tab20', len(sids))

        for i, sid in enumerate(sids):
            sid_int = int(sid)
            if sid_int in exclude:
                continue
            g_ids, lc_ids = fro_obj.get_surface_nodes(sid_int)
            if lc_ids is None:
                # Some FroFile APIs return only local; ensure g_ids is list of globals
                g_ids = g_ids if g_ids is not None else []
            # Append this surface's self.points
            order.extend(list(g_ids))
            # Same color for this surface's self.points
            c = cmap(i)
            colors.extend([c] * len(g_ids))

        self._dbg_order = np.asarray(order, dtype=int)
        self._dbg_colors = np.asarray(colors)

    def _set_axes_equal(self, ax, pts):
        # Equal aspect for 3D
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        centers = (mins + maxs) / 2.0
        ranges = (maxs - mins)
        r = ranges.max() * 0.5
        ax.set_xlim(centers[0]-r, centers[0]+r)
        ax.set_ylim(centers[1]-r, centers[1]+r)
        ax.set_zlim(centers[2]-r, centers[2]+r)
        ax.set_aspect('equal')

    @pyqtSlot(object, str)
    def _update_debug_plot_mpl(self, fro_obj, title):
        # GUI thread: fast in-place update of one scatter with all self.points (no sampling)
        self._ensure_mpl_canvas()

        # First time: build fixed order & scatter
        if self._dbg_scatter is None or self._dbg_order is None:
            self._build_point_order_and_colors(fro_obj)
            if self._dbg_order.size == 0:
                # Nothing to plot (all excluded?) — just return gracefully
                return

            pts = fro_obj.nodes[self._dbg_order]
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

            self.plot_ax.clear()
            self.plot_ax.set_xlabel("X"); self.plot_ax.set_ylabel("Y"); self.plot_ax.set_zlabel("Z")
            self._dbg_scatter = self.plot_ax.scatter(
                x, y, z, s=2, c=self._dbg_colors, marker='.', depthshade=False
            )
            self._set_axes_equal(self.plot_ax, pts)
            self.plot_ax.set_title(title)
            self.canvas.draw_idle()
            return

        # Subsequent calls: update the _offsets3d of the existing Path3DCollection
        pts = fro_obj.nodes[self._dbg_order]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        # Matplotlib trick: update scatter3D data without recreating the artist
        self._dbg_scatter._offsets3d = (x, y, z)   # noqa: private attr

        # Throttle to ~30 FPS
        self._dbg_last_title = title
        if self._dbg_pending:
            return
        self._dbg_pending = True

        def _do_draw():
            try:
                self.plot_ax.set_title(self._dbg_last_title)
                # keep axes bounds stable for visual continuity; comment next line to auto-rescale
                # self._set_axes_equal(self.plot_ax, pts)
                self.canvas.draw_idle()
            finally:
                self._dbg_pending = False

        QTimer.singleShot(33, _do_draw)
        
        self.open_debug_btn.setVisible(True)
        
    def _open_last_debug_html(self):
        import webbrowser, glob
        html_files = glob.glob(os.path.join(self.output_dir, "*.html"))
        if not html_files:
            self.log("[WARN] No debug HTML found.")
            return
        latest = max(html_files, key=os.path.getctime)
        webbrowser.open(f"file://{os.path.abspath(latest)}")
        self.log(f"[INFO] Opened {latest} in browser.")
        
    def _surface_id_from_actor_name(self, name: str) -> str:
        """Extract the surface ID you use elsewhere from the actor's name."""
        return name.split()[-1] if "Surface" in name else name

    def reset_surfaces(self):
        for _, actor in self.surface_actors.items():
            actor.SetVisibility(True)
        if hasattr(self, "_highlight_actor") and self._highlight_actor is not None:
            try:
                self.plotter.remove_actor(self._highlight_actor)
            except Exception:
                pass
            self._highlight_actor = None
        if hasattr(self, "summary_table"):
            self.summary_table.clearSelection()
        self.plotter.render()
        self.hidden_surfaces.clear()

    def toggle_hide_mode(self):
        # If currently enabling hide mode
        if not self.hide_mode_enabled:
            # Turn off T/C selection mode if it's active
            if hasattr(self, 'select_mode_enabled') and self.select_mode_enabled:
                self.plotter.disable_picking()
                self.plotter.remove_actor("select_label")
                self.select_mode_enabled = False

            self.plotter.enable_mesh_picking(callback=self._hide_surface, show_message=True, use_actor=True, font_size=12)
            self.plotter.add_text("Hide mode ON", name="hide_label", position='lower_right', font_size=6)
            self.hide_btn.setText("Disable Hide Mode")
            self.hide_mode_enabled = True

        # If disabling hide mode
        else:
            self.plotter.disable_picking()
            self.plotter.remove_actor("hide_label")
            self.hide_btn.setText("Hide Surface")
            self.hide_mode_enabled = False

    def _hide_surface(self, actor):
        for name, act in self.surface_actors.items():
            if act == actor:
                act.SetVisibility(False)
                self.hidden_surfaces.add(name)
                break
        self.plotter.render()

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.render()

    def export_visible_mesh(self):
        visible_points = []
        for name, actor in self.surface_actors.items():
            if actor.GetVisibility():
                pts = self.mesh_obj.get_surface_points(name)
                visible_points.append(pv.PolyData(pts))

        if not visible_points:
            self.log("No surfaces visible.")
            return

        combined = visible_points[0]
        for part in visible_points[1:]:
            combined = combined.merge(part)

        save_path = os.path.join(self.output_dir, "exported_visible_mesh.vtk")
        combined.save(save_path)
        self.log(f"Visible mesh exported to: {save_path}")

    def tc_surfaces(self):
        for btn in [self.tc_btn, self.export_btn]:
            btn.setVisible(False)

        # T/U/C selection buttons
        self.T_btn = QPushButton("Select T Surfaces")
        self.T_btn.clicked.connect(self.select_T_surfaces)
        self.main_layout.addWidget(self.T_btn)

        self.C_btn = QPushButton("Select C Surfaces")
        self.C_btn.clicked.connect(self.select_C_surfaces)
        self.main_layout.addWidget(self.C_btn)

        self.U_btn = QPushButton("Select U Surfaces")
        self.U_btn.clicked.connect(self.select_U_surfaces)
        self.main_layout.addWidget(self.U_btn)
        
        self.edit_btn = QPushButton("Edit Surface Selections")
        self.edit_btn.clicked.connect(self.open_edit_dialog)
        self.main_layout.addWidget(self.edit_btn)

        self.finish_btn = QPushButton("Done")
        self.finish_btn.clicked.connect(self.finish_selection)
        self.main_layout.addWidget(self.finish_btn)

    def _enter_select_mode(self, callback):
        """Common select-mode enter logic."""
        if getattr(self, "hide_mode_enabled", False):
            self.plotter.disable_picking()
            try:
                self.plotter.remove_actor("hide_label")
            except Exception:
                pass
            self.hide_mode_enabled = False

        self.plotter.disable_picking()
        try:
            self.plotter.remove_actor("select_label")
        except Exception:
            pass

        self.plotter.enable_mesh_picking(
            callback=callback,
            show_message=True,
            font_size=12,
            use_actor=True
        )
        self.plotter.add_text("Select mode ON", name="select_label",
                            position='lower_right', font_size=6)
        self.select_mode_enabled = True


    def select_T_surfaces(self):
        self._enter_select_mode(self.mark_T_surface)

    def select_C_surfaces(self):
        self._enter_select_mode(self.mark_C_surface)

    def select_U_surfaces(self):
        self._enter_select_mode(self.mark_U_surface)

    # --- marking callbacks ---
    def mark_T_surface(self, actor):
        for name, act in self.surface_actors.items():
            if act == actor:
                sid = self.mesh_obj.get_surface_id(name)
                if sid not in self.TSurfaces:
                    self.TSurfaces.append(sid)   
                    if not hasattr(self, "T_names"): self.T_names = []
                    self.T_names.append(name)    
                break

    def _remove_selected(self, list_widget, backing_list):
        for item in list_widget.selectedItems():
            sid = item.text()
            list_widget.takeItem(list_widget.row(item))
            try:
                backing_list.remove(int(sid))
            except Exception:
                try:
                    backing_list.remove(sid)
                except:
                    pass
        self.log(f"Updated selections: T={self.TSurfaces}, U={self.USurfaces}, C={self.CSurfaces}")

    def mark_C_surface(self, actor):
        for name, act in self.surface_actors.items():
            if act == actor:
                sid = self.mesh_obj.get_surface_id(name)
                if sid not in self.CSurfaces:
                    self.CSurfaces.append(sid)   
                    if not hasattr(self, "C_names"): self.C_names = []
                    self.C_names.append(name)    
                break

    def mark_U_surface(self, actor):
        for name, act in self.surface_actors.items():
            if act == actor:
                sid = self.mesh_obj.get_surface_id(name)
                if sid not in self.USurfaces:
                    self.USurfaces.append(sid)   
                    if not hasattr(self, "U_names"): self.U_names = []
                    self.U_names.append(name)    
                break

    def manual_select_control_nodes(self, t_mesh: pv.PolyData, n_pick: int):
        import vtk
        from PyQt5.QtCore import Qt

        # reset view
        try:
            self.plotter.close()
        except Exception:
            pass
        self.plotter = QtInteractor(self)
        self.main_layout.addWidget(self.plotter)

        self._manual_pick_targets = int(n_pick)
        self._manual_pick_ids = []
        self._manual_pick_pts = []
        self._manual_pick_actors = []

        # Keep reference to the mesh we are picking on
        self._manual_t_poly = t_mesh
        self._manual_t_points = np.asarray(t_mesh.points, float)

        # Render as a proper surface (cells exist)
        self.plotter.add_mesh(t_mesh, color="#52b7ba", style="points", show_edges=False, pickable=True)
        self.plotter.add_text(
            f"Manual CN pick: RIGHT click {n_pick} points on T surface\n"
            f"Press 'Finish Manual Pick' when done",
            name="manual_pick_label",
            position="upper_left",
            font_size=10,
        )
        self.plotter.reset_camera()
        self.plotter.render()

        # ensure widget takes mouse events
        self.plotter.setFocusPolicy(Qt.StrongFocus)
        self.plotter.setFocus()
        self.plotter.setMouseTracking(True)

        # VTK picker
        self._vtk_picker = vtk.vtkCellPicker()
        self._vtk_picker.SetTolerance(0.05)

        def _qt_right_click(x, y):
            self.log(f"[DEBUG] Qt Right click at ({x},{y})")

            ren = self.plotter.renderer
            dpr = 1.0
            try:
                dpr = float(self.plotter.devicePixelRatioF())
            except Exception:
                try:
                    dpr = float(self.plotter.devicePixelRatio())
                except Exception:
                    dpr = 1.0

            xp = int(round(x * dpr))
            yp = int(round((self.plotter.height() - y) * dpr))  # flip Y for VTK

            ok = self._vtk_picker.Pick(xp, yp, 0, self.plotter.renderer)
            
            if not ok:
                self.log("[DEBUG] Picker: no hit.")
                return

            pos = np.array(self._vtk_picker.GetPickPosition(), float)
            pid = int(self._manual_t_poly.find_closest_point(pos))
            if pid < 0:
                self.log("[DEBUG] Picker hit but no closest point.")
                return

            picked_pt = np.array(self._manual_t_poly.points[pid], float)
            self._on_manual_point_picked(picked_pt)

        # remove old filter (if any) + install new
        try:
            if hasattr(self, "_vtk_rc_filter") and self._vtk_rc_filter is not None:
                self.plotter.removeEventFilter(self._vtk_rc_filter)
        except Exception:
            pass

        self._vtk_rc_filter = VtkRightClickFilter(self.plotter, _qt_right_click)
        self.plotter.installEventFilter(self._vtk_rc_filter)
        self.log("[INFO] Manual pick: Qt right-click filter installed.")

        # buttons
        self._manual_finish_btn = QPushButton("Finish Manual Pick")
        self._manual_finish_btn.clicked.connect(self._finalize_manual_pick)
        self.main_layout.addWidget(self._manual_finish_btn)

        self._manual_clear_btn = QPushButton("Clear Picks")
        self._manual_clear_btn.clicked.connect(self._clear_manual_pick)
        self.main_layout.addWidget(self._manual_clear_btn)

        self.log("[INFO] Manual picking enabled: RIGHT click on the surface mesh.")


    def _on_manual_point_picked(self, picked_point):
        if picked_point is None:
            self.log("[INFO] No point detected.")
            return
        if len(self._manual_pick_ids) >= self._manual_pick_targets:
            self.log("[INFO] Already picked required number of control nodes.")
            return

        p = np.asarray(picked_point, float).reshape(1, 3)
        # snap to nearest actual T node
        dif = self._manual_t_points - p
        i = int(np.argmin(np.einsum("ij,ij->i", dif, dif)))

        if i in self._manual_pick_ids:
            return  # prevent duplicates

        self._manual_pick_ids.append(i)
        pt = self._manual_t_points[i]
        self._manual_pick_pts.append(pt)

        # draw a marker
        marker = pv.Sphere(radius=0.01 * (np.linalg.norm(self._manual_t_points.max(0)-self._manual_t_points.min(0)) + 1e-12),
                        center=pt)
        act = self.plotter.add_mesh(marker, color="black")
        if not hasattr(self, "_manual_pick_actors"):
            self._manual_pick_actors = []
        self._manual_pick_actors.append(act)

        self.log(f"[INFO] Picked {len(self._manual_pick_ids)}/{self._manual_pick_targets} control nodes.")
        if len(self._manual_pick_ids) == self._manual_pick_targets:
            self.log("[INFO] Required picks reached. Click 'Finish Manual Pick' to continue.")

    def _clear_manual_pick(self):
        self._manual_pick_ids = []
        self._manual_pick_pts = []
        for a in getattr(self, "_manual_pick_actors", []):
            try:
                self.plotter.remove_actor(a)
            except Exception:
                pass
        self._manual_pick_actors = []
        self.plotter.render()
        self.log("[INFO] Manual picks cleared.")

    def _finalize_manual_pick(self):
        if len(self._manual_pick_pts) == 0:
            self.log("[WARN] No points selected.")
            return
        if len(self._manual_pick_pts) != self._manual_pick_targets:
            self.log(f"[WARN] Selected {len(self._manual_pick_pts)} but expected {self._manual_pick_targets}.")
            return

        self.points = self._manual_t_points
        self.control_nodes = np.asarray(self._manual_pick_pts, float)

        # compute normals like auto path
        surf_normals = _surface_normals(self.points, knn=16)
        self.control_normals = _map_normals_to_control(self.control_nodes, self.points, surf_normals, k=12)

        # compute patch scale (same as plot_T_surfaces)
        try:
            pts = np.asarray(self.points, float)
            d = pts.max(axis=0) - pts.min(axis=0)
            self.t_patch_scale = float(np.linalg.norm(d))
            self.log(f"[INFO] T-patch scale (manual pick) = {self.t_patch_scale:.6g}")
        except Exception as e:
            self.t_patch_scale = None
            self.log(f"[WARN] Failed to compute T-patch scale: {e}")

        # remove manual pick buttons
        try:
            self._manual_finish_btn.setParent(None)
            self._manual_clear_btn.setParent(None)
        except Exception:
            pass

        
        self.plot_T_surfaces()
        
    def auto_select_control_nodes(self, output_path, num_input):
        self.points, self.control_nodes = selectControlNodes(output_path, self.output_dir, num_input)
        surf_normals = _surface_normals(self.points, knn=16)
        self.control_normals = _map_normals_to_control(self.control_nodes, self.points, surf_normals, k=12)

        try:
            pts = np.asarray(self.points, float)
            d = pts.max(axis=0) - pts.min(axis=0)
            self.t_patch_scale = float(np.linalg.norm(d))  # bbox diagonal
            self.log(f"[INFO] T-patch scale (from plot_T_surfaces points) = {self.t_patch_scale:.6g}")
        except Exception as e:
            self.t_patch_scale = None
            self.log(f"[WARN] Failed to compute T-patch scale in plot_T_surfaces: {e}")
        
        self.plotter.close()
        
        self.plot_T_surfaces()

    # --- finish selection ---
    def finish_selection(self):
        if not hasattr(self, 'output_dir') or not self.output_dir:
            default_dir = os.path.join(os.getcwd(), "Outputs")
            self.set_output_directory(default_dir)
            self.log(f"[Info] Output directory auto-set to: {default_dir}")

        # --- Helper: robust int list ---
        def _as_int_list(lst):
            out = []
            for x in lst:
                try:
                    out.append(int(x))
                except Exception:
                    pass
            return _dedup_preserve_order(out)

        # --- Preserve NAME selections for preview BEFORE overwriting with ints ---
        # Assumption: your surface_actors keys are the same "names" used by mesh_obj.get_surface_points(name)
        T_names = list(self.TSurfaces) if self.TSurfaces else []
        C_names = list(self.CSurfaces) if self.CSurfaces else []
        U_names = list(self.USurfaces) if self.USurfaces else []

        # --- Universe of surfaces as IDs ---
        all_ids = _as_int_list([self.mesh_obj.get_surface_id(nm) for nm in self.surface_actors.keys()])

        def _names_or_ids_to_ids(seq):
            ids = []
            for item in (seq or []):
                # if it's already a number-like string/int
                try:
                    ids.append(int(item))
                    continue
                except Exception:
                    pass
                # otherwise treat as name
                try:
                    sid = self.mesh_obj.get_surface_id(item)
                    if sid is not None:
                        ids.append(int(sid))
                except Exception:
                    pass
            return _dedup_preserve_order(ids)

        T = _names_or_ids_to_ids(T_names)
        U = _names_or_ids_to_ids(U_names)

        # Auto-compute U from the universe (recommended)
        C = _dedup_preserve_order([sid for sid in all_ids if sid not in (set(T) | set(U))])

        # Save IDs back (these are what should go to morph_config)
        self.TSurfaces = T
        self.CSurfaces = C
        self.USurfaces = U

        # Logging (accurate)
        self.log(f"Marked T surfaces (IDs): {T}")
        self.log(f"Marked C surfaces (IDs): {C}")
        self.log(f"Marked U surfaces (IDs): {U}")
        self.log(f"[CHECK] overlap(T,C)={set(T) & set(C)}")

        # Hide selection widgets (unchanged)
        for wdg in [self.hide_btn, self.reset_btn, self.cam_btn, self.T_btn, self.C_btn,
                    self.U_btn, self.edit_btn, self.finish_btn, self.export_btn, self.debug_checkbox]:
            try:
                wdg.setVisible(False)
            except Exception:
                pass

        dlg = ControlNodeSelectionDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        mode, num_input = dlg.get_choice()

        # --- Build preview PolyData from T surface NAMES ---
        t_mesh = None
        for nm in getattr(self, "T_names", []):   # IMPORTANT: use names, not IDs
            surf = self.mesh_obj.get_surface_mesh(nm)
            if surf is None or surf.n_cells == 0:
                continue
            t_mesh = surf.copy() if t_mesh is None else t_mesh.merge(surf)

        if t_mesh is None or t_mesh.n_cells == 0:
            self.log("[WARN] No T-surface cells available for manual selection.")
            return

        # Make it “picker-friendly”
        t_mesh = t_mesh.extract_surface().triangulate().clean()

        # If you still want to save for downstream, save the real surface mesh
        output_path = os.path.join(self.output_dir, "surfaces", "output.vtk")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        t_mesh.save(output_path)
        
        
        if mode == "auto":
            self.auto_select_control_nodes(output_path, num_input)
        else:
            # manual selection from T mesh nodes
            self.manual_select_control_nodes(t_mesh, num_input)


    def open_edit_dialog(self):
        # Use stored name lists (fall back to empty if not set yet)
        T_names = getattr(self, "T_names", [])
        U_names = getattr(self, "U_names", [])
        C_names = getattr(self, "C_names", [])

        all_surface_names = list(self.mesh_obj.norm_label_to_id.keys())
        
        dlg = SurfaceEditDialog(all_surface_names, T_names, U_names, C_names, self)
        dlg.show()
        dlg.accepted.connect(lambda: self._apply_edit_results(dlg))
        
    def _apply_edit_results(self, dlg):
        T_names, U_names, C_names = dlg.get_results()
        self.T_names, self.U_names, self.C_names = T_names, U_names, C_names
        self.TSurfaces = [self.mesh_obj.friendly_names[nm] for nm in T_names]
        self.USurfaces = [self.mesh_obj.friendly_names[nm] for nm in U_names]
        self.CSurfaces = [self.mesh_obj.friendly_names[nm] for nm in C_names]
        self.log(f"[EDIT] Updated surfaces: T={self.TSurfaces}, U={self.USurfaces}, C={self.CSurfaces}")

    def plot_T_surfaces(self):
        try:
            self.plotter.close()
        except Exception:
            pass
        
        self.plotter = QtInteractor(self)
        self.main_layout.addWidget(self.plotter)

        polydata = pv.PolyData(self.points)
        self.plotter.add_mesh(polydata, color="#52b7ba", show_edges=True, opacity=0.3)

        # Plot control nodes directly
        cn = pv.PolyData()
        cn.points = self.control_nodes
        verts = np.hstack([[1, i] for i in range(len(self.control_nodes))])
        cn.verts = verts
        self.plotter.add_mesh(cn, color='black', point_size=12.0)

        self.plotter.reset_camera()
        self.plotter.render()

        # ===== NEW: modal / spectral controls =====
        from PyQt5.QtWidgets import QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox

        self.form = QFormLayout()

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.back_to_surface_selection)
        self.main_layout.addWidget(self.back_btn)
        
        # k_modes
        n_cn = len(self.control_nodes)
        self.k_modes_spin = QSpinBox()
        if n_cn == 1:
            self.k_modes_spin.setRange(1, n_cn)
            self.k_modes_spin.setValue(min(getattr(self, "k_modes", 6), n_cn))
        else:
            self.k_modes_spin.setRange(1, n_cn-1)
            self.k_modes_spin.setValue(min(getattr(self, "k_modes", 6), n_cn-1))    
        self.form.addRow("Number of modes (k):", self.k_modes_spin)

        # spectral decay p
        self.decay_p_spin = QDoubleSpinBox()
        self.decay_p_spin.setRange(0.1, 6.0)
        self.decay_p_spin.setDecimals(2)
        self.decay_p_spin.setSingleStep(0.1)
        self.decay_p_spin.setValue(getattr(self, "spectral_p", 2.0))
        self.form.addRow("Spectral decay p:", self.decay_p_spin)

        # coefficient amplitude fraction
        self.coeff_frac_spin = QDoubleSpinBox()
        self.coeff_frac_spin.setRange(0.01, 1.0)
        self.coeff_frac_spin.setDecimals(3)
        self.coeff_frac_spin.setSingleStep(0.01)
        self.coeff_frac_spin.setValue(getattr(self, "coeff_frac", 0.15))
        self.form.addRow("Coeff amplitude (frac):", self.coeff_frac_spin)

        # random seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1_000_000, 1_000_000)
        self.seed_spin.setValue(getattr(self, "seed", 0))  # 0 = deterministic default
        self.form.addRow("Random seed (0=deterministic):", self.seed_spin)

        # normal projection
        self.normal_proj_cb = QCheckBox("Project along surface normals")
        self.normal_proj_cb.setChecked(getattr(self, "normal_project", True))
        self.form.addRow(self.normal_proj_cb)
        
        self.rigid_c_cb = QCheckBox("Translate C surfaces rigidly with boundaries")
        # Persist last choice if present
        self.rigid_c_cb.setChecked(getattr(self, "rigid_boundary_translation", False))
        self.form.addRow(self.rigid_c_cb)
        
        
        from PyQt5.QtWidgets import QLineEdit, QLabel
        # ---------- PCA reduced-space controls ----------
        self.use_pca_cb = QCheckBox("Use PCA reduced space")
        default_use_pca = bool(getattr(self.main_window, "use_pca_reduced", False))
        self.use_pca_cb.setChecked(default_use_pca)

        self.pca_M_spin = QSpinBox()
        self.pca_M_spin.setRange(20, 5000)
        self.pca_M_spin.setValue(300)
        self.pca_M_spin.setSingleStep(50)

        self.pca_energy_spin = QDoubleSpinBox()
        self.pca_energy_spin.setDecimals(3)
        self.pca_energy_spin.setRange(0.50, 0.999)
        self.pca_energy_spin.setValue(0.99)
        self.pca_energy_spin.setSingleStep(0.01)

        self.pca_k_spin = QSpinBox()
        self.pca_k_spin.setRange(0, 5000)  # 0 => auto from energy
        self.pca_k_spin.setValue(0)

        pca_row = QHBoxLayout()
        pca_row.addWidget(self.use_pca_cb)
        pca_row.addWidget(QLabel("PCA M"))
        pca_row.addWidget(self.pca_M_spin)
        pca_row.addWidget(QLabel("Energy"))
        pca_row.addWidget(self.pca_energy_spin)
        pca_row.addWidget(QLabel("k (0=auto)"))
        pca_row.addWidget(self.pca_k_spin)

        self.form.addRow(pca_row)

        # Optional bump window controls
        self.bump_enable_cb = QCheckBox("Enable bump window")
        self.bump_enable_cb.setChecked(getattr(self, "bump_enable", False))
        self.form.addRow(self.bump_enable_cb)

        bump_row = QHBoxLayout()
        self.bump_cx = QLineEdit(); self.bump_cx.setPlaceholderText("cx")
        self.bump_cy = QLineEdit(); self.bump_cy.setPlaceholderText("cy")
        self.bump_cz = QLineEdit(); self.bump_cz.setPlaceholderText("cz")
        self.bump_r  = QLineEdit(); self.bump_r.setPlaceholderText("radius")
        bump_row.addWidget(QLabel("Center (x,y,z), Radius:"))
        bump_row.addWidget(self.bump_cx); bump_row.addWidget(self.bump_cy); bump_row.addWidget(self.bump_cz); bump_row.addWidget(self.bump_r)
        self.form.addRow(bump_row)

        self.bump_one_sided_cb = QCheckBox("One-sided (+normal only)")
        self.bump_one_sided_cb.setChecked(getattr(self, "bump_one_sided", False))
        self.form.addRow(self.bump_one_sided_cb)

        # mount the form
        from PyQt5.QtWidgets import QWidget
        self.form_widget = QWidget()
        self.form_widget.setLayout(self.form)
        self.main_layout.addWidget(self.form_widget)

        # Save button
        self.cn_btn = QPushButton("Save Selected Control Nodes")
        self.cn_btn.clicked.connect(self.save_controlnodes)
        self.main_layout.addWidget(self.cn_btn)
        
    def back_to_surface_selection(self):
        """Return to T/U/C surface selection view."""
        try:
            # Remove control node widgets
            self.plotter.close()
            if hasattr(self, "form_widget"):
                self.form_widget.setParent(None)
            if hasattr(self, "cn_btn"):
                self.cn_btn.setParent(None)
            if hasattr(self, "back_btn"):
                self.back_btn.setParent(None)

            # Recreate the mesh + T/U/C surface selection UI
            self._setup_controls()
            self._add_mesh_to_plotter()
        except Exception as e:
            self.log(f"[ERROR] Could not go back: {e}")

    def save_controlnodes(self):
        self.cn_btn.setVisible(False)

        self.k_modes = int(self.k_modes_spin.value())
        self.spectral_p = float(self.decay_p_spin.value())
        self.coeff_frac = float(self.coeff_frac_spin.value())
        self.seed = int(self.seed_spin.value())
        self.normal_project = bool(self.normal_proj_cb.isChecked())
        self.rigid_boundary_translation = bool(self.rigid_c_cb.isChecked())

        # Bump (optional)
        self.bump_enable = bool(self.bump_enable_cb.isChecked())
        self.bump_center = None
        self.bump_radius = None
        self.bump_one_sided = bool(self.bump_one_sided_cb.isChecked())

        if self.bump_enable:
            try:
                cx = float(self.bump_cx.text()); cy = float(self.bump_cy.text()); cz = float(self.bump_cz.text())
                r  = float(self.bump_r.text())
                self.bump_center = (cx, cy, cz)
                self.bump_radius = r
            except Exception:
                self.bump_enable = False
                self.bump_center = None
                self.bump_radius = None

        # PCA settings
        self.use_pca = bool(self.use_pca_cb.isChecked())
        self.pca_train_M = int(self.pca_M_spin.value())
        self.pca_energy = float(self.pca_energy_spin.value())
        k_tmp = int(self.pca_k_spin.value())
        self.pca_k_red = None if k_tmp <= 0 else k_tmp

        # ---- Persist control nodes + normals to output directory ----
        try:
            cn_dir = os.path.join(self.output_dir, "Control Nodes")
            os.makedirs(cn_dir, exist_ok=True)

            cn = np.asarray(self.control_nodes, float).reshape((-1, 3))
            nn = np.asarray(self.control_normals, float).reshape((-1, 3)) if getattr(self, "control_normals", None) is not None else None

            np.save(os.path.join(cn_dir, "control_nodes.npy"), cn)
            if nn is not None:
                np.save(os.path.join(cn_dir, "control_normals.npy"), nn)

            import json
            cn_meta = {
                "k_modes": int(self.k_modes),
                "spectral_p": float(self.spectral_p),
                "coeff_frac": float(self.coeff_frac),
                "seed": int(self.seed),
                "normal_project": bool(self.normal_project),
                "rigid_boundary_translation": bool(self.rigid_boundary_translation),
                "t_patch_scale": None if getattr(self, "t_patch_scale", None) is None else float(self.t_patch_scale),
                "use_pca": bool(self.use_pca),
                "pca_train_M": int(self.pca_train_M),
                "pca_energy": float(self.pca_energy),
                "pca_k_red": (None if self.pca_k_red is None else int(self.pca_k_red)),
            }
            with open(os.path.join(cn_dir, "control_nodes_meta.json"), "w", encoding="utf-8") as f:
                json.dump(cn_meta, f, indent=2)

        except Exception as e:
            self.log(f"[WARN] Failed to save control node files: {e}")
            # still emit ready so UI doesn't deadlock
            self.control_ready.emit()
            return

        # ---- Build PCA cache if enabled ----
        if self.use_pca:
            if nn is None:
                self.log("[WARN] PCA enabled but control_normals missing; disabling PCA.")
                self.use_pca = False
            else:
                pca_dir = os.path.join(cn_dir, "pca")
                os.makedirs(pca_dir, exist_ok=True)
                pca_path = os.path.join(pca_dir, "pca_basis.npz")

                sig = make_signature(
                    control_nodes=cn,
                    control_normals=nn,
                    normal_project=bool(self.normal_project),
                    k_modes=int(self.k_modes),
                    knn=12,
                    spectral_p=float(self.spectral_p),
                    coeff_frac=float(self.coeff_frac),
                    amp_alpha=0.0005,
                    t_patch_scale=getattr(self, "t_patch_scale", None),
                    train_M=int(self.pca_train_M),
                    energy=float(self.pca_energy),
                    k_red=self.pca_k_red,
                )

                need_build = True
                if os.path.exists(pca_path):
                    try:
                        pca_existing = load_pca_basis(pca_path)  # PCABasis object
                        if pca_existing.meta.get("signature") == sig:
                            need_build = False
                    except Exception:
                        need_build = True

                if need_build:
                    from MeshGeneration.controlNodeDisp import getDisplacements

                    X = []
                    for i in range(int(self.pca_train_M)):
                        d = getDisplacements(
                            output_dir=self.output_dir,          # IMPORTANT (not cn_dir)
                            seed=int(self.seed) + i,
                            control_nodes=cn,
                            normals=nn,
                            coeffs=None,
                            k_modes=int(self.k_modes),
                            normal_project=bool(self.normal_project),
                            t_patch_scale=getattr(self, "t_patch_scale", None),
                            amp_alpha=0.02,
                        )
                        X.append(np.asarray(d, float).reshape(-1))
                    X = np.vstack(X)

                    mean, V, sigma, explained = build_pca_basis(
                        X, energy=float(self.pca_energy), k_red=self.pca_k_red
                    )

                    save_pca_basis(
                        pca_path,
                        mean, V, sigma, explained,
                        signature=sig,
                        meta={"signature": sig,
                            "train_M": int(self.pca_train_M),
                            "energy": float(self.pca_energy),
                            "k_red": int(V.shape[1])},
                    )

                # stash for Aeropt export
                self.pca_cache_path = pca_path

        self.control_ready.emit()
        
    @pyqtSlot()
    def plot_control_displacements(self):
        # debounce / non-reentrant guard
        if self._cn_preview_busy:
            # try again shortly (coalesces rapid emits)
            QTimer.singleShot(50, self.plot_control_displacements)
            return
        self._cn_preview_busy = True
        try:
            # --- sanity
            if not hasattr(self, "plotter") or self.plotter is None:
                self.log("[WARN] Plotter not initialised."); return
            if not hasattr(self, "cn_points") or self.cn_points is None or len(self.cn_points) == 0:
                self.log("[WARN] No control-node self.points."); return
            if not hasattr(self, "cn_targets") or self.cn_targets is None or len(self.cn_targets) == 0:
                self.log("[WARN] No displaced control-node targets."); return
            if not hasattr(self, "TSurfaces") or not self.TSurfaces:
                self.log("[WARN] No T-surfaces set."); return

            # --- remove previous preview actors ONLY (do not clear the whole scene)
            for a in getattr(self, "_cn_preview_actors", []):
                try:
                    self.plotter.remove_actor(a, render=False)
                except Exception:
                    pass
            self._cn_preview_actors = []

            # --- build T-surface context (merged once per call)
            t_merge = None
            for sid in self.TSurfaces:
                try:
                    name = self.mesh_obj.get_surface_name(int(sid)) if str(sid).isdigit() else sid
                    surf = self.mesh_obj.get_surface_mesh(name)
                    if surf is None: 
                        continue
                    t_merge = surf.copy() if t_merge is None else t_merge.merge(surf)
                except Exception as e:
                    self.log(f"[WARN] T-surface {sid} load failed: {e}")

            if t_merge is not None and t_merge.n_points > 0:
                act_t = self.plotter.add_mesh(t_merge, color=(0.85,0.85,0.9),
                                              opacity=0.35, show_edges=True, render=False)
                self._cn_preview_actors.append(act_t)

                b = np.array(t_merge.bounds, float)
                ext = np.array([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
                L   = float(np.linalg.norm(ext)) or 1.0
                lmin = float(max(ext.min(), 1e-12))
            else:
                # fallback scale from CNs bbox
                P = np.asarray(self.cn_points, float)
                bmin, bmax = P.min(0), P.max(0)
                ext = bmax - bmin
                L   = float(np.linalg.norm(ext)) or 1.0
                lmin = float(max(ext.min(), 1e-12))

            # --- scaled overlay
            cnP  = np.asarray(self.cn_points, float)
            tgtP = np.asarray(self.cn_targets, float)
            vecs = tgtP - cnP
            dmax = float(np.linalg.norm(vecs, axis=1).max() or 1.0)
            auto_scale = 0.02 * L / dmax if dmax < 0.05*L else 1.0
            scale = float(getattr(self, "preview_scale", auto_scale))

            # lift off the surface to avoid z-fighting
            lift = 1e-4 * L
            cnP_lift  = cnP.copy();  cnP_lift[:,2]  += lift
            tgtP_lift = (cnP + scale*vecs).copy(); tgtP_lift[:,2] += lift

            # spheres for self.points
            r = 0.012 * lmin
            sph_black = pv.Sphere(radius=r); sph_red = pv.Sphere(radius=r)
            cn_poly   = pv.PolyData(cnP_lift)
            tgt_poly  = pv.PolyData(tgtP_lift)
            cn_glyphs  = cn_poly.glyph(geom=sph_black, scale=False)
            tgt_glyphs = tgt_poly.glyph(geom=sph_red,   scale=False)

            act1 = self.plotter.add_mesh(cn_glyphs,  color="black", lighting=False, render=False)
            act2 = self.plotter.add_mesh(tgt_glyphs, color="red",   lighting=False, render=False)
            self._cn_preview_actors += [act1, act2]

            # thick segments for vectors
            pts = np.vstack([cnP_lift, tgtP_lift])
            n   = cnP_lift.shape[0]
            lines = np.hstack([[2, i, i+n] for i in range(n)]).astype(np.int64)
            segs = pv.PolyData(pts, lines=lines)
            act3 = self.plotter.add_mesh(segs, color="red", line_width=3, render_lines_as_tubes=True, render=False)
            self._cn_preview_actors.append(act3)

            # overlay text
            act_txt = self.plotter.add_text(f"Black=orig CNs | Red=displaced | scale×{scale:.1f}", font_size=10)
            self._cn_preview_actors.append(act_txt)

            self.plotter.render()
            self.log(f"[INFO] CN preview: N={len(cnP)}, max|d|={dmax:.3e}, L={L:.3e}, scale={scale:.2f}")
        finally:
            self._cn_preview_busy = False


    def enqueue_plot_control_displacements(self):
        """Thread-safe enqueue from workers."""
        QTimer.singleShot(0, self.plot_control_displacements)

 
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline


    def morphMesh(self):
        # If we have a main_window with a pipeline in HPC mode, delegate
        mw = getattr(self, "main_window", None)
        pipe = getattr(self, "pipeline", None)

        is_hpc = bool(mw and getattr(mw, "run_mode", "") == "HPC")

        if is_hpc:
            # Keep this function usable if called directly (but ideally Run Morph button uses MainWindow.run_morph)
            from PyQt5.QtWidgets import QInputDialog

            n_cases, ok = QInputDialog.getInt(
                self,
                "Morph + Volume on HPC",
                "How many morphed meshes would you like to generate?",
                value=5, min=1, max=500, step=1
            )
            if not ok:
                self.log("[INFO] Morph cancelled by user.")
                return

            if pipe is None:
                self.log("[ERROR] No pipeline attached; cannot submit HPC batch.")
                return

            try:
                jobid = pipe.submit_mesh_batch(
                    n_cases=int(n_cases),
                    do_volume=True,
                    source=getattr(mw, "control_node_source", "mesh"),
                )
                self.log(f"[MORPH] Submitted mesh-batch orchestrator job {jobid}.")
            except Exception as e:
                self.log(f"[ERROR] Failed to submit mesh batch: {e}")
            return

        self.log("[MORPH] Local mode: running local morph workflow (not HPC orchestrator).")


    def _on_morph_finished(self, result):
        self.log("[INFO] Mesh deformation complete.")
        # Re-enable UI controls as before...
        for btn_name in ["reset_btn", "hide_btn", "export_btn"]:
            btn = getattr(self, btn_name, None)
            if btn:
                btn.setEnabled(True)

    def _on_morph_failed(self, msg):
        self.log(f"[ERROR] Morph failed: {msg}")
        # Re-enable controls on failure too
        for btn_name in ["reset_btn", "hide_btn", "export_btn"]:
            btn = getattr(self, btn_name, None)
            if btn:
                btn.setEnabled(True)

    def _setup_morphed(self):
        self.plotter = QtInteractor(self)
        self.main_layout.addWidget(self.plotter)
        self.plotter.reset_camera()
        
        self.reset_btn = QPushButton("Reset Surfaces")
        self.reset_btn.clicked.connect(self.reset_surfaces)
        self.main_layout.addWidget(self.reset_btn)

        self.hide_btn = QPushButton("Hide Surface")
        self.hide_btn.clicked.connect(self.toggle_hide_mode)
        self.main_layout.addWidget(self.hide_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_plotter)
        self.main_layout.addWidget(self.close_btn)
    
    def plotMorphedMesh(self, blocks):
        self._setup_morphed()

        cmap = cm.get_cmap("tab20")
        surface_names = list(blocks.keys())
        self.surface_actors = {}

        for i, name in enumerate(surface_names):
            block = blocks[name]
            if block is None or block.n_cells == 0:
                continue
            color = cmap(i % 20)[:3]
            actor = self.plotter.add_mesh(block, color=color, show_edges=True, pickable=True)
            self.surface_actors[name] = actor

        self.plotter.reset_camera()
        self.plotter.render()
        self.plotter.enable_anti_aliasing()

    def close_plotter(self):
        self.plotter.close()
        self.reset_btn.setVisible(False)
        self.hide_btn.setVisible(False)
        self.close_btn.setVisible(False)
        self.placeholder = QLabel("No mesh loaded")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.placeholder)
        
        
        
class SurfaceEditDialog(QDialog):
    def __init__(self, all_surface_names, T_names, U_names, C_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Surface Selections")
        self.resize(600, 400)

        layout = QHBoxLayout(self)

        # Create list widgets
        self.T_list = QListWidget(); self.T_list.addItems(T_names)
        self.U_list = QListWidget(); self.U_list.addItems(U_names)
        self.C_list = QListWidget(); self.C_list.addItems(C_names)

        layout.addWidget(QLabel("T Surfaces")); layout.addWidget(self.T_list)
        layout.addWidget(QLabel("U Surfaces")); layout.addWidget(self.U_list)
        layout.addWidget(QLabel("C Surfaces")); layout.addWidget(self.C_list)
        
        self.T_list.itemClicked.connect(lambda item: self._focus_on_surface(item.text()))
        self.U_list.itemClicked.connect(lambda item: self._focus_on_surface(item.text()))
        self.C_list.itemClicked.connect(lambda item: self._focus_on_surface(item.text()))

        # Controls
        btns = QVBoxLayout()
        self.rm_btn = QPushButton("Remove Selected")
        self.add_dropdown = QComboBox(); self.add_dropdown.addItems(all_surface_names)
        self.addT = QPushButton("Add to T")
        self.addU = QPushButton("Add to U")
        self.addC = QPushButton("Add to C")
        self.ok_btn = QPushButton("OK")

        btns.addWidget(self.rm_btn)
        btns.addWidget(self.add_dropdown)
        btns.addWidget(self.addT); btns.addWidget(self.addU); btns.addWidget(self.addC)
        btns.addStretch()
        btns.addWidget(self.ok_btn)
        layout.addLayout(btns)

        # Connections
        self.rm_btn.clicked.connect(self.remove_selected)
        self.addT.clicked.connect(lambda: self.add_to_list(self.T_list))
        self.addU.clicked.connect(lambda: self.add_to_list(self.U_list))
        self.addC.clicked.connect(lambda: self.add_to_list(self.C_list))
        self.ok_btn.clicked.connect(self.accept)

    def remove_selected(self):
        for lw in [self.T_list, self.U_list, self.C_list]:
            for item in lw.selectedItems():
                lw.takeItem(lw.row(item))

    def add_to_list(self, lw):
        surf = self.add_dropdown.currentText()
        if surf and not any(lw.item(i).text() == surf for i in range(lw.count())):
            lw.addItem(surf)

    def get_results(self):
        return (
            [self.T_list.item(i).text() for i in range(self.T_list.count())],
            [self.U_list.item(i).text() for i in range(self.U_list.count())],
            [self.C_list.item(i).text() for i in range(self.C_list.count())],
        )
        
    def _focus_on_surface(self, name):
        parent = self.parent()
        if not (hasattr(parent, "mesh_obj") and hasattr(parent, "plotter")):
            return
        try:
            # Reset all surfaces to default (light gray, no edges)
            for sname, actor in parent.surface_actors.items():
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # light gray0
                actor.GetProperty().SetEdgeVisibility(False)

            # Highlight the selected one (red + edges on)
            actor = parent.surface_actors.get(name)
            if actor:
                actor.GetProperty().SetColor(1.0, 0.0, 0.0)   # red
                actor.GetProperty().SetEdgeVisibility(True)

                # Zoom to its bounding box
                mesh = parent.mesh_obj.get_surface_mesh(name)
                parent.plotter.reset_camera(mesh)

            parent.plotter.render()
        except Exception as e:
            print(f"[DEBUG] Failed to highlight {name}: {e}")
            
    def reset_surfaces(self):
        parent = self.parent()
        if hasattr(parent, "surface_actors"):
            for actor in parent.surface_actors.values():
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # light gray
                actor.GetProperty().SetEdgeVisibility(False)
            parent.plotter.reset_camera()
            parent.plotter.render()
            
            
            
class ControlNodeSelectionDialog(QDialog):
    """
    Lets the user choose either:
      - Auto: pick N control nodes automatically
      - Manual: pick N control nodes by clicking points on the T patch
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Node Selection")
        self.resize(420, 180)

        layout = QVBoxLayout(self)

        self.mode_auto = QCheckBox("Auto-select control nodes (random)")
        self.mode_manual = QCheckBox("Manually select control nodes (click points on T patch)")
        self.mode_auto.setChecked(True)
        self.mode_manual.setChecked(False)

        # make them mutually exclusive
        self.mode_auto.stateChanged.connect(lambda s: self._sync("auto"))
        self.mode_manual.stateChanged.connect(lambda s: self._sync("manual"))

        layout.addWidget(self.mode_auto)
        layout.addWidget(self.mode_manual)

        row = QHBoxLayout()
        row.addWidget(QLabel("Number of control nodes (N):"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 5000)
        self.n_spin.setValue(6)
        row.addWidget(self.n_spin)
        row.addStretch(1)
        layout.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _sync(self, which):
        if which == "auto" and self.mode_auto.isChecked():
            self.mode_manual.setChecked(False)
        if which == "manual" and self.mode_manual.isChecked():
            self.mode_auto.setChecked(False)
        # ensure at least one is checked
        if not self.mode_auto.isChecked() and not self.mode_manual.isChecked():
            self.mode_auto.setChecked(True)

    def get_choice(self):
        mode = "manual" if self.mode_manual.isChecked() else "auto"
        return mode, int(self.n_spin.value())
