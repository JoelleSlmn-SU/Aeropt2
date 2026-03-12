import os, sys
import numpy as np
import matplotlib.cm as cm
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, pyqtSlot, QMetaObject, Q_ARG, QObject, QEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy, QDialog, QComboBox, QSpinBox, QFormLayout,
    QFileDialog, QLineEdit, QHBoxLayout, QInputDialog, QCheckBox, QListWidget, QDialogButtonBox, QTableWidget
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
    build_pca_cache
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
        self.global_modes_selected = False

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
            "form_widget","back_btn","debug_checkbox","open_debug_btn"
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
        from PyQt5.QtWidgets import QTableWidgetItem, QDockWidget, QAbstractItemView

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
                actor = self.plotter.add_mesh(
                    mesh,
                    color=color,
                    show_edges=True,
                    pickable=True,
                    show_scalar_bar=True,
                    scalar_bar_args=sargs
                )
                self.surface_actors[name] = actor
            except Exception as e:
                self.log(f"Failed to plot surface '{name}': {e}")

        # ---- NEW: show axes + orientation marker ----
        try:
            self.plotter.add_axes(
                line_width=2,
                labels_off=False
            )
        except Exception as e:
            self.log(f"[WARN] Failed to add axes: {e}")

        # optional but very useful:
        try:
            self.plotter.show_bounds(
                grid='back',
                location='outer',
                all_edges=True,
                xtitle='X',
                ytitle='Y',
                ztitle='Z'
            )
        except Exception as e:
            self.log(f"[WARN] Failed to show bounds: {e}")

        self.plotter.set_visible(True) if hasattr(self.plotter, "set_visible") else None
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
        
    def load_saved_control_nodes(self, control_nodes_path, control_normals_path=None):
        if not control_nodes_path or not os.path.exists(control_nodes_path):
            self.log(f"[ERROR] Control nodes file not found: {control_nodes_path}")
            return

        try:
            self.control_nodes = np.asarray(np.load(control_nodes_path), float)
        except Exception as e:
            self.log(f"[ERROR] Failed to load control nodes from '{control_nodes_path}': {e}")
            return

        if self.control_nodes.ndim != 2 or self.control_nodes.shape[1] != 3:
            self.log(f"[ERROR] Loaded control nodes must have shape (N, 3), got {self.control_nodes.shape}")
            return

        # Keep the T-surface points as the parent cloud for plotting / normal mapping
        if not hasattr(self, "points") or self.points is None:
            self.log("[ERROR] T-surface points are not available.")
            return

        try:
            if control_normals_path and os.path.exists(control_normals_path):
                self.control_normals = np.asarray(np.load(control_normals_path), float)
                if self.control_normals.shape != self.control_nodes.shape:
                    raise ValueError(
                        f"control_normals shape {self.control_normals.shape} "
                        f"does not match control_nodes shape {self.control_nodes.shape}"
                    )
                self.log(f"[INFO] Loaded control normals from: {control_normals_path}")
            else:
                surf_normals = _surface_normals(self.points, knn=16)
                self.control_normals = _map_normals_to_control(
                    self.control_nodes,
                    self.points,
                    surf_normals,
                    k=12
                )
                self.log("[INFO] No control_normals.npy provided; normals mapped from T-surface points.")
        except Exception as e:
            self.log(f"[ERROR] Failed to create/load control normals: {e}")
            return

        try:
            pts = np.asarray(self.points, float)
            d = pts.max(axis=0) - pts.min(axis=0)
            self.t_patch_scale = float(np.linalg.norm(d))
            self.log(f"[INFO] T-patch scale (loaded CNs) = {self.t_patch_scale:.6g}")
        except Exception as e:
            self.t_patch_scale = None
            self.log(f"[WARN] Failed to compute T-patch scale: {e}")

        self.log(f"[INFO] Loaded {len(self.control_nodes)} control nodes from: {control_nodes_path}")

        try:
            self.plotter.close()
        except Exception:
            pass

        self.plot_T_surfaces()
        
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
        
    def _on_global_modes_toggled(self, checked):
        if checked:
            existing = getattr(self, "global_mode_config", [])
            dlg = GlobalModesDialog(existing=existing, parent=self)
            if dlg.exec_() == QDialog.Accepted:
                self.global_mode_config = dlg.get_selected_modes()
                self.log(f"[INFO] Global modes selected: {self.global_mode_config}")
            else:
                # user cancelled -> undo checkbox
                self.global_modes_cb.blockSignals(True)
                self.global_modes_cb.setChecked(False)
                self.global_modes_cb.blockSignals(False)
                self.global_modes_selected = False
                self.global_mode_config = []
        else:
            self.global_modes_selected = False
            self.global_mode_config = []

    def edit_global_modes(self):
        existing = getattr(self, "global_mode_config", [])
        dlg = GlobalModesDialog(existing=existing, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.global_mode_config = dlg.get_selected_modes()
            self.log(f"[INFO] Global modes updated: {self.global_mode_config}")
            
    def _clear_param_widgets(self):
        for attr in [
            "back_btn", "save_btn", "edit_global_modes_btn",
            "global_modes_cb", "global_only_cb", "use_local_modes_cb",
            "k_modes_spin", "decay_p_spin", "coeff_frac_spin", "seed_spin",
            "normal_project_cb", "vector_mode_combo", "frame_knn_spin",
            "rigid_translation_cb", "amp_alpha_spin",
            "use_pca_cb", "pca_train_spin", "pca_energy_spin", "pca_k_red_spin",
            "direct_mode_combo", "direct_amp_alpha_spin"
        ]:
            try:
                w = getattr(self, attr, None)
                if w is not None:
                    w.setParent(None)
            except Exception:
                pass

        try:
            if hasattr(self, "form_container") and self.form_container is not None:
                self.form_container.setParent(None)
        except Exception:
            pass
        
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

        choice = dlg.get_choice()
        mode = choice["selection_mode"]
        num_input = choice["num_nodes"]

        # store the selected parameterisation family for later UI branching
        self.parameterisation_method = choice["parameterisation_method"]
        self.control_node_selection_mode = mode
        self.loaded_control_nodes_path = choice["control_nodes_path"]
        self.loaded_control_normals_path = choice["control_normals_path"]

        # --- Build preview PolyData from T surface NAMES ---
        t_mesh = None
        for nm in getattr(self, "T_names", []):
            surf = self.mesh_obj.get_surface_mesh(nm)
            if surf is None or surf.n_cells == 0:
                continue
            t_mesh = surf.copy() if t_mesh is None else t_mesh.merge(surf)

        if t_mesh is None or t_mesh.n_cells == 0:
            self.log("[WARN] No T-surface cells available for control-node selection.")
            return

        # picker-friendly mesh
        t_mesh = t_mesh.extract_surface().triangulate().clean()

        # save preview mesh for downstream / auto selection path
        output_path = os.path.join(self.output_dir, "surfaces", "output.vtk")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        t_mesh.save(output_path)

        # keep the point cloud available for all branches, including loaded CNs
        self.points = np.asarray(t_mesh.points, float)

        if mode == "auto":
            self.auto_select_control_nodes(output_path, num_input)
        elif mode == "manual":
            self.manual_select_control_nodes(t_mesh, num_input)
        elif mode == "load":
            self.load_saved_control_nodes(
                choice["control_nodes_path"],
                choice["control_normals_path"]
            )
        else:
            self.log(f"[ERROR] Unknown control-node selection mode: {mode}")


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

        self._clear_param_widgets()

        self.plotter = QtInteractor(self)
        self.main_layout.addWidget(self.plotter)

        # ----------------------------
        # base cloud / T patch preview
        # ----------------------------
        polydata = pv.PolyData(self.points)
        self.plotter.add_mesh(polydata, color="#52b7ba", show_edges=True, opacity=0.3)

        # plot control nodes
        cn = pv.PolyData()
        cn.points = self.control_nodes
        verts = np.hstack([[1, i] for i in range(len(self.control_nodes))])
        cn.verts = verts
        self.plotter.add_mesh(cn, color="black", point_size=12.0)

        # axes
        try:
            self.plotter.add_axes(line_width=2, labels_off=False)
        except Exception:
            pass

        try:
            self.plotter.show_bounds(
                grid="back",
                location="outer",
                all_edges=True,
                xtitle="X",
                ytitle="Y",
                ztitle="Z",
            )
        except Exception:
            pass

        self.plotter.reset_camera()
        self.plotter.render()

        # ----------------------------
        # defaults
        # ----------------------------
        if not hasattr(self, "parameterisation_method"):
            self.parameterisation_method = "modal"
        
        from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox

        # shared buttons / form
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.back_to_surface_selection)
        self.main_layout.addWidget(self.back_btn)

        self.form = QFormLayout()
        self.form_container = QWidget()
        self.form_container.setLayout(self.form)

        # show chosen family (read-only label)
        method_label = QLabel(
            "Direct Control Node Displacement"
            if self.parameterisation_method == "direct"
            else "Modal Parameterisation"
        )
        self.form.addRow("Parameterisation:", method_label)

        # ============================
        # DIRECT PARAMETERISATION UI
        # ============================
        if self.parameterisation_method == "direct":

            # subtype
            self.direct_mode_combo = QComboBox()
            self.direct_mode_combo.addItems([
                "Cartesian (x,y,z)",
                "Normal displacement (d·n)",
            ])
            prev_direct_mode = getattr(self, "direct_parameterisation_subtype", "xyz")
            self.direct_mode_combo.setCurrentIndex(0 if prev_direct_mode == "xyz" else 1)
            self.form.addRow("Direct displacement type:", self.direct_mode_combo)

            # amplitude scale
            self.direct_amp_alpha_spin = QDoubleSpinBox()
            self.direct_amp_alpha_spin.setRange(1e-6, 10.0)
            self.direct_amp_alpha_spin.setDecimals(6)
            self.direct_amp_alpha_spin.setSingleStep(0.001)
            self.direct_amp_alpha_spin.setValue(float(getattr(self, "amp_alpha", 0.01)))
            self.form.addRow("Amplitude scale:", self.direct_amp_alpha_spin)

            # rigid translation
            self.rigid_translation_cb = QCheckBox("Enable rigid boundary translation")
            self.rigid_translation_cb.setChecked(getattr(self, "rigid_boundary_translation", False))
            self.form.addRow(self.rigid_translation_cb)

            # optional future extension note
            direct_note = QLabel(
                "Direct mode uses per-control-node displacement variables.\n"
                "Modal/global/PCA settings are hidden for this parameterisation."
            )
            direct_note.setWordWrap(True)
            self.form.addRow(direct_note)

        # ============================
        # MODAL PARAMETERISATION UI
        # ============================
        else:
            from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox

            n_cn = len(self.control_nodes)

            # number of modes
            self.k_modes_spin = QSpinBox()
            if n_cn == 1:
                self.k_modes_spin.setRange(1, n_cn)
                self.k_modes_spin.setValue(min(getattr(self, "k_modes", 6), n_cn))
            else:
                self.k_modes_spin.setRange(1, max(1, n_cn - 1))
                self.k_modes_spin.setValue(min(getattr(self, "k_modes", 6), max(1, n_cn - 1)))
            self.form.addRow("Number of modes (k):", self.k_modes_spin)

            # spectral decay
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

            # seed
            self.seed_spin = QSpinBox()
            self.seed_spin.setRange(0, 10**9)
            self.seed_spin.setValue(getattr(self, "seed", 0))
            self.form.addRow("Random seed:", self.seed_spin)

            # amp alpha
            self.amp_alpha_spin = QDoubleSpinBox()
            self.amp_alpha_spin.setRange(1e-6, 10.0)
            self.amp_alpha_spin.setDecimals(6)
            self.amp_alpha_spin.setSingleStep(0.001)
            self.amp_alpha_spin.setValue(float(getattr(self, "amp_alpha", 0.01)))
            self.form.addRow("Amplitude scale:", self.amp_alpha_spin)

            # normal projection
            self.normal_project_cb = QCheckBox("Project local modes along normals")
            self.normal_project_cb.setChecked(getattr(self, "normal_project", True))
            self.form.addRow(self.normal_project_cb)

            # vector mode for non-normal case
            self.vector_mode_combo = QComboBox()
            self.vector_mode_combo.addItems([
                "Local frame (t1,t2,n)",
                "Cartesian (x,y,z)",
            ])
            prev_vm = getattr(self, "vector_mode", "local_frame")
            self.vector_mode_combo.setCurrentIndex(0 if prev_vm == "local_frame" else 1)
            self.form.addRow("Vector mode (if normals off):", self.vector_mode_combo)

            # local frame knn
            self.frame_knn_spin = QSpinBox()
            self.frame_knn_spin.setRange(3, 200)
            self.frame_knn_spin.setValue(int(getattr(self, "frame_knn", 16)))
            self.form.addRow("Local-frame kNN:", self.frame_knn_spin)

            # local/global toggles
            self.use_local_modes_cb = QCheckBox("Use Laplacian local modes")
            self.use_local_modes_cb.setChecked(getattr(self, "use_local_modes", True))
            self.form.addRow(self.use_local_modes_cb)

            self.global_modes_cb = QCheckBox("Use global aerodynamic modes")
            self.global_modes_cb.setChecked(getattr(self, "global_modes_selected", False))
            self.global_modes_cb.toggled.connect(self._on_global_modes_toggled)
            self.form.addRow(self.global_modes_cb)

            self.global_only_cb = QCheckBox("Global modes only")
            self.global_only_cb.setChecked(getattr(self, "global_only", False))
            self.form.addRow(self.global_only_cb)

            self.edit_global_modes_btn = QPushButton("Edit global modes")
            self.edit_global_modes_btn.clicked.connect(self.edit_global_modes)
            self.form.addRow(self.edit_global_modes_btn)

            # PCA
            self.use_pca_cb = QCheckBox("Use PCA-reduced basis")
            self.use_pca_cb.setChecked(getattr(self, "use_pca", False))
            self.form.addRow(self.use_pca_cb)

            self.pca_train_spin = QSpinBox()
            self.pca_train_spin.setRange(10, 100000)
            self.pca_train_spin.setValue(int(getattr(self, "pca_train_M", 300)))
            self.form.addRow("PCA training samples:", self.pca_train_spin)

            self.pca_energy_spin = QDoubleSpinBox()
            self.pca_energy_spin.setRange(0.5, 0.9999)
            self.pca_energy_spin.setDecimals(4)
            self.pca_energy_spin.setSingleStep(0.01)
            self.pca_energy_spin.setValue(float(getattr(self, "pca_energy", 0.99)))
            self.form.addRow("PCA energy target:", self.pca_energy_spin)

            self.pca_k_red_spin = QSpinBox()
            self.pca_k_red_spin.setRange(0, 100000)
            self.pca_k_red_spin.setValue(int(getattr(self, "pca_k_red", 0) or 0))
            self.form.addRow("PCA fixed reduced k (0=auto):", self.pca_k_red_spin)

            # rigid translation
            self.rigid_translation_cb = QCheckBox("Enable rigid boundary translation")
            self.rigid_translation_cb.setChecked(getattr(self, "rigid_boundary_translation", False))
            self.form.addRow(self.rigid_translation_cb)

            # enable/disable vector widgets based on normal projection
            def _sync_modal_widgets():
                normals_on = self.normal_project_cb.isChecked()
                self.vector_mode_combo.setEnabled(not normals_on)
                self.frame_knn_spin.setEnabled(not normals_on)

                global_only = self.global_only_cb.isChecked()
                self.use_local_modes_cb.setEnabled(not global_only)
                if global_only:
                    self.use_local_modes_cb.setChecked(False)

            self.normal_project_cb.toggled.connect(_sync_modal_widgets)
            self.global_only_cb.toggled.connect(_sync_modal_widgets)
            _sync_modal_widgets()

        # ----------------------------
        # save button
        # ----------------------------
        self.save_btn = QPushButton("Save Control Nodes / Basis")
        self.save_btn.clicked.connect(self.save_controlnodes)
        self.main_layout.addWidget(self.form_container)
        self.main_layout.addWidget(self.save_btn)
        
    def back_to_surface_selection(self):
        """Return to T/U/C surface selection view."""
        try:
            # Remove control node widgets
            self.plotter.close()
            if hasattr(self, "form_widget"):
                self.form_widget.setParent(None)
            if hasattr(self, "back_btn"):
                self.back_btn.setParent(None)

            # Recreate the mesh + T/U/C surface selection UI
            self._setup_controls()
            self._add_mesh_to_plotter()
        except Exception as e:
            self.log(f"[ERROR] Could not go back: {e}")

    def save_controlnodes(self):

        # --------------------------------------------------
        # top-level parameterisation family
        # --------------------------------------------------
        self.save_btn.setVisible(False)
        
        self.parameterisation_method = getattr(self, "parameterisation_method", "modal")

        # defaults shared by both families
        self.global_mode_config = getattr(self, "global_mode_config", [])
        self.bump_enable = False
        self.bump_center = None
        self.bump_radius = None
        self.bump_one_sided = None

        # optional body-frame mapping for globals
        # for now default to world xyz unless you add a dedicated UI later
        self.basis_axes = getattr(self, "basis_axes", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # keep track of how CNs were obtained
        self.control_node_selection_mode = getattr(self, "control_node_selection_mode", "auto")
        self.loaded_control_nodes_path = getattr(self, "loaded_control_nodes_path", None)
        self.loaded_control_normals_path = getattr(self, "loaded_control_normals_path", None)

        # --------------------------------------------------
        # DIRECT CONTROL-NODE PARAMETERISATION
        # --------------------------------------------------
        if self.parameterisation_method == "direct":
            self.direct_parameterisation_subtype = (
                "xyz" if self.direct_mode_combo.currentIndex() == 0 else "normal"
            )

            self.amp_alpha = float(self.direct_amp_alpha_spin.value())
            self.rigid_boundary_translation = bool(self.rigid_translation_cb.isChecked())

            # disable modal-only settings
            self.k_modes = 0
            self.spectral_p = None
            self.coeff_frac = None
            self.seed = 0
            self.normal_project = None
            self.vector_mode = None
            self.frame_knn = None

            self.use_local_modes = False
            self.global_modes_selected = False
            self.global_only = False

            self.use_pca = False
            self.pca_train_M = None
            self.pca_energy = None
            self.pca_k_red = None
            self.pca_k_final = None
            self.pca_cache_path = None

        # --------------------------------------------------
        # MODAL PARAMETERISATION
        # --------------------------------------------------
        else:
            self.direct_parameterisation_subtype = None

            self.k_modes = int(self.k_modes_spin.value())
            self.spectral_p = float(self.decay_p_spin.value())
            self.coeff_frac = float(self.coeff_frac_spin.value())
            self.seed = int(self.seed_spin.value())
            self.amp_alpha = float(self.amp_alpha_spin.value())

            self.normal_project = bool(self.normal_project_cb.isChecked())
            self.vector_mode = "local_frame" if self.vector_mode_combo.currentIndex() == 0 else "xyz"
            self.frame_knn = int(self.frame_knn_spin.value())

            self.use_local_modes = bool(self.use_local_modes_cb.isChecked())
            self.global_modes_selected = bool(self.global_modes_cb.isChecked())
            self.global_only = bool(self.global_only_cb.isChecked())
            self.rigid_boundary_translation = bool(self.rigid_translation_cb.isChecked())

            # if global-only is checked, force local modes off
            if self.global_only:
                self.use_local_modes = False

            self.use_pca = bool(self.use_pca_cb.isChecked())
            self.pca_train_M = int(self.pca_train_spin.value())
            self.pca_energy = float(self.pca_energy_spin.value())
            k_tmp = int(self.pca_k_red_spin.value())
            self.pca_k_red = None if k_tmp <= 0 else k_tmp
            self.pca_k_final = getattr(self, "pca_k_final", None)
            self.pca_cache_path = getattr(self, "pca_cache_path", None)

        # --------------------------------------------------
        # persist control nodes + normals to output directory
        # --------------------------------------------------
        try:
            cn_dir = os.path.join(self.output_dir, "Control Nodes")
            os.makedirs(cn_dir, exist_ok=True)

            cn_path = os.path.join(cn_dir, "control_nodes.npy")
            np.save(cn_path, np.asarray(self.control_nodes, float))
            self.log(f"[INFO] Saved control nodes -> {cn_path}")

            if getattr(self, "control_normals", None) is not None:
                cn_normals_path = os.path.join(cn_dir, "control_normals.npy")
                np.save(cn_normals_path, np.asarray(self.control_normals, float))
                self.log(f"[INFO] Saved control normals -> {cn_normals_path}")

            meta = {
                "parameterisation_method": self.parameterisation_method,
                "direct_parameterisation_subtype": self.direct_parameterisation_subtype,
                "selection_mode": self.control_node_selection_mode,
                "loaded_control_nodes_path": self.loaded_control_nodes_path,
                "loaded_control_normals_path": self.loaded_control_normals_path,
                "k_modes": self.k_modes,
                "spectral_p": self.spectral_p,
                "coeff_frac": self.coeff_frac,
                "seed": self.seed,
                "normal_project": self.normal_project,
                "vector_mode": self.vector_mode,
                "frame_knn": self.frame_knn,
                "use_local_modes": self.use_local_modes,
                "global_modes": self.global_modes_selected,
                "global_only": self.global_only,
                "global_mode_config": self.global_mode_config,
                "basis_axes": self.basis_axes,
                "t_patch_scale": None if getattr(self, "t_patch_scale", None) is None else float(self.t_patch_scale),
                "amp_alpha": float(self.amp_alpha),
                "rigid_translation": bool(self.rigid_boundary_translation),
                "use_pca": self.use_pca,
                "pca_train_M": self.pca_train_M,
                "pca_energy": self.pca_energy,
                "pca_k_red": self.pca_k_red,
                "pca_k_final": self.pca_k_final,
                "pca_cache_path": self.pca_cache_path,
            }

            meta_path = os.path.join(cn_dir, "control_nodes_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            self.log(f"[INFO] Saved control-node metadata -> {meta_path}")

        except Exception as e:
            self.log(f"[WARN] Failed to save control nodes/metadata: {e}")

        # --------------------------------------------------
        # PCA cache generation only for modal family
        # --------------------------------------------------
        if self.parameterisation_method == "modal" and self.use_pca:
            try:
                self.log("[PCA] Building PCA cache...")

                cache = build_pca_cache(
                    output_dir=self.output_dir,
                    control_nodes=np.asarray(self.control_nodes, float),
                    normals=np.asarray(self.control_normals, float),
                    k_modes=int(self.k_modes),
                    M=int(self.pca_train_M),
                    energy=float(self.pca_energy),
                    k_red=self.pca_k_red,
                    seed=int(self.seed),
                    normal_project=bool(self.normal_project),
                    t_patch_scale=self.t_patch_scale,
                    amp_alpha=float(self.amp_alpha),
                    vector_mode=self.vector_mode,
                    frame_knn=int(self.frame_knn) if self.frame_knn is not None else 12,
                    global_modes=bool(self.global_modes_selected),
                    global_mode_config=self.global_mode_config,
                    basis_axes=self.basis_axes,
                )
                self.pca_cache_path = cache["cache_path"]
                self.pca_k_final = int(cache["k_red"])
                self.log(f"[PCA] Cache built: {self.pca_cache_path} (k_red={self.pca_k_final})")
            except Exception as e:
                self.log(f"[ERROR] PCA cache build failed: {e}")
                self.use_pca = False
                self.pca_cache_path = None
                self.pca_k_final = None

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
    Lets the user choose:
      - how to obtain control nodes:
          * auto
          * manual
          * load saved .npy
      - which parameterisation family to use:
          * direct
          * modal
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Node Selection")
        self.resize(560, 280)

        layout = QVBoxLayout(self)

        # ----------------------------
        # selection mode
        # ----------------------------
        self.mode_auto = QCheckBox("Auto-select control nodes (random)")
        self.mode_manual = QCheckBox("Manually select control nodes (click points on T patch)")
        self.mode_load = QCheckBox("Load saved control nodes (.npy)")
        self.mode_auto.setChecked(True)

        self.mode_auto.stateChanged.connect(lambda s: self._sync("auto"))
        self.mode_manual.stateChanged.connect(lambda s: self._sync("manual"))
        self.mode_load.stateChanged.connect(lambda s: self._sync("load"))

        layout.addWidget(self.mode_auto)
        layout.addWidget(self.mode_manual)
        layout.addWidget(self.mode_load)

        # ----------------------------
        # node count
        # ----------------------------
        row_n = QHBoxLayout()
        row_n.addWidget(QLabel("Number of control nodes (N):"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 5000)
        self.n_spin.setValue(6)
        row_n.addWidget(self.n_spin)
        row_n.addStretch(1)
        layout.addLayout(row_n)

        # ----------------------------
        # load saved files
        # ----------------------------
        row_cn = QHBoxLayout()
        row_cn.addWidget(QLabel("Control nodes file:"))
        self.cn_path_edit = QLineEdit()
        self.cn_path_edit.setPlaceholderText("Path to control_nodes.npy")
        self.cn_browse_btn = QPushButton("Browse...")
        self.cn_browse_btn.clicked.connect(self._browse_cn_file)
        row_cn.addWidget(self.cn_path_edit)
        row_cn.addWidget(self.cn_browse_btn)
        layout.addLayout(row_cn)

        row_normals = QHBoxLayout()
        row_normals.addWidget(QLabel("Control normals file (optional):"))
        self.normals_path_edit = QLineEdit()
        self.normals_path_edit.setPlaceholderText("Optional path to control_normals.npy")
        self.normals_browse_btn = QPushButton("Browse...")
        self.normals_browse_btn.clicked.connect(self._browse_normals_file)
        row_normals.addWidget(self.normals_path_edit)
        row_normals.addWidget(self.normals_browse_btn)
        layout.addLayout(row_normals)

        # ----------------------------
        # parameterisation family
        # ----------------------------
        row_param = QHBoxLayout()
        row_param.addWidget(QLabel("Parameterisation method:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems([
            "Direct Control Node Displacement",
            "Modal Parameterisation",
        ])
        row_param.addWidget(self.param_combo)
        row_param.addStretch(1)
        layout.addLayout(row_param)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._update_enabled_state()

    def _sync(self, which):
        if which == "auto" and self.mode_auto.isChecked():
            self.mode_manual.setChecked(False)
            self.mode_load.setChecked(False)
        elif which == "manual" and self.mode_manual.isChecked():
            self.mode_auto.setChecked(False)
            self.mode_load.setChecked(False)
        elif which == "load" and self.mode_load.isChecked():
            self.mode_auto.setChecked(False)
            self.mode_manual.setChecked(False)

        if not self.mode_auto.isChecked() and not self.mode_manual.isChecked() and not self.mode_load.isChecked():
            self.mode_auto.setChecked(True)

        self._update_enabled_state()

    def _update_enabled_state(self):
        load_mode = self.mode_load.isChecked()
        self.n_spin.setEnabled(not load_mode)
        self.cn_path_edit.setEnabled(load_mode)
        self.cn_browse_btn.setEnabled(load_mode)
        self.normals_path_edit.setEnabled(load_mode)
        self.normals_browse_btn.setEnabled(load_mode)

    def _browse_cn_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select control_nodes.npy",
            "",
            "NumPy files (*.npy)"
        )
        if path:
            self.cn_path_edit.setText(path)

    def _browse_normals_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select control_normals.npy",
            "",
            "NumPy files (*.npy)"
        )
        if path:
            self.normals_path_edit.setText(path)

    def get_choice(self):
        if self.mode_load.isChecked():
            selection_mode = "load"
        elif self.mode_manual.isChecked():
            selection_mode = "manual"
        else:
            selection_mode = "auto"

        param_text = self.param_combo.currentText()
        parameterisation_method = (
            "direct"
            if param_text == "Direct Control Node Displacement"
            else "modal"
        )

        return {
            "selection_mode": selection_mode,
            "num_nodes": int(self.n_spin.value()),
            "control_nodes_path": self.cn_path_edit.text().strip() or None,
            "control_normals_path": self.normals_path_edit.text().strip() or None,
            "parameterisation_method": parameterisation_method,
        }


class GlobalModesDialog(QDialog):
    """
    Lets the user build a list of global modes, each with:
      - mode type
      - direction
    """
    def __init__(self, existing=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Global Modes")
        self.resize(520, 350)

        self.mode_rows = []

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Enable", "Mode Type", "Direction"])
        layout.addWidget(self.table)

        row_btns = QHBoxLayout()
        add_btn = QPushButton("+ Add Mode")
        del_btn = QPushButton("– Remove Selected")
        add_btn.clicked.connect(self.add_row)
        del_btn.clicked.connect(self.remove_selected_rows)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(del_btn)
        row_btns.addStretch(1)
        layout.addLayout(row_btns)

        defaults = existing if existing else [
            {"type": "stretch", "direction": "x"},
            {"type": "flatten", "direction": "z"},
        ]
        for item in defaults:
            self.add_row(item)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def add_row(self, data=None):
        data = data or {"type": "stretch", "direction": "x"}

        row = self.table.rowCount()
        self.table.insertRow(row)

        enabled_cb = QCheckBox()
        enabled_cb.setChecked(True)

        mode_combo = QComboBox()
        mode_combo.addItems([
            "stretch",
            "flatten",
            "bulge",
            "camber",
            "twist",
            "taper",
            "bend"
        ])
        mode_combo.setCurrentText(data.get("type", "stretch"))

        dir_combo = QComboBox()
        dir_combo.addItems(["x", "y", "z"])
        dir_combo.setCurrentText(data.get("direction", "x"))

        self.table.setCellWidget(row, 0, enabled_cb)
        self.table.setCellWidget(row, 1, mode_combo)
        self.table.setCellWidget(row, 2, dir_combo)

    def remove_selected_rows(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def get_selected_modes(self):
        out = []
        for row in range(self.table.rowCount()):
            enabled_cb = self.table.cellWidget(row, 0)
            mode_combo = self.table.cellWidget(row, 1)
            dir_combo = self.table.cellWidget(row, 2)

            if enabled_cb is not None and enabled_cb.isChecked():
                out.append({
                    "type": mode_combo.currentText(),
                    "direction": dir_combo.currentText()
                })
        return out