# geom_gui.py (GeometryPanel.__init__)
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QDialogButtonBox, QDialog, QSlider, QListWidget,
                             QDockWidget, QAbstractItemView, QHBoxLayout, QPushButton, QMessageBox, QFileDialog, QPlainTextEdit, QLineEdit,
                             QDoubleSpinBox, QSpinBox, QListWidgetItem, QListWidget, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import os, sys
import numpy as np
sys.path.append(os.path.dirname("FileRW"))
from FileRW.StpFile import OCCViewer
from FileRW.BacFile import BacFile
from FileRW.BppFile import BppFile
from GUI.mesh_gui import SurfaceEditDialog
from ShapeParameterization.surfaceFitting import farthest_point_sampling
from ShapeParameterization.cadd_ffd import CADFFDManager, FaceRole

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                             QListWidget, QAbstractItemView, QPushButton, 
                             QTabWidget, QPlainTextEdit, QFileDialog, QMessageBox, QInputDialog)

class BacFileDialog(QDialog):
    def __init__(self, base_name, output_dir, dat_path=None, cad_path=None, parent=None):
        super().__init__(parent)
        self.base_name = base_name
        self.output_dir = output_dir
        self.dat_path = dat_path     # may be None if not generated yet
        self.cad_path = cad_path     # STEP/IGES path for fromGeometry
        self.setWindowTitle("Generate BAC File")
        self.resize(600, 400)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ---------- Automatic tab ----------
        auto_tab = QVBoxLayout()
        auto_widget = QWidget(); auto_widget.setLayout(auto_tab)

        q_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0); self.slider.setMaximum(100); self.slider.setValue(50)
        q_layout.addWidget(QLabel("Coarse"))
        q_layout.addWidget(self.slider)
        q_layout.addWidget(QLabel("Fine"))
        auto_tab.addLayout(q_layout)

        auto_tab.addWidget(QLabel("Select focus surfaces:"))
        self.surface_list = QListWidget()
        self.surface_list.setSelectionMode(QAbstractItemView.MultiSelection)

        # Try populate surfaces if DAT is already available
        if self.dat_path and os.path.exists(self.dat_path):
            try:
                from FileRW.DatFile import DatFile
                df = DatFile(self.dat_path)
                s4_blocks = df._parse_surfaces4(df.lines, df.idx["Surfaces4"])
                for blk in s4_blocks:
                    sid = blk["header_vals"][0]
                    self.surface_list.addItem(f"Surface {sid}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to parse DAT:\n{e}")

        # If no DAT but CAD exists, fallback: list faces by traversing STEP geometry
        elif self.cad_path and os.path.exists(self.cad_path):
            try:
                from OCP.STEPControl import STEPControl_Reader
                from OCP.IFSelect import IFSelect_RetDone
                from OCP.TopExp import TopExp_Explorer
                from OCP.TopAbs import TopAbs_FACE
                rdr = STEPControl_Reader()
                if rdr.ReadFile(self.cad_path) == IFSelect_RetDone:
                    rdr.TransferRoots()
                    shape = rdr.OneShape()
                    exp = TopExp_Explorer(shape, TopAbs_FACE)
                    face_id = 0
                    while exp.More():
                        face_id += 1
                        self.surface_list.addItem(f"Face {face_id}")
                        exp.Next()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to parse CAD for faces:\n{e}")

        auto_tab.addWidget(self.surface_list)
        self.tabs.addTab(auto_widget, "Automatic")

        # ---------- Manual tab ----------
        man_tab = QVBoxLayout()
        man_widget = QWidget(); man_widget.setLayout(man_tab)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("# BAC file content here…")
        man_tab.addWidget(self.editor)

        load_btn = QPushButton("Load BAC…")
        load_btn.clicked.connect(self._load_bac)
        man_tab.addWidget(load_btn)

        self.tabs.addTab(man_widget, "Manual")

        # ---------- Submit ----------
        btn = QPushButton("Generate")
        btn.clicked.connect(self._generate)
        layout.addWidget(btn)

    def _load_bac(self):
        # Open dialog in output_dir (if available)
        start_dir = os.path.join(os.getcwd(), "templates")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open BAC", start_dir, "BAC files (*.bac);;All files (*)"
        )
        if path:
            try:
                with open(path, "r") as f:
                    self.editor.setPlainText(f.read())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load BAC:\n{e}")

    def _generate(self):
        tab = self.tabs.currentIndex()
        bac_path = os.path.join(self.output_dir, f"{self.base_name}.bac")

        try:
            if tab == 0:  # Automatic
                slider_val = self.slider.value()
                selected = [item.text() for item in self.surface_list.selectedItems()]
                focus_ids = [int(s.split()[-1]) for s in selected]

                from FileRW.BacFile import BacFile

                if self.dat_path and os.path.exists(self.dat_path):
                    # Generate from DAT
                    bac = BacFile.fromDat(
                        self.dat_path,
                        name=self.base_name,
                        quality=slider_val,
                        focus_surfaces=focus_ids
                    )
                else:
                    # Fall back to geometry (STEP/IGES)
                    if not self.cad_path:
                        raise RuntimeError("No DAT or CAD path available for BAC generation.")
                    bac = BacFile.fromGeometry(
                        name=self.base_name,
                        path=self.cad_path,
                        slider_val=slider_val,
                        focus_surfaces=focus_ids
                    )

                with open(bac_path, "w") as f:
                    f.write(str(bac))

            else:  # Manual
                text = self.editor.toPlainText()
                with open(bac_path, "w") as f:
                    f.write(text)

            QMessageBox.information(self, "Success", f"BAC saved to:\n{bac_path}")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"BAC generation failed:\n{e}")

class BppDialog(QDialog):
    def __init__(self, mesh_obj, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BPP Settings")

        layout = QVBoxLayout()

        # Farfield selection
        layout.addWidget(QLabel("Select Farfield Surfaces:"))
        self.farfield_list = QListWidget()
        self.farfield_list.setSelectionMode(QListWidget.MultiSelection)
        for nm in mesh_obj.get_surface_names():
            item = QListWidgetItem(nm)
            self.farfield_list.addItem(item)
        layout.addWidget(self.farfield_list)

        # Symmetry selection
        layout.addWidget(QLabel("Select Symmetry Surfaces:"))
        self.symm_list = QListWidget()
        self.symm_list.setSelectionMode(QListWidget.MultiSelection)
        for nm in mesh_obj.get_surface_names():
            item = QListWidgetItem(nm)
            self.symm_list.addItem(item)
        layout.addWidget(self.symm_list)

        # Inflation layers
        layout.addWidget(QLabel("Number of inflation layers:"))
        self.n_layers = QSpinBox(); self.n_layers.setRange(0, 100)
        layout.addWidget(self.n_layers)

        layout.addWidget(QLabel("First layer thickness:"))
        self.first_layer = QDoubleSpinBox(); self.first_layer.setDecimals(6); self.first_layer.setValue(0.001)
        layout.addWidget(self.first_layer)

        layout.addWidget(QLabel("Growth ratio (optional):"))
        self.growth = QDoubleSpinBox(); self.growth.setDecimals(3); self.growth.setValue(1.2)
        layout.addWidget(self.growth)

        self.manual_entry = QLineEdit()
        self.manual_entry.setPlaceholderText("Or enter thicknesses manually (comma separated)")
        layout.addWidget(self.manual_entry)

        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

        self.setLayout(layout)

    def get_values(self):
        farfield = [item.text() for item in self.farfield_list.selectedItems()]
        symm = [item.text() for item in self.symm_list.selectedItems()]

        if self.manual_entry.text():
            layers = [float(x) for x in self.manual_entry.text().split(",")]
        else:
            n = self.n_layers.value()
            t0 = self.first_layer.value()
            r  = self.growth.value()
            layers = [t0 * (r**i) for i in range(n)]
        return farfield, symm, layers


class CtlEditDialog(QDialog):
    def __init__(self, title, initial_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)

        self.path = initial_path
        self.editor = QPlainTextEdit(self)
        self.editor.setPlaceholderText("# Control file content goes here…")

        # Try load if path provided
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                    self.editor.setPlainText(f.read())
            except Exception as e:
                QMessageBox.warning(self, "Open failed", f"Could not read:\n{self.path}\n\n{e}")

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        run_btn = btns.addButton("Run Surface Mesh", QDialogButtonBox.ActionRole)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        run_btn.clicked.connect(self._run_surface_mesh)

        pick_btn = QPushButton("Load…")
        pick_btn.clicked.connect(self._pick)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel(os.path.basename(self.path) if self.path else "(no file)"))
        top.addStretch(1)
        top.addWidget(pick_btn)
        layout.addLayout(top)
        layout.addWidget(self.editor)
        layout.addWidget(btns)

    def _save_and_accept(self):
        """Save text back to file + set variable on main_window."""
        text = self.editor.toPlainText()

        # Always save back to the input file if path is available
        if self.path:
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                QMessageBox.warning(self, "Save failed", f"Could not save to:\n{self.path}\n\n{e}")

        # Store it in memory on the main_window (accessible to pipeline)
        if hasattr(self.parent(), "main_window"):
            self.parent().main_window.surface_ctl_text = text

        self.accept()
        
    def _run_surface_mesh(self):
        from workers import SurfaceWorker
        from PyQt5.QtCore import QThread

        # get pipeline through GeometryPanel → MainWindow
        geo_panel = self.parent()   # GeometryPanel was passed as parent
        pipeline = getattr(geo_panel.main_window, "pipeline", None)

        if pipeline is None:
            QMessageBox.warning(self, "Error", "No pipeline available to run surface mesh.")
            return

        self.thread = QThread()
        self.worker = SurfaceWorker(pipeline=pipeline, debug=True)   # 👈 change here
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # hook logs + errors
        self.worker.log.connect(lambda msg: geo_panel.main_window.logger.log(msg))
        self.worker.failed.connect(lambda e: QMessageBox.critical(self, "Surface Mesh Error", str(e)))

        self.thread.start()
        self.accept()

    def _pick(self):
        start_dir = os.path.join(os.getcwd(), "templates")
        path, _ = QFileDialog.getOpenFileName(self, "Open control file", start_dir, "CTL files (*.ctl);;All files (*)")
        if path:
            self.path = path
            try:
                with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                    self.editor.setPlainText(f.read())
            except Exception as e:
                QMessageBox.warning(self, "Open failed", f"Could not read:\n{self.path}\n\n{e}")

    def get_text(self):
        return self.editor.toPlainText()

class GeometryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)

        # Viewer
        self.viewer = OCCViewer(self)
        self.layout.addWidget(self.viewer)

        # Path label
        self.path_lbl = QLabel("No CAD loaded")
        self.path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.layout.addWidget(self.path_lbl)

        # --- Main geometry layout (hide/reset + mesher buttons) ---
        geom = QVBoxLayout()
        self.layout.addLayout(geom)

        # State
        self.cad_path = None
        self.cad_ffd: Optional[CADFFDManager] = None
        self.dat_path = None
        self.dat_generated = False
        self.bpp_generated = False

        self.use_cad_param = False
        self.control_nodes_ready = False

        self.T_names = []
        self.U_names = []
        self.C_names = []

        # --- CAD parameterisation widget (initially hidden) ---
        self.param_widget = QWidget()
        self.param_layout = QVBoxLayout(self.param_widget)
        self.param_widget.setVisible(False)
        self.layout.addWidget(self.param_widget)

        # Row of T/U/C selection buttons (operate on CAD face table)
        self.btn_sel_T = QPushButton("Select T Surfaces")
        self.btn_sel_C = QPushButton("Select C Surfaces")
        self.btn_sel_U = QPushButton("Select U Surfaces")
        self.btn_edit_surfaces = QPushButton("Edit Surface Selections")
        self.btn_save_cn = QPushButton("Done")

        self.param_layout.addWidget(self.btn_sel_T)
        self.param_layout.addWidget(self.btn_sel_C)
        self.param_layout.addWidget(self.btn_sel_U)
        self.param_layout.addWidget(self.btn_edit_surfaces)
        self.param_layout.addWidget(self.btn_save_cn)

        # Wire CAD-param buttons
        self.btn_sel_T.clicked.connect(self._select_T_surfaces)
        self.btn_sel_C.clicked.connect(self._select_C_surfaces)
        self.btn_sel_U.clicked.connect(self._select_U_surfaces)
        self.btn_edit_surfaces.clicked.connect(self._open_param_surface_dialog)
        self.btn_save_cn.clicked.connect(self._cad_param_save)

        # --- Hide / reset controls ---
        ctrl = QHBoxLayout()
        self.hide_btn  = QPushButton("Hide selected")
        self.reset_cam = QPushButton("Reset camera")
        self.reset_vis = QPushButton("Reset surfaces")
        ctrl.addWidget(self.hide_btn)
        ctrl.addWidget(self.reset_cam)
        ctrl.addWidget(self.reset_vis)
        geom.addLayout(ctrl)

        # --- Mesher / file-generation buttons ---
        mesher1 = QHBoxLayout()
        self.btn_dat = QPushButton("Generate DAT file")
        self.btn_dat.setToolTip("Convert the loaded CAD to a DAT file; optionally clean it.")
        self.btn_bac = QPushButton("Generate BAC file")
        self.btn_bac.setToolTip("Generate a BAC file; select grid size and focus surfaces.")
        self.btn_bpp = QPushButton("Generate BPP file")
        self.btn_bpp.setToolTip("Generate a BPP file; specify surface types and inflation layers.")
        mesher1.addWidget(self.btn_dat)
        mesher1.addWidget(self.btn_bac)
        mesher1.addWidget(self.btn_bpp)
        geom.addLayout(mesher1)

        mesher2 = QHBoxLayout()
        self.btn_surf = QPushButton("Run Surface Mesh")
        self.btn_surf.setToolTip("Edit/save a surface meshing control (.ctl); hook to your mesher later.")
        self.btn_vol  = QPushButton("Run Volume Mesh")
        self.btn_vol.setToolTip("Edit/save a volume meshing control (.ctl); hook to your mesher later.")
        mesher2.addWidget(self.btn_surf)
        mesher2.addWidget(self.btn_vol)
        geom.addLayout(mesher2)
        geom.addStretch(1)

        # Group mesher-related buttons for easy show/hide
        self.mesh_buttons = (
            self.btn_dat, self.btn_bac, self.btn_bpp,
            self.btn_surf, self.btn_vol
        )

        # Initially hide / disable mesher buttons until CAD is loaded
        for b in self.mesh_buttons:
            b.setVisible(False)
            b.setEnabled(False)

        self.summary_dock = None
        self.summary_table = None

        # Wire controls
        self.hide_btn.clicked.connect(self._hide_selected_rows)
        self.reset_cam.clicked.connect(self.viewer.reset_camera)
        self.reset_vis.clicked.connect(self._reset_surfaces)

        self.btn_dat.clicked.connect(self._generate_dat)
        self.btn_bac.clicked.connect(self._generate_bac)
        self.btn_bpp.clicked.connect(self._generate_bpp)
        self.btn_surf.clicked.connect(self.run_surface_mesher)
        self.btn_vol.clicked.connect(self.run_volume_mesher)
        
        self.layout = QVBoxLayout(self)

    def _face_role_from_ui(self, face_id: int) -> FaceRole:
        """
        Map face index -> FaceRole using the T_names, C_names, U_names lists.
        Face names in the table are like 'Face 0', 'Face 1', ...
        """
        name = f"Face {face_id}"
        if name in getattr(self, "T_names", []):
            return FaceRole.T
        elif name in getattr(self, "C_names", []):
            return FaceRole.C
        elif name in getattr(self, "U_names", []):
            return FaceRole.U
        else:
            # default: treat as U if not explicitly tagged
            return FaceRole.U


    def _init_cad_ffd_manager(self):
        """
        Create and initialise CADFFDManager once CAD is loaded and T/C/U are set.
        """
        from FileRW.StpFile import StpFile

        if not getattr(self, "cad_path", None):
            if hasattr(self.main_window, "logger"):
                self.main_window.logger.log("[CAD-FFD] No CAD file loaded.")
            return

        # Load STEP and get shape + faces
        stp = StpFile(self.cad_path)
        shape = stp.shape
        faces = stp.get_faces()   # you already iterate faces elsewhere

        # Create manager
        self.cad_ffd = CADFFDManager(shape)

        for face_id, face in enumerate(faces):
            role = self._face_role_from_ui(face_id)
            bspline = stp.as_bspline(face)   # you’ll need a helper that returns Geom_BSplineSurface
            self.cad_ffd.register_face(face_id, face, role, bspline)

        # Build FFD around T faces and embed T poles
        self.cad_ffd.build_ffd_from_T_faces()
        self.cad_ffd.embed_T_faces()

        if hasattr(self.main_window, "logger"):
            self.main_window.logger.log("[CAD-FFD] FFD lattice built and T poles embedded.")

    
    def set_input_filepath(self, path):
        self.input_filepath = path
        
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        
    def set_logger(self, logger):
        self.logger = logger
    
    def load_cad(self, path: str):
        self.cad_path = path
        self.path_lbl.setText(path)

        # draw CAD in the OCC viewer + face table
        if hasattr(self.main_window, "logger"):
            self.main_window.logger.log(f"[INFO] Loading CAD into OCC viewer: {path}")
        try:
            self.viewer.display_cad(path)
            self._build_face_summary()
            if hasattr(self.main_window, "logger"):
                self.main_window.logger.log("[INFO] CAD displayed (OCC/AIS).")
        except Exception as e:
            if hasattr(self.main_window, "logger"):
                self.main_window.logger.log(f"[ERROR] CAD display failed: {e}")
            QMessageBox.critical(self, "CAD error", f"Failed to display CAD:\n{e}")
            return

        # --- Ask user if they want CAD-based parameterisation ---
        reply = QMessageBox.question(
            self,
            "CAD parameterisation",
            "Enable CAD-based parameterisation (select T/U/C faces and define control nodes)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        self.use_cad_param = (reply == QMessageBox.Yes)
        self.control_nodes_ready = False

        if self.use_cad_param:
            # Show the CAD parameterisation panel, hide mesher controls for now
            self.param_widget.setVisible(True)
            for b in self.mesh_buttons:
                b.setVisible(False)
                b.setEnabled(False)

            if hasattr(self.main_window, "logger"):
                self.main_window.logger.log(
                    "[GeomParam] CAD parameterisation enabled. "
                    "Mesher controls will appear after control nodes are saved."
                )
        else:
            # No CAD param: behave like original – show/enable mesher buttons
            self.param_widget.setVisible(False)
            for b in self.mesh_buttons:
                b.setVisible(True)
                b.setEnabled(True)

            if hasattr(self.main_window, "logger"):
                self.main_window.logger.log("[GeomParam] CAD parameterisation disabled (original workflow).")

    def _on_param_toggled(self, checked: bool):
        self.param_mesh_enabled = self.cb_mesh_param.isChecked()
        self.param_cad_enabled  = self.cb_cad_param.isChecked()

        enable = (self.param_mesh_enabled or self.param_cad_enabled) and bool(
            getattr(self, "cad_path", None)
        )
        self.btn_param_surfaces.setEnabled(enable)

    def _build_face_summary(self):
        names = [f"Face {i}" for i in range(len(self.viewer.faces))]
        self.all_cad_surface_names = names
        
        table = QTableWidget(); table.setRowCount(len(names)); table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Face"])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i, nm in enumerate(names):
            table.setItem(i, 0, QTableWidgetItem(nm))

        dock = QDockWidget("CAD Face Summary", self); dock.setWidget(table)
        if self.summary_dock and hasattr(self.main_window, "removeDockWidget"):
            self.main_window.removeDockWidget(self.summary_dock)
        self.summary_dock = dock; self.summary_table = table
        if hasattr(self.main_window, "addDockWidget"):
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, dock)

        table.itemSelectionChanged.connect(self._on_row_selected)
    
    def _face_role_from_ui(self, face_id: int) -> FaceRole:
        """
        Map face index -> FaceRole using the T_names, C_names, U_names lists.
        Face names in the table are like 'Face 0', 'Face 1', ...
        """
        name = f"Face {face_id}"  # matches _build_face_summary() naming
        if name in self.T_names:
            return FaceRole.T
        elif name in self.C_names:
            return FaceRole.C
        elif name in self.U_names:
            return FaceRole.U
        else:
            # default: treat as U (unaffected) if not explicitly tagged
            return FaceRole.U
    
    # ---------- CAD parameterisation helpers ----------

    def _get_selected_face_names(self):
        """
        Read selected rows from the CAD face summary table and return their names.
        """
        if not self.summary_table:
            QMessageBox.warning(self, "No faces", "No CAD face table available. Load a CAD file first.")
            return []

        rows = sorted({idx.row() for idx in self.summary_table.selectedIndexes()})
        if not rows:
            QMessageBox.information(
                self,
                "No selection",
                "Select one or more faces in the table, then click the T / U / C button."
            )
            return []

        names = []
        for r in rows:
            item = self.summary_table.item(r, 0)
            if item:
                nm = item.text()
                if nm not in names:
                    names.append(nm)
        return names

    def _get_all_face_names(self):
        """
        Return a list of all face names in the CAD face table.
        """
        if not self.summary_table:
            return []
        names = []
        for r in range(self.summary_table.rowCount()):
            item = self.summary_table.item(r, 0)
            if item:
                names.append(item.text())
        return names

    def _select_T_surfaces(self):
        names = self._get_selected_face_names()
        if not names:
            return
        for nm in names:
            if nm not in self.T_names:
                self.T_names.append(nm)
        if hasattr(self.main_window, "logger"):
            self.main_window.logger.log(f"[GeomParam] T surfaces: {self.T_names}")

    def _select_U_surfaces(self):
        names = self._get_selected_face_names()
        if not names:
            return
        for nm in names:
            if nm not in self.U_names:
                self.U_names.append(nm)
        if hasattr(self.main_window, "logger"):
            self.main_window.logger.log(f"[GeomParam] U surfaces: {self.U_names}")

    def _select_C_surfaces(self):
        names = self._get_selected_face_names()
        if not names:
            return
        for nm in names:
            if nm not in self.C_names:
                self.C_names.append(nm)
        if hasattr(self.main_window, "logger"):
            self.main_window.logger.log(f"[GeomParam] C surfaces: {self.C_names}")

    def _open_param_surface_dialog(self):
        """
        Open the same T/U/C edit dialog used in mesh_gui, to tweak selections.
        """
        all_names = self._get_all_face_names()
        if not all_names:
            QMessageBox.warning(self, "No faces", "No CAD faces found to edit.")
            return

        dlg = SurfaceEditDialog(all_names, self.T_names, self.U_names, self.C_names, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.T_names, self.U_names, self.C_names = dlg.get_results()
            if hasattr(self.main_window, "logger"):
                self.main_window.logger.log(
                    f"[GeomParam] Updated selections – T={self.T_names}, U={self.U_names}, C={self.C_names}"
                )
    
    def apply_cad_ffd_design(self, dP_lattice):
        """Called by optimiser or a 'Preview' button to apply one FFD design."""
        if self.cad_ffd is None:
            self.logger.log("[CAD-FFD] Manager not initialised.")
            return

        self.cad_ffd.apply_design(dP_lattice, propagate_C=True, propagate_U=True)

        # Now update viewer from deformed shape
        deformed_shape = self.cad_ffd.get_deformed_shape()
        self.viewer.display_shape(deformed_shape)  # or equivalent OCC viewer call

    
    def _cad_param_save(self):
        """
        Full CAD control-node generation:
        - extract NURBS poles on T faces
        - downsample to requested number
        - show modal/basis form
        """
        # Hide the buttons
        for btn in [self.btn_sel_C, self.btn_sel_T, self.btn_sel_U, self.btn_edit_surfaces, self.btn_save_cn]:
            btn.setVisible(False)

        # Request number of control nodes
        num_cn, ok = QInputDialog.getInt(
            self, "Control Node Count",
            "How many control nodes would you like to use?",
            value=50, min=1, max=5000
        )
        if not ok:
            return

        # Ensure T surfaces selected
        if not self.T_names:
            QMessageBox.information(
                self, "No T-surfaces",
                "Please select at least one T surface before generating control nodes."
            )
            return

        # Convert names ("Face 3") → indices (3)
        try:
            T_indices = [int(nm.split()[1]) for nm in self.T_names]
        except Exception:
            QMessageBox.warning(self, "Error", "Could not parse face indices for T surfaces.")
            return

 
                # --- CAD control-node extraction ---
        cn_all, face_ids, face_slices = self.viewer.get_control_nodes_from_faces(T_indices)

        if cn_all is None or len(cn_all) == 0:
            ...
            return

        # Keep all poles for plotting (full surface points)
        cn_all = np.asarray(cn_all, float)
        self.cn_all = cn_all          # <-- for preview

        # Downsample to requested number of control nodes
        n_all = cn_all.shape[0]
        k = min(num_cn, cn_all.shape[0])
        self.control_nodes = farthest_point_sampling(cn_all, k)

        # Log baseline
        if hasattr(self.main_window, "logger"):
            self.main_window.logger.log(
                f"[CAD] Extracted {n_all} poles from T faces; using {k} as control nodes."
            )
            
        # ---- Now show the modal-basis + bump window forms, like plot_T_surfaces ----
        self._show_basis_form()
        self.preview_cad_control_nodes()

        self.control_nodes_ready = True
        
        try:
            self._init_cad_ffd_manager()
            self.preview_cad_ffd_lattice()
        except Exception as e:
            if hasattr(self.main_window, "logger"):
                self.main_window.logger.log(f"[CAD-FFD] Init/preview failed: {e}")
                
    def test_ffd_bump(self):
        if not self.cad_ffd or not self.cad_ffd.ffd:
            return
        ffd = self.cad_ffd.ffd
        n_xi, n_eta, n_zeta = ffd.n_ctrl
        dP = np.zeros((n_xi, n_eta, n_zeta, 3))
        dP[:, :, -1, 2] = 0.01
        self.cad_ffd.apply_design(dP)
        deformed_shape = self.cad_ffd.get_deformed_shape()
        self.viewer.display_shape(deformed_shape)
            
    def _show_basis_form(self):
        """
        Replicates the modal/spectral parameter form from mesh_gui.plot_T_surfaces(),
        but without PyVista preview.
        """

        # Remove old form if any
        if hasattr(self, "basis_form_widget"):
            self.basis_form_widget.setParent(None)
        if hasattr(self, "basis_save_btn"):
            self.basis_save_btn.setParent(None)

        from PyQt5.QtWidgets import (
            QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QLineEdit, QWidget
        )

        form = QFormLayout()

        # ======== All settings exactly like mesh_gui ========
        # k_modes
        self.k_modes_spin = QSpinBox()
        self.k_modes_spin.setRange(1, 64)
        self.k_modes_spin.setValue(getattr(self, "k_modes", 6))
        form.addRow("Number of modes (k):", self.k_modes_spin)

        # spectral decay p
        self.decay_p_spin = QDoubleSpinBox()
        self.decay_p_spin.setRange(0.1, 6.0)
        self.decay_p_spin.setDecimals(2)
        self.decay_p_spin.setValue(getattr(self, "spectral_p", 2.0))
        form.addRow("Spectral decay p:", self.decay_p_spin)

        # coefficient amplitude fraction
        self.coeff_frac_spin = QDoubleSpinBox()
        self.coeff_frac_spin.setRange(0.01, 1.0)
        self.coeff_frac_spin.setDecimals(3)
        self.coeff_frac_spin.setSingleStep(0.01)
        self.coeff_frac_spin.setValue(getattr(self, "coeff_frac", 0.15))
        form.addRow("Coeff amplitude (frac):", self.coeff_frac_spin)

        # random seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1_000_000, 1_000_000)
        self.seed_spin.setValue(getattr(self, "seed", 0))
        form.addRow("Random seed (0=deterministic):", self.seed_spin)

        # normal projection
        self.normal_proj_cb = QCheckBox("Project along surface normals")
        self.normal_proj_cb.setChecked(getattr(self, "normal_project", True))
        form.addRow(self.normal_proj_cb)

        # bump enable
        self.bump_enable_cb = QCheckBox("Enable bump window")
        self.bump_enable_cb.setChecked(getattr(self, "bump_enable", False))
        form.addRow(self.bump_enable_cb)

        # bump parameters
        bump_row = QHBoxLayout()
        self.bump_cx = QLineEdit(); self.bump_cx.setPlaceholderText("cx")
        self.bump_cy = QLineEdit(); self.bump_cy.setPlaceholderText("cy")
        self.bump_cz = QLineEdit(); self.bump_cz.setPlaceholderText("cz")
        self.bump_r  = QLineEdit(); self.bump_r.setPlaceholderText("radius")
        bump_row.addWidget(QLabel("Center (x,y,z), Radius:"))
        bump_row.addWidget(self.bump_cx); bump_row.addWidget(self.bump_cy)
        bump_row.addWidget(self.bump_cz); bump_row.addWidget(self.bump_r)
        form.addRow(bump_row)

        self.bump_one_sided_cb = QCheckBox("One-sided (+normal only)")
        self.bump_one_sided_cb.setChecked(getattr(self, "bump_one_sided", False))
        form.addRow(self.bump_one_sided_cb)

        # Mount widget
        self.basis_form_widget = QWidget()
        self.basis_form_widget.setLayout(form)
        self.layout.addWidget(self.basis_form_widget)

        # Save button
        self.basis_save_btn = QPushButton("Save control-node settings")
        self.basis_save_btn.clicked.connect(self._finalise_cad_control_nodes)
        self.layout.addWidget(self.basis_save_btn)
        
    def preview_cad_control_nodes(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        """
        Preview CAD control nodes using Matplotlib:

        - grey dots  : all poles on the selected T-faces (self.cn_all)
        - black dots : chosen control nodes (self.control_nodes)
        - red dots + lines (optional): displaced control nodes (self.cn_targets)
        """
        if not hasattr(self, "control_nodes") or self.control_nodes is None or len(self.control_nodes) == 0:
            QMessageBox.information(self, "No control nodes", "No CAD control nodes to preview yet.")
            return

        pts_cn = np.asarray(self.control_nodes, float)
        pts_all = getattr(self, "cn_all", None)
        if pts_all is not None and len(pts_all):
            pts_all = np.asarray(pts_all, float)
        else:
            pts_all = None

        # Optional displacement (same shape as control_nodes)
        cn_targets = getattr(self, "cn_targets", None)
        if cn_targets is not None:
            cn_targets = np.asarray(cn_targets, float)
            if cn_targets.shape != pts_cn.shape:
                cn_targets = None
        # else: leave as None

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.canvas.manager.set_window_title("CAD Control Nodes")

        # all poles on T surface(s)
        if pts_all is not None:
            ax.scatter(
                pts_all[:, 0], pts_all[:, 1], pts_all[:, 2],
                s=5, c="lightgrey", alpha=0.9, label="surface poles"
            )

        # selected control nodes
        ax.scatter(
            pts_cn[:, 0], pts_cn[:, 1], pts_cn[:, 2],
            s=40, c="black", label="control nodes"
        )

        # displaced control nodes + connecting segments (optional)
        if cn_targets is not None:
            ax.scatter(
                cn_targets[:, 0], cn_targets[:, 1], cn_targets[:, 2],
                s=40, c="red", label="displaced CNs"
            )
            for p, q in zip(pts_cn, cn_targets):
                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    [p[2], q[2]],
                    "r-",
                    linewidth=1.5,
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="best")

        # make aspect roughly equal
        xs = np.concatenate([
            pts_all[:, 0] if pts_all is not None else pts_cn[:, 0],
            pts_cn[:, 0]
        ])
        ys = np.concatenate([
            pts_all[:, 1] if pts_all is not None else pts_cn[:, 1],
            pts_cn[:, 1]
        ])
        zs = np.concatenate([
            pts_all[:, 2] if pts_all is not None else pts_cn[:, 2],
            pts_cn[:, 2]
        ])
        max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min())
        if max_range > 0:
            mid_x = 0.5 * (xs.max() + xs.min())
            mid_y = 0.5 * (ys.max() + ys.min())
            mid_z = 0.5 * (zs.max() + zs.min())
            r = 0.5 * max_range
            ax.set_xlim(mid_x - r, mid_x + r)
            ax.set_ylim(mid_y - r, mid_y + r)
            ax.set_zlim(mid_z - r, mid_z + r)

        plt.tight_layout()
        plt.show()

    def preview_cad_ffd_lattice(self):
        """
        Show a Matplotlib 3D plot with:
        - FFD lattice control points/lines in grey
        - selected CAD control nodes in red
        """
        if self.cad_ffd is None or self.cad_ffd.ffd is None:
            QMessageBox.information(self, "CAD FFD", "CAD FFD manager not initialised yet.")
            return

        if not hasattr(self, "control_nodes") or self.control_nodes is None or len(self.control_nodes) == 0:
            QMessageBox.information(self, "CAD FFD", "No CAD control nodes to plot.")
            return

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        ffd = self.cad_ffd.ffd
        ctrl = ffd.ctrl_pts  # shape (nξ, nη, nζ, 3)
        n_xi, n_eta, n_zeta, _ = ctrl.shape

        pts_cn = np.asarray(self.control_nodes, float)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.canvas.manager.set_window_title("CAD FFD Lattice + Control Nodes")

        # --- Plot lattice as wireframe (grey) ---
        # Lines varying in zeta (k) for each (i,j)
        for i in range(n_xi):
            for j in range(n_eta):
                pts = ctrl[i, j, :, :]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=0.5, color="lightgrey")

        # Lines varying in eta (j) for each (i,k)
        for i in range(n_xi):
            for k in range(n_zeta):
                pts = ctrl[i, :, k, :]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=0.5, color="lightgrey")

        # Lines varying in xi (i) for each (j,k)
        for j in range(n_eta):
            for k in range(n_zeta):
                pts = ctrl[:, j, k, :]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=0.5, color="lightgrey")

        # --- Plot selected control nodes in red ---
        ax.scatter(
            pts_cn[:, 0], pts_cn[:, 1], pts_cn[:, 2],
            s=40, c="red", label="control nodes"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        plt.show()

    def _finalise_cad_control_nodes(self):
        """
        Store all parameters, attach them to mesh_viewer (so pipeline sees them),
        and reveal DAT/BAC/BPP/mesher buttons.
        """

        # Store settings
        self.k_modes      = int(self.k_modes_spin.value())
        self.spectral_p   = float(self.decay_p_spin.value())
        self.coeff_frac   = float(self.coeff_frac_spin.value())
        self.seed         = int(self.seed_spin.value())
        self.normal_project = bool(self.normal_proj_cb.isChecked())

        # bump
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

        # Push settings & control nodes into mesh_viewer (pipeline expects it here)
        mv = getattr(self.main_window, "mesh_viewer", None)
        if mv:
            mv.control_nodes    = self.control_nodes
            mv.k_modes          = self.k_modes
            mv.spectral_p       = self.spectral_p
            mv.coeff_frac       = self.coeff_frac
            mv.seed             = self.seed
            mv.normal_project   = self.normal_project

            mv.bump_enable      = self.bump_enable
            mv.bump_center      = self.bump_center
            mv.bump_radius      = self.bump_radius
            mv.bump_one_sided   = self.bump_one_sided

            try:
                mv.control_ready.emit()
            except:
                pass

        # Reveal DAT/BAC/BPP + mesher buttons
        for b in self.mesh_buttons:
            b.setVisible(True)
            b.setEnabled(True)

        # UI feedback
        if hasattr(self.main_window, "logger"):
            self.main_window.logger.log(
                f"[GeomParam] CAD control nodes saved ({len(self.control_nodes)} nodes), "
                f"k={self.k_modes}, p={self.spectral_p}"
            )

        QMessageBox.information(self, "Done", "Control-node settings saved.")


    def _on_row_selected(self):
        row = self.summary_table.currentRow()
        if row >= 0:
            self.viewer.highlight_face(row)

    def _hide_selected_rows(self):
        rows = sorted({idx.row() for idx in self.summary_table.selectedIndexes()})
        if rows:
            self.viewer.hide_faces(rows)

    def _reset_surfaces(self):
        self.viewer.show_all_faces()
        if self.summary_table:
            self.summary_table.clearSelection()

    def reset(self):
        # clear panel on New Project / Close
        if self.summary_dock and hasattr(self.main_window, "removeDockWidget"):
            try: self.main_window.removeDockWidget(self.summary_dock)
            except Exception: pass
        self.summary_dock = None; self.summary_table = None
        self.path_lbl.setText("No CAD loaded")
        try: self.viewer.clear()
        except Exception: pass
        
    def _generate_dat(self):
        if not self.cad_path:
            QMessageBox.information(self, "No CAD", "Please open a STEP/IGES file first.")
            return

        # choose output DAT path
        base_name = os.path.splitext(os.path.basename(self.main_window.input_file_path))[0]
        default_dir = getattr(self.main_window, "output_directory", os.getcwd())
        out_path, _ = QFileDialog.getSaveFileName(self, "Save DAT as", os.path.join(default_dir, f"{base_name}.dat"),
                                                  "DAT files (*.dat);;All files (*)")
        if not out_path:
            return

        # TODO: integrate your actual converter here when ready, e.g.:
        # from ConvertFileType.step_to_dat import convert
        # convert(self.cad_path, out_path, clean=clean)
        self.dat_generated = True
        if hasattr(self.main_window, "logger"):
            self.main_window.logger.log(f"[WARN] DAT generation is stubbed. Would convert:\n"
                                        f"       CAD={self.cad_path}\n       OUT={out_path}\n       clean=")
        QMessageBox.information(self, "Generate DAT", "Stub: hooked for later. (Logged the intended action.)")
        
        self.dat_path = out_path

    def _generate_bac(self):
        # Always allow the user to try — the dialog will handle DAT vs CAD fallback
        dat_path = getattr(self, "dat_path", None)
        cad_path = getattr(self, "cad_path", None)
        base_name = os.path.splitext(os.path.basename(self.main_window.input_file_path))[0]
        
        dlg = BacFileDialog(
            base_name=base_name,
            output_dir=self.main_window.output_directory,
            dat_path=dat_path,
            cad_path=cad_path,
            parent=self
        )
        dlg.exec_()
        
    def _generate_bpp(self):
        """
        Open the BPP dialog using the *mesh* (if available) and write a custom BPP
        into main_window.output_directory/base_name.bpp.

        If no mesh is loaded yet, we just warn the user.
        """
        base_name = os.path.splitext(os.path.basename(self.main_window.input_file_path))[0]

        mesh_viewer = getattr(self.main_window, "mesh_viewer", None)
        mesh_obj = getattr(mesh_viewer, "mesh_obj", None) if mesh_viewer else None
        if mesh_obj is None:
            QMessageBox.warning(
                self,
                "No surface mesh",
                "Load a surface mesh in the Mesh tab before creating a BPP file."
            )
            return

        dlg = BppDialog(mesh_obj, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            farfield, symm, layers = dlg.get_values()

            bpp = BppFile(base_name)
            bpp.FarField_Surf_Names = farfield
            bpp.Symmetry_Surf_Names = symm
            bpp.Layer_Heights = layers

            # NOTE: HPCPipelineManager.surface() looks in output_directory, not /surfaces/…
            bpp_path = os.path.join(self.main_window.output_directory, f"{base_name}.bpp")
            os.makedirs(self.main_window.output_directory, exist_ok=True)
            with open(bpp_path, "w") as f:
                f.write(str(bpp))

            if hasattr(self.main_window, "logger"):
                self.main_window.logger.log(f"[Geometry] Custom BPP file written to {bpp_path}")

            self.bpp_generated = True
            QMessageBox.information(self, "BPP generated", f"Custom BPP file saved:\n{bpp_path}")

    def run_surface_mesher(self):
        self._open_ctl_editor(kind="surf")
        
    def run_volume_mesher(self):
        self._open_ctl_editor(kind="vol")

    def _open_ctl_editor(self, kind="surf"):
        # decide default file name
        default_dir = getattr(self.main_window, "output_directory", os.getcwd())
        if kind == "surf":
            title = "Surface Mesh Control"
            default_name = "Surf3D_v25.ctl"
        else:
            title = "Volume Mesh Control"
            default_name = "Mesh3D_v50.ctl"

        # Try pre-existing file in output_dir; else start empty
        default_path = os.path.join(default_dir or os.getcwd(), default_name)
        dlg = CtlEditDialog(title, initial_path=default_path if os.path.exists(default_path) else None, parent=self)

        if dlg.exec_() == QDialog.Accepted:
            text = dlg.get_text()
            # Save to output_dir with the standard name
            os.makedirs(default_dir, exist_ok=True)
            save_path = os.path.join(default_dir, default_name)
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text)
                if hasattr(self.main_window, "logger"):
                    self.main_window.logger.log(f"[INFO] Saved {title} to: {save_path}")
                QMessageBox.information(self, title, f"Saved to:\n{save_path}\n")
            except Exception as e:
                QMessageBox.critical(self, "Save failed", f"Could not save:\n{save_path}\n\n{e}")