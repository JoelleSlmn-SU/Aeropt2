# geom_gui.py (GeometryPanel.__init__)
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QDialogButtonBox, QDialog, QSlider, QListWidget,
                             QDockWidget, QAbstractItemView, QHBoxLayout, QPushButton, QMessageBox, QFileDialog, QPlainTextEdit, QLineEdit,
                             QDoubleSpinBox, QSpinBox, QListWidgetItem, QListWidget)
from PyQt5.QtCore import Qt
import os, sys
sys.path.append(os.path.dirname("FileRW"))
from FileRW.StpFile import OCCViewer
from FileRW.BacFile import BacFile
from FileRW.BppFile import BppFile

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                             QListWidget, QAbstractItemView, QPushButton, 
                             QTabWidget, QPlainTextEdit, QFileDialog, QMessageBox)

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
        self.path_lbl = QLabel("No CAD loaded"); self.path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.layout.addWidget(self.path_lbl)

        # 🔧 Controls row (like Mesh)
        geom = QVBoxLayout()
        self.layout.addLayout(geom)
        
        ctrl = QHBoxLayout()
        self.hide_btn  = QPushButton("Hide selected")
        self.reset_cam = QPushButton("Reset camera")
        self.reset_vis = QPushButton("Reset surfaces")
        ctrl.addWidget(self.hide_btn); ctrl.addWidget(self.reset_cam); ctrl.addWidget(self.reset_vis)
        geom.addLayout(ctrl)
        
        mesher1 = QHBoxLayout()
        self.btn_dat = QPushButton("Generate DAT file")
        self.btn_dat.setToolTip("Convert the loaded CAD to a DAT file; optionally clean it.")
        self.btn_bac = QPushButton("Generate BAC file")
        self.btn_bac.setToolTip("Generate a BAC file; select grid size and focus surfaces.")
        self.btn_bpp = QPushButton("Generate BPP file")
        self.btn_bpp.setToolTip("Generate a BPP file; specify surface types and inflation layers.")
        mesher1.addWidget(self.btn_dat); mesher1.addWidget(self.btn_bac); mesher1.addWidget(self.btn_bpp)
        geom.addLayout(mesher1)
        
        mesher2 = QHBoxLayout()
        self.btn_surf = QPushButton("Run Surface Mesh")
        self.btn_surf.setToolTip("Edit/save a surface meshing control (.ctl); hook to your mesher later.")
        self.btn_vol  = QPushButton("Run Volume Mesh")
        self.btn_vol.setToolTip("Edit/save a volume meshing control (.ctl); hook to your mesher later.")
        mesher2.addWidget(self.btn_surf); mesher2.addWidget(self.btn_vol)
        geom.addLayout(mesher2)
        geom.addStretch(1)

        for b in (self.btn_dat, self.btn_bac, self.btn_bpp, self.btn_surf, self.btn_vol):
            b.setEnabled(True)
        # (keep your existing DAT / Surf / Vol buttons right below)
        # self.btn_dat, self.btn_surf, self.btn_vol …

        self.dat_generated = False
        self.bpp_generated = False
        
        self.summary_dock = None
        self.summary_table = None

        # Wire controls
        self.hide_btn.clicked.connect(self._hide_selected_rows)
        self.reset_cam.clicked.connect(self.viewer.reset_camera)
        self.reset_vis.clicked.connect(self._reset_surfaces)
        
        self.btn_dat.clicked.connect(self._generate_dat)
        self.btn_bac.clicked.connect(self._generate_bac)
        self.btn_bpp.clicked.connect(self._generate_bpp)
        self.btn_surf.clicked.connect(lambda: self.run_surface_mesher())
        self.btn_vol.clicked.connect(lambda: self.run_volume_mesher())

        # disabled until a CAD is loaded
        for b in (self.btn_dat, self.btn_surf, self.btn_vol):
            b.setEnabled(False)

    def set_input_filepath(self, path):
        self.input_filepath = path
        
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        
    def set_logger(self, logger):
        self.logger = logger
    
    def load_cad(self, path: str):
        self.cad_path = path                # <-- add this
        self.path_lbl.setText(path)

        # enable CAD-dependent buttons
        for b in (self.btn_dat, self.btn_surf, self.btn_vol):
            b.setEnabled(True)              # <-- add this

        # draw
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

    def _build_face_summary(self):
        names = [f"Face {i}" for i in range(len(self.viewer.faces))]
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
        base_name = os.path.splitext(os.path.basename(self.main_window.input_file_path))[0]
        dlg = BppDialog(self.mesh_viewer.mesh_obj, parent=self)
        
        if dlg.exec_() == QDialog.Accepted:
            farfield, symm, layers = dlg.get_values()
            bpp = BppFile(self.base_name)
            bpp.FarField_Surf_Names = farfield
            bpp.Symmetry_Surf_Names = symm
            bpp.Layer_Heights = layers
            bpp_path = os.path.join(self.main_window.output_directory, "surfaces", f"{base_name}.bpp")
            with open(bpp_path, "w") as f:
                f.write(str(bpp))
            self.logger.log(f"[Geometry] Custom BPP file written to {bpp_path}")
            self.bpp_generated = True

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