# geom_gui.py
# ----------------------------------------------------------------------
# Mesh-oriented GUI page that now exposes the same "Generate BAC / BPP /
# Edit Surf CTL" affordances as geometry_gui.
#
# - Writes:
#     <output_dir>/<base>.bac
#     <output_dir>/<base>.bpp
#     <output_dir>/Surf3D_v25.ctl
#     <output_dir>/boundaries.json   (when using "Generate BPP")
#
# - Reuses dialogs implemented in geometry_gui:
#     BacFileDialog, BppDialog, CtlEditDialog
#
# - Assumptions:
#   * self.main_window has attributes:
#       - input_file_path (str)            # currently loaded file path
#       - output_directory (str)           # chosen output dir
#       - logger.log(str)                  # logging method
#   * A "mesh_obj" can be set via set_mesh_obj(...) if your BPP dialog needs it.
#
# You can wire this page into your stacked widget same as before.
# ----------------------------------------------------------------------

import os
import json
import shutil
from typing import Optional, Dict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QDialog, QFileDialog, QTextEdit, QSplitter, QLabel
)

# Reuse the richer dialogs from geometry_gui to keep UX consistent
try:
    from geometry_gui import (
        BacFileDialog as GeoBacDialog,
        BppDialog as GeoBppDialog,
        CtlEditDialog as GeoCtlDialog,
    )
except Exception as e:
    # If imports fail, we provide minimal fallbacks so the file still loads.
    GeoBacDialog = None
    GeoBppDialog = None
    GeoCtlDialog = None


# Optional imports for writing files directly when needed
try:
    # If your project structure puts these under FileRW.*, adjust imports accordingly
    from FileRW.BppFile import BppFile          # or: from FileRW.BppFile import BppFile
except Exception:
    try:
        from FileRW.BppFile import BppFile
    except Exception:
        BppFile = None

try:
    from FileRW.BacFile import BacFile          # or: from FileRW.BacFile import BacFile
except Exception:
    try:
        from FileRW.BacFile import BacFile
    except Exception:
        BacFile = None


class GeomWindow(QWidget):
    """
    Mesh GUI page (surface-centric) with parity to geometry_gui's buttons/options:
      - Generate BAC
      - Generate BPP
      - Edit Surf CTL
    """

    def __init__(self, main_window=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window
        self.mesh_obj = None  # set via set_mesh_obj(...)
        self._init_ui()

    # -------------------------- UI ------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)

        # Top controls row
        btn_row = QHBoxLayout()

        self.btn_generate_bac = QPushButton("Generate BAC")
        self.btn_generate_bpp = QPushButton("Generate BPP")
        self.btn_edit_ctl     = QPushButton("Edit Surf CTL")

        btn_row.addWidget(self.btn_generate_bac)
        btn_row.addWidget(self.btn_generate_bpp)
        btn_row.addWidget(self.btn_edit_ctl)
        btn_row.addStretch(1)

        # Optional: a tiny status label + placeholder area
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #666;")

        # Optional center area (splitter) if you want to put a mesh preview or notes
        self.center_split = QSplitter(Qt.Vertical)
        self.center_split.addWidget(self.status_lbl)
        # Placeholder for future content; keeping minimal for a drop-in replacement
        placeholder = QTextEdit()
        placeholder.setReadOnly(True)
        placeholder.setPlainText(
            "geom_gui page loaded.\n\n"
            "Use the buttons above to generate BAC / BPP / Surf CTL.\n"
            "Files are written into the current output directory."
        )
        self.center_split.addWidget(placeholder)
        self.center_split.setSizes([60, 240])

        root.addLayout(btn_row)
        root.addWidget(self.center_split)
        self.setLayout(root)

        # Wire actions
        self.btn_generate_bac.clicked.connect(self.generate_bac_file)
        self.btn_generate_bpp.clicked.connect(self.generate_bpp_file)
        self.btn_edit_ctl.clicked.connect(self.edit_surf_ctl)

    # --------------------- External setters ----------------------------
    def set_main_window(self, mw):
        self.main_window = mw

    def set_mesh_obj(self, mesh_obj):
        """Provide the loaded surface mesh object (whatever your BppDialog expects)."""
        self.mesh_obj = mesh_obj

    # ------------------------- Utilities -------------------------------
    def _log(self, msg: str):
        if getattr(self.main_window, "logger", None):
            try:
                self.main_window.logger.log(msg)
                return
            except Exception:
                pass
        print(msg)

    def _base_name(self) -> str:
        try:
            path = self.main_window.input_file_path
        except Exception:
            path = None
        if not path:
            return "model"
        return os.path.splitext(os.path.basename(path))[0]

    def _output_dir(self) -> str:
        outd = getattr(self.main_window, "output_directory", "") or ""
        if not outd:
            # Prompt if not set
            chosen = QFileDialog.getExistingDirectory(self, "Select Output Directory", os.getcwd())
            if chosen:
                outd = chosen
                try:
                    self.main_window.output_directory = outd
                    if hasattr(self.main_window, "output_directory_set"):
                        self.main_window.output_directory_set = True
                except Exception:
                    pass
        if not outd:
            outd = os.getcwd()
        os.makedirs(outd, exist_ok=True)
        return outd

    def _is_cad(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in (".step", ".stp", ".iges", ".igs")

    # ------------------------- Actions ---------------------------------
    def generate_bac_file(self):
        """
        Invoke the *same* BAC generation dialog as geometry_gui, then save <base>.bac into output_dir.
        """
        outd = self._output_dir()
        base = self._base_name()
        cad_path = None

        try:
            p = self.main_window.input_file_path
            if p and self._is_cad(p):
                cad_path = p
        except Exception:
            pass

        dat_path = None  # set if you have a dat readily available

        if GeoBacDialog is None:
            # Minimal fallback: attempt to create a generic BAC if dialog is unavailable
            if BacFile is None:
                QMessageBox.critical(self, "BAC", "BacFile class not available and dialog import failed.")
                return
            bac_path = os.path.join(outd, f"{base}.bac")
            bac = None
            try:
                bac = BacFile.defaultCRMFineMesh(name=base)
            except Exception:
                bac = BacFile(base) if hasattr(BacFile, "__call__") else None
            if bac is None:
                QMessageBox.critical(self, "BAC", "Could not create a BAC file without the dialog.")
                return
            with open(bac_path, "w", newline="\n") as f:
                f.write(str(bac))
            self._log(f"[BAC] Saved fallback BAC → {bac_path}")
            QMessageBox.information(self, "BAC", f"Saved:\n{bac_path}")
            return

        dlg = GeoBacDialog(
            base_name=base,
            output_dir=outd,
            dat_path=dat_path,
            cad_path=cad_path,
            parent=self,
        )
        dlg.exec_()

        bac_path = os.path.join(outd, f"{base}.bac")
        if os.path.exists(bac_path):
            self._log(f"[BAC] Saved: {bac_path}")
            QMessageBox.information(self, "BAC", f"Saved:\n{bac_path}")
        else:
            QMessageBox.warning(self, "BAC", "BAC was not created.")

    def generate_bpp_file(self):
        """
        Invoke the *same* BPP dialog as geometry_gui. Also persist a boundaries.json
        that SimulationWorker (or the pipeline) can reuse if the user skips this step later.
        """
        outd = self._output_dir()
        base = self._base_name()

        if GeoBppDialog is None:
            if BppFile is None:
                QMessageBox.critical(self, "BPP", "BppFile class not available and dialog import failed.")
                return
            # Minimal fallback: write an empty-but-valid BPP
            bpp = BppFile(base)
            bpp_path = os.path.join(outd, f"{base}.bpp")
            with open(bpp_path, "w", newline="\n") as f:
                f.write(str(bpp))
            self._log(f"[BPP] Saved fallback BPP → {bpp_path}")
            QMessageBox.information(self, "BPP", f"Saved:\n{bpp_path}")
            return

        if self.mesh_obj is None:
            # In your app, the dialog may still work without a mesh_obj if it queries names from elsewhere.
            # We'll warn the user but still try to open it.
            self._log("[BPP] Warning: mesh_obj not set; proceeding with dialog anyway.")

        dlg = GeoBppDialog(self.mesh_obj, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            farfield, symmetry, layers = dlg.get_values()

            # Persist mapping for worker fallbacks
            mapping: Dict[str, str] = {name: "Farfield" for name in farfield}
            mapping.update({name: "Symmetry" for name in symmetry})
            with open(os.path.join(outd, "boundaries.json"), "w") as f:
                json.dump(mapping, f, indent=2)

            if BppFile is None:
                # If BppFile is not importable, at least save the mapping and notify the user.
                QMessageBox.information(
                    self, "BPP",
                    "Saved boundaries.json. BppFile class not available, so the BPP file was not written."
                )
                return

            # Write the BPP itself
            bpp = BppFile(base)
            # Set lists expected by your serializer
            try:
                bpp.FarField_Surf_Names = list(farfield)
            except Exception:
                setattr(bpp, "FarField_Surf_Names", list(farfield))
            try:
                bpp.Symmetry_Surf_Names = list(symmetry)
            except Exception:
                setattr(bpp, "Symmetry_Surf_Names", list(symmetry))
            try:
                bpp.UnChecking_Surf_Names = list(farfield) + list(symmetry)
            except Exception:
                setattr(bpp, "UnChecking_Surf_Names", list(farfield) + list(symmetry))
            # Layers
            try:
                bpp.Layer_Heights = list(layers)
            except Exception:
                setattr(bpp, "Layer_Heights", list(layers))

            bpp_path = os.path.join(outd, f"{base}.bpp")
            with open(bpp_path, "w", newline="\n") as f:
                f.write(str(bpp))

            self._log(f"[BPP] Saved: {bpp_path}")
            QMessageBox.information(self, "BPP", f"Saved:\n{bpp_path}")

    def edit_surf_ctl(self):
        """
        Open the same Surf control editor as geometry_gui for Surf3D_v25.ctl in output_dir.
        If the file does not exist, we seed from templates/Surf3D_v25.ctl or create a stub.
        """
        outd = self._output_dir()
        ctl_name = "Surf3D_v25.ctl"
        ctl_path = os.path.join(outd, ctl_name)

        if not os.path.exists(ctl_path):
            # Seed from templates if present
            tpl = os.path.join(os.getcwd(), "templates", ctl_name)
            if os.path.exists(tpl):
                try:
                    shutil.copyfile(tpl, ctl_path)
                except Exception:
                    with open(ctl_path, "w") as f:
                        f.write("# Surf control\n")
            else:
                with open(ctl_path, "w") as f:
                    f.write("# Surf control\n")

        if GeoCtlDialog is None:
            # Minimal fallback: open the file in a quick text editor
            self._log("[CTL] geometry_gui dialog not found; opening a simple editor fallback.")
            self._open_simple_text_editor(ctl_path, title="Surf Control (fallback editor)")
            return

        dlg = GeoCtlDialog("Surface Mesh Control", initial_path=ctl_path, parent=self)
        dlg.exec_()

    # -------------------- Simple fallback editor -----------------------
    def _open_simple_text_editor(self, file_path: str, title: str = "Editor"):
        w = QDialog(self)
        w.setWindowTitle(title)
        v = QVBoxLayout(w)
        edit = QTextEdit(w)
        btns = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_close = QPushButton("Close")
        btns.addStretch(1)
        btns.addWidget(btn_save)
        btns.addWidget(btn_close)

        # Load if exists
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                edit.setPlainText(f.read())
        except Exception:
            pass

        def save_now():
            try:
                with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(edit.toPlainText())
                self._log(f"[Editor] Saved: {file_path}")
                QMessageBox.information(w, "Saved", f"Saved:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(w, "Save failed", str(e))

        btn_save.clicked.connect(save_now)
        btn_close.clicked.connect(w.close)

        v.addWidget(edit)
        v.addLayout(btns)
        w.resize(700, 500)
        w.exec_()
