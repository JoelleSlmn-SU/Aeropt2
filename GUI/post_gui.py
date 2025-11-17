# post_gui.py — upgraded PostViewer with embedded QtInteractor, hide/reset/clip,
# and derived fields (Density, |U|, Mach, p_static, p_total).
#
# Adapts interaction patterns from MeshViewer (hide mode, reset, actor map).
# Joelle 2025-11-13

import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os, io, fnmatch
import numpy as np
import posixpath
import tempfile

# ---------- SFTP utilities ----------
def stat_isdir(st_mode):
    import stat
    return stat.S_ISDIR(st_mode)

def sftp_walk(sftp, remote_path):
    dirnames, filenames = [], []
    for attr in sftp.listdir_attr(remote_path):
        name = attr.filename
        rpath = posixpath.join(remote_path, name)
        if stat_isdir(attr.st_mode):
            dirnames.append(name)
        else:
            filenames.append(name)
    yield remote_path, dirnames, filenames
    for d in dirnames:
        for x in sftp_walk(sftp, posixpath.join(remote_path, d)):
            yield x

def download_tree(sftp, remote_root, local_root, patterns=None):
    os.makedirs(local_root, exist_ok=True)
    for dirpath, _, filenames in sftp_walk(sftp, remote_root):
        rel = os.path.relpath(dirpath, remote_root).replace("\\", "/")
        ldir = os.path.join(local_root, rel) if rel != "." else local_root
        os.makedirs(ldir, exist_ok=True)
        for fn in filenames:
            if patterns and not any(fnmatch.fnmatch(fn, p) for p in patterns):
                continue
            sftp.get(posixpath.join(dirpath, fn), os.path.join(ldir, fn))

# ---------- helpers for derived variables ----------
_GAMMA_DEFAULT = 1.4

def _find_array_name(names, candidates):
    """Return the first matching candidate (case-insensitive) in `names`."""
    low = {n.lower(): n for n in names or [] if isinstance(n, str)}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _is_vector(arr):
    try:
        return arr.ndim == 2 and arr.shape[1] in (2,3)
    except Exception:
        return False

def _append_or_replace(arrdict, name, data):
    """Add `data` under `name` to a pyvista dataset's point_data or cell_data dict-like."""
    try:
        arrdict[name] = data
    except Exception:
        # Some composite blocks may be read-only; ignore silently
        pass

# ---------- Main widget ----------
class PostViewer(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window

        self._mesh = None                # pv.MultiBlock or dataset
        self._actors = {}                # name -> actor
        self._hidden = set()
        self._clip_widgets = []          # active plane widgets
        self._clipped_actor = None
        self._gamma = _GAMMA_DEFAULT

        # ---------- UI ----------
        root = QVBoxLayout(self)

        # Row 1: HPC file controls + variable chooser
        row1 = QHBoxLayout()
        self.refresh_btn  = QPushButton("Refresh post-proc jobs")
        self.download_btn = QPushButton("Download && Open")
        row1.addWidget(self.refresh_btn)
        row1.addWidget(self.download_btn)
        row1.addStretch(1)

        row1.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(220)
        row1.addWidget(self.field_combo)

        self.gamma_edit = QLineEdit(str(self._gamma))
        self.gamma_edit.setFixedWidth(50)
        self.gamma_edit.setToolTip("γ (ratio of specific heats) used for Mach and p₀")
        row1.addWidget(QLabel("γ:"))
        row1.addWidget(self.gamma_edit)

        root.addLayout(row1)

        # Row 2: actions (hide/reset/clip)
        row2 = QHBoxLayout()
        self.reset_cam_btn = QPushButton("Reset Camera")
        self.reset_vis_btn = QPushButton("Reset Surfaces")
        self.hide_mode_btn = QPushButton("Hide Surface (pick)")
        self.add_clip_btn  = QPushButton("Add Clip Plane")
        self.clear_clip_btn= QPushButton("Clear Clips")
        self.hide_orig_chk   = QCheckBox("Hide Original")
        self.freeze_clip_btn = QPushButton("Freeze Clip")
        
        row2.addWidget(self.reset_cam_btn)
        row2.addWidget(self.reset_vis_btn)
        row2.addWidget(self.hide_mode_btn)
        row2.addWidget(self.add_clip_btn)
        row2.addWidget(self.clear_clip_btn)
        row2.addWidget(self.hide_orig_chk)
        row2.addWidget(self.freeze_clip_btn)
        row2.addStretch(1)
        root.addLayout(row2)

        for btn in [self.reset_cam_btn, self.reset_vis_btn, self.hide_mode_btn, self.add_clip_btn, self.clear_clip_btn, self.hide_orig_chk, self.freeze_clip_btn]:
            btn.setEnabled(False)
        
        self.hide_orig_chk.toggled.connect(self._toggle_original_visibility)
        self.freeze_clip_btn.clicked.connect(self._freeze_current_clip)
        
        # Jobs list
        self.jobs = QListWidget()
        self.jobs.setFixedHeight(100)
        root.addWidget(self.jobs)

        # Embedded PyVista view
        self.plotter = QtInteractor(self)
        self.plotter.setVisible(True)
        root.addWidget(self.plotter)

        # Wiring
        self.refresh_btn.clicked.connect(self.refresh_jobs)
        self.download_btn.clicked.connect(self.download_and_open)
        self.field_combo.currentTextChanged.connect(self._update_scalars)
        self.reset_cam_btn.clicked.connect(self._reset_camera)
        self.reset_vis_btn.clicked.connect(self._reset_visibility)
        self.hide_mode_btn.clicked.connect(self._toggle_hide_mode)
        self.add_clip_btn.clicked.connect(self._add_clip_plane)
        self.clear_clip_btn.clicked.connect(self._clear_clips)

        self._hide_mode = False

    # ---------- HPC postprocessed root ----------
    def _post_root(self):
        if not getattr(self.mw, "remote_output_dir", None):
            QMessageBox.information(self, "HPC output not set",
                                    "Please set an output directory and mirror to HPC first.")
            return None
        return posixpath.join(self.mw.remote_output_dir, "postprocessed")

    # ---------- Populate jobs ----------
    def refresh_jobs(self):
        self.jobs.clear()
        try:
            sftp = self.mw.ssh_client.open_sftp()
        except Exception as e:
            QMessageBox.critical(self, "SSH error", str(e)); return

        try:
            root = self._post_root()
            if not root:
                return

            def safe_listdir(path):
                try:
                    return sftp.listdir_attr(path)
                except Exception:
                    return []

            for attr_n in safe_listdir(root):
                if not stat_isdir(attr_n.st_mode) or not attr_n.filename.startswith("n_"): continue
                npath = posixpath.join(root, attr_n.filename)
                for attr_c in safe_listdir(npath):
                    if not stat_isdir(attr_c.st_mode) or not attr_c.filename.startswith("cond_"): continue
                    cpath = posixpath.join(npath, attr_c.filename)
                    for attr_f in safe_listdir(cpath):
                        if attr_f.filename.lower().endswith(".case"):
                            self.jobs.addItem(posixpath.join(cpath, attr_f.filename))
        finally:
            sftp.close()

    # ---------- Download case and open ----------
    def download_and_open(self):
        it = self.jobs.currentItem()
        if not it:
            QMessageBox.information(self, "No selection", "Select a .case job first.")
            return

        remote_case = it.text()
        remote_dir  = posixpath.dirname(remote_case)

        local_root = QFileDialog.getExistingDirectory(self, "Select local folder to store EnSight case")
        if not local_root:
            local_root = tempfile.mkdtemp(prefix="ensight_")
        local_dir = os.path.join(local_root, os.path.basename(remote_dir))

        # SFTP copy with simple progress
        try:
            sftp = self.mw.ssh_client.open_sftp()
        except Exception as e:
            QMessageBox.critical(self, "SSH error", str(e)); return

        total = 0
        stack = [remote_dir]
        try:
            while stack:
                d = stack.pop()
                for a in sftp.listdir_attr(d):
                    if stat_isdir(a.st_mode):
                        stack.append(posixpath.join(d, a.filename))
                    else:
                        total += 1
        except Exception:
            pass

        prog = QProgressDialog("Downloading EnSight case...", "Cancel", 0, max(1,total), self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
        done = 0
        os.makedirs(local_dir, exist_ok=True)
        stack = [remote_dir]
        ok = True
        while stack and ok:
            d = stack.pop()
            rel = os.path.relpath(d, remote_dir).replace("\\","/")
            ldir = os.path.join(local_dir, rel) if rel!="." else local_dir
            os.makedirs(ldir, exist_ok=True)
            for a in sftp.listdir_attr(d):
                if prog.wasCanceled(): ok=False; break
                rpath = posixpath.join(d, a.filename)
                if stat_isdir(a.st_mode):
                    stack.append(rpath)
                else:
                    try:
                        sftp.get(rpath, os.path.join(ldir, a.filename))
                    except Exception as ee:
                        QMessageBox.critical(self, "SFTP error", f"Failed to download:\n{rpath}\n\n{ee}")
                        ok=False; break
                    done += 1; prog.setValue(min(done,total)); QApplication.processEvents()
        try: sftp.close()
        except: pass
        prog.close()
        if not ok: return

        local_case = os.path.join(local_dir, os.path.basename(remote_case))
        if not os.path.exists(local_case):
            QMessageBox.critical(self, "Read error",
                                 f"Case file not found in the downloaded folder:\n{local_case}")
            return

        try:
            self._mesh = pv.read(local_case)
        except Exception as e:
            QMessageBox.critical(self, "Read error", f"Failed to read EnSight case:\n{e}")
            return

        # Build scene
        self._build_scene()
        self._populate_field_menu()
        
        for btn in [self.reset_cam_btn, self.reset_vis_btn, self.hide_mode_btn, self.add_clip_btn, self.clear_clip_btn, self.hide_orig_chk, self.freeze_clip_btn]:
            btn.setEnabled(True)

        QMessageBox.information(self, "Loaded", f"Downloaded to:\n{local_dir}\nOpened in viewer.")

    # ---------- Scene management ----------
    def _build_scene(self):
        self.plotter.clear()
        self._actors.clear()
        self._hidden.clear()
        self._clear_clips()

        if isinstance(self._mesh, pv.MultiBlock):
            # Add each block with an actor (color cycling handled by PyVista)
            for i, blk in enumerate(self._mesh):
                if blk is None: continue
                name = f"Block {i}"
                actor = self.plotter.add_mesh(blk, name=name, show_edges=False, pickable=True)
                self._actors[name] = actor
        else:
            name = "Dataset"
            actor = self.plotter.add_mesh(self._mesh, name=name, show_edges=False, pickable=True)
            self._actors[name] = actor

        self.plotter.reset_camera()
        self.plotter.enable_anti_aliasing()
        self.plotter.render()

    def _reset_camera(self):
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def _reset_visibility(self):
        for name, actor in self._actors.items():
            try:
                actor.SetVisibility(True)
            except Exception:
                pass
        self._hidden.clear()
        self.plotter.render()

    # ---------- Hide mode ----------
    def _toggle_hide_mode(self):
        if not self._hide_mode:
            # enable
            self.plotter.enable_mesh_picking(callback=self._hide_actor, show_message=True,
                                             use_actor=True, font_size=12)
            self.plotter.add_text("Hide mode ON", name="hide_label",
                                  position='lower_right', font_size=6)
            self.hide_mode_btn.setText("Exit Hide Mode")
            self._hide_mode = True
        else:
            # disable
            self.plotter.disable_picking()
            try: self.plotter.remove_actor("hide_label")
            except Exception: pass
            self.hide_mode_btn.setText("Hide Surface (pick)")
            self._hide_mode = False

    def _hide_actor(self, actor):
        for name, act in self._actors.items():
            if act == actor:
                try:
                    act.SetVisibility(False)
                    self._hidden.add(name)
                    self.plotter.render()
                except Exception:
                    pass
                break

    # ---------- Clip planes ----------
    def _add_clip_plane(self):
        if self._mesh is None:
            return

        # On-demand clipped overlay (not destructive to original)
        def _cb(normal, origin):
            try:
                src = self._mesh
                # Build a clipped dataset (composite-aware)
                if isinstance(src, pv.MultiBlock):
                    clipped = pv.MultiBlock()
                    for i, blk in enumerate(src):
                        if blk is None: 
                            clipped.append(None); continue
                        clipped.append(blk.clip(normal=normal, origin=origin))
                else:
                    clipped = src.clip(normal=normal, origin=origin)

                # Remove previous clipped actor overlay
                if self._clipped_actor is not None:
                    try: self.plotter.remove_actor(self._clipped_actor)
                    except Exception: pass
                    self._clipped_actor = None

                # Add overlay (semi-opaque)
                self._clipped_actor = self.plotter.add_mesh(
                    clipped, opacity=0.7, show_edges=False, pickable=False
                )
                self.plotter.render()
            except Exception as e:
                QMessageBox.warning(self, "Clip error", str(e))

        # Plane widget (user can drag)
        wid = self.plotter.add_plane_widget(
            callback=_cb, normal='x', origin=None, assign_to_axis=None
        )
        self._clip_widgets.append(wid)

    def _clear_clips(self):
        # Remove plane widgets and overlay actor
        for wid in self._clip_widgets:
            try:
                self.plotter.remove_actor(wid)  # plane widgets are actors internally
            except Exception:
                pass
        self._clip_widgets = []
        if self._clipped_actor is not None:
            try: self.plotter.remove_actor(self._clipped_actor)
            except Exception: pass
            self._clipped_actor = None
        self.plotter.render()
        
    def _toggle_original_visibility(self, hide):
        # hide/show the uncut actors while you position the plane
        for name, actor in self._actors.items():
            try:
                actor.SetVisibility(not hide)
            except Exception:
                pass
        self.plotter.render()

    def _freeze_current_clip(self):
        """Replace the current scene mesh with the latest clipped overlay."""
        if self._clipped_actor is None:
            QMessageBox.information(self, "Freeze Clip", "No active clip to freeze.")
            return

        try:
            # Get the dataset back from the actor’s mapper input (PyVista helper)
            clipped_ds = self._clipped_actor.mapper().GetInputAsDataSet()
            # Convert to a PyVista dataset
            clipped_pv = pv.wrap(clipped_ds)

            # Replace the source mesh and rebuild the scene
            self._mesh = clipped_pv
            self._build_scene()
            # Re-apply current field coloring
            self._update_scalars(self.field_combo.currentText())

            # Clear plane widgets/overlay
            self._clear_clips()
            self.hide_orig_chk.setChecked(False)
        except Exception as e:
            QMessageBox.warning(self, "Freeze Clip", f"Could not freeze clip:\n{e}")

    # ---------- Field menu + derived arrays ----------
    def _populate_field_menu(self):
        # Base derived options
        derived = [
            "Density (ρ)",
            "Velocity Magnitude |U|",
            "Mach",
            "Pressure (static)",
            "Pressure (total)",
        ]

        # Raw arrays found in the case (point or cell arrays across blocks)
        raw = set()
        def _collect(ds):
            try:
                for n in (ds.array_names or []):
                    if isinstance(n, str):
                        raw.add(n)
            except Exception:
                pass

        if isinstance(self._mesh, pv.MultiBlock):
            for blk in self._mesh:
                if blk is not None:
                    _collect(blk)
        else:
            _collect(self._mesh)

        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(derived + ["—"] + sorted(raw))
        self.field_combo.setCurrentText("Mach")
        self.field_combo.blockSignals(False)

        # Compute and shade initial field
        self._update_scalars(self.field_combo.currentText())

    def _update_scalars(self, label):
        if self._mesh is None or not label:
            return

        # parse gamma
        try:
            self._gamma = float(self.gamma_edit.text())
        except Exception:
            self._gamma = _GAMMA_DEFAULT
            self.gamma_edit.setText(str(self._gamma))

        # For derived fields we (re)compute arrays and then color by that array name
        if label in ("Density (ρ)", "Velocity Magnitude |U|", "Mach", "Pressure (static)", "Pressure (total)"):
            self._compute_and_color(label)
        elif label == "—":
            # no coloring
            self._shade_by(None)
        else:
            # raw array name
            self._shade_by(label)

    def _compute_and_color(self, what):
        if isinstance(self._mesh, pv.MultiBlock):
            for blk in self._mesh:
                if blk is None: continue
                self._attach_derived_arrays(blk)
        else:
            self._attach_derived_arrays(self._mesh)

        key = None
        if what == "Density (ρ)":
            key = "_rho_"
        elif what == "Velocity Magnitude |U|":
            key = "_Umag_"
        elif what == "Mach":
            key = "_Mach_"
        elif what == "Pressure (static)":
            key = "_p_"              # this will be p_fromE if computed, else native p
        elif what == "Pressure (total)":
            key = "_p0_"

        self._shade_by(key)

    def _shade_by(self, array_name):
        # Re-add meshes with scalars applied (fast path: update mapper on existing actors)
        self.plotter.clear()
        self._actors.clear()

        def _add(ds, name):
            try:
                if array_name and array_name in (ds.array_names or []):
                    actor = self.plotter.add_mesh(ds, scalars=array_name, show_edges=False, pickable=True)
                else:
                    actor = self.plotter.add_mesh(ds, show_edges=False, pickable=True)
                self._actors[name] = actor
            except Exception:
                pass

        if isinstance(self._mesh, pv.MultiBlock):
            for i, blk in enumerate(self._mesh):
                if blk is None: continue
                _add(blk, f"Block {i}")
        else:
            _add(self._mesh, "Dataset")

        # keep hidden set honoured
        for name in list(self._hidden):
            act = self._actors.get(name)
            if act is not None:
                try: act.SetVisibility(False)
                except Exception: pass

        self.plotter.add_scalar_bar(title=array_name or "")
        self.plotter.reset_camera()
        self.plotter.render()

    def _attach_derived_arrays(self, ds):
        """
        Compute ρ, |U|, p (from energy), Mach, p0 and store them as:
        _rho_, _Umag_, _p_fromE_, _Mach_, _p0_
        Works whether the dataset is point or cell based; we place results on POINT data.
        """
        try:
            names = list(ds.array_names or [])
        except Exception:
            names = []

        # --- find canonical arrays (case-insensitive) ---
        rho_name = _find_array_name(names, ["density", "rho", "RHO"])
        e_name   = _find_array_name(names, ["energy", "E", "Energy"])
        U_name   = _find_array_name(names, ["velocity", "U", "VEL"])

        # Try component fallback for velocity if needed
        U = None
        if U_name is not None:
            U = ds[U_name]
        else:
            ux = _find_array_name(names, ["u","Ux","velocity_0","vel_0"])
            uy = _find_array_name(names, ["v","Uy","velocity_1","vel_1"])
            uz = _find_array_name(names, ["w","Uz","velocity_2","vel_2"])
            if ux and uy and uz:
                U = np.column_stack([ds[ux], ds[uy], ds[uz]])

        # --- expose density ---
        if rho_name is not None:
            rho = np.asarray(ds[rho_name])
            _append_or_replace(ds.point_data, "_rho_", rho)
        else:
            rho = None

        # --- |U| ---
        Umag = None
        if U is not None:
            Umag = np.linalg.norm(U, axis=1) if _is_vector(U) else np.asarray(U)
            _append_or_replace(ds.point_data, "_Umag_", Umag)

        # --- static p from energy (ParaView formula you use) ---
        # p = (E - 0.5*rho*|U|^2)*(gamma-1)
        p_fromE = None
        if (e_name is not None) and (rho is not None) and (Umag is not None):
            E = np.asarray(ds[e_name])
            p_fromE = (E - 0.5 * rho * (Umag**2)) * (self._gamma - 1.0)
            # numeric hygiene
            p_fromE = np.where(np.isfinite(p_fromE), p_fromE, 0.0)
            p_fromE = np.clip(p_fromE, a_min=0.0, a_max=None)
            _append_or_replace(ds.point_data, "_p_fromE_", p_fromE)

        # If a native pressure exists, stash it too (optional)
        p_native_name = _find_array_name(names, ["p", "pressure", "Pressure", "P"])
        if p_native_name is not None:
            _append_or_replace(ds.point_data, "_p_native_", np.asarray(ds[p_native_name]))

        # --- Mach & p0 using p_fromE if available (else native p if present) ---
        # M = |U| / sqrt(gamma * p / rho)
        if (rho is not None) and (Umag is not None):
            p_for_M = p_fromE
            if p_for_M is None and p_native_name is not None:
                p_for_M = np.asarray(ds[p_native_name])

            if p_for_M is not None:
                a = np.sqrt(np.maximum(self._gamma * p_for_M / np.maximum(rho, 1e-20), 1e-30))
                Mach = Umag / a
                Mach = np.where(np.isfinite(Mach), Mach, 0.0)
                _append_or_replace(ds.point_data, "_Mach_", Mach)

                p0 = p_for_M * np.power(1.0 + 0.5*(self._gamma - 1.0)*Mach*Mach,
                                        self._gamma/(self._gamma - 1.0))
                p0 = np.where(np.isfinite(p0), p0, 0.0)
                _append_or_replace(ds.point_data, "_p0_", p0)

        # Also expose “plain” static pressure key for coloring convenience:
        # prefer p_fromE, then native p if present.
        if p_fromE is not None:
            _append_or_replace(ds.point_data, "_p_", p_fromE)
        elif p_native_name is not None:
            _append_or_replace(ds.point_data, "_p_", np.asarray(ds[p_native_name]))
