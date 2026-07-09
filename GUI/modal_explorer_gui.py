"""
modal_slider_explorer.py

Standalone PyQt5 + PyVista GUI for visually exploring Laplacian/modal
control-node deformation modes before integrating them into AerOpt.

What it does
------------
1. Loads a surface mesh (.vtk, .vtp, .ply, .stl, .obj, .vtm supported by PyVista).
2. Loads control nodes from .npy/.txt/.csv.
3. Loads optional control-node normals from .npy/.txt/.csv.
   If normals are not supplied, they are estimated from the mesh points.
4. Builds a graph-Laplacian modal basis on the control nodes.
5. Exposes one slider per modal coefficient with 0.01 resolution.
6. Propagates the resulting control-node displacement to the mesh using RBF.
7. Updates the mesh interactively and colours it by displacement magnitude.

Typical usage
-------------
python modal_slider_explorer.py

or

python modal_slider_explorer.py \
    --mesh path/to/surface.vtk \
    --control-nodes path/to/control_nodes.npy \
    --control-normals path/to/control_normals.npy \
    --k-modes 8

Dependencies
------------
PyQt5, pyvista, pyvistaqt, numpy, scipy, scikit-learn

Notes
-----
- This is intentionally standalone. It does not require AerOpt imports.
- For interpretation, start with one mode at a time using the "Solo selected mode"
  workflow, then combine modes gradually.
- The RBF interpolation is for visual exploration. Your production AerOpt morphing
  may use a different RBF/constraint treatment; this GUI is meant to reveal the
  qualitative behaviour of the modal coefficients.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.path.dirname("ShapeParameterization"))
from ShapeParameterization.controlNodeDisp import getDisplacements

# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------


def load_array(path: str, cols: int = 3) -> np.ndarray:
    """Load an array from .npy, .txt, or .csv and force shape (N, cols)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        delimiter = "," if ext == ".csv" else None
        arr = np.loadtxt(path, delimiter=delimiter)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, cols))
    if arr.shape[1] != cols:
        raise ValueError(f"Expected array with {cols} columns. Got shape {arr.shape} from {path}")
    return arr


def estimate_normals_from_mesh_points(query_pts: np.ndarray, mesh_pts: np.ndarray, knn: int = 16) -> np.ndarray:
    """Estimate normals at query points by local PCA on nearby mesh points."""
    from sklearn.neighbors import NearestNeighbors

    query_pts = np.asarray(query_pts, dtype=float)
    mesh_pts = np.asarray(mesh_pts, dtype=float)
    k = min(max(4, int(knn)), max(1, mesh_pts.shape[0] - 1))

    nbrs = NearestNeighbors(n_neighbors=k).fit(mesh_pts)
    idx = nbrs.kneighbors(query_pts, return_distance=False)

    centroid = mesh_pts.mean(axis=0)
    normals = np.zeros_like(query_pts, dtype=float)

    for i, row in enumerate(idx):
        q = mesh_pts[row]
        qc = q - q.mean(axis=0, keepdims=True)
        cov = qc.T @ qc
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0]
        n /= np.linalg.norm(n) + 1e-12

        # crude but stable outward orientation relative to mesh centroid
        sign = np.sign(np.dot(query_pts[i] - centroid, n))
        if sign == 0:
            sign = 1.0
        normals[i] = sign * n

    return normals


def build_knn_laplacian_basis(control_nodes: np.ndarray, k_modes: int, knn: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a symmetric kNN graph Laplacian basis on the control nodes.

    Returns
    -------
    evals : (k_modes,) eigenvalues, excluding the constant zero mode when possible
    phi   : (N, k_modes) eigenvectors/modes
    """
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(control_nodes, dtype=float)
    n = x.shape[0]
    if n < 2:
        raise ValueError("At least two control nodes are required for a modal basis.")

    k_modes = int(min(max(1, k_modes), n - 1))
    knn = int(min(max(2, knn), n))

    nbrs = NearestNeighbors(n_neighbors=knn).fit(x)
    dists, idx = nbrs.kneighbors(x)

    # characteristic distance for Gaussian weights
    nonzero_d = dists[:, 1:].reshape(-1)
    sigma = float(np.median(nonzero_d[nonzero_d > 1e-12])) if np.any(nonzero_d > 1e-12) else 1.0

    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        for jj, d in zip(idx[i, 1:], dists[i, 1:]):
            wij = np.exp(-0.5 * (float(d) / (sigma + 1e-12)) ** 2)
            w[i, int(jj)] = max(w[i, int(jj)], wij)
            w[int(jj), i] = max(w[int(jj), i], wij)

    degree = np.sum(w, axis=1)
    lap = np.diag(degree) - w

    evals_all, evecs_all = np.linalg.eigh(lap)
    order = np.argsort(evals_all)
    evals_all = evals_all[order]
    evecs_all = evecs_all[:, order]

    # Exclude the constant mode if possible. This makes mode 1 the first non-rigid spatial variation.
    start = 1 if n > 1 else 0
    evals = evals_all[start : start + k_modes]
    phi = evecs_all[:, start : start + k_modes]

    # IMPORTANT:
    # Do NOT normalise the modes here. The eigenvectors returned by np.linalg.eigh
    # are already L2-normalised. Keeping them untouched means the GUI shows the
    # actual modal displacement implied by the coefficient value, rather than a
    # visually-rescaled version of each mode.
    return evals, phi


def make_control_displacement(
    phi: np.ndarray,
    coeffs: np.ndarray,
    normals: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Convert modal coefficients into 3D control-node displacements."""
    scalar = phi @ coeffs.reshape((-1,))
    return float(amplitude) * scalar[:, None] * normals


def rbf_propagate(
    control_nodes: np.ndarray,
    control_disp: np.ndarray,
    target_pts: np.ndarray,
    kernel: str = "thin_plate_spline",
    smoothing: float = 1e-8,
    neighbors: Optional[int] = None,
) -> np.ndarray:
    """Propagate control-node displacement to mesh points using scipy RBFInterpolator."""
    from scipy.interpolate import RBFInterpolator

    control_nodes = np.asarray(control_nodes, dtype=float)
    control_disp = np.asarray(control_disp, dtype=float)
    target_pts = np.asarray(target_pts, dtype=float)

    if np.linalg.norm(control_disp) < 1e-15:
        return np.zeros_like(target_pts)

    # RBFInterpolator can map vector-valued data directly.
    rbf = RBFInterpolator(
        control_nodes,
        control_disp,
        kernel=kernel,
        smoothing=float(smoothing),
        neighbors=neighbors,
    )
    return np.asarray(rbf(target_pts), dtype=float)


def safe_read_mesh(path: str) -> pv.DataSet:
    """Read mesh and extract a displayable surface if needed."""
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.clean()


@dataclass
class ModalState:
    mesh_path: Optional[str] = None
    control_nodes_path: Optional[str] = None
    control_normals_path: Optional[str] = None

    control_nodes: Optional[np.ndarray] = None
    control_normals: Optional[np.ndarray] = None
    output_dir: Optional[str] = None

    k_modes: int = 6
    knn: int = 6
    seed: int = 0

    amp_alpha: float = 0.005
    t_patch_scale: Optional[float] = None
    normal_project: bool = True
    vector_mode: str = "local_frame"
    frame_knn: int = 12
    global_modes: bool = False
    global_mode_config: Optional[list] = None

    deform_scale: float = 1.0
    rbf_kernel: str = "thin_plate_spline"
    rbf_smoothing: float = 1e-8
    
    graph_method: str = "mutual_knn"
    delaunay_cutoff_factor: float = 2.5

    use_protection: bool = False
    protected_control_nodes: Optional[list] = None
    protection_radius: Optional[float] = None

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


class ModalSliderExplorer(QMainWindow):
    def __init__(self, initial: Optional[ModalState] = None):
        super().__init__()
        self.setWindowTitle("Modal Coefficient Explorer")
        self.resize(1500, 900)

        self.state = initial or ModalState()

        self.mesh: Optional[pv.PolyData] = None
        self.mesh_actor = None
        self.base_points: Optional[np.ndarray] = None
        self.current_disp: Optional[np.ndarray] = None
        self.current_ctrl_disp: Optional[np.ndarray] = None
        self.control_nodes_deformed_poly = None
        self.control_nodes_base_poly = None
        self.control_node_label_actors = []

        self.control_nodes: Optional[np.ndarray] = None
        self.control_normals: Optional[np.ndarray] = None
        self.evals: Optional[np.ndarray] = None
        self.phi: Optional[np.ndarray] = None

        self.coeff_sliders: list[QSlider] = []
        self.coeff_values: list[QDoubleSpinBox] = []
        self.slider_min = -200
        self.slider_max = 200
        self.slider_scale = 100.0  # integer slider value / 100 => coefficient step of 0.01
        self._pending_update = False

        self._build_ui()
        self._closing = False

        if self.state.mesh_path and (
            self.state.control_nodes_path or self.state.control_nodes is not None
        ):
            self.load_case(
                self.state.mesh_path,
                self.state.control_nodes_path,
                self.state.control_normals_path,
                self.state.k_modes,
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        self.setCentralWidget(central)

        # Left controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setFixedWidth(590)
        root.addWidget(left)

        # Right side: PyVista viewer + bottom per-control-node modal table
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(right, stretch=1)

        self.plotter = QtInteractor(self)
        right_layout.addWidget(self.plotter, stretch=5)
        self.plotter.set_background("white")
        try:
            self.plotter.add_axes(line_width=2, labels_off=False)
        except Exception:
            pass

        self.cn_table = QTableWidget(0, 0)
        self.cn_table.setMinimumHeight(190)
        self.cn_table.setMaximumHeight(260)
        self.cn_table.setAlternatingRowColors(True)
        right_layout.addWidget(self.cn_table, stretch=1)

        file_box = QGroupBox("Inputs")
        file_form = QFormLayout(file_box)

        self.mesh_label = QLabel("No mesh loaded")
        self.cn_label = QLabel("No control nodes loaded")
        self.nn_label = QLabel("Optional")

        mesh_btn = QPushButton("Load surface mesh")
        mesh_btn.clicked.connect(self.on_load_mesh)
        cn_btn = QPushButton("Load control nodes")
        cn_btn.clicked.connect(self.on_load_control_nodes)
        nn_btn = QPushButton("Load control normals")
        nn_btn.clicked.connect(self.on_load_control_normals)

        file_form.addRow(mesh_btn, self.mesh_label)
        file_form.addRow(cn_btn, self.cn_label)
        file_form.addRow(nn_btn, self.nn_label)
        left_layout.addWidget(file_box)

        settings_box = QGroupBox("Modal/RBF settings")
        settings_form = QFormLayout(settings_box)

        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 1000)
        self.k_spin.setValue(self.state.k_modes)
        settings_form.addRow("Number of modes", self.k_spin)

        self.knn_spin = QSpinBox()
        self.knn_spin.setRange(2, 100)
        self.knn_spin.setValue(self.state.knn)
        settings_form.addRow("Graph kNN", self.knn_spin)

        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setDecimals(6)
        self.amp_spin.setRange(1e-9, 1e9)
        self.amp_spin.setSingleStep(0.001)
        self.amp_spin.setValue(self.state.amp_alpha)
        settings_form.addRow("amp_alpha", self.amp_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(3)
        self.scale_spin.setRange(0.001, 1000.0)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setValue(self.state.deform_scale)
        self.scale_spin.valueChanged.connect(lambda _v: self.schedule_update())
        settings_form.addRow("Visual scale", self.scale_spin)

        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems([
            "thin_plate_spline",
            "cubic",
            "quintic",
            "linear",
            "multiquadric",
            "inverse_multiquadric",
            "inverse_quadratic",
            "gaussian",
        ])
        self.kernel_combo.setCurrentText(self.state.rbf_kernel)
        self.kernel_combo.currentTextChanged.connect(lambda _v: self.schedule_update())
        settings_form.addRow("RBF kernel", self.kernel_combo)

        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setDecimals(12)
        self.smooth_spin.setRange(0.0, 1.0)
        self.smooth_spin.setSingleStep(1e-6)
        self.smooth_spin.setValue(self.state.rbf_smoothing)
        self.smooth_spin.valueChanged.connect(lambda _v: self.schedule_update())
        settings_form.addRow("RBF smoothing", self.smooth_spin)

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems([
            "Displacement magnitude |u|",
            "Signed normal displacement u·n",
            "X displacement ux",
            "Y displacement uy",
            "Z displacement uz",
        ])
        self.color_mode_combo.currentTextChanged.connect(lambda _v: self.refresh_actor())
        settings_form.addRow("Colour field", self.color_mode_combo)

        self.show_baseline_check = QCheckBox("Show baseline wireframe")
        self.show_baseline_check.setChecked(True)
        self.show_baseline_check.stateChanged.connect(lambda _v: self.refresh_actor())
        settings_form.addRow(self.show_baseline_check)

        self.show_cn_base_check = QCheckBox("Show original control nodes")
        self.show_cn_base_check.setChecked(True)
        self.show_cn_base_check.stateChanged.connect(lambda _v: self.refresh_control_node_actors())
        settings_form.addRow(self.show_cn_base_check)

        self.show_cn_deformed_check = QCheckBox("Show deformed control nodes")
        self.show_cn_deformed_check.setChecked(True)
        self.show_cn_deformed_check.stateChanged.connect(lambda _v: self.refresh_control_node_actors())
        settings_form.addRow(self.show_cn_deformed_check)

        self.show_cn_labels_check = QCheckBox("Show control-node modal labels")
        self.show_cn_labels_check.setChecked(False)
        self.show_cn_labels_check.stateChanged.connect(lambda _v: self.update_control_node_labels())
        settings_form.addRow(self.show_cn_labels_check)

        self.edges_check = QCheckBox("Show mesh edges")
        self.edges_check.setChecked(False)
        self.edges_check.stateChanged.connect(lambda _v: self.refresh_actor())
        settings_form.addRow(self.edges_check)

        build_btn = QPushButton("Load modal basis cache")
        build_btn.clicked.connect(self.on_build_basis)
        settings_form.addRow(build_btn)
        left_layout.addWidget(settings_box)

        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)

        zero_btn = QPushButton("Reset all coefficients to zero")
        zero_btn.clicked.connect(self.zero_all_coefficients)
        actions_layout.addWidget(zero_btn)

        random_btn = QPushButton("Random small coefficients")
        random_btn.clicked.connect(self.random_coefficients)
        actions_layout.addWidget(random_btn)

        left_layout.addWidget(actions_box)

        # Scrollable sliders
        slider_group = QGroupBox("Modal coefficients")
        slider_group.setMinimumHeight(330)
        slider_outer = QVBoxLayout(slider_group)
        self.slider_scroll = QScrollArea()
        self.slider_scroll.setWidgetResizable(True)
        self.slider_container = QWidget()
        self.slider_layout = QVBoxLayout(self.slider_container)
        self.slider_scroll.setWidget(self.slider_container)
        slider_outer.addWidget(self.slider_scroll)
        left_layout.addWidget(slider_group, stretch=1)

        # Mode metrics table
        self.mode_table = QTableWidget(0, 4)
        self.mode_table.setHorizontalHeaderLabels(["Mode", "Eigenvalue", "Roughness", "Notes"])
        self.mode_table.setMaximumHeight(135)
        left_layout.addWidget(self.mode_table)

        self.status = QLabel("Load a mesh and control nodes to begin.")
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)
        
    def closeEvent(self, event):
        self._closing = True
        self._pending_update = False

        try:
            for actor in getattr(self, "control_node_label_actors", []):
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.control_node_label_actors = []

            if getattr(self, "plotter", None) is not None:
                try:
                    self.plotter.disable_picking()
                except Exception:
                    pass
                try:
                    self.plotter.clear()
                except Exception:
                    pass
                try:
                    rw = getattr(self.plotter, "ren_win", None)
                    if rw is not None:
                        rw.Finalize()
                except Exception:
                    pass
                try:
                    iren = getattr(self.plotter, "interactor", None)
                    if iren is not None:
                        iren.TerminateApp()
                except Exception:
                    pass
                try:
                    self.plotter.close()
                except Exception:
                    pass

            self.plotter = None
            self.mesh = None
            self.base_points = None
            self.current_disp = None
            self.current_ctrl_disp = None

        except Exception as e:
            print(f"[DEBUG] Modal explorer close cleanup failed: {e}")

        event.accept()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def on_load_mesh(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load surface mesh",
            "",
            "Mesh files (*.vtk *.vtp *.vtu *.vtm *.ply *.stl *.obj);;All files (*)",
        )
        if path:
            self.state.mesh_path = path
            self.mesh_label.setText(os.path.basename(path))
            self.try_load_current_case()

    def on_load_control_nodes(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load control nodes",
            "",
            "Array files (*.npy *.txt *.csv);;All files (*)",
        )
        if path:
            self.state.control_nodes_path = path
            self.cn_label.setText(os.path.basename(path))
            self.try_load_current_case()

    def on_load_control_normals(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load control normals",
            "",
            "Array files (*.npy *.txt *.csv);;All files (*)",
        )
        if path:
            self.state.control_normals_path = path
            self.nn_label.setText(os.path.basename(path))
            self.try_load_current_case()

    def try_load_current_case(self):
        if self.state.mesh_path and (
            self.state.control_nodes_path or self.state.control_nodes is not None
        ):
            self.load_case(
                self.state.mesh_path,
                self.state.control_nodes_path,
                self.state.control_normals_path,
                int(self.k_spin.value()),
            )

    def load_case(self, mesh_path: str, control_nodes_path: str, control_normals_path: Optional[str], k_modes: int):
        try:
            self.mesh = safe_read_mesh(mesh_path)
            self.base_points = np.asarray(self.mesh.points, dtype=float).copy()
            self.current_disp = np.zeros_like(self.base_points)

            if self.state.control_nodes is not None:
                self.control_nodes = np.asarray(self.state.control_nodes, dtype=float).reshape((-1, 3))
            else:
                self.control_nodes = load_array(control_nodes_path, cols=3)

            if self.state.control_normals is not None:
                self.control_normals = np.asarray(self.state.control_normals, dtype=float).reshape((-1, 3))
            elif control_normals_path:
                self.control_normals = load_array(control_normals_path, cols=3)
            else:
                self.control_normals = estimate_normals_from_mesh_points(
                    self.control_nodes,
                    self.base_points,
                    knn=16,
                )
                self.nn_label.setText("Estimated from mesh")

            self.mesh_label.setText(os.path.basename(mesh_path))

            if control_nodes_path:
                self.cn_label.setText(os.path.basename(control_nodes_path))
            else:
                self.cn_label.setText(f"Loaded from MeshViewer ({len(self.control_nodes)} nodes)")

            if control_normals_path:
                self.nn_label.setText(os.path.basename(control_normals_path))
            elif self.state.control_normals is not None:
                self.nn_label.setText("Loaded from MeshViewer")

            max_modes = int(self.state.k_modes)
            self.k_spin.setMaximum(max_modes)
            self.k_spin.setValue(min(int(k_modes), max_modes))

            self.on_build_basis()
            self.draw_initial_scene()

            self.status.setText(
                f"Loaded mesh with {self.mesh.n_points} points and {self.mesh.n_cells} cells. "
                f"Loaded {self.control_nodes.shape[0]} control nodes."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self.status.setText(f"Load failed: {exc}")

    # ------------------------------------------------------------------
    # Basis/sliders
    # ------------------------------------------------------------------

    def on_build_basis(self):
        if self.control_nodes is None:
            return
        try:
            modal_cache_path = os.path.join(
            self.state.output_dir,
                "Control Nodes",
                "modal_basis_T_surface.npz"
            )

            cache = np.load(modal_cache_path)
            phi_T = np.asarray(cache["phi_T"], float)
            idx = np.asarray(cache["control_node_point_indices"], dtype=int)

            self.phi = phi_T[idx]
            self.evals = np.arange(self.phi.shape[1], dtype=float)  # or save/load eigenvalues later
            self.rebuild_sliders()
            self.update_mode_table()
            self.update_control_node_table()
            self.schedule_update()
            self.status.setText(f"Built modal basis: phi shape = {self.phi.shape}")
        except Exception as exc:
            QMessageBox.critical(self, "Basis build failed", str(exc))
            self.status.setText(f"Basis build failed: {exc}")

    def rebuild_sliders(self):
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.coeff_sliders = []
        self.coeff_values = []
        if self.phi is None:
            return

        for i in range(self.phi.shape[1]):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"a{i + 1}")
            label.setFixedWidth(35)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(self.slider_min, self.slider_max)
            slider.setSingleStep(1)       # 0.01 coefficient step
            slider.setPageStep(10)        # 0.10 coefficient step
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(50)    # every 0.50

            value_box = QDoubleSpinBox()
            value_box.setDecimals(2)
            value_box.setRange(self.slider_min / self.slider_scale, self.slider_max / self.slider_scale)
            value_box.setSingleStep(0.01)
            value_box.setValue(0.0)
            value_box.setFixedWidth(75)

            def _slider_changed(v, box=value_box):
                coeff = float(v) / self.slider_scale
                box.blockSignals(True)
                box.setValue(coeff)
                box.blockSignals(False)
                self.schedule_update()

            def _box_changed(v, sl=slider):
                iv = int(round(float(v) * self.slider_scale))
                sl.blockSignals(True)
                sl.setValue(iv)
                sl.blockSignals(False)
                self.schedule_update()

            slider.valueChanged.connect(_slider_changed)
            value_box.valueChanged.connect(_box_changed)

            solo_pos_btn = QPushButton("Solo +")
            solo_pos_btn.setFixedWidth(65)
            solo_pos_btn.clicked.connect(lambda _checked=False, idx=i: self.solo_mode(idx, value=1.0))

            solo_neg_btn = QPushButton("Solo -")
            solo_neg_btn.setFixedWidth(65)
            solo_neg_btn.clicked.connect(lambda _checked=False, idx=i: self.solo_mode(idx, value=-1.0))

            zero_btn = QPushButton("0")
            zero_btn.setFixedWidth(35)
            zero_btn.clicked.connect(lambda _checked=False, sl=slider: sl.setValue(0))

            row_layout.addWidget(label)
            row_layout.addWidget(slider, stretch=1)
            row_layout.addWidget(value_box)
            row_layout.addWidget(solo_pos_btn)
            row_layout.addWidget(solo_neg_btn)
            row_layout.addWidget(zero_btn)

            self.slider_layout.addWidget(row)
            self.coeff_sliders.append(slider)
            self.coeff_values.append(value_box)

        self.slider_layout.addStretch(1)

    def update_mode_table(self):
        if self.phi is None or self.evals is None:
            return
        k = self.phi.shape[1]
        self.mode_table.setRowCount(k)
        for i in range(k):
            eig = float(self.evals[i])
            mode = self.phi[:, i]
            roughness = eig  # for graph Laplacian modes, eigenvalue is a roughness/frequency proxy
            if i < max(3, k // 4):
                note = "low-frequency/global"
            elif i > 3 * k // 4:
                note = "higher-frequency/local"
            else:
                note = "intermediate"

            for col, text in enumerate([
                str(i + 1),
                f"{eig:.4e}",
                f"{roughness:.4e}",
                note,
            ]):
                self.mode_table.setItem(i, col, QTableWidgetItem(text))

        self.mode_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Scene/update
    # ------------------------------------------------------------------

    def draw_initial_scene(self):
        if self.mesh is None:
            return
        self.plotter.clear()
        try:
            self.plotter.add_axes(line_width=2, labels_off=False)
        except Exception:
            pass

        self.refresh_actor(reset_camera=True)

        if self.control_nodes is not None:
            self.control_nodes_base_poly = pv.PolyData(self.control_nodes.copy())
            self.control_nodes_deformed_poly = pv.PolyData(self.control_nodes.copy())
            self.refresh_control_node_actors()
            self.update_control_node_table()

        self.plotter.add_text("Modal coefficient explorer", position="upper_left", font_size=11, color="black")
        self.plotter.reset_camera()
        self.plotter.render()

    def _current_coefficients(self) -> np.ndarray:
        """Current modal coefficients alpha_k from the slider positions."""
        if not self.coeff_sliders:
            return np.zeros(0, dtype=float)
        return np.asarray([s.value() / self.slider_scale for s in self.coeff_sliders], dtype=float)

    def _active_mode_index(self, coeffs: np.ndarray) -> Optional[int]:
        """Return the single active mode index if exactly one coefficient is non-zero."""
        active = np.where(np.abs(coeffs) > 1e-12)[0]
        return int(active[0]) if len(active) == 1 else None

    def refresh_control_node_actors(self):
        """Add/remove original and deformed control-node point actors."""
        if self.control_nodes is None:
            return

        for name in ["control_nodes_base", "control_nodes_deformed"]:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass

        show_base = not hasattr(self, "show_cn_base_check") or self.show_cn_base_check.isChecked()
        show_def = not hasattr(self, "show_cn_deformed_check") or self.show_cn_deformed_check.isChecked()

        if self.control_nodes_base_poly is None:
            self.control_nodes_base_poly = pv.PolyData(self.control_nodes.copy())
        if self.control_nodes_deformed_poly is None:
            self.control_nodes_deformed_poly = pv.PolyData(self.control_nodes.copy())

        if show_base:
            self.plotter.add_mesh(
                self.control_nodes_base_poly,
                color="black",
                point_size=11,
                render_points_as_spheres=True,
                name="control_nodes_base",
                pickable=False,
            )

        if show_def:
            self.plotter.add_mesh(
                self.control_nodes_deformed_poly,
                color="red",
                point_size=15,
                render_points_as_spheres=True,
                name="control_nodes_deformed",
                pickable=False,
            )

        self.update_control_node_labels()
        try:
            self.plotter.render()
        except Exception:
            pass

    def update_control_node_labels(self):
        """Optional compact labels beside control nodes showing phi_k, alpha_k and alpha_k*phi_k."""
        if self.control_nodes is None or self.phi is None:
            return

        for actor in getattr(self, "control_node_label_actors", []):
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.control_node_label_actors = []

        if hasattr(self, "show_cn_labels_check") and not self.show_cn_labels_check.isChecked():
            try:
                self.plotter.render()
            except Exception:
                pass
            return

        coeffs = self._current_coefficients()
        if coeffs.size == 0:
            return

        active_mode = self._active_mode_index(coeffs)
        scalar = self.phi @ coeffs

        pts = self.control_nodes.copy()
        if self.current_ctrl_disp is not None:
            pts = self.control_nodes + float(self.scale_spin.value()) * self.current_ctrl_disp

        labels = []
        for i in range(self.control_nodes.shape[0]):
            if active_mode is None:
                labels.append(f"CN {i+1}\nΣαφ={scalar[i]:+.3e}")
            else:
                k = active_mode
                phi_val = self.phi[i, k]
                alpha_val = coeffs[k]
                labels.append(
                    f"CN {i+1}\n"
                    f"φ{k+1}={phi_val:+.3e}\n"
                    f"α{k+1}={alpha_val:+.3e}\n"
                    f"αφ={alpha_val*phi_val:+.3e}\n"
                    f"Σαφ={scalar[i]:+.3e}"
                )

        actor = self.plotter.add_point_labels(
            pts,
            labels,
            font_size=9,
            text_color="black",
            point_color="red",
            point_size=8,
            render_points_as_spheres=True,
            always_visible=True,
            name="control_node_modal_values",
        )
        self.control_node_label_actors.append(actor)
        try:
            self.plotter.render()
        except Exception:
            pass

    def update_control_node_table(self):
        """Bottom table: one row per control node with phi_k, alpha_k, alpha_k*phi_k and displacement."""
        if not hasattr(self, "cn_table"):
            return

        if self.control_nodes is None or self.control_normals is None or self.phi is None:
            self.cn_table.setRowCount(0)
            self.cn_table.setColumnCount(0)
            return

        coeffs = self._current_coefficients()
        k = self.phi.shape[1]
        if coeffs.size != k:
            coeffs = np.zeros(k, dtype=float)

        scalar = self.phi @ coeffs
        products = self.phi * coeffs.reshape(1, -1)

        if self.current_ctrl_disp is None:
            ctrl_disp = make_control_displacement(
                self.phi,
                coeffs,
                self.control_normals,
                amplitude=float(self.amp_spin.value()),
            )
        else:
            ctrl_disp = np.asarray(self.current_ctrl_disp, dtype=float)

        disp_mag = np.linalg.norm(ctrl_disp, axis=1)

        headers = (
            ["CN", "x", "y", "z", "nx", "ny", "nz"]
            + [f"phi_{j+1}" for j in range(k)]
            + [f"alpha_{j+1}" for j in range(k)]
            + [f"alpha*phi_{j+1}" for j in range(k)]
            + ["sum_alpha_phi", "dx", "dy", "dz", "|dx|"]
        )

        n = self.control_nodes.shape[0]
        self.cn_table.blockSignals(True)
        self.cn_table.setColumnCount(len(headers))
        self.cn_table.setRowCount(n)
        self.cn_table.setHorizontalHeaderLabels(headers)

        def fmt(v, nd=5):
            try:
                return f"{float(v):.{nd}e}"
            except Exception:
                return str(v)

        for i in range(n):
            vals = (
                [i + 1]
                + list(self.control_nodes[i])
                + list(self.control_normals[i])
                + list(self.phi[i, :])
                + list(coeffs)
                + list(products[i, :])
                + [scalar[i], ctrl_disp[i, 0], ctrl_disp[i, 1], ctrl_disp[i, 2], disp_mag[i]]
            )
            for j, val in enumerate(vals):
                item = QTableWidgetItem(str(val) if j == 0 else fmt(val))
                if j == 0:
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.cn_table.setItem(i, j, item)

        self.cn_table.resizeColumnsToContents()
        self.cn_table.blockSignals(False)

    def _mesh_scalar_field(self) -> tuple[str, np.ndarray, str]:
        """Return scalar name, scalar values, scalar-bar title for the current colour mode."""
        if self.current_disp is None:
            vals = np.zeros(self.mesh.n_points)
            return "disp_mag", vals, "|u|"

        mode = self.color_mode_combo.currentText() if hasattr(self, "color_mode_combo") else "Displacement magnitude |u|"
        u = np.asarray(self.current_disp, float)

        if mode.startswith("Signed normal"):
            # For mesh points, estimate sign by projecting displacement onto the radial direction
            # from the mesh centroid. This makes positive/negative regions visible.
            c = self.base_points.mean(axis=0, keepdims=True)
            r = self.base_points - c
            r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
            vals = np.einsum("ij,ij->i", u, r)
            return "signed_un", vals, "signed u"
        if mode.startswith("X displacement"):
            return "ux", u[:, 0], "ux"
        if mode.startswith("Y displacement"):
            return "uy", u[:, 1], "uy"
        if mode.startswith("Z displacement"):
            return "uz", u[:, 2], "uz"

        return "disp_mag", np.linalg.norm(u, axis=1), "|u|"

    def refresh_actor(self, reset_camera: bool = False):
        if self.mesh is None:
            return

        for name in ["baseline_wire", "deformed_mesh"]:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass

        # Baseline overlay, like your animated_morph visualisation.
        if getattr(self, "show_baseline_check", None) is None or self.show_baseline_check.isChecked():
            base = self.mesh.copy(deep=True)
            if self.base_points is not None:
                base.points = self.base_points.copy()
            self.plotter.add_mesh(
                base,
                color="black",
                style="wireframe",
                opacity=0.18,
                line_width=1.0,
                name="baseline_wire",
                show_scalar_bar=False,
            )

        scalar_name, vals, title = self._mesh_scalar_field()
        self.mesh.point_data[scalar_name] = vals

        clim = None
        if vals.size:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if vmin < 0.0 < vmax:
                m = max(abs(vmin), abs(vmax), 1e-14)
                clim = [-m, m]
            elif abs(vmax - vmin) > 1e-14:
                clim = [vmin, vmax]

        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars=scalar_name,
            clim=clim,
            show_edges=self.edges_check.isChecked(),
            smooth_shading=True,
            name="deformed_mesh",
            scalar_bar_args={"title": title},
        )

        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def schedule_update(self):
        if getattr(self, "_closing", False):
            return
        if self._pending_update:
            return
        self._pending_update = True
        QTimer.singleShot(30, self.update_deformation)

    def update_deformation(self):
        self._pending_update = False
        if getattr(self, "_closing", False):
            return
        if self.mesh is None or self.base_points is None or self.phi is None or self.control_normals is None:
            return

        try:
            coeffs = np.asarray([s.value() / self.slider_scale for s in self.coeff_sliders], dtype=float)

            ctrl_disp = getDisplacements(
                output_dir=self.state.output_dir or os.getcwd(),
                seed=int(self.state.seed),
                control_nodes=self.control_nodes,
                normals=self.control_normals,
                coeffs=coeffs,
                k_modes=int(self.k_spin.value()),
                normal_project=bool(self.state.normal_project),
                t_patch_scale=self.state.t_patch_scale,
                amp_alpha=float(self.amp_spin.value()),
                vector_mode=str(self.state.vector_mode),
                frame_knn=int(self.state.frame_knn),
                global_modes=bool(self.state.global_modes),
                global_mode_config=self.state.global_mode_config or [],
                graph_method=str(self.state.graph_method),
                delaunay_cutoff_factor=float(self.state.delaunay_cutoff_factor),
                protected_nodes=(
                    self.state.protected_control_nodes
                    if bool(self.state.use_protection)
                    else []
                ),
                radius=(
                    self.state.protection_radius
                    if bool(self.state.use_protection)
                    else None
                ),
            )
            mesh_disp = rbf_propagate(
                self.control_nodes,
                ctrl_disp,
                self.base_points,
                kernel=self.kernel_combo.currentText(),
                smoothing=float(self.smooth_spin.value()),
                neighbors=30,
            )
            self.current_disp = mesh_disp
            self.current_ctrl_disp = ctrl_disp
            visual_scale = float(self.scale_spin.value())
            self.mesh.points = self.base_points + visual_scale * mesh_disp

            # Move the red control-node markers with the same visual scale.
            if self.control_nodes_base_poly is not None:
                self.control_nodes_base_poly.points = self.control_nodes.copy()
                try:
                    self.control_nodes_base_poly.Modified()
                except Exception:
                    pass
            if self.control_nodes_deformed_poly is not None:
                self.control_nodes_deformed_poly.points = self.control_nodes + visual_scale * ctrl_disp
                try:
                    self.control_nodes_deformed_poly.Modified()
                except Exception:
                    pass

            try:
                self.mesh.Modified()
            except Exception:
                pass

            # Re-add the mesh actor so the scalar range updates correctly.
            self.refresh_actor(reset_camera=False)
            self.refresh_control_node_actors()
            self.update_control_node_table()

            cn_rms = float(np.sqrt(np.mean(np.sum(ctrl_disp**2, axis=1))))
            mesh_rms = float(np.sqrt(np.mean(np.sum(mesh_disp**2, axis=1))))
            mesh_max = float(np.max(np.linalg.norm(mesh_disp, axis=1)))
            self.status.setText(
                f"Control RMS={cn_rms:.6e} | Mesh RMS={mesh_rms:.6e} | Mesh max={mesh_max:.6e} | visual scale={float(self.scale_spin.value()):.3g}"
            )
        except Exception as exc:
            self.status.setText(f"Update failed: {exc}")

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    def zero_all_coefficients(self):
        for s in self.coeff_sliders:
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)
        for b in self.coeff_values:
            b.blockSignals(True)
            b.setValue(0.0)
            b.blockSignals(False)
        self.schedule_update()

    def solo_mode(self, idx: int, value: float = 1.0):
        target = int(round(float(value) * self.slider_scale))
        for i, s in enumerate(self.coeff_sliders):
            v = target if i == idx else 0
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
            if i < len(self.coeff_values):
                self.coeff_values[i].blockSignals(True)
                self.coeff_values[i].setValue(v / self.slider_scale)
                self.coeff_values[i].blockSignals(False)
        self.schedule_update()

    def random_coefficients(self):
        rng = np.random.default_rng()
        for i, s in enumerate(self.coeff_sliders):
            # Decay amplitudes with mode index so the initial random shape remains smooth.
            val = rng.normal(0.0, 0.35 / ((i + 1) ** 1.5))
            val = float(np.clip(val, self.slider_min / self.slider_scale, self.slider_max / self.slider_scale))
            iv = int(round(val * self.slider_scale))
            s.blockSignals(True)
            s.setValue(iv)
            s.blockSignals(False)
            if i < len(self.coeff_values):
                self.coeff_values[i].blockSignals(True)
                self.coeff_values[i].setValue(iv / self.slider_scale)
                self.coeff_values[i].blockSignals(False)
        self.schedule_update()

    def export_current_mesh(self):
        if self.mesh is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current deformed mesh",
            "modal_deformed_mesh.vtk",
            "VTK files (*.vtk);;VTP files (*.vtp);;PLY files (*.ply);;All files (*)",
        )
        if path:
            try:
                self.mesh.save(path)
                self.status.setText(f"Saved deformed mesh: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save screenshot",
            "modal_explorer.png",
            "PNG files (*.png);;All files (*)",
        )
        if path:
            try:
                self.plotter.screenshot(path)
                self.status.setText(f"Saved screenshot: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Screenshot failed", str(exc))


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


'''def parse_args(argv=None) -> ModalState:
    parser = argparse.ArgumentParser(description="Standalone modal coefficient explorer GUI")
    parser.add_argument("--mesh", default=None, help="Path to surface mesh file")
    parser.add_argument("--control-nodes", default=None, help="Path to control_nodes.npy/.txt/.csv")
    parser.add_argument("--control-normals", default=None, help="Optional path to control_normals.npy/.txt/.csv")
    parser.add_argument("--k-modes", type=int, default=6, help="Number of modes to build")
    parser.add_argument("--knn", type=int, default=6, help="kNN used to build control-node graph")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Physical amplitude multiplier")
    parser.add_argument("--deform-scale", type=float, default=1.0, help="Visual deformation scale")
    
    
    
    return ModalState(
        mesh_path=parser.parse_args(argv).mesh,
        control_nodes_path=parser.parse_args(argv).control_nodes,
        control_normals_path=parser.parse_args(argv).control_normals,
        k_modes=parser.parse_args(argv).k_modes,
        knn=parser.parse_args(argv).knn,
        amplitude=parser.parse_args(argv).amplitude,
        deform_scale=parser.parse_args(argv).deform_scale,
    )'''
    
# ----------------------------------------------------------------------
# USER INPUTS
# ----------------------------------------------------------------------

def parse_args() -> ModalState:
    return ModalState(
        output_dir=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt_lap",
        k_modes=5,
        knn=3,
        amp_alpha=0.005,
        deform_scale=25.0,
        rbf_kernel="thin_plate_spline",
        rbf_smoothing=1e-6,
    )


def main(argv=None):
    state = parse_args()
    app = QApplication(sys.argv)
    win = ModalSliderExplorer(initial=state)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
