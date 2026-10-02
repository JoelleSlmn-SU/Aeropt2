"""
modal_explorer_gui.py

PyQt5 + PyVista explorer for design-vector / modal-coefficient deformations
that runs the SAME code path as the production morph.

Pipeline replicated (shared modules, not copies)
------------------------------------------------
  design vector
    -> ShapeParameterization.designDisplacement.control_displacements_from_basis
         (same pad/truncate + getDisplacements kwargs as pipeline_cluster)
    -> control-node displacement d_ctrl (N_cn, 3)
    -> MeshGeneration.morphPropagation
         classify_regions : D = T ∪ U, C fixed, D<->C anchors (from the .fro)
         rbf_parameters   : adaptive Wendland C4 support (beta=2.2, etc.)
         deform_region    : MorphModel.transformT with the production flags
    -> surface-node displacement on D (T and U both deform; C is fixed)

"Verify vs MorphMesh" and "Export morphed .fro" call MeshGeneration.Morph.MorphMesh
itself, so you can prove the live preview equals the production output.

What is NOT production-equivalent (display only)
------------------------------------------------
  * "Visual scale" multiplies the displayed displacement. The whole chain is
    linear in the coefficients, so this is the same as scaling the design
    vector, but the numbers in the status bar are always at scale 1.
  * Slider positions are quantised to 0.01; the spin boxes and loaded designs
    are exact (the coefficient array, not the slider, is what gets morphed).

Inputs
------
  * morph_basis.json (preferred - exactly what the cluster reads), or the
    equivalent settings passed in ModalState from the mesh GUI.
  * The baseline surface mesh (.fro / .vtm / .vtk / .case), converted exactly
    as Remote/remoteMorph.py does.
  * output_dir containing "Control Nodes/modal_basis_T_surface.npz" (the cache
    that is uploaded to the cluster). If it is missing or stale (different CNs,
    k, or output.vtk) it is rebuilt with exactly the call
    MeshViewer.save_controlnodes() makes, so opening the explorer before Save
    previews the same basis Save will write.

Launch paths
------------
  * From the mesh GUI: MeshViewer.open_modal_explorer passes
    ModalState(basis=build_morph_basis(live_overlay(viewer)), baseline_mesh_path,
    output_dir) - the same dict Aeropt uploads as morph_basis.json.
  * Standalone: load <output_dir>/Control Nodes/morph_basis.json (written by
    Save Control Nodes / Basis); output_dir and the baseline mesh are taken
    from it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass, field
from typing import List, Optional

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
    QLineEdit,
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

# ---------------------------------------------------------------------------
# Make the project root importable (same approach as Remote/remoteMorph.py)
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while not os.path.isdir(os.path.join(_root, "FileRW")):
    _parent = os.path.dirname(_root)
    if _parent == _root:
        _root = None
        break
    _root = _parent
if _root and _root not in sys.path:
    sys.path.insert(0, _root)

from FileRW.FroFile import FroFile  # noqa: E402
from MeshGeneration.MorphModel import MorphModel  # noqa: E402
from MeshGeneration.morphPropagation import (  # noqa: E402
    RBF_BETA,
    RBF_K_NN,
    RBF_MAX_CLIP_FRAC,
    RBF_MIN_CLIP_FRAC,
    TRANSFORM_ANCHOR_TAPER,
    TRANSFORM_BOUNDARY_RECOVER,
    classify_regions,
    deform_region,
    rbf_parameters,
    seam_report,
)
from ShapeParameterization.designDisplacement import (  # noqa: E402
    MODAL_CACHE_REL,
    basis_settings,
    canonical_design_length,
    control_displacements_from_basis,
    describe_design_vector,
    load_modal_cache,
)

# -----------------------------------------------------------------------------
# Helpers
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


def load_baseline_fro(path: str) -> FroFile:
    """Load the baseline surface mesh exactly as Remote/remoteMorph.py does."""
    ext = os.path.splitext(path)[1].lower()
    base = os.path.splitext(os.path.basename(path))[0]
    scratch = os.path.join(tempfile.gettempdir(), "aeropt_modal_explorer")
    os.makedirs(scratch, exist_ok=True)
    fro_out = os.path.join(scratch, f"{base}.fro")

    if ext in (".vtk", ".vtm"):
        from ConvertFileType.convertVtmtoFro import vtm_to_fro
        m = vtm_to_fro(path, fro_out)
    elif ext == ".case":
        # engeo_to_fro returns a writer object, not a FroFile -> re-read the .fro
        from ConvertFileType.convertEnGeoToFro import engeo_to_fro
        engeo_to_fro(path, fro_out)
        m = FroFile(fro_out)
    elif ext == ".fro":
        m = FroFile(path)
    else:
        raise ValueError(f"Unsupported baseline mesh type: {ext}")

    if m.node_count == 0 or (len(m.boundary_triangles) + len(m.boundary_quads)) == 0:
        raise RuntimeError(f"Baseline mesh is empty after loading: {path}")
    return m


def subset_polydata(nodes: np.ndarray, tris: np.ndarray, quads: np.ndarray,
                    tri_mask: np.ndarray, quad_mask: np.ndarray):
    """PolyData made of the selected faces. Returns (poly, global_node_ids)."""
    t = tris[tri_mask, :3] if tris.size else np.zeros((0, 3), int)
    q = quads[quad_mask, :4] if quads.size else np.zeros((0, 4), int)
    used = np.unique(np.concatenate([t.ravel(), q.ravel()]).astype(np.int64))
    if used.size == 0:
        return None, used
    remap = -np.ones(len(nodes), dtype=np.int64)
    remap[used] = np.arange(used.size)
    cells = []
    if t.size:
        cells.append(np.hstack([np.full((len(t), 1), 3), remap[t]]).ravel())
    if q.size:
        cells.append(np.hstack([np.full((len(q), 1), 4), remap[q]]).ravel())
    poly = pv.PolyData(np.asarray(nodes[used], float).copy(), np.concatenate(cells))
    return poly, used


def _parse_int_list(text: str) -> List[int]:
    text = (text or "").replace(";", ",").replace(" ", ",")
    return [int(tok) for tok in text.split(",") if tok.strip()]


@dataclass
class ModalState:
    # --- production inputs (new) ---
    morph_basis_path: Optional[str] = None      # morph_basis.json (preferred)
    basis: Optional[dict] = None                # same schema as morph_basis.json
    baseline_mesh_path: Optional[str] = None    # .fro / .vtm / .vtk / .case
    t_surfaces: Optional[list] = None
    u_surfaces: Optional[list] = None
    c_surfaces: Optional[list] = None
    rigid_translation: bool = False

    # --- settings used to build a basis dict when no json/dict is given ---
    control_nodes_path: Optional[str] = None
    control_normals_path: Optional[str] = None
    control_nodes: Optional[np.ndarray] = None
    control_normals: Optional[np.ndarray] = None
    output_dir: Optional[str] = None

    parameterisation_method: str = "modal"
    direct_parameterisation_subtype: Optional[str] = None
    k_modes: int = 6
    seed: int = 0
    amp_alpha: float = 0.005
    t_patch_scale: Optional[float] = None
    normal_project: bool = True
    vector_mode: str = "local_frame"
    frame_knn: int = 12
    use_local_modes: bool = True
    global_modes: bool = False
    global_only: bool = False
    global_mode_config: Optional[list] = None
    basis_axes: Optional[list] = None
    use_pca: bool = False
    pca_cache_path: Optional[str] = None
    pca_k_final: Optional[int] = None
    graph_method: str = "mutual_knn"
    delaunay_cutoff_factor: float = 2.5
    use_protection: bool = False
    protected_control_nodes: Optional[list] = None
    protection_radius: Optional[float] = None

    deform_scale: float = 1.0

    # --- accepted for backwards compatibility with older callers; unused ---
    mesh_path: Optional[str] = None
    u_mesh_path: Optional[str] = None
    knn: int = 6
    rbf_kernel: str = "thin_plate_spline"
    rbf_smoothing: float = 1e-8


def basis_from_state(state: ModalState) -> dict:
    """Build a morph_basis.json-shaped dict from ModalState (mesh-GUI launch path)."""
    cn = state.control_nodes
    if cn is None and state.control_nodes_path:
        cn = load_array(state.control_nodes_path, 3)
    if cn is None and state.output_dir:
        p = os.path.join(state.output_dir, "Control Nodes", "control_nodes.npy")
        if os.path.exists(p):
            cn = np.load(p)
    if cn is None:
        raise ValueError("No control nodes: give morph_basis.json, control_nodes or control_nodes_path.")
    nn = state.control_normals
    if nn is None and state.control_normals_path:
        nn = load_array(state.control_normals_path, 3)
    if nn is None and state.output_dir:
        p = os.path.join(state.output_dir, "Control Nodes", "control_normals.npy")
        if os.path.exists(p):
            nn = np.load(p)

    return {
        "control_nodes": np.asarray(cn, float).reshape((-1, 3)).tolist(),
        "control_normals": None if nn is None else np.asarray(nn, float).reshape((-1, 3)).tolist(),
        "parameterisation_method": state.parameterisation_method,
        "direct_parameterisation_subtype": state.direct_parameterisation_subtype,
        "t_patch_scale": state.t_patch_scale,
        "amp_alpha": state.amp_alpha,
        "TSurfaces": list(state.t_surfaces or []),
        "USurfaces": list(state.u_surfaces or []),
        "CSurfaces": list(state.c_surfaces or []),
        "k_modes": state.k_modes,
        "seed": state.seed,
        "normal_project": state.normal_project,
        "vector_mode": state.vector_mode,
        "frame_knn": state.frame_knn,
        "use_local_modes": state.use_local_modes,
        "global_modes": state.global_modes,
        "global_only": state.global_only,
        "global_mode_config": state.global_mode_config or [],
        "basis_axes": state.basis_axes,
        "use_pca": state.use_pca,
        "pca_cache_path": state.pca_cache_path,
        "pca_k_final": state.pca_k_final,
        "use_protection": state.use_protection,
        "protected_control_nodes": list(state.protected_control_nodes or []),
        "protection_radius": state.protection_radius,
        "graph_method": state.graph_method,
        "delaunay_cutoff_factor": state.delaunay_cutoff_factor,
        "rigid_translation": state.rigid_translation,
    }


class _QuietLogger:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(str(msg))
        print(msg)


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


class ModalSliderExplorer(QMainWindow):
    def __init__(self, initial: Optional[ModalState] = None):
        super().__init__()
        self.setWindowTitle("Modal Coefficient Explorer (production morph)")
        self.resize(1500, 900)
        self._closing = False

        self.state = initial or ModalState()

        # case data
        self.basis: Optional[dict] = None
        self.settings: Optional[dict] = None
        self.output_dir: Optional[str] = None
        self.fro: Optional[FroFile] = None
        self._fro_path_loaded: Optional[str] = None
        self.model: Optional[MorphModel] = None
        self.regions = None
        self.rbf_params: Optional[dict] = None
        self.control_nodes: Optional[np.ndarray] = None
        self.control_normals: Optional[np.ndarray] = None
        self.layout_info: List[dict] = []
        self.cache: Optional[dict] = None
        self.phi_cn: Optional[np.ndarray] = None
        self.case_warnings: List[str] = []

        # scene data: role -> dict(poly, ids, base)
        self.parts = {}
        self.d_normals: Optional[np.ndarray] = None
        self.current_node_disp: Optional[np.ndarray] = None   # (node_count, 3), scale 1
        self.current_ctrl_disp: Optional[np.ndarray] = None   # (N_cn, 3), scale 1
        self.current_seam: Optional[dict] = None
        self.control_nodes_base_poly = None
        self.control_nodes_deformed_poly = None
        self.control_node_label_actors = []
        self.mesh_actor = None

        # design vector
        self.coeffs = np.zeros(0)
        self.coeff_sliders: list[QSlider] = []
        self.coeff_values: list[QDoubleSpinBox] = []
        self.slider_scale = 100.0  # slider int / 100 -> 0.01 step
        self._pending_update = False

        self._build_ui()

        if self.state.morph_basis_path:
            self.basis_label.setText(os.path.basename(self.state.morph_basis_path))
        if self.state.baseline_mesh_path:
            self.baseline_label.setText(os.path.basename(self.state.baseline_mesh_path))
        if self.state.output_dir:
            self.outdir_label.setText(self.state.output_dir)

        if self._have_minimum_inputs():
            self.load_case()
        else:
            self.status.setText(
                "Load morph_basis.json (or launch from the mesh GUI), the baseline mesh, "
                "and the output directory holding 'Control Nodes/modal_basis_T_surface.npz'."
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        self.setCentralWidget(central)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(610)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        root.addWidget(left_scroll)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setFixedWidth(590)
        left_scroll.setWidget(left)

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

        # ---------------- inputs ----------------
        file_box = QGroupBox("Inputs (production)")
        file_form = QFormLayout(file_box)

        self.basis_label = QLabel("From mesh GUI (live form)" if self.state.basis is not None else "Not loaded")
        self.baseline_label = QLabel("No baseline mesh")
        self.outdir_label = QLabel("No output dir")
        self.outdir_label.setWordWrap(True)

        basis_btn = QPushButton("Load morph_basis.json")
        basis_btn.clicked.connect(self.on_load_basis)
        base_btn = QPushButton("Load baseline mesh")
        base_btn.clicked.connect(self.on_load_baseline)
        out_btn = QPushButton("Set output dir")
        out_btn.clicked.connect(self.on_set_output_dir)

        file_form.addRow(basis_btn, self.basis_label)
        file_form.addRow(base_btn, self.baseline_label)
        file_form.addRow(out_btn, self.outdir_label)

        self.t_edit = QLineEdit()
        self.u_edit = QLineEdit()
        self.c_edit = QLineEdit()
        for w in (self.t_edit, self.u_edit, self.c_edit):
            w.setPlaceholderText("from morph_basis.json")
        file_form.addRow("T surfaces", self.t_edit)
        file_form.addRow("U surfaces", self.u_edit)
        file_form.addRow("C surfaces", self.c_edit)

        reload_btn = QPushButton("(Re)load case")
        reload_btn.clicked.connect(self.load_case)
        file_form.addRow(reload_btn)
        left_layout.addWidget(file_box)

        # ---------------- production summary ----------------
        prod_box = QGroupBox("Production settings in use (read-only)")
        prod_layout = QVBoxLayout(prod_box)
        self.prod_label = QLabel("-")
        self.prod_label.setWordWrap(True)
        self.prod_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        prod_layout.addWidget(self.prod_label)
        left_layout.addWidget(prod_box)

        # ---------------- display ----------------
        disp_box = QGroupBox("Display")
        disp_form = QFormLayout(disp_box)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(3)
        self.scale_spin.setRange(0.001, 1000.0)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setValue(self.state.deform_scale)
        self.scale_spin.valueChanged.connect(lambda _v: self.apply_display())
        disp_form.addRow("Visual scale (display only)", self.scale_spin)

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems([
            "Displacement magnitude |u|",
            "Normal displacement u·n (mesh normals)",
            "X displacement ux",
            "Y displacement uy",
            "Z displacement uz",
        ])
        self.color_mode_combo.currentTextChanged.connect(lambda _v: self.refresh_actor())
        disp_form.addRow("Colour field", self.color_mode_combo)

        self.cbar_vertical_check = QCheckBox("Vertical colour bar")
        self.cbar_vertical_check.stateChanged.connect(lambda _v: self.refresh_actor())
        disp_form.addRow(self.cbar_vertical_check)

        cbar_pos_row = QHBoxLayout()
        self.cbar_x_spin = QDoubleSpinBox()
        self.cbar_x_spin.setRange(0.0, 0.95)
        self.cbar_x_spin.setSingleStep(0.05)
        self.cbar_x_spin.setDecimals(2)
        self.cbar_x_spin.setValue(0.7)
        self.cbar_x_spin.valueChanged.connect(lambda _v: self.refresh_actor())
        self.cbar_y_spin = QDoubleSpinBox()
        self.cbar_y_spin.setRange(0.0, 0.95)
        self.cbar_y_spin.setSingleStep(0.05)
        self.cbar_y_spin.setDecimals(2)
        self.cbar_y_spin.setValue(0.05)
        self.cbar_y_spin.valueChanged.connect(lambda _v: self.refresh_actor())
        cbar_reset_btn = QPushButton("Reset")
        cbar_reset_btn.clicked.connect(self.reset_colorbar_position)
        cbar_pos_row.addWidget(QLabel("x"))
        cbar_pos_row.addWidget(self.cbar_x_spin)
        cbar_pos_row.addWidget(QLabel("y"))
        cbar_pos_row.addWidget(self.cbar_y_spin)
        cbar_pos_row.addWidget(cbar_reset_btn)
        cbar_pos_widget = QWidget()
        cbar_pos_widget.setLayout(cbar_pos_row)
        disp_form.addRow("Colour bar position", cbar_pos_widget)

        self.robust_clim_check = QCheckBox("Robust colour limits (clip to 1st-99th percentile)")
        self.robust_clim_check.stateChanged.connect(lambda _v: self.refresh_actor())
        disp_form.addRow(self.robust_clim_check)

        self.cbar_range_label = QLabel("Data range: -")
        self.cbar_range_label.setWordWrap(True)
        disp_form.addRow(self.cbar_range_label)

        self.show_t_check = QCheckBox("Show T surfaces (deforming)")
        self.show_t_check.setChecked(True)
        self.show_t_check.stateChanged.connect(lambda _v: self.refresh_actor())
        disp_form.addRow(self.show_t_check)

        self.show_u_check = QCheckBox("Show U surfaces (deforming)")
        self.show_u_check.setChecked(True)
        self.show_u_check.stateChanged.connect(lambda _v: self.refresh_actor())
        disp_form.addRow(self.show_u_check)

        self.show_baseline_check = QCheckBox("Show baseline wireframe (visible T / U only)")
        self.show_baseline_check.setChecked(True)
        self.show_baseline_check.stateChanged.connect(lambda _v: self.refresh_actor())
        disp_form.addRow(self.show_baseline_check)

        self.show_c_check = QCheckBox("Show C surfaces (fixed)")
        self.show_c_check.setChecked(True)
        self.show_c_check.stateChanged.connect(lambda _v: self.refresh_context_actors())
        disp_form.addRow(self.show_c_check)

        self.show_other_check = QCheckBox("Show other surfaces (e.g. farfield)")
        self.show_other_check.setChecked(False)
        self.show_other_check.stateChanged.connect(lambda _v: self.refresh_context_actors())
        disp_form.addRow(self.show_other_check)

        self.show_anchor_check = QCheckBox("Show D↔C seam (anchor) nodes")
        self.show_anchor_check.setChecked(False)
        self.show_anchor_check.stateChanged.connect(lambda _v: self.refresh_context_actors())
        disp_form.addRow(self.show_anchor_check)

        self.show_cn_base_check = QCheckBox("Show original control nodes")
        self.show_cn_base_check.setChecked(True)
        self.show_cn_base_check.stateChanged.connect(lambda _v: self.refresh_control_node_actors())
        disp_form.addRow(self.show_cn_base_check)

        self.show_cn_deformed_check = QCheckBox("Show deformed control nodes")
        self.show_cn_deformed_check.setChecked(True)
        self.show_cn_deformed_check.stateChanged.connect(lambda _v: self.refresh_control_node_actors())
        disp_form.addRow(self.show_cn_deformed_check)

        self.show_cn_labels_check = QCheckBox("Show control-node labels")
        self.show_cn_labels_check.setChecked(False)
        self.show_cn_labels_check.stateChanged.connect(lambda _v: self.update_control_node_labels())
        disp_form.addRow(self.show_cn_labels_check)

        self.edges_check = QCheckBox("Show mesh edges")
        self.edges_check.stateChanged.connect(lambda _v: self.refresh_actor())
        disp_form.addRow(self.edges_check)
        left_layout.addWidget(disp_box)

        # ---------------- actions ----------------
        actions_box = QGroupBox("Design vector / verification")
        actions_layout = QVBoxLayout(actions_box)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Slider range ±"))
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setDecimals(2)
        self.range_spin.setRange(0.1, 100.0)
        self.range_spin.setValue(2.0)
        self.range_spin.valueChanged.connect(lambda _v: self.rebuild_sliders())
        range_row.addWidget(self.range_spin)
        range_w = QWidget()
        range_w.setLayout(range_row)
        actions_layout.addWidget(range_w)

        for text, slot in [
            ("Reset all coefficients to zero", self.zero_all_coefficients),
            ("Random small coefficients", self.random_coefficients),
            ("Load design from morph_config_n_*.json", self.on_load_design),
            ("Save current design vector (.json)", self.on_save_design),
            ("Verify preview against MorphMesh", self.verify_against_morphmesh),
            ("Export morphed .fro (via MorphMesh)", self.export_morphed_fro),
        ]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            actions_layout.addWidget(b)
        left_layout.addWidget(actions_box)

        # ---------------- picture / export ----------------
        export_box = QGroupBox("Picture / export")
        export_form = QFormLayout(export_box)

        self.picture_view_check = QCheckBox("Picture view (hide axes / title / scalar bar)")
        self.picture_view_check.stateChanged.connect(lambda _v: self.toggle_picture_view())
        export_form.addRow(self.picture_view_check)

        self.screenshot_scale_spin = QSpinBox()
        self.screenshot_scale_spin.setRange(1, 8)
        self.screenshot_scale_spin.setValue(2)
        export_form.addRow("Image scale", self.screenshot_scale_spin)

        png_btn_row = QHBoxLayout()
        save_png_btn = QPushButton("Save PNG")
        save_png_btn.clicked.connect(lambda: self.save_screenshot(transparent=False))
        save_png_transparent_btn = QPushButton("Save PNG (transparent bg)")
        save_png_transparent_btn.clicked.connect(lambda: self.save_screenshot(transparent=True))
        png_btn_row.addWidget(save_png_btn)
        png_btn_row.addWidget(save_png_transparent_btn)
        png_btn_widget = QWidget()
        png_btn_widget.setLayout(png_btn_row)
        export_form.addRow(png_btn_widget)
        left_layout.addWidget(export_box)

        # ---------------- sliders ----------------
        slider_group = QGroupBox("Design vector (as passed to the pipeline)")
        slider_group.setMinimumHeight(330)
        slider_outer = QVBoxLayout(slider_group)
        self.slider_scroll = QScrollArea()
        self.slider_scroll.setWidgetResizable(True)
        self.slider_container = QWidget()
        self.slider_layout = QVBoxLayout(self.slider_container)
        self.slider_scroll.setWidget(self.slider_container)
        slider_outer.addWidget(self.slider_scroll)
        left_layout.addWidget(slider_group, stretch=1)

        self.mode_table = QTableWidget(0, 3)
        self.mode_table.setHorizontalHeaderLabels(["Design var", "Drives", "Eigenvalue λ"])
        self.mode_table.setMaximumHeight(160)
        left_layout.addWidget(self.mode_table)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
                for fn in ("disable_picking", "clear"):
                    try:
                        getattr(self.plotter, fn)()
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
            self.parts = {}
            self.fro = None
        except Exception as e:
            print(f"[DEBUG] Modal explorer close cleanup failed: {e}")
        event.accept()

    # ------------------------------------------------------------------
    # Input selection
    # ------------------------------------------------------------------

    def _have_minimum_inputs(self) -> bool:
        has_basis = bool(self.state.morph_basis_path) or self.state.basis is not None or (
            self.state.control_nodes is not None or self.state.control_nodes_path is not None
        )
        return has_basis and bool(self._resolve_baseline()) and bool(self._resolve_output_dir())

    def _basis_json_in_control_nodes(self) -> Optional[str]:
        """<out>/Control Nodes/morph_basis.json -> <out> (the saved-by-GUI layout)."""
        p = self.state.morph_basis_path
        if p:
            d = os.path.dirname(os.path.abspath(p))
            if os.path.basename(d) == "Control Nodes":
                return os.path.dirname(d)
        return None

    def _resolve_output_dir(self) -> Optional[str]:
        # Only the explicit value, or the layout save_controlnodes writes
        # (<out>/Control Nodes/morph_basis.json). No other guessing, because
        # getDisplacements BUILDS a cache under <output_dir>/Control Nodes/.
        return self.state.output_dir or self._basis_json_in_control_nodes()

    def _resolve_baseline(self) -> Optional[str]:
        if self.state.baseline_mesh_path:
            return self.state.baseline_mesh_path
        b = self.state.basis
        if b is None and self.state.morph_basis_path and os.path.exists(self.state.morph_basis_path):
            try:
                with open(self.state.morph_basis_path, "r", encoding="utf-8") as f:
                    b = json.load(f)
            except Exception:
                b = None
        p = (b or {}).get("baseline_mesh_path")
        return p if p and os.path.exists(p) else None

    def on_load_basis(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load morph_basis.json", "", "JSON (*.json);;All files (*)")
        if path:
            self.state.morph_basis_path = path
            self.basis_label.setText(os.path.basename(path))
            for w in (self.t_edit, self.u_edit, self.c_edit):
                w.clear()
            if self._have_minimum_inputs():
                self.load_case()

    def on_load_baseline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load baseline surface mesh", "",
            "Mesh (*.fro *.vtm *.vtk *.case);;All files (*)",
        )
        if path:
            self.state.baseline_mesh_path = path
            self.baseline_label.setText(os.path.basename(path))
            if self._have_minimum_inputs():
                self.load_case()

    def on_set_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Output dir (contains 'Control Nodes/')")
        if path:
            self.state.output_dir = path
            self.outdir_label.setText(path)
            if self._have_minimum_inputs():
                self.load_case()

    # ------------------------------------------------------------------
    # Case loading
    # ------------------------------------------------------------------

    def _resolve_basis(self) -> dict:
        if self.state.morph_basis_path:
            with open(self.state.morph_basis_path, "r", encoding="utf-8") as f:
                return json.load(f)
        if self.state.basis is not None:
            return dict(self.state.basis)
        return basis_from_state(self.state)

    def _set_busy(self, msg: str):
        self.status.setText(msg)
        try:
            QApplication.processEvents()
        except Exception:
            pass

    def load_case(self):
        try:
            self.case_warnings = []
            self._set_busy("Loading case...")

            # 1) basis (settings + CNs)
            self.basis = self._resolve_basis()
            self.settings = basis_settings(self.basis)
            self.output_dir = self._resolve_output_dir()
            if not self.output_dir:
                raise ValueError("Output dir not set (needed for the modal cache).")
            self.outdir_label.setText(self.output_dir)

            self.control_nodes = np.asarray(self.basis["control_nodes"], float).reshape((-1, 3))
            nn = self.basis.get("control_normals", None)
            self.control_normals = None if nn is None else np.asarray(nn, float).reshape((-1, 3))

            # 2) surface roles: basis first, UI fields override (with a warning)
            def pick(edit: QLineEdit, key: str, fallback) -> List[int]:
                from_basis = list(map(int, self.basis.get(key, []) or (fallback or [])))
                if edit.text().strip():
                    ui = _parse_int_list(edit.text())
                    if ui != from_basis:
                        self.case_warnings.append(f"{key} overridden in the UI: {ui} (basis: {from_basis})")
                    return ui
                edit.setText(", ".join(map(str, from_basis)))
                return from_basis

            t_s = pick(self.t_edit, "TSurfaces", self.state.t_surfaces)
            u_s = pick(self.u_edit, "USurfaces", self.state.u_surfaces)
            c_s = pick(self.c_edit, "CSurfaces", self.state.c_surfaces)
            if not t_s:
                raise ValueError("No T surfaces defined.")

            # 3) baseline mesh (converted like remoteMorph.py; cached by path)
            if not self.state.baseline_mesh_path:
                self.state.baseline_mesh_path = self._resolve_baseline()
            if not self.state.baseline_mesh_path:
                raise ValueError("No baseline mesh selected.")
            if self.fro is None or self._fro_path_loaded != self.state.baseline_mesh_path:
                self._set_busy("Converting / reading baseline mesh...")
                self.fro = load_baseline_fro(self.state.baseline_mesh_path)
                self._fro_path_loaded = self.state.baseline_mesh_path
            self.baseline_label.setText(
                f"{os.path.basename(self.state.baseline_mesh_path)} ({self.fro.node_count} nodes)"
            )
            missing = sorted(set(t_s + u_s + c_s) - set(self.fro.get_surface_ids()))
            if missing:
                self.case_warnings.append(f"Surface ids not in baseline mesh: {missing}")

            # 4) morph model exactly as remoteMorph.run_surf_morph_from_config builds it
            self.model = MorphModel(
                f_name=None, con=False, pb=[], path=None, T=[], U=[],
                cn=self.control_nodes.tolist(), displacement_vector=[],
            )
            self.model.t_surfaces = list(map(int, t_s))
            self.model.u_surfaces = list(map(int, u_s))
            self.model.c_surfaces = list(map(int, c_s))
            self.model.rigid_boundary_translation = bool(self.basis.get("rigid_translation", False))
            if self.model.rigid_boundary_translation:
                self.case_warnings.append(
                    "rigid_translation=True: MorphMesh applies zero rigid translations "
                    "(bt=0), so this is a no-op in production as well."
                )

            # 5) regions + RBF parameters (independent of the design vector)
            self._set_busy("Classifying T/U/C regions...")
            self.regions = classify_regions(self.fro, self.model)
            if len(self.regions.anchor_gids) == 0:
                self.case_warnings.append("No D–C anchor nodes found (R0 falls back to fallback_R_frac).")
            self.rbf_params = rbf_parameters(
                self.model.control_nodes, self.regions.d_verts, self.regions.anchor_points
            )

            # CNs vs D nodes (transformT snaps CNs to the nearest D node)
            from scipy.spatial import cKDTree
            dsnap, _ = cKDTree(self.regions.d_verts).query(self.control_nodes, k=1)
            L = float(np.linalg.norm(self.regions.d_verts.max(0) - self.regions.d_verts.min(0)))
            self.snap_max = float(dsnap.max()) if dsnap.size else 0.0
            if self.snap_max > 1e-6 * max(L, 1e-12):
                self.case_warnings.append(
                    f"Control nodes are not exact D nodes (max snap {self.snap_max:.3e}); "
                    "transformT snaps them, same as production."
                )

            # 6) modal cache: make sure it exists exactly as production would see it
            self._check_modal_cache()

            # 7) design-vector layout
            n_design = canonical_design_length(self.basis)
            self._set_layout(n_design)

            # 8) scene
            self._build_scene_parts()
            self.draw_initial_scene()
            self.update_production_summary()
            self.schedule_update()

            msg = (
                f"Loaded: |D|={len(self.regions.d_gids)} (T={len(self.regions.t_gids)}, "
                f"U={len(self.regions.u_gids)}), |C|={len(self.regions.c_gids)}, "
                f"anchors={len(self.regions.anchor_gids)}, CNs={len(self.control_nodes)}, "
                f"design length={n_design}."
            )
            if self.case_warnings:
                msg += "\nWARNINGS:\n - " + "\n - ".join(self.case_warnings)
            self.status.setText(msg)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load failed", str(exc))
            self.status.setText(f"Load failed: {exc}")

    def _cache_problems(self, cache) -> List[str]:
        """Reasons the on-disk modal cache does not match the current basis."""
        s = self.settings
        if cache is None:
            return ["missing"]
        out = []
        k_cache = int(cache["phi_T"].shape[1])
        if k_cache != int(s["k"]):
            out.append(f"k={k_cache} in cache vs k_modes={s['k']}")
        idx = np.asarray(cache["control_node_point_indices"], int)
        pts = np.asarray(cache["points_T"], float)
        if len(idx) != len(self.control_nodes):
            out.append(f"{len(idx)} CNs in cache vs {len(self.control_nodes)}")
        elif float(np.max(np.linalg.norm(pts[idx] - self.control_nodes, axis=1))) > 1e-8:
            out.append("CN positions changed")
        vtk_path = os.path.join(self.output_dir, "surfaces", "output.vtk")
        if os.path.exists(vtk_path):
            try:
                if pv.read(vtk_path).n_points != len(pts):
                    out.append("surfaces/output.vtk changed")
            except Exception:
                pass
        return out

    def _check_modal_cache(self):
        s = self.settings
        self.cache = None
        self.phi_cn = None
        if s["parameterisation_method"] == "direct" or s["use_pca"] or not s["use_local_modes"]:
            return

        cache_path = os.path.join(self.output_dir, MODAL_CACHE_REL)
        cache = load_modal_cache(self.output_dir)
        why = self._cache_problems(cache)
        if why:
            # Rebuild with exactly the arguments MeshViewer.save_controlnodes()
            # uses, into the canonical path. Save will rewrite the same basis,
            # and it is this file that gets uploaded to the cluster.
            from ShapeParameterization.controlNodeDisp import build_t_surface_modal_cache
            self._set_busy(f"Rebuilding modal cache ({'; '.join(why)})...")
            build_t_surface_modal_cache(
                output_dir=self.output_dir,
                control_nodes=np.asarray(self.control_nodes, float),
                k_modes=int(s["k"]),
                frame_knn=int(s["frame_knn"] or 12),
                graph_method=s["graph_method"],
                delaunay_cutoff_factor=float(s["delaunay_cutoff_factor"]),
            )
            self.case_warnings.append(
                f"Modal cache rebuilt ({'; '.join(why)}) with k={s['k']}, graph kNN={s['frame_knn']}, "
                f"graph={s['graph_method']} - same call as 'Save Control Nodes / Basis'."
            )
            cache = load_modal_cache(self.output_dir)
            still = self._cache_problems(cache)
            if still:
                self.case_warnings.append(f"Cache still differs after rebuild: {'; '.join(still)}")

        if cache is None:
            raise RuntimeError(f"Modal cache could not be loaded from {cache_path}")
        self.cache = cache
        idx = np.asarray(cache["control_node_point_indices"], int)
        if len(idx) == len(self.control_nodes):
            self.phi_cn = np.asarray(cache["phi_T"], float)[idx]

    def _set_layout(self, n_design: int):
        self.layout_info = describe_design_vector(self.basis, self.output_dir, int(n_design))
        old = self.coeffs
        self.coeffs = np.zeros(int(n_design))
        m = min(len(old), len(self.coeffs))
        self.coeffs[:m] = old[:m]
        self.rebuild_sliders()
        self.update_mode_table()

    def update_production_summary(self):
        s = self.settings
        p = self.rbf_params or {}
        lines = [
            f"<b>Displacement</b>: {s['parameterisation_method']}"
            + (f" / {s['direct_subtype']}" if s['parameterisation_method'] == 'direct' else "")
            + (" / PCA" if s['use_pca'] else ""),
            f"local modes={s['use_local_modes']} (k basis={s['k']}"
            + (f", k cache={self.cache['phi_T'].shape[1]}" if self.cache is not None else "") + ")"
            + f", normal_project={s['normal_project']}, vector_mode={s['vector_mode']}",
            f"global modes={s['global_modes']} config={s['global_mode_config']}",
            f"amp_alpha={s['amp_alpha']}, t_patch_scale={s['t_patch_scale']}"
            + ("  (None → CN 3rd-neighbour spacing d_ref)" if s['t_patch_scale'] is None else ""),
            f"protection={s['use_protection']} nodes={len(s['protected_nodes'])} radius={s['protection_radius']}",
            f"graph={s['graph_method']} (only used if the cache is rebuilt)",
            "",
            f"<b>Propagation</b>: MorphModel.transformT, Wendland C4, λ={getattr(self.model, 'rbf_lambda', 1e-10)}",
            f"k_nn={RBF_K_NN}, β={RBF_BETA}, clip=[{RBF_MIN_CLIP_FRAC}, {RBF_MAX_CLIP_FRAC}]·L",
            f"min_R_frac={p.get('min_R_frac', float('nan')):.4g}, fallback_R_frac={p.get('fallback_R_frac', float('nan')):.4g}, "
            f"R_scale={p.get('R_scale', float('nan')):.3g}",
            f"anchor_taper={TRANSFORM_ANCHOR_TAPER}, boundary_recover={TRANSFORM_BOUNDARY_RECOVER}",
            f"D = T ∪ U deforms; C fixed; T={self.model.t_surfaces} U={self.model.u_surfaces} C={self.model.c_surfaces}",
        ]
        self.prod_label.setText("<br>".join(lines))

    # ------------------------------------------------------------------
    # Sliders / tables
    # ------------------------------------------------------------------

    def _slider_bounds(self):
        r = float(self.range_spin.value())
        return int(round(-r * self.slider_scale)), int(round(r * self.slider_scale))

    def rebuild_sliders(self):
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.coeff_sliders = []
        self.coeff_values = []
        smin, smax = self._slider_bounds()

        for i in range(len(self.coeffs)):
            info = self.layout_info[i] if i < len(self.layout_info) else {"label": f"x{i + 1}", "kind": "?"}
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(info["label"])
            label.setFixedWidth(150)
            label.setToolTip(info["label"])
            if info["kind"] == "unused":
                label.setStyleSheet("color: #999;")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(smin, smax)
            slider.setSingleStep(1)
            slider.setPageStep(10)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(50)

            value_box = QDoubleSpinBox()
            value_box.setDecimals(6)
            value_box.setRange(-1e6, 1e6)
            value_box.setSingleStep(0.01)
            value_box.setFixedWidth(95)

            slider.blockSignals(True)
            slider.setValue(int(np.clip(round(self.coeffs[i] * self.slider_scale), smin, smax)))
            slider.blockSignals(False)
            value_box.blockSignals(True)
            value_box.setValue(float(self.coeffs[i]))
            value_box.blockSignals(False)

            def _slider_changed(v, idx=i, box=value_box):
                self.coeffs[idx] = float(v) / self.slider_scale
                box.blockSignals(True)
                box.setValue(self.coeffs[idx])
                box.blockSignals(False)
                self.schedule_update()

            def _box_changed(v, idx=i, sl=slider):
                self.coeffs[idx] = float(v)  # exact value is what gets morphed
                lo, hi = self._slider_bounds()
                sl.blockSignals(True)
                sl.setValue(int(np.clip(round(float(v) * self.slider_scale), lo, hi)))
                sl.blockSignals(False)
                self.schedule_update()

            slider.valueChanged.connect(_slider_changed)
            value_box.valueChanged.connect(_box_changed)

            solo_pos_btn = QPushButton("Solo +")
            solo_pos_btn.setFixedWidth(55)
            solo_pos_btn.clicked.connect(lambda _c=False, idx=i: self.solo_mode(idx, 1.0))
            solo_neg_btn = QPushButton("Solo -")
            solo_neg_btn.setFixedWidth(55)
            solo_neg_btn.clicked.connect(lambda _c=False, idx=i: self.solo_mode(idx, -1.0))
            zero_btn = QPushButton("0")
            zero_btn.setFixedWidth(30)
            zero_btn.clicked.connect(lambda _c=False, box=value_box: box.setValue(0.0))

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

    def _sync_widgets_from_coeffs(self):
        smin, smax = self._slider_bounds()
        for i, (s, b) in enumerate(zip(self.coeff_sliders, self.coeff_values)):
            s.blockSignals(True)
            s.setValue(int(np.clip(round(self.coeffs[i] * self.slider_scale), smin, smax)))
            s.blockSignals(False)
            b.blockSignals(True)
            b.setValue(float(self.coeffs[i]))
            b.blockSignals(False)

    def update_mode_table(self):
        n = len(self.layout_info)
        self.mode_table.setRowCount(n)
        evals = None
        if self.cache is not None and "evals_T" in self.cache:
            evals = np.asarray(self.cache["evals_T"], float).reshape(-1)
        for i, info in enumerate(self.layout_info):
            ev = ""
            if info["kind"] == "local" and evals is not None and info["mode"] < evals.size:
                ev = f"{evals[info['mode']]:.4e}"
            drives = {
                "local": f"local Laplacian mode {info['mode'] + 1 if info['mode'] is not None else ''}"
                         + (f" ({info['component']})" if info.get("component") not in (None, "n") else ""),
                "global": "global mode",
                "direct": "direct CN displacement",
                "pca": "PCA component",
                "unused": "nothing (pad/truncate)",
            }.get(info["kind"], "?")
            for col, text in enumerate([info["label"], drives, ev]):
                self.mode_table.setItem(i, col, QTableWidgetItem(text))
        self.mode_table.resizeColumnsToContents()

    def update_control_node_table(self):
        if self.control_nodes is None:
            self.cn_table.setRowCount(0)
            self.cn_table.setColumnCount(0)
            return
        d = self.current_ctrl_disp if self.current_ctrl_disp is not None else np.zeros_like(self.control_nodes)
        dmag = np.linalg.norm(d, axis=1)
        nrm = self.control_normals
        show_phi = (
            self.phi_cn is not None
            and self.settings["normal_project"]
            and self.settings["use_local_modes"]
            and self.settings["parameterisation_method"] == "modal"
            and not self.settings["use_pca"]
        )
        k = self.phi_cn.shape[1] if show_phi else 0

        headers = ["CN", "x", "y", "z", "dx", "dy", "dz", "|d|"]
        if nrm is not None:
            headers += ["d·n"]
        headers += [f"phi_{j + 1}" for j in range(k)]

        n = self.control_nodes.shape[0]
        self.cn_table.blockSignals(True)
        self.cn_table.setColumnCount(len(headers))
        self.cn_table.setRowCount(n)
        self.cn_table.setHorizontalHeaderLabels(headers)
        for i in range(n):
            vals = [i + 1] + list(self.control_nodes[i]) + list(d[i]) + [dmag[i]]
            if nrm is not None:
                vals += [float(np.dot(d[i], nrm[i]))]
            if k:
                vals += list(self.phi_cn[i, :])
            for j, val in enumerate(vals):
                item = QTableWidgetItem(str(val) if j == 0 else f"{float(val):.5e}")
                item.setTextAlignment(Qt.AlignCenter if j == 0 else (Qt.AlignRight | Qt.AlignVCenter))
                self.cn_table.setItem(i, j, item)
        self.cn_table.resizeColumnsToContents()
        self.cn_table.blockSignals(False)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _build_scene_parts(self):
        fro = self.fro
        nodes = np.asarray(fro.nodes, float)
        tris = np.asarray(fro.boundary_triangles, int).reshape((-1, 4))
        quads = np.asarray(fro.boundary_quads, int).reshape((-1, 5))
        t_sid = tris[:, 3] if tris.size else np.zeros(0, int)
        q_sid = quads[:, 4] if quads.size else np.zeros(0, int)

        d_s = set(self.model.t_surfaces) | set(self.model.u_surfaces)
        if self.model.c_surfaces:
            c_s = set(self.model.c_surfaces) - d_s
        else:
            # MorphModel.get_c_node_gids fallback: complement of T ∪ U minus farfield
            c_s = set(fro.get_surface_ids()) - d_s - set(getattr(fro, "farfield_ids", []) or [])

        # T and U are displayed as separate parts so each can be toggled.
        # A surface id listed in both T and U is shown under T only (no double draw).
        t_only = set(self.model.t_surfaces)
        u_only = set(self.model.u_surfaces) - t_only

        self.parts = {}
        for role, sids in (("T", t_only), ("U", u_only), ("C", c_s)):
            if not sids:
                continue
            poly, ids = subset_polydata(nodes, tris, quads,
                                        np.isin(t_sid, list(sids)), np.isin(q_sid, list(sids)))
            if poly is not None:
                self.parts[role] = {"poly": poly, "ids": ids, "base": np.asarray(poly.points).copy()}
        other = set(fro.get_surface_ids()) - d_s - c_s
        if other:
            poly, ids = subset_polydata(nodes, tris, quads,
                                        np.isin(t_sid, list(other)), np.isin(q_sid, list(other)))
            if poly is not None:
                self.parts["O"] = {"poly": poly, "ids": ids, "base": np.asarray(poly.points).copy()}

        # Point normals from the .fro face orientation (for u·n colouring), computed on
        # the whole of D = T ∪ U so seam nodes get the same averaged normal as before,
        # then scattered into a global (node_count, 3) array indexed by node gid.
        self.d_normals = None
        if self.parts.get("T") is not None or self.parts.get("U") is not None:
            try:
                d_poly, d_ids = subset_polydata(nodes, tris, quads,
                                                np.isin(t_sid, list(d_s)), np.isin(q_sid, list(d_s)))
                if d_poly is not None:
                    nm = d_poly.compute_normals(
                        point_normals=True, cell_normals=False, split_vertices=False,
                        consistent_normals=False, auto_orient_normals=False,
                    )
                    g = np.full((len(nodes), 3), np.nan)
                    g[np.asarray(d_ids, int)] = np.asarray(nm.point_data["Normals"], float)
                    self.d_normals = g
            except Exception as exc:
                print(f"[WARN] normal computation failed: {exc}")

        self.current_node_disp = np.zeros((fro.node_count, 3))
        self.anchor_poly = pv.PolyData(np.asarray(self.regions.anchor_points, float).reshape((-1, 3))) \
            if len(self.regions.anchor_points) else None

    def draw_initial_scene(self):
        if not self.parts:
            return
        self.plotter.clear()
        self.control_node_label_actors = []
        picture_view = self.picture_view_check.isChecked()
        try:
            self.plotter.add_axes(line_width=2, labels_off=False)
            if picture_view:
                self.plotter.hide_axes()
        except Exception:
            pass

        self.control_nodes_base_poly = pv.PolyData(self.control_nodes.copy())
        self.control_nodes_deformed_poly = pv.PolyData(self.control_nodes.copy())

        self.refresh_actor(reset_camera=True)
        self.refresh_context_actors()
        self.refresh_control_node_actors()
        self.update_control_node_table()

        if not picture_view:
            self.plotter.add_text("Modal coefficient explorer (production morph)",
                                  position="upper_left", font_size=11, color="black", name="title_text")
        self.plotter.reset_camera()
        self.plotter.render()

    def apply_display(self):
        """Push current_node_disp * visual scale onto every displayed part."""
        if not self.parts or self.current_node_disp is None:
            return
        sc = float(self.scale_spin.value())
        for part in self.parts.values():
            part["poly"].points = part["base"] + sc * self.current_node_disp[part["ids"]]
        if self.control_nodes_deformed_poly is not None and self.current_ctrl_disp is not None:
            self.control_nodes_deformed_poly.points = self.control_nodes + sc * self.current_ctrl_disp
        if self.anchor_poly is not None:
            self.anchor_poly.points = (
                np.asarray(self.regions.anchor_points, float)
                + sc * self.current_node_disp[np.asarray(self.regions.anchor_gids, int)]
            )
        self.refresh_actor()
        self.refresh_context_actors()
        self.refresh_control_node_actors()

    def _visible_deforming_roles(self):
        """Deforming parts (T, U) that exist and are ticked in the Display box."""
        roles = []
        if "T" in self.parts and self.show_t_check.isChecked():
            roles.append("T")
        if "U" in self.parts and self.show_u_check.isChecked():
            roles.append("U")
        return roles

    def _mesh_scalar_field(self, role: str):
        part = self.parts[role]
        ids = np.asarray(part["ids"], int)
        u = self.current_node_disp[ids]
        mode = self.color_mode_combo.currentText()
        if mode.startswith("Normal displacement") and self.d_normals is not None:
            return "u_dot_n", np.einsum("ij,ij->i", u, self.d_normals[ids]), "u·n"
        if mode.startswith("X displacement"):
            return "ux", u[:, 0], "ux"
        if mode.startswith("Y displacement"):
            return "uy", u[:, 1], "uy"
        if mode.startswith("Z displacement"):
            return "uz", u[:, 2], "uz"
        return "disp_mag", np.linalg.norm(u, axis=1), "|u|"

    def refresh_actor(self, reset_camera: bool = False):
        if self.plotter is None or not ("T" in self.parts or "U" in self.parts):
            return
        for name in ("baseline_wire_T", "baseline_wire_U", "deformed_mesh_T", "deformed_mesh_U",
                     "baseline_wire", "deformed_mesh"):
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        self.mesh_actor = None

        roles = self._visible_deforming_roles()
        if not roles:
            self.cbar_range_label.setText("Data range: - (no T / U surfaces shown)")
            for bar_title in list(getattr(self.plotter, "scalar_bars", {}).keys()):
                try:
                    self.plotter.remove_scalar_bar(bar_title)
                except Exception:
                    pass
            if reset_camera:
                self.plotter.reset_camera()
            self.plotter.render()
            return

        # Scalars per visible part; the colour range is shared across them so T and U
        # are directly comparable, and it reflects only what is on screen.
        fields = {}
        title = ""
        for role in roles:
            scalar_name, vals, title = self._mesh_scalar_field(role)
            self.parts[role]["poly"].point_data[scalar_name] = vals
            fields[role] = (scalar_name, vals)
        all_vals = np.concatenate([v for _, v in fields.values()])
        all_vals = all_vals[np.isfinite(all_vals)]

        clim = None
        vmin_full = vmax_full = 0.0
        robust = self.robust_clim_check.isChecked()
        if all_vals.size:
            vmin_full = float(np.min(all_vals))
            vmax_full = float(np.max(all_vals))
            if robust and all_vals.size > 1:
                vmin, vmax = (float(v) for v in np.percentile(all_vals, [1.0, 99.0]))
            else:
                vmin, vmax = vmin_full, vmax_full
            if vmin < 0.0 < vmax:
                m = max(abs(vmin), abs(vmax), 1e-14)
                clim = [-m, m]
            elif abs(vmax - vmin) > 1e-14:
                clim = [vmin, vmax]

        shown = " + ".join(roles)
        if robust and clim is not None and (abs(clim[0] - vmin_full) > 1e-12 or abs(clim[1] - vmax_full) > 1e-12):
            self.cbar_range_label.setText(
                f"Data range ({shown}): [{vmin_full:.3e}, {vmax_full:.3e}]\n"
                f"Bar clipped to 1-99th pct: [{clim[0]:.3e}, {clim[1]:.3e}]"
            )
        else:
            self.cbar_range_label.setText(f"Data range ({shown}): [{vmin_full:.3e}, {vmax_full:.3e}]")

        picture_view = self.picture_view_check.isChecked()
        scalar_bar_args = {
            "title": title,
            "interactive": True,
            "position_x": float(self.cbar_x_spin.value()),
            "position_y": float(self.cbar_y_spin.value()),
            "vertical": bool(self.cbar_vertical_check.isChecked()),
        }
        show_edges = self.edges_check.isChecked()
        for i, role in enumerate(roles):
            part = self.parts[role]
            if self.show_baseline_check.isChecked():
                base = part["poly"].copy(deep=True)
                base.points = part["base"].copy()
                self.plotter.add_mesh(base, color="black", style="wireframe", opacity=0.18,
                                      line_width=1.0, name=f"baseline_wire_{role}", show_scalar_bar=False)
            scalar_name, _ = fields[role]
            # Only the first visible part owns the scalar bar; the others share its clim.
            actor = self.plotter.add_mesh(
                part["poly"], scalars=scalar_name, clim=clim, show_edges=show_edges,
                smooth_shading=True, name=f"deformed_mesh_{role}",
                scalar_bar_args=scalar_bar_args if i == 0 else None,
                show_scalar_bar=(i == 0) and not picture_view,
            )
            if i == 0:
                self.mesh_actor = actor
            if clim is not None:
                try:
                    actor.mapper.scalar_range = tuple(clim)
                except Exception:
                    pass

        # Re-assert the range (interactive scalar-bar widgets cache the first one)
        if clim is not None and not picture_view:
            try:
                self.plotter.update_scalar_bar_range(clim, name=title)
            except Exception:
                pass

        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def refresh_context_actors(self):
        if self.plotter is None or not self.parts:
            return
        for name in ("c_surfaces", "other_surfaces", "anchor_nodes"):
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        if "C" in self.parts and self.show_c_check.isChecked():
            self.plotter.add_mesh(self.parts["C"]["poly"], color=(0.62, 0.64, 0.70), opacity=0.55,
                                  show_edges=self.edges_check.isChecked(), edge_color=(0.3, 0.3, 0.35),
                                  name="c_surfaces", pickable=False, show_scalar_bar=False)
        if "O" in self.parts and self.show_other_check.isChecked():
            self.plotter.add_mesh(self.parts["O"]["poly"], color=(0.8, 0.8, 0.8), opacity=0.15,
                                  style="wireframe", name="other_surfaces", pickable=False,
                                  show_scalar_bar=False)
        if self.anchor_poly is not None and self.show_anchor_check.isChecked():
            self.plotter.add_mesh(self.anchor_poly, color="magenta", point_size=6,
                                  render_points_as_spheres=True, name="anchor_nodes", pickable=False)
        try:
            self.plotter.render()
        except Exception:
            pass

    def refresh_control_node_actors(self):
        if self.control_nodes is None or self.plotter is None:
            return
        for name in ["control_nodes_base", "control_nodes_deformed"]:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        if self.control_nodes_base_poly is None:
            self.control_nodes_base_poly = pv.PolyData(self.control_nodes.copy())
        if self.control_nodes_deformed_poly is None:
            self.control_nodes_deformed_poly = pv.PolyData(self.control_nodes.copy())
        if self.show_cn_base_check.isChecked():
            self.plotter.add_mesh(self.control_nodes_base_poly, color="black", point_size=11,
                                  render_points_as_spheres=True, name="control_nodes_base", pickable=False)
        if self.show_cn_deformed_check.isChecked():
            self.plotter.add_mesh(self.control_nodes_deformed_poly, color="red", point_size=15,
                                  render_points_as_spheres=True, name="control_nodes_deformed", pickable=False)
        self.update_control_node_labels()
        try:
            self.plotter.render()
        except Exception:
            pass

    def update_control_node_labels(self):
        if self.control_nodes is None or self.plotter is None:
            return
        for actor in self.control_node_label_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.control_node_label_actors = []
        if not self.show_cn_labels_check.isChecked():
            try:
                self.plotter.render()
            except Exception:
                pass
            return
        d = self.current_ctrl_disp if self.current_ctrl_disp is not None else np.zeros_like(self.control_nodes)
        pts = self.control_nodes + float(self.scale_spin.value()) * d
        labels = []
        for i in range(len(pts)):
            lab = f"CN {i + 1}\n|d|={np.linalg.norm(d[i]):.3e}"
            if self.control_normals is not None:
                lab += f"\nd·n={float(np.dot(d[i], self.control_normals[i])):+.3e}"
            labels.append(lab)
        actor = self.plotter.add_point_labels(
            pts, labels, font_size=9, text_color="black", point_color="red", point_size=8,
            render_points_as_spheres=True, always_visible=True, name="control_node_modal_values",
        )
        self.control_node_label_actors.append(actor)
        try:
            self.plotter.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # The production morph (live)
    # ------------------------------------------------------------------

    def schedule_update(self):
        if self._closing or self._pending_update:
            return
        self._pending_update = True
        QTimer.singleShot(30, self.update_deformation)

    def compute_morph(self, coeffs: np.ndarray):
        """design vector -> (d_ctrl, d_verts_m) through the shared production code."""
        d_ctrl = control_displacements_from_basis(
            self.output_dir, self.basis, np.asarray(coeffs, float), log=lambda m: None, tag="[explorer]"
        )
        self.model.control_nodes = self.control_nodes.tolist()
        self.model.displacement_vector = d_ctrl.tolist()
        d_verts_m = np.asarray(deform_region(self.model, self.regions, self.rbf_params), float)
        return d_ctrl, d_verts_m

    def update_deformation(self):
        self._pending_update = False
        if self._closing or self.model is None or self.regions is None or not self.parts:
            return
        try:
            d_ctrl, d_verts_m = self.compute_morph(self.coeffs)
            node_disp = np.zeros((self.fro.node_count, 3))
            node_disp[np.asarray(self.regions.d_gids, int)] = d_verts_m - self.regions.d_verts
            self.current_node_disp = node_disp
            self.current_ctrl_disp = d_ctrl
            self.current_seam = seam_report(self.regions, d_verts_m)

            self.apply_display()
            self.update_control_node_table()

            sr = self.current_seam
            cn_rms = float(np.sqrt(np.mean(np.sum(d_ctrl ** 2, axis=1)))) if d_ctrl.size else 0.0
            self.status.setText(
                f"CN RMS={cn_rms:.4e} | D RMS={sr['d_rms']:.4e} | D max={sr['d_max']:.4e}\n"
                f"Seam (anchors, n={sr['n_anchor']}, shared={sr['n_shared']}): "
                f"max={sr['seam_max']:.4e}, rms={sr['seam_rms']:.4e}, "
                f"max seam / max D = {sr['seam_ratio']:.3f}\n"
                f"(values at scale 1; display ×{float(self.scale_spin.value()):.3g})"
                + ("\nWARNINGS:\n - " + "\n - ".join(self.case_warnings) if self.case_warnings else "")
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.status.setText(f"Update failed: {exc}")

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def zero_all_coefficients(self):
        self.coeffs[:] = 0.0
        self._sync_widgets_from_coeffs()
        self.schedule_update()

    def solo_mode(self, idx: int, value: float = 1.0):
        self.coeffs[:] = 0.0
        self.coeffs[idx] = float(value)
        self._sync_widgets_from_coeffs()
        self.schedule_update()

    def random_coefficients(self):
        rng = np.random.default_rng()
        r = float(self.range_spin.value())
        for i in range(len(self.coeffs)):
            val = rng.normal(0.0, 0.35 / ((i + 1) ** 1.5))
            self.coeffs[i] = float(np.clip(round(val, 2), -r, r))
        self._sync_widgets_from_coeffs()
        self.schedule_update()

    def on_load_design(self):
        if self.basis is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load design (morph_config_n_*.json or saved design)",
                                              "", "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            coeffs = np.asarray(cfg.get("modal_coeffs", cfg.get("design", [])), float).reshape(-1)
            if coeffs.size == 0:
                raise ValueError("No 'modal_coeffs' in file.")
            need = float(np.max(np.abs(coeffs))) if coeffs.size else 0.0
            if need > self.range_spin.value():
                self.range_spin.blockSignals(True)
                self.range_spin.setValue(float(np.ceil(need)))
                self.range_spin.blockSignals(False)
            self.coeffs = coeffs.copy()
            self.layout_info = describe_design_vector(self.basis, self.output_dir, coeffs.size)
            self.rebuild_sliders()
            self.update_mode_table()

            report = [f"Loaded {coeffs.size} coefficients from {os.path.basename(path)}."]
            d_ctrl, _ = self.compute_morph(self.coeffs)
            if "displacement_vector" in cfg and cfg["displacement_vector"]:
                ref = np.asarray(cfg["displacement_vector"], float).reshape((-1, 3))
                if ref.shape == d_ctrl.shape:
                    err = float(np.max(np.abs(ref - d_ctrl)))
                    scale = float(np.max(np.abs(ref))) or 1.0
                    report.append(f"d_ctrl vs config displacement_vector: max|Δ|={err:.3e} (rel {err / scale:.2e})")
                else:
                    report.append(f"Config displacement_vector shape {ref.shape} != explorer {d_ctrl.shape}")
            if "control_nodes" in cfg and cfg["control_nodes"]:
                cref = np.asarray(cfg["control_nodes"], float).reshape((-1, 3))
                if cref.shape == self.control_nodes.shape:
                    report.append(f"CN positions vs config: max|Δ|={float(np.max(np.abs(cref - self.control_nodes))):.3e}")
                else:
                    report.append(f"Config has {len(cref)} CNs, explorer has {len(self.control_nodes)}")
            for key, attr in (("t_surfaces", "t_surfaces"), ("u_surfaces", "u_surfaces"), ("c_surfaces", "c_surfaces")):
                if key in cfg and list(map(int, cfg[key])) != list(getattr(self.model, attr)):
                    report.append(f"{key} differ: config {cfg[key]} vs explorer {getattr(self.model, attr)}")
            QMessageBox.information(self, "Design loaded", "\n".join(report))
            self.schedule_update()
        except Exception as exc:
            QMessageBox.critical(self, "Load design failed", str(exc))

    def on_save_design(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save design vector", "design.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"modal_coeffs": self.coeffs.tolist(),
                       "labels": [i["label"] for i in self.layout_info]}, f, indent=2)
        self.status.setText(f"Saved design vector: {path}")

    def _run_morphmesh(self):
        """Run the real production MorphMesh on the current design (no files written)."""
        from MeshGeneration.Morph import MorphMesh

        d_ctrl = control_displacements_from_basis(self.output_dir, self.basis, self.coeffs, log=lambda m: None)
        mm = MorphModel(f_name=None, con=False, pb=[], path=None, T=[], U=[],
                        cn=self.control_nodes.tolist(), displacement_vector=d_ctrl.tolist())
        mm.t_surfaces = list(self.model.t_surfaces)
        mm.u_surfaces = list(self.model.u_surfaces)
        mm.c_surfaces = list(self.model.c_surfaces)
        mm.rigid_boundary_translation = self.model.rigid_boundary_translation
        logger = _QuietLogger()
        viewer = types.SimpleNamespace(logger=logger)  # not None -> MorphMesh writes nothing
        mesh_out = MorphMesh(self.fro, "explorer", mm, viewer, output_dir=None, debug=False)
        return mesh_out, logger

    def verify_against_morphmesh(self):
        if self.model is None:
            return
        try:
            self._set_busy("Running MorphMesh for verification...")
            mesh_out, logger = self._run_morphmesh()
            if mesh_out.node_count != self.fro.node_count:
                QMessageBox.warning(self, "Verify", "MorphMesh output has a different node count "
                                    "(FroFile.copy() compacted unreferenced nodes) - cannot compare by id.")
                return
            prod = np.asarray(mesh_out.nodes, float) - np.asarray(self.fro.nodes, float)
            err = float(np.max(np.abs(prod - self.current_node_disp)))
            ref = float(np.max(np.abs(prod))) or 1.0
            seam = [l for l in logger.lines if l.startswith("[SEAM]")]
            msg = (f"max |u_MorphMesh - u_explorer| = {err:.3e}  (relative {err / ref:.2e})\n"
                   f"max |u| (MorphMesh) = {ref:.3e}\n" + ("\n".join(seam)))
            QMessageBox.information(self, "Verify vs MorphMesh", msg)
            self.status.setText(msg)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Verify failed", str(exc))

    def export_morphed_fro(self):
        if self.model is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export morphed .fro", "explorer_morph.fro", "FRO (*.fro)")
        if not path:
            return
        try:
            self._set_busy("Running MorphMesh...")
            mesh_out, _ = self._run_morphmesh()
            mesh_out.write_file(path)
            self.status.setText(f"Wrote {path} (production MorphMesh, visual scale NOT applied)")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def reset_colorbar_position(self):
        for spin, v in ((self.cbar_x_spin, 0.7), (self.cbar_y_spin, 0.05)):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        self.refresh_actor()

    def toggle_picture_view(self):
        if not self.parts or self.plotter is None:
            return
        picture = self.picture_view_check.isChecked()
        try:
            self.plotter.hide_axes() if picture else self.plotter.show_axes()
        except Exception:
            pass
        try:
            if picture:
                self.plotter.remove_actor("title_text")
            else:
                self.plotter.add_text("Modal coefficient explorer (production morph)", position="upper_left",
                                      font_size=11, color="black", name="title_text")
        except Exception:
            pass
        self.refresh_actor(reset_camera=False)

    def save_screenshot(self, transparent: bool = False):
        if not self.parts or self.plotter is None:
            self.status.setText("Nothing to export yet - load a case first.")
            return
        default_name = "modal_explorer_transparent.png" if transparent else "modal_explorer.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", default_name, "PNG files (*.png)")
        if not path:
            return
        scale = int(self.screenshot_scale_spin.value())
        try:
            self.plotter.screenshot(path, transparent_background=bool(transparent), scale=scale)
            self.status.setText(f"Saved screenshot: {path} (scale x{scale})")
        except Exception as exc:
            QMessageBox.critical(self, "Screenshot failed", str(exc))


# -----------------------------------------------------------------------------
# Entrypoint (standalone use)
# -----------------------------------------------------------------------------

def parse_args() -> ModalState:
    import argparse
    ap = argparse.ArgumentParser(description="Modal explorer using the production morph")
    ap.add_argument("--basis", default=None, help="morph_basis.json")
    ap.add_argument("--mesh", default=None, help="baseline mesh (.fro/.vtm/.vtk/.case)")
    ap.add_argument("--output-dir", default=None, help="dir containing 'Control Nodes/modal_basis_T_surface.npz'")
    ap.add_argument("--scale", type=float, default=10.0, help="visual scale")
    a = ap.parse_args()
    return ModalState(morph_basis_path=a.basis, baseline_mesh_path=a.mesh,
                      output_dir=a.output_dir, deform_scale=a.scale)


def main(argv=None):
    state = parse_args()
    app = QApplication(sys.argv)
    win = ModalSliderExplorer(initial=state)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()