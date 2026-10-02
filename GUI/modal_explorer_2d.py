"""
modal_explorer_2d.py

Interactive explorer for 2D edge-mode morphs (MeshGeneration.Morph2D).

One slider per T-edge mode. Each change runs the SAME Morph2D.morph() the
pipeline uses (a few ms), redraws the domain mesh near T, and shows the
quality gate result (inverted triangles, boundary crossings, min area ratio).

Slider ranges are the mesh-validity limits from Morph2D.max_safe_amplitude,
i.e. where the morph stops producing a valid mesh, NOT physically sensible
bounds.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QSlider,
    QDoubleSpinBox, QPushButton, QScrollArea, QFileDialog, QCheckBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

_STEPS = 1000


class ModalExplorer2D(QMainWindow):
    def __init__(self, morpher, limits, output_dir=None, base_name="mesh", logger=None, parent=None):
        """
        morpher : MeshGeneration.Morph2D.Morph2D (already set up)
        limits  : list of (a_min, a_max) per mode (slider ranges)
        """
        super().__init__(parent)
        self.mo = morpher
        self.mesh = morpher.mesh
        self.limits = [(float(a), float(b)) for a, b in limits]
        self.output_dir = output_dir
        self.base_name = base_name
        self.logger = logger
        self.coeffs = np.zeros(self.mo.n_modes)
        self.last_report = None
        self.last_mesh = None

        self.setWindowTitle(f"2D Edge-Mode Explorer: T edges {self.mo.t_edges}, {self.mo.n_modes} modes")
        self.resize(1400, 820)
        self._prepare_view_subset()
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.update_morph)
        self.update_morph()

    # ------------------------------------------------------------------ setup
    def _log(self, msg):
        if self.logger is not None:
            try:
                self.logger.log(msg)
                return
            except Exception:
                pass
        print(msg)

    def _prepare_view_subset(self):
        """Only draw triangles that can move (within R of T) plus a margin."""
        P = self.mesh.nodes[:, :2]
        T = self.mesh.triangles
        tx = self.mesh.nodes[self.mo.t_ids, :2]
        lo, hi = tx.min(0), tx.max(0)
        pad = 1.25 * self.mo.R
        self.view_box = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
        c = P[T].mean(axis=1)
        x0, x1, y0, y1 = self.view_box
        self.view_tris = T[(c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)]
        # boundary segments for context
        segs = []
        for e in self.mesh.edges.values():
            segs.append((e.id, e.nodes))
        self.edge_segs = segs

    def _build_ui(self):
        root = QWidget()
        lay = QHBoxLayout(root)

        # plot
        left = QVBoxLayout()
        self.fig = Figure(figsize=(9, 7))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        left.addWidget(NavToolbar(self.canvas, self))
        left.addWidget(self.canvas)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        lay.addLayout(left, 3)

        # sliders
        right = QVBoxLayout()
        info = QLabel(
            f"T length L = {self.mo.modes.length:.4g}, RBF radius R = {self.mo.R:.4g}\n"
            f"Coefficient a_k = amplitude of mode k (mesh units).\n"
            f"Slider range = mesh-validity limit, not a design bound."
        )
        info.setWordWrap(True)
        right.addWidget(info)

        grid = QGridLayout()
        self.sliders, self.boxes = [], []
        for k in range(self.mo.n_modes):
            lo, hi = self.limits[k]
            grid.addWidget(QLabel(f"Mode {k + 1}  (λ={self.mo.modes.lam[k]:.3g})"), 2 * k, 0, 1, 2)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, _STEPS)
            sl.setValue(self._to_slider(k, 0.0))
            box = QDoubleSpinBox()
            box.setDecimals(5)
            box.setRange(lo, hi)
            box.setSingleStep(max((hi - lo) / 200.0, 1e-6))
            box.setValue(0.0)
            sl.valueChanged.connect(lambda v, i=k: self._slider_changed(i, v))
            box.valueChanged.connect(lambda v, i=k: self._box_changed(i, v))
            grid.addWidget(sl, 2 * k + 1, 0)
            grid.addWidget(box, 2 * k + 1, 1)
            self.sliders.append(sl)
            self.boxes.append(box)
        holder = QWidget()
        holder.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        right.addWidget(scroll, 1)

        self.show_mesh_cb = QCheckBox("Show mesh")
        self.show_mesh_cb.setChecked(True)
        self.show_mesh_cb.stateChanged.connect(lambda _: self.redraw())
        right.addWidget(self.show_mesh_cb)

        for text, fn in [("Zero all", self.zero_all),
                         ("Random (within 50% of limits)", self.random_coeffs),
                         ("Export morphed mesh", self.export_mesh),
                         ("Save design", self.save_design),
                         ("Load design", self.load_design)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            right.addWidget(b)
        lay.addLayout(right, 1)
        self.setCentralWidget(root)

    # ------------------------------------------------------------ sliders
    def _to_slider(self, k, a):
        lo, hi = self.limits[k]
        return int(round((a - lo) / max(hi - lo, 1e-300) * _STEPS))

    def _from_slider(self, k, v):
        lo, hi = self.limits[k]
        return lo + (hi - lo) * v / _STEPS

    def _slider_changed(self, k, v):
        a = self._from_slider(k, v)
        self.boxes[k].blockSignals(True)
        self.boxes[k].setValue(a)
        self.boxes[k].blockSignals(False)
        self.coeffs[k] = a
        self._timer.start(40)

    def _box_changed(self, k, a):
        self.sliders[k].blockSignals(True)
        self.sliders[k].setValue(self._to_slider(k, a))
        self.sliders[k].blockSignals(False)
        self.coeffs[k] = a
        self._timer.start(40)

    def set_coeffs(self, coeffs):
        c = np.asarray(coeffs, float).ravel()
        if c.size != self.mo.n_modes:
            raise ValueError(f"expected {self.mo.n_modes} coefficients, got {c.size}")
        for k, a in enumerate(c):
            lo, hi = self.limits[k]
            a = float(np.clip(a, lo, hi))
            self.coeffs[k] = a
            for w, val in ((self.boxes[k], a), (self.sliders[k], self._to_slider(k, a))):
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)
        self.update_morph()

    # ------------------------------------------------------------ morph + draw
    def update_morph(self):
        self.last_mesh, self.last_report = self.mo.morph(self.coeffs)
        self.redraw()

    def redraw(self):
        ax = self.ax
        ax.clear()
        P0 = self.mesh.nodes
        P1 = self.last_mesh.nodes
        if self.show_mesh_cb.isChecked() and len(self.view_tris):
            ax.triplot(P1[:, 0], P1[:, 1], self.view_tris, lw=0.25, color="#4c78a8")
        for eid, nodes in self.edge_segs:
            ax.plot(P1[nodes, 0], P1[nodes, 1], "-", color="0.35", lw=1.5)
        t = self.mo.t_ids
        ax.plot(P0[t, 0], P0[t, 1], "--", color="k", lw=1.2, label="baseline T")
        ax.plot(P1[t, 0], P1[t, 1], "-", color="#d62728", lw=2.2, label="morphed T")
        if len(self.mo.u_ids):
            u = self.mo.u_ids
            ax.plot(P1[u, 0], P1[u, 1], ".", color="#ff7f0e", ms=3, label="U nodes")
        x0, x1, y0, y1 = self.view_box
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        rep = self.last_report
        ax.set_title("OK" if rep.ok else f"INVALID: {rep.reason}",
                     color="#2ca02c" if rep.ok else "#d62728")
        self.canvas.draw_idle()
        self.status.setText(str(rep))

    # ------------------------------------------------------------ actions
    def zero_all(self):
        self.set_coeffs(np.zeros(self.mo.n_modes))

    def random_coeffs(self):
        rng = np.random.default_rng()
        c = [rng.uniform(0.5 * lo, 0.5 * hi) for lo, hi in self.limits]
        self.set_coeffs(c)

    def _default_dir(self):
        d = self.output_dir or os.getcwd()
        out = os.path.join(d, "surfaces", "n_0")
        os.makedirs(out, exist_ok=True)
        return out

    def export_mesh(self, path=None):
        if not self.last_report.ok:
            self._log(f"[2D-EXPLORER][WARN] Exporting an INVALID morph: {self.last_report.reason}")
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export morphed mesh",
                os.path.join(self._default_dir(), f"{self.base_name}_2d_morph.fro"),
                "FLITE fro (*.fro);;VTK (*.vtu)")
            if not path:
                return None
        if path.lower().endswith(".vtu"):
            self.last_mesh.write_vtk(path)
        else:
            self.last_mesh.write_file(path)
        self._log(f"[2D-EXPLORER] Exported morphed mesh -> {path}  ({self.last_report})")
        return path

    def save_design(self, path=None):
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save design", os.path.join(self._default_dir(), "design_2d.json"), "JSON (*.json)")
            if not path:
                return None
        rep = self.last_report
        with open(path, "w") as f:
            json.dump({"coeffs": self.coeffs.tolist(), "t_edges": self.mo.t_edges,
                       "u_edges": self.mo.u_edges, "ok": rep.ok, "reason": rep.reason,
                       "min_area_ratio": rep.min_area_ratio}, f, indent=2)
        self._log(f"[2D-EXPLORER] Saved design -> {path}")
        return path

    def load_design(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load design", self._default_dir(), "JSON (*.json)")
            if not path:
                return
        with open(path) as f:
            d = json.load(f)
        self.set_coeffs(d["coeffs"])
        self._log(f"[2D-EXPLORER] Loaded design <- {path}")