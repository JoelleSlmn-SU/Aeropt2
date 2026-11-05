"""
run_browser_2d.py

What the Solver view (GUI/solver_monitor_2d.py) and the Post-Processing view
(GUI/post_view_2d.py) of a local 2D case have in common:

    Design [n = 3 ...]   Condition [1]   [x] follow ...                [buttons]
    one status line

and the machinery behind it: reading the output folder once a second while the
view is on screen (Local/flite2d_monitor.py says which files, and why those),
keeping the list of designs up to date, and deciding which design and flow
condition are on show. A subclass builds its own body and draws it in _show().

The views only read files. They therefore work the same for 'Run Simulation'
(a worker thread of the main window), for 'Run Optimisation' (another process),
and for designs of a run that ended earlier.

The check box keeps the view on the run the subclass names in _follow_target().
Picking a design or condition by hand turns it off; a new run turns it back on.
"""
import math
import os

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QVBoxLayout, QWidget)

from Local import flite2d_monitor as fm


class PictureView(QWidget):
    """A picture file scaled to the space available (aspect ratio kept). The file is read
    again only when it has changed. Double-click opens it in the system's picture viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setMinimumSize(1, 1)
        self._label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)
        self._pix, self._key, self.path = None, None, None

    def note(self):
        """The text shown in place of a picture ('' while a picture is on show)."""
        return "" if self.path else self._label.text()

    def show_picture(self, path, note=""):
        """Returns True if what is shown changed."""
        key = None
        if path:
            try:
                st = os.stat(path)
                key = (path, st.st_mtime, st.st_size)
            except OSError:
                path, note = None, note or "The picture file is not there any more."
        if key is None:
            key = (None, note)
        if key == self._key:
            return False
        if path:
            pix = QPixmap(path)
            if pix.isNull():                    # e.g. being replaced right now: look again next time
                self._key = None
                return False
            self._pix, self.path = pix, path
            self._label.setToolTip(f"{path}\nDouble-click to open the file.")
            self._rescale()
        else:
            self._pix, self.path = None, None
            self._label.setToolTip("")
            self._label.setPixmap(QPixmap())
            self._label.setText(note)
        self._key = key
        return True

    def _rescale(self):
        if self._pix is not None and self._label.width() > 1 and self._label.height() > 1:
            self._label.setPixmap(self._pix.scaled(self._label.size(), Qt.KeepAspectRatio,
                                                   Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def mouseDoubleClickEvent(self, event):
        if self.path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))


class RunBrowser2D(QWidget):
    POLL_MS = 1000
    NAME = "view"                                   # used in the one warning this class can log
    TWO_ROWS = False                                # selectors on one line, check box and buttons on the next
    FOLLOW_TEXT = "Follow the run"
    FOLLOW_TIP = ("Stay on the design and flow condition that were started last.\n"
                  "Picking a design or condition by hand turns this off.")
    NO_OUTPUT_TEXT = "Set the output directory (toolbar) to see the runs of a 2D case here."
    NO_DESIGN_TEXT = ("No design yet in this output directory. Save the 2D basis (it writes the baseline "
                      "solver mesh), then press Run Simulation or Run Optimisation.")

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.mw = main_window
        self._out = None                 # output directory the state below belongs to
        self._n, self._nc = None, None   # design and condition on show
        self._designs, self._design_labels = [], []
        self._conds = None
        self._latest = None              # (n, cond) started last, from the files
        self._opt, self._opt_mtime = {}, None
        self._tick = 0
        self._error_logged = False
        self._init_state()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self._build_top(lay)
        self._build_body(lay)
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self.refresh)

    # ----------------------------------------------------------- for subclasses
    def _init_state(self):
        """Attributes of the subclass that _build_body and _show need."""

    def _build_body(self, layout):
        raise NotImplementedError

    def _top_buttons(self):
        """Extra buttons for the right end of the top row (before 'Open folder')."""
        return []

    def _extra_selectors(self, top):
        """Add selectors to the top row, in front of 'Design:' (e.g. a generation chooser)."""

    def _filter_designs(self, designs):
        """The designs to list, out of those on disk (called at every rescan, after the
        optimisation's table has been read into self._opt)."""
        return designs

    def _follow_target(self, out):
        """(design, condition or None) to stay on while the check box is ticked; None = no change."""
        return self._latest

    def _show(self, out):
        """Draw design self._n, condition self._nc (which may be None)."""
        raise NotImplementedError

    def _reset_body(self):
        """The output directory changed: forget what was cached for the old one."""

    def _clear_body(self):
        """Nothing to show (no output directory, or no design in it)."""

    # ------------------------------------------------------------------ top row
    def _build_top(self, lay):
        top = QHBoxLayout()
        self._extra_selectors(top)
        top.addWidget(QLabel("Design:"))
        self.design_combo = QComboBox()
        # the entries can be long (generation, objective, status): the box takes the room there
        # is and may be narrower than its text; the list that drops down is always wide enough
        self.design_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.design_combo.setMinimumContentsLength(14)
        self.design_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.design_combo.setMaximumWidth(460)
        self.design_combo.setToolTip("Designs with a solver mesh or results in the output directory.\n"
                                     "n = 0 is the baseline; in an optimisation the others are its designs,\n"
                                     "listed with their generation and, once computed, their objective.")
        self.design_combo.activated.connect(self._on_design_picked)      # user action only
        top.addWidget(self.design_combo, 2)
        top.addWidget(QLabel("Condition:"))
        self.cond_combo = QComboBox()
        self.cond_combo.setMinimumWidth(60)
        self.cond_combo.activated.connect(self._on_cond_picked)
        top.addWidget(self.cond_combo)
        row = top
        if self.TWO_ROWS:
            top.addStretch(1)
            lay.addLayout(top)
            row = QHBoxLayout()
        self.follow_cb = QCheckBox(self.FOLLOW_TEXT)
        self.follow_cb.setChecked(True)
        self.follow_cb.setToolTip(self.FOLLOW_TIP)
        self.follow_cb.toggled.connect(self._on_follow_toggled)
        row.addWidget(self.follow_cb)
        row.addStretch(1)
        for btn in self._top_buttons():
            row.addWidget(btn)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setToolTip("Open the folder of the run on show (solutions/n_<n>/cond_<i>).")
        self.open_btn.clicked.connect(self._open_folder)
        row.addWidget(self.open_btn)
        lay.addLayout(row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)

    # ------------------------------------------------------------ main window
    def output_dir(self):
        return getattr(self.mw, "output_directory", None) or None

    def _run_active(self):
        return (getattr(self.mw, "sim2d_thread", None) is not None
                or getattr(self.mw, "opt2d_proc", None) is not None)

    def run_started(self):
        """Called by the main window when a local 2D simulation or optimisation starts."""
        self.follow_cb.setChecked(True)
        if self.isVisible():
            self.refresh(rescan=True)

    def run_finished(self):
        """Called by the main window when it ends: show the end state now."""
        if self.isVisible():
            self.refresh(rescan=True)

    def select(self, n, cond=None):
        """Show design n (and flow condition cond), as if the user had picked them."""
        if n is None:
            return
        self.follow_cb.setChecked(False)
        self._n, self._nc = int(n), (int(cond) if cond is not None else None)
        self.refresh(rescan=True)

    def current(self):
        return self._n, self._nc

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh(rescan=True)
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    # ----------------------------------------------------------------- polling
    def refresh(self, rescan=False):
        try:
            self._refresh(bool(rescan))
        except Exception as e:                   # a timer callback must never raise into Qt
            if not self._error_logged:
                self._error_logged = True
                self._log(f"[2D][MONITOR][WARN] The {self.NAME} could not be updated: {type(e).__name__}: {e}")

    def _log(self, msg):
        logger = getattr(self.mw, "logger", None)
        try:
            logger.log(msg) if logger is not None else print(msg)
        except Exception:
            print(msg)

    def _refresh(self, rescan):
        out = self.output_dir()
        if not out or not os.path.isdir(out):
            self._clear(self.NO_OUTPUT_TEXT)
            return
        if out != self._out:
            self._out, self._n, self._nc = out, None, None
            self._designs, self._design_labels, self._conds = [], [], None
            self._opt, self._opt_mtime = {}, None
            self._reset_body()
            rescan = True
        self._tick += 1
        if rescan or self._tick % 3 == 0 or not self._designs:
            self._rescan(out)
        self._latest = fm.latest_run(out)
        if not self._designs:
            self._clear(self.NO_DESIGN_TEXT)
            return
        target = self._follow_target(out) if self.follow_cb.isChecked() else None
        if target is not None:
            if target[0] not in self._designs:
                self._rescan(out)
            if target[0] in self._designs:
                self._n, self._nc = target
        if self._n not in self._designs:
            self._n, self._nc = self._designs[-1], None
        conds = fm.condition_numbers(out, self._n)
        keep_none = target is not None and target == (self._n, None)       # e.g. its preprocessor is running
        if self._nc not in conds and not keep_none:
            self._nc = self._default_condition(out, self._n, conds)
        self._sync_combos(conds)
        self._show(out)

    def _default_condition(self, out, n, conds):
        """The condition of design n whose solver was started last (None if it has none)."""
        best, when = None, -1.0
        for nc in conds:
            t = fm.run_state(out, n, nc).started or 0.0
            if t >= when:
                best, when = nc, t
        return best

    def _rescan(self, out):
        designs = fm.design_numbers(out)
        opt_file = os.path.join(out, "opt", "designs.json")
        try:
            mt = os.stat(opt_file).st_mtime
        except OSError:
            mt = None
        if mt != self._opt_mtime:
            self._opt_mtime, self._opt = mt, fm.opt_designs(out)
        designs = self._filter_designs(designs)
        labels = [self._design_label(n) for n in designs]
        if designs != self._designs:
            self.design_combo.blockSignals(True)
            self.design_combo.clear()
            for n, text in zip(designs, labels):
                self.design_combo.addItem(text, n)
            self.design_combo.blockSignals(False)
        else:
            for i, text in enumerate(labels):
                if text != self._design_labels[i]:
                    self.design_combo.setItemText(i, text)
        if labels != self._design_labels and labels:
            try:                                    # the drop-down list shows every entry in full
                metrics = self.design_combo.fontMetrics()
                self.design_combo.view().setMinimumWidth(max(metrics.horizontalAdvance(t) for t in labels) + 48)
            except Exception:
                pass
        self._designs, self._design_labels = designs, labels

    def _generation(self, n):
        """Generation of design n in the optimisation's table (None if it is not in it)."""
        row = self._opt.get(n)
        g = row.get("gen") if row else None
        return int(g) if isinstance(g, (int, float)) else None

    def _objective(self, n):
        """Objective of design n from the optimisation's table (None if it has none yet)."""
        row = self._opt.get(n)
        obj = row.get("objective") if row and row.get("done") else None
        return float(obj) if isinstance(obj, (int, float)) and math.isfinite(obj) else None

    def _design_label(self, n, with_generation=True):
        text = f"n = {n}" + (" (baseline)" if n == 0 else "")
        row = self._opt.get(n)
        if row:
            if with_generation and row.get("gen") is not None:
                text += f"    generation {row['gen']}"
            if row.get("done"):
                obj = self._objective(n)
                if obj is not None:
                    text += f"    objective {obj:.5g}"
                status = row.get("status")
                if status and status != "ok":
                    text += f"    [{status}]"
        return text

    def _sync_combos(self, conds):
        i = self._designs.index(self._n)
        if self.design_combo.currentIndex() != i:
            self.design_combo.blockSignals(True)
            self.design_combo.setCurrentIndex(i)
            self.design_combo.blockSignals(False)
        items = list(conds) if self._nc is not None or conds else []
        shown = (self._n, tuple(items), self._nc)
        if shown != self._conds:
            self.cond_combo.blockSignals(True)
            self.cond_combo.clear()
            if self._nc is None:
                self.cond_combo.addItem("-", None)
            for nc in items:
                self.cond_combo.addItem(str(nc), nc)
            idx = self.cond_combo.findData(self._nc)
            self.cond_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.cond_combo.blockSignals(False)
            self._conds = shown

    def _on_follow_toggled(self, on):
        if on and self.isVisible():
            self.refresh()

    def _on_design_picked(self, index):
        n = self.design_combo.itemData(index)
        if n is None:
            return
        self.follow_cb.setChecked(False)
        self._n, self._nc = int(n), None         # refresh picks its most recent condition
        self.refresh()

    def _on_cond_picked(self, index):
        nc = self.cond_combo.itemData(index)
        if nc is None:
            return
        self.follow_cb.setChecked(False)
        self._nc = int(nc)
        self.refresh()

    def _open_folder(self):
        out = self.output_dir()
        if not out:
            return
        for path in ((fm.cond_dir(out, self._n, self._nc) if None not in (self._n, self._nc) else None),
                     (os.path.join(out, "solutions", f"n_{self._n}") if self._n is not None else None), out):
            if path and os.path.isdir(path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
                return

    def _clear(self, text):
        if self.status.text() == text and not self._designs and self._n is None:
            return                                  # already showing exactly this
        self.status.setText(text)
        self._n = self._nc = None
        self._conds = None
        self.design_combo.blockSignals(True)
        self.design_combo.clear()
        self.design_combo.blockSignals(False)
        self.cond_combo.blockSignals(True)
        self.cond_combo.clear()
        self.cond_combo.blockSignals(False)
        self._designs, self._design_labels = [], []
        self._clear_body()

    # ---------------------------------------------------------- shared wording
    def _state_text(self, st):
        """In words, what the files say about the condition on show."""
        current = self._run_active() and self._latest is not None and self._latest[0] == self._n
        if st is None or not st.exists or st.started is None:
            if current and self._latest[1] is None:
                return "preprocessor running - the solver has not started on this design yet"
            return "not run"
        if st.finished:
            if st.status == "ok" and (st.metrics or {}).get("converged_by") == "forces":
                return "finished (time limit; forces settled)"
            return {"ok": "finished"}.get(st.status, f"ended: {st.status}" if st.status else "ended")
        if current and self._latest[1] == st.cond:
            return "solver running"
        return "no result (the run was interrupted)"

    def _headline(self, st):
        n, nc = self._n, self._nc
        bits = [f"n = {n}" + (" (baseline)" if n == 0 else "")]
        if nc is not None:
            bits.append(f"condition {nc}" + (f" ({st.describe_condition()})" if st and st.describe_condition()
                                             else ""))
        return bits
