"""
solver_monitor_2d.py

The Solver view for a local 2D case (FLITE 2D PrePro.exe / Solver.exe):

    Design [n = 3 ...]   Condition [1]   [x] Follow the run        [Pictures] [Open folder]
    one status line: state, cycle, residual, lift, drag, moment
    +--------------------------------------------------------------------------+
    | geometry of the design (wall outline; the baseline dashed underneath)    |
    +--------------------------------------------------------------------------+
    | (toolbar)                                     Solver running  02:13      |
    | convergence, live: residual, lift, drag, moment against multigrid cycle, |
    | read from the .rsd while the solver writes it                            |
    +--------------------------------------------------------------------------+

The clock on the right is the time of the run on show: counting while its
preprocessor or solver is running, and the solver's run time once it has ended.

The pictures written when a solver run has ended (surface pressure, Mach, ...)
are in the Post-Processing view (GUI/post_view_2d.py); 'Pictures' opens it on
the design shown here.

Selection, polling and "Follow the run" are in GUI/run_browser_2d.py.
"""
import os
import time
import warnings

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from Local import flite2d_monitor as fm
from Local.flite2d_post import draw_convergence, draw_message, draw_wall_outline

try:
    from GUI.run_browser_2d import PictureView, RunBrowser2D        # noqa: F401  (PictureView: old import path)
except ImportError:
    from run_browser_2d import PictureView, RunBrowser2D            # noqa: F401

# zooming an equal-scale plot with the toolbar makes matplotlib say, on every redraw, that it
# widened one direction to keep the scales equal; that is what is wanted here
warnings.filterwarnings("ignore", message="Ignoring fixed [xy] limits to fulfill fixed data aspect")


def _toolbar(canvas, parent):
    tb = NavigationToolbar2QT(canvas, parent)
    tb.setIconSize(QSize(16, 16))
    return tb


class Solver2DMonitor(RunBrowser2D):
    NAME = "Solver view"

    def _init_state(self):
        self._follower = fm.RsdFollower()
        self._conv_run = self._conv_meta = None
        self._geom_key, self._geom_fail = None, 0
        self._walls = {}                 # mesh path -> (mtime, walls)
        self._clock = time.time          # replaced in tests

    # ------------------------------------------------------------------ layout
    def _top_buttons(self):
        self.pictures_btn = QPushButton("Pictures")
        self.pictures_btn.setToolTip("Open the Post-Processing view on this design: surface pressure, Mach,\n"
                                     "pressure and total-pressure pictures, written when a solver run has ended.")
        self.pictures_btn.clicked.connect(self._open_pictures)
        return [self.pictures_btn]

    def _build_body(self, lay):
        self.splitter = QSplitter(Qt.Vertical)
        geom = QWidget()
        gl = QVBoxLayout(geom)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(0)
        self.geom_fig = Figure(facecolor="white")
        self.geom_canvas = FigureCanvas(self.geom_fig)
        self.geom_canvas.setMinimumHeight(110)
        gl.addWidget(_toolbar(self.geom_canvas, geom))
        gl.addWidget(self.geom_canvas, 1)
        self.splitter.addWidget(geom)

        conv = QWidget()
        cl = QVBoxLayout(conv)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self.conv_fig = Figure(facecolor="white")
        self.conv_canvas = FigureCanvas(self.conv_fig)
        self.conv_canvas.setMinimumHeight(160)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 6, 0)
        bar.addWidget(_toolbar(self.conv_canvas, conv), 1)
        self.timer_label = QLabel("")
        self.timer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.timer_label.setToolTip("Time of the run on show: counting while its preprocessor or solver is running;\n"
                                    "the solver's run time once it has ended.")
        font = self.timer_label.font()
        font.setBold(True)
        self.timer_label.setFont(font)
        bar.addWidget(self.timer_label)
        cl.addLayout(bar)
        cl.addWidget(self.conv_canvas, 1)
        self.splitter.addWidget(conv)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 5)
        self.splitter.setSizes([230, 560])
        lay.addWidget(self.splitter, 1)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(200)
        self._resize_timer.timeout.connect(self._redraw)
        for canvas in (self.geom_canvas, self.conv_canvas):
            canvas.mpl_connect("resize_event", lambda _e: self._resize_timer.start())

    def _open_pictures(self):
        show = getattr(self.mw, "show_results_2d", None)
        if callable(show):
            show(self._n, self._nc)

    # ------------------------------------------------------------- base hooks
    def _reset_body(self):
        self._conv_run = self._geom_key = None
        self._walls = {}
        self._follower.set_path(None)

    def _clear_body(self):
        self._conv_run = self._geom_key = None
        self._follower.set_path(None)
        self.timer_label.setText("")
        for fig, canvas in ((self.geom_fig, self.geom_canvas), (self.conv_fig, self.conv_canvas)):
            fig.clear()
            canvas.draw_idle()

    def _show(self, out):
        self._show_geometry(out)
        st, state = self._show_convergence(out)
        self._show_timer(out, st, state)
        self._show_status(st, state)

    # ------------------------------------------------------------------ pieces
    def _read_walls(self, path):
        mt = os.stat(path).st_mtime
        hit = self._walls.get(path)
        if hit is None or hit[0] != mt:
            hit = (mt, fm.read_wall_outline(path))
            if len(self._walls) > 8:               # the baseline and a few designs are enough
                self._walls.pop(next(iter(self._walls)))
            self._walls[path] = hit
        return hit[1]

    def _show_geometry(self, out, force=False):
        n = self._n
        dat = fm.mesh_path(out, n)
        base = fm.mesh_path(out, 0) if n != 0 else None

        def stamp(p):
            try:
                return os.stat(p).st_mtime if p else None
            except OSError:
                return None
        key = (n, dat, stamp(dat), base, stamp(base))
        if key == self._geom_key and not force:
            return
        title = f"Geometry - design n = {n}" + (" (baseline)" if n == 0 else "")
        walls = base_walls = None
        problem = None
        if not dat:
            problem = "This design has no solver mesh (volumes/n_%d)." % n
        else:
            try:
                walls = self._read_walls(dat)
                if base and base != dat:
                    try:
                        base_walls = self._read_walls(base)
                    except (OSError, ValueError):
                        base_walls = None
                if not walls:
                    problem = "The solver mesh of this design has no wall (boundary flag 1 or 5)."
            except (OSError, ValueError) as e:
                # most likely the mesh is being written at this moment: look again on the next
                # ticks, then leave the message until the file changes
                self._geom_fail += 1
                if self._geom_fail <= 5:
                    key = None
                problem = f"The solver mesh could not be read yet ({e})."
        if problem is None:
            self._geom_fail = 0
            sub = f"{sum(len(P) for P, _ in walls):,} wall nodes"
            moved = fm.max_wall_displacement(walls, base_walls)
            if moved is not None:
                sub += f"; largest wall displacement from the baseline {moved:.4g}"
            draw_wall_outline(self.geom_fig, walls, base_walls, title=title, subtitle=sub)
        else:
            draw_message(self.geom_fig, problem, title=title)
        self._geom_key = key
        self.geom_canvas.draw_idle()

    def _show_convergence(self, out, force=False):
        n, nc = self._n, self._nc
        st = fm.run_state(out, n, nc) if nc is not None else None
        run = (n, nc, st.started if st else None)
        if run != self._conv_run:
            self._follower.set_path(st.rsd if st is not None else None)
            self._conv_run, force = run, True
        changed = self._follower.poll()
        state = self._state_text(st)
        meta = (state, st.target_orders if st else None)
        if changed or force or meta != self._conv_meta:
            self._conv_meta = meta
            H = self._follower.history
            head = f"n = {n}" + (f", condition {nc}" if nc is not None else "")
            bits = [st.describe_condition() if st else "", f"{int(H[-1, 0])} multigrid cycles" if len(H) else "",
                    state]
            title, sub = f"Convergence - {head}", "; ".join(b for b in bits if b)
            if len(H):
                draw_convergence(self.conv_fig, H, title=title, subtitle=sub,
                                 target_orders=st.target_orders if st else None)
            else:
                draw_message(self.conv_fig,
                             "Waiting for the first residual..." if state == "solver running" else
                             "The preprocessor is running. The residuals appear here when the solver starts."
                             if state.startswith("preprocessor") else
                             "No residual history for this design and condition.", title=title, subtitle=sub)
            self.conv_canvas.draw_idle()
        return st, state

    def _time_limit_s(self):
        """Time limit of a solver run, in seconds, as set on the Solver Settings page (None if none)."""
        try:
            minutes = float(getattr(self.mw, "flite2d_time_limit_min", 0) or 0)
        except (TypeError, ValueError):
            minutes = 0.0
        return 60.0 * minutes if minutes > 0 else None

    def _show_timer(self, out, st, state):
        """The clock of the run on show (see the top of this file)."""
        now, text = self._clock(), ""
        if state == "solver running":
            limit = self._time_limit_s()
            text = "Solver running  " + fm.format_duration(st.run_time(now))
            if limit:
                text += f"  /  {fm.format_duration(limit)} limit"
        elif state.startswith("preprocessor"):
            t0 = fm.prepro_started(out, self._n)
            text = "Preprocessor running" + (f"  {fm.format_duration(max(0.0, now - t0))}" if t0 else "")
        elif st is not None and st.finished:
            took = fm.format_duration(st.run_time())
            text = f"Solver run time  {took}" if took else ""
        if text != self.timer_label.text():
            self.timer_label.setText(text)

    def _show_status(self, st, state):
        bits = self._headline(st)
        bits.append(state)
        H = self._follower.history
        if len(H):
            it, res, cl, cd, cm = H[-1, :5]
            bits.append(f"cycle {int(it)}   residual {res:.4g}   lift {cl:.6g}   drag {cd:.6g}   "
                        f"moment {cm:.6g}")
        obj = self._objective(self._n)
        if obj is not None:
            bits.append(f"objective {obj:.6g}")
        if self._follower.error:
            bits.append("(the residual file could not be read this second; trying again)")
        self.status.setText("   |   ".join(bits))

    def _redraw(self):
        """After a resize: the margins of the figures are in inches, so draw them again."""
        if self._out is None or self._n is None or not self.isVisible():
            return
        try:
            self._show_geometry(self._out, force=True)
            self._show_convergence(self._out, force=True)
        except Exception:
            pass
