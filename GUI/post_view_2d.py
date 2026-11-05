"""
post_view_2d.py

The Post-Processing view for a local 2D case: the pictures that
Local/flite2d_post.py writes when a solver run has ended, for whichever
generation, design and flow condition is chosen.

    Generation [1 (1 design)]   Design [n = 6    generation 1    objective ...]   Condition [1]
    [x] Show the latest finished run                                         [Open folder]
    one line: how the run ended, cycles, residual, lift, drag, moment, objective, run time
    +--------------------------------------------------------------------------------+
    | Surface Cp | Mach | Cp | Total pressure | Convergence | Surface mesh | History  |
    |   the picture, scaled to the page; double-click opens the file                 |
    +--------------------------------------------------------------------------------+

"History" is the optimisation's own picture (<output>/opt/history.png); it does
not depend on the design chosen.

"Generation" lists the generations of the local optimisation (its table,
<output>/opt/designs.json, says which design belongs to which): generation 0 is
the baseline with the initial sample, every later generation is one optimiser
step. Choosing one limits the design list to its designs; "All" also lists
designs that are not in the table (a single 'Run Simulation', or folders left
by an earlier optimisation).

With the check box ticked the view moves to each run as it ends, so during an
optimisation the pictures of the design that has just been computed are on
show. Picking a generation, design or condition by hand turns it off.

Selection and polling are in GUI/run_browser_2d.py; nothing here writes a file.
"""
import os

from PyQt5.QtWidgets import QComboBox, QLabel, QTabWidget

from Local import flite2d_monitor as fm

try:
    from GUI.run_browser_2d import PictureView, RunBrowser2D
except ImportError:
    from run_browser_2d import PictureView, RunBrowser2D

HISTORY_TAB = "History"
NO_HISTORY = ("No optimisation history yet. The picture (<output>/opt/history.png) is written by a local 2D "
              "optimisation after each generation.")


class Post2DViewer(RunBrowser2D):
    NAME = "Post-Processing view"
    FOLLOW_TEXT = "Show the latest finished run"
    FOLLOW_TIP = ("Move to each solver run as it ends, so the newest pictures are on show.\n"
                  "Picking a generation, design or condition by hand turns this off.")
    ALL_GENERATIONS = -1                             # item data of "All" in the generation list
    TWO_ROWS = True                                  # three selectors need the width of the page

    def _init_state(self):
        self.titles = [t for _, t in fm.RUN_PICTURES] + ["Surface mesh", HISTORY_TAB]
        self._gen = None                             # generation chosen (None = all designs)
        self._gen_items = None                       # what the generation list holds now

    # ------------------------------------------------------- generation chooser
    def _extra_selectors(self, top):
        top.addWidget(QLabel("Generation:"))
        self.gen_combo = QComboBox()
        self.gen_combo.setMinimumWidth(150)
        self.gen_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.gen_combo.setToolTip("Generations of the local optimisation (from <output>/opt/designs.json).\n"
                                  "0 is the baseline with the initial sample; each later one is an optimiser step.\n"
                                  "Choosing one limits 'Design' to the designs of that generation.")
        self.gen_combo.addItem("All", self.ALL_GENERATIONS)
        self.gen_combo.activated.connect(self._on_generation_picked)   # user action only
        top.addWidget(self.gen_combo)

    def _filter_designs(self, designs):
        count = {}
        for n in designs:
            g = self._generation(n)
            if g is not None:
                count[g] = count.get(g, 0) + 1
        if self._gen is not None and self._gen not in count:
            self._gen = None                         # that generation is gone (e.g. a new optimisation)
        items = [(self.ALL_GENERATIONS, "All")] + [
            (g, f"{g}   ({count[g]} design{'' if count[g] == 1 else 's'})") for g in sorted(count)]
        self.gen_combo.blockSignals(True)
        if items != self._gen_items:
            self.gen_combo.clear()
            for g, text in items:
                self.gen_combo.addItem(text, g)
            self._gen_items = items
        i = self.gen_combo.findData(self.ALL_GENERATIONS if self._gen is None else self._gen)
        if i >= 0 and i != self.gen_combo.currentIndex():
            self.gen_combo.setCurrentIndex(i)
        self.gen_combo.blockSignals(False)
        self.gen_combo.setEnabled(bool(count))       # nothing to choose without an optimisation table
        return designs if self._gen is None else [n for n in designs if self._generation(n) == self._gen]

    def _design_label(self, n, with_generation=True):
        # with one generation chosen, every entry of the design list belongs to it
        return super()._design_label(n, with_generation=self._gen is None)

    def _on_generation_picked(self, index):
        g = self.gen_combo.itemData(index)
        if g is None:
            return
        self.follow_cb.setChecked(False)
        self._gen = None if int(g) == self.ALL_GENERATIONS else int(g)
        if self._gen is not None and self._generation(self._n) != self._gen:
            self._n = self._nc = None                # refresh shows the last design of that generation
        self.refresh(rescan=True)

    def select(self, n, cond=None):
        if n is not None and self._gen is not None and self._generation(int(n)) != self._gen:
            self._gen = None                         # the design asked for is outside the chosen generation
        super().select(n, cond)

    def _build_body(self, lay):
        self.tabs = QTabWidget()
        self.picture_views = []
        for title in self.titles:
            view = PictureView()
            self.picture_views.append(view)
            self.tabs.addTab(view, title)
        self.tabs.currentChanged.connect(lambda _i: self._show_picture())
        lay.addWidget(self.tabs, 1)

    # ------------------------------------------------------------- base hooks
    def _follow_target(self, out):
        target = fm.latest_finished_run(out)
        if target is not None and self._gen is not None and self._generation(target[0]) != self._gen:
            self._gen = self._generation(target[0])  # go with it (the base class then lists that generation)
        return target

    def _reset_body(self):
        self._gen = None

    def _clear_body(self):
        self._gen = None
        self._filter_designs([])
        for view in self.picture_views:
            view.show_picture(None, "")

    def _show(self, out):
        n, nc = self._n, self._nc
        st = fm.run_state(out, n, nc) if nc is not None else None
        bits = self._headline(st)
        bits.append(self._state_text(st))
        m = st.metrics if st is not None and st.finished else {}
        if m:
            def num(key):
                v = m.get(key)
                return f"{v:.6g}" if isinstance(v, (int, float)) else "-"
            it = m.get("iterations")
            bits.append(f"cycle {int(it) if isinstance(it, (int, float)) else '-'}   residual {num('residual')}   "
                        f"lift {num('CL')}   drag {num('CD')}   moment {num('CM')}")
            if st.status != "ok":
                bits.append("the numbers are those of the last cycle reached, not of a finished run")
            fc = m.get("force_check") or {}
            if m.get("converged_by") == "forces" and fc.get("CL_spread") is not None:
                bits.append(f"time limit reached; over the last {fc.get('window')} cycles the lift coefficient "
                            f"moved {fc['CL_spread']:.2g}, drag {fc['CD_spread']:.2g} (tol {fc.get('tol'):.2g})")
        obj = self._objective(n)
        if obj is not None:
            bits.append(f"objective {obj:.6g}")
        took = fm.format_duration(st.run_time()) if st is not None and st.finished else ""
        if took:
            bits.append(f"solver run time {took}")
        self.status.setText("   |   ".join(bits))
        self._show_picture()

    def _show_picture(self):
        """Load the picture of the tab on show (the others are loaded when their tab is opened)."""
        out, i = self._out, self.tabs.currentIndex()
        if out is None or i < 0:
            return
        if self.titles[i] == HISTORY_TAB:
            path = fm.opt_history_path(out)
            self.picture_views[i].show_picture(path, "" if path else NO_HISTORY)
            return
        if self._n is None:
            return
        title, path, note = fm.run_pictures(out, self._n, self._nc)[i]
        self.picture_views[i].show_picture(path, note)

    def picture_path(self, title):
        """Path of the picture now shown in the tab called `title` (None if it shows a note)."""
        return self.picture_views[self.titles.index(title)].path
