"""
Tests of the two views of a local 2D case: the Solver view (GUI/solver_monitor_2d.py: geometry, live
convergence, run clock) and the Post-Processing view (GUI/post_view_2d.py: the pictures of a run).

    python -m pytest tests/test_solver_monitor_2d.py -q      or      python tests/test_solver_monitor_2d.py

The widgets are built without a screen (Qt "offscreen" platform), next to runs
made by the real runner with the stand-in programs of tests/standins. Skipped
if PyQt5 is not installed. What a figure looks like has to be judged by eye;
checked here is what the views decide to show, and when.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:                                                               # pragma: no cover
    QApplication = None

from test_edge_modes_2d import naca0012, o_mesh                                   # noqa: E402
from FileRW.FliteMesh2D import default_edge_flags, write_flite2d_mesh             # noqa: E402
from Local import flite2d_monitor as fm                                           # noqa: E402
from Local.flite2d_local import Flite2DCase, PreProOptions, set_namelist_value    # noqa: E402

PREPRO = [sys.executable, os.path.join(HERE, "standins", "fake_prepro.py")]
SOLVER = [sys.executable, os.path.join(HERE, "standins", "fake_solver.py")]
with open(os.path.join(os.path.dirname(HERE), "templates", "flite2d_solver.inp")) as _fh:
    TEMPLATE = _fh.read()


class _Window:
    """The attributes of the main window that the views read."""
    def __init__(self, out):
        self.output_directory, self.sim2d_thread, self.opt2d_proc = out, None, None
        self.flite2d_time_limit_min = 0
        self.messages, self.opened = [], []
        self.logger = self

    def log(self, msg):
        self.messages.append(msg)

    def show_results_2d(self, n=None, cond=None):
        self.opened.append((n, cond))


def _case(out, n=0, scale_y=1.0):
    m, _ = o_mesh(naca0012(40), n_layers=10)
    m.nodes[:, 1] *= scale_y
    case = Flite2DCase(out, "naca", n=n, log=lambda s: None, prepro_cmd=PREPRO, solver_cmd=SOLVER)
    write_flite2d_mesh(m, case.mesh_path, default_edge_flags(m))
    return case


def _pump(app, seconds=0.0):
    t0 = time.time()
    app.processEvents()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


def _tab(view, title):
    """Open the tab called `title` of the Post-Processing view; returns its picture widget."""
    i = view.titles.index(title)
    view.tabs.setCurrentIndex(i)
    return view.picture_views[i]


def test_views_follow_a_run_and_let_the_user_browse():
    if QApplication is None:
        try:
            import pytest
            pytest.skip("PyQt5 is not installed")
        except ImportError:
            return
    from GUI.post_view_2d import NO_HISTORY, Post2DViewer
    from GUI.solver_monitor_2d import Solver2DMonitor
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out put dir")
        win = _Window(None)
        mon, post = Solver2DMonitor(win), Post2DViewer(win)
        for view in (mon, post):
            view.resize(900, 800)
            view.show()
        _pump(app)
        for view in (mon, post):
            assert "output directory" in view.status.text() and view.design_combo.count() == 0
        win.output_directory = out
        os.makedirs(out)
        for view in (mon, post):
            view.refresh()
            assert "No design yet" in view.status.text()
        assert post.titles == ["Surface Cp", "Mach", "Cp", "Total pressure", "Convergence", "Surface mesh", "History"]
        assert not hasattr(mon, "picture_views")                               # the pictures live in one place

        # ---- a finished run of two conditions ------------------------------------------------
        control = set_namelist_value(TEMPLATE, "writeToFileInterval", 2)
        res = _case(out, 0).run(PreProOptions(), control, [{"Mach": 0.5, "AoA": 1.0}, {"Mach": 0.7, "AoA": 2.0}],
                                boundary_flags=[1, 3])
        mon.refresh()
        assert (mon._n, mon._nc) == (0, 2) and mon._designs == [0]             # the run started last
        assert [mon.cond_combo.itemText(i) for i in range(mon.cond_combo.count())] == ["1", "2"]
        assert mon._follower.rows == 5 and "finished" in mon.status.text() and "cycle 5" in mon.status.text()
        assert "Mach 0.7, alpha 2 deg" in mon.status.text() and "baseline" in mon.status.text()
        assert len(mon.conv_fig.axes) == 4 and len(mon.geom_fig.axes) == 1
        assert mon.geom_fig.texts[0].get_text() == "Geometry - design n = 0 (baseline)"
        # the clock of a run that has ended is the solver's own run time
        assert mon.timer_label.text() == "Solver run time  " + fm.format_duration(res[1]["wall_time_s"])
        mon.pictures_btn.click()
        assert win.opened == [(0, 2)]                                          # 'Pictures' opens the other view here

        post.refresh()
        assert (post._n, post._nc) == (0, 2)                                   # the run that ended last
        text = post.status.text()
        assert "finished" in text and "cycle 5" in text and "lift" in text and "solver run time" in text
        assert "Mach 0.7, alpha 2 deg" in text and "not of a finished run" not in text
        for title, name in (("Surface Cp", "surface_cp.png"), ("Mach", "mach.png"), ("Cp", "cp.png"),
                            ("Total pressure", "total_pressure.png"), ("Convergence", "convergence.png"),
                            ("Surface mesh", "naca_0.png")):
            view = _tab(post, title)                                           # loaded when the tab is opened
            assert view.path.endswith(name) and os.sep + "cond_2" + os.sep in view.path or name == "naca_0.png", \
                (title, view.path)
            assert post.picture_path(title) == view.path
        assert _tab(post, "History").note() == NO_HISTORY
        os.makedirs(os.path.join(out, "opt"))
        shutil.copy(os.path.join(out, "surfaces", "n_0", "naca_0.png"), os.path.join(out, "opt", "history.png"))
        post.refresh()
        assert post.picture_path("History") == os.path.join(out, "opt", "history.png")
        _tab(post, "Mach")

        # ---- a second design: both views move to it -------------------------------------------
        time.sleep(0.05)
        _case(out, 1, scale_y=1.3).run(PreProOptions(), control, [{"Mach": 0.5, "AoA": 1.0}], boundary_flags=[1, 3])
        for view in (mon, post):
            view.refresh(rescan=True)
            assert (view._n, view._nc) == (1, 1) and view._designs == [0, 1]
            assert view.design_combo.currentText() == "n = 1"
        sub = mon.geom_fig.texts[1].get_text()
        assert "largest wall displacement from the baseline" in sub and len(mon.geom_fig.axes[0].lines) == 2
        assert post.picture_path("Mach").endswith(os.path.join("n_1", "cond_1", "mach.png"))

        # ---- by hand: the check box goes off and the view stays on the choice ------------------
        for view in (mon, post):
            view._on_design_picked(0)
            assert not view.follow_cb.isChecked() and (view._n, view._nc) == (0, 2)
            view._on_cond_picked(view.cond_combo.findData(1))
            assert (view._n, view._nc) == (0, 1) and "Mach 0.5, alpha 1 deg" in view.status.text()
            view.refresh()
            assert (view._n, view._nc) == (0, 1)                               # not pulled back to n = 1
        assert post.picture_path("Mach").endswith(os.path.join("n_0", "cond_1", "mach.png"))
        post.select(1)                                                         # what 'Pictures' does
        assert (post._n, post._nc) == (1, 1) and not post.follow_cb.isChecked()

        # ---- a run in progress ------------------------------------------------------------------
        stop, result = {"now": False}, {}
        case0 = _case(out, 0)
        no_field = set_namelist_value(TEMPLATE, "writeToFileInterval", 1000000)   # stopped before any .res

        def run():
            result["res"] = case0.run(PreProOptions(), no_field, [{"Mach": 8.88}], boundary_flags=[1, 3],
                                      should_stop=lambda: stop["now"])
        thread = threading.Thread(target=run)
        win.sim2d_thread = thread                                              # what the main window holds
        thread.start()
        try:
            mon.run_started()
            post.run_started()
            assert mon.follow_cb.isChecked() and post.follow_cb.isChecked()
            seen, t0 = [], time.time()
            while time.time() - t0 < 40.0 and len(seen) < 3:
                _pump(app, 0.3)
                mon.refresh()
                if (mon._n, mon._nc) == (0, 1) and mon._follower.rows and \
                        (not seen or mon._follower.rows > seen[-1]):
                    seen.append(mon._follower.rows)
                    assert "solver running" in mon.status.text(), mon.status.text()
            assert len(seen) == 3, seen
            # the clock counts from the moment the solver was started
            started = fm.run_state(out, 0, 1).started
            mon._clock = lambda: started + 75.0
            mon.refresh()
            assert mon.timer_label.text() == "Solver running  01:15"
            win.flite2d_time_limit_min = 10
            mon.refresh()
            assert mon.timer_label.text() == "Solver running  01:15  /  10:00 limit"
            mon._clock = time.time
            # the Post-Processing view stays on the last run that ENDED, with its pictures ...
            post.refresh()
            assert (post._n, post._nc) == (1, 1) and post.picture_path("Mach").endswith(
                os.path.join("n_1", "cond_1", "mach.png"))
            # ... and if the running design is chosen by hand, its old pictures are not shown as new
            post.select(0, 1)
            assert post.picture_path("Mach") is None and "earlier run" in _tab(post, "Mach").note()
            post.follow_cb.setChecked(True)
        finally:
            stop["now"] = True
            thread.join(40.0)
        win.sim2d_thread = None
        mon.run_finished()
        post.run_finished()
        assert result["res"][0]["status"] == "stopped"
        assert "ended: stopped" in mon.status.text() and mon._follower.rows == result["res"][0]["rows"]
        assert mon.timer_label.text() == "Solver run time  " + fm.format_duration(result["res"][0]["wall_time_s"])
        assert (post._n, post._nc) == (0, 1)                                   # it has just ended
        assert "ended: stopped" in post.status.text() and "not of a finished run" in post.status.text()
        note = _tab(post, "Cp").note()                                         # never written by this run
        assert "This run ended (stopped) without this picture" in note and "earlier run" in note
        assert _tab(post, "Convergence").path.endswith("convergence.png")      # this one it did write
        assert not [m for m in win.messages if "MONITOR" in m], win.messages

        # ---- the preprocessor of a new design: the clock runs from its start ----------------------
        time.sleep(0.05)
        _case(out, 2, scale_y=0.8)
        pre = os.path.join(out, "preprocessed", "n_2")
        os.makedirs(pre)
        open(os.path.join(pre, "runPrePro.inp"), "w").close()
        win.sim2d_thread = object()                                            # "a run is active"
        t_pre = fm.prepro_started(out, 2)
        mon._clock = lambda: t_pre + 7.0
        mon.refresh(rescan=True)
        assert (mon._n, mon._nc) == (2, None) and "preprocessor running" in mon.status.text()
        assert mon.timer_label.text() == "Preprocessor running  00:07"
        post.refresh(rescan=True)
        assert (post._n, post._nc) == (0, 1)                                   # nothing new has ended
        win.sim2d_thread = None
        mon._clock = time.time
        mon.refresh()
        assert mon.timer_label.text() == ""                                    # not run: no clock

        # ---- another output directory: nothing of the old one is left --------------------------
        win.output_directory = os.path.join(tmp, "empty")
        os.makedirs(win.output_directory)
        for view in (mon, post):
            view.refresh()
            assert view.design_combo.count() == 0 and "No design yet" in view.status.text()
            view.hide()
            assert not view._timer.isActive()
        assert mon._follower.rows == 0 and mon.timer_label.text() == "" and post.picture_path("Mach") is None
        mon.close()
        post.close()


def test_post_view_lists_the_designs_of_the_generation_chosen():
    if QApplication is None:
        try:
            import pytest
            pytest.skip("PyQt5 is not installed")
        except ImportError:
            return
    from GUI.post_view_2d import Post2DViewer
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out")
        control = set_namelist_value(TEMPLATE, "writeToFileInterval", 2)
        for n in range(5):                                                     # five designs, run in this order
            _case(out, n, scale_y=1.0 + 0.1 * n).run(PreProOptions(), control, [{"Mach": 0.5, "AoA": 1.0}],
                                                     boundary_flags=[1, 3])
            time.sleep(0.05)
        win = _Window(out)
        post = Post2DViewer(win)
        post.resize(900, 700)
        post.show()
        _pump(app)
        gens = lambda: [post.gen_combo.itemText(i) for i in range(post.gen_combo.count())]   # noqa: E731

        # no optimisation table: every design is listed, there is no generation to choose
        assert gens() == ["All"] and not post.gen_combo.isEnabled()
        assert post._designs == [0, 1, 2, 3, 4] and (post._n, post._nc) == (4, 1)

        # the optimisation's table: generation 0 = baseline + two samples, then one design per step;
        # design 4 is on disk but not in the table (left by an earlier run)
        rows = [{"n": 0, "gen": 0, "done": True, "status": "ok", "objective": 1.5, "baseline": True},
                {"n": 1, "gen": 0, "done": True, "status": "diverged", "objective": None},
                {"n": 2, "gen": 0, "done": True, "status": "ok", "objective": 1.25},
                {"n": 3, "gen": 1, "done": True, "status": "ok", "objective": 1.125}]
        os.makedirs(os.path.join(out, "opt"))
        with open(os.path.join(out, "opt", "designs.json"), "w") as fh:
            json.dump({"designs": rows}, fh)
        post.refresh(rescan=True)
        assert gens() == ["All", "0   (3 designs)", "1   (1 design)"] and post.gen_combo.isEnabled()
        assert post.gen_combo.currentText() == "All" and post._designs == [0, 1, 2, 3, 4]
        assert post.follow_cb.isChecked() and post._n == 4                    # still on the run that ended last

        # choose generation 0: its three designs, the last of them on show, following off
        post._on_generation_picked(1)
        assert post._designs == [0, 1, 2] and (post._n, post._nc) == (2, 1) and not post.follow_cb.isChecked()
        assert [post.design_combo.itemText(i) for i in range(3)] == [       # no need to repeat the generation
            "n = 0 (baseline)    objective 1.5", "n = 1    [diverged]", "n = 2    objective 1.25"]
        assert "objective 1.25" in post.status.text()
        assert _tab(post, "Mach").path.endswith(os.path.join("n_2", "cond_1", "mach.png"))
        post._on_design_picked(1)                                              # a design within the generation
        assert (post._n, post._nc) == (1, 1) and post.gen_combo.currentText().startswith("0 ")
        assert post.picture_path("Mach").endswith(os.path.join("n_1", "cond_1", "mach.png"))
        post.refresh()
        post.refresh(rescan=True)
        assert post._designs == [0, 1, 2] and post._n == 1                    # it stays there

        # generation 1 has one design: it is shown straight away
        post._on_generation_picked(2)
        assert post._designs == [3] and (post._n, post._nc) == (3, 1)
        assert post.picture_path("Mach").endswith(os.path.join("n_3", "cond_1", "mach.png"))

        # a design outside the chosen generation is asked for (the Solver view's 'Pictures'): back to All
        post.select(4, 1)
        assert post.gen_combo.currentText() == "All" and post._designs == [0, 1, 2, 3, 4] and post._n == 4
        assert [post.design_combo.itemText(i) for i in (2, 3, 4)] == [         # under All it is part of the entry
            "n = 2    generation 0    objective 1.25", "n = 3    generation 1    objective 1.125", "n = 4"]
        post.select(0)
        post._on_generation_picked(2)
        assert post._n == 3
        post.select(3, 1)                                                      # inside it: the choice is kept
        assert post.gen_combo.currentText().startswith("1 ") and post._designs == [3]

        # "show the latest finished run" goes to the generation of that run
        post._on_generation_picked(1)
        time.sleep(0.05)
        _case(out, 2, scale_y=1.2).run(PreProOptions(), control, [{"Mach": 0.5, "AoA": 1.0}], boundary_flags=[1, 3])
        post.follow_cb.setChecked(True)
        post.refresh()
        assert post._n == 2 and post.gen_combo.currentText().startswith("0 ")
        post._on_generation_picked(2)                                          # by hand again: generation 1
        post.follow_cb.setChecked(True)                                        # ... and back to following
        assert post._n == 2 and post.gen_combo.currentText().startswith("0 ") and post._designs == [0, 1, 2]
        time.sleep(0.05)
        _case(out, 4, scale_y=1.4).run(PreProOptions(), control, [{"Mach": 0.5, "AoA": 1.0}], boundary_flags=[1, 3])
        post.refresh()                                                         # latest run is not in the table
        assert post._n == 4 and post.gen_combo.currentText() == "All" and post._designs == [0, 1, 2, 3, 4]

        # a new optimisation with fewer generations: a choice that no longer exists falls back to All
        post._on_generation_picked(2)
        assert post._designs == [3]
        with open(os.path.join(out, "opt", "designs.json"), "w") as fh:
            json.dump({"designs": rows[:3]}, fh)
        time.sleep(0.05)
        os.utime(os.path.join(out, "opt", "designs.json"))
        post.refresh(rescan=True)
        assert gens() == ["All", "0   (3 designs)"] and post.gen_combo.currentText() == "All"
        assert post._designs == [0, 1, 2, 3, 4] and post._n == 3              # the design on show is kept
        assert not [m for m in win.messages if "MONITOR" in m], win.messages
        post.close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
