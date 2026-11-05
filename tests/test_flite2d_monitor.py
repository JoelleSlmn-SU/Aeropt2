"""
Tests of the read-only run monitor (Local/flite2d_monitor.py) and of the two
drawings the Solver view of the GUI shares with the pictures on disk.

    python -m pytest tests/test_flite2d_monitor.py -q      or      python tests/test_flite2d_monitor.py

The runs are made by the real runner (Local/flite2d_local.py) with the stand-in
programs of tests/standins, so what is checked is that the monitor reads the
folders the way the runner writes them.
"""
import os
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from test_edge_modes_2d import naca0012, o_mesh                                   # noqa: E402
from FileRW.FliteMesh2D import default_edge_flags, read_flite2d_mesh, write_flite2d_mesh   # noqa: E402
from Local import flite2d_monitor as fm                                           # noqa: E402
from Local.flite2d_local import Flite2DCase, PreProOptions, set_namelist_value    # noqa: E402
from Local.flite2d_post import (draw_convergence, draw_wall_outline, read_rsd_history,   # noqa: E402
                                wall_chains)

PREPRO = [sys.executable, os.path.join(HERE, "standins", "fake_prepro.py")]
SOLVER = [sys.executable, os.path.join(HERE, "standins", "fake_solver.py")]
with open(os.path.join(os.path.dirname(HERE), "templates", "flite2d_solver.inp")) as _fh:
    TEMPLATE = _fh.read()


def _case(out, n=0, scale_y=1.0, logs=None):
    m, _ = o_mesh(naca0012(40), n_layers=10)
    if scale_y != 1.0:
        m.nodes[:, 1] *= scale_y
    case = Flite2DCase(out, "naca", n=n, log=(logs.append if logs is not None else (lambda s: None)),
                       prepro_cmd=PREPRO, solver_cmd=SOLVER)
    write_flite2d_mesh(m, case.mesh_path, default_edge_flags(m))
    return case


def _figure(w, h):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=100)
    FigureCanvasAgg(fig)
    return fig


# ---------------------------------------------------------------------------
# the residual file while it is being written
# ---------------------------------------------------------------------------
def test_follower_reads_only_whole_new_rows():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.rsd")
        f = fm.RsdFollower(path)
        assert not f.poll() and f.rows == 0                         # no file yet: nothing, no error
        row = "{:7d} {:13.5E} {:13.5E} {:13.5E} {:13.5E} {:13.5E} {:13.5E} {:13.5E}\n"
        with open(path, "w") as fh:
            fh.write(row.format(1, 0.0, 0.1, 0.01, 0.5, 0.1, 0.01, 0.0))
            fh.write(row.format(2, -0.5, 0.2, 0.02, 0.6, 0.2, 0.02, 0.0)[:30])     # half a line, no newline
            fh.flush()
            assert f.poll() and f.rows == 1                         # the half line is NOT a row
            assert not f.poll()                                     # nothing new: no change reported
            fh.write(row.format(2, -0.5, 0.2, 0.02, 0.6, 0.2, 0.02, 0.0)[30:])
            fh.write("   3  -0.90000E+00   0.30000E+00   0.30000E-01\n")            # 4 columns only
            fh.flush()
            assert f.poll() and f.rows == 3
        H = f.history
        assert H.shape == (3, 8) and H[1].tolist() == [2, -0.5, 0.2, 0.02, 0.6, 0.2, 0.02, 0.0]
        assert H[2, :4].tolist() == [3, -0.9, 0.3, 0.03] and np.isnan(H[2, 4:]).all()
        assert np.array_equal(H, read_rsd_history(path), equal_nan=True)            # same as the full reader


def test_follower_starts_again_when_the_solver_does():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.rsd")
        f = fm.RsdFollower(path)
        with open(path, "w") as fh:
            for i in range(1, 51):
                fh.write(f"{i:7d} {-0.1 * i:13.5E} {0.5:13.5E} {0.01:13.5E} {0.0:13.5E}\n")
        assert f.poll() and f.rows == 50
        # new run, shorter file
        with open(path, "w") as fh:
            fh.write(f"{1:7d} {0.0:13.5E} {0.7:13.5E} {0.02:13.5E} {0.0:13.5E}\n")
        assert f.poll() and f.rows == 1 and f.history[0, 2] == 0.7
        # new run whose file is already LONGER than what was read (size alone would not show it):
        # the iteration number going back is what gives it away
        with open(path, "w") as fh:
            for i in range(1, 4):
                fh.write(f"{i:7d} {0.0:13.5E} {0.9:13.5E} {0.03:13.5E} {0.0:13.5E} {0.0:13.5E} {0.0:13.5E}\n")
        assert f.poll() and f.rows == 3 and f.history[:, 0].tolist() == [1, 2, 3] and (f.history[:, 2] == 0.9).all()
        os.remove(path)
        assert f.poll() and f.rows == 0                             # file gone: history gone
        assert not f.poll()
        f.set_path(None)
        assert not f.poll() and f.history.shape == (0, 8)
        # NaN rows (a diverged run) are rows too
        with open(path, "w") as fh:
            fh.write("      1           NaN           NaN           NaN           NaN\n")
        f.set_path(path)
        assert f.poll() and f.rows == 1 and np.isnan(f.history[0, 1])


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_wall_outline_is_the_wall_of_the_full_reader():
    with tempfile.TemporaryDirectory() as d:
        base = _case(d, n=0)
        thick = _case(d, n=1, scale_y=1.5)
        full = read_flite2d_mesh(base.mesh_path)
        chains = wall_chains(full["boundary"])
        walls = fm.read_wall_outline(base.mesh_path)
        assert len(walls) == len(chains) == 1 and walls[0][1] is True
        assert np.array_equal(walls[0][0], full["nodes"][chains[0][0]])             # same points, same order
        walls1 = fm.read_wall_outline(thick.mesh_path)
        moved = fm.max_wall_displacement(walls1, walls)
        assert abs(moved - 0.5 * np.abs(walls[0][0][:, 1]).max()) < 1e-12           # y scaled by 1.5
        assert fm.max_wall_displacement(walls, walls) == 0.0
        assert fm.max_wall_displacement(walls, [(walls[0][0][:-1], True)]) is None  # not the same wall
        assert fm.max_wall_displacement(walls, None) is None
        assert fm.mesh_path(d, 1) == thick.mesh_path and fm.mesh_path(d, 7) is None
        try:
            fm.read_wall_outline(os.path.join(os.path.dirname(HERE), "templates", "flite2d_solver.inp"))
            raise AssertionError("a control file was accepted as a mesh")
        except ValueError:
            pass
        # the drawing: both walls inside the axes, equal scales
        fig = _figure(7.0, 2.2)
        ax = draw_wall_outline(fig, walls1, walls, title="t", subtitle="s")
        fig.canvas.draw()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        P = np.vstack([walls1[0][0], walls[0][0]])
        assert x0 < P[:, 0].min() and x1 > P[:, 0].max() and y0 < P[:, 1].min() and y1 > P[:, 1].max()
        box = ax.get_window_extent()
        assert abs((x1 - x0) / box.width - (y1 - y0) / box.height) < 1e-6 * (x1 - x0)
        draw_wall_outline(_figure(4, 2), [], None)                                  # nothing to draw: no error


# ---------------------------------------------------------------------------
# what the folders say about a run
# ---------------------------------------------------------------------------
def test_states_pictures_and_latest_run_after_a_run():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out put dir")                      # blank in the path on purpose
        assert fm.design_numbers(out) == [] and fm.latest_run(out) is None and fm.latest_finished_run(out) is None
        case = _case(out, n=0)
        assert fm.design_numbers(out) == [0] and fm.condition_numbers(out, 0) == []
        assert fm.latest_run(out) == (0, None)                      # a mesh, nothing run yet
        assert not fm.run_state(out, 0, 1).exists
        pics = fm.run_pictures(out, 0, None)
        assert [t for t, _, _ in pics] == ["Surface Cp", "Mach", "Cp", "Total pressure", "Convergence",
                                           "Surface mesh"]
        assert all(p is None and note for _, p, note in pics)
        assert fm.latest_finished_run(out) is None and fm.prepro_started(out, 0) is None
        assert fm.run_state(out, 0, 1).run_time(now=1e9) is None    # never started: no clock

        control = set_namelist_value(TEMPLATE, "writeToFileInterval", 2)
        res = case.run(PreProOptions(), control, [{"Mach": 0.5, "AoA": 1.0}, {"Mach": 0.7, "AoA": 2.0}],
                       boundary_flags=[1, 3])
        assert [r["status"] for r in res] == ["ok", "ok"]
        assert fm.condition_numbers(out, 0) == [1, 2]
        assert fm.latest_run(out) == (0, 2)                         # the condition started last
        assert fm.latest_finished_run(out) == (0, 2)                # ... and the one that ended last
        st = fm.run_state(out, 0, 2)
        assert st.exists and st.finished and st.status == "ok"
        assert st.ended >= st.started >= fm.prepro_started(out, 0)
        assert st.run_time() == st.run_time(now=1e12) == res[1]["wall_time_s"]   # ended: the runner's own figure
        assert (st.mach, st.alpha, st.reynolds, st.target_orders) == (0.7, 2.0, 0.0, 3.0)
        assert st.describe_condition() == "Mach 0.7, alpha 2 deg"
        assert st.rsd == os.path.join(fm.cond_dir(out, 0, 2), "naca_0.rsd") and os.path.isfile(st.rsd)
        f = fm.RsdFollower(st.rsd)
        assert f.poll() and f.rows == res[1]["rows"] == 5 and f.history[-1, 2] == res[1]["CL"]
        pics = fm.run_pictures(out, 0, 2)
        assert all(p and os.path.isfile(p) and not note for _, p, note in pics)
        assert pics[-1][1] == case.mesh_image_path == fm.mesh_image_path(out, 0)

        # a second design: the one whose preprocessor started last is the current one
        time.sleep(0.05)
        case1 = _case(out, n=1, scale_y=1.2)
        assert fm.design_numbers(out) == [0, 1] and fm.latest_run(out) == (1, None)
        assert fm.latest_finished_run(out) == (0, 2)                # nothing of design 1 has ended yet
        case1.run(PreProOptions(), control, [{"Mach": 0.5}], boundary_flags=[1, 3])
        assert fm.latest_run(out) == (1, 1) and fm.latest_finished_run(out) == (1, 1)

        # design 0, condition 1 is started again: until that run has ended its old pictures
        # and its old metrics are not shown as the result of the new run
        d1 = fm.cond_dir(out, 0, 1)
        later = time.time() + 30.0
        os.utime(os.path.join(d1, "runSolver.inp"), (later, later))
        st = fm.run_state(out, 0, 1)
        assert st.started == later and not st.finished and st.status is None
        assert st.run_time(now=later + 12.5) == 12.5 and st.run_time() is None   # running: time since the start
        assert st.run_time(now=later - 3.0) == 0.0                  # never negative
        assert fm.latest_finished_run(out) == (1, 1)                # the old metrics of (0, 1) do not count
        pics = fm.run_pictures(out, 0, 1)
        assert all(p is None and "earlier run" in note for _, p, note in pics[:-1])
        assert pics[-1][1]                                          # the mesh picture belongs to the design
        assert fm.latest_run(out) == (0, 1)
        # ... and if the preprocessor of design 0 is started after that, no condition is current
        os.utime(os.path.join(out, "preprocessed", "n_0", "runPrePro.inp"), (later + 5, later + 5))
        assert fm.latest_run(out) == (0, None)
        assert fm.prepro_started(out, 0) == later + 5
        assert fm.opt_designs(out) == {} and fm.opt_history_path(out) is None


def test_durations_are_written_as_a_clock():
    f = fm.format_duration
    assert [f(v) for v in (0, 0.4, 59.4, 59.6, 75, 600, 3599.4)] == ["00:00", "00:00", "00:59", "01:00", "01:15",
                                                                    "10:00", "59:59"]
    assert [f(v) for v in (3599.6, 3725, 36000)] == ["1:00:00", "1:02:05", "10:00:00"]
    assert f(None) == "" and f(float("nan")) == "" and f(-5) == "00:00"


def test_following_a_solver_that_is_running():
    """The stand-in at Mach 8.88 writes a row every 0.05 s and never ends: the monitor sees the
    rows arrive, reports the run as not finished, and sees the status once it has been stopped."""
    with tempfile.TemporaryDirectory() as tmp:
        case = _case(tmp, n=0)
        stop = {"now": False}
        result = {}

        def run():
            result["res"] = case.run(PreProOptions(), TEMPLATE, [{"Mach": 8.88}], boundary_flags=[1, 3],
                                     should_stop=lambda: stop["now"])
        t = threading.Thread(target=run)
        t.start()
        try:
            f, seen, t0 = fm.RsdFollower(), [], time.time()
            while time.time() - t0 < 30.0 and (len(seen) < 3 or seen[-1] < 20):
                time.sleep(0.2)
                if fm.latest_run(tmp) != (0, 1):
                    continue                                        # still in the preprocessor
                st = fm.run_state(tmp, 0, 1)
                if f.path != st.rsd:
                    f.set_path(st.rsd)
                if f.poll():
                    seen.append(f.rows)
                    assert not st.finished and st.status is None    # running: no metrics yet
                    assert st.mach == 8.88 and st.target_orders == 3.0
            assert len(seen) >= 3 and seen == sorted(set(seen)), seen     # it grew, poll after poll
            H = f.history
            assert H[:, 0].tolist() == list(range(1, len(H) + 1))        # no row lost or read twice
        finally:
            stop["now"] = True
            t.join(30.0)
        assert not t.is_alive() and result["res"][0]["status"] == "stopped"
        st = fm.run_state(tmp, 0, 1)
        assert st.finished and st.status == "stopped"
        f.poll()
        assert f.rows == result["res"][0]["rows"]                   # the follower ended on the last row


# ---------------------------------------------------------------------------
# the convergence drawing on a figure of another size
# ---------------------------------------------------------------------------
def test_convergence_drawing_on_any_figure_and_without_data():
    it = np.arange(1, 201, dtype=float)
    H = np.column_stack([it, -it / 50.0, 0.3 + 0.2 * np.exp(-it / 20.0), 0.01 + 0.0 * it, 0.0 * it,
                         0.0 * it, 0.0 * it, 0.0 * it])
    for size in ((10.0, 6.4), (6.0, 3.0), (3.0, 1.8)):              # the picture, a panel, a very small panel
        fig = _figure(*size)
        axes = draw_convergence(fig, H, title="Convergence - t", subtitle="Mach 0.5", target_orders=3)
        fig.canvas.draw()
        assert axes.shape == (2, 2)
        top = max(ax.get_position().y1 for ax in axes.ravel())
        assert 0.5 < top < 1.0 and all(ax.get_position().height > 0.05 for ax in axes.ravel())
        assert axes[0, 0].get_title(loc="left").endswith("-4")       # last residual in the panel title
        lo, hi = axes[0, 1].get_ylim()
        assert lo < 0.3 < hi < 0.45                                  # lift panel scaled to the last 80 %
    fig = _figure(10.0, 6.4)
    assert abs(max(ax.get_position().y1 for ax in draw_convergence(fig, H).ravel()) - 0.86) < 1e-9
    for empty in (np.zeros((0, 8)), [], None):
        axes = draw_convergence(_figure(6.0, 3.0), empty, title="t")
        assert axes[0, 1].get_title(loc="left") == "lift" and not axes[0, 1].lines[0].get_xdata().size
    draw_convergence(fig, H[:1])                                     # one row: no error


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
