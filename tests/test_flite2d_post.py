"""
Tests of the FLITE 2D pictures and tables (Local/flite2d_post.py).

    python -m pytest tests/test_flite2d_post.py -q      or      python tests/test_flite2d_post.py

These check the numbers behind the pictures (formulas, wall ordering, file
reading) and that every picture is written; what a picture looks like has to
be judged by eye.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from test_edge_modes_2d import naca0012, o_mesh                                   # noqa: E402
from FileRW.FliteMesh2D import default_edge_flags, read_flite2d_mesh, write_flite2d_mesh   # noqa: E402
from Local.flite2d_post import (body_window, critical_cp, flow_fields, plot_field, plot_surface_mesh,   # noqa: E402
                                read_res, read_rsd_history, save_mesh_image, surface_table, wall_chains)


def _mesh(d):
    m, _ = o_mesh(naca0012(40), n_layers=10)
    path = os.path.join(d, "m.dat")
    write_flite2d_mesh(m, path, default_edge_flags(m))
    return path, read_flite2d_mesh(path)


def test_free_stream_gives_mach_inf_zero_cp_and_unit_total_pressure():
    for mach in (0.3, 0.8, 2.0):
        p_inf = 1.0 / (1.4 * mach ** 2)
        R = np.array([[1.0, np.cos(0.1), np.sin(0.1), 0.0, p_inf]])            # rho, u, v, E, p
        F = flow_fields(R, mach)
        assert abs(F["mach"][0] - mach) < 1e-12 and abs(F["cp"][0]) < 1e-12 and abs(F["p0_ratio"][0] - 1) < 1e-12
    # stagnation point reached isentropically: Mach 0, total pressure unchanged, Cp = Cp0
    mach = 0.5
    p0 = (1.0 / (1.4 * mach ** 2)) * (1 + 0.2 * mach ** 2) ** 3.5
    F = flow_fields(np.array([[1.1, 0.0, 0.0, 0.0, p0]]), mach)
    assert F["mach"][0] == 0.0 and abs(F["p0_ratio"][0] - 1) < 1e-12
    assert abs(F["cp"][0] - 1.0641) < 1e-3                                       # 1 + M^2/4 + M^4/40 ...
    # critical Cp: textbook value -0.4346 at Mach 0.8; none for supersonic free stream
    assert abs(critical_cp(0.8) - (-0.4346)) < 1e-4 and critical_cp(1.2) is None
    # ... and it is the Cp of a sonic point with free-stream total pressure
    p_star = (1.0 / (1.4 * 0.64)) * ((1 + 0.2 * 0.64) / 1.2) ** 3.5
    assert abs(2 * (p_star - 1.0 / (1.4 * 0.64)) - critical_cp(0.8)) < 1e-12


def test_wall_chains_follow_the_file_and_close():
    with tempfile.TemporaryDirectory() as d:
        _, m = _mesh(d)
        B = m["boundary"]
        chains = wall_chains(B)
        assert len(chains) == 1 and chains[0][1]                                # one closed loop
        nodes = chains[0][0]
        wall = B[B[:, 2] == 1]
        assert len(nodes) == len(wall) == len(set(nodes.tolist()))
        nxt = dict(zip(wall[:, 0].tolist(), wall[:, 1].tolist()))
        assert all(nxt[int(a)] == int(b) for a, b in zip(nodes, np.roll(nodes, -1)))   # walking order of the file
        # an open wall (three edges) and a separate loop given out of order
        open_b = np.array([[5, 6, 1], [6, 7, 1], [7, 8, 1], [20, 21, 3], [11, 12, 5], [10, 11, 5], [12, 10, 5]])
        got = sorted((c.tolist(), closed) for c, closed in wall_chains(open_b))
        assert got[0] == ([5, 6, 7, 8], False) and got[1][1] and sorted(got[1][0]) == [10, 11, 12]
        assert wall_chains(np.array([[1, 2, 3]])) == []                          # no wall at all
        # the picture frame: the walls plus half their size on every side
        x0, x1, y0, y1 = body_window(m["nodes"], B)
        P = m["nodes"][nodes]
        size = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
        assert abs(x0 - (P[:, 0].min() - 0.5 * size)) < 1e-9 and abs(y1 - (P[:, 1].max() + 0.5 * size)) < 1e-9


def test_res_reader_ignores_the_node_column_and_refuses_a_cut_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.res")
        rows = ["99999    0.1008E+01    0.9876E+00    0.2543E-01    0.3289E+01    0.1129E+01",
                "*****    0.1009E+01   -0.9877E+00    0.2541E-01    0.3289E+01    0.1130E+01",
                "*****    0.1000E+01    0.9995E+00    0.2147E-01    0.3292E+01    0.1117E+01"]
        with open(p, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        R = read_res(p, n_nodes=3)
        assert R.shape == (3, 5) and R[1, 0] == 1.009 and R[1, 1] == -0.9877 and R[2, 4] == 1.117
        with open(p, "w") as fh:
            fh.write("\n".join(rows[:2]) + "\n*****    0.1000E+01    0.99")     # killed while writing
        try:
            read_res(p, n_nodes=3)
            raise AssertionError("a cut result file was accepted")
        except ValueError as e:
            assert "2 rows" in str(e)
        h = os.path.join(d, "a.rsd")
        with open(h, "w") as fh:
            fh.write("      1   0.0   0.1   0.2   0.3   0.1   0.2   0.0\n      2  -0.5   NaN   0.2   0.3\n   3  -0.")
        H = read_rsd_history(h)
        assert H.shape == (2, 8) and np.isnan(H[1, 2]) and np.isnan(H[1, 7]) and H[1, 1] == -0.5
        assert read_rsd_history(os.path.join(d, "missing.rsd")).shape == (0, 8)


def test_surface_table_and_pictures():
    with tempfile.TemporaryDirectory() as d:
        dat, m = _mesh(d)
        X, T, B = m["nodes"], m["triangles"], m["boundary"]
        F = flow_fields(np.column_stack([np.ones(len(X)), 1 + 0.1 * X[:, 1], 0.0 * X[:, 0], np.zeros(len(X)),
                                         2.857 + 0.2 * X[:, 0]]), 0.5)
        tab = surface_table(X, B, F)
        assert len(tab) == 1 and tab[0]["closed"] and tab[0]["nodes"][0] == tab[0]["nodes"][-1]
        assert np.all(np.diff(tab[0]["s"]) > 0)                                  # arc length increases
        assert np.allclose(tab[0]["cp"], 2 * (2.857 + 0.2 * tab[0]["x"] - 1 / (1.4 * 0.25)))
        # pictures: a normal field, a uniform one (degenerate colour scale) and one with holes
        for name, v, kind in (("a.png", F["mach"], "sequential"), ("b.png", np.full(len(X), 0.5), "sequential"),
                              ("c.png", np.zeros(len(X)), "diverging")):
            out = plot_field(X, T, B, v, os.path.join(d, name), "q", "title", kind=kind, line_at=1.0)
            assert out and os.path.getsize(out) > 1000
        holes = F["mach"].copy()
        holes[::7] = np.nan
        assert plot_field(X, T, B, holes, os.path.join(d, "d.png"), "q", "t")
        assert plot_field(X, T, B, np.full(len(X), np.nan), os.path.join(d, "e.png"), "q", "t") is None
        # surface mesh: baseline, and a deformed design drawn over the baseline
        png = save_mesh_image(dat, os.path.join(d, "surfaces", "n_0", "m_0.png"))
        assert os.path.getsize(png) > 1000
        X2 = X.copy()
        X2[:, 1] *= 1.2
        png2 = plot_surface_mesh(X2, T, B, os.path.join(d, "m_1.png"), baseline_X=X, baseline_boundary=B)
        assert os.path.getsize(png2) > 1000


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and fn.__module__ == "__main__":
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:                                  # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
