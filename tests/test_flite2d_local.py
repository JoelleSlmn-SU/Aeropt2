"""
Tests of the local FLITE 2D runner (Local/flite2d_local.py).

    python -m pytest tests/test_flite2d_local.py -q      or      python tests/test_flite2d_local.py

PrePro.exe and Solver.exe are replaced by the stand-in programs in
tests/standins, which ask the same questions in the same order. These tests
therefore check the ORCHESTRATION (folders, piped answers, relative paths,
control-file editing, time limit, failure handling), not the real solver.
"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from test_edge_modes_2d import naca0012, o_mesh                                  # noqa: E402
from FileRW.FliteMesh2D import default_edge_flags, write_flite2d_mesh            # noqa: E402
from Local.flite2d_local import (Flite2DCase, PreProOptions, RunStopped, apply_condition,    # noqa: E402
                                 check_control, get_namelist_value, is_control_file, read_rsd,
                                 set_namelist_value)

PREPRO = [sys.executable, os.path.join(HERE, "standins", "fake_prepro.py")]
SOLVER = [sys.executable, os.path.join(HERE, "standins", "fake_solver.py")]
with open(os.path.join(os.path.dirname(HERE), "templates", "flite2d_solver.inp")) as _fh:
    TEMPLATE = _fh.read()


PICTURES = ["convergence.png", "surface_cp.png", "naca_0_surface.csv", "mach.png", "cp.png",
            "total_pressure.png"]


def _case(d, logs, n=0):
    m, _ = o_mesh(naca0012(40), n_layers=10)
    case = Flite2DCase(os.path.join(d, "out put dir"), "naca", n=n, log=logs.append,   # blank in the path on purpose
                       prepro_cmd=PREPRO, solver_cmd=SOLVER)
    write_flite2d_mesh(m, case.mesh_path, default_edge_flags(m))
    return case


def test_template_is_a_control_file_and_namelist_editing():
    assert is_control_file(TEMPLATE) and not is_control_file("TITLE\n 3D solver input\n")
    assert get_namelist_value(TEMPLATE, "MachNumber") == "0.5"
    assert get_namelist_value(TEMPLATE, "numberOfMGIterations") == "-3"
    t = set_namelist_value(TEMPLATE, "MachNumber", 0.75)
    assert get_namelist_value(t, "MachNumber") == "0.75"
    assert t.count("ivd%MachNumber") == 1 and len(t.splitlines()) == len(TEMPLATE.splitlines())
    assert get_namelist_value(set_namelist_value(t, "useMatrixDissipation", True), "useMatrixDissipation") == ".true."
    t2 = set_namelist_value(TEMPLATE, "engineBCRelaxation", 0.5)          # not in the template: added before '/'
    assert get_namelist_value(t2, "engineBCRelaxation") == "0.5" and t2.rstrip().endswith("/")
    assert get_namelist_value(set_namelist_value(TEMPLATE, "tripNodes(1)", 17), "tripNodes(1)") == "17"
    assert get_namelist_value(TEMPLATE, "tripNodes(2)") == "252"          # the other entry is untouched


def test_conditions_and_checks():
    t, notes = apply_condition(TEMPLATE, {"AoA": 2.0, "Mach": 0.8, "Re": 0.0, "TurbModel": 0,
                                           "Altitude": 36000.0, "EngineFlow": 2, "MassFlow": 1.0})
    assert [get_namelist_value(t, k) for k in ("alpha", "MachNumber", "ReynoldsNumber", "turbulenceModel")] == \
        ["2", "0.8", "0", "0"]
    assert notes and "not used" in notes[0]
    assert apply_condition(TEMPLATE, {})[0] == TEMPLATE                    # no condition: template untouched

    err, warn = check_control(TEMPLATE, n_grids=5, boundary_flags=[1, 3], time_limit_s=600)
    assert not err and any("numberOfMGIterations" in w for w in warn)
    assert check_control(TEMPLATE, n_grids=3)[0]                           # uses 5 grids, only 3 generated
    assert check_control("some 3D input file", n_grids=5)[0]
    viscous = set_namelist_value(TEMPLATE, "ReynoldsNumber", 4.0e5)
    assert check_control(viscous, 5, boundary_flags=[1, 3])[0]             # viscous flow, inviscid walls
    assert not check_control(viscous, 5, boundary_flags=[5, 3])[0]
    assert check_control(TEMPLATE, 5, boundary_flags=[5, 3])[1]            # inviscid flow, viscous walls: warning
    capped = set_namelist_value(TEMPLATE, "numberOfMGIterations", 500)
    assert not any("numberOfMGIterations" in w for w in check_control(capped, 5, [1, 3])[1])


def test_prepro_options():
    o = PreProOptions(n_grids=4, hybrid=False, directionality="0", n_directional=0)
    assert o.answers("m.dat", "m.sol") == ["m.dat", ".false.", "4", "0", "0", "m.sol"]
    # the count of directional agglomerations is only asked for a non-zero directionality;
    # the visualization mode (always 0) comes just before the output file
    assert PreProOptions(n_grids=4, n_directional=3).answers("m.dat", "m.sol") == \
        ["m.dat", ".false.", "4", "0", "0", "m.sol"]
    assert PreProOptions(n_grids=4, directionality="1", n_directional=3).answers("m.dat", "m.sol") == \
        ["m.dat", ".false.", "4", "1", "3", "0", "m.sol"]
    assert PreProOptions.from_dict(o.to_dict()) == o
    for bad in (dict(n_grids=0), dict(n_directional=-1), dict(directionality="abc")):
        try:
            PreProOptions(**bad)
            raise AssertionError(f"{bad} accepted")
        except ValueError:
            pass


def test_full_chain_two_conditions():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case = _case(d, logs)
        case.run_prepro(PreProOptions(n_grids=3, directionality="1", n_directional=2))   # 7 answers
        res = case.run(PreProOptions(n_grids=5), TEMPLATE,
                       conds=[{"AoA": 2.0, "Mach": 0.5, "Re": 0.0}, {"AoA": 4.0, "Mach": 0.7, "Re": 0.0}],
                       boundary_flags=[1, 3], solver_time_limit_s=60)
        assert [r["status"] for r in res] == ["ok", "ok"]
        assert abs(res[0]["CL"] - 0.22) < 1e-9 and abs(res[1]["CL"] - 0.44) < 1e-9     # 0.11 * alpha
        assert abs(res[1]["CD"] - 0.0007) < 1e-12 and res[0]["iterations"] == 5
        # files where the layout says, answers with short relative names
        with open(os.path.join(case.pre_dir, "runPrePro.inp")) as fh:
            a = fh.read().split("\n")
        assert a[:6] == [os.path.join("..", "..", "volumes", "n_0", "naca_0.dat"), ".false.", "5", "0", "0",
                         "naca_0.sol"]
        assert os.path.getsize(case.sol_path) > 0
        for i in (1, 2):
            cd = case.cond_dir(i)
            assert sorted(os.listdir(cd)) == sorted(
                ["metrics.json", "naca.inp", "naca_0.res", "naca_0.rsd", "runSolver.inp", "solver.log"] + PICTURES)
            assert all(os.path.getsize(os.path.join(cd, f)) > 1000 for f in PICTURES)
            with open(os.path.join(cd, "runSolver.inp")) as fh:
                assert fh.read().split("\n")[:6] == [
                    "naca.inp", os.path.join("..", "..", "..", "preprocessed", "n_0", "naca_0.sol"), "",
                    "naca_0.res", "naca_0.rsd", ""]
            with open(os.path.join(cd, "metrics.json")) as fh:
                m = json.load(fh)
            assert m["status"] == "ok" and m["res_complete"] and m["res_iteration"] == 5
            assert sorted(os.path.basename(f) for f in m["plots"]) == sorted(PICTURES)
        assert os.path.getsize(case.mesh_image_path) > 1000          # surfaces/n_0/naca_0.png
        assert case.mesh_image_path.endswith(os.path.join("surfaces", "n_0", "naca_0.png"))
        with open(os.path.join(case.cond_dir(2), "naca.inp")) as fh:
            assert get_namelist_value(fh.read(), "MachNumber") == "0.7"
        assert any("[2D][PREPRO] done" in ln for ln in logs) and any("cond 2 finished" in ln for ln in logs)


def test_nothing_runs_if_a_control_file_is_wrong():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case = _case(d, logs)
        try:
            case.run(PreProOptions(n_grids=5), TEMPLATE, conds=[{"Mach": 0.5}, {"Re": 1.0e6}],
                     boundary_flags=[1, 3])
            raise AssertionError("viscous condition on inviscid walls was run")
        except ValueError as e:
            assert "cond 2" in str(e)
        assert not os.path.exists(case.pre_dir) and not os.path.exists(case.sol_root)


def test_diverged_failed_and_timeout_are_not_numbers():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case = _case(d, logs)
        case.run_prepro(PreProOptions(n_grids=5))
        nan = case.run_solver(set_namelist_value(TEMPLATE, "MachNumber", 9.99), nc=1, time_limit_s=60)
        assert nan["status"] == "diverged" and nan["CL"] != nan["CL"]
        dead = case.run_solver(set_namelist_value(TEMPLATE, "MachNumber", 7.77), nc=2, time_limit_s=60)
        assert dead["status"] == "failed" and dead["rows"] == 0
        died = case.run_solver(set_namelist_value(TEMPLATE, "MachNumber", 6.66), nc=5, time_limit_s=60)
        assert died["status"] == "crashed" and died["rows"] == 1      # rows were written, still not a result
        assert any("fix_flite2d_exes.py" in ln for ln in logs)        # the known defect is pointed out
        slow = case.run_solver(set_namelist_value(TEMPLATE, "MachNumber", 8.88), nc=3, time_limit_s=1.5)
        assert slow["status"] == "timeout" and slow["wall_time_s"] < 10
        assert any("time limit reached" in ln and "NOT converged" in ln for ln in logs)
        # a stale result from an earlier good run must not survive a failed re-run
        good = case.run_solver(TEMPLATE, nc=4, time_limit_s=60)
        assert good["status"] == "ok"
        again = case.run_solver(set_namelist_value(TEMPLATE, "MachNumber", 7.77), nc=4, time_limit_s=60)
        assert again["status"] == "failed" and not os.path.exists(os.path.join(case.cond_dir(4), "naca_0.rsd"))


def test_prepro_failures_raise():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case = _case(d, logs)
        try:
            case.run_prepro(PreProOptions(n_grids=12))            # stand-in refuses > 9 grids, exit code 0
            raise AssertionError("PrePro without a .sol file was accepted")
        except RuntimeError as e:
            assert "no .sol file" in str(e) and "To many grids" in str(e)
        try:
            case.run_prepro(PreProOptions(n_grids=8))             # stand-in dies after a partial .sol
            raise AssertionError("crashed PrePro was accepted")
        except RuntimeError as e:
            assert "run-time error" in str(e) and not os.path.exists(case.sol_path)
            assert "fix_flite2d_exes.py" in str(e)
        os.remove(case.mesh_path)
        try:
            case.run_prepro(PreProOptions())
            raise AssertionError("missing mesh accepted")
        except FileNotFoundError:
            pass
        try:
            case.run_solver(TEMPLATE)                              # no .sol
            raise AssertionError("solver ran without a .sol file")
        except FileNotFoundError:
            pass


def test_stop_request_and_read_rsd():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case = _case(d, logs)
        case.run_prepro(PreProOptions())
        m = case.run_solver(set_namelist_value(TEMPLATE, "MachNumber", 8.88), nc=1, time_limit_s=60,
                            should_stop=lambda: True)
        assert m["status"] == "stopped"
        p = os.path.join(d, "x.rsd")
        assert read_rsd(p)["status"] == "no_data"
        with open(p, "w") as fh:
            fh.write("header line\n   1  -0.5  0.1  0.01  0.0\n   2  -1.0  0.2  0.02  0.3\n")
        r = read_rsd(p)
        assert (r["status"], r["iterations"], r["CL"], r["CD"], r["CM"], r["rows"]) == ("ok", 2, 0.2, 0.02, 0.3, 2)


def test_stop_while_the_solver_runs_keeps_what_was_computed():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case = _case(d, logs)
        endless = set_namelist_value(set_namelist_value(TEMPLATE, "MachNumber", 8.88), "writeToFileInterval", 10)
        rsd = os.path.join(case.cond_dir(1), "naca_0.rsd")

        def stop():                                            # once the solver has done 25 iterations
            return os.path.isfile(rsd) and read_rsd(rsd)["rows"] >= 25
        res = case.run(PreProOptions(n_grids=5), endless, conds=[{}, {"AoA": 3.0}], boundary_flags=[1, 3],
                       solver_time_limit_s=120, should_stop=stop)
        assert len(res) == 1                                   # the second condition was never started
        m = res[0]
        assert m["status"] == "stopped" and m["wall_time_s"] < 30 and m["iterations"] >= 10
        # the flow field is the last one the solver wrote, whole, and from a multiple of the interval
        assert m["res_complete"] and m["res_iteration"] % 10 == 0
        assert 0 < m["res_iteration"] <= m["iterations"]
        assert m["CL"] == m["CL"]                              # the last lift is kept (it is flagged, not hidden)
        cd = case.cond_dir(1)
        assert all(os.path.getsize(os.path.join(cd, f)) > 1000 for f in PICTURES)
        assert not os.path.exists(case.cond_dir(2))
        assert any("stopped by the user" in ln and "NOT converged" in ln for ln in logs)
        assert any("were not run" in ln for ln in logs)


def test_stop_while_the_preprocessor_runs():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case = _case(d, logs)
        t0 = time.time()
        try:
            case.run(PreProOptions(n_grids=7), set_namelist_value(TEMPLATE, "numberOfGridsToUse", 5),
                     boundary_flags=[1, 3], should_stop=lambda: time.time() - t0 > 1.0)   # stand-in hangs for 7
            raise AssertionError("a stopped preprocessor was not reported")
        except RunStopped:
            pass
        assert time.time() - t0 < 20 and not os.path.exists(case.sol_path) and not os.path.exists(case.sol_root)


def test_pictures_can_be_switched_off_and_never_stop_a_run():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case = _case(d, logs)
        case.make_plots = False
        res = case.run(PreProOptions(n_grids=5), TEMPLATE, boundary_flags=[1, 3])
        assert res[0]["status"] == "ok" and "plots" not in res[0]
        assert not os.path.exists(case.mesh_image_path)
        assert not os.path.exists(os.path.join(case.cond_dir(1), "mach.png"))
        # a result file cut short (solver killed mid-write): convergence picture only, run still reported
        case.make_plots = True
        res_file = os.path.join(case.cond_dir(1), "naca_0.res")
        with open(res_file) as fh:
            rows = fh.read().splitlines()
        with open(res_file, "w") as fh:
            fh.write("\n".join(rows[:len(rows) // 2]) + "\n    7    0.1E+0")
        m = dict(res[0])
        made = case.write_plots(1, m, TEMPLATE)
        assert [os.path.basename(f) for f in made] == ["convergence.png"]
        assert any("stopped while it was writing" in ln for ln in logs)


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
