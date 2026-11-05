"""
Tests of the local 2D optimisation (Local/local2DOpt.py).

    python -m pytest tests/test_local2d_opt.py -q      or      python tests/test_local2d_opt.py

PrePro.exe and Solver.exe are replaced by the stand-ins in tests/standins. The
stand-in solver is started with --geometry, so its drag depends on the wall
shape (minimum at a mean |y| of 0.03; the NACA 0012 baseline has about 0.04).
These tests check the ORCHESTRATION - design numbering, files, the handling of
designs without a result, stop and resume, the set-up checks - with the real
optimiser, the real morph and the real mesh writer. They say nothing about the
flow solver.
"""
import csv
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from test_edge_modes_2d import naca0012, o_mesh                                   # noqa: E402
from MeshGeneration.meshFile import load_mesh                                     # noqa: E402
from ShapeParameterization.edgeModes2D import EdgeModes2D                         # noqa: E402
from Local.local2DOpt import Design2DBuilder, run, tail_range                     # noqa: E402

PREPRO = [sys.executable, os.path.join(HERE, "standins", "fake_prepro.py")]
SOLVER = [sys.executable, os.path.join(HERE, "standins", "fake_solver.py"), "--geometry"]
with open(os.path.join(os.path.dirname(HERE), "templates", "flite2d_solver.inp")) as _fh:
    TEMPLATE = _fh.read()


def make_case(d, bound=0.006, n_modes=3, n_samples=4, count_limit=3, expression="CD", constraints=(),
              conditions=({"AoA": 2.0, "Mach": 0.5, "Re": 0.0, "Weight": 1.0},), **extra):
    """A complete case folder as the GUI leaves it, plus <case>/opt with the two json files."""
    case = os.path.join(d, "case dir")                           # blank in the path on purpose
    os.makedirs(os.path.join(case, "Control Nodes"))
    m0, _ = o_mesh(naca0012(40), n_layers=10)
    fro = m0.write_file(os.path.join(case, "naca.fro"))
    mesh = load_mesh(fro).mesh2d
    modes = EdgeModes2D.from_mesh(mesh, [1], n_modes=n_modes)
    modes.save(os.path.join(case, "Control Nodes", "edge_modes_2d.json"))
    with open(os.path.join(case, "Control Nodes", "morph_basis.json"), "w") as fh:
        json.dump({"dim": 2, "parameterisation_method": "edge_modes_2d", "t_edges": [1], "u_edges": [],
                   "support_frac": 1.0, "min_area_ratio": 0.05, "solver_bc_flags": {"1": 1, "2": 3, "3": 3},
                   "solver_mesh_scale": 1.0, "baseline_mesh_path": fro, "dat_path": None,
                   "edge_modes_file": "edge_modes_2d.json"}, fh)
    with open(os.path.join(case, "naca.inp"), "w") as fh:
        fh.write(TEMPLATE)
    run_dir = os.path.join(case, "opt")
    os.makedirs(run_dir)
    settings = {"n_dim": modes.n_modes, "n_obj": 1, "lb": [-bound] * modes.n_modes, "ub": [bound] * modes.n_modes,
                "count_limit": count_limit, "n_samples": n_samples, "kernel": "Mat52Kern",
                "acquisition_function": "Expected Improvement", "base_name": "naca", "output_dir": case,
                "prepro_cmd": PREPRO, "solver_cmd": SOLVER, "constraints": list(constraints),
                "af_maxfevals": 300, "mll_maxfevals": 200, "seed": 3, "bounds_check_samples": 20,
                "make_plots": False}
    settings.update(extra)
    with open(os.path.join(run_dir, "bo_settings.json"), "w") as fh:
        json.dump(settings, fh)
    with open(os.path.join(run_dir, "objective.json"), "w") as fh:
        json.dump({"objective_type": "Custom", "expression": expression, "terms": [],
                   "conditions": list(conditions), "constraints": []}, fh)
    return case, run_dir


def designs(run_dir):
    with open(os.path.join(run_dir, "designs.json")) as fh:
        return sorted(json.load(fh)["designs"], key=lambda r: r["n"])


def test_full_optimisation_numbers_files_and_improvement():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case, run_dir = make_case(d, make_plots=True)
        assert run(run_dir, echo=logs.append) == 0, "\n".join(logs[-15:])
        D = designs(run_dir)
        # baseline + 4 initial samples in generation 0, then one design per optimiser step
        assert [r["n"] for r in D] == list(range(8)) and [r["gen"] for r in D] == [0] * 5 + [1, 2, 3]
        assert D[0]["baseline"] and D[0]["x"] == [0.0, 0.0, 0.0] and not any(r["baseline"] for r in D[1:])
        assert all(r["done"] and r["status"] == "ok" and r["objective"] is not None for r in D)
        assert all(r["objective"] == r["objective_used"] for r in D)       # nothing was imputed
        for r in D:                                                         # files of every design
            n = r["n"]
            assert os.path.isfile(os.path.join(case, "volumes", f"n_{n}", f"naca_{n}.dat"))
            with open(os.path.join(case, "volumes", f"n_{n}", "design.json")) as fh:
                assert json.load(fh)["x"] == r["x"]
            assert os.path.isfile(os.path.join(case, "solutions", f"n_{n}", "cond_1", "metrics.json"))
            assert os.path.isfile(os.path.join(case, "surfaces", f"n_{n}", f"naca_{n}.png"))
            assert not os.path.exists(os.path.join(case, "preprocessed", f"n_{n}", f"naca_{n}.sol"))   # keep_sol off
            assert all(-0.006 <= v <= 0.006 for v in r["x"])
        # the objective in the table is the drag of that design's own run
        with open(os.path.join(case, "solutions", "n_3", "cond_1", "metrics.json")) as fh:
            assert abs(json.load(fh)["CD"] - D[3]["objective"]) < 1e-12
        with open(os.path.join(run_dir, "optimization_results.json")) as fh:
            res = json.load(fh)
        best = min(D, key=lambda r: r["objective"])
        assert res["best_design_n"] == best["n"] and abs(res["Y_best"] - best["objective"]) < 1e-9
        assert res["baseline_objective"] == D[0]["objective"] and res["n_without_result"] == 0
        assert best["objective"] < D[0]["objective"]                        # thinner than the baseline is better
        with open(os.path.join(run_dir, "designs.csv")) as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == 9 and rows[0][:4] == ["n", "generation", "status", "objective"]
        assert float(rows[1 + best["n"]][3]) == float(f"{best['objective']:.10g}")
        for f in ("history.png", "bo_data.mcsv", "local_opt.log"):
            assert os.path.getsize(os.path.join(run_dir, f)) > 0
        assert any("Best design: n=" in ln and "baseline" in ln for ln in logs)
        assert any("% give a valid mesh" in ln for ln in logs)


def test_stop_and_resume_recomputes_nothing_that_was_finished():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case, run_dir = make_case(d)

        def echo(msg):                         # press "stop" when the third design starts
            logs.append(msg)
            if "---- design n=2 " in msg:
                open(os.path.join(run_dir, "STOP"), "w").close()
        assert run(run_dir, echo=echo) == 3
        D = designs(run_dir)
        assert [r["n"] for r in D if r["done"]] == [0, 1] and len(D) == 5
        assert any("STOPPED" in ln and "2 design(s) are kept" in ln for ln in logs)
        assert not os.path.exists(os.path.join(run_dir, "STOP"))
        first = {r["n"]: (r["x"], r["objective"]) for r in D}
        stamp = os.path.getmtime(os.path.join(case, "solutions", "n_1", "cond_1", "metrics.json"))

        logs2 = []
        assert run(run_dir, echo=logs2.append) == 0, "\n".join(logs2[-15:])
        D2 = designs(run_dir)
        assert [r["n"] for r in D2] == list(range(8)) and all(r["done"] for r in D2)
        assert sum("already computed, result reused" in ln for ln in logs2) == 2
        assert os.path.getmtime(os.path.join(case, "solutions", "n_1", "cond_1", "metrics.json")) == stamp
        for n in (0, 1):                       # same designs, same numbers, as before the stop
            assert np.allclose(D2[n]["x"], first[n][0], rtol=0.0, atol=1e-7)
        # (the objective is attached to a design when its generation completes)
        assert all(r["objective"] is not None for r in D2)
        for n in (2, 3, 4):                    # the rest of the initial sample is the one drawn before the stop
            assert np.allclose(D2[n]["x"], first[n][0], rtol=0.0, atol=1e-7)
        # a finished optimisation started again does nothing
        logs3 = []
        assert run(run_dir, echo=logs3.append) == 0
        assert not any("---- design" in ln for ln in logs3)


def test_stop_simulation_ends_one_design_and_the_optimisation_goes_on():
    """SKIP (the GUI's Stop Simulation during an optimisation): the design being computed ends
    without a result, its remaining flow conditions are not run, everything else carries on."""
    logs = []
    with tempfile.TemporaryDirectory() as d:
        # condition 1 takes two seconds in the stand-in solver (Mach 3.33), condition 2 is immediate
        case, run_dir = make_case(d, n_samples=2, count_limit=1,
                                  conditions=({"AoA": 2.0, "Mach": 3.33, "Re": 0.0, "Weight": 1.0},
                                              {"AoA": 1.0, "Mach": 0.5, "Re": 0.0, "Weight": 1.0}))
        skip = os.path.join(run_dir, "SKIP")

        def echo(msg):
            logs.append(msg)
            if "[2D][SOLVER] n=1 cond 1" in msg:           # the solver of design 1 is about to start
                open(skip, "w").close()
            if "---- design n=2 " in msg:                  # nothing is running at this moment
                open(skip, "w").close()
        t0 = time.time()
        assert run(run_dir, echo=echo) == 0, "\n".join(logs[-15:])
        D = designs(run_dir)
        assert [r["n"] for r in D] == [0, 1, 2, 3] and all(r["done"] for r in D)
        assert [r["status"] for r in D] == ["ok", "skipped", "ok", "ok"]
        assert [m["status"] for m in D[1]["metrics"]] == ["skipped", "skipped"]
        assert D[1]["objective"] is None and all(r["objective"] is not None for r in (D[0], D[2], D[3]))
        # it is given the penalty: worse than every design that has a result
        assert D[1]["objective_used"] > max(r["objective"] for r in (D[0], D[2], D[3]))
        # what the run left behind: the solver was killed part-way, condition 2 was never started
        with open(os.path.join(case, "solutions", "n_1", "cond_1", "metrics.json")) as fh:
            m = json.load(fh)
        assert m["status"] == "stopped" and 0 < m["rows"] < 40
        assert not os.path.exists(os.path.join(case, "solutions", "n_1", "cond_2"))
        assert os.path.isfile(os.path.join(case, "solutions", "n_2", "cond_2", "metrics.json"))
        assert D[1]["metrics"][0]["iterations"] == m["iterations"]         # where it was stopped is on record
        assert not os.path.exists(skip)
        assert sum("'Stop Simulation' - ending the simulation of this design" in ln for ln in logs) == 1
        assert sum("arrived while no simulation was running" in ln for ln in logs) == 1
        assert any("n=1: objective = none (skipped)" in ln for ln in logs)
        assert D[1]["wall_time_s"] < 2.0 < min(D[2]["wall_time_s"], D[3]["wall_time_s"])   # it really was cut short
        # a skipped design is final: nothing is computed again when the run is started again
        logs2 = []
        assert run(run_dir, echo=logs2.append) == 0
        assert not any("---- design" in ln for ln in logs2) and designs(run_dir)[1]["status"] == "skipped"


def test_stop_simulation_during_the_preprocessor_and_on_the_baseline():
    import Local.local2DOpt as L
    real_run = L.Flite2DCase.run
    with tempfile.TemporaryDirectory() as d:
        case, run_dir = make_case(d, n_samples=2, count_limit=1)
        skip = os.path.join(run_dir, "SKIP")

        def run_with_a_hanging_preprocessor(self, opts, control, conds=None, flags=None,
                                            solver_time_limit_s=3600.0, should_stop=None, force_check=None):
            if self.n != 2:
                return real_run(self, opts, control, conds, flags, solver_time_limit_s=solver_time_limit_s,
                                should_stop=should_stop, force_check=force_check)
            open(skip, "w").close()                         # pressed while "the preprocessor" runs
            t0 = time.time()
            while not should_stop() and time.time() - t0 < 20.0:
                time.sleep(0.02)
            raise L.RunStopped("stopped while the preprocessor was running; nothing was computed.")
        L.Flite2DCase.run = run_with_a_hanging_preprocessor
        logs = []
        try:
            assert run(run_dir, echo=logs.append) == 0, "\n".join(logs[-15:])
        finally:
            L.Flite2DCase.run = real_run
        D = designs(run_dir)
        assert [r["status"] for r in D] == ["ok", "ok", "skipped", "ok"]
        assert [m["status"] for m in D[2]["metrics"]] == ["skipped"] and D[2]["objective"] is None
        assert any("n=2: stopped before the solver had started" in ln for ln in logs)

    # the baseline: without it there is nothing to compare with, so the run ends, and the next
    # start computes the baseline again instead of reusing "skipped"
    with tempfile.TemporaryDirectory() as d:
        case, run_dir = make_case(d, n_samples=2, count_limit=1,
                                  conditions=({"AoA": 2.0, "Mach": 3.33, "Re": 0.0, "Weight": 1.0},))
        logs = []

        def echo(msg):
            logs.append(msg)
            if "[2D][SOLVER] n=0 cond 1" in msg:
                open(os.path.join(run_dir, "SKIP"), "w").close()
        assert run(run_dir, echo=echo) == 1
        assert any("baseline (n=0) was ended with 'Stop Simulation'" in ln for ln in logs)
        D = designs(run_dir)
        assert D[0]["status"] == "skipped" and not D[0]["done"]
        logs2 = []
        assert run(run_dir, echo=logs2.append) == 0, "\n".join(logs2[-15:])
        D2 = designs(run_dir)
        assert any("---- design n=0 " in ln for ln in logs2)
        assert [r["status"] for r in D2] == ["ok"] * 4 and all(r["done"] for r in D2)


def test_a_changed_problem_is_not_resumed():
    with tempfile.TemporaryDirectory() as d:
        case, run_dir = make_case(d, count_limit=1, n_samples=2)
        assert run(run_dir, echo=lambda m: None) == 0
        with open(os.path.join(run_dir, "objective.json")) as fh:
            obj = json.load(fh)
        obj["expression"] = "-CL"
        with open(os.path.join(run_dir, "objective.json"), "w") as fh:
            json.dump(obj, fh)
        logs = []
        assert run(run_dir, echo=logs.append) == 1
        assert any("DIFFERENT problem" in ln for ln in logs)


def test_designs_without_a_result_are_penalised_not_counted():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        # a box far too large for this mesh: many morphs turn triangles inside out
        case, run_dir = make_case(d, bound=0.6, n_samples=8, count_limit=2, seed=1)
        assert run(run_dir, echo=logs.append) == 0, "\n".join(logs[-15:])
        D = designs(run_dir)
        bad = [r for r in D if r["status"] == "invalid_mesh"]
        good = [r for r in D if r["objective"] is not None]
        assert bad and good and len(bad) + len(good) == len(D)
        for r in bad:
            assert r["objective"] is None and not r["morph"]["ok"]
            assert r["objective_used"] > max(g["objective"] for g in good if g["gen"] <= r["gen"])
            assert not os.path.exists(os.path.join(case, "solutions", f"n_{r['n']}"))    # no CFD was run
        with open(os.path.join(run_dir, "optimization_results.json")) as fh:
            res = json.load(fh)
        assert res["n_without_result"] == len(bad) and res["best_design_n"] in [g["n"] for g in good]
        assert any("of the box gives an invalid mesh" in ln and "[WARN]" in ln for ln in logs)   # said before any CFD
        assert any("the morph gives an invalid mesh" in ln for ln in logs)


def test_timed_out_designs_with_settled_forces_are_accepted():
    """Mach 8.88: the stand-in never finishes, with constant lift and drag. With the force check,
    every design hits the time limit and is accepted; without it, the baseline has no result."""
    with tempfile.TemporaryDirectory() as d:
        logs = []
        case, run_dir = make_case(d, n_samples=2, count_limit=1, conditions=({"AoA": 2.0, "Mach": 8.88, "Re": 0.0},),
                                  solver_time_limit_min=0.02, force_check={"enabled": True, "window": 5, "tol": 1e-4})
        assert run(run_dir, echo=logs.append) == 0, "\n".join(logs[-15:])
        D = [r for r in designs(run_dir) if r["done"]]
        assert D and all(r["status"] == "ok" and r["objective"] is not None for r in D)
        for r in D:
            m = r["metrics"][0]
            assert m["converged_by"] == "forces" and m["force_check"]["passed"]
            assert m["force_check"]["CL_spread"] == 0.0 and m["force_check"]["window"] == 5
        assert any("time limit is ACCEPTED" in ln for ln in logs)
        assert any("forces had settled" in ln and "ACCEPTED" in ln for ln in logs)
        with open(os.path.join(run_dir, "designs.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert rows and all(row["cond1_converged_by"] == "forces" for row in rows if row["status"] == "ok")
    with tempfile.TemporaryDirectory() as d:
        logs = []
        case, run_dir = make_case(d, n_samples=2, count_limit=1, conditions=({"AoA": 2.0, "Mach": 8.88, "Re": 0.0},),
                                  solver_time_limit_min=0.02, force_check={"enabled": False})
        assert run(run_dir, echo=logs.append) == 1
        assert designs(run_dir)[0]["status"] == "timeout"
        assert any("force check is off" in ln for ln in logs)
    with tempfile.TemporaryDirectory() as d:
        logs = []
        case, run_dir = make_case(d, force_check={"enabled": True, "tol": -1})
        assert run(run_dir, echo=logs.append) == 1
        assert any("tol must be a positive number" in ln and "[ERROR]" in ln for ln in logs)


def test_a_failing_baseline_stops_the_run_at_once():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        # Mach 9.99 makes the stand-in solver return NaN: every design "diverges"
        case, run_dir = make_case(d, conditions=({"AoA": 0.0, "Mach": 9.99, "Re": 0.0},))
        assert run(run_dir, echo=logs.append) == 1
        assert any("baseline" in ln and "[ERROR]" in ln for ln in logs)
        assert len([r for r in designs(run_dir) if r["done"]]) == 1          # only the baseline was run


def test_constraint_values_and_feasibility():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        # stand-in lift = 0.11 alpha + 5 camber = 0.22 + 5 camber; ask for at least the baseline's lift
        case, run_dir = make_case(d, constraints=["CL >= 0.22"], n_samples=6, count_limit=2)
        assert run(run_dir, echo=logs.append) == 0, "\n".join(logs[-15:])
        D = designs(run_dir)
        for r in D:
            with open(os.path.join(case, "solutions", f"n_{r['n']}", "cond_1", "metrics.json")) as fh:
                cl = json.load(fh)["CL"]
            assert abs(r["constraints"]["CL"] - cl) < 1e-12 and r["feasible"] == (cl >= 0.22)
        with open(os.path.join(run_dir, "optimization_results.json")) as fh:
            res = json.load(fh)
        feas = [r for r in D if r["feasible"]]
        assert feas and res["best_design_n"] == min(feas, key=lambda r: r["objective"])["n"]
        assert any("Constraints: ['CL >= 0.22']" in ln for ln in logs)


def test_set_up_errors_are_reported_before_any_run():
    def fails(expect, **kw):
        logs = []
        with tempfile.TemporaryDirectory() as d:
            mutate = kw.pop("mutate", None)
            case, run_dir = make_case(d, **kw)
            if mutate:
                mutate(case, run_dir)
            assert run(run_dir, echo=logs.append) == 1, expect
            assert any(expect in ln for ln in logs), (expect, logs[-3:])
            assert not os.path.exists(os.path.join(case, "solutions")) and \
                not os.path.exists(os.path.join(run_dir, "bo_data.mcsv"))

    fails("uses ['PR_s5']", expression="-PR_s5")                                   # a 3D monitor symbol
    fails("no edge is a viscous wall", conditions=({"AoA": 2.0, "Mach": 0.5, "Re": 1.0e6},))
    fails("without a minus sign", constraints=["-CL <= -0.2"])
    fails("has 3 mode(s)", lb=[-0.01, -0.01], ub=[0.01, 0.01], n_dim=2)           # bounds of an older basis
    fails("solver control file not found", mutate=lambda case, rd: os.remove(os.path.join(case, "naca.inp")))
    fails("Baseline mesh not found", mutate=lambda case, rd: os.remove(os.path.join(case, "naca.fro")))
    fails("unknown kernel", kernel="NoSuchKernel")

    def drop_a_flag(case, rd):
        p = os.path.join(case, "Control Nodes", "morph_basis.json")
        with open(p) as fh:
            b = json.load(fh)
        b["solver_bc_flags"].pop("3")
        with open(p, "w") as fh:
            json.dump(b, fh)
    fails("solver boundary types", mutate=drop_a_flag)


def test_check_mode_and_helpers():
    logs = []
    with tempfile.TemporaryDirectory() as d:
        case, run_dir = make_case(d, bound=0.6, bounds_check_samples=40)
        assert run(run_dir, check_only=True, echo=logs.append) == 0
        assert not os.path.exists(os.path.join(case, "volumes")) and not os.path.exists(os.path.join(run_dir, "designs.json"))
        assert any("No CFD was run" in ln for ln in logs) and any("tighten the bounds" in ln for ln in logs)
        b = Design2DBuilder(os.path.join(case, "Control Nodes", "morph_basis.json"), log=lambda m: None)
        assert b.n_modes == 3 and b.wall_displacement([0, 0, 0]) == 0.0
        x = [0.004, -0.002, 0.001]
        mesh, rep = b.morphed(x)
        wall = b.morph.t_ids
        moved = np.linalg.norm(mesh.nodes[wall, :2] - b.mesh.nodes[wall, :2], axis=1).max()
        assert rep.ok and abs(moved - b.wall_displacement(x)) < 1e-12              # the reported displacement is real
        r = b.box_report([-0.001] * 3, [0.001] * 3, 10)
        assert r["valid_fraction"] == 1.0 and 0 < r["wall_disp_max"] < 0.004
    assert tail_range([1, 2, 3]) != tail_range([1, 2, 3])                            # too short: NaN
    assert tail_range(list(range(100))) == 19.0 and tail_range([5.0] * 50) == 0.0


def test_optimiser_data_file_keeps_every_value_exactly():
    # bo_data.mcsv is re-read by the optimiser before every generation. Written with numpy's
    # print defaults it was abbreviated with "..." above 1000 values (unreadable), wrapped
    # long rows and kept 8 digits.
    from FileRW.MultiArrayCsvFile import MultiArrayCsvFile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        f = MultiArrayCsvFile(os.path.join(d, "bo_data.mcsv"))
        for shape in ((60, 20), (200, 40), (5, 1), (1, 6)):
            X = rng.normal(size=shape) * 10.0 ** rng.integers(-6, 3)
            Y = rng.normal(size=(shape[0], 1)) * 1e-5
            f.write({"X": X, "Y": Y, "X_uneval": np.array([]), "gen_num": [7], "C__CL": Y})
            r = f.read()
            assert np.array_equal(r["X"], X) and np.array_equal(r["Y"], Y) and np.array_equal(r["C__CL"], Y)
            assert r["X_uneval"].size == 0 and int(np.ravel(r["gen_num"])[0]) == 7


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
