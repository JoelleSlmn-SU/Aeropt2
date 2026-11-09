"""Sampling methods, seed, random-search strategy, geometric (pre-CFD) constraints."""
import csv
import json
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from Optimisation.BayesianOptimisation.design_space import (   # noqa: E402
    sample_box, GeometricConstraints, Linearised,
)
from Optimisation.BayesianOptimisation.optimiser import BayesianOptimiser   # noqa: E402
from Optimisation.BayesianOptimisation.kernels import Mat52Kern              # noqa: E402
from Optimisation.BayesianOptimisation.acquisition_functions import EI       # noqa: E402


# ------------------------------------------------------------------ sampling
@pytest.mark.parametrize("method", ["lhs", "lhs_maximin", "random", "sobol"])
def test_sampling_in_box_and_repeatable(method):
    lb, ub = np.array([-1.0, 0.0, 2.0]), np.array([1.0, 0.5, 3.0])
    a = sample_box(method, 16, lb, ub, np.random.default_rng(3))
    b = sample_box(method, 16, lb, ub, np.random.default_rng(3))
    c = sample_box(method, 16, lb, ub, np.random.default_rng(4))
    assert a.shape == (16, 3) and np.all(a >= lb) and np.all(a <= ub)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    if method.startswith("lhs"):          # one point per stratum in every dimension
        for j in range(3):
            strata = np.floor((a[:, j] - lb[j]) / (ub[j] - lb[j]) * 16).astype(int)
            assert sorted(np.clip(strata, 0, 15)) == list(range(16))


def test_lhs_does_not_disturb_global_numpy_state():
    np.random.seed(11)
    ref = np.random.random(3)
    np.random.seed(11)
    sample_box("lhs", 8, [0, 0], [1, 1], np.random.default_rng(0))
    assert np.array_equal(np.random.random(3), ref)


# ------------------------------------------------------------------ geometric constraints
def _lin_geom(X):
    X = np.atleast_2d(X)
    return {"geom_a": X[:, 0] + X[:, 1],                         # scalar per design
            "geom_v": np.column_stack([X[:, 0], -X[:, 1] + 0.5])}  # per station


def test_linearised_exact_for_linear_and_detects_nonlinear():
    rng = np.random.default_rng(0)
    L = Linearised(_lin_geom, [-1, -1], [1, 1], rng, log=lambda m: None)
    assert not L.exact
    X = rng.uniform(-1, 1, (5, 2))
    for k, v in _lin_geom(X).items():
        assert np.allclose(L(X)[k], np.asarray(v).reshape(5, -1))
    Q = Linearised(lambda X: {"geom_q": np.atleast_2d(X)[:, 0] ** 2}, [-1, -1], [1, 1], rng, log=lambda m: None)
    assert Q.exact_names == ["geom_q"]
    assert np.allclose(Q(np.array([[0.5, 0.0]]))["geom_q"], 0.25)


def test_geometric_feasibility_and_rejection_sampling():
    specs = [{"metric": "geom_a", "sense": ">=", "limit": 0.0},
             {"metric": "geom_v", "sense": ">=", "limit": 0.0}]
    g = GeometricConstraints(specs, _lin_geom, [-1, -1], [1, 1], np.random.default_rng(0), log=lambda m: None)
    X, rate = g.sample_feasible("lhs", 20, [-1, -1], [1, 1], np.random.default_rng(1))
    assert len(X) == 20 and 0 < rate < 1
    assert np.all(X[:, 0] >= 0) and np.all(X[:, 1] <= 0.5) and np.all(X.sum(1) >= 0)
    assert g.violation([[0.5, 0.0]])[0] == 0 and g.violation([[-0.5, 0.0]])[0] > 0
    with pytest.raises(ValueError, match="no geometry function"):
        GeometricConstraints(specs, None, [-1, -1], [1, 1], np.random.default_rng(0), log=lambda m: None)


# ------------------------------------------------------------------ optimiser end to end (analytic)
def _settings(d, **kw):
    s = {"n_dim": 2, "n_obj": 1, "lb": [-1.0, -1.0], "ub": [1.0, 1.0], "n_samples": 6, "count_limit": 4,
         "kernel": Mat52Kern, "acquisition_function": EI, "sim_dir": d + "/", "af_maxfevals": 300,
         "mll_maxfevals": 200}
    s.update(kw)
    return s


def _run(d, **kw):
    f = lambda X: np.sum((np.atleast_2d(X) - 0.3) ** 2, axis=1)     # noqa: E731
    bo = BayesianOptimiser(_settings(d, **kw), eval_func=lambda X, g: f(X), init_func=lambda X, g: None,
                           geom_func=kw.pop("geom_func", None) if "geom_func" in kw else None)
    bo.optimise(cont=True)
    return bo


@pytest.mark.parametrize("strategy", ["bo", "random_search"])
def test_seeded_runs_repeat(strategy, tmp_path):
    a = _run(str(tmp_path / "a"), seed=5, strategy=strategy, sampling_method="sobol")
    b = _run(str(tmp_path / "b"), seed=5, strategy=strategy, sampling_method="sobol")
    assert np.allclose(np.atleast_2d(a.X), np.atleast_2d(b.X))
    assert len(a.Y) == 6 + 4
    rows = open(os.path.join(str(tmp_path / "a"), "optimiser_timing.csv")).read().splitlines()
    assert rows[0].startswith("generation,strategy") and all(strategy in r for r in rows[1:])


def test_bo_respects_geometric_constraint(tmp_path):
    # optimum of f at (0.3, 0.3) is infeasible for x0 + x1 <= 0.2
    geom = lambda X: {"geom_s": np.atleast_2d(X).sum(axis=1)}       # noqa: E731
    s = _settings(str(tmp_path), seed=1, constraints=["geom_s <= 0.2"], count_limit=6)
    f = lambda X: np.sum((np.atleast_2d(X) - 0.3) ** 2, axis=1)       # noqa: E731
    bo = BayesianOptimiser(s, eval_func=lambda X, g: f(X), init_func=lambda X, g: None, geom_func=geom)
    assert bo.constraints == [] and len(bo.geom_specs) == 1           # no GP for it
    bo.optimise(cont=True)
    X = np.atleast_2d(bo.X)
    assert np.all(X.sum(axis=1) <= 0.2 + 1e-9)                         # nothing infeasible was evaluated
    assert abs(bo.X_best.sum() - 0.2) < 0.15                           # pushed onto the constraint


def test_unknown_strategy_or_sampler_is_refused(tmp_path):
    with pytest.raises(ValueError):
        BayesianOptimiser(_settings(str(tmp_path), strategy="grid"), eval_func=None)
    with pytest.raises(ValueError):
        BayesianOptimiser(_settings(str(tmp_path), sampling_method="halton"), eval_func=None)


# ------------------------------------------------------------------ 2D wall geometry
def _modes(n_modes=4):
    from test_edge_modes_2d import naca0012, o_mesh
    from ShapeParameterization.edgeModes2D import EdgeModes2D
    from MeshGeneration.meshFile import load_mesh
    d = tempfile.mkdtemp()
    m0, _ = o_mesh(naca0012(60), n_layers=6)
    mesh = load_mesh(m0.write_file(os.path.join(d, "n.fro"))).mesh2d
    return EdgeModes2D.from_mesh(mesh, [1], n_modes=n_modes)


def test_wall_geometry_2d_baseline_and_signs():
    from ShapeParameterization.geomConstraints2D import WallGeometry2D
    modes = _modes()
    g = WallGeometry2D(modes, n_stations=51)
    z = np.zeros(modes.n_modes)
    G = g(z)
    assert np.allclose(G["geom_envelope"], 0) and np.allclose(G["geom_thickness_ratio"], 1)
    assert np.isclose(G["geom_area_ratio"][0, 0], 1) and np.allclose(G["geom_normal_disp"], 0)
    # a pure outward displacement (every node moves out) satisfies everything and grows the area
    phi = modes.phi
    c = np.linalg.lstsq(phi, np.ones(len(phi)) * g.out_sign * 1e-3, rcond=None)[0]
    G = g(c)
    assert np.isclose(G["geom_area_ratio"][0, 0], 1, atol=0.2) and G["geom_area_ratio"][0, 0] > 1
    assert np.mean(G["geom_envelope"]) > 0
    assert np.mean(g(-c)["geom_envelope"]) < 0


def test_adodg_style_run_never_evaluates_an_infeasible_design():
    """local2DOpt end to end (stand-in solver): geom_envelope >= 0, random search."""
    from test_local2d_opt import make_case, designs
    from Local.local2DOpt import run
    d = tempfile.mkdtemp()
    _, run_dir = make_case(d, bound=0.01, n_modes=4, n_samples=4, count_limit=2,
                        constraints=["geom_envelope >= 0"], strategy="random_search", seed=3,
                        sampling_method="random")
    assert run(run_dir, echo=lambda m: None) == 0
    rows = designs(run_dir)
    assert len(rows) == 1 + 4 + 2
    for r in rows:
        assert r["constraints"]["geom_envelope"] >= -1e-9, r
    assert any(r["constraints"]["geom_envelope"] > 1e-6 for r in rows)   # designs actually moved


# ------------------------------------------------------------------ cost report
SACCT = """\
101|morph_n0_1|COMPLETED|60|1|60|2026-11-06T10:00:00|2026-11-06T10:01:00|2026-11-06T10:02:00
102|vol_n0_1|COMPLETED|600|40|24000|2026-11-06T10:00:01|2026-11-06T10:02:00|2026-11-06T10:12:00
103|pre_n0_1|COMPLETED|120|1|120|2026-11-06T10:00:02|2026-11-06T10:12:00|2026-11-06T10:14:00
104|sol_n0_1_cond_1|COMPLETED|3600|80|288000|2026-11-06T10:00:03|2026-11-06T10:14:00|2026-11-06T11:14:00
105|guard_n0_1_cond_1|COMPLETED|10|1|10|2026-11-06T10:00:04|2026-11-06T11:14:30|2026-11-06T11:14:40
106|solrestart_n0_1_cond_1|TIMEOUT|1800|80|144000|2026-11-06T11:14:35|2026-11-06T11:20:00|2026-11-06T11:50:00
107|sol_n1_1_cond_1|RUNNING|100|80|8000|2026-11-06T12:00:00|2026-11-06T12:00:10|Unknown
108|interactive|COMPLETED|5|1|5|2026-11-06T09:00:00|2026-11-06T09:00:00|2026-11-06T09:00:05
"""


def test_parse_sacct_and_cluster_report(tmp_path):
    from Optimisation.cost_report import parse_sacct, cluster_report
    rows = parse_sacct(SACCT)
    assert len(rows) == 7 and {r["stage"] for r in rows} == {"morph", "vol", "pre", "sol", "guard", "solrestart"}
    assert rows[3]["core_h"] == 80.0 and rows[3]["queue_s"] == 14 * 60 - 3
    root = tmp_path / "root"
    run = root / "opt"
    st = root / "solutions" / "n_0" / "cond_1" / "1"
    st.mkdir(parents=True)
    run.mkdir()
    (st / "convergence_state.json").write_text(json.dumps({"attempt": 1, "converged": False}))
    (run / "optimiser_timing.csv").write_text("generation,strategy,gp_fit_s,acquisition_s,n_evaluated\n"
                                              "0,bo,1.5,2.5,6\n")
    s = cluster_report(str(run), str(root), sacct_text=SACCT)
    assert s["designs"] == 2 and s["solver_restarts"] == 1
    assert abs(s["cfd_core_h"] - (80 + 40 + 8000 / 3600)) < 1e-3
    assert s["gp_fit_s"] == 1.5 and s["acquisition_s"] == 2.5
    per = list(csv.DictReader(open(run / "cost_per_design.csv")))
    assert per[0]["restarts"] == "1" and float(per[0]["total_core_h"]) > float(per[0]["cfd_core_h"])


def test_local2d_cost_written_during_run():
    from test_local2d_opt import make_case
    from Local.local2DOpt import run
    d = tempfile.mkdtemp()
    _, run_dir = make_case(d, n_samples=3, count_limit=1)
    assert run(run_dir, echo=lambda m: None) == 0
    s = json.load(open(os.path.join(run_dir, "cost_summary.json")))
    assert s["designs"] == 1 + 3 + 1 and s["cfd_core_h"] > 0 and s["optimiser_steps"] >= 1
