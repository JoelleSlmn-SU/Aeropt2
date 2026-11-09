"""Cluster convergence guard: toggle, decision logic, and the guard batch script itself
(run under bash with stand-in makeplot / splitplot / sbatch)."""
import json
import os
import stat
import subprocess
import sys
import types

import pytest

from Remote import convergenceGuard as cg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- decide()
def _inp(tmp_path, n=0):
    p = tmp_path / "model_1.inp"
    p.write_text(f"&inputVariables\n ivd%restartNumber = {n},\n ivd%alpha = 2.0,\n/\n")
    return p


def _check(results):
    it = iter(results)
    return lambda *a, **k: next(it)


def test_converged_first_time(tmp_path):
    _inp(tmp_path)
    a = cg.decide(str(tmp_path), "model_1.inp", "model_1.rsd", "out", -3, 2,
                  check=_check([{"converged": True, "reason": "ok"}]))
    st = json.loads((tmp_path / cg.STATE).read_text())
    assert a == "done" and st["converged"] and st["final"] and st["attempt"] == 0


def test_restarts_then_gives_up(tmp_path):
    inp = _inp(tmp_path)
    no = {"converged": False, "reason": "final residual -2.1"}
    actions = [cg.decide(str(tmp_path), "model_1.inp", "r", "o", -3, 2, check=_check([no])) for _ in range(3)]
    assert actions == ["restart", "restart", "done"]
    st = json.loads((tmp_path / cg.STATE).read_text())
    assert st["attempt"] == 2 and st["final"] and not st["converged"] and len(st["history"]) == 3
    assert "ivd%restartNumber = 2," in inp.read_text()          # bumped in the solver's own file


def test_crash_is_not_restarted(tmp_path):
    _inp(tmp_path)
    (tmp_path / "SOLVER_RC").write_text("137\n")
    a = cg.decide(str(tmp_path), "model_1.inp", "r", "o", -3, 2,
                  check=_check([{"converged": False, "reason": "no data"}]))
    assert a == "done" and "code 137" in json.loads((tmp_path / cg.STATE).read_text())["reason"]


def test_missing_restart_line_ends_run(tmp_path):
    (tmp_path / "model_1.inp").write_text("&inputVariables\n/\n")
    a = cg.decide(str(tmp_path), "model_1.inp", "r", "o", -3, 2,
                  check=_check([{"converged": False, "reason": "x"}]))
    assert a == "done"


def test_real_check_on_rsd(tmp_path):
    """End to end through convergenceCheck.py on a residual file."""
    _inp(tmp_path)
    rows = "\n".join(f"{i} {-0.01 * i:.4f} 0.3 0.02" for i in range(1, 401))  # reaches -4.0
    (tmp_path / "model_1.rsd").write_text(rows + "\n")
    a = cg.decide(str(tmp_path), "model_1.inp", "model_1.rsd", "solver_output", -3, 2)
    assert a == "done" and json.loads((tmp_path / cg.STATE).read_text())["converged"]


# ---------------------------------------------------------------- pipeline
@pytest.fixture
def pipe_factory(tmp_path, monkeypatch):
    import Remote.pipeline_cluster as pc
    calls = []
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if not (isinstance(cmd, str) and cmd.startswith("sbatch")):
            return real_run(cmd, *args, **kwargs)     # pc.subprocess IS the global module
        calls.append(cmd)
        return types.SimpleNamespace(stdout=f"Submitted batch job {100 + len(calls)}\n", stderr="", returncode=0)

    monkeypatch.setattr(pc.subprocess, "run", fake_run)

    def make(check):
        cfg = {"remote_output": str(tmp_path), "base_name": "model", "convergence_check": check,
               "convergence_max_retries": 3, "convergence_res_threshold": -4.0,
               "makeplot_exe": "makeplot", "splitplot_exe": "splitplot", "combine_exe": "makeplot"}
        return pc.ClusterPipelineManager(cfg, gen=0, n=1), calls
    return make


def _sol(tmp_path):
    return tmp_path / "solutions" / "n_0" / "cond_1" / "1"


def test_check_off_submits_no_guard(tmp_path, pipe_factory):
    pipe, calls = pipe_factory(False)
    pipe.solver({"Mach": 0.85, "AoA": 0.0}, nc=1)
    sol = _sol(tmp_path)
    assert len(calls) == 1 and "guard" not in calls[0]
    assert not list(sol.glob("batchfile_guard*")) and not list(sol.glob("batchfile_solrestart*"))
    txt = (sol / "batchfile_sol_n0_1_cond_1").read_text()
    assert txt.rstrip().endswith("touch SOLVER_DONE")


def test_check_on_writes_consistent_guard(tmp_path, pipe_factory):
    pipe, calls = pipe_factory(True)
    pipe.solver({"Mach": 0.85, "AoA": 0.0}, nc=1)
    sol = _sol(tmp_path)
    assert len(calls) == 2 and "--dependency=afterany:101" in calls[1]
    first = (sol / "batchfile_sol_n0_1_cond_1").read_text()
    restart = (sol / "batchfile_solrestart_n0_1_cond_1").read_text()
    guard = (sol / "batchfile_guard_n0_1_cond_1").read_text()
    assert first.rstrip().endswith("touch SOLVER_RUN_DONE") and "SOLVER_DONE\n" not in first.replace("rm -f SOLVER_DONE", "")
    assert "/orig/" not in restart and "interpu" not in restart.lower() and "model_1.inp" in restart and restart.rstrip().endswith("touch SOLVER_RUN_DONE")
    for name in ("model_1.inp", "model_1.rsd", "model_1.res", "model_1.unk", "model_1.rst",
                 "batchfile_solrestart_n0_1_cond_1", "batchfile_guard_n0_1_cond_1",
                 "--max-retries 3", "--threshold -4.0"):
        assert name in guard, name
    assert (sol / "convergenceGuard.py").exists() and (sol / "convergenceCheck.py").exists()


def _stub(bindir, name, body):
    p = bindir / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


@pytest.mark.parametrize("converged", [False, True])
def test_guard_script_runs(tmp_path, pipe_factory, converged):
    """Execute the generated guard body with stand-ins on PATH."""
    pipe, _ = pipe_factory(True)
    pipe.solver({"Mach": 0.85}, nc=1)
    sol = _sol(tmp_path)
    (sol / "model_1.inp").write_text(" ivd%restartNumber = 0,\n")
    (sol / "model_1.res").write_text("res")
    (sol / "SOLVER_RC").write_text("0")
    final = -4.5 if converged else -2.0
    (sol / "model_1.rsd").write_text("\n".join(f"{i} {final} 0.3 0.02" for i in range(1, 20)) + "\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "sbatch.log"
    _stub(bindir, "makeplot", 'read a; read b; read c; echo unk > "$c"\n')
    _stub(bindir, "splitplot", 'read a; read b; read c; echo rst > "$c"\n')
    _stub(bindir, "sbatch", f'echo "$@" >> {log}; echo $((RANDOM+1))\n')
    _stub(bindir, "python3", f'exec {sys.executable} "$@"\n')
    body = [l for l in (sol / "batchfile_guard_n0_1_cond_1").read_text().splitlines()
            if not l.startswith(("#", "module", "source", "conda"))]
    script = tmp_path / "guard.sh"
    script.write_text("\n".join(body) + "\n")
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, cwd=str(sol))
    assert r.returncode == 0, r.stderr
    st = json.loads((sol / "convergence_state.json").read_text())
    if converged:
        assert (sol / "SOLVER_DONE").exists() and st["converged"] and not log.exists()
    else:
        assert not (sol / "SOLVER_DONE").exists() and st["attempt"] == 1
        assert (sol / "model_1.rst").exists()
        sub = log.read_text().splitlines()
        assert "batchfile_solrestart_n0_1_cond_1" in sub[0]
        assert "--dependency=afterany:" in sub[1] and "batchfile_guard_n0_1_cond_1" in sub[1]


def test_guard_ends_run_when_restart_cannot_be_prepared(tmp_path, pipe_factory):
    pipe, _ = pipe_factory(True)
    pipe.solver({"Mach": 0.85}, nc=1)
    sol = _sol(tmp_path)
    (sol / "model_1.inp").write_text(" ivd%restartNumber = 0,\n")
    (sol / "model_1.rsd").write_text("1 -1.0 0.3 0.02\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _stub(bindir, "makeplot", "exit 1\n")
    _stub(bindir, "splitplot", "exit 1\n")
    _stub(bindir, "sbatch", "exit 1\n")
    _stub(bindir, "python3", f'exec {sys.executable} "$@"\n')
    body = [l for l in (sol / "batchfile_guard_n0_1_cond_1").read_text().splitlines()
            if not l.startswith(("#", "module", "source", "conda"))]
    script = tmp_path / "guard.sh"
    script.write_text("\n".join(body) + "\n")
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, cwd=str(sol))
    assert r.returncode == 0, r.stderr
    st = json.loads((sol / "convergence_state.json").read_text())
    assert (sol / "SOLVER_DONE").exists() and st["final"] and "restart aborted" in st["reason"]


# ---------------------------------------------------------------- remoteOpt side
def test_apply_convergence_state(tmp_path):
    from Remote.remoteOpt import apply_convergence_state
    (tmp_path / "convergence_state.json").write_text(json.dumps({"converged": False, "attempt": 2}))
    m = {"CL": 0.5, "CD": 0.02}
    assert apply_convergence_state(str(tmp_path), m, {}) == m                       # check off
    acc = apply_convergence_state(str(tmp_path), m, {"convergence_check": True})
    assert acc["CD"] == 0.02 and acc["converged"] == 0.0 and acc["n_restarts"] == 2.0
    rej = apply_convergence_state(str(tmp_path), m, {"convergence_check": True, "convergence_on_fail": "reject"})
    assert rej["CD"] != rej["CD"] and rej["converged"] == 0.0
