"""
local2DOpt.py

Bayesian optimisation of a 2D case on THIS machine: the local counterpart of
Remote/remoteOpt.py. Same inputs (a run folder holding bo_settings.json and
objective.json), same optimiser (Optimisation/BayesianOptimisation), same
objective and constraint expressions, same treatment of failed designs. What
differs is how a design is evaluated: no Slurm, no waiting for markers - each
design is morphed, preprocessed and solved here, one after the other.

    python Local/local2DOpt.py <run_dir>             run, or resume, the optimisation
    python Local/local2DOpt.py --check <run_dir>     check the set-up only; no CFD

How one design x (the edge-mode coefficients) is evaluated
----------------------------------------------------------
    morph    baseline mesh + saved edge modes (Control Nodes/edge_modes_2d.json)
             -> volumes/n_<n>/<base>_<n>.dat          (MeshGeneration.Morph2D)
    prepro   Solver/PrePro.exe                        (Local.flite2d_local)
    solver   Solver/Solver.exe, once per flow condition
    metrics  last row of the .rsd -> CL, CD, CM       (per flow condition)
    objective = sum over conditions of Weight x expression(metrics)

Design numbers
--------------
Every design gets its own number n, counted up in the order the optimiser asks
for it, and all its files live under ...\\n_<n> exactly as for a single
simulation. n = 0 is the baseline (x = 0): it is evaluated together with the
initial Latin-hypercube sample and given to the optimiser as a training point,
so every result can be compared with it ("include_baseline": false turns this
off). <run_dir>/designs.csv is the table of all designs.

Designs that give no number
---------------------------
  * the morph produces an invalid mesh  -> nothing is run       ("invalid_mesh")
  * PrePro fails                                                ("prepro_failed")
  * the solver diverges, crashes, writes nothing or hits the time limit
The objective of such a design is NaN and is then replaced, exactly as in the
3D chain, by a deliberately bad finite value (worst value seen + 10 % of the
spread) so the optimiser moves away from it. Only solver runs whose status is
in "accept_statuses" (default: ["ok"]) count as results.
If the BASELINE gives no result, or no design of the initial sample does, the
run ends there with an error: the set-up is wrong, not the designs.

Stop and resume
---------------
Creating the file <run_dir>/STOP (the GUI's Stop button does this) ends the
running program and the optimisation. Nothing is lost: running the same
command again continues from bo_data.mcsv, and every design that had finished
is taken from designs.json instead of being computed again. The design that
was running when the stop came is computed again from the start.

A resumed run must be the SAME problem. The objective, constraints, flow
conditions, solver control file, preprocessor options and morph basis are
fingerprinted when the run starts; if any of them has changed since, the run
refuses to continue (mixing results of two different problems in one surrogate
model is an error nobody would see afterwards).

Extra keys read from bo_settings.json (all optional)
----------------------------------------------------
    output_dir             case folder holding volumes/, solutions/ ... (default: parent of run_dir)
    base_name              file stem of the case (default: "model")
    morph_basis_json       default <output_dir>/Control Nodes/morph_basis.json
    solver_control_file    default <output_dir>/<base_name>.inp
    prepro                 {"n_grids": .., "hybrid": .., "directionality": .., "n_directional": ..}
                           (default: <output_dir>/flite2d_settings.json, else PrePro defaults)
    solver_time_limit_min  per flow condition (default: flite2d_settings.json, else 60)
    solver_dir             folder of PrePro.exe / Solver.exe (default <repo>/Solver)
    include_baseline       true
    accept_statuses        ["ok"]
    make_plots             true   (pictures per design, see Local.flite2d_post)
    keep_sol               false  (the .sol of a finished design is deleted: it is large and
                                   can be regenerated from the .dat)
    bounds_check_samples   100    (designs morphed at start to report how much of the box is usable)
    seed                   none   (numpy seed for the initial sample)
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")      # the optimiser imports pyplot; never open a window

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Local.flite2d_local import (Flite2DCase, PreProOptions, RunStopped, apply_condition,      # noqa: E402
                                 check_control, find_executables, get_namelist_value, _as_float)

STOP_FILE = "STOP"
# what a 2D run can deliver to an objective or constraint expression
SYMBOLS_2D = ("CL", "CD", "CM", "CL_over_CD", "min_area_ratio", "max_wall_displacement")
FINAL_STATUSES = ("ok", "diverged", "crashed", "failed", "timeout", "invalid_mesh", "prepro_failed")
NAN = float("nan")


class OptAbort(RuntimeError):
    """The optimisation cannot sensibly continue (e.g. the baseline itself gives no result)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
class OptLog:
    """Lines go to stdout (the GUI reads them) and, time-stamped, to <run_dir>/local_opt.log."""

    def __init__(self, run_dir: Optional[str] = None, echo: Optional[Callable[[str], None]] = None):
        self.path = os.path.join(run_dir, "local_opt.log") if run_dir else None
        self.echo = echo
        self.out = sys.stdout             # kept: sys.stdout is diverted while the optimiser runs

    def __call__(self, msg: str):
        msg = str(msg)
        if self.echo is not None:
            self.echo(msg)
        else:
            try:
                print(msg, file=self.out, flush=True)
            except UnicodeEncodeError:
                print(msg.encode("ascii", "replace").decode(), file=self.out, flush=True)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
            except OSError:
                pass


class _PrintsToFile:
    """
    While active, everything other modules print() goes to a file instead of
    the console. The optimiser and its GP print the kernel and the acquisition
    result every generation; that is worth keeping but would bury the log.
    """

    def __init__(self, path: str):
        self.path, self._old, self._fh = path, None, None

    def __enter__(self):
        try:
            self._fh = open(self.path, "a", encoding="utf-8", errors="replace")
        except OSError:
            return self
        self._old, sys.stdout = sys.stdout, self._fh
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            sys.stdout = self._old
            self._fh.close()
        return False


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def tail_range(values, frac: float = 0.2, min_rows: int = 5) -> float:
    """Largest minus smallest value over the last `frac` of a history (NaN if too short)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < min_rows:
        return NAN
    w = max(min_rows, int(math.ceil(frac * len(v))))
    return float(np.ptp(v[-w:]))


# ---------------------------------------------------------------------------
# design x -> solver mesh
# ---------------------------------------------------------------------------
class Design2DBuilder:
    """
    The baseline mesh and the saved edge modes; turns a design vector into a
    solver mesh. The modes are TAKEN FROM THE FILE the GUI saved (not rebuilt),
    so a coefficient means the same shape here as in the GUI where the bounds
    were set; from_saved also checks that the mesh is the one they belong to.
    """

    def __init__(self, morph_basis_json: str, mesh=None, log: Callable[[str], None] = print):
        from FileRW.FliteMesh2D import flags_from_basis
        from MeshGeneration.Morph2D import Morph2D
        from ShapeParameterization.edgeModes2D import EdgeModes2D
        self.basis_path = os.path.abspath(morph_basis_json)
        self.basis = _read_json(self.basis_path)
        b = self.basis
        if int(b.get("dim", 3)) != 2 or b.get("parameterisation_method") != "edge_modes_2d":
            raise ValueError(f"{self.basis_path} is not a 2D edge-mode basis; save the 2D basis in the GUI first.")
        modes_path = os.path.join(os.path.dirname(self.basis_path), b.get("edge_modes_file") or "edge_modes_2d.json")
        if not os.path.isfile(modes_path):
            raise FileNotFoundError(f"Saved edge modes not found: {modes_path}. Re-save the 2D basis in the GUI.")
        self.modes_path = modes_path
        if mesh is None:
            path = b.get("baseline_mesh_path") or ""
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Baseline mesh not found: {path!r} (from {self.basis_path}). "
                                        "It was moved or renamed since the basis was saved; re-save the basis.")
            from MeshGeneration.meshFile import load_mesh
            t0 = time.time()
            mesh = load_mesh(path, b.get("dat_path") or None).mesh2d
            log(f"[LOCAL-OPT] Baseline mesh {os.path.basename(path)}: {mesh.node_count} nodes "
                f"({time.time() - t0:.1f} s)")
        self.mesh = mesh
        modes = EdgeModes2D.from_saved(mesh, modes_path)
        self.morph = Morph2D(mesh, b["t_edges"], b.get("u_edges") or [],
                             support_frac=float(b.get("support_frac", 1.0)),
                             min_area_ratio=float(b.get("min_area_ratio", 0.05)), modes=modes)
        self.flags = flags_from_basis(b)
        if set(self.flags) != set(int(e) for e in mesh.edges):
            raise ValueError(f"the solver boundary types saved in {self.basis_path} are for edges "
                             f"{sorted(self.flags)}, but the mesh has edges {sorted(int(e) for e in mesh.edges)}. "
                             "Re-save the 2D basis in the GUI.")
        self.scale = float(b.get("solver_mesh_scale", 1.0) or 1.0)
        self.wall_length = float(modes.length)

    @property
    def n_modes(self) -> int:
        return int(self.morph.n_modes)

    def wall_displacement(self, x) -> float:
        """Largest displacement of a T node for design x (mesh units)."""
        d = np.asarray(self.morph.modes.displacement(np.asarray(x, float)), float)
        return float(np.linalg.norm(d[:, :2], axis=1).max()) if len(d) else 0.0

    def morphed(self, x):
        """(morphed EdgeMesh2D, MorphReport) for design x."""
        return self.morph.morph(np.asarray(x, float))

    def write(self, mesh, path: str):
        from FileRW.FliteMesh2D import write_flite2d_mesh
        return write_flite2d_mesh(mesh, path, self.flags, scale=self.scale)

    def box_report(self, lb, ub, n_samples: int = 100, seed: int = 12345) -> dict:
        """
        Morph `n_samples` Latin-hypercube designs of the box [lb, ub] (no CFD):
        the share that gives a valid mesh and the wall displacements they imply.
        The sample is seeded separately, so it does not disturb the optimiser's.
        """
        from Utilities.lhs import lhs
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            X = lhs(len(lb), samples=int(n_samples), lb=np.asarray(lb, float), ub=np.asarray(ub, float))
        finally:
            np.random.set_state(state)
        ok, disp, reasons = [], [], {}
        for x in np.atleast_2d(X):
            rep = self.morphed(x)[1]
            ok.append(bool(rep.ok))
            disp.append(self.wall_displacement(x))
            if not rep.ok:
                key = "inverted triangles" if rep.n_inverted else (rep.reason or "quality gate")
                reasons[key] = reasons.get(key, 0) + 1
        return {"n": len(ok), "valid_fraction": float(np.mean(ok)), "reasons": reasons,
                "wall_disp_median": float(np.median(disp)), "wall_disp_max": float(np.max(disp)),
                "wall_length": self.wall_length}


# ---------------------------------------------------------------------------
# the test manager (same two calls as Remote.remoteOpt.ClusterTestManager)
# ---------------------------------------------------------------------------
class Local2DTestManager:
    """
    init_generation(X, gen, conds)     registers the designs (numbers, x)
    evaluate_generation(X, gen, conds) runs them one after the other and returns,
                                       per design, one metrics dict per flow condition
    """

    def __init__(self, output_dir: str, base_name: str, builder: Design2DBuilder, control_text: str,
                 prepro_options: PreProOptions, run_dir: str, solver_time_limit_s: Optional[float] = 3600.0,
                 solver_dir: Optional[str] = None, log: Callable[[str], None] = print,
                 should_stop: Optional[Callable[[], bool]] = None, accept_statuses: Sequence[str] = ("ok",),
                 make_plots: bool = True, keep_sol: bool = False, prepro_cmd=None, solver_cmd=None,
                 fingerprint: str = "", x_tol=None):
        """x_tol: per-variable tolerance when a design is recognised again after a resume
        (the optimiser's data file keeps 8 decimals). Default 1e-4 of |x| + 1e-7."""
        self.output_dir = os.path.abspath(output_dir)
        self.base_name, self.builder = str(base_name), builder
        self.control_text, self.opts = control_text, prepro_options
        self.run_dir = os.path.abspath(run_dir)
        self.time_limit_s, self.solver_dir = solver_time_limit_s, solver_dir
        self.log, self.should_stop = log, (should_stop or (lambda: False))
        self.accept = tuple(accept_statuses)
        self.make_plots, self.keep_sol = bool(make_plots), bool(keep_sol)
        self.prepro_cmd, self.solver_cmd = prepro_cmd, solver_cmd
        self.x_tol = None if x_tol is None else np.asarray(x_tol, float)
        self._said = set()
        self.registry_path = os.path.join(self.run_dir, "designs.json")
        self.csv_path = os.path.join(self.run_dir, "designs.csv")
        os.makedirs(self.run_dir, exist_ok=True)
        self.fingerprint = fingerprint
        self.designs: List[dict] = []
        if os.path.isfile(self.registry_path):
            reg = _read_json(self.registry_path)
            if fingerprint and reg.get("fingerprint") and reg["fingerprint"] != fingerprint:
                raise ValueError(
                    "this run folder holds designs of a DIFFERENT problem (the objective, constraints, flow "
                    "conditions, solver file, preprocessor options or morph basis have changed since it was "
                    "started). Their results cannot be mixed with new ones: start a new optimisation.")
            self.designs = list(reg.get("designs", []))

    # -- registry ---------------------------------------------------------------
    def _save(self):
        _write_json(self.registry_path, {"fingerprint": self.fingerprint, "designs": self.designs})
        self.write_csv()

    def _find(self, gen: int, index: int) -> Optional[dict]:
        for d in self.designs:
            if d["gen"] == int(gen) and d["index"] == int(index):
                return d
        return None

    def _register(self, gen: int, index: int, x) -> dict:
        x = [float(v) for v in np.asarray(x, float).ravel()]
        rec = self._find(gen, index)
        if rec is not None:
            # (generation, index) identifies the design; x is only a cross-check. bo_data.mcsv
            # keeps 8 decimals, so the x read back after a save differs slightly from the original.
            tol = self.x_tol if self.x_tol is not None else 1e-4 * np.abs(np.asarray(rec["x"], float)) + 1e-7
            if len(rec["x"]) == len(x) and np.all(np.abs(np.asarray(rec["x"], float) - np.asarray(x, float)) <= tol):
                return rec
            self.log(f"[LOCAL-OPT][WARN] generation {gen} design {index}: x differs from the one recorded "
                     f"for n={rec['n']}; it is computed again.")
            rec.update(x=x, done=False, status="pending", metrics=None, objective=None, objective_used=None)
            return rec
        used = {d["n"] for d in self.designs}
        is_baseline = all(v == 0.0 for v in x)
        n = 0 if (is_baseline and 0 not in used) else max([0] + list(used)) + 1
        rec = {"n": n, "gen": int(gen), "index": int(index), "x": x, "done": False, "status": "pending",
               "baseline": bool(is_baseline and n == 0), "metrics": None, "objective": None,
               "objective_used": None, "morph": None, "wall_time_s": None}
        self.designs.append(rec)
        return rec

    def init_generation(self, X_list, gen_num: int, conds: List[dict]) -> None:
        recs = [self._register(gen_num, i, x) for i, x in enumerate(np.atleast_2d(np.asarray(X_list, float)))]
        self._save()
        self.log(f"[LOCAL-OPT] Generation {gen_num}: {len(recs)} design(s) -> n = {[r['n'] for r in recs]}")

    # -- one design ---------------------------------------------------------------
    def _nan_metrics(self, status: str, rep=None, n_conds: int = 1) -> List[dict]:
        m = {"CL": NAN, "CD": NAN, "CM": NAN, "CL_over_CD": NAN, "status": status, "iterations": 0,
             "residual": NAN, "CL_tail_range": NAN, "CD_tail_range": NAN,
             "min_area_ratio": float(rep.min_area_ratio) if rep is not None else NAN}
        return [dict(m, cond=i) for i in range(1, n_conds + 1)]

    def _cond_metrics(self, case: Flite2DCase, r: dict, rep, wall_disp: float) -> dict:
        from Local.flite2d_post import read_rsd_history
        usable = r.get("status") in self.accept
        H = read_rsd_history(os.path.join(case.cond_dir(r["cond"]), case.stem + ".rsd"))
        cl, cd, cm = (float(r.get(k, NAN)) for k in ("CL", "CD", "CM"))
        return {
            "CL": cl if usable else NAN, "CD": cd if usable else NAN, "CM": cm if usable else NAN,
            "CL_over_CD": (cl / cd) if (usable and _finite(cl) and _finite(cd) and cd != 0.0) else NAN,
            "status": r.get("status"), "cond": int(r["cond"]), "iterations": int(r.get("iterations", 0)),
            "residual": float(r.get("residual", NAN)),
            # how much the forces were still moving at the end: the numerical
            # resolution of this design's objective
            "CL_tail_range": tail_range(H[:, 2]) if len(H) else NAN,
            "CD_tail_range": tail_range(H[:, 3]) if len(H) else NAN,
            "min_area_ratio": float(rep.min_area_ratio), "max_wall_displacement": float(wall_disp),
        }

    def _case_log(self, msg: str):
        """The runner repeats its set-up warnings for every design; show each one once."""
        if msg.startswith("[2D][WARN]"):
            if msg in self._said:
                return
            self._said.add(msg)
        self.log(msg)

    def _run_design(self, rec: dict, conds: List[dict]) -> List[dict]:
        n, x = rec["n"], rec["x"]
        case = Flite2DCase(self.output_dir, self.base_name, n=n, solver_dir=self.solver_dir, log=self._case_log,
                           prepro_cmd=self.prepro_cmd, solver_cmd=self.solver_cmd, make_plots=self.make_plots)
        _write_json_dir = os.path.join(self.output_dir, "volumes", f"n_{n}")
        os.makedirs(_write_json_dir, exist_ok=True)
        mesh, rep = self.builder.morphed(x)
        wall_disp = self.builder.wall_displacement(x)
        rec["morph"] = {"ok": bool(rep.ok), "reason": rep.reason, "n_inverted": int(rep.n_inverted),
                        "min_area_ratio": float(rep.min_area_ratio), "max_displacement": float(rep.max_disp),
                        "max_wall_displacement": wall_disp}
        _write_json(os.path.join(_write_json_dir, "design.json"),
                    {"n": n, "generation": rec["gen"], "x": x, "morph": rec["morph"]})
        if not rep.ok:
            self.log(f"[LOCAL-OPT][WARN] n={n}: the morph gives an invalid mesh ({rep.reason or 'quality gate'}; "
                     f"min area ratio {rep.min_area_ratio:.3g}). Nothing is run for this design.")
            return self._nan_metrics("invalid_mesh", rep, len(conds))
        try:
            info = self.builder.write(mesh, case.mesh_path)
            self.log(f"[LOCAL-OPT] n={n}: mesh written ({info}); largest wall displacement {wall_disp:.4g}, "
                     f"min area ratio {rep.min_area_ratio:.3f}")
        except ValueError as e:
            self.log(f"[LOCAL-OPT][WARN] n={n}: the solver mesh could not be written ({e}). Nothing is run.")
            return self._nan_metrics("invalid_mesh", rep, len(conds))
        try:
            results = case.run(self.opts, self.control_text, conds, self.builder.flags.values(),
                               solver_time_limit_s=self.time_limit_s, should_stop=self.should_stop)
        except RunStopped:
            raise
        except RuntimeError as e:                    # PrePro crashed, wrote nothing or timed out
            self.log(f"[LOCAL-OPT][WARN] n={n}: preprocessor failed; the design gets no result.\n{e}")
            return self._nan_metrics("prepro_failed", rep, len(conds))
        finally:
            if not self.keep_sol:
                try:
                    os.remove(case.sol_path)
                except OSError:
                    pass
        if len(results) < len(conds) or any(r.get("status") == "stopped" for r in results) or self.should_stop():
            raise RunStopped(f"stopped while design n={n} was running.")
        return [self._cond_metrics(case, r, rep, wall_disp) for r in results]

    def evaluate_generation(self, X_list, gen_num: int, conds: List[dict]) -> List[List[dict]]:
        out = []
        X = np.atleast_2d(np.asarray(X_list, float))
        for i, x in enumerate(X):
            rec = self._register(gen_num, i, x)
            if rec.get("done") and rec.get("status") in FINAL_STATUSES and rec.get("metrics"):
                self.log(f"[LOCAL-OPT] n={rec['n']} (generation {gen_num}): already computed, result reused.")
                out.append(rec["metrics"])
                continue
            if self.should_stop():
                raise RunStopped("stopped between designs.")
            self.log(f"[LOCAL-OPT] ---- design n={rec['n']} (generation {gen_num}, {i + 1} of {len(X)})"
                     f"{' - BASELINE' if rec.get('baseline') else ''}: x = "
                     f"[{', '.join(f'{v:.5g}' for v in rec['x'])}]")
            t0 = time.time()
            rec["x"] = [float(v) for v in x]          # exactly the x the optimiser holds for this design
            metrics = self._run_design(rec, conds)
            statuses = [m["status"] for m in metrics]
            rec.update(metrics=metrics, done=True, wall_time_s=round(time.time() - t0, 1),
                       status=("ok" if all(s in self.accept for s in statuses) else
                               next(s for s in statuses if s not in self.accept)))
            self._save()
            out.append(metrics)
            if rec.get("baseline") and rec["status"] != "ok":
                # if the unmodified geometry cannot be computed, the set-up is wrong;
                # running fifty more designs would only produce fifty penalties
                raise OptAbort(f"the baseline (n=0) gave no result: {rec['status']}. Fix the case first "
                               "(run it once with 'Run Simulation' and look at solutions/n_0), then start again.")
        return out

    # -- results ------------------------------------------------------------------
    def record_objectives(self, gen_num: int, raw, used, constraints: Optional[Dict[str, Sequence[float]]] = None,
                          feasible: Optional[Sequence[bool]] = None):
        """raw: objective per design (NaN = no result); used: what the optimiser was given;
        constraints: {name: observed values, NaN = no result}; feasible: per design."""
        for i, (y, yu) in enumerate(zip(raw, used)):
            rec = self._find(gen_num, i)
            if rec is None:
                continue
            rec["objective"] = float(y) if _finite(y) else None
            rec["objective_used"] = float(yu)
            if constraints:
                rec["constraints"] = {k: (float(v[i]) if _finite(v[i]) else None) for k, v in constraints.items()}
            if feasible is not None:
                rec["feasible"] = bool(feasible[i])
        self._save()

    def baseline(self) -> Optional[dict]:
        return next((d for d in self.designs if d.get("baseline")), None)

    def best(self) -> Optional[dict]:
        """The design with the smallest objective among those with a result that meet the constraints."""
        good = [d for d in self.designs if d.get("objective") is not None and d.get("feasible", True)]
        return min(good, key=lambda d: d["objective"]) if good else None

    def write_csv(self) -> None:
        n_x = max([len(d["x"]) for d in self.designs] or [0])
        n_c = max([len(d["metrics"] or []) for d in self.designs] or [0])
        cons = sorted({k for d in self.designs for k in (d.get("constraints") or {})})
        head = (["n", "generation", "status", "objective", "objective_used_by_optimiser"]
                + [f"constraint: {c}" for c in cons] + [f"x{k + 1}" for k in range(n_x)]
                + ["max_wall_displacement", "min_area_ratio", "wall_time_s"])
        for c in range(1, n_c + 1):
            head += [f"cond{c}_{k}" for k in ("status", "CL", "CD", "CM", "iterations", "residual",
                                                "CL_tail_range", "CD_tail_range")]
        def num(v):                                   # blank for "no value", 8 significant digits otherwise
            return f"{float(v):.8g}" if _finite(v) else ""
        tmp = self.csv_path + ".tmp"
        try:
            with open(tmp, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(head)
                for d in sorted(self.designs, key=lambda d: d["n"]):
                    mo = d.get("morph") or {}
                    row = [d["n"], d["gen"], d["status"],
                           "" if d.get("objective") is None else f"{d['objective']:.10g}",
                           "" if d.get("objective_used") is None else f"{d['objective_used']:.10g}"]
                    row += ["" if (d.get("constraints") or {}).get(c) is None else f"{d['constraints'][c]:.10g}"
                            for c in cons]
                    row += [f"{v:.10g}" for v in d["x"]] + [""] * (n_x - len(d["x"]))
                    row += [num(mo.get("max_wall_displacement")), num(mo.get("min_area_ratio")),
                            d.get("wall_time_s") or ""]
                    for c in range(n_c):
                        m = (d["metrics"] or [])[c] if c < len(d["metrics"] or []) else {}
                        row += [m.get("status", "")] + [num(m.get(k)) for k in ("CL", "CD", "CM")]
                        row += [m.get("iterations", "")] + [num(m.get(k)) for k in ("residual", "CL_tail_range",
                                                                                     "CD_tail_range")]
                    w.writerow(row)
            os.replace(tmp, self.csv_path)
        except OSError as e:                          # e.g. the table is open in Excel
            self.log(f"[LOCAL-OPT][WARN] designs.csv not updated ({e}); designs.json is up to date.")


# ---------------------------------------------------------------------------
# set-up: everything that can be checked before the first design is run
# ---------------------------------------------------------------------------
def _objective_problems(objective: dict, constraints: Sequence) -> List[str]:
    """Symbols the 2D chain cannot deliver, and monitor terms (not available in 2D)."""
    from Remote.remoteOpt import _expression_symbols
    from Optimisation.BayesianOptimisation.optimiser import _parse_constraint
    problems = []
    monitor = [t.get("symbol") or t.get("metric") for t in (objective.get("terms") or [])
               if str(t.get("source", "")).lower() == "monitor"]
    if monitor:
        problems.append(f"objective term(s) {monitor} come from surface monitors, which the 2D chain does not "
                        f"have. Available in 2D: {', '.join(SYMBOLS_2D)}.")
    expr = (objective.get("expression") or "").strip()
    if expr and expr.lower() not in ("drag", "lift", "lift-to-drag", "lift to drag"):
        missing = _expression_symbols(expr) - set(SYMBOLS_2D)
        if missing:
            problems.append(f"expression '{expr}' uses {sorted(missing)}; a 2D run provides {list(SYMBOLS_2D)}.")
    for c in constraints or []:
        missing = _expression_symbols(_parse_constraint(c)["metric"]) - set(SYMBOLS_2D)
        if missing:
            problems.append(f"constraint '{c}' uses {sorted(missing)}; a 2D run provides {list(SYMBOLS_2D)}.")
    return problems


class Setup:
    """Everything read from <run_dir>; raises ValueError / FileNotFoundError with a message for the user."""

    def __init__(self, run_dir: str, log: Callable[[str], None], mesh=None):
        self.run_dir = os.path.abspath(run_dir)
        s_path, o_path = os.path.join(self.run_dir, "bo_settings.json"), os.path.join(self.run_dir, "objective.json")
        for p in (s_path, o_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"not found: {p}")
        self.settings, self.objective = _read_json(s_path), _read_json(o_path)
        s = self.settings
        self.output_dir = os.path.abspath(s.get("output_dir") or os.path.dirname(self.run_dir))
        self.base_name = s.get("base_name", "model")
        case_cfg = {}
        cfg_path = os.path.join(self.output_dir, "flite2d_settings.json")
        if os.path.isfile(cfg_path):
            case_cfg = _read_json(cfg_path)
        self.opts = PreProOptions.from_dict(s.get("prepro") or case_cfg.get("prepro"))
        limit_min = s.get("solver_time_limit_min", case_cfg.get("solver_time_limit_min", 60))
        self.time_limit_s = 60.0 * float(limit_min) if limit_min else None
        control = s.get("solver_control_file") or os.path.join(self.output_dir, f"{self.base_name}.inp")
        if not os.path.isfile(control):
            raise FileNotFoundError(f"solver control file not found: {control} (press 'Save Solver Settings').")
        with open(control, "r", encoding="utf-8", errors="ignore") as fh:
            self.control_text = fh.read()
        self.conds = list(self.objective.get("conditions") or []) or [{}]
        self.constraints = list(s.get("constraints") or []) or list(self.objective.get("constraints") or [])
        self.accept = tuple(s.get("accept_statuses") or ("ok",))
        self.prepro_cmd, self.solver_cmd = s.get("prepro_cmd"), s.get("solver_cmd")     # stand-ins (tests)
        self.solver_dir = s.get("solver_dir") or None

        problems = _objective_problems(self.objective, self.constraints)
        for c in self.constraints:
            from Optimisation.BayesianOptimisation.optimiser import _parse_constraint
            name = _parse_constraint(c)["metric"]
            if "-" in name:
                # bo_data.mcsv stores each constraint under "C__<name>-<rows>:"; a '-' in the
                # name makes the file unreadable when the run is resumed
                problems.append(f"constraint '{c}': write the left-hand side without a minus sign "
                                "(e.g. 'CL >= 0.3' instead of '-CL <= -0.3').")
        if problems:
            raise ValueError(" ".join(problems))
        basis = s.get("morph_basis_json") or os.path.join(self.output_dir, "Control Nodes", "morph_basis.json")
        if not os.path.isfile(basis):
            raise FileNotFoundError(f"morph basis not found: {basis} (press 'Save 2D Basis').")
        self.builder = Design2DBuilder(basis, mesh=mesh, log=log)
        self.lb, self.ub = np.asarray(s.get("lb"), float), np.asarray(s.get("ub"), float)
        if self.lb.shape != self.ub.shape or self.lb.ndim != 1 or len(self.lb) != self.builder.n_modes:
            raise ValueError(f"the optimiser has {self.lb.size} bound(s) but the saved basis has "
                             f"{self.builder.n_modes} mode(s). The bounds are from an older basis: open "
                             "Optimisation Settings and save them again.")
        if int(s.get("n_dim", len(self.lb))) != len(self.lb):
            raise ValueError(f"n_dim = {s.get('n_dim')} but {len(self.lb)} bounds are given.")
        if not np.all(self.ub > self.lb):
            raise ValueError("every upper bound must be larger than its lower bound.")

        # every flow condition's control file, checked before anything runs
        self.warnings: List[str] = []
        flags = list(self.builder.flags.values())
        for i, cond in enumerate(self.conds, 1):
            text, notes = apply_condition(self.control_text, cond)
            errors, warns = check_control(text, self.opts.n_grids, flags, self.time_limit_s)
            if errors:
                raise ValueError(f"flow condition {i}: " + " ".join(errors))
            self.warnings += [f"flow condition {i}: {w}" for w in dict.fromkeys(notes + warns)]
        its = _as_float(get_namelist_value(self.control_text, "numberOfMGIterations"))
        if its is not None and its < 0 and "timeout" not in self.accept:
            self.warnings.append(
                f"ivd%numberOfMGIterations = {int(its)}: a design that does not reach {int(-its)} orders within "
                "the time limit is REJECTED (status 'timeout') and gets the penalty value. If some designs "
                "converge and others do not, the surrogate is fitted to penalties, not physics. A fixed, "
                "positive number of cycles gives every design a result at the same cost.")
        if not (self.prepro_cmd and self.solver_cmd):
            find_executables(self.solver_dir)

    def fingerprint(self) -> str:
        with open(self.builder.basis_path, "rb") as fh:
            basis = fh.read()
        with open(self.builder.modes_path, "rb") as fh:
            modes = fh.read()
        blob = json.dumps({
            "expression": self.objective.get("expression"), "objective_type": self.objective.get("objective_type"),
            "constraints": self.constraints, "conditions": self.conds, "control": self.control_text,
            "prepro": self.opts.to_dict(), "accept": list(self.accept),
            "lb": self.lb.tolist(), "ub": self.ub.tolist(),
        }, sort_keys=True).encode() + basis + modes
        return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# the optimisation
# ---------------------------------------------------------------------------
def run(run_dir: str, check_only: bool = False, mesh=None, echo: Optional[Callable[[str], None]] = None) -> int:
    """
    Run (or resume) the optimisation in `run_dir`. Returns 0 finished, 1 set-up
    error, 3 stopped by the STOP file. `mesh`: a ready baseline EdgeMesh2D
    (tests); normally it is loaded from the path in morph_basis.json.
    """
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    log = OptLog(run_dir, echo)
    stop_file = os.path.join(run_dir, STOP_FILE)
    try:
        os.remove(stop_file)                       # a STOP left over from the last run must not stop this one
    except OSError:
        pass
    log(f"[LOCAL-OPT] {'Checking' if check_only else 'Starting'} in {run_dir}")
    try:
        from Remote.remoteOpt import _build_objective_callable, _impute_nonfinite
        from Optimisation.BayesianOptimisation.objective_evaluator import ConstraintSet
        from Optimisation.BayesianOptimisation.optimiser import BayesianOptimiser, _parse_constraint
        from Optimisation.BayesianOptimisation.kernels import (RBFKernel, SquaredExponentialKernel,
                                                               ExponentialKernel, Mat12Kern, Mat32Kern, Mat52Kern)
        from Optimisation.BayesianOptimisation.acquisition_functions import EI, POI, UCB
        setup = Setup(run_dir, log, mesh=mesh)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, KeyError) as e:
        log(f"[LOCAL-OPT][ERROR] {e}")
        return 1
    s, objective, conds = setup.settings, setup.objective, setup.conds

    obj_func, obj_expr = _build_objective_callable(objective)
    cons_parsed = [_parse_constraint(c) for c in setup.constraints]
    cset = ConstraintSet([c if isinstance(c, str) else f"{c['metric']} {c.get('sense', '<=')} {c['limit']}"
                          for c in setup.constraints])
    log(f"[LOCAL-OPT] Case folder: {setup.output_dir}  (base name '{setup.base_name}')")
    log(f"[LOCAL-OPT] Objective (minimised): {obj_expr}")
    log(f"[LOCAL-OPT] Constraints: {setup.constraints if setup.constraints else 'none'}")
    for i, c in enumerate(conds, 1):
        log(f"[LOCAL-OPT] Flow condition {i}: {c if c else 'as in the solver control file'}")
    for w in setup.warnings:
        log(f"[LOCAL-OPT][WARN] {w}")

    # how much of the design box is usable, and how far it moves the wall
    n_check = int(s.get("bounds_check_samples", 100) or 0)
    if n_check > 0:
        r = setup.builder.box_report(setup.lb, setup.ub, n_check)
        pct = 100.0 * r["wall_disp_max"] / max(r["wall_length"], 1e-300)
        log(f"[LOCAL-OPT] Design box, {r['n']} sampled designs: {100 * r['valid_fraction']:.0f} % give a valid "
            f"mesh; largest wall displacement median {r['wall_disp_median']:.4g}, maximum {r['wall_disp_max']:.4g} "
            f"= {pct:.1f} % of the wall length ({r['wall_length']:.4g}).")
        if r["valid_fraction"] < 0.9:
            log(f"[LOCAL-OPT][WARN] {100 * (1 - r['valid_fraction']):.0f} % of the box gives an invalid mesh "
                f"({r['reasons']}). Each such design costs an optimiser step and teaches the surrogate a "
                "penalty, not physics: tighten the bounds.")
        if pct > 5.0:
            log("[LOCAL-OPT][WARN] The bounds come from mesh validity, not from the shapes you want to allow: "
                f"designs in this box move the wall by up to {pct:.0f} % of its length. Check that this is "
                "intended (Optimisation Settings > Edit Bounds).")
    if check_only:
        log("[LOCAL-OPT] Set-up is consistent. No CFD was run.")
        return 0

    try:
        tm = Local2DTestManager(setup.output_dir, setup.base_name, setup.builder, setup.control_text, setup.opts,
                                run_dir, solver_time_limit_s=setup.time_limit_s, solver_dir=setup.solver_dir,
                                log=log, should_stop=lambda: os.path.exists(stop_file),
                                accept_statuses=setup.accept, make_plots=bool(s.get("make_plots", True)),
                                keep_sol=bool(s.get("keep_sol", False)), prepro_cmd=setup.prepro_cmd,
                                solver_cmd=setup.solver_cmd, fingerprint=setup.fingerprint(),
                                x_tol=1e-4 * (setup.ub - setup.lb))
    except ValueError as e:
        log(f"[LOCAL-OPT][ERROR] {e}")
        return 1
    if tm.designs and not os.path.isfile(os.path.join(run_dir, "bo_data.mcsv")):
        log("[LOCAL-OPT][ERROR] designs.json exists but bo_data.mcsv does not: the run folder is incomplete. "
            "Start a new optimisation.")
        return 1

    kern_map = {"RBFKernel": RBFKernel, "Squared Exponential Kernel": SquaredExponentialKernel,
                "Exponential Kernel": ExponentialKernel, "Mat12Kern": Mat12Kern, "Mat32Kern": Mat32Kern,
                "Mat52Kern": Mat52Kern}
    acq_map = {"Expected Improvement": EI, "Probability of Improvement": POI, "Upper Confidence Bound": UCB}
    settings = dict(s)
    try:
        settings["kernel"] = kern_map[s.get("kernel", "Mat52Kern")]
        settings["acquisition_function"] = acq_map[s.get("acquisition_function", "Expected Improvement")]
    except KeyError as e:
        log(f"[LOCAL-OPT][ERROR] unknown kernel or acquisition function {e}.")
        return 1
    settings.update(sim_dir=run_dir, constraints=setup.constraints, n_dim=len(setup.lb),
                    lb=setup.lb.tolist(), ub=setup.ub.tolist())
    settings.setdefault("n_obj", 1)
    settings.setdefault("n_samples", 5)
    if bool(s.get("include_baseline", True)):
        settings["initial_points"] = [[0.0] * len(setup.lb)]
        outside = bool(np.any(setup.lb > 0.0) or np.any(setup.ub < 0.0))
        log("[LOCAL-OPT] The baseline (x = 0) is evaluated with the initial sample as design n=0"
            + (" (it lies outside the bounds; it is still used as a training point)." if outside else "."))
    if s.get("seed") is not None:
        np.random.seed(int(s["seed"]))
        log(f"[LOCAL-OPT] numpy seed {int(s['seed'])} (fixes the initial sample; the acquisition search "
            "(CMA-ES) draws its own random numbers).")

    def relabel(msg):
        log(str(msg).replace("[REMOTE-OPT]", "[LOCAL-OPT]"))

    def init_func(X_list, gen_num):
        tm.init_generation(X_list, gen_num, conds)

    def eval_func(X_list, gen_num):
        per_design = tm.evaluate_generation(X_list, gen_num, conds)
        raw = []
        for metrics_per_cond in per_design:
            y = 0.0
            for cond, m in zip(conds, metrics_per_cond):
                # a condition without a result makes the design's objective NaN (imputed below)
                y += float(cond.get("Weight", 1.0)) * (obj_func(m) if m.get("status") in setup.accept else NAN)
            raw.append(float(y))
        if gen_num == 0 and not any(_finite(y) for y in raw):
            raise OptAbort("none of the initial designs gave a result, so there is nothing to fit a model to. "
                           f"See {tm.csv_path} for what happened to each.")
        Y = _impute_nonfinite(raw, bo.Y, +1, f"generation {gen_num} objective", relabel)
        C, C_raw = {}, {}
        feas = [_finite(y) for y in raw]                 # no result -> never "feasible"
        if cons_parsed:
            per_point = [cset.evaluate(mpc) for mpc in per_design]
            for cons in cons_parsed:
                name = cons["metric"]
                vals = np.array([pp.get(name, np.nan) for pp in per_point], dtype=float)
                vals[~np.isfinite(vals)] = np.nan
                C_raw[name] = vals.copy()
                for i, v in enumerate(vals):
                    feas[i] = feas[i] and _finite(v) and (v <= cons["limit"] if cons["sense"] == "<=" else
                                                          v >= cons["limit"])
                direction = +1 if cons["sense"] == "<=" else -1
                if np.any(~np.isfinite(vals)):
                    pool = np.concatenate([np.asarray(bo.C.get(name, []), float).ravel(),
                                           vals[np.isfinite(vals)], [cons["limit"]]])
                    vals = _impute_nonfinite(vals, pool, direction, f"generation {gen_num} constraint {name}",
                                             relabel)
                C[name] = vals
        tm.record_objectives(gen_num, raw, Y, C_raw or None, feas)
        for i, (y, yu) in enumerate(zip(raw, Y)):
            rec = tm._find(gen_num, i)
            desc = "".join(f" | {k} = {C_raw[k][i]:.5g}" for k in C_raw) + \
                   (f" | feasible: {'yes' if feas[i] else 'NO'}" if cons_parsed else "")
            log(f"[LOCAL-OPT] n={rec['n']}: objective = "
                + (f"{y:.6g}" if _finite(y) else f"none ({rec['status']}); the optimiser is given {yu:.6g}") + desc)
        _progress(tm, log, run_dir, obj_expr)
        return (Y, C) if cons_parsed else Y

    log(f"[LOCAL-OPT] {len(setup.lb)} design variable(s); {settings['n_samples']} initial sample(s)"
        f"{' + baseline' if 'initial_points' in settings else ''}, then {settings.get('count_limit', 5)} "
        "optimiser step(s), one design each, run one after the other.")
    bo = BayesianOptimiser(settings, eval_func=eval_func, init_func=init_func)

    class _Tee:                                       # the optimiser's own messages, into this log
        def log(self, message):
            log(f"[BO] {message}")
    bo.logger = _Tee()
    _plain_convergence = bo.convergence

    def _safe_convergence(*a, **k):                   # its plot must never end a run
        try:
            return _plain_convergence(*a, **k)
        except Exception as e:
            log(f"[LOCAL-OPT][WARN] the optimiser's convergence plot was not written: {e}")
    bo.convergence = _safe_convergence
    _plain_write = bo.mac.write

    def _atomic_write(payload):                       # a kill in mid-write must not cost the history
        name = bo.mac.filename
        bo.mac.filename = name + ".tmp"
        try:
            _plain_write(payload)
        finally:
            bo.mac.filename = name
        os.replace(name + ".tmp", name)
    bo.mac.write = _atomic_write

    try:
        with _PrintsToFile(os.path.join(run_dir, "optimiser_output.txt")):
            X_best, Y_best = bo.optimise(cont=True)
    except Exception as e:
        if isinstance(e, OptAbort):
            log(f"[LOCAL-OPT][ERROR] {e}")
            return 1
        if not isinstance(e, RunStopped):
            import traceback
            log(f"[LOCAL-OPT][ERROR] The optimisation ended with an error: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}\n[LOCAL-OPT] Finished designs are kept; once the cause is fixed, "
                "run the optimisation again to continue.")
            return 1
        log(f"[LOCAL-OPT] STOPPED: {e} Run the optimisation again to continue from here "
            f"({sum(1 for d in tm.designs if d.get('done'))} design(s) are kept).")
        try:
            os.remove(stop_file)
        except OSError:
            pass
        return 3

    best, base = tm.best(), tm.baseline()
    result = {
        "X_best": None if X_best is None else np.asarray(X_best, float).tolist(),
        "Y_best": None if Y_best is None else float(Y_best),
        "best_design_n": None if best is None else best["n"],
        "baseline_objective": None if base is None else base.get("objective"),
        "n_designs": len(tm.designs),
        "n_without_result": sum(1 for d in tm.designs if d.get("objective") is None),
        "objective_minimised": obj_expr, "settings": s, "objective": objective,
    }
    _write_json(os.path.join(run_dir, "optimization_results.json"), result)
    log("[LOCAL-OPT] OPTIMISATION COMPLETE")
    if best is None:
        log("[LOCAL-OPT][WARN] No design gave a usable" + (" feasible" if cons_parsed else "") + " result.")
    else:
        line = f"[LOCAL-OPT] Best design: n={best['n']}, objective {best['objective']:.6g}"
        if base is not None and base.get("objective") is not None:
            d = best["objective"] - base["objective"]
            line += (f"; baseline {base['objective']:.6g} (change {d:+.6g}"
                     + (f", {100 * d / abs(base['objective']):+.2f} %" if base["objective"] else "") + ")")
        log(line)
        log(f"[LOCAL-OPT] Its files: {os.path.join(setup.output_dir, 'solutions', 'n_' + str(best['n']))}; "
            f"all designs: {tm.csv_path}")
        tails = {k: [m.get(f"{k}_tail_range") for d in tm.designs for m in (d.get("metrics") or [])] for k in ("CL", "CD")}
        tails = {k: [v for v in vals if _finite(v)] for k, vals in tails.items()}
        if tails["CL"] and tails["CD"] and (max(tails["CL"]) > 0 or max(tails["CD"]) > 0):
            log("[LOCAL-OPT] Numerical resolution: over the last 20 % of the solver cycles the forces were still "
                f"moving by {np.median(tails['CL']):.3g} (lift) and {np.median(tails['CD']):.3g} (drag), median over "
                f"the designs; largest {max(tails['CL']):.3g} and {max(tails['CD']):.3g}. Differences between "
                "designs smaller than this are not resolved (columns *_tail_range of designs.csv).")
    return 0


def _progress(tm: Local2DTestManager, log, run_dir: str, obj_expr: str) -> None:
    """history.png, updated after every generation. Never raises."""
    try:
        from Local.flite2d_post import plot_opt_history
        plot_opt_history(tm.designs, os.path.join(run_dir, "history.png"), title=f"Optimisation history - {obj_expr}")
    except Exception as e:
        log(f"[LOCAL-OPT][WARN] history.png not written: {e}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    return run(argv[0], check_only=check)


if __name__ == "__main__":
    sys.exit(main())
