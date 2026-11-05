"""
flite2d_local.py

Run the FLITE 2D preprocessor (PrePro.exe, "Aggl2d") and solver (Solver.exe,
"MGNS2D") locally for one design. No Qt in here, so the same code serves the
"Run Simulation" button and, later, the 2D optimisation loop.

Folder layout (same convention as the 3D chain):

    <output>/volumes/n_<n>/<base>_<n>.dat            solver mesh (FileRW.FliteMesh2D)
    <output>/preprocessed/n_<n>/runPrePro.inp        answers piped to PrePro.exe
    <output>/preprocessed/n_<n>/<base>_<n>.sol       PrePro output
    <output>/preprocessed/n_<n>/prepro.log
    <output>/solutions/n_<n>/cond_<i>/<base>.inp     solver control file (namelist)
    <output>/solutions/n_<n>/cond_<i>/runSolver.inp  answers piped to Solver.exe
    <output>/solutions/n_<n>/cond_<i>/<base>_<n>.res / .rsd / solver.log / metrics.json
    <output>/solutions/n_<n>/cond_<i>/*.png, <base>_<n>_surface.csv   pictures (Local.flite2d_post)
    <output>/surfaces/n_<n>/<base>_<n>.png           picture of the wall and the mesh around it

How the executables are driven
------------------------------
Both read their answers from standard input, exactly as the original MATLAB
scripts did with `system('PrePro.exe < runPrePro.inp')`:

    PrePro : mesh file, hybrid (.true./.false.), number of grids,
             directionality parameter,
             [number of directional agglomerations - asked ONLY when the
              directionality parameter is not 0],
             visualization mode (0 = none), output file
    Solver : control file, computation (.sol) file, startup file (blank = none),
             result file, residual file

Each executable runs with its working directory set to the folder holding its
files, and is given SHORT RELATIVE names. Absolute paths are avoided on
purpose: the length of the Fortran file-name variables is unknown, and a path
under "OneDrive - Swansea University\\..." is long and contains blanks.

What counts as success
----------------------
Old outputs are deleted before each run, so a file left over from an earlier
run can never be taken for a new result.
  * PrePro: return code 0, no Fortran run-time error in its log, and a
    non-empty .sol file. (A PrePro that crashes while writing leaves a partial
    .sol behind; that file is deleted so the solver cannot pick it up.)
  * Solver: return code 0, no Fortran run-time error in its log, and a .rsd
    whose last row is finite. A NaN row is reported as "diverged", a time-out
    as "timeout", a run-time error as "crashed" (even if some iterations were
    written first); none of these is turned into a number.

The node-number column of the solver's .res file is 5 characters wide and
turns into '*****' from node 100,000 on. Read .res rows by position: they are
in the node order of the mesh file.

Stopping a run
--------------
`should_stop` (a function returning True once the user has pressed Stop) is
polled twice a second while either program runs. The programs have no way of
being asked to finish, so the process is killed. What is left behind:
  * during PrePro: nothing usable; RunStopped is raised and no solver is run.
  * during the solver: the .rsd up to the last completed iteration, and the
    .res the solver last wrote. It rewrites the .res from scratch every
    `ivd%writeToFileInterval` iterations (about 0.3 s for 140,000 nodes), so
    the flow field belongs to the last multiple of that interval, NOT to the
    last row of the .rsd; metrics.json records it as "res_iteration". Before
    killing, the runner waits for a write in progress to finish, and the row
    count of the .res is checked afterwards ("res_complete").
    The case is marked "stopped"; lift/drag are those of the last iteration
    reached, at whatever convergence that happened to be.
The same applies to a run that hits the time limit ("timeout"), with one
exception: a timed-out run whose lift and drag had stopped moving is accepted.

Accepting a timed-out run whose forces have settled
---------------------------------------------------
With a negative `ivd%numberOfMGIterations` the solver stops on the residual,
but the quantity the optimiser uses is the force. When the time limit ends a
run, the .rsd is checked (see check_force_settling): over the last `window`
multigrid cycles, the largest minus the smallest value of the lift and of the
drag COEFFICIENT (force / reference length; the .rsd holds forces per q per
unit of the solver's length) must both be <= `tol`. If so, the status becomes
"ok", "converged_by" is "forces", and the numbers are those of the last cycle.
The reference length is the x-extent of the walls of the solver mesh, unless
one is given. The result of the check is kept in metrics.json under
"force_check" for EVERY run (also those that reached the residual target), so
how far the forces still moved is on record either way.
A user stop is never turned into a result. Note that the flow field on disk is
still that of "res_iteration", up to writeToFileInterval cycles earlier.

The solver's `ivd%numberOfMGIterations` may be NEGATIVE, meaning "run until
the residual has dropped by that many orders of magnitude". A case that never
gets there would run forever, so `time_limit_s` is enforced here: when it
expires the process is killed and the case is marked "timeout".
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

VISCOUS_WALL_FLAG = 5
_NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0      # CREATE_NO_WINDOW

# flow-condition keys (GUI "Parallel Flow Conditions") -> solver namelist variables
CONDITION_TO_NAMELIST = {
    "AoA": ("alpha", float),
    "Mach": ("MachNumber", float),
    "Re": ("ReynoldsNumber", float),
    "TurbModel": ("turbulenceModel", int),
}
CONDITION_KEYS_NOT_USED_IN_2D = ("Altitude", "EngineFlow", "MassFlow")


_FIX_HINT = ("[2D] 'access violation' and 'stack overflow' are known defects of the 2013 PrePro.exe / "
             "Solver.exe (see Utilities/fix_flite2d_exes.py). Repair both once with\n"
             "    python Utilities/fix_flite2d_exes.py Solver\n"
             "(run from the repository root) and run again.")


# ---------------------------------------------------------------------------
# executables
# ---------------------------------------------------------------------------
def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_solver_dir() -> str:
    """<repository>/Solver, where PrePro.exe and Solver.exe live."""
    return os.path.join(repo_root(), "Solver")


def find_executables(solver_dir: Optional[str] = None) -> Tuple[str, str]:
    d = solver_dir or default_solver_dir()
    pre, sol = os.path.join(d, "PrePro.exe"), os.path.join(d, "Solver.exe")
    missing = [p for p in (pre, sol) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("2D executable(s) not found: " + ", ".join(missing))
    if not sys.platform.startswith("win"):
        raise RuntimeError("PrePro.exe and Solver.exe are Windows executables; "
                           "the 2D solver can only run on a Windows machine.")
    return pre, sol


def unfixed_executables(solver_dir: Optional[str] = None) -> List[str]:
    """Names of the 2D executables in `solver_dir` that still have the known defects
    (see Utilities/fix_flite2d_exes.py). Empty if both are repaired or cannot be checked."""
    d = solver_dir or default_solver_dir()
    util = os.path.join(repo_root(), "Utilities")
    try:
        if util not in sys.path:
            sys.path.insert(0, util)
        from Utilities.fix_flite2d_exes import is_fixed
    except Exception:
        return []
    return [n for n in ("PrePro.exe", "Solver.exe")
            if os.path.isfile(os.path.join(d, n)) and not is_fixed(os.path.join(d, n))]


# ---------------------------------------------------------------------------
# preprocessor options
# ---------------------------------------------------------------------------
@dataclass
class PreProOptions:
    n_grids: int = 5                 # multigrid levels to generate
    hybrid: bool = False             # "Is grid hybrid": our meshes are all triangles
    directionality: str = "0"        # passed to PrePro verbatim; 0 = no directional agglomeration
    n_directional: int = 0           # number of directional agglomerations (only asked if
                                     # directionality != 0)

    def __post_init__(self):
        self.n_grids = int(self.n_grids)
        self.n_directional = int(self.n_directional)
        self.hybrid = bool(self.hybrid)
        self.directionality = str(self.directionality).strip() or "0"
        if self.n_grids < 1:
            raise ValueError("number of grids must be >= 1")
        if self.n_directional < 0:
            raise ValueError("number of directional agglomerations must be >= 0")
        float(self.directionality)   # must be a number; raises ValueError otherwise

    def answers(self, mesh_name: str, sol_name: str) -> List[str]:
        # Prompt order checked against PrePro.exe itself: "Number of directional
        # agglomerations" appears only for a non-zero directionality parameter, and
        # "Visualization mode (0: none)" always follows. Visualization is kept off:
        # any other mode makes PrePro ask for more file names.
        a = [mesh_name, ".true." if self.hybrid else ".false.", str(self.n_grids), self.directionality]
        if float(self.directionality) != 0.0:
            a.append(str(self.n_directional))
        return a + ["0", sol_name]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PreProOptions":
        d = d or {}
        return cls(**{k: d[k] for k in ("n_grids", "hybrid", "directionality", "n_directional") if k in d})


# ---------------------------------------------------------------------------
# solver control file (Fortran namelist  &inputVariables  ivd%name = value, ... /)
# ---------------------------------------------------------------------------
def _nml_pattern(name: str):
    return re.compile(r"(?im)^(\s*ivd%" + re.escape(name) + r"\s*=\s*)([^,\n!/]*)(\s*,?)")


def is_control_file(text: str) -> bool:
    return bool(re.search(r"(?i)&\s*inputVariables", text or "")) and "ivd%" in (text or "")


def get_namelist_value(text: str, name: str) -> Optional[str]:
    m = _nml_pattern(name).search(text)
    return m.group(2).strip() if m else None


def set_namelist_value(text: str, name: str, value) -> str:
    """Replace ivd%<name>, or add it before the closing '/' if it is not there."""
    if isinstance(value, bool):
        val = ".true." if value else ".false."
    elif isinstance(value, int):
        val = str(value)
    elif isinstance(value, float):
        val = f"{value:.12g}"
    else:
        val = str(value)
    pat = _nml_pattern(name)
    if pat.search(text):
        return pat.sub(lambda m: f"{m.group(1)}{val},", text, count=1)
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "/":
            lines.insert(i, f" ivd%{name} = {val},")
            return "\n".join(lines) + "\n"
    raise ValueError("solver control file has no closing '/' line; not a namelist.")


def _as_float(s: Optional[str]) -> Optional[float]:
    try:
        return float(str(s).lower().replace("d", "e"))
    except (TypeError, ValueError):
        return None


def apply_condition(text: str, cond: Optional[dict]) -> Tuple[str, List[str]]:
    """
    Control-file text for one flow condition. Only AoA, Mach, Re and TurbModel
    are written; keys that are absent leave the template value untouched.
    Returns (text, notes).
    """
    notes = []
    for key, (name, typ) in CONDITION_TO_NAMELIST.items():
        if cond and key in cond and cond[key] is not None and str(cond[key]) != "":
            text = set_namelist_value(text, name, typ(float(cond[key])) if typ is int else float(cond[key]))
    ignored = [k for k in CONDITION_KEYS_NOT_USED_IN_2D if cond and k in cond]
    if ignored:
        notes.append(f"flow-condition field(s) {ignored} are not used by the 2D solver set-up.")
    return text, notes


def check_control(text: str, n_grids: Optional[int] = None,
                  boundary_flags: Optional[Iterable[int]] = None,
                  time_limit_s: Optional[float] = None) -> Tuple[List[str], List[str]]:
    """
    (errors, warnings) for a control file before it is run.
    boundary_flags: the solver boundary flags present in the mesh, if known.
    """
    errors, warnings = [], []
    if not is_control_file(text):
        return ["this is not a FLITE 2D solver control file (no '&inputVariables' namelist with "
                "'ivd%...' entries). Load the 2D template, not the 3D one."], warnings
    used = _as_float(get_namelist_value(text, "numberOfGridsToUse"))
    if n_grids is not None and used is not None and used > n_grids:
        errors.append(f"ivd%numberOfGridsToUse = {int(used)} but the preprocessor only generates "
                      f"{n_grids} grid(s).")
    for name in ("MachNumber", "alpha"):
        if _as_float(get_namelist_value(text, name)) is None:
            errors.append(f"ivd%{name} is missing or not a number.")
    re_ = _as_float(get_namelist_value(text, "ReynoldsNumber"))
    if boundary_flags is not None and re_ is not None:
        viscous_walls = VISCOUS_WALL_FLAG in set(int(f) for f in boundary_flags)
        if re_ > 0.0 and not viscous_walls:
            errors.append(f"ivd%ReynoldsNumber = {re_:g} (viscous) but no edge is a viscous wall (flag "
                          f"{VISCOUS_WALL_FLAG}). Set the walls to 'Viscous wall' and re-save the basis, "
                          "or set the Reynolds number to 0 for an inviscid run.")
        if re_ == 0.0 and viscous_walls:
            warnings.append("viscous walls in the mesh but ivd%ReynoldsNumber = 0 (inviscid).")
    turb = _as_float(get_namelist_value(text, "turbulenceModel"))
    if re_ == 0.0 and turb:
        warnings.append(f"ivd%turbulenceModel = {int(turb)} with ivd%ReynoldsNumber = 0: the run is inviscid, "
                        "so a turbulence model has no meaning here (set TurbModel to 0).")
    its = _as_float(get_namelist_value(text, "numberOfMGIterations"))
    if its is not None and its < 0:
        stop = (f"the time limit ({time_limit_s / 60.0:g} min) is the only other stop" if time_limit_s
                else "there is NO time limit set, so a case that stalls runs forever")
        warnings.append(f"ivd%numberOfMGIterations = {int(its)}: the solver runs until the residual has "
                        f"dropped {int(-its)} orders; {stop}.")
    return errors, warnings


# ---------------------------------------------------------------------------
# residual file
# ---------------------------------------------------------------------------
def read_rsd(path: str) -> dict:
    """
    .rsd rows: iteration, residual, lift, drag, moment, pressure lift,
    pressure drag, friction drag. Returns the last row as
    {"status", "iterations", "residual", "CL", "CD", "CM", "rows"} with
    status "ok" | "diverged" (NaN/inf in the last row) | "no_data".
    Lift/drag are forces / q per unit of the solver's length, i.e. coefficients
    only if the mesh was scaled to a reference length of 1.
    """
    nan = float("nan")
    out = {"status": "no_data", "iterations": 0, "residual": nan, "CL": nan, "CD": nan, "CM": nan, "rows": 0}
    try:
        with open(path, "r", errors="ignore") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    rows = []
    for ln in lines:
        toks = ln.replace(",", " ").split()
        try:
            vals = [float(t.lower().replace("d", "e")) for t in toks]
        except ValueError:
            continue
        if len(vals) >= 4:
            rows.append(vals)
    if not rows:
        return out
    last = rows[-1]
    out.update(rows=len(rows), iterations=int(last[0]) if math.isfinite(last[0]) else len(rows),
               residual=last[1], CL=last[2], CD=last[3], CM=last[4] if len(last) > 4 else nan)
    finite = all(math.isfinite(v) for v in last[1:4])
    out["status"] = "ok" if finite else "diverged"
    return out


# ---------------------------------------------------------------------------
# have the forces settled? (accepting a timed-out run, see the top of the file)
# ---------------------------------------------------------------------------
# 1e-4 in a coefficient is one drag count. The .rsd prints about five significant
# digits, so with forces of order one per unit length a tolerance much below 1e-5
# asks for the printed number not to change at all.
FORCE_CHECK_DEFAULT = {"enabled": True, "window": 500, "tol": 1e-4}


def force_check_settings(cfg=None) -> Optional[dict]:
    """
    The force check to apply, from a settings entry
    {"enabled": bool, "window": cycles, "tol": coefficient, "ref_length": optional}:
    returns {"window", "tol", "ref_length"} or None when the check is off.
    None -> FORCE_CHECK_DEFAULT; False -> off. Raises ValueError on a bad entry.
    """
    if cfg is None:
        cfg = FORCE_CHECK_DEFAULT
    if cfg is False:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"force_check must be a dictionary, not {cfg!r}")
    if not cfg.get("enabled", True):
        return None
    try:
        window = int(cfg.get("window", FORCE_CHECK_DEFAULT["window"]))
        tol = float(cfg.get("tol", FORCE_CHECK_DEFAULT["tol"]))
        ref = cfg.get("ref_length")
        ref = None if ref in (None, "", 0) else float(ref)
    except (TypeError, ValueError):
        raise ValueError(f"force_check: window must be a whole number of cycles and tol a number ({cfg!r})")
    if window < 1:
        raise ValueError(f"force_check: window must be at least 1 cycle (got {window})")
    if not (math.isfinite(tol) and tol > 0):
        raise ValueError(f"force_check: tol must be a positive number (got {tol})")
    if ref is not None and not (math.isfinite(ref) and ref > 0):
        raise ValueError(f"force_check: ref_length must be a positive number (got {ref})")
    return {"window": window, "tol": tol, "ref_length": ref}


def check_force_settling(history, window: int, tol: float, ref_length: float = 1.0) -> dict:
    """
    history: .rsd rows (iteration, residual, lift, drag, ...), see
    Local.flite2d_post.read_rsd_history. Over the cycles last-window .. last,
    the spread (max - min) of lift / ref_length and drag / ref_length.
    Returns {"passed", "reason", "window", "tol", "ref_length", "first_iteration",
    "last_iteration", "rows", "CL_spread", "CD_spread"}; "passed" is False with a
    reason when the history is shorter than the window or holds a non-finite value.
    """
    out = {"passed": False, "reason": "", "window": int(window), "tol": float(tol),
           "ref_length": float(ref_length), "first_iteration": None, "last_iteration": None, "rows": 0,
           "CL_spread": None, "CD_spread": None}
    H = np.asarray(history, float)
    if H.ndim != 2 or H.shape[1] < 4 or len(H) < 2:
        out["reason"] = "fewer than 2 rows in the residual file"
        return out
    it = H[:, 0]
    last = it[-1]
    if not math.isfinite(last):
        out["reason"] = "the last iteration number is not a number"
        return out
    out["last_iteration"] = int(last)
    if not (it[0] <= last - window):
        out["reason"] = f"the run is shorter than the window ({int(last - it[0])} of {int(window)} cycles)"
        return out
    sel = np.isfinite(it) & (it >= last - window)
    W = H[sel]
    out["rows"], out["first_iteration"] = int(len(W)), int(W[0, 0])
    if len(W) < 2:
        out["reason"] = "fewer than 2 rows in the window"
        return out
    if not np.all(np.isfinite(W[:, 2:4])):
        out["reason"] = "a lift or drag value in the window is not finite"
        return out
    L = float(ref_length) if ref_length and ref_length > 0 else 1.0
    out["CL_spread"] = float(np.ptp(W[:, 2]) / L)
    out["CD_spread"] = float(np.ptp(W[:, 3]) / L)
    worst = max(out["CL_spread"], out["CD_spread"])
    out["passed"] = bool(worst <= tol)
    out["reason"] = ("settled" if out["passed"] else
                     ("lift" if out["CL_spread"] > tol else "drag") + " still moving")
    return out


# ---------------------------------------------------------------------------
# running a process with piped answers, a log file and a time limit
# ---------------------------------------------------------------------------
def _run_piped(cmd: Sequence[str], cwd: str, answers_file: str, log_file: str,
               time_limit_s: Optional[float] = None, poll_s: float = 0.5,
               on_poll: Optional[Callable[[float], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None,
               settle: Sequence[str] = ()) -> Tuple[str, Optional[int], float]:
    """Returns (status, return code, wall time); status = finished | timeout | stopped.
    settle: files the program rewrites while it runs; before the process is killed
    (stop or time limit) a write in progress is given time to finish."""
    t0 = time.time()
    with open(answers_file, "r") as fin, open(log_file, "w") as flog:
        proc = subprocess.Popen(list(cmd), cwd=cwd, stdin=fin, stdout=flog, stderr=subprocess.STDOUT,
                                creationflags=_NO_WINDOW)
        status = "finished"
        try:
            while True:
                try:
                    proc.wait(timeout=poll_s)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.time() - t0
                    if on_poll:
                        on_poll(elapsed)
                    if should_stop and should_stop():
                        status = "stopped"
                    elif time_limit_s and elapsed > time_limit_s:
                        status = "timeout"
                    if status != "finished":
                        if settle:
                            _wait_settled(settle)
                        proc.kill()
                        proc.wait()
                        break
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    return status, proc.returncode, time.time() - t0


class RunStopped(RuntimeError):
    """The user stopped the run before it produced anything (i.e. during PrePro)."""


def _wait_settled(paths: Sequence[str], max_wait_s: float = 3.0, step_s: float = 0.1, need: int = 3) -> None:
    """Return once the sizes of `paths` have not changed for `need` checks in a row
    (a file being rewritten is growing), or after max_wait_s."""
    def sizes():
        out = []
        for p in paths:
            try:
                out.append(os.stat(p).st_size)
            except OSError:
                out.append(-1)
        return out
    prev, same, t0 = sizes(), 0, time.time()
    while same < need and time.time() - t0 < max_wait_s:
        time.sleep(step_s)
        cur = sizes()
        same = same + 1 if cur == prev else 0
        prev = cur


def _tail(path: str, n: int = 12) -> str:
    try:
        with open(path, "r", errors="ignore") as fh:
            return "\n".join(fh.read().splitlines()[-n:])
    except OSError:
        return ""


def _runtime_error(log_file: str) -> bool:
    """True if the Intel Fortran run-time reported an error ("forrtl: severe ...") in the log."""
    try:
        with open(log_file, "r", errors="ignore") as fh:
            return "forrtl:" in fh.read().lower()
    except OSError:
        return False


def _remove(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# one design
# ---------------------------------------------------------------------------
class Flite2DCase:
    def __init__(self, output_dir: str, base: str, n: int = 0, solver_dir: Optional[str] = None,
                 log: Callable[[str], None] = print, prepro_cmd: Optional[Sequence[str]] = None,
                 solver_cmd: Optional[Sequence[str]] = None, make_plots: bool = True):
        """
        prepro_cmd / solver_cmd: full command lines, for tests with stand-in
        programs. Normally left None: <solver_dir>/PrePro.exe and Solver.exe.
        make_plots: write the pictures of Local.flite2d_post (surface mesh per
        design; convergence, surface pressure and flow field per condition).
        """
        self.output_dir = os.path.abspath(output_dir)
        self.base, self.n = str(base), int(n)
        self.log = log
        self._prepro_cmd = list(prepro_cmd) if prepro_cmd else None
        self._solver_cmd = list(solver_cmd) if solver_cmd else None
        self.solver_dir = solver_dir
        self.stem = f"{self.base}_{self.n}"
        self.vol_dir = os.path.join(self.output_dir, "volumes", f"n_{self.n}")
        self.pre_dir = os.path.join(self.output_dir, "preprocessed", f"n_{self.n}")
        self.sol_root = os.path.join(self.output_dir, "solutions", f"n_{self.n}")
        self.mesh_path = os.path.join(self.vol_dir, self.stem + ".dat")
        self.sol_path = os.path.join(self.pre_dir, self.stem + ".sol")
        self.make_plots = bool(make_plots)
        self.mesh_image_path = os.path.join(self.output_dir, "surfaces", f"n_{self.n}", self.stem + ".png")
        self._mesh = None                 # (mtime of the .dat, dict of read_flite2d_mesh)

    # -- commands ------------------------------------------------------------
    def _commands(self):
        if self._prepro_cmd and self._solver_cmd:
            return self._prepro_cmd, self._solver_cmd
        pre, sol = find_executables(self.solver_dir)
        return self._prepro_cmd or [pre], self._solver_cmd or [sol]

    def cond_dir(self, nc: int) -> str:
        return os.path.join(self.sol_root, f"cond_{int(nc)}")

    # -- pictures ----------------------------------------------------------------
    def _mesh_data(self) -> dict:
        from FileRW.FliteMesh2D import read_flite2d_mesh
        mt = os.path.getmtime(self.mesh_path)
        if self._mesh is None or self._mesh[0] != mt:
            self._mesh = (mt, read_flite2d_mesh(self.mesh_path))
        return self._mesh[1]

    def reference_length(self) -> Optional[float]:
        """x-extent of the walls of the solver mesh (None if it cannot be read)."""
        try:
            from Local.flite2d_monitor import read_wall_outline
            walls = read_wall_outline(self.mesh_path)
            if not walls:
                return None
            x = np.concatenate([P[:, 0] for P, _ in walls])
            L = float(np.ptp(x))
            return L if math.isfinite(L) and L > 0 else None
        except (OSError, ValueError, IndexError):
            return None

    def _force_check(self, rsd: str, settings: Optional[dict]) -> Optional[dict]:
        """check_force_settling on the .rsd of a run (None when the check is off). Never raises."""
        if not settings:
            return None
        try:
            from Local.flite2d_post import read_rsd_history
            L, where = settings.get("ref_length"), "given"
            if not L:
                L, where = self.reference_length(), "x-extent of the walls"
            if not L:
                L, where = 1.0, "1 (the wall could not be read from the mesh)"
            out = check_force_settling(read_rsd_history(rsd), settings["window"], settings["tol"], L)
            out["ref_length_from"] = where
            return out
        except Exception as e:                       # a diagnostic must never cost the run its result
            return {"passed": False, "reason": f"check failed: {e}"}

    def mesh_node_count(self) -> Optional[int]:
        """Number of nodes from the header of the solver mesh (None if unreadable)."""
        try:
            with open(self.mesh_path, "r", errors="replace") as fh:
                head = [fh.readline() for _ in range(4)]
            return int(float(head[3].split()[1]))
        except (OSError, IndexError, ValueError):
            return None

    def save_mesh_image(self, force: bool = False) -> Optional[str]:
        """
        <output>/surfaces/n_<n>/<base>_<n>.png : the wall and the mesh around it;
        for n > 0 the baseline wall (volumes/n_0) is drawn underneath. Skipped if
        the picture is already newer than the mesh. Never raises: a picture that
        cannot be made is reported and the run goes on.
        """
        png = self.mesh_image_path
        try:
            if not force and os.path.isfile(png) and os.path.getmtime(png) >= os.path.getmtime(self.mesh_path):
                return png
            from Local.flite2d_post import plot_surface_mesh
            from FileRW.FliteMesh2D import read_flite2d_mesh
            m = self._mesh_data()
            bX = bB = None
            base_dat = os.path.join(self.output_dir, "volumes", "n_0", f"{self.base}_0.dat")
            if self.n != 0 and os.path.isfile(base_dat):
                b = read_flite2d_mesh(base_dat)
                bX, bB = b["nodes"], b["boundary"]
            plot_surface_mesh(m["nodes"], m["triangles"], m["boundary"], png,
                              title=f"{self.stem} - surface mesh" + ("" if self.n else " (baseline)"),
                              baseline_X=bX, baseline_boundary=bB)
            self.log(f"[2D] Surface-mesh picture -> {os.path.relpath(png, self.output_dir)}")
            return png
        except Exception as e:
            self.log(f"[2D][PLOT][WARN] surface-mesh picture not written: {e}")
            return None

    def write_plots(self, nc: int, metrics: dict, control_text: str) -> List[str]:
        """Pictures of one flow condition (see Local.flite2d_post.write_run_plots). Never raises."""
        try:
            from Local.flite2d_post import write_run_plots
            its = _as_float(get_namelist_value(control_text, "numberOfMGIterations"))
            paths = write_run_plots(self.cond_dir(nc), self.stem, self._mesh_data(), metrics,
                                    gamma=_as_float(get_namelist_value(control_text, "gamma")) or 1.4,
                                    target_orders=-its if its is not None and its < 0 else None, log=self.log)
        except Exception as e:
            self.log(f"[2D][PLOT][WARN] pictures not written: {e}")
            return []
        if paths:
            self.log(f"[2D] Pictures -> {os.path.relpath(self.cond_dir(nc), self.output_dir)}: "
                     + ", ".join(os.path.basename(p) for p in paths))
        return [os.path.relpath(p, self.output_dir) for p in paths]

    # -- preprocessor ----------------------------------------------------------
    def write_prepro_input(self, opts: PreProOptions) -> str:
        os.makedirs(self.pre_dir, exist_ok=True)
        mesh_rel = os.path.relpath(self.mesh_path, self.pre_dir)
        path = os.path.join(self.pre_dir, "runPrePro.inp")
        with open(path, "w", newline="\n") as fh:
            fh.write("\n".join(opts.answers(mesh_rel, os.path.basename(self.sol_path))) + "\n")
        return path

    def run_prepro(self, opts: PreProOptions, time_limit_s: Optional[float] = 1800.0,
                   should_stop: Optional[Callable[[], bool]] = None) -> str:
        if not os.path.isfile(self.mesh_path):
            raise FileNotFoundError(f"Solver mesh not found: {self.mesh_path}")
        cmd, _ = self._commands()
        inp = self.write_prepro_input(opts)
        log_file = os.path.join(self.pre_dir, "prepro.log")
        _remove(self.sol_path)
        self.log(f"[2D][PREPRO] n={self.n}: {opts.n_grids} grid(s), mesh "
                 f"{os.path.relpath(self.mesh_path, self.output_dir)}")
        status, rc, wall = _run_piped(cmd, self.pre_dir, inp, log_file, time_limit_s, should_stop=should_stop)
        has_sol = os.path.isfile(self.sol_path) and os.path.getsize(self.sol_path) > 0
        crashed = _runtime_error(log_file) or rc not in (0, None)
        if status == "stopped":
            _remove(self.sol_path)
            raise RunStopped("stopped while the preprocessor was running; nothing was computed.")
        if status != "finished" or crashed or not has_sol:
            why = ("time limit reached" if status == "timeout" else "stopped" if status == "stopped" else
                   f"it stopped with a run-time error (return code {rc})" if crashed and has_sol else
                   f"no .sol file was written (return code {rc})")
            _remove(self.sol_path)                    # never leave a partial .sol for the solver
            tail = _tail(log_file)
            hint = ("\n" + _FIX_HINT) if ("access violation" in tail.lower() or
                                          "stack overflow" in tail.lower()) else ""
            raise RuntimeError(f"PrePro failed: {why}. Last lines of {log_file}:\n{tail}{hint}")
        self.log(f"[2D][PREPRO] done in {wall:.1f} s -> {os.path.relpath(self.sol_path, self.output_dir)} "
                 f"({os.path.getsize(self.sol_path) / 1e6:.1f} MB)")
        return self.sol_path

    # -- solver ----------------------------------------------------------------
    def write_solver_inputs(self, control_text: str, nc: int) -> Tuple[str, str]:
        d = self.cond_dir(nc)
        os.makedirs(d, exist_ok=True)
        control = os.path.join(d, f"{self.base}.inp")
        with open(control, "w", newline="\n") as fh:
            fh.write(control_text if control_text.endswith("\n") else control_text + "\n")
        answers = os.path.join(d, "runSolver.inp")
        with open(answers, "w", newline="\n") as fh:
            fh.write("\n".join([os.path.basename(control), os.path.relpath(self.sol_path, d), "",
                                self.stem + ".res", self.stem + ".rsd", ""]) + "\n")
        return control, answers

    def run_solver(self, control_text: str, nc: int = 1, time_limit_s: Optional[float] = 3600.0,
                   progress_every_s: float = 15.0,
                   should_stop: Optional[Callable[[], bool]] = None, force_check=None) -> dict:
        """force_check: a settings entry for force_check_settings (None -> the default,
        False -> off); decides whether a timed-out run whose forces had settled is accepted."""
        fc_settings = force_check_settings(force_check)
        if not os.path.isfile(self.sol_path):
            raise FileNotFoundError(f"Preprocessor output not found: {self.sol_path}")
        _, cmd = self._commands()
        d = self.cond_dir(nc)
        _, answers = self.write_solver_inputs(control_text, nc)
        res, rsd = os.path.join(d, self.stem + ".res"), os.path.join(d, self.stem + ".rsd")
        log_file, metrics_file = os.path.join(d, "solver.log"), os.path.join(d, "metrics.json")
        _remove(res, rsd, metrics_file)
        self.log(f"[2D][SOLVER] n={self.n} cond {nc}: Mach {get_namelist_value(control_text, 'MachNumber')}, "
                 f"alpha {get_namelist_value(control_text, 'alpha')}, "
                 f"Re {get_namelist_value(control_text, 'ReynoldsNumber')}")

        last = {"t": 0.0, "rows": -1}

        def on_poll(elapsed):
            if elapsed - last["t"] < progress_every_s:
                return
            last["t"] = elapsed
            m = read_rsd(rsd)
            if m["rows"] != last["rows"]:
                last["rows"] = m["rows"]
                self.log(f"[2D][SOLVER]   {elapsed:6.0f} s  it {m['iterations']}  residual {m['residual']:.4g}  "
                         f"lift {m['CL']:.6g}  drag {m['CD']:.6g}")

        status, rc, wall = _run_piped(cmd, d, answers, log_file, time_limit_s, on_poll=on_poll,
                                      should_stop=should_stop, settle=(res,))
        m = read_rsd(rsd)
        if status != "finished":
            m["status"] = status
        elif m["status"] == "no_data":
            m["status"] = "failed"
        elif m["status"] == "ok" and (_runtime_error(log_file) or rc not in (0, None)):
            m["status"] = "crashed"                   # some iterations were written, then it died
        m.update(return_code=rc, wall_time_s=round(wall, 2), n=self.n, cond=int(nc),
                 mach=_as_float(get_namelist_value(control_text, "MachNumber")),
                 alpha=_as_float(get_namelist_value(control_text, "alpha")),
                 reynolds=_as_float(get_namelist_value(control_text, "ReynoldsNumber")),
                 has_result_file=bool(os.path.isfile(res) and os.path.getsize(res) > 0))
        # which iteration the flow field on disk belongs to, and whether it is whole
        interval = _as_float(get_namelist_value(control_text, "writeToFileInterval"))
        m["res_iteration"] = None
        if m["has_result_file"]:
            if m["status"] == "ok":
                m["res_iteration"] = m["iterations"]                  # written once more at the end
            elif interval and interval >= 1:
                m["res_iteration"] = int(m["iterations"] // int(interval) * int(interval)) or None
        m["res_complete"] = False
        if m["has_result_file"]:
            try:
                with open(res, "r", errors="ignore") as fh:
                    m["res_complete"] = sum(1 for ln in fh if ln.strip()) == self.mesh_node_count()
            except OSError:
                pass
        # how far lift and drag still moved at the end; a timed-out run whose forces had
        # settled is accepted with the numbers of its last cycle (see the top of the file)
        fc = self._force_check(rsd, fc_settings) if m["status"] in ("ok", "timeout") and m["rows"] else None
        m["force_check"] = fc
        m["converged_by"] = "residual" if m["status"] == "ok" else None
        if m["status"] == "timeout" and fc and fc.get("passed"):
            m["status"], m["converged_by"] = "ok", "forces"

        def moved():
            if not fc or fc.get("CL_spread") is None:
                return ""
            return (f"over the last {fc['window']} cycles the lift coefficient moved {fc['CL_spread']:.3g} and "
                    f"the drag coefficient {fc['CD_spread']:.3g} (tolerance {fc['tol']:.3g}, reference length "
                    f"{fc['ref_length']:.6g})")
        if m["converged_by"] == "forces":
            field = (f"the flow field on disk is that of iteration {m['res_iteration']}" if m["res_complete"]
                     else "no complete flow field on disk")
            self.log(f"[2D][SOLVER] cond {nc}: time limit reached after {wall:.0f} s at iteration "
                     f"{m['iterations']} (residual {m['residual']:.4g}, stop criterion NOT reached), but the "
                     f"forces had settled: {moved()}. ACCEPTED with the values of iteration {m['iterations']}: "
                     f"lift {m['CL']:.6g}, drag {m['CD']:.6g}, moment {m['CM']:.6g}; {field}.")
        elif m["status"] == "ok":
            self.log(f"[2D][SOLVER] cond {nc} finished in {wall:.0f} s: {m['iterations']} iterations, residual "
                     f"{m['residual']:.4g}, lift {m['CL']:.6g}, drag {m['CD']:.6g}, moment {m['CM']:.6g}"
                     + (f"; {moved()}" if moved() else ""))
        elif m["status"] in ("stopped", "timeout") and m["rows"]:
            why = "stopped by the user" if m["status"] == "stopped" else "time limit reached"
            field = (f"flow field on disk is that of iteration {m['res_iteration']}" if m["res_complete"]
                     else "no complete flow field on disk")
            forces = ""
            if m["status"] == "timeout":
                if fc is None:
                    forces = " The force check is off."
                elif fc.get("CL_spread") is not None:
                    forces = f" Forces NOT settled: {moved()}."
                else:
                    forces = f" Forces not checked: {fc.get('reason')}."
            self.log(f"[2D][SOLVER] cond {nc}: {why} after {wall:.0f} s at iteration {m['iterations']}, residual "
                     f"{m['residual']:.4g}, lift {m['CL']:.6g}, drag {m['CD']:.6g} (NOT converged to the "
                     f"stop criterion); {field}.{forces}")
        else:
            tail = _tail(log_file)
            self.log(f"[2D][SOLVER][ERROR] cond {nc}: {m['status']} after {wall:.0f} s "
                     f"(return code {rc}). Last lines of {log_file}:\n{tail}")
            if "access violation" in tail.lower() or "stack overflow" in tail.lower():
                self.log(_FIX_HINT)
        if self.make_plots:
            m["plots"] = self.write_plots(nc, m, control_text)
        with open(metrics_file, "w") as fh:
            json.dump(m, fh, indent=2)
        return m

    # -- whole chain -------------------------------------------------------------
    def run(self, opts: PreProOptions, control_template: str, conds: Optional[List[dict]] = None,
            boundary_flags: Optional[Iterable[int]] = None, solver_time_limit_s: Optional[float] = 3600.0,
            should_stop: Optional[Callable[[], bool]] = None, force_check=None) -> List[dict]:
        """
        PrePro once, then the solver for each flow condition. Every control
        file is checked BEFORE anything is run. Returns one metrics dict per
        condition (see read_rsd; "status" tells whether the numbers are usable).
        Raises RunStopped if should_stop() turns True before the solver has started.
        force_check: see run_solver (checked here, before anything runs).
        """
        force_check_settings(force_check)              # a bad entry fails now, not after PrePro
        conds = list(conds) if conds else [{}]
        texts = []
        for i, cond in enumerate(conds, 1):
            text, notes = apply_condition(control_template, cond)
            errors, warnings = check_control(text, opts.n_grids, boundary_flags, solver_time_limit_s)
            for w in dict.fromkeys(notes + warnings):
                self.log(f"[2D][WARN] cond {i}: {w}")
            if errors:
                raise ValueError(f"cond {i}: " + " ".join(errors))
            texts.append(text)
        if not (self._prepro_cmd and self._solver_cmd):
            bad = unfixed_executables(self.solver_dir)
            if bad:
                self.log(f"[2D][WARN] {' and '.join(bad)} still have known defects that can stop a run "
                         "(stack overflow / access violation). Repair them once with: "
                         "python Utilities/fix_flite2d_exes.py Solver")
        if self.make_plots and os.path.isfile(self.mesh_path):
            self.save_mesh_image()
        if should_stop and should_stop():
            raise RunStopped("stopped before the preprocessor started; nothing was computed.")
        self.run_prepro(opts, should_stop=should_stop)
        out = []
        for i, text in enumerate(texts, 1):
            if should_stop and should_stop():
                self.log(f"[2D] Stopped: flow condition(s) {i} to {len(texts)} were not run.")
                break
            out.append(self.run_solver(text, nc=i, time_limit_s=solver_time_limit_s, should_stop=should_stop,
                                       force_check=force_check))
        return out
