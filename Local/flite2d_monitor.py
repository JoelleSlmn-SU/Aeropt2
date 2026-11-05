"""
flite2d_monitor.py

Read-only view of the files a local FLITE 2D run leaves behind (see
Local/flite2d_local.py for who writes what). Used by the GUI to follow a run
while it is going on (Solver view, GUI/solver_monitor_2d.py) and to browse the
pictures of finished designs (Post-Processing view, GUI/post_view_2d.py). No Qt in here, and nothing in here writes or deletes a file.

Which run is "the current one"
------------------------------
A simulation and an optimisation run in a worker thread or in another process,
so this module does not ask them: it reads the folders. Three files are written
(and closed) at fixed moments of a design, which makes their modification times
reliable time stamps:

    volumes/n_<n>/<base>_<n>.dat                 the mesh of the design exists
    preprocessed/n_<n>/runPrePro.inp             the preprocessor is about to start
    solutions/n_<n>/cond_<i>/runSolver.inp       the solver is about to start on condition i

The files the solver has open (.rsd, solver.log) are NOT used as time stamps:
Windows does not update the modification time of a file reliably while another
program is still writing it.

    solutions/n_<n>/cond_<i>/metrics.json        removed when the solver starts, written when
                                                 it has ended and the pictures are done

so a condition folder with runSolver.inp and no metrics.json is a run in
progress (or one that was killed with the program that started it).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# pictures Local.flite2d_post writes per flow condition: (file name, tab title)
RUN_PICTURES = [("surface_cp.png", "Surface Cp"), ("mach.png", "Mach"), ("cp.png", "Cp"),
                ("total_pressure.png", "Total pressure"), ("convergence.png", "Convergence")]
WALL_FLAGS = (1, 5)                      # inviscid wall, viscous wall (as in Local.flite2d_post)


def _mtime(path: str) -> Optional[float]:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def format_duration(seconds: Optional[float]) -> str:
    """mm:ss, or h:mm:ss from one hour on; "" for no value."""
    if seconds is None or not math.isfinite(seconds):
        return ""
    s = int(max(0.0, seconds) + 0.5)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _numbered(root: str, prefix: str) -> List[int]:
    out = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", name)
        if m and os.path.isdir(os.path.join(root, name)):
            out.append(int(m.group(1)))
    return sorted(out)


# ---------------------------------------------------------------------------
# what is on disk
# ---------------------------------------------------------------------------
def design_numbers(output_dir: str) -> List[int]:
    """Designs that have a solver mesh or a solution folder, in increasing order."""
    with_mesh = [n for n in _numbered(os.path.join(output_dir, "volumes"), "n_") if mesh_path(output_dir, n)]
    return sorted(set(with_mesh) | set(_numbered(os.path.join(output_dir, "solutions"), "n_")))


def condition_numbers(output_dir: str, n: int) -> List[int]:
    return _numbered(os.path.join(output_dir, "solutions", f"n_{int(n)}"), "cond_")


def cond_dir(output_dir: str, n: int, nc: int) -> str:
    return os.path.join(output_dir, "solutions", f"n_{int(n)}", f"cond_{int(nc)}")


def mesh_path(output_dir: str, n: int) -> Optional[str]:
    """Solver mesh of design n: volumes/n_<n>/<base>_<n>.dat (None if there is none)."""
    hits = sorted(glob.glob(os.path.join(glob.escape(os.path.join(output_dir, "volumes", f"n_{int(n)}")),
                                         f"*_{int(n)}.dat")))
    return hits[0] if hits else None


def mesh_image_path(output_dir: str, n: int) -> Optional[str]:
    """surfaces/n_<n>/<base>_<n>.png, written before the preprocessor runs (None if missing)."""
    hits = sorted(glob.glob(os.path.join(glob.escape(os.path.join(output_dir, "surfaces", f"n_{int(n)}")),
                                         f"*_{int(n)}.png")))
    return hits[0] if hits else None


def _first(folder: str, pattern: str, exclude: Tuple[str, ...] = ()) -> Optional[str]:
    hits = [p for p in sorted(glob.glob(os.path.join(glob.escape(folder), pattern)))
            if os.path.basename(p) not in exclude]
    return hits[0] if hits else None


@dataclass
class RunState:
    """One flow condition of one design, as far as the files tell."""
    n: int
    cond: int
    folder: str
    exists: bool = False
    started: Optional[float] = None          # modification time of runSolver.inp
    finished: bool = False                   # metrics.json of THIS start is there
    ended: Optional[float] = None            # modification time of that metrics.json
    status: Optional[str] = None             # metrics.json "status" (ok, timeout, failed, ...)
    rsd: Optional[str] = None                # the residual file (may not exist yet)
    mach: Optional[float] = None
    alpha: Optional[float] = None
    reynolds: Optional[float] = None
    target_orders: Optional[float] = None    # |numberOfMGIterations| when it is negative
    metrics: Dict = field(default_factory=dict)

    def describe_condition(self) -> str:
        bits = []
        if self.mach is not None:
            bits.append(f"Mach {self.mach:g}")
        if self.alpha is not None:
            bits.append(f"alpha {self.alpha:g} deg")
        if self.reynolds:
            bits.append(f"Re {self.reynolds:g}")
        return ", ".join(bits)

    def run_time(self, now: Optional[float] = None) -> Optional[float]:
        """
        Seconds the solver has been running (pass the present time as `now`), or ran: the wall
        time the runner measured ("wall_time_s" of metrics.json; the pictures made afterwards are
        not part of it). None if the run was never started, or was started and left without a
        result (then nobody knows when it ended).
        """
        if self.started is None:
            return None
        if self.finished:
            t = self.metrics.get("wall_time_s")
            if isinstance(t, (int, float)) and math.isfinite(t):
                return float(t)
            return max(0.0, self.ended - self.started) if self.ended is not None else None
        return max(0.0, now - self.started) if now is not None else None


def _namelist_float(text: str, name: str) -> Optional[float]:
    m = re.search(r"(?im)^\s*ivd%" + re.escape(name) + r"\s*=\s*([^,\n!/]*)", text)
    if not m:
        return None
    try:
        return float(m.group(1).strip().lower().replace("d", "e"))
    except ValueError:
        return None


def run_state(output_dir: str, n: int, nc: int) -> RunState:
    d = cond_dir(output_dir, n, nc)
    st = RunState(n=int(n), cond=int(nc), folder=d, exists=os.path.isdir(d))
    if not st.exists:
        return st
    st.started = _mtime(os.path.join(d, "runSolver.inp"))
    # the .rsd is named in the answers the solver was given (5th line); fall back to any .rsd
    try:
        with open(os.path.join(d, "runSolver.inp"), "r", errors="ignore") as fh:
            answers = fh.read().splitlines()
        if len(answers) >= 5 and answers[4].strip():
            st.rsd = os.path.join(d, answers[4].strip())
        control = os.path.join(d, answers[0].strip()) if answers and answers[0].strip() else None
    except OSError:
        control = None
    if st.rsd is None:
        st.rsd = _first(d, "*.rsd")
    control = control if control and os.path.isfile(control) else _first(d, "*.inp", exclude=("runSolver.inp",))
    if control:
        try:
            with open(control, "r", errors="ignore") as fh:
                text = fh.read()
            st.mach = _namelist_float(text, "MachNumber")
            st.alpha = _namelist_float(text, "alpha")
            st.reynolds = _namelist_float(text, "ReynoldsNumber")
            its = _namelist_float(text, "numberOfMGIterations")
            st.target_orders = -its if its is not None and its < 0 else None
        except OSError:
            pass
    mfile = os.path.join(d, "metrics.json")
    mt = _mtime(mfile)
    if mt is not None and (st.started is None or mt >= st.started - 1.0):
        try:
            with open(mfile, "r") as fh:
                st.metrics = json.load(fh)
            st.finished = True
            st.ended = mt
            st.status = str(st.metrics.get("status", "")) or None
        except (OSError, ValueError):
            pass                              # being written right now: read it on the next look
    return st


def latest_run(output_dir: str) -> Optional[Tuple[int, Optional[int]]]:
    """
    (design, condition) that was started last, from the time stamps described at
    the top of this file. The condition is None while that design is still in
    the preprocessor (its solver has not been started since). None if there is
    no design at all.
    """
    best = None
    for n in design_numbers(output_dir):
        t_mesh = _mtime(mesh_path(output_dir, n) or "") or 0.0
        t_pre = _mtime(os.path.join(output_dir, "preprocessed", f"n_{n}", "runPrePro.inp")) or 0.0
        t_cond, cond = 0.0, None
        for nc in condition_numbers(output_dir, n):
            t = _mtime(os.path.join(cond_dir(output_dir, n, nc), "runSolver.inp")) or 0.0
            if t >= t_cond and t > 0.0:
                t_cond, cond = t, nc
        if cond is not None and t_cond < t_pre:
            cond = None                       # preprocessor started again after the last solver run
        t = max(t_mesh, t_pre, t_cond)
        if t > 0.0 and (best is None or t > best[0]):
            best = (t, n, cond)
    return (best[1], best[2]) if best else None


def prepro_started(output_dir: str, n: int) -> Optional[float]:
    """When the preprocessor of design n was last started (modification time of its answers file)."""
    return _mtime(os.path.join(output_dir, "preprocessed", f"n_{int(n)}", "runPrePro.inp"))


def latest_finished_run(output_dir: str) -> Optional[Tuple[int, int]]:
    """(design, condition) of the solver run that ENDED last (its metrics.json is the newest one
    that belongs to the run's current start). None if no run has ended yet."""
    best = None
    for n in design_numbers(output_dir):
        for nc in condition_numbers(output_dir, n):
            d = cond_dir(output_dir, n, nc)
            mt = _mtime(os.path.join(d, "metrics.json"))
            if mt is None:
                continue
            started = _mtime(os.path.join(d, "runSolver.inp"))
            if started is not None and mt < started - 1.0:
                continue                      # metrics of an earlier run; this one is still going
            if best is None or mt > best[0]:
                best = (mt, n, nc)
    return (best[1], best[2]) if best else None


def opt_history_path(output_dir: str) -> Optional[str]:
    """<output>/opt/history.png of the local optimisation (None if it has not been written)."""
    path = os.path.join(output_dir, "opt", "history.png")
    return path if os.path.isfile(path) else None


def run_pictures(output_dir: str, n: int, nc: Optional[int]) -> List[Tuple[str, Optional[str], str]]:
    """
    [(tab title, path or None, note)] for the pictures of one condition plus the
    surface-mesh picture of the design. A picture older than the start of the
    condition's current run belongs to an earlier run: it is reported as missing,
    with a note saying so, rather than shown as if it were the new result.
    """
    out = []
    started, ended = None, ""
    d = cond_dir(output_dir, n, nc) if nc is not None else None
    if d:
        st = run_state(output_dir, n, nc)
        started = st.started
        if st.finished:                       # the run is over: a picture that is not there will not come
            ended = (f"This run ended ({st.status or 'no status'}) without this picture"
                     + (": it has no complete flow field (the solver writes one every "
                        "ivd%writeToFileInterval cycles)." if not st.metrics.get("res_complete") else "."))
    for name, title in RUN_PICTURES:
        path = os.path.join(d, name) if d else None
        mt = _mtime(path) if path else None
        if mt is None:
            out.append((title, None, ended or "Not written yet. The pictures are made when the solver run "
                                              "has ended."))
        elif started is not None and mt < started - 1.0:
            out.append((title, None, (ended + " The file on disk is from an earlier run.") if ended else
                        "Not written yet for this run (the picture on disk is from an earlier run)."))
        else:
            out.append((title, path, ""))
    png = mesh_image_path(output_dir, n)
    out.append(("Surface mesh", png, "" if png else "No surface-mesh picture for this design."))
    return out


def opt_designs(output_dir: str) -> Dict[int, dict]:
    """{n: row of <output>/opt/designs.json} of the local optimisation; {} if there is none."""
    try:
        with open(os.path.join(output_dir, "opt", "designs.json"), "r", encoding="utf-8") as fh:
            rows = json.load(fh).get("designs", [])
        return {int(r["n"]): r for r in rows if isinstance(r, dict) and "n" in r}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def read_wall_outline(dat_path: str) -> List[Tuple[np.ndarray, bool]]:
    """
    Walls of a FLITE 2D solver mesh as [(points (k, 2), closed)], in the walking
    direction of the file. Only the boundary section and the coordinates of the
    wall nodes are parsed (the triangles are skipped), so this is quick enough
    for the GUI thread even on a large mesh. Raises ValueError on a file that is
    not a solver mesh.
    """
    from Local.flite2d_post import wall_chains
    with open(dat_path, "r", errors="replace") as fh:
        L = fh.read().split("\n")
    try:
        if L[4].strip().lower() != "connectivities":
            raise ValueError("no 'connectivities' line")
        ne, npn, nb = (int(float(v)) for v in L[3].split()[:3])
        ic = 5 + ne
        if L[ic].strip().lower() != "coordinates":
            raise ValueError(f"expected 'coordinates' on line {ic + 1}")
        ib = ic + 1 + npn
        if L[ib].strip().lower() != "boundaries":
            raise ValueError(f"expected 'boundaries' on line {ib + 1}")
        B = np.array([[int(float(v)) for v in L[ib + 1 + i].split()[:3]] for i in range(nb)],
                     np.int64).reshape(-1, 3)
        B[:, :2] -= 1
        chains = wall_chains(B, WALL_FLAGS)
        if not chains:
            return []
        ids = np.unique(np.concatenate([c for c, _ in chains]))
        if ids.min() < 0 or ids.max() >= npn:
            raise ValueError("wall node ids out of range")
        xy = {int(i): [float(v) for v in L[ic + 1 + int(i)].split()[1:3]] for i in ids}
        return [(np.array([xy[int(i)] for i in nodes], float).reshape(-1, 2), bool(closed))
                for nodes, closed in chains]
    except (IndexError, ValueError) as e:
        raise ValueError(f"{dat_path}: not a readable FLITE 2D solver mesh ({e}).")


def max_wall_displacement(walls, baseline_walls) -> Optional[float]:
    """Largest distance between corresponding wall nodes, or None if the two sets do not correspond
    (different number of chains or nodes). Every deformation of one baseline keeps the numbering."""
    if not walls or not baseline_walls or len(walls) != len(baseline_walls):
        return None
    worst = 0.0
    for (P, _), (Q, _) in zip(walls, baseline_walls):
        if P.shape != Q.shape:
            return None
        if len(P):
            worst = max(worst, float(np.hypot(*(P - Q).T).max()))
    return worst


# ---------------------------------------------------------------------------
# residual file of a run in progress
# ---------------------------------------------------------------------------
class RsdFollower:
    """
    Reads a .rsd incrementally: each poll() parses only what the solver has
    written since the last one. A file that disappears, shrinks, no longer
    holds what was read from it, or starts counting iterations again (the
    solver was started anew) resets the history.

        f = RsdFollower(path)
        if f.poll():            # True when the history changed
            H = f.history       # array (rows, 8): iteration, residual, lift, drag, moment,
                                #                  pressure lift, pressure drag, friction drag

    A row is used only once its line is complete (ends with a newline), so a
    line the solver is in the middle of writing is never read as a short number.
    """
    COLS = 8
    _CHECK = 64                          # bytes before the read position that are compared at every poll

    def __init__(self, path: Optional[str] = None):
        self.path = None
        self.error: Optional[str] = None
        self.set_path(path)

    def set_path(self, path: Optional[str]):
        self.path = path
        self.reset()

    def reset(self):
        self._pos = 0
        self._seen = b""                  # the last bytes read (to notice a file that was replaced)
        self._partial = b""
        self._rows: List[List[float]] = []
        self._array: Optional[np.ndarray] = None

    @property
    def history(self) -> np.ndarray:
        if self._array is None:
            self._array = np.array(self._rows, float).reshape(-1, self.COLS)
        return self._array

    @property
    def rows(self) -> int:
        return len(self._rows)

    def poll(self) -> bool:
        self.error = None
        had = len(self._rows)
        if not self.path:
            if had:
                self.reset()
            return bool(had)
        try:
            fh = open(self.path, "rb")
        except FileNotFoundError:
            if had or self._pos:
                self.reset()
            return bool(had)
        except OSError as e:                  # e.g. locked for a moment: keep what was read, try again
            self.error = str(e)
            return False
        changed = False
        try:
            with fh:
                # the size is taken from the open file, not from the directory: Windows can report
                # an old size for a file another program is still writing
                size = fh.seek(0, os.SEEK_END)
                if size < self._pos:          # replaced by a new, shorter file
                    self.reset()
                    changed = bool(had)
                if self._seen:
                    # a new file that has already grown past the old read position looks, by its
                    # size, like the old one with more rows: compare what precedes the position
                    fh.seek(self._pos - len(self._seen))
                    if fh.read(len(self._seen)) != self._seen:
                        self.reset()
                        changed = bool(had)
                fh.seek(self._pos)
                data = fh.read()
        except OSError as e:
            self.error = str(e)
            return changed
        if not data:
            return changed
        self._pos += len(data)
        self._seen = (self._seen + data)[-self._CHECK:]
        lines = (self._partial + data).split(b"\n")
        self._partial = lines.pop()           # what follows the last newline is not a whole line yet
        for raw in lines:
            try:
                v = [float(t.lower().replace("d", "e"))
                     for t in raw.decode("ascii", "ignore").replace(",", " ").split()]
            except ValueError:
                continue
            if len(v) < 4:
                continue
            if self._rows and math.isfinite(v[0]) and math.isfinite(self._rows[-1][0]) and v[0] <= self._rows[-1][0]:
                self._rows = []               # iteration count went back: these rows are a new run
            self._rows.append((v + [math.nan] * self.COLS)[:self.COLS])
            changed = True
        if changed:
            self._array = None
        return changed
