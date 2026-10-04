"""
flite2d_post.py

Pictures and tables of a local FLITE 2D run (see Local/flite2d_local.py). No Qt
and no pyplot in here: every figure is a matplotlib `Figure` drawn with the Agg
canvas, so the functions are safe to call from a worker thread while the GUI
(pyvista / Qt) is running, and on a machine without a display.

What is written
---------------
Per design n (once, before the preprocessor runs, and whenever the GUI writes
the solver mesh):

    <output>/surfaces/n_<n>/<base>_<n>.png      wall + near-body mesh; for n > 0 the
                                                baseline wall is drawn underneath

Per flow condition, in  <output>/solutions/n_<n>/cond_<i>/ :

    convergence.png        residual, lift, drag, moment against iteration (from the .rsd)
    surface_cp.png         pressure coefficient along every wall
    <base>_<n>_surface.csv the same wall data as numbers (x, y, arc length, Cp, Mach, p)
    mach.png               Mach number near the body, sonic line marked
    cp.png                 pressure coefficient near the body
    total_pressure.png     p0 / p0_inf near the body

Quantities
----------
The .res rows are rho, u, v, E, p in the solver's units: rho_inf = 1,
|u_inf| = 1, p_inf = 1 / (gamma M_inf^2). From these

    Mach      = |u| / sqrt(gamma p / rho)
    Cp        = (p - p_inf) / (rho_inf u_inf^2 / 2) = 2 (p - p_inf)
    p0/p0_inf = p (1 + (gamma-1)/2 M^2)^(gamma/(gamma-1)) / the same at infinity

In a steady inviscid flow p0/p0_inf is exactly 1 everywhere except behind
shocks. Any other departure from 1 in an inviscid run is numerical (artificial
dissipation, coarse mesh), which makes that picture the quickest check of how
much of a computed drag is real.

The .res file is rewritten by the solver every `ivd%writeToFileInterval`
iterations. A run that is stopped or timed out therefore has a flow field from
the last multiple of that interval, not from the last row of the .rsd; the
titles say which iteration the field belongs to.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WALL_FLAGS = (1, 5)                      # inviscid wall, viscous wall

# figure colours (fixed; text never takes a series colour)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd7"
BODY, MESH = "#d5d4cf", "#8f8e88"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DPI = 200


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def read_res(path: str, n_nodes: Optional[int] = None) -> np.ndarray:
    """
    Solver result file -> array (n, 5): rho, u, v, E, p, in the node order of
    the solver mesh. The first column (node number) is ignored: it is 5
    characters wide and reads '*****' from node 100,000 on.
    Raises ValueError if the file is incomplete (the solver was killed while
    writing it) - pass n_nodes to have the row count checked.
    """
    try:
        R = np.loadtxt(path, usecols=(1, 2, 3, 4, 5), ndmin=2)
    except (ValueError, IndexError):
        rows = []
        with open(path, "r", errors="ignore") as fh:
            for ln in fh:
                t = ln.split()
                if len(t) < 6:
                    break
                try:
                    rows.append([float(v.lower().replace("d", "e")) for v in t[-5:]])
                except ValueError:
                    break
        R = np.array(rows, float).reshape(-1, 5)
    if n_nodes is not None and len(R) != int(n_nodes):
        raise ValueError(f"{os.path.basename(path)} has {len(R)} rows, the mesh has {int(n_nodes)} nodes "
                         "(the solver was stopped while it was writing the file).")
    return R


def read_rsd_history(path: str) -> np.ndarray:
    """All complete rows of the .rsd as an array (rows, 8); missing columns are NaN."""
    rows = []
    try:
        with open(path, "r", errors="ignore") as fh:
            for ln in fh:
                try:
                    v = [float(t.lower().replace("d", "e")) for t in ln.replace(",", " ").split()]
                except ValueError:
                    continue
                if len(v) >= 4:
                    rows.append((v + [math.nan] * 8)[:8])
    except OSError:
        pass
    return np.array(rows, float).reshape(-1, 8)


def flow_fields(R: np.ndarray, mach_inf: float, gamma: float = 1.4) -> Dict[str, np.ndarray]:
    """Mach, Cp and p0/p0_inf at every node from the .res columns (see the module text)."""
    rho, u, v, p = R[:, 0], R[:, 1], R[:, 2], R[:, 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        mach = np.hypot(u, v) / np.sqrt(gamma * p / rho)
        p_inf = 1.0 / (gamma * mach_inf ** 2)
        ex = gamma / (gamma - 1.0)
        p0 = p * (1.0 + 0.5 * (gamma - 1.0) * mach ** 2) ** ex
        p0_inf = p_inf * (1.0 + 0.5 * (gamma - 1.0) * mach_inf ** 2) ** ex
        return {"mach": mach, "cp": 2.0 * (p - p_inf), "p0_ratio": p0 / p0_inf, "p": p}


def critical_cp(mach_inf: float, gamma: float = 1.4) -> Optional[float]:
    """Cp at which the local flow is sonic (isentropic). None for M_inf >= 1."""
    if not (0.0 < mach_inf < 1.0):
        return None
    g = gamma
    return 2.0 / (g * mach_inf ** 2) * (((2.0 + (g - 1.0) * mach_inf ** 2) / (g + 1.0)) ** (g / (g - 1.0)) - 1.0)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def wall_chains(boundary: np.ndarray, wall_flags: Sequence[int] = WALL_FLAGS) -> List[Tuple[np.ndarray, bool]]:
    """
    Wall boundary edges linked into chains of node indices, each in the walking
    direction of the file. Returns [(nodes, closed)]; a closed chain does not
    repeat its first node.
    """
    B = np.asarray(boundary)
    rows = B[np.isin(B[:, 2], list(wall_flags))][:, :2] if len(B) else np.zeros((0, 2), int)
    if not len(rows):
        return []
    nxt: Dict[int, int] = {}
    for a, b in rows:
        if int(a) in nxt:                 # a node with two outgoing wall edges: fall back to row order
            return _chains_by_row_order(rows)
        nxt[int(a)] = int(b)
    targets = set(nxt.values())
    starts = [a for a in nxt if a not in targets]          # open chains begin here
    seen, chains = set(), []
    for s in starts + list(nxt):
        if s in seen or s not in nxt:
            continue
        chain, a = [s], s
        seen.add(s)
        while a in nxt and nxt[a] not in seen:
            a = nxt[a]
            chain.append(a)
            seen.add(a)
        closed = a in nxt and nxt[a] == chain[0] and len(chain) > 2
        chains.append((np.array(chain, int), bool(closed)))
    return chains


def _chains_by_row_order(rows: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
    chains, cur = [], [int(rows[0, 0]), int(rows[0, 1])]
    for a, b in rows[1:]:
        if int(a) == cur[-1]:
            cur.append(int(b))
        else:
            chains.append(cur)
            cur = [int(a), int(b)]
    chains.append(cur)
    out = []
    for c in chains:
        closed = len(c) > 3 and c[0] == c[-1]
        out.append((np.array(c[:-1] if closed else c, int), closed))
    return out


def body_window(X: np.ndarray, boundary: np.ndarray, pad: float = 0.5) -> Tuple[float, float, float, float]:
    """
    (xmin, xmax, ymin, ymax) around the walls: their bounding box grown on every
    side by pad x its larger dimension. The whole mesh if there is no wall.
    """
    B = np.asarray(boundary)
    ids = np.unique(B[np.isin(B[:, 2], list(WALL_FLAGS))][:, :2]) if len(B) else np.array([], int)
    P = X[ids] if len(ids) else X
    lo, hi = P.min(axis=0), P.max(axis=0)
    if not len(ids):
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])
    d = pad * float(max(hi - lo))
    glo, ghi = X.min(axis=0), X.max(axis=0)                # never beyond the mesh itself
    return (float(max(lo[0] - d, glo[0])), float(min(hi[0] + d, ghi[0])),
            float(max(lo[1] - d, glo[1])), float(min(hi[1] + d, ghi[1])))


def _triangles_in(X, T, win, margin=0.02):
    x0, x1, y0, y1 = win
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    inside = (X[:, 0] >= x0 - mx) & (X[:, 0] <= x1 + mx) & (X[:, 1] >= y0 - my) & (X[:, 1] <= y1 + my)
    return T[inside[T].any(axis=1)]


# ---------------------------------------------------------------------------
# figure plumbing
# ---------------------------------------------------------------------------
def _figure(width, height):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(width, height), dpi=DPI, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _style(ax, grid=True):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=8, width=0.6, length=3)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def _titles(fig, title, subtitle=None):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=11, color=INK, fontweight="bold")
    if subtitle:
        fig.text(0.012, 0.985 - 0.26 / fig.get_figheight(), subtitle, ha="left", va="top",
                 fontsize=8.5, color=MUTED)


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=DPI, facecolor="white")
    try:
        os.replace(tmp, path)
    except OSError as e:                  # e.g. the old picture is open in a viewer that locks it
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise OSError(f"could not replace {path} (is it open in another program?): {e}")
    return path


def _draw_walls(ax, X, chains, fill=True, lw=0.9):
    for nodes, closed in chains:
        P = X[nodes]
        if closed:
            if fill:
                ax.fill(P[:, 0], P[:, 1], color=BODY, linewidth=0, zorder=3)
            P = np.vstack([P, P[:1]])
        ax.plot(P[:, 0], P[:, 1], color=INK, linewidth=lw, zorder=4, solid_joinstyle="round")


def _window_axes(win, width=9.0, extra_w=0.0, top=0.62, bottom=0.5, left=0.7, right=0.25):
    """Figure and axes whose data area has the aspect ratio of the window."""
    x0, x1, y0, y1 = win
    aw = width - left - right - extra_w
    ah = aw * (y1 - y0) / max(x1 - x0, 1e-300)
    ah = min(max(ah, 1.6), 9.0)
    height = ah + top + bottom
    fig = _figure(width, height)
    ax = fig.add_axes([left / width, bottom / height, aw / width, ah / height])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    return fig, ax, (left, bottom, aw, ah, width, height)


# ---------------------------------------------------------------------------
# surface mesh image (one per design)
# ---------------------------------------------------------------------------
def plot_surface_mesh(X: np.ndarray, T: np.ndarray, boundary: np.ndarray, path: str,
                      title: str = "Surface mesh", baseline_X: Optional[np.ndarray] = None,
                      baseline_boundary: Optional[np.ndarray] = None, pad: float = 0.15) -> str:
    """
    Walls (line + nodes) on top of the mesh around them. With a baseline given,
    its walls are drawn underneath and the largest wall displacement is stated.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    chains = wall_chains(boundary)
    win = body_window(X, boundary, pad=pad)
    if baseline_X is not None and baseline_boundary is not None:      # same picture frame for every design
        b = body_window(baseline_X, baseline_boundary, pad=pad)
        win = (min(win[0], b[0]), max(win[1], b[1]), min(win[2], b[2]), max(win[3], b[3]))
    fig, ax, _ = _window_axes(win, width=12.0)
    _style(ax, grid=False)

    Tw = _triangles_in(X, T, win)
    E = np.vstack([Tw[:, [0, 1]], Tw[:, [1, 2]], Tw[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    ax.add_collection(LineCollection(X[E], colors=MESH, linewidths=0.25, zorder=1))

    wall_ids = np.unique(np.concatenate([c for c, _ in chains])) if chains else np.array([], int)
    handles, moved = [], None
    if baseline_X is not None and baseline_boundary is not None:
        for nodes, closed in wall_chains(baseline_boundary):
            P = baseline_X[nodes]
            P = np.vstack([P, P[:1]]) if closed else P
            ax.plot(P[:, 0], P[:, 1], color=SERIES[1], linewidth=1.1, linestyle=(0, (4, 2)), zorder=5)
        handles.append(Line2D([], [], color=SERIES[1], linewidth=1.1, linestyle=(0, (4, 2)), label="baseline wall"))
        if baseline_X.shape == X.shape and len(wall_ids):
            moved = float(np.linalg.norm(X[wall_ids] - baseline_X[wall_ids], axis=1).max())
    _draw_walls(ax, X, chains, fill=False, lw=1.0)
    if len(wall_ids):
        ax.plot(X[wall_ids, 0], X[wall_ids, 1], linestyle="none", marker="o", markersize=1.8,
                color=SERIES[0], zorder=6)
    handles.append(Line2D([], [], color=INK, linewidth=1.0, marker="o", markersize=3,
                          markerfacecolor=SERIES[0], markeredgecolor=SERIES[0], label="wall and its nodes"))
    handles.append(Line2D([], [], color=MESH, linewidth=0.6, label="mesh"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
              edgecolor=GRID, labelcolor=MUTED)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    sub = f"{len(X):,} nodes, {len(T):,} triangles, {len(wall_ids):,} wall nodes"
    if moved is not None:
        sub += f"; largest wall displacement from the baseline {moved:.4g}"
    _titles(fig, title, sub)
    return _save(fig, path)


def save_mesh_image(mesh_dat: str, png_path: str, title: Optional[str] = None,
                    baseline_dat: Optional[str] = None) -> str:
    """Surface-mesh picture of a solver mesh file (see plot_surface_mesh)."""
    from FileRW.FliteMesh2D import read_flite2d_mesh
    m = read_flite2d_mesh(mesh_dat)
    bX = bB = None
    if baseline_dat and os.path.isfile(baseline_dat) and \
            os.path.abspath(baseline_dat) != os.path.abspath(mesh_dat):
        b = read_flite2d_mesh(baseline_dat)
        bX, bB = b["nodes"], b["boundary"]
    return plot_surface_mesh(m["nodes"], m["triangles"], m["boundary"], png_path,
                             title=title or os.path.splitext(os.path.basename(mesh_dat))[0],
                             baseline_X=bX, baseline_boundary=bB)


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------
def plot_convergence(history: np.ndarray, path: str, title: str = "Convergence",
                     subtitle: Optional[str] = None, target_orders: Optional[float] = None) -> str:
    """
    Residual, lift, drag and moment against iteration, one panel each (the four
    have unrelated scales). The force panels are scaled to the last 80 % of the
    run so the start-up transient does not flatten the part that matters.
    """
    H = np.asarray(history, float)
    it = H[:, 0]
    fig = _figure(10.0, 6.4)
    axes = fig.subplots(2, 2, sharex=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.085, hspace=0.3, wspace=0.24)
    panels = [(1, "residual  (log10 of residual / first residual)"), (2, "lift"), (3, "drag"), (4, "moment")]
    for ax, (col, name) in zip(axes.ravel(), panels):
        _style(ax)
        y = H[:, col]
        ax.plot(it, y, color=SERIES[0], linewidth=1.4)
        last = y[-1] if len(y) else math.nan
        ax.set_title(f"{name}:  {last:.6g}" if math.isfinite(last) else f"{name}:  not a number",
                     loc="left", fontsize=9, color=INK)
        ok = np.isfinite(y)
        if col == 1:
            if target_orders:
                ax.axhline(-abs(target_orders), color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
                ax.annotate(f"stop criterion {-abs(target_orders):g}", xy=(1.0, -abs(target_orders)),
                            xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                            ha="right", fontsize=7.5, color=MUTED)
        elif ok.sum() > 20:
            tail = y[int(0.2 * len(y)):]
            tail = tail[np.isfinite(tail)]
            if len(tail):
                lo, hi = float(tail.min()), float(tail.max())
                m = 0.15 * (hi - lo) if hi > lo else (abs(hi) * 0.05 or 1.0)
                ax.set_ylim(lo - m, hi + m)
    for ax in axes[1]:
        ax.set_xlabel("multigrid cycle")
    _titles(fig, title, subtitle)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# surface pressure
# ---------------------------------------------------------------------------
def surface_table(X: np.ndarray, boundary: np.ndarray, F: Dict[str, np.ndarray]) -> List[dict]:
    """One dict per wall chain: nodes (0-based, file numbering), closed, x, y, s, cp, mach, p."""
    out = []
    for nodes, closed in wall_chains(boundary):
        ids = np.append(nodes, nodes[0]) if closed else nodes
        P = X[ids]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        out.append({"nodes": ids, "closed": closed, "x": P[:, 0], "y": P[:, 1], "s": s,
                    "cp": F["cp"][ids], "mach": F["mach"][ids], "p": F["p"][ids]})
    return out


def write_surface_csv(table: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wall", "node", "x", "y", "arc_length", "cp", "mach", "p"])
        for k, c in enumerate(table, 1):
            for i in range(len(c["nodes"])):
                w.writerow([k, int(c["nodes"][i]) + 1] + [f"{c[q][i]:.8g}" for q in ("x", "y", "s", "cp", "mach", "p")])
    return path


def plot_surface_cp(table: List[dict], path: str, title: str = "Surface pressure",
                    subtitle: Optional[str] = None, cp_crit: Optional[float] = None) -> str:
    """Cp against x for every wall, suction upwards (the aerodynamic convention)."""
    fig = _figure(9.0, 5.4)
    ax = fig.add_axes([0.085, 0.1, 0.895, 0.74])
    _style(ax)
    many = len(table) > len(SERIES)
    for k, c in enumerate(table):
        ax.plot(c["x"], c["cp"], color=SERIES[0] if many else SERIES[k], linewidth=1.5,
                label=None if many else f"wall {k + 1}")
    if 1 < len(table) <= len(SERIES):
        ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    ax.axhline(0.0, color=MUTED, linewidth=0.6)
    if cp_crit is not None and math.isfinite(cp_crit):
        ax.axhline(cp_crit, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(f"sonic, Cp* = {cp_crit:.3f}", xy=(1.0, cp_crit), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3), textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("Cp  (suction up)")
    _titles(fig, title, subtitle)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# flow-field pictures
# ---------------------------------------------------------------------------
def plot_field(X: np.ndarray, T: np.ndarray, boundary: np.ndarray, values: np.ndarray, path: str,
               label: str, title: str, subtitle: Optional[str] = None, kind: str = "sequential",
               centre: float = 0.0, line_at: Optional[float] = None, line_label: Optional[str] = None,
               window: Optional[Tuple[float, float, float, float]] = None, n_levels: int = 40) -> Optional[str]:
    """
    Filled contours of a nodal quantity near the body.
    kind "sequential": levels from the smallest to the largest value in view.
    kind "diverging" : two-sided scale symmetric about `centre` (neutral there).
    line_at: also draw the contour of this value (e.g. Mach 1) in ink.
    Returns None (nothing written) if there is no finite value in view.
    """
    from matplotlib.tri import Triangulation
    from matplotlib.lines import Line2D
    win = window or body_window(X, boundary)
    Tw = _triangles_in(X, T, win)
    v = np.asarray(values, float)
    Tw = Tw[np.isfinite(v[Tw]).all(axis=1)]
    if not len(Tw):
        return None
    seen = v[np.unique(Tw)]
    lo, hi = float(seen.min()), float(seen.max())
    if kind == "diverging":
        a = max(abs(hi - centre), abs(lo - centre), 1e-9 * max(abs(centre), 1.0))
        levels, cmap = np.linspace(centre - a, centre + a, n_levels + 1), "RdBu_r"
    else:
        if hi - lo < 1e-9 * max(abs(lo), 1.0):          # a uniform field still gets a picture
            hi = lo + 1e-9 * max(abs(lo), 1.0)
        levels, cmap = np.linspace(lo, hi, n_levels + 1), "viridis"

    fig, ax, (left, bottom, aw, ah, width, height) = _window_axes(win, width=9.0, extra_w=0.95)
    _style(ax, grid=False)
    ax.set_facecolor(BODY)
    tri = Triangulation(X[:, 0], X[:, 1], Tw)
    filled = ax.tricontourf(tri, np.where(np.isfinite(v), v, lo), levels=levels, cmap=cmap, zorder=1)
    if line_at is not None and lo < line_at < hi:
        ax.tricontour(tri, np.where(np.isfinite(v), v, lo), levels=[line_at], colors=[INK],
                      linewidths=0.9, zorder=2)
        ax.legend(handles=[Line2D([], [], color=INK, linewidth=0.9, label=line_label or f"{label} = {line_at:g}")],
                  loc="upper right", fontsize=8, frameon=True, framealpha=0.9, edgecolor=GRID, labelcolor=MUTED)
    _draw_walls(ax, X, wall_chains(boundary))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cax = fig.add_axes([(left + aw + 0.18) / width, bottom / height, 0.16 / width, ah / height])
    cb = fig.colorbar(filled, cax=cax)
    cb.set_label(label, fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED, width=0.6, length=3)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(MUTED)
    _titles(fig, title, (subtitle + "   " if subtitle else "") + f"range in view {lo:.4g} to {hi:.4g}")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# everything for one flow condition
# ---------------------------------------------------------------------------
def write_run_plots(cond_dir: str, stem: str, mesh: dict, metrics: dict, gamma: float = 1.4,
                    target_orders: Optional[float] = None, log=print) -> List[str]:
    """
    Write every picture that the files in `cond_dir` allow. `mesh` is the dict
    of FileRW.FliteMesh2D.read_flite2d_mesh for the mesh that was run; `metrics`
    the dict of Flite2DCase.run_solver (status, iterations, mach, alpha,
    res_iteration ...). One broken picture never stops the others: the reason
    is logged and the rest are still written. Returns the paths written.
    """
    written = []
    rsd, res = os.path.join(cond_dir, stem + ".rsd"), os.path.join(cond_dir, stem + ".res")
    mach_inf, alpha = metrics.get("mach"), metrics.get("alpha")
    status = metrics.get("status", "")
    head = f"{stem}, condition {metrics.get('cond', '')}"
    cond = ", ".join(s for s in (f"Mach {mach_inf:g}" if mach_inf is not None else "",
                                 f"alpha {alpha:g} deg" if alpha is not None else "") if s)
    state = "" if status == "ok" else f"   [{status.upper()} - not a finished run]"

    def attempt(what, fn):
        try:
            p = fn()
            if p:
                written.append(p)
        except Exception as e:                       # a picture must never cost the run its result
            log(f"[2D][PLOT][WARN] {what} not written: {e}")

    H = read_rsd_history(rsd)
    if len(H):
        attempt("convergence.png", lambda: plot_convergence(
            H, os.path.join(cond_dir, "convergence.png"), title=f"Convergence - {head}",
            subtitle=f"{cond}; {int(H[-1, 0])} multigrid cycles{state}", target_orders=target_orders))
    else:
        log("[2D][PLOT][WARN] no residual history, so no convergence picture.")

    if not (os.path.isfile(res) and os.path.getsize(res) > 0):
        log("[2D][PLOT][WARN] no flow field on disk (the solver writes it every ivd%writeToFileInterval "
            "iterations and the run ended before the first one), so no flow pictures.")
        return written
    if mach_inf is None or not mach_inf > 0:
        log("[2D][PLOT][WARN] free-stream Mach number unknown, so no flow pictures.")
        return written
    X, T, B = mesh["nodes"], mesh["triangles"], mesh["boundary"]
    try:
        R = read_res(res, n_nodes=len(X))
    except (OSError, ValueError) as e:
        log(f"[2D][PLOT][WARN] no flow pictures: {e}")
        return written
    F = flow_fields(R, float(mach_inf), gamma)
    at = metrics.get("res_iteration")
    sub = cond + (f"; flow field of cycle {at}" if at else "") + state
    table = surface_table(X, B, F)
    if table:
        attempt("surface_cp.png", lambda: plot_surface_cp(
            table, os.path.join(cond_dir, "surface_cp.png"), title=f"Surface pressure - {head}",
            subtitle=sub, cp_crit=critical_cp(float(mach_inf), gamma)))
        attempt("surface csv", lambda: write_surface_csv(table, os.path.join(cond_dir, stem + "_surface.csv")))
    win = body_window(X, B)
    attempt("mach.png", lambda: plot_field(
        X, T, B, F["mach"], os.path.join(cond_dir, "mach.png"), "Mach number", f"Mach number - {head}", sub,
        line_at=1.0, line_label="sonic line (Mach 1)", window=win))
    attempt("cp.png", lambda: plot_field(
        X, T, B, F["cp"], os.path.join(cond_dir, "cp.png"), "Cp", f"Pressure coefficient - {head}", sub,
        kind="diverging", centre=0.0, window=win))
    attempt("total_pressure.png", lambda: plot_field(
        X, T, B, F["p0_ratio"], os.path.join(cond_dir, "total_pressure.png"), "p0 / p0 free stream",
        f"Total pressure - {head}", sub, kind="diverging", centre=1.0, window=win))
    return written


# ---------------------------------------------------------------------------
# optimisation history (Local.local2DOpt)
# ---------------------------------------------------------------------------
def plot_opt_history(designs: List[dict], path: str, title: str = "Optimisation history") -> Optional[str]:
    """
    Objective of every design against its number n, with the best value so far.
    `designs`: the records of Local2DTestManager (keys n, gen, objective,
    baseline, feasible). Designs without a result are marked on the bottom
    edge; designs that break a constraint are drawn hollow and do not count
    for "best so far". Returns None if no design has a result yet.
    """
    from matplotlib.lines import Line2D
    D = sorted([d for d in designs if d.get("done")], key=lambda d: d["n"])
    have = [d for d in D if d.get("objective") is not None]
    if not have:
        return None
    fig = _figure(9.0, 5.0)
    ax = fig.add_axes([0.09, 0.11, 0.89, 0.72])
    _style(ax)
    ok = [d for d in have if d.get("feasible", True)]
    bad = [d for d in have if not d.get("feasible", True)]
    none = [d for d in D if d.get("objective") is None]
    ys = [d["objective"] for d in have]
    lo, hi = min(ys), max(ys)
    pad = 0.08 * (hi - lo) if hi > lo else (0.05 * abs(hi) or 1.0)
    ax.set_ylim(lo - 2.2 * pad, hi + pad)

    best, bx, by = math.inf, [], []
    for d in D:                                            # best so far, in the order designs were run
        if d.get("objective") is not None and d.get("feasible", True) and d["objective"] < best:
            best = d["objective"]
        if math.isfinite(best):
            bx.append(d["n"])
            by.append(best)
    handles = []
    if bx:
        ax.step(bx, by, where="post", color=SERIES[1], linewidth=2.0, zorder=2)
        handles.append(Line2D([], [], color=SERIES[1], linewidth=2.0, label="best so far"))
    if ok:
        ax.plot([d["n"] for d in ok], [d["objective"] for d in ok], linestyle="none", marker="o", markersize=6,
                color=SERIES[0], markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6, color=SERIES[0], label="design"))
    if bad:
        ax.plot([d["n"] for d in bad], [d["objective"] for d in bad], linestyle="none", marker="o", markersize=6,
                markerfacecolor="white", markeredgecolor=SERIES[0], markeredgewidth=1.3, zorder=3)
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor="white",
                              markeredgecolor=SERIES[0], markeredgewidth=1.3, label="breaks a constraint"))
    if none:
        ax.plot([d["n"] for d in none], [lo - 1.5 * pad] * len(none), linestyle="none", marker="x", markersize=6,
                color=MUTED, markeredgewidth=1.3, zorder=3, clip_on=False)
        handles.append(Line2D([], [], linestyle="none", marker="x", markersize=6, color=MUTED,
                              markeredgewidth=1.3, label="no result (invalid mesh or failed run)"))
    base = next((d for d in D if d.get("baseline") and d.get("objective") is not None), None)
    if base is not None:
        ax.axhline(base["objective"], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(f"baseline {base['objective']:.5g}", xy=(1.0, base["objective"]),
                    xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=7.5, color=MUTED)
    first = [d["n"] for d in D if d["gen"] == 0]
    later = [d["n"] for d in D if d["gen"] > 0]
    if first and later:
        x_split = 0.5 * (max(first) + min(later))
        ax.axvline(x_split, color=GRID, linewidth=1.0, zorder=0)
        ax.annotate("initial sample ", xy=(x_split, 1.0), xycoords=("data", "axes fraction"), xytext=(-3, -3),
                    textcoords="offset points", ha="right", va="top", fontsize=7.5, color=MUTED)
        ax.annotate(" optimiser", xy=(x_split, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -3),
                    textcoords="offset points", ha="left", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(min(d["n"] for d in D) - 0.6, max(d["n"] for d in D) + 0.6)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("design number n")
    ax.set_ylabel("objective (minimised)")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(handles), fontsize=8,
              frameon=False, labelcolor=MUTED)
    fig.axes[0].set_position([0.09, 0.17, 0.89, 0.66])
    sub = f"{len(D)} design(s), {len(none)} without a result"
    if math.isfinite(best):
        sub += f"; best {best:.6g}"
        if base is not None and base["objective"]:
            sub += f" ({100.0 * (best - base['objective']) / abs(base['objective']):+.2f} % against the baseline)"
    _titles(fig, title, sub)
    return _save(fig, path)
