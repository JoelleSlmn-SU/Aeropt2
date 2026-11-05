"""Non-dominated set of the evaluated DSI designs (Mach 1.6 sheet 'Results').

Usage:  python pareto_dsi.py DSI_Optimisation_Results.xlsx [output_dir]

Reads column A = folder number, PR, Normalised Drag, Weighted Combination, DC60.
Writes pareto_PR_CD.png, pareto_PR_DC60.png and nondominated.csv.
The batch of a design is taken from its folder number (see BATCHES): edit that
table when new designs are added.
"""
import sys
from pathlib import Path

import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# folder range (inclusive) -> (label, colour, marker)
BATCHES = [
    ((15, 28), ("LHS, old parameterisation (14)", "#2a78d6", "o")),
    ((2, 14),  ("LHS, new parameterisation (13)", "#eb6834", "s")),
    ((29, 31), ("Optimiser, weighted J (3)",       "#1baf7a", "^")),
    ((32, 34), ("Optimiser, drag only (3)",        "#1baf7a", "v")),
]
BASELINE = 1
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read_results(xlsx):
    ws = openpyxl.load_workbook(xlsx, data_only=True)["Results"]
    rows = [r[:5] for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    a = np.array(rows, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Results sheet has empty or non-numeric cells")
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 4]      # n, PR, CD, DC60


def nondominated(cost):
    """cost: (n, m) array, every column to be minimised. Returns boolean mask."""
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        better_eq = np.all(cost <= cost[i], axis=1)
        strictly = np.any(cost < cost[i], axis=1)
        keep[i] = not np.any(better_eq & strictly)
    return keep


def plot(n, x, y, xlabel, ylabel, title, path, label_offsets=None):
    label_offsets = label_offsets or {}
    front = nondominated(np.column_stack([x, -y]))               # min x, max y
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    order = np.argsort(x[front])
    fx, fy = x[front][order], y[front][order]
    ax.plot(fx, fy, drawstyle="steps-post",
            color=INK, lw=1.4, zorder=2, label="Non-dominated set")

    for (lo, hi), (lab, col, mk) in BATCHES:
        m = (n >= lo) & (n <= hi)
        ax.scatter(x[m], y[m], s=70, c=col, marker=mk, edgecolors=SURFACE,
                   linewidths=1.2, zorder=3, label=lab)
    b = n == BASELINE
    ax.scatter(x[b], y[b], s=230, c=INK, marker="*", edgecolors=SURFACE,
               linewidths=1.2, zorder=5, label="Baseline")

    for i in np.where(front | b)[0]:
        dx, dy = label_offsets.get(int(n[i]), (8, 6))
        txt = "Baseline" if n[i] == BASELINE else f"{n[i]}"
        ax.annotate(txt, (x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=10, color=INK, fontweight="bold", zorder=6)

    ax.annotate("better", xy=(0.035, 0.955), xytext=(0.12, 0.84), xycoords="axes fraction",
                fontsize=10, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_xlabel(xlabel, fontsize=11, color=INK)
    ax.set_ylabel(ylabel, fontsize=11, color=INK)
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.legend(loc="lower left", fontsize=9.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return front


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    xlsx = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else xlsx.parent
    out.mkdir(parents=True, exist_ok=True)
    n, pr, cd, dc = read_results(xlsx)

    f1 = plot(n, cd, pr, "Normalised drag  (lower is better)", "Pressure recovery  (higher is better)",
              "Mach 1.6: pressure recovery against drag, 34 evaluated designs",
              out / "pareto_PR_CD.png", {1: (-62, -14), 15: (-22, -4), 19: (-22, 4), 18: (8, 2), 7: (8, 6)})
    f2 = plot(n, dc, pr, "DC60  (lower is better)", "Pressure recovery  (higher is better)",
              "Mach 1.6: pressure recovery against distortion, 34 evaluated designs",
              out / "pareto_PR_DC60.png", {1: (-62, 4), 12: (8, -12), 15: (-24, 5)})
    f3 = nondominated(np.column_stack([-pr, cd, dc]))
    with open(out / "nondominated.csv", "w") as fh:
        fh.write("design,PR,normalised_drag,DC60,front_PR_CD,front_PR_DC60,front_PR_CD_DC60\n")
        for i in range(len(n)):
            fh.write(f"{n[i]},{pr[i]:.5f},{cd[i]:.5f},{dc[i]:.4f},{int(f1[i])},{int(f2[i])},{int(f3[i])}\n")
    print("PR-CD      :", n[f1].tolist())
    print("PR-DC60    :", n[f2].tolist())
    print("PR-CD-DC60 :", n[f3].tolist())


if __name__ == "__main__":
    main()