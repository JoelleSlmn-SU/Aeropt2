import numpy as np
import matplotlib.pyplot as plt

def plot_control_nodes_iteration(P, D, ids, title="", removed_ids=set(), new_ids=set(), scale=1.0):
    """
    P: (N,2) or (N,3) positions (plotting uses x,y)
    D: (N,2) or (N,3) displacements (plotting uses dx,dy)
    ids: (N,) integer IDs
    removed_ids/new_ids: sets of IDs to highlight
    """
    P = np.asarray(P); D = np.asarray(D); ids = np.asarray(ids)
    x, y = P[:,0], P[:,1]
    u, v = D[:,0], D[:,1]

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect("equal", adjustable="box")

    # Base styling buckets
    mask_new = np.isin(ids, list(new_ids)) if new_ids else np.zeros(len(ids), dtype=bool)
    mask_removed = np.isin(ids, list(removed_ids)) if removed_ids else np.zeros(len(ids), dtype=bool)
    mask_keep = ~(mask_new | mask_removed)

    ax.scatter(x[mask_keep], y[mask_keep], s=30, label="kept")
    if mask_new.any():
        ax.scatter(x[mask_new], y[mask_new], s=50, marker="*", label="new")
    if mask_removed.any():
        ax.scatter(x[mask_removed], y[mask_removed], s=50, marker="x", label="removed")

    # Displacement vectors (skip removed if you want)
    ax.quiver(x[mask_keep], y[mask_keep], u[mask_keep], v[mask_keep],
              angles="xy", scale_units="xy", scale=1/scale, width=0.003)

    # Labels
    for xi, yi, nid in zip(x, y, ids):
        ax.text(xi, yi, str(int(nid)), fontsize=9, ha="left", va="bottom")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax

def diff_ids(prev_ids, curr_ids):
    prev_ids = set(map(int, prev_ids))
    curr_ids = set(map(int, curr_ids))
    new_ids = curr_ids - prev_ids
    removed_ids = prev_ids - curr_ids
    kept_ids = curr_ids & prev_ids
    return new_ids, removed_ids, kept_ids


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_control_nodes(history, out_path="control_nodes.mp4", fps=20):
    """
    history: list of dicts, each:
      {
        "P": (N,2) or (N,3),
        "D": (N,2) or (N,3),
        "ids": (N,)
      }
    """
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    scat_keep = ax.scatter([], [], s=30)
    scat_new  = ax.scatter([], [], s=50, marker="*")
    scat_rem  = ax.scatter([], [], s=50, marker="x")
    quiv = None
    texts = []

    def clear_texts():
        nonlocal texts
        for t in texts: t.remove()
        texts = []

    def update(k):
        nonlocal quiv
        ax.set_title(f"Iteration {k}")

        P = np.asarray(history[k]["P"])
        D = np.asarray(history[k]["D"])
        ids = np.asarray(history[k]["ids"]).astype(int)

        if k == 0:
            new_ids, removed_ids = set(ids), set()
        else:
            prev_ids = history[k-1]["ids"]
            new_ids, removed_ids, _ = diff_ids(prev_ids, ids)

        mask_new = np.isin(ids, list(new_ids)) if new_ids else np.zeros(len(ids), bool)
        mask_rem = np.isin(ids, list(removed_ids)) if removed_ids else np.zeros(len(ids), bool)
        mask_keep = ~(mask_new | mask_rem)

        x, y = P[:,0], P[:,1]
        u, v = D[:,0], D[:,1]

        scat_keep.set_offsets(np.c_[x[mask_keep], y[mask_keep]])
        scat_new.set_offsets(np.c_[x[mask_new], y[mask_new]] if mask_new.any() else np.empty((0,2)))
        scat_rem.set_offsets(np.c_[x[mask_rem], y[mask_rem]] if mask_rem.any() else np.empty((0,2)))

        if quiv is not None:
            quiv.remove()
        quiv = ax.quiver(x[mask_keep], y[mask_keep], u[mask_keep], v[mask_keep],
                         angles="xy", scale_units="xy", scale=1.0, width=0.003)

        clear_texts()
        for xi, yi, nid in zip(x, y, ids):
            texts.append(ax.text(xi, yi, str(nid), fontsize=9, ha="left", va="bottom"))

        return scat_keep, scat_new, scat_rem, quiv, *texts

    ani = FuncAnimation(fig, update, frames=len(history), interval=1000/fps, blit=False)

    if out_path.lower().endswith(".gif"):
        ani.save(out_path, writer="pillow", fps=fps)
    else:
        ani.save(out_path, writer="ffmpeg", fps=fps)

    plt.close(fig)
    return out_path



def plot_convergence_history(
    X,
    Y,
    training_data,
    count_limit=None,
    normalize_y=True,
    objective="min",          # "min" (drag) or "max" (pressure recovery)
    percent_mode=None,        # None | "reduction" | "increase" (auto if None based on objective)
    save_prefix=None,
    out_dir=".",
    gen_num=None,
    logger=None,
    show=False,
    var = "",
):
    """
    Standalone convergence plot (your same style), now supports:
      - minimisation (e.g., drag): % reduction, best = lowest
      - maximisation (e.g., pressure recovery): % increase, best = highest

    Parameters
    ----------
    X : array-like
        Accepted for signature consistency; not used for x-axis in this plot style.
    Y : array-like
        History values ordered by evaluation time.
    training_data : int
        Number of initial points treated as iteration 0.
    count_limit : int or None
        X-axis limit (iterations). If None, uses len(Y) - training_data.
    normalize_y : bool
        If True, use percent change relative to the first Y (y0).
    objective : {"min","max"}
        Whether "best" means minimum or maximum.
    percent_mode : {None,"reduction","increase"}
        If None: chooses "reduction" for objective="min" and "increase" for objective="max".
    save_prefix, out_dir, gen_num, logger, show : see previous version.

    Returns
    -------
    dict with xs, ys, Y_plot
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    Y = list(Y) if Y is not None else []
    if len(Y) == 0:
        return None

    training_data = int(training_data)
    training_data = max(0, min(training_data, len(Y)))

    if count_limit is None:
        count_limit = max(0, len(Y) - training_data)

    objective = str(objective).lower().strip()
    if objective not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'")

    if percent_mode is None:
        percent_mode = "reduction" if objective == "min" else "increase"
    percent_mode = str(percent_mode).lower().strip()
    if percent_mode not in {"reduction", "increase", "none"}:
        raise ValueError("percent_mode must be None/'none', 'reduction', or 'increase'")

    # ---- Build plotted Y ----
    if normalize_y and percent_mode != "none":
        y0 = float(Y[0])
        if y0 == 0.0:
            Y_plot = np.array(Y, dtype=float)
            ylabel = "Y(x) (y0=0, not normalized)"
            orig_line_val = float(Y_plot[0])
        else:
            # Base percent change
            pct = np.array([100.0 * (float(y) - y0) / y0 for y in Y], dtype=float)

            # For "reduction", invert sign so "positive = improvement" (matches your drag plot vibe)
            if percent_mode == "reduction":
                Y_plot = -pct
                ylabel = f"\% Reduction in {var}"
            else:  # "increase"
                Y_plot = pct
                ylabel = f"\% Increase in {var}"

            orig_line_val = float(Y_plot[0])
    else:
        Y_plot = np.array(Y, dtype=float)
        ylabel = "Y(x)"
        orig_line_val = None

    training_Y = Y_plot[:training_data] if training_data > 0 else np.array([], dtype=float)
    iteration_Y = Y_plot[training_data:]

    # ---- X/Y point construction (same as your original) ----
    xs, ys = [], []
    for y in training_Y:
        xs.append(0)
        ys.append(float(y))
    for i, d in enumerate(iteration_Y):
        xs.append(i + 1)
        ys.append(float(d))

    # ---- helpers for "best" ----
    def best_val(arr):
        if arr.size == 0:
            return None
        return float(np.min(arr) if objective == "min" else np.max(arr))

    best_initial = best_val(training_Y)
    best_overall = best_val(Y_plot)

    # ---- Plot ----
    count_limit = 20
    plt.figure()
    plt.xlabel("Iteration")
    plt.xlim([0, count_limit])
    plt.xticks(np.arange(-1, count_limit + 1, 1.0))
    plt.grid(which="both")
    plt.ylabel(ylabel)

    if orig_line_val is not None:
        plt.axhline(y=orig_line_val, color="red", linestyle="dashed", label="Original")

    plt.scatter(xs, ys, color="black", marker="x")

    y_min = float(np.min(Y_plot))
    y_max = float(np.max(Y_plot))
    plt.ylim([y_min - 1, y_max + 1])

    if best_initial is not None:
        plt.axhline(
            y=best_initial,
            color="orange",
            linestyle="dotted",
            label="Best Initial",
        )
    if best_overall is not None:
        plt.axhline(
            y=best_overall,
            color="green",
            linestyle="solid",
            label="Best Overall",
        )

    msg = f"{'Max' if objective=='max' else 'Min'} y = {best_overall}"
    if logger is not None:
        if callable(logger):
            logger(msg)
        elif hasattr(logger, "log") and callable(getattr(logger, "log")):
            logger.log(msg)

    plt.legend(prop={"size": 14})

    if save_prefix is not None:
        os.makedirs(out_dir, exist_ok=True)
        g = "NA" if gen_num is None else str(gen_num)
        base = f"{save_prefix}_n_{training_data}_g_{g}"
        plt.savefig(os.path.join(out_dir, base + ".png"))
        plt.savefig(os.path.join(out_dir, base + ".pdf"))

    if show:
        plt.show()

    plt.close("all")
    return {"xs": xs, "ys": ys, "Y_plot": Y_plot.tolist()}


Y2 = [0.976, 0.966, 0.9438, 0.905, 0.8843, 0.946, 0.916]
Y1 = [1.67, 1.594, 1.683, 1.677, 1.675, 1.658, 1.629]
# 0 1 2 3 4 8 1 1 
X  = [0, 0, 0, 0, 0, 0, 1, 2]
training_data = 5

out_dir_1 = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\drag_plt.png"
out_dir_2 = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\PR_plt.png"

plot_convergence_history(X, Y1, training_data, objective="min", out_dir=out_dir_1, show=True, var="Drag")
plot_convergence_history(X, Y2, training_data, objective="max", out_dir=out_dir_2, show=True, var="PR")


def animate_design_variable_gif_pretty(
    values,
    var_name="Design Variable",
    xlim=(-1, 1),
    gif_path="design_variable.gif",
    duration_ms=1000,
    dpi=180,
    figsize=(7.2, 2.4),
    trail_alpha=0.25,
    track_height=0.18,
    show_ticks=True,
):
    """
    Aesthetic 1D design-variable GIF:
      - x-axis = variable value
      - each frame adds points cumulatively (with a faded trail)
      - current point highlighted
      - clean, paper-ish styling

    Requirements:
      pip install pillow
    (Uses Pillow via matplotlib's PillowWriter; no imageio needed.)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return

    # ---- Figure / axes ----
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Limits
    ax.set_xlim(xlim)
    ax.set_ylim(-0.6, 0.6)

    # Remove y clutter
    ax.set_yticks([])

    # Spines: clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)

    # Grid: subtle
    ax.grid(True, axis="x", linewidth=0.8, alpha=0.25)
    ax.grid(False, axis="y")

    # Ticks
    if show_ticks:
        ax.tick_params(axis="x", labelsize=12, length=4, width=1)
    else:
        ax.set_xticks([])

    # Labels / title
    ax.set_xlabel(var_name, fontsize=13, labelpad=8)
    title = ax.set_title("Iteration 1", fontsize=16, pad=10)

    # ---- "Track" (a soft band) ----
    # A light horizontal band makes the single-line plot feel intentional.
    y0 = 0.0
    ax.fill_between(
        [xlim[0], xlim[1]],
        y0 - track_height / 2,
        y0 + track_height / 2,
        alpha=0.08,
        linewidth=0,
    )
    ax.hlines(y0, xlim[0], xlim[1], linewidth=1.2, alpha=0.35)

    # ---- Artists: trail + current point ----
    # Trail: all previous points faint
    trail_scatter = ax.scatter([], [], s=20, alpha=trail_alpha, edgecolors="none")
    # Current: emphasized
    current_scatter = ax.scatter([], [], s=20, edgecolors="black", linewidths=0.8, zorder=3)
    # Optional marker line to show current position
    vline = ax.vlines([], -0.35, 0.35, linewidth=1.4, alpha=0.25)

    # A simple color progression (uses matplotlib default colormap)
    # We don't hardcode colors; cmap choice is fine and respects your style.
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    def init():
        trail_scatter.set_offsets(np.empty((0, 2)))
        current_scatter.set_offsets(np.empty((0, 2)))
        vline.set_segments([])
        title.set_text("Modal Coefficient 1 Value")
        return trail_scatter, current_scatter, vline, title

    def update(i):
        # points up to i
        xi = x[: i + 1]
        yi = np.zeros_like(xi)

        # Trail = all previous
        if i > 0:
            trail_offsets = np.column_stack([xi[:-1], yi[:-1]])
            trail_scatter.set_offsets(trail_offsets)
            trail_scatter.set_facecolor(colors[:i])
        else:
            trail_scatter.set_offsets(np.empty((0, 2)))

        # Current point
        current_scatter.set_offsets([[xi[-1], 0.0]])
        current_scatter.set_facecolor([colors[i]])

        # Vertical hint line at current x
        vline.set_segments([[[xi[-1], -0.35], [xi[-1], 0.35]]])

        return trail_scatter, current_scatter, vline, title

    anim = FuncAnimation(fig, update, frames=n, init_func=init, blit=True)

    writer = PillowWriter(
            fps=max(1, int(1000 / duration_ms))
        )
    anim.save(gif_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Saved GIF: {gif_path}")



coeff_history = [0.0, -0.30904, 0.06748, -0.3307, 0.2463, -0.32, 0.6847]
out = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\dv_trace_6.gif"

animate_design_variable_gif_pretty(
    coeff_history[:7],
    var_name="Ramp Angle Coefficient",
    gif_path=out,
    duration_ms=1000
)