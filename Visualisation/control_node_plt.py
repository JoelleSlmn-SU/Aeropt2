import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.dirname("FileRW"))
from FileRW.MultiArrayCsvFile import MultiArrayCsvFile

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
    objective="min",
    percent_mode=None,
    save_prefix=None,
    out_dir=".",
    gen_num=None,
    logger=None,
    show=False,
    var="",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    Y_raw = np.asarray(Y, dtype=float).flatten()
    if len(Y_raw) == 0:
        return None

    training_data = int(training_data)
    gen0_count = training_data + 1  # baseline + initial LHS samples

    objective = str(objective).lower().strip()
    if objective not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'")

    if percent_mode is None:
        percent_mode = "reduction" if objective == "min" else "increase"

    # ------------------------------------------------------------
    # Build x-axis:
    # baseline + LHS -> generation 0
    # BO samples     -> generation 1, 2, 3, ...
    # ------------------------------------------------------------
    n_total = len(Y_raw)
    n_bo = max(0, n_total - gen0_count)

    xs = [0] * min(gen0_count, n_total)
    xs += list(range(1, n_bo + 1))
    xs = np.asarray(xs, dtype=int)

    # ------------------------------------------------------------
    # Convert Y to plotted values
    # ------------------------------------------------------------
    y0 = float(Y_raw[0])

    if normalize_y and y0 != 0.0:
        pct = 100.0 * (Y_raw - y0) / y0

        if percent_mode == "reduction":
            Y_plot = -pct
            ylabel = f"% Reduction in {var}"
        elif percent_mode == "increase":
            Y_plot = pct
            ylabel = f"% Increase in {var}"
        else:
            Y_plot = Y_raw.copy()
            ylabel = var if var else "Y"
    else:
        Y_plot = Y_raw.copy()
        ylabel = var if var else "Y"

    # ------------------------------------------------------------
    # Best values computed from RAW objective values
    # ------------------------------------------------------------
    gen0_raw = Y_raw[:gen0_count]

    if objective == "min":
        best_initial_idx = int(np.argmin(gen0_raw))
        best_overall_idx = int(np.argmin(Y_raw))
    else:
        best_initial_idx = int(np.argmax(gen0_raw))
        best_overall_idx = int(np.argmax(Y_raw))

    best_initial_line = float(Y_plot[best_initial_idx])
    best_overall_line = float(Y_plot[best_overall_idx])

    print(f"Generation mapping:")
    for xval, yval in zip(xs, Y_raw):
        print(f"{yval:.5f} -> {xval}")

    print(f"Best initial raw Y = {Y_raw[best_initial_idx]:.5f} at generation {xs[best_initial_idx]}")
    print(f"Best overall raw Y = {Y_raw[best_overall_idx]:.5f} at generation {xs[best_overall_idx]}")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    count_limit = 20
    if count_limit is None:
        count_limit = max(xs)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(xs, Y_plot, color="black", marker="x")

    ax.axhline(
        y=0.0,
        color="red",
        linestyle="dashed",
        label="Original",
    )

    ax.axhline(
        y=best_initial_line,
        color="orange",
        linestyle="dotted",
        label="Best Initial",
    )

    ax.axhline(
        y=best_overall_line,
        color="green",
        linestyle="solid",
        label="Best Overall",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)

    ax.set_xlim(-0.5, count_limit + 0.5)
    ax.set_xticks(np.arange(0, count_limit + 1, 1))

    y_min = float(np.min(Y_plot))
    y_max = float(np.max(Y_plot))
    pad = max(1.0, 0.1 * abs(y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.grid(True, which="both")
    ax.legend(prop={"size": 14})

    plt.tight_layout()

    if save_prefix is not None:
        os.makedirs(out_dir, exist_ok=True)
        g = "NA" if gen_num is None else str(gen_num)
        base = f"{save_prefix}_n_{training_data}_g_{g}"
        plt.savefig(os.path.join(out_dir, base + ".png"), dpi=300)
        plt.savefig(os.path.join(out_dir, base + ".pdf"))

    if show:
        plt.show()

    plt.close(fig)

    return {
        "xs": xs.tolist(),
        "Y_raw": Y_raw.tolist(),
        "Y_plot": Y_plot.tolist(),
        "best_initial_idx": best_initial_idx,
        "best_overall_idx": best_overall_idx,
    }

mcsv_file = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\bo_data.mcsv"
out_dir = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\sphere_opt\conv_plot.png"

training_data = 5          # number of initial LHS samples
objective = "min"          # "max" for pressure recovery♀
variable_name = "CD"

# ------------------------------------------------------------------
# Load BO data
# ------------------------------------------------------------------

mac = MultiArrayCsvFile(mcsv_file)
data = mac.read()

X = np.asarray(data["X"])
Y = np.asarray(data["Y"]).flatten()
print(X)
print(Y)

print(f"Loaded {len(Y)} evaluations")

# ------------------------------------------------------------------
# Plot convergence history
# ------------------------------------------------------------------

plot_convergence_history(
    X=X,
    Y=Y,
    training_data=training_data,
    objective=objective,
    out_dir = out_dir,
    show=True,
    var=variable_name,
)


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



#coeff_history = [0.0, -0.30904, 0.06748, -0.3307, 0.2463, -0.32, 0.6847]
#out = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\dv_trace_6.gif"

#animate_design_variable_gif_pretty(
#    coeff_history[:7],
#    var_name="Ramp Angle Coefficient",
#    gif_path=out,
#    duration_ms=1000
#)