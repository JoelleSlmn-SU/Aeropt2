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
    count_limit = 5
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


def plot_design_variable_heatmap(
    X,
    xlower,
    xupper,
    var_names=None,
    row_labels=None,
    gen0_count=None,
    fail_mask=None,
    annotate=True,
    fmt="{:.2f}",
    cmap_diverging="RdBu_r",
    title="Design Variable Coefficients per Case",
    save_prefix=None,
    out_dir=".",
    show=False,
    figsize=None,
):
    """
    Matrix view of every evaluated design vector: rows = cases (in evaluation
    order), columns = design variables (modal coefficients). Cell COLOR encodes
    where the value sits within its own [lower, upper] bound (-1 = at the lower
    bound, 0 = mid-range, +1 = at the upper bound), so variables with different
    physical ranges stay visually comparable on one shared scale; the cell TEXT
    gives the exact raw value, so nothing is lost to the normalization.

    Diverging (not sequential) color is deliberate: 0 is a physically
    meaningful midpoint here (the undeformed/baseline shape for a modal
    coefficient set to 0), so "which side of baseline, and how far" is the
    actual question this plot answers per cell.

    Parameters
    ----------
    X : (N, n_vars) array
        Design vectors, one row per evaluated case, in evaluation order.
    xlower, xupper : scalar or (n_vars,) array
        Per-variable bounds. A scalar is broadcast to all variables.
    var_names : list[str], optional
        Length n_vars column labels (defaults to "v1".."v{n_vars}").
    row_labels : list[str], optional
        Length N row labels (defaults to case index 0..N-1).
    gen0_count : int, optional
        Number of rows belonging to the initial baseline+LHS batch (rows
        0..gen0_count-1). If given, draws a line separating them from the
        subsequent BO iterations -- consistent with the gen0/BO split used in
        plot_convergence_history, so the two figures read together.
    fail_mask : (N,) bool array, optional
        Marks failed/penalized evaluations (see the fail_threshold convention
        used for the convergence plot). The design VECTOR for a failed case is
        still drawn -- it's a real, valid DV combination that happened to
        crash the solver, which is itself useful to see in DV space -- but the
        row gets a red outline and a "FAILED" label so it can't be mistaken
        for a converged, trustworthy result.

    Returns
    -------
    dict with the normalized bound-utilization matrix and the fail mask, for
    downstream inspection/testing.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_cases, n_vars = X.shape

    xlower = np.broadcast_to(np.asarray(xlower, dtype=float), (n_vars,)).copy()
    xupper = np.broadcast_to(np.asarray(xupper, dtype=float), (n_vars,)).copy()
    if np.any(xupper <= xlower):
        raise ValueError("xupper must be strictly greater than xlower for every variable.")

    if var_names is None:
        var_names = [f"v{i+1}" for i in range(n_vars)]
    elif len(var_names) != n_vars:
        raise ValueError(f"var_names has {len(var_names)} entries but X has {n_vars} columns.")

    if row_labels is None:
        row_labels = [str(i) for i in range(n_cases)]
    elif len(row_labels) != n_cases:
        raise ValueError(f"row_labels has {len(row_labels)} entries but X has {n_cases} rows.")

    if fail_mask is None:
        fail_mask = np.zeros(n_cases, dtype=bool)
    else:
        fail_mask = np.asarray(fail_mask, dtype=bool)
        if fail_mask.shape != (n_cases,):
            raise ValueError(f"fail_mask must have shape ({n_cases},), got {fail_mask.shape}.")

    out_of_bounds = (X < xlower) | (X > xupper)
    if out_of_bounds.any():
        bad_cases, bad_vars = np.where(out_of_bounds)
        print(f"[WARN] {len(bad_cases)} (case, variable) value(s) fall outside the "
              f"supplied bounds -- e.g. case {row_labels[bad_cases[0]]}, "
              f"{var_names[bad_vars[0]]} = {X[bad_cases[0], bad_vars[0]]:.4f} not in "
              f"[{xlower[bad_vars[0]]:.4f}, {xupper[bad_vars[0]]:.4f}]. Clipped for the "
              f"color scale only -- the annotated text still shows the true value, so "
              f"check whether this is a bound-definition mismatch or a genuine "
              f"optimizer excursion past the constraint.")

    # Signed bound-utilization: -1 at lower bound, 0 at mid-range, +1 at upper bound.
    mid = 0.5 * (xlower + xupper)
    half_range = 0.5 * (xupper - xlower)
    util = np.clip((X - mid) / half_range, -1.0, 1.0)

    if figsize is None:
        figsize = (max(6.0, 1.1 * n_vars + 2.0), max(4.0, 0.32 * n_cases + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap(cmap_diverging)
    cnorm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    im = ax.imshow(util, cmap=cmap, norm=cnorm, aspect="auto")

    ax.set_xticks(np.arange(n_vars))
    ax.set_xticklabels(var_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_cases))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_ylabel("Case (evaluation order)")
    ax.set_title(title)

    # Cell gridlines (offset minor ticks so they fall between cells, not on them).
    ax.set_xticks(np.arange(-0.5, n_vars, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cases, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(n_cases):
            for j in range(n_vars):
                rgba = cmap(cnorm(util[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                ax.text(j, i, fmt.format(X[i, j]), ha="center", va="center",
                         fontsize=6.5, color=txt_color)

    if gen0_count is not None and 0 < gen0_count < n_cases:
        ax.axhline(gen0_count - 0.5, color="black", linewidth=1.5)
        ax.text(n_vars - 0.5, gen0_count - 0.5, " BO starts ", fontsize=8,
                 va="center", ha="left", color="black",
                 bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85))

    for i in np.flatnonzero(fail_mask):
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), n_vars, 1, fill=False,
                                     edgecolor="crimson", linewidth=2.2, zorder=5))
        ax.text(-0.7, i, "FAILED", fontsize=7, color="crimson", ha="right",
                 va="center", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Position within bound  (-1 = lower, 0 = mid, +1 = upper)")

    plt.tight_layout()

    if save_prefix is not None:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, save_prefix)
        fig.savefig(base + ".png", dpi=300)
        fig.savefig(base + ".pdf")
        print(f"[OK] Saved: {base}.png / .pdf")

    if show:
        plt.show()
    plt.close(fig)

    return {"util": util.tolist(), "fail_mask": fail_mask.tolist()}


def plot_design_variable_case(
    x_case,
    xlower,
    xupper,
    var_names=None,
    case_label="",
    y_value=None,
    y_label="Y",
    is_failed=False,
    baseline_case=None,
    cmap_diverging="RdBu_r",
    ax=None,
    figsize=(7, 4.5),
):
    """
    Single-case lollipop view: one design vector's modal coefficients plotted
    against their bounds. Each variable gets its own column with a shaded band
    for [lower, upper], a dashed line at 0 (the undeformed/baseline
    coefficient), and a stem+marker at this case's value, colored the same way
    (and on the same -1..+1 bound-utilization scale) as
    plot_design_variable_heatmap so a reader can move between the two figures
    without re-learning the color mapping.

    x_case  : (n_vars,) design vector for ONE case
    xlower, xupper : scalar or (n_vars,) bounds, same convention as the heatmap
    baseline_case  : optional (n_vars,) reference vector (e.g. case 0) drawn as
                     small grey ticks, so you can see how far this case moved
                     from the baseline design at a glance
    ax      : draw into an existing Axes instead of creating a figure (used by
              the batch export below); if None, a new figure is created

    Returns (fig, ax) -- fig is None if `ax` was supplied by the caller.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    x_case = np.asarray(x_case, dtype=float).flatten()
    n_vars = x_case.shape[0]

    xlower = np.broadcast_to(np.asarray(xlower, dtype=float), (n_vars,)).copy()
    xupper = np.broadcast_to(np.asarray(xupper, dtype=float), (n_vars,)).copy()
    if np.any(xupper <= xlower):
        raise ValueError("xupper must be strictly greater than xlower for every variable.")

    if var_names is None:
        var_names = [f"v{i+1}" for i in range(n_vars)]
    elif len(var_names) != n_vars:
        raise ValueError(f"var_names has {len(var_names)} entries but x_case has {n_vars}.")

    mid = 0.5 * (xlower + xupper)
    half_range = 0.5 * (xupper - xlower)
    util = np.clip((x_case - mid) / half_range, -1.0, 1.0)
    out_of_bounds = (x_case < xlower) | (x_case > xupper)

    cmap = plt.get_cmap(cmap_diverging)
    cnorm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    xpos = np.arange(n_vars)

    for i in xpos:
        ax.add_patch(plt.Rectangle((i - 0.35, xlower[i]), 0.7, xupper[i] - xlower[i],
                                     facecolor="0.92", edgecolor="0.75", zorder=0))

    ax.axhline(0.0, color="0.5", linestyle="dashed", linewidth=1.0, zorder=1)

    if baseline_case is not None:
        baseline_case = np.asarray(baseline_case, dtype=float).flatten()
        ax.scatter(xpos, baseline_case, marker="_", s=260, color="0.45",
                    linewidths=1.6, zorder=2, label="Baseline (case 0)")

    colors = [cmap(cnorm(u)) for u in util]
    ax.vlines(xpos, 0.0, x_case, color=colors, linewidth=2.2, zorder=3)
    ax.scatter(xpos, x_case, s=90, color=colors, edgecolors="0.2", linewidths=0.8, zorder=4)

    for i in xpos:
        va = "bottom" if x_case[i] >= 0 else "top"
        offset = 0.03 * (xupper[i] - xlower[i])
        ax.text(i, x_case[i] + (offset if x_case[i] >= 0 else -offset),
                 f"{x_case[i]:.3f}", ha="center", va=va, fontsize=8)
        if out_of_bounds[i]:
            ax.text(i, xupper[i] + 0.08 * (xupper[i] - xlower[i]), "OUT OF BOUNDS",
                     ha="center", va="bottom", fontsize=7, color="crimson", fontweight="bold")

    ax.set_xticks(xpos)
    ax.set_xticklabels(var_names, rotation=20, ha="right", fontsize=9)
    lo_all, hi_all = float(np.min(xlower)), float(np.max(xupper))
    pad = 0.15 * (hi_all - lo_all)
    ax.set_ylim(lo_all - pad, hi_all + pad)
    ax.set_xlim(-0.6, n_vars - 0.4)
    ax.set_ylabel("Coefficient value")
    ax.grid(axis="y", alpha=0.25)

    title = f"Case {case_label}" if case_label != "" else "Case"
    if y_value is not None:
        title += f"  |  {y_label} = {y_value:.5f}"
    if is_failed:
        title += "   [FAILED]"
        for spine in ax.spines.values():
            spine.set_edgecolor("crimson")
            spine.set_linewidth(2.0)
    ax.set_title(title, fontsize=11)

    if baseline_case is not None:
        ax.legend(loc="upper right", fontsize=8)

    if fig is not None:
        plt.tight_layout()

    return fig, ax


def plot_design_variables_case_by_case(
    X,
    xlower,
    xupper,
    var_names=None,
    row_labels=None,
    Y=None,
    y_label="Y",
    fail_mask=None,
    baseline_idx=0,
    out_dir=".",
    save_prefix="dv_case",
    combine_pdf=True,
    dpi=200,
):
    """
    Calls plot_design_variable_case() once per row of X, saves one PNG per
    case (dv_case_000.png, dv_case_001.png, ...), and -- if PIL is available
    -- stitches them into a single multi-page PDF you can page through, the
    same pattern pngs_to_pdf() uses for the x-sweep contours in Paraview.py.

    baseline_idx: row of X drawn as the grey reference ticks on every case's
                  plot (default 0, i.e. the undeformed baseline design -- set
                  to None to disable the overlay).
    """
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_cases, n_vars = X.shape

    if row_labels is None:
        row_labels = [str(i) for i in range(n_cases)]
    elif len(row_labels) != n_cases:
        raise ValueError(f"row_labels has {len(row_labels)} entries but X has {n_cases} rows.")

    if fail_mask is None:
        fail_mask = np.zeros(n_cases, dtype=bool)
    else:
        fail_mask = np.asarray(fail_mask, dtype=bool)
        if fail_mask.shape != (n_cases,):
            raise ValueError(f"fail_mask must have shape ({n_cases},), got {fail_mask.shape}.")

    baseline_case = X[baseline_idx] if baseline_idx is not None else None

    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []
    for i in range(n_cases):
        y_val = float(Y[i]) if Y is not None else None
        fig, ax = plot_design_variable_case(
            x_case=X[i],
            xlower=xlower,
            xupper=xupper,
            var_names=var_names,
            case_label=row_labels[i],
            y_value=y_val,
            y_label=y_label,
            is_failed=bool(fail_mask[i]),
            baseline_case=baseline_case,
        )
        path = os.path.join(out_dir, f"{save_prefix}_{i:03d}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved_paths.append(path)
    print(f"[OK] Saved {len(saved_paths)} per-case plots -> {out_dir} "
          f"({save_prefix}_000.png .. {save_prefix}_{n_cases-1:03d}.png)")

    if combine_pdf:
        try:
            from PIL import Image
        except Exception:
            print("[WARN] PIL not available; skipping PDF build (individual PNGs are still saved).")
            return saved_paths
        frames = sorted(glob.glob(os.path.join(out_dir, f"{save_prefix}_*.png")))
        frames = [p for p in frames if os.path.getsize(p) > 0]
        if not frames:
            print("[WARN] No per-case PNGs found to combine into a PDF.")
            return saved_paths
        imgs = [Image.open(p).convert("RGB") for p in frames]
        pdf_path = os.path.join(out_dir, f"{save_prefix}.pdf")
        imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:])
        print(f"[OK] PDF written: {pdf_path} ({len(imgs)} pages)")

    return saved_paths


mcsv_file = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\corner_optimisation_cd\bo_data.mcsv"
out_dir = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\corner_optimisation_cd\dv_case_plots"

training_data = 27          # number of initial LHS samples
objective = "MIN"          # "max" for pressure recovery♀
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

# ------------------------------------------------------------------
# Plot design variable (modal coefficient) coverage per case
# ------------------------------------------------------------------
# NOTE: bounds below default to (-1, 1) for every coefficient, matching the
# range X actually spans in bo_data.mcsv and the xlim used elsewhere in this
# file (animate_design_variable_gif_pretty). If Aeropt2's modal
# parameterization defines different (e.g. per-mode or asymmetric) bounds,
# replace these with the real arrays -- xlower/xupper each take either a
# scalar or a length-n_vars array.
n_vars = X.shape[1]
dv_xlower = -1.0
dv_xupper = 1.0
dv_names = [f"Modal Coeff {i+1}" for i in range(n_vars)]

# Same "solver crashed" sentinel convention flagged for the convergence plot:
# CD/PR values are O(1), so anything at/above this magnitude is a penalty
# value, not a real evaluation.
fail_threshold = 1e6
fail_mask = np.abs(Y) >= fail_threshold

plot_design_variable_heatmap(
    X=X,
    xlower=dv_xlower,
    xupper=dv_xupper,
    var_names=dv_names,
    gen0_count=training_data + 1,
    fail_mask=fail_mask,
    save_prefix="dv_heatmap",
    out_dir=out_dir,
    show=True,
)

# ------------------------------------------------------------------
# Same design variables, one plot per case (paged PDF)
# ------------------------------------------------------------------
dv_case_dir = os.path.join(out_dir if os.path.isdir(out_dir) else ".", "dv_case_plots")

plot_design_variables_case_by_case(
    X=X,
    xlower=dv_xlower,
    xupper=dv_xupper,
    var_names=dv_names,
    Y=Y,
    y_label=variable_name,
    fail_mask=fail_mask,
    baseline_idx=0,
    out_dir=dv_case_dir,
    save_prefix="dv_case",
    combine_pdf=True,
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