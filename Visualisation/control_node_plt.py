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
