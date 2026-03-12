#!/usr/bin/env python3
"""
Mode shape visualisations:
- Laplacian eigenmodes (normal-projected)
- PCA principal deformation modes (from pca_basis.npz)

Outputs PNGs suitable for slides.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname("MeshGeneration"))

from MeshGeneration.modalBasis import build_laplacian_basis, expand_modal_coeffs
from MeshGeneration.controlNodeDisp import estimate_normals
from MeshGeneration.pcaBasis import load_pca_basis, reconstruct_disp_flat  # PCABasis cache

def plot_quiver(control_nodes, disp, out_png, title="", scale=1.0, stride=1):
    """
    Simple 3D quiver plot of displacement vectors.
    stride: plot every nth node to reduce clutter.
    """
    P = np.asarray(control_nodes, float)
    D = np.asarray(disp, float)

    if stride > 1:
        P = P[::stride]
        D = D[::stride]

    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=6)
    ax.quiver(
        P[:, 0], P[:, 1], P[:, 2],
        D[:, 0], D[:, 1], D[:, 2],
        length=scale, normalize=False
    )
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def laplacian_mode_shapes(control_nodes, k_modes=8, n_show=3, knn=6, outdir="figs_modes",
                          normal_project=True, stride=2, vis_scale=1.0):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    evals, phi = build_laplacian_basis(control_nodes, k_modes=k_modes, knn=knn)

    if normal_project:
        normals = estimate_normals(np.asarray(control_nodes, float), knn=12)
        # Plot first n_show eigenmodes as unit coefficients in that mode
        for i in range(min(n_show, phi.shape[1])):
            c = np.zeros(phi.shape[1]); c[i] = 1.0
            d = expand_modal_coeffs(phi, c, normals=normals)
            # scale up only for visualisation
            d_vis = d * float(vis_scale)
            plot_quiver(
                control_nodes, d_vis,
                outdir / f"laplacian_mode_{i+1}.png",
                title=f"Laplacian Mode {i+1} (normal-projected)",
                scale=1.0, stride=stride
            )
    else:
        # per-axis basis would need 3k coefficients; usually not what you want for slides
        raise NotImplementedError("Use normal_project=True for clean mode visualisation.")

def pca_mode_shapes(pca_cache_path, n_show=3, outdir="figs_modes",
                    stride=2, vis_scale=1.0):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    p = load_pca_basis(str(pca_cache_path))  # mean (d,), V (d,k) :contentReference[oaicite:6]{index=6}
    d = int(p.mean.size)
    N = d // 3
    if 3 * N != d:
        raise ValueError(f"PCA mean size {d} not divisible by 3 (expected flattened N×3).")

    # A PCA "mode shape" can be shown by taking a = unit vector in component i,
    # then reconstructing d = mean + V@a; for pure mode, plot V[:,i] only (mean removed).
    for i in range(min(n_show, p.V.shape[1])):
        a = np.zeros(p.V.shape[1]); a[i] = 1.0
        d_flat = reconstruct_disp_flat(np.zeros_like(p.mean), p.V, a)  # = V @ a
        d_ctrl = d_flat.reshape((N, 3))
        d_vis = d_ctrl * float(vis_scale)

        # For PCA mode plots, use a point cloud of control nodes if you have it saved
        # next to cache (common pattern). If not, you'll pass CNs explicitly.
        # Here we assume you also saved control_nodes.npy somewhere accessible.
        raise RuntimeError(
            "For PCA mode plots, call pca_mode_shapes_with_nodes() (needs control_nodes)."
        )

def pca_mode_shapes_with_nodes(control_nodes, pca_cache_path, n_show=3, outdir="figs_modes",
                               stride=2, vis_scale=1.0):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    p = load_pca_basis(str(pca_cache_path))
    d = int(p.mean.size)
    N = d // 3
    if len(control_nodes) != N:
        raise ValueError(f"control_nodes has N={len(control_nodes)} but PCA expects N={N}")

    for i in range(min(n_show, p.V.shape[1])):
        a = np.zeros(p.V.shape[1]); a[i] = 1.0
        d_flat = reconstruct_disp_flat(np.zeros_like(p.mean), p.V, a)  # pure mode = V[:,i]
        d_ctrl = d_flat.reshape((N, 3))
        d_vis = d_ctrl * float(vis_scale)

        plot_quiver(
            control_nodes, d_vis,
            outdir / f"pca_mode_{i+1}.png",
            title=f"PCA Principal Mode {i+1}",
            scale=1.0, stride=stride
        )

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_nodes", required=True, help="Path to control_nodes.npy")
    ap.add_argument("--outdir", default="figs_modes")
    ap.add_argument("--lap_k_modes", type=int, default=8)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--vis_scale", type=float, default=50.0,
                    help="Purely visual multiplier so vectors are visible in plots")
    ap.add_argument("--pca_cache", default="", help="Path to pca_basis.npz (optional)")
    args = ap.parse_args()

    cn = np.load(args.control_nodes)
    
    # Laplacian modes
    laplacian_mode_shapes(
        cn,
        k_modes=args.lap_k_modes,
        n_show=args.show,
        outdir=args.outdir,
        stride=args.stride,
        vis_scale=args.vis_scale
    )

    # PCA modes (if provided)
    if args.pca_cache:
        pca_mode_shapes_with_nodes(
            cn,
            args.pca_cache,
            n_show=args.show,
            outdir=args.outdir,
            stride=args.stride,
            vis_scale=args.vis_scale
        )

    print("Saved mode figures to:", args.outdir)

if __name__ == "__main__":
    main()