# visualize_tuc_vtm.py

import os, sys
import json
import argparse
import pyvista as pv

sys.path.append(os.path.dirname("MeshGeneration"))
from MeshGeneration.meshFile import load_mesh


def _find_vtm_from_config(cfg_path: str, cfg: dict) -> str:
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    base = os.path.splitext(os.path.basename(cfg.get("vtk_name", "")))[0]

    candidates = []

    if base:
        candidates.append(os.path.join(cfg_dir, f"{base}.vtm"))

        out_dir = cfg.get("output_directory", "")
        gen = cfg.get("gen", 0)
        candidates.append(os.path.join(out_dir, "surfaces", f"n_{gen}", f"{base}.vtm"))
        candidates.append(os.path.join(out_dir, f"{base}.vtm"))

    try:
        for fn in os.listdir(cfg_dir):
            if fn.lower().endswith(".vtm"):
                candidates.append(os.path.join(cfg_dir, fn))
    except Exception:
        pass

    for p in candidates:
        if p and os.path.exists(p):
            return p

    raise FileNotFoundError("Could not locate .vtm file.")


def visualize_tuc_from_config(
    config_json: str,
    vtm_path: str | None = None,
    show_edges: bool = False,
    opacity: float = 1.0,
    exclude_surfaces: list[int] | None = None,
    line_width: float = 1.5,
):
    with open(config_json, "r") as f:
        cfg = json.load(f)

    T = set(map(int, cfg.get("t_surfaces", [])))
    U = set(map(int, cfg.get("u_surfaces", [])))
    C = set(map(int, cfg.get("c_surfaces", [])))

    # If exclude not provided from CLI, try config
    if exclude_surfaces is None:
        exclude_surfaces = cfg.get("exclude_surfaces", [])

    exclude_surfaces = set(map(int, exclude_surfaces))

    if vtm_path is None:
        vtm_path = _find_vtm_from_config(config_json, cfg)

    mesh = load_mesh(vtm_path)

    p = pv.Plotter()
    p.add_text(f"T/U/C preview\n{os.path.basename(vtm_path)}", font_size=10)

    legend = [
        ("T surfaces", "blue"),
        ("U surfaces", "green"),
        ("C surfaces", "red"),
        ("Other", "lightgray"),
    ]

    nT = nU = nC = nO = nExcluded = 0

    for name, blk in getattr(mesh, "blocks", []):

        sid = int(mesh.get_surface_id(name))

        # --- EXCLUSION LOGIC ---
        if sid in exclude_surfaces:
            nExcluded += 1
            continue  # do NOT plot

        if sid in T:
            color = "#3246a8"; nT += 1
        elif sid in U:
            color = "#2fad35"; nU += 1
        elif sid in C:
            color = "#a83238"; nC += 1
        else:
            color = "lightgray"; nO += 1

        # Edges only (no filled surface): each patch's cell edges, colour-coded by
        # T/U/C/Other. `show_edges` is kept as a "bold vs. thin" toggle now that
        # edges are the whole plot, rather than "overlay edges on the surface".
        p.add_mesh(
            blk,
            style="wireframe",
            color=color,
            opacity=float(opacity),
            line_width=float(line_width) * (1.6 if bool(show_edges) else 1.0),
            pickable=True,
        )

    # p.add_legend(legend, bcolor="white")
    # p.add_text(
    #     f"T={nT}  U={nU}  C={nC}  Other={nO}  Excluded={nExcluded}",
    #     position="lower_left",
    #     font_size=9
    # )

    p.show()


def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--config", required=True)
    # ap.add_argument("--vtm", default=None)
    # ap.add_argument("--edges", action="store_true")
    # ap.add_argument("--opacity", type=float, default=1.0)
    # ap.add_argument(
    #     "--exclude",
    #     nargs="*",
    #     type=int,
    #     default=None,
    #     help="Surface IDs to exclude from plotting"
    # )

    # args = ap.parse_args()
    
    config = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\crm_prelim\surfaces\n_0\morph_config_n_1.json"
    vtm = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\crm_prelim\crm.vtm"
    edges= "store_true"
    opacity = 1
    exclude = [15, 16]

    visualize_tuc_from_config(
        config_json=config,
        vtm_path=vtm,
        show_edges=edges,
        opacity=opacity,
        exclude_surfaces=exclude,
    )


if __name__ == "__main__":
    main()