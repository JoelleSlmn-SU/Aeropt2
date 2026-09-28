"""
designDisplacement.py

Single source of truth for:  design vector (BO coefficients) + morph_basis.json
                              -> control-node displacement d_ctrl (N, 3)

Used by:
  * Remote.pipeline_cluster.ClusterPipelineManager._write_morph_config
  * GUI.modal_explorer_gui

The coefficient normalisation (pad / truncate) and every getDisplacements
keyword are taken from the basis dict here, so the explorer and the cluster
interpret a design vector identically.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from ShapeParameterization.controlNodeDisp import getDisplacements, build_global_modes

MODAL_CACHE_REL = os.path.join("Control Nodes", "modal_basis_T_surface.npz")


# ---------------------------------------------------------------------------
# Settings read from morph_basis.json (defaults = pipeline_cluster defaults)
# ---------------------------------------------------------------------------
def basis_settings(basis: dict) -> dict:
    s = dict(
        parameterisation_method=str(basis.get("parameterisation_method", "modal")).strip().lower(),
        direct_subtype=basis.get("direct_parameterisation_subtype", None),
        use_pca=bool(basis.get("use_pca", False)),
        pca_cache_path=basis.get("pca_cache_path", None),
        pca_k_final=basis.get("pca_k_final", None),
        normal_project=bool(basis.get("normal_project", True)),
        vector_mode=str(basis.get("vector_mode", "local_frame")),
        frame_knn=basis.get("frame_knn", 12),
        global_modes=bool(basis.get("global_modes", False)),
        global_mode_config=basis.get("global_mode_config", []),
        basis_axes=basis.get("basis_axes", None),
        use_local_modes=bool(basis.get("use_local_modes", True)),
        global_only=bool(basis.get("global_only", False)),
        k=int(basis.get("k_modes", 5)),
        seed=int(basis.get("seed", 0)),
        t_patch_scale=basis.get("t_patch_scale", None),
        amp_alpha=float(basis.get("amp_alpha", 0.005)),
        graph_method=basis.get("graph_method", "mutual_knn"),
        delaunay_cutoff_factor=float(basis.get("delaunay_cutoff_factor", 2.5)),
    )
    if s["global_only"]:
        s["use_local_modes"] = False

    use_protection = bool(basis.get("use_protection", False))
    protected_nodes = [int(i) for i in basis.get("protected_control_nodes", []) or []]
    protection_radius = basis.get("protection_radius", None)
    if protection_radius is not None:
        protection_radius = float(protection_radius)
    if not use_protection:
        protected_nodes = []
        protection_radius = None
    s.update(use_protection=use_protection, protected_nodes=protected_nodes,
             protection_radius=protection_radius)
    return s


def _pipeline_n_global(s: dict) -> int:
    """n_global as pipeline_cluster / pipelineMorph assume it (8 if config empty)."""
    return (
        len(s["global_mode_config"])
        if s["global_modes"] and s["global_mode_config"]
        else (8 if s["global_modes"] else 0)
    )


def _normalise_coeffs(s: dict, coeffs: np.ndarray, n_cn: int) -> np.ndarray:
    """Pad / truncate exactly as pipeline_cluster._write_morph_config did."""
    method = s["parameterisation_method"]

    if method == "direct":
        subtype = str(s["direct_subtype"] or "").strip().lower()
        if subtype == "xyz":
            expected_len = 3 * n_cn
        elif subtype == "normal":
            expected_len = n_cn
        else:
            raise RuntimeError(f"Unknown direct_parameterisation_subtype: {s['direct_subtype']}")

    elif s["use_pca"]:
        if s["pca_k_final"] is None:
            return coeffs  # passed through unchanged (pipeline behaviour)
        expected_len = int(s["pca_k_final"])

    else:
        k = s["k"]
        n_global = _pipeline_n_global(s)
        if s["use_local_modes"]:
            if s["normal_project"]:
                valid_local, default_local = (k,), k
            elif s["vector_mode"] == "xyz":
                valid_local, default_local = (3 * k,), 3 * k
            else:
                valid_local, default_local = (k, 2 * k, 3 * k), 3 * k
        else:
            valid_local, default_local = (0,), 0

        valid_full = tuple(n_global + v for v in valid_local)
        if coeffs.size in valid_local or coeffs.size in valid_full:
            expected_len = int(coeffs.size)
        else:
            expected_len = n_global + default_local

    if coeffs.size < expected_len:
        coeffs = np.pad(coeffs, (0, expected_len - coeffs.size))
    elif coeffs.size > expected_len:
        coeffs = coeffs[:expected_len]
    return coeffs


def control_displacements_from_basis(
    output_dir: str,
    basis: dict,
    coeffs,
    log: Callable[[str], None] = print,
    tag: str = "",
) -> np.ndarray:
    """
    Design vector -> (N_cn, 3) control-node displacement, exactly as the
    cluster pipeline computes it for morph_config_n_*.json.

    output_dir : directory containing "Control Nodes/modal_basis_T_surface.npz"
                 (remote_output on the cluster, the GUI output_dir locally).
    """
    cn = np.asarray(basis["control_nodes"], float).reshape((-1, 3))
    cn_normals = basis.get("control_normals", None)
    cn_normals = None if cn_normals is None else np.asarray(cn_normals, float)

    s = basis_settings(basis)
    coeffs = np.asarray([] if coeffs is None else coeffs, dtype=float).reshape(-1)

    log(
        f"[PIPELINE] {tag} "
        f"param={s['parameterisation_method']} "
        f"use_pca={s['use_pca']} coeffs_len={int(coeffs.size)}"
    )

    coeffs = _normalise_coeffs(s, coeffs, len(cn))
    common = dict(
        control_nodes=cn,
        normals=cn_normals,
        t_patch_scale=s["t_patch_scale"],
        amp_alpha=s["amp_alpha"],
        protected_nodes=s["protected_nodes"],
        radius=s["protection_radius"],
        graph_method=s["graph_method"],
        delaunay_cutoff_factor=s["delaunay_cutoff_factor"],
    )

    if s["parameterisation_method"] == "direct":
        d_ctrl = getDisplacements(
            output_dir,
            coeffs=coeffs,
            parameterisation_method="direct",
            direct_parameterisation_subtype=str(s["direct_subtype"] or "").strip().lower(),
            **common,
        )

    elif s["use_pca"]:
        if not s["pca_cache_path"]:
            raise RuntimeError("use_pca=True but no pca_cache_path provided in morph_basis.json")
        d_ctrl = getDisplacements(
            output_dir,
            use_pca=True,
            pca_cache_path=s["pca_cache_path"],
            pca_coeffs=coeffs,
            normal_project=s["normal_project"],
            vector_mode=s["vector_mode"],
            frame_knn=s["frame_knn"],
            global_modes=s["global_modes"],
            global_mode_config=s["global_mode_config"],
            basis_axes=s["basis_axes"],
            use_local_modes=s["use_local_modes"],
            global_only=s["global_only"],
            **common,
        )

    else:
        d_ctrl = getDisplacements(
            output_dir,
            seed=s["seed"],
            coeffs=coeffs,
            k_modes=s["k"],
            normal_project=s["normal_project"],
            vector_mode=s["vector_mode"],
            frame_knn=s["frame_knn"],
            global_modes=s["global_modes"],
            global_mode_config=s["global_mode_config"],
            basis_axes=s["basis_axes"],
            parameterisation_method="modal",
            use_local_modes=s["use_local_modes"],
            global_only=s["global_only"],
            **common,
        )

    return np.asarray(d_ctrl, dtype=float)


# ---------------------------------------------------------------------------
# Design-vector layout (for labelling sliders / tables)
# ---------------------------------------------------------------------------
def canonical_design_length(basis: dict) -> int:
    """Design-vector length pipelineMorph.orchestrate_run samples for this basis."""
    s = basis_settings(basis)
    n_cn = len(basis.get("control_nodes", []))
    if s["parameterisation_method"] == "direct":
        sub = str(s["direct_subtype"] or "").strip().lower()
        return 3 * n_cn if sub == "xyz" else n_cn
    if s["use_pca"]:
        return int(s["pca_k_final"] or 0)
    k = s["k"]
    local_len = 0
    if s["use_local_modes"]:
        local_len = k if s["normal_project"] else 3 * k
    return _pipeline_n_global(s) + local_len


def load_modal_cache(output_dir: str) -> Optional[dict]:
    p = os.path.join(output_dir, MODAL_CACHE_REL)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def describe_design_vector(basis: dict, output_dir: str, n_design: int) -> List[dict]:
    """
    For each design-vector index i, what getDisplacements will do with it
    after the pipeline pad/truncate. Each entry:
        {"label": str, "kind": "global"|"local"|"direct"|"pca"|"unused",
         "mode": int|None, "component": str|None}
    """
    s = basis_settings(basis)
    cn = np.asarray(basis.get("control_nodes", []), float).reshape((-1, 3))
    n_cn = len(cn)
    out: List[dict] = []

    # what the pipeline keeps
    probe = _normalise_coeffs(s, np.zeros(n_design), n_cn) if n_design else np.zeros(0)
    n_kept = min(n_design, probe.size)

    def unused(i, why="truncated"):
        return {"label": f"x{i + 1}: unused ({why})", "kind": "unused", "mode": None, "component": None}

    if s["parameterisation_method"] == "direct":
        sub = str(s["direct_subtype"] or "").strip().lower()
        for i in range(n_design):
            if i >= n_kept:
                out.append(unused(i)); continue
            if sub == "xyz":
                out.append({"label": f"x{i + 1}: CN{i // 3 + 1} d{'xyz'[i % 3]}",
                            "kind": "direct", "mode": i // 3, "component": "xyz"[i % 3]})
            else:
                out.append({"label": f"x{i + 1}: CN{i + 1} normal",
                            "kind": "direct", "mode": i, "component": "n"})
        return out

    if s["use_pca"]:
        for i in range(n_design):
            out.append({"label": f"x{i + 1}: PCA z{i + 1}", "kind": "pca", "mode": i, "component": None}
                       if i < n_kept else unused(i))
        return out

    # modal: how many globals does getDisplacements actually build?
    gnames: List[str] = []
    if s["global_modes"] and n_cn >= 1:
        try:
            _, gnames = build_global_modes(cn, axes=s["basis_axes"], mode_config=s["global_mode_config"])
        except Exception:
            gnames = []
    n_g = len(gnames)

    cache = load_modal_cache(output_dir) if s["use_local_modes"] else None
    k_c = int(cache["phi_T"].shape[1]) if cache is not None else s["k"]

    if s["use_local_modes"]:
        if s["normal_project"]:
            comps = ["n"]
        elif s["vector_mode"] == "xyz":
            comps = ["x", "y", "z"]
        else:
            # local_frame: length k -> [n], 2k -> [t1, n], 3k -> [t1, t2, n]
            n_loc = max(0, n_kept - n_g)
            comps = {k_c: ["n"], 2 * k_c: ["t1", "n"]}.get(n_loc, ["t1", "t2", "n"])
    else:
        comps = []
    local_len = len(comps) * k_c

    for i in range(n_design):
        if i >= n_kept:
            out.append(unused(i)); continue
        if i < n_g:
            out.append({"label": f"x{i + 1}: global {gnames[i]}", "kind": "global", "mode": i, "component": None})
            continue
        j = i - n_g
        if j < local_len:
            c = comps[j // k_c]
            m = j % k_c
            lab = f"x{i + 1}: mode {m + 1}" + ("" if c == "n" and len(comps) == 1 else f" ({c})")
            out.append({"label": lab, "kind": "local", "mode": m, "component": c})
        else:
            out.append(unused(i, "beyond getDisplacements layout"))
    return out
