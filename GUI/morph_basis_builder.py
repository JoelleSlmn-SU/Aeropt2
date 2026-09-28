"""
morph_basis_builder.py

ONE place that turns the MeshViewer state into the morph_basis.json dict.

Used by:
  * Aeropt.export_morph_basis_for_opt   -> uploaded to the cluster
  * MeshViewer.save_controlnodes         -> <output_dir>/Control Nodes/morph_basis.json
  * MeshViewer.open_modal_explorer       -> passed to the explorer in memory

so the explorer always previews exactly the basis the optimiser will use.

`live_overlay(mv)` reproduces what save_controlnodes() *would* commit from the
on-screen form, without mutating the viewer. That lets the explorer be opened
before "Save Control Nodes / Basis" and still see the same values Save writes.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# What save_controlnodes() commits (kept in step with mesh_gui.save_controlnodes)
# ---------------------------------------------------------------------------
def _widget(fn, default):
    try:
        return fn()
    except Exception:  # widget missing or already deleted by Qt
        return default


def committed_settings(mv) -> dict:
    g = lambda name, default=None: getattr(mv, name, default)  # noqa: E731
    method = g("parameterisation_method", "modal")
    o = {
        "parameterisation_method": method,
        "global_mode_config": g("global_mode_config", []),
        "bump_enable": False,
        "bump_center": None,
        "bump_radius": None,
        "bump_one_sided": None,
        "basis_axes": g("basis_axes", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "control_node_selection_mode": g("control_node_selection_mode", "auto"),
        "loaded_control_nodes_path": g("loaded_control_nodes_path", None),
        "loaded_control_normals_path": g("loaded_control_normals_path", None),
    }

    if method == "direct":
        o["direct_parameterisation_subtype"] = _widget(
            lambda: "xyz" if mv.direct_mode_combo.currentIndex() == 0 else "normal",
            g("direct_parameterisation_subtype", "xyz"),
        )
        o["amp_alpha"] = _widget(lambda: float(mv.direct_amp_alpha_spin.value()), g("amp_alpha", 0.01))
        o["rigid_boundary_translation"] = _widget(
            lambda: bool(mv.direct_rigid_translation_cb.isChecked()), g("rigid_boundary_translation", False)
        )
        o.update(
            k_modes=0, spectral_p=None, coeff_frac=None, seed=0,
            normal_project=None, vector_mode=None, frame_knn=None,
            use_local_modes=False, global_modes_selected=False, global_only=False,
            use_pca=False, pca_train_M=None, pca_energy=None, pca_k_red=None,
            pca_k_final=None, pca_cache_path=None,
        )
    else:
        o["direct_parameterisation_subtype"] = None
        o["k_modes"] = _widget(lambda: int(mv.k_modes_spin.value()), int(g("k_modes", 5)))
        o["amp_alpha"] = _widget(lambda: float(mv.amp_alpha_spin.value()), float(g("amp_alpha", 0.005)))
        o["normal_project"] = _widget(lambda: bool(mv.normal_project_cb.isChecked()), bool(g("normal_project", True)))
        o["rigid_boundary_translation"] = _widget(
            lambda: bool(mv.rigid_translation_cb.isChecked()), bool(g("rigid_boundary_translation", False))
        )
        o["spectral_p"] = float(g("spectral_p", 2.0) or 2.0)
        o["coeff_frac"] = float(g("coeff_frac", 0.15) or 0.15)
        o["seed"] = int(g("seed", 0) or 0)
        o["vector_mode"] = str(g("vector_mode", "local_frame") or "local_frame")
        o["frame_knn"] = int(g("frame_knn", 12) or 12)
        o["use_local_modes"] = bool(g("use_local_modes", True))
        o["global_modes_selected"] = bool(g("global_modes_selected", False))
        o["global_only"] = bool(g("global_only", False))
        if o["global_only"]:
            o["use_local_modes"] = False
        o["use_pca"] = bool(g("use_pca", False))
        o["pca_train_M"] = int(g("pca_train_M", 300) or 300)
        o["pca_energy"] = float(g("pca_energy", 0.99) or 0.99)
        o["pca_k_red"] = g("pca_k_red", None)
        o["pca_k_final"] = g("pca_k_final", None)
        o["pca_cache_path"] = g("pca_cache_path", None)
        o["use_protection"] = bool(g("use_protection", False))
        o["protection_radius"] = g("protection_radius", None)
        o["protected_control_nodes"] = g("protected_control_nodes", [])
    return o


class _Overlay:
    """Attribute view: overrides first, then the real viewer."""

    def __init__(self, mv, overrides: dict):
        object.__setattr__(self, "_mv", mv)
        object.__setattr__(self, "_ov", overrides)

    def __getattr__(self, name):
        ov = object.__getattribute__(self, "_ov")
        if name in ov:
            return ov[name]
        return getattr(object.__getattribute__(self, "_mv"), name)


def live_overlay(mv):
    return _Overlay(mv, committed_settings(mv))


# ---------------------------------------------------------------------------
# The morph_basis.json dict (moved verbatim from Aeropt.export_morph_basis_for_opt)
# ---------------------------------------------------------------------------
def _tolist(x):
    return None if x is None else np.asarray(x).tolist()


def build_morph_basis(mv) -> dict:
    return {
        "control_nodes": np.asarray(mv.control_nodes).tolist(),
        "control_normals": _tolist(getattr(mv, "control_normals", None)),

        "parameterisation_method": getattr(mv, "parameterisation_method", "modal"),
        "direct_parameterisation_subtype": getattr(mv, "direct_parameterisation_subtype", None),

        "selection_mode": getattr(mv, "control_node_selection_mode", None),
        "loaded_control_nodes_path": getattr(mv, "loaded_control_nodes_path", None),
        "loaded_control_normals_path": getattr(mv, "loaded_control_normals_path", None),

        "t_patch_scale": getattr(mv, "t_patch_scale", None),
        "amp_alpha": getattr(mv, "amp_alpha", 0.001),

        "TSurfaces": [int(s) for s in getattr(mv, "TSurfaces", [])],
        "USurfaces": [int(s) for s in getattr(mv, "USurfaces", [])],
        "CSurfaces": [int(s) for s in getattr(mv, "CSurfaces", [])],

        # preliminary response-surface / regional screening metadata
        "prelim_enabled": bool(getattr(mv, "prelim_enabled", False)),
        "prelim_regions": int(getattr(mv, "prelim_regions", 1) or 1),
        "prelim_final_control_nodes": int(getattr(mv, "prelim_final_control_nodes", 0) or 0),
        "prelim_keep_fraction": float(getattr(mv, "prelim_keep_fraction", 0.67) or 0.67),
        "prelim_doe_amplitude": float(getattr(mv, "prelim_doe_amplitude", 1.0) or 1.0),
        "prelim_morris_trajectories": int(getattr(mv, "prelim_morris_trajectories", 6) or 6),
        "prelim_morris_levels": int(getattr(mv, "prelim_morris_levels", 4) or 4),
        "t_surface_points": (
            np.asarray(getattr(mv, "points", []), dtype=float).reshape((-1, 3)).tolist()
            if np.asarray(getattr(mv, "points", [])).size else None
        ),

        "point_region_ids": _tolist(getattr(mv, "point_region_ids", None)),
        "control_node_region_ids": _tolist(getattr(mv, "control_node_region_ids", None)),
        "region_centres": _tolist(getattr(mv, "region_centres", None)),
        "control_node_point_indices": _tolist(getattr(mv, "control_node_point_indices", None)),

        "k_modes": getattr(mv, "k_modes", 0),
        "spectral_p": getattr(mv, "spectral_p", None),
        "coeff_frac": getattr(mv, "coeff_frac", None),
        "seed": getattr(mv, "seed", 0),

        "normal_project": getattr(mv, "normal_project", None),
        "vector_mode": getattr(mv, "vector_mode", None),
        "frame_knn": getattr(mv, "frame_knn", None),

        "use_local_modes": getattr(mv, "use_local_modes", False),
        "global_modes": getattr(mv, "global_modes_selected", False),
        "global_only": getattr(mv, "global_only", False),
        "global_mode_config": getattr(mv, "global_mode_config", []),
        "basis_axes": getattr(mv, "basis_axes", None),

        "use_pca": getattr(mv, "use_pca", False),
        "pca_cache_path": getattr(mv, "pca_cache_path", None),
        "pca_train_M": getattr(mv, "pca_train_M", None),
        "pca_energy": getattr(mv, "pca_energy", None),
        "pca_k_red": getattr(mv, "pca_k_red", None),
        "pca_k_final": getattr(mv, "pca_k_final", None),

        "bump_enable": getattr(mv, "bump_enable", False),
        "bump_center": getattr(mv, "bump_center", None),
        "bump_radius": getattr(mv, "bump_radius", None),
        "bump_one_sided": getattr(mv, "bump_one_sided", False),

        "use_protection": bool(
            getattr(mv, "use_protection", bool(getattr(mv, "protected_control_nodes", [])))
        ),
        "protected_control_nodes": [int(i) for i in (getattr(mv, "protected_control_nodes", []) or [])],
        "protection_radius": (
            float(getattr(mv, "protection_radius", 0.0))
            if getattr(mv, "protection_radius", None) is not None
            else None
        ),

        "graph_method": getattr(mv, "graph_method", "mutual_knn"),
        "delaunay_cutoff_factor": float(getattr(mv, "delaunay_cutoff_factor", 2.5)),

        "rigid_translation": getattr(mv, "rigid_boundary_translation", True),
    }
