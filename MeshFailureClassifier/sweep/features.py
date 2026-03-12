from __future__ import annotations

import argparse
import json
import os, sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

# Your existing feature core (good stuff) lives here:
# - area/edge ratios, flipped tris, laplacian energy, etc.

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# .../Scripts/MeshFailureClassifier/sweep
# go up 2 levels -> .../Scripts
SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

FILERW_DIR = os.path.join(SCRIPTS_DIR, "FileRW")
if FILERW_DIR not in sys.path:
    sys.path.insert(0, FILERW_DIR)
    
from classifier_inputs import compute_mesh_fail_features


# ----------------------------
# Helpers
# ----------------------------

def _bbox_diag(X: np.ndarray) -> float:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return float(np.linalg.norm(maxs - mins))

def _triangle_quality(P: np.ndarray, tri_nodes: np.ndarray) -> np.ndarray:
    """
    Simple triangle quality metric in (0,1]:
        q = 4*sqrt(3)*A / sum(e^2)
    equilateral -> 1, degenerate -> 0
    """
    a = P[tri_nodes[:, 0]]
    b = P[tri_nodes[:, 1]]
    c = P[tri_nodes[:, 2]]
    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)
    sume2 = e0**2 + e1**2 + e2**2

    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    q = (4.0 * np.sqrt(3.0) * A) / np.maximum(sume2, 1e-15)
    return q

def _tri_strain_proxy(X0: np.ndarray, U: np.ndarray, tri_nodes: np.ndarray) -> np.ndarray:
    """
    Edge-based displacement-gradient proxy (dimensionless-ish):
      for each triangle, compute max over edges of ||Δu|| / ||Δx||
    This catches "sharp" deformation that often correlates with bad volume meshing.
    """
    a = tri_nodes[:, 0]
    b = tri_nodes[:, 1]
    c = tri_nodes[:, 2]

    def edge_ratio(i, j):
        dx = np.linalg.norm(X0[j] - X0[i], axis=1)
        du = np.linalg.norm(U[j] - U[i], axis=1)
        return du / np.maximum(dx, 1e-15)

    r0 = edge_ratio(a, b)
    r1 = edge_ratio(b, c)
    r2 = edge_ratio(c, a)
    return np.maximum(np.maximum(r0, r1), r2)

def _load_morph_basis(basis_json: Optional[str]) -> Dict[str, Any]:
    if not basis_json:
        return {}
    basis_json = basis_json.strip()
    if not basis_json:
        return {}
    with open(basis_json, "r") as f:
        return json.load(f)

def _get_surface_node_gids(ff: Any, sid: int) -> np.ndarray:
    """
    FroFile commonly returns (global_ids, local_ids) from get_surface_nodes.
    We'll attempt to interpret both.
    """
    out = ff.get_surface_nodes(int(sid))
    if out is None:
        return np.empty((0,), dtype=np.int64)

    if isinstance(out, tuple) and len(out) >= 1:
        g = out[0]
        # sometimes g might be list of ints already
        return np.asarray(g, dtype=np.int64)

    # fallback
    return np.asarray(out, dtype=np.int64)

def _select_deforming_gids(ff0: Any, basis: Dict[str, Any]) -> np.ndarray:
    """
    Prefer deforming region = nodes on TSurfaces ∪ USurfaces (from morph_basis.json).
    Fallback to all nodes if missing.
    """
    t_surf = list(map(int, basis.get("TSurfaces", []) or []))
    u_surf = list(map(int, basis.get("USurfaces", []) or []))

    gids = []
    for sid in (t_surf + u_surf):
        try:
            gids.append(_get_surface_node_gids(ff0, sid))
        except Exception:
            continue

    if gids:
        D = np.unique(np.concatenate(gids))
        return D.astype(np.int64)

    # fallback: everything
    try:
        n = len(ff0.nodes)
    except Exception:
        n = 0
    return np.arange(n, dtype=np.int64)

def _select_anchor_gids(ff0: Any, D: np.ndarray, basis: Dict[str, Any]) -> np.ndarray:
    """
    Anchors = nodes in deforming region that touch CSurfaces (via node_connections).
    This approximates your previous anchor logic without needing MorphModel.
    """
    c_surf = list(map(int, basis.get("CSurfaces", []) or []))
    if not c_surf:
        return np.empty((0,), dtype=np.int64)

    Cg = []
    for sid in c_surf:
        try:
            Cg.append(_get_surface_node_gids(ff0, sid))
        except Exception:
            continue
    if not Cg:
        return np.empty((0,), dtype=np.int64)
    C = set(np.unique(np.concatenate(Cg)).astype(np.int64).tolist())

    anchors = []
    conn = getattr(ff0, "node_connections", {}) or {}
    for g in D:
        nb = conn.get(int(g), [])
        if any(int(n) in C for n in nb):
            anchors.append(int(g))
    if not anchors:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(sorted(set(anchors)), dtype=np.int64)

def _filter_boundary_tris_by_surfaces(ff0: Any, basis: Dict[str, Any]) -> np.ndarray:
    """
    Return boundary triangles filtered to T/U surfaces if surf ids exist.
    If boundary triangles don't carry surface id, return all.
    """
    tri = np.asarray(getattr(ff0, "boundary_triangles", []), dtype=np.int64)
    if tri.size == 0:
        return tri

    if tri.ndim != 2 or tri.shape[1] < 3:
        return np.empty((0, 3), dtype=np.int64)

    # If surf id present, filter to T/U
    if tri.shape[1] >= 4:
        t_surf = set(map(int, basis.get("TSurfaces", []) or []))
        u_surf = set(map(int, basis.get("USurfaces", []) or []))
        keep = t_surf | u_surf
        if keep:
            mask = np.isin(tri[:, 3], np.asarray(sorted(keep), dtype=np.int64))
            tri = tri[mask]

    # return node triplets
    return tri[:, :3].astype(np.int64)


# ----------------------------
# Main feature computation
# ----------------------------

def compute_features(
    orig_fro: str,
    morphed_fro: str,
    run_dir: Optional[str] = None,
    case_id: Optional[str] = None,
    morph_basis_json: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns a JSON-serializable dict:
      - metadata (paths, ids)
      - morph features (scalars)
    """
    # Import here to avoid hard dependency when not available locally
    from FileRW.FroFile import FroFile  # type: ignore

    ff0 = FroFile()
    ff0.read_file(orig_fro)

    ff1 = FroFile()
    ff1.read_file(morphed_fro)

    basis = _load_morph_basis(morph_basis_json)

    # --- core features from your existing implementation ---
    # We pass no morph_model here (we handle region selection ourselves),
    # but compute_mesh_fail_features still does useful global signals.
    core = compute_mesh_fail_features(ff0, ff1, morph_model=None)

    # --- region-aware extras ---
    X0 = np.asarray(ff0.nodes, dtype=float)
    X1 = np.asarray(ff1.nodes, dtype=float)
    U = X1 - X0

    D = _select_deforming_gids(ff0, basis)
    anchors = _select_anchor_gids(ff0, D, basis)

    XD0 = X0[D] if D.size else X0
    Ld = max(_bbox_diag(XD0), 1e-12)

    umagD = np.linalg.norm(U[D], axis=1) if D.size else np.linalg.norm(U, axis=1)

    feats: Dict[str, Any] = {}
    feats.update(core)

    # Make it explicit what region we used
    feats["D_count"] = int(D.size)
    feats["anchor_count"] = int(anchors.size)

    # Tri-based quality + strain proxy (filtered to T/U surfaces when possible)
    tri_nodes = _filter_boundary_tris_by_surfaces(ff0, basis)
    if tri_nodes.size:
        q0 = _triangle_quality(X0, tri_nodes)
        q1 = _triangle_quality(X1, tri_nodes)

        # displacement gradient / "strain" proxy
        strain = _tri_strain_proxy(X0, U, tri_nodes)

        feats.update({
            "tri_q0_p01": float(np.quantile(q0, 0.01)),
            "tri_q0_p50": float(np.quantile(q0, 0.50)),
            "tri_q0_min": float(np.min(q0)),
            "tri_q1_p01": float(np.quantile(q1, 0.01)),
            "tri_q1_p50": float(np.quantile(q1, 0.50)),
            "tri_q1_min": float(np.min(q1)),
            "tri_q_ratio_p01": float(np.quantile(q1 / np.maximum(q0, 1e-15), 0.01)),
            "tri_q_ratio_min": float(np.min(q1 / np.maximum(q0, 1e-15))),
            "tri_deg_frac_q1_lt_1e-6": float(np.mean(q1 < 1e-6)),
            "strain_p50": float(np.quantile(strain, 0.50)),
            "strain_p95": float(np.quantile(strain, 0.95)),
            "strain_max": float(np.max(strain)),
        })
    else:
        feats.update({
            "tri_q0_p01": 1.0, "tri_q0_p50": 1.0, "tri_q0_min": 1.0,
            "tri_q1_p01": 1.0, "tri_q1_p50": 1.0, "tri_q1_min": 1.0,
            "tri_q_ratio_p01": 1.0, "tri_q_ratio_min": 1.0,
            "tri_deg_frac_q1_lt_1e-6": 0.0,
            "strain_p50": 0.0, "strain_p95": 0.0, "strain_max": 0.0,
        })

    # Compose final record
    import datetime
    x = datetime.datetime.now()
    case_id = x.strftime("%d%b%I") + case_id
    
    rec: Dict[str, Any] = {
        "case_id": case_id or "",
        "morphed_fro": morphed_fro,
        "features": feats,
    }

    if extra:
        rec.update(extra)

    return rec


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ----------------------------
# CLI
# ----------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="Path to original .fro")
    ap.add_argument("--morphed", required=True, help="Path to morphed .fro")
    ap.add_argument("--basis", default="", help="Path to morph_basis.json (optional)")
    ap.add_argument("--run-dir", default="", help="Run directory (optional)")
    ap.add_argument("--case-id", default="", help="Case identifier (optional)")
    ap.add_argument("--out-jsonl", default="", help="Append record to this dataset.jsonl (optional)")
    args = ap.parse_args()

    rec = compute_features(
        orig_fro=args.orig,
        morphed_fro=args.morphed,
        run_dir=args.run_dir or None,
        case_id=args.case_id or None,
        morph_basis_json=args.basis or None,
    )

    if args.out_jsonl:
        append_jsonl(args.out_jsonl, rec)
    else:
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    _cli()
