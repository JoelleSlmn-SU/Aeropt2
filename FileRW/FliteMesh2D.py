"""
FliteMesh2D.py

Mesh file of the FLITE 2D preprocessor (PrePro.exe, "Aggl2d") and solver
(Solver.exe, "MGNS2D"): writer, reader and checks.

This is NOT the FLITE 2D *geometry* .dat that FileRW.EdgeMesh2D.read_dat_2d
reads (boundary curves for the mesh generator). Both use the extension .dat.
`is_flite2d_mesh` tells them apart, and the solver mesh is never written next
to the input mesh as <stem>.dat (MeshGeneration.meshFile.find_dat_for would
pick it up as a geometry file).

File layout (as written by the original MATLAB save_mesh.m, list-directed
Fortran read):

    1
    title
    ne  np  nb
    <ne> <np> <nb>
    connectivities
    <elem id> <n1> <n2> <n3> 1            (ne rows, 1-based node ids)
    coordinates
    <node id> <x> <y>                     (np rows)
    boundaries
    <n1> <n2> <flag>                      (nb rows, one per boundary edge)

Boundary flags known from the solver tutorial:
    1  inviscid wall
    3  far field
    5  viscous wall
The solver also has engine-face / inflow / outflow conditions; their flag
numbers are not documented here, so any other integer is passed through
unchanged and it is the user's job to know what it means.

What this writer guarantees
---------------------------
* Coordinates in full double precision. The MATLAB writer used dlmwrite's
  default 5 significant digits, which merged neighbouring nodes on a fine
  mesh (zero-length edge, zero-area triangle -> NaN from the first solver
  iteration).
* Triangles counter-clockwise, every boundary edge listed exactly once.
* Boundary nodes numbered first, walls before everything else, each loop in
  walking order - the numbering the original mesh generator produced, so
  solver inputs that refer to wall node numbers (ivd%tripNodes) keep their
  meaning. `node_order` in the returned info maps the file numbering back to
  the EdgeMesh2D numbering; the solver's .res rows are in file numbering.
* Refuses a mesh with a degenerate triangle, a zero-length edge or a
  boundary edge without a flag, instead of writing a file the solver would
  turn into NaN.

Not verified here (no Windows executables in the test environment): that
PrePro.exe needs none of the above beyond what the original files had, and
whether it cares about the direction of the boundary edges. In the original
example the two wall loops were wound in opposite directions, so it probably
does not; this writer uses fluid-on-the-left throughout.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

FLAG_INVISCID_WALL = 1
FLAG_FAR_FIELD = 3
FLAG_VISCOUS_WALL = 5
FLAG_LABELS = {FLAG_INVISCID_WALL: "Inviscid wall", FLAG_FAR_FIELD: "Far field",
               FLAG_VISCOUS_WALL: "Viscous wall"}
WALL_FLAGS = (FLAG_INVISCID_WALL, FLAG_VISCOUS_WALL)

# bc codes of the FLITE 2D GEOMETRY .dat (EdgeMesh2D.BC_NAMES) -> solver flag
_GEOMETRY_BC_FAR_FIELD = 1
_GEOMETRY_BC_WALL = 4


@dataclass
class FliteMesh2DInfo:
    path: str
    n_elements: int
    n_nodes: int
    n_boundary: int
    scale: float
    flags: Dict[int, int]            # edge id -> solver flag
    flag_counts: Dict[int, int]      # solver flag -> number of boundary edges
    node_order: np.ndarray           # file node k (0-based) = mesh node node_order[k]
    boundary_edge_id: np.ndarray     # EdgeMesh2D edge id of each boundary row
    map_path: Optional[str] = None

    def __str__(self):
        counts = ", ".join(f"{FLAG_LABELS.get(f, 'flag')} ({f}): {n}" for f, n in sorted(self.flag_counts.items()))
        return (f"{self.n_elements} triangles, {self.n_nodes} nodes, {self.n_boundary} boundary edges "
                f"[{counts}]" + (f", scale {self.scale:g}" if self.scale != 1.0 else ""))


# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------
def _loop_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def default_edge_flags(mesh, viscous: bool = False) -> Dict[int, Optional[int]]:
    """
    A first guess of the solver flag of every edge, for the user to confirm.

    * edges labelled from a FLITE geometry .dat: far field -> 3, wall -> 1
      (5 if viscous); any other geometry code -> None (must be set by hand);
    * otherwise: edges on the OUTER boundary loop -> far field, edges on inner
      loops (bodies) -> wall. This is right for a body in a far field and
      WRONG for a duct/intake domain, where the outer loop also carries walls,
      inflow and outflow.
    """
    wall = FLAG_VISCOUS_WALL if viscous else FLAG_INVISCID_WALL
    areas = [abs(_loop_area(mesh.nodes[L, :2])) for L in mesh.loops]
    outer = int(np.argmax(areas)) if areas else 0
    out: Dict[int, Optional[int]] = {}
    for e in mesh.edges.values():
        if e.bc is not None:
            out[e.id] = (FLAG_FAR_FIELD if e.bc == _GEOMETRY_BC_FAR_FIELD else
                         wall if e.bc == _GEOMETRY_BC_WALL else None)
        else:
            out[e.id] = FLAG_FAR_FIELD if e.loop == outer else wall
    return out


def parse_flag(text) -> int:
    """'Far field (3)' / '3' / 3 -> 3. Raises ValueError otherwise."""
    if isinstance(text, (int, np.integer)):
        return int(text)
    s = str(text).strip()
    if "(" in s and s.endswith(")"):
        s = s[s.rindex("(") + 1:-1]
    v = int(s)
    if v < 0:
        raise ValueError(f"boundary flag must be >= 0, got {v}")
    return v


def flag_text(flag: int) -> str:
    return f"{FLAG_LABELS[flag]} ({flag})" if flag in FLAG_LABELS else str(int(flag))


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
def _boundary_rows(mesh, flags: Dict[int, int]):
    """(a, b, flag, edge id) for every boundary segment, loop by loop, walls first."""
    seg_edge = {}
    for e in mesh.edges.values():
        nd = [int(n) for n in e.nodes]
        for a, b in zip(nd[:-1], nd[1:]):
            if a != b:
                seg_edge[(a, b)] = e.id
    loops = []
    for L in mesh.loops:
        rows = []
        for k in range(len(L)):
            a, b = int(L[k]), int(L[(k + 1) % len(L)])
            eid = seg_edge.get((a, b))
            if eid is None:
                raise ValueError(f"Boundary segment {a}-{b} belongs to no labelled edge; "
                                 "the edge labels do not cover the whole boundary.")
            if eid not in flags or flags[eid] is None:
                raise ValueError(f"Edge {eid} has no solver boundary flag.")
            rows.append((a, b, int(flags[eid]), int(eid)))
        loops.append(rows)
    # loops that carry a wall first (stable), as the original mesh generator did
    loops.sort(key=lambda rows: 0 if any(r[2] in WALL_FLAGS for r in rows) else 1)
    return [r for rows in loops for r in rows]


def write_flite2d_mesh(mesh, path: str, edge_flags: Dict[int, int], scale: float = 1.0,
                       title: str = "title", write_map: bool = False) -> FliteMesh2DInfo:
    """
    Write an EdgeMesh2D as a FLITE 2D preprocessor mesh.

    edge_flags : {edge id: solver boundary flag} for EVERY edge of the mesh.
    scale      : coordinates are multiplied by this (e.g. 1e-3 for a mesh in mm
                 that the solver should see with a chord of 1). The solver is
                 non-dimensional: forces are per unit of ITS length, and the
                 Reynolds number is per unit of its length.
    write_map  : also write <stem>.map.npz (node_order, boundary_edge_id,
                 edge ids and flags) next to the file, to map solver output
                 back to the mesh. The numbering depends only on the mesh
                 topology and the flags, so one map serves every deformation
                 of the same baseline.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    flags = {int(k): (None if v is None else parse_flag(v)) for k, v in dict(edge_flags).items()}
    unknown = set(flags) - set(mesh.edges)
    if unknown:
        raise ValueError(f"Flags given for unknown edge ids {sorted(unknown)}; mesh has {sorted(mesh.edges)}.")

    P = np.asarray(mesh.nodes[:, :2], float)
    T = np.asarray(mesh.triangles, np.int64)
    N = len(P)
    if not np.all(np.isfinite(P)):
        raise ValueError("Mesh has non-finite coordinates.")

    rows = _boundary_rows(mesh, flags)

    # ---- checks: the solver turns these into NaN, so stop here instead ----
    area = mesh.triangle_areas()
    if np.any(area <= 0.0):
        bad = np.flatnonzero(area <= 0.0)
        raise ValueError(f"{len(bad)} triangle(s) with zero or negative area (first: element {int(bad[0]) + 1}); "
                         "the mesh is inverted or degenerate.")
    E = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    if float(np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1).min()) <= 0.0:
        raise ValueError("Mesh has coincident nodes (a zero-length element edge).")

    # ---- numbering: boundary nodes first, in the order of the boundary rows ----
    order, seen = [], np.zeros(N, bool)
    for a, _b, _f, _e in rows:
        if not seen[a]:
            seen[a] = True
            order.append(a)
    if len(order) != len(rows):
        raise ValueError("Boundary loops share a node; not supported.")
    order = np.concatenate([np.asarray(order, np.int64), np.flatnonzero(~seen)])
    new_of = np.empty(N, np.int64)
    new_of[order] = np.arange(N)

    ne, nb = len(T), len(rows)
    Tn = new_of[T] + 1
    Pn = P[order] * scale

    lines = ["1", str(title).strip() or "title", "ne\tnp\tnb\t", f"{ne:11d}\t{N:11d}\t{nb:11d}", "connectivities"]
    lines += [f"{i + 1:11d}\t{t[0]:11d}\t{t[1]:11d}\t{t[2]:11d}\t{1:11d}" for i, t in enumerate(Tn)]
    lines.append("coordinates")
    lines += [f"{i + 1:d}\t{p[0]:.16e}\t{p[1]:.16e}" for i, p in enumerate(Pn)]
    lines.append("boundaries")
    lines += [f"{new_of[a] + 1:11d}\t{new_of[b] + 1:11d}\t{f:11d}" for a, b, f, _e in rows]

    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as fh:           # unix newlines, as the original
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)                              # never leave a half-written mesh

    bflag = np.asarray([r[2] for r in rows], np.int64)
    info = FliteMesh2DInfo(
        path=path, n_elements=ne, n_nodes=N, n_boundary=nb, scale=scale,
        flags={k: v for k, v in flags.items() if v is not None},
        flag_counts={int(f): int((bflag == f).sum()) for f in np.unique(bflag)},
        node_order=order, boundary_edge_id=np.asarray([r[3] for r in rows], np.int64))
    if write_map:
        info.map_path = os.path.splitext(path)[0] + ".map.npz"
        np.savez_compressed(info.map_path, node_order=order, boundary_edge_id=info.boundary_edge_id,
                            boundary_flag=bflag, scale=np.asarray(scale),
                            edge_ids=np.asarray(sorted(info.flags), np.int64),
                            edge_flags=np.asarray([info.flags[k] for k in sorted(info.flags)], np.int64))
    return info


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
def is_flite2d_mesh(path: str) -> bool:
    """True if `path` looks like a FLITE 2D solver mesh (not a geometry .dat)."""
    try:
        with open(path, "r", errors="replace") as fh:
            head = [fh.readline().strip().lower() for _ in range(5)]
    except OSError:
        return False
    return len(head) >= 5 and head[4] == "connectivities" and head[2].split()[:3] == ["ne", "np", "nb"]


def read_flite2d_mesh(path: str) -> dict:
    """
    Read a FLITE 2D solver mesh. Returns
        {"title", "nodes" (np, 2), "triangles" (ne, 3) 0-based,
         "boundary" (nb, 3): node a, node b (0-based), flag}
    """
    with open(path, "r", errors="replace") as fh:
        L = [ln.rstrip("\r\n") for ln in fh]
    if not is_flite2d_mesh(path):
        raise ValueError(f"{path}: not a FLITE 2D solver mesh (expected 'ne np nb' / 'connectivities').")
    try:
        ne, npn, nb = (int(float(v)) for v in L[3].split()[:3])
        i0 = 5
        T = np.array([[int(float(v)) for v in L[i0 + i].split()[1:4]] for i in range(ne)], np.int64) - 1
        ic = i0 + ne
        if L[ic].strip().lower() != "coordinates":
            raise ValueError(f"expected 'coordinates' on line {ic + 1}")
        X = np.array([[float(v) for v in L[ic + 1 + i].split()[1:3]] for i in range(npn)], float)
        ib = ic + 1 + npn
        if L[ib].strip().lower() != "boundaries":
            raise ValueError(f"expected 'boundaries' on line {ib + 1}")
        B = np.array([[int(float(v)) for v in L[ib + 1 + i].split()[:3]] for i in range(nb)], np.int64)
    except (IndexError, ValueError) as e:
        raise ValueError(f"{path}: malformed FLITE 2D mesh ({e}).")
    B[:, :2] -= 1
    if T.size and (T.min() < 0 or T.max() >= npn) or B.size and (B[:, :2].min() < 0 or B[:, :2].max() >= npn):
        raise ValueError(f"{path}: node ids out of range.")
    return {"title": L[1].strip(), "nodes": X, "triangles": T, "boundary": B}


def check_flite2d_mesh(path: str) -> dict:
    """
    Problems the solver cannot survive, for a file already on disk (e.g. one
    written by another tool). Returns counts; all zero means the file is sound.
    """
    m = read_flite2d_mesh(path)
    P, T, B = m["nodes"], m["triangles"], m["boundary"]
    a, b = P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]
    area = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    E = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    und, cnt = np.unique(np.sort(E, axis=1), axis=0, return_counts=True)
    topo = {tuple(e) for e in und[cnt == 1]}
    listed = {tuple(sorted((int(x), int(y)))) for x, y in B[:, :2]}
    return {
        "zero_area_triangles": int((area == 0.0).sum()),
        "clockwise_triangles": int((area < 0.0).sum()),
        "coincident_node_pairs": int(len(P) - len(np.unique(P, axis=0))),
        "zero_length_boundary_edges": int((np.linalg.norm(P[B[:, 0]] - P[B[:, 1]], axis=1) == 0.0).sum()),
        "boundary_edges_missing": len(topo - listed),
        "boundary_edges_not_on_boundary": len(listed - topo),
    }


# ---------------------------------------------------------------------------
# convenience: flags stored in morph_basis.json
# ---------------------------------------------------------------------------
def flags_from_basis(basis: dict) -> Dict[int, int]:
    """{edge id: flag} from morph_basis.json["solver_bc_flags"] (keys are strings in JSON)."""
    raw = basis.get("solver_bc_flags") or {}
    if not raw:
        raise ValueError("morph_basis.json has no 'solver_bc_flags'; re-save the 2D basis from the GUI.")
    return {int(k): parse_flag(v) for k, v in raw.items()}


def write_from_basis(mesh, path: str, basis_or_path, write_map: bool = False) -> FliteMesh2DInfo:
    """
    Write `mesh` (baseline or morphed EdgeMesh2D) with the flags and scale
    saved in morph_basis.json. This is the call for "after each deformation".
    """
    basis = basis_or_path
    if isinstance(basis_or_path, str):
        with open(basis_or_path, "r", encoding="utf-8") as fh:
            basis = json.load(fh)
    return write_flite2d_mesh(mesh, path, flags_from_basis(basis),
                              scale=float(basis.get("solver_mesh_scale", 1.0) or 1.0), write_map=write_map)
