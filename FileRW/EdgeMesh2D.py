"""
EdgeMesh2D.py

2D planar domain mesh (triangles) with labelled boundary edges.

Purpose
-------
Lets the morph / optimisation code treat a 2D case the way it treats a 3D
surface mesh: boundary *edges* play the role that boundary *surfaces* play in
3D. The class exposes the FroFile-style accessors the rest of the code uses
(nodes, node_count, node_connections, get_surface_ids, get_surface_nodes,
convert_node_ids_to_coordinates, farfield_ids, copy, write_file).

Inputs
------
* the domain mesh: .fro (2D triangulation written as a single "surface"),
  or .vtm / .vtk / .vtu read through pyvista;
* optionally the FLITE 2D geometry .dat, used to name the boundary edges.

Edge labelling
--------------
1. From the .dat (preferred). Each .dat curve's two end points must coincide
   with mesh boundary nodes (mesh generators always place a node at curve
   end points). The boundary loop is cut at those nodes, and each piece takes
   the curve id and bc code. If any end point does not match, we raise:
   a .dat that does not describe this mesh must never label it silently.
   We do NOT label node-by-node by distance to the .dat polylines: .dat
   points are spline control points, and the chord sag between them can be
   larger than the mesh spacing (about 0.37 on a R=25 far field with 20 deg
   spacing), so no fixed distance tolerance is reliable.
2. Fallback (no .dat): cut each loop at corners whose turning angle exceeds
   `corner_deg`; edges get generic names and no bc code.

Conventions
-----------
* Triangles are re-oriented counter-clockwise on load.
* Boundary loops and every edge's node list run with the FLUID ON THE LEFT.
  So the left normal (-t_y, t_x) of the walking direction points into the
  fluid, with no centroid heuristics.
* Node ids are 0-based everywhere in Python, 1-based only in written files.
* Only planes of constant z are supported. A planar mesh in another plane
  raises, rather than being rotated silently.
"""
from __future__ import annotations

import copy as _copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# bc codes of the FLITE 2D .dat. Single place to edit.
# 2 is assumed to be outflow (vertical boundaries at the exit) - UNCONFIRMED.
# ---------------------------------------------------------------------------
BC_NAMES = {1: "farfield", 2: "outflow", 4: "wall"}
FARFIELD_BC_CODES = {1}


# ---------------------------------------------------------------------------
# Dimension detection
# ---------------------------------------------------------------------------
def detect_dim(points, rel_tol: float = 1e-8) -> int:
    """
    2 if the point cloud is planar, else 3.

    Uses the thinnest principal direction relative to the largest. Run this on
    ALL blocks merged: a 3D intake contains planar blocks (symmetry planes,
    flat ramps) that would be misclassified one block at a time.
    """
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 3:
        raise ValueError(f"points must be (N>=3, 3); got {P.shape}")
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    if s[0] <= 0.0:
        raise ValueError("Degenerate point cloud (all points coincide).")
    return 2 if s[-1] / s[0] < rel_tol else 3


def _require_constant_z(P, rel_tol=1e-8):
    span = float(np.ptp(P[:, 2]))
    size = float(np.linalg.norm(np.ptp(P, axis=0)))
    if span > rel_tol * max(size, 1e-300):
        raise ValueError(
            "Mesh is planar but not in a z = const plane "
            f"(z span {span:.3e}). Rotate it into the xy-plane before loading."
        )


# ---------------------------------------------------------------------------
# FLITE 2D .dat
# ---------------------------------------------------------------------------
@dataclass
class DatCurve:
    id: int
    bc: int
    point_ids: List[int]
    xy: np.ndarray  # (n, 2)


@dataclass
class Dat2D:
    points: Dict[int, np.ndarray]
    curves: List[DatCurve]
    path: str = ""


def read_dat_2d(path: str) -> Dat2D:
    """
    FLITE 2D geometry file:

        npoin  nseg  nvseg  nlayer
        54 5 0 0 0.000
        <id> <x> <y>            (npoin lines)
        <seg id> <npts> <bc>    (nseg blocks of two lines)
        <point ids ...>         (may wrap over several lines)
    """
    with open(path, "r", errors="replace") as f:
        lines = [ln.strip() for ln in f.read().splitlines()]
    lines = [ln for ln in lines if ln]

    try:
        hdr = lines[1].split()
        npoin, nseg = int(hdr[0]), int(hdr[1])
    except Exception as e:
        raise ValueError(f"{path}: cannot read 'npoin nseg' from line 2: {e}")

    points: Dict[int, np.ndarray] = {}
    for k in range(npoin):
        t = lines[2 + k].split()
        points[int(t[0])] = np.array([float(t[1]), float(t[2])])

    # remaining tokens are segment headers + id lists (lists may wrap)
    toks = " ".join(lines[2 + npoin:]).split()
    pos = 0
    curves: List[DatCurve] = []
    for _ in range(nseg):
        if pos + 3 > len(toks):
            raise ValueError(f"{path}: file ends inside the segment list")
        sid, n, bc = int(toks[pos]), int(toks[pos + 1]), int(toks[pos + 2])
        pos += 3
        ids = [int(x) for x in toks[pos:pos + n]]
        pos += n
        if len(ids) != n:
            raise ValueError(f"{path}: segment {sid} expects {n} points, found {len(ids)}")
        missing = [i for i in ids if i not in points]
        if missing:
            raise ValueError(f"{path}: segment {sid} references unknown points {missing}")
        curves.append(DatCurve(sid, bc, ids, np.array([points[i] for i in ids])))
    return Dat2D(points=points, curves=curves, path=path)


# ---------------------------------------------------------------------------
# Edge record
# ---------------------------------------------------------------------------
@dataclass
class Edge2D:
    id: int
    name: str
    nodes: np.ndarray            # ordered node ids, fluid on the left
    bc: Optional[int] = None
    loop: int = 0
    source: str = "corner"       # "dat" or "corner"
    reversed_wrt_dat: bool = False

    @property
    def bc_name(self) -> str:
        if self.bc is None:
            return "-"
        return BC_NAMES.get(self.bc, f"bc{self.bc}")


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def _point_polyline_dist(Q, poly):
    """Distance from each point of Q (m,2) to polyline poly (n,2)."""
    Q = np.asarray(Q, float)
    A, B = poly[:-1], poly[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-300)
    AQ = Q[:, None, :] - A[None, :, :]
    t = np.clip((AQ * AB[None]).sum(2) / L2[None], 0.0, 1.0)
    C = A[None] + t[..., None] * AB[None]
    return np.sqrt(((Q[:, None, :] - C) ** 2).sum(2)).min(1)


def _fortran_e(v: float) -> str:
    """0.dddddddddddddE+xx, the layout FLITE writes."""
    v = float(v)
    if v == 0.0 or not np.isfinite(v):
        return " 0.000000000000E+00" if v == 0.0 else f" {v}"
    e = int(np.floor(np.log10(abs(v)))) + 1
    m = v / 10.0 ** e
    s = f"{m:.12f}"
    if s.lstrip("-").startswith("1."):          # rounding up to 1.0
        e += 1
        s = f"{v / 10.0 ** e:.12f}"
    if not s.startswith("-"):
        s = " " + s
    return f"{s}E{e:+03d}"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class EdgeMesh2D:
    dim = 2

    def __init__(self, nodes, triangles, *, name: str = "", header_tokens=None):
        P = np.asarray(nodes, float)
        if P.ndim != 2 or P.shape[1] not in (2, 3):
            raise ValueError(f"nodes must be (N,2) or (N,3); got {P.shape}")
        if P.shape[1] == 2:
            P = np.column_stack([P, np.zeros(len(P))])
        if detect_dim(P) != 2:
            raise ValueError("Mesh is not planar: this is a 3D geometry.")
        _require_constant_z(P)

        T = np.asarray(triangles, dtype=np.int64)
        if T.ndim != 2 or T.shape[1] != 3:
            raise ValueError(f"triangles must be (M,3); got {T.shape}")
        if T.min() < 0 or T.max() >= len(P):
            raise ValueError("triangle node ids out of range")

        self.name = name
        self.nodes = P
        self.triangles = self._orient_ccw(P, T)
        self._header_tokens = list(header_tokens) if header_tokens else None

        self.loops: List[np.ndarray] = self._boundary_loops()
        self.edges: Dict[int, Edge2D] = {}
        self.label_source = "none"
        self._node_connections = None

    # ------------------------------------------------------------------ io
    @classmethod
    def from_file(cls, mesh_path: str, dat_path: Optional[str] = None, *,
                  corner_deg: float = 30.0, endpoint_tol_frac: float = 1e-5,
                  on_dat_mismatch: str = "raise") -> "EdgeMesh2D":
        """
        Load a 2D mesh and label its edges.

        on_dat_mismatch: "raise" (default) or "corners" (warn, fall back to
        corner splitting). Keep "raise" for anything automated.
        """
        ext = os.path.splitext(mesh_path)[1].lower()
        if ext == ".fro":
            m = cls._from_fro(mesh_path)
        elif ext in (".vtm", ".vtk", ".vtu", ".vtp"):
            m = cls._from_pyvista(mesh_path)
        else:
            raise ValueError(f"Unsupported 2D mesh format: {ext}")

        if dat_path:
            try:
                m.label_from_dat(read_dat_2d(dat_path), endpoint_tol_frac=endpoint_tol_frac)
            except ValueError as e:
                if on_dat_mismatch != "corners":
                    raise
                print(f"[EdgeMesh2D][WARN] {e}\n[EdgeMesh2D][WARN] Falling back to corner splitting.")
                m.label_by_corners(corner_deg)
        else:
            m.label_by_corners(corner_deg)
        return m

    @classmethod
    def _from_fro(cls, path: str) -> "EdgeMesh2D":
        with open(path) as f:
            lines = f.read().splitlines()
        hdr = lines[0].split()
        if len(hdr) == 8:
            n_quad, n_tri, n_node = 0, int(hdr[0]), int(hdr[1])
        else:
            n_quad, n_tri, n_node = int(hdr[0]), int(hdr[1]), int(hdr[2])
        if n_quad:
            raise NotImplementedError("2D meshes with quads are not supported yet.")
        P = np.array([[float(v) for v in lines[1 + i].split()[1:4]] for i in range(n_node)])
        i0 = 1 + n_node
        T = np.array([[int(v) - 1 for v in lines[i0 + i].split()[1:4]] for i in range(n_tri)],
                     dtype=np.int64)
        sids = {int(lines[i0 + i].split()[4]) for i in range(n_tri)}
        if len(sids) > 1:
            print(f"[EdgeMesh2D][WARN] {path}: triangles carry several surface ids {sorted(sids)}; "
                  "treating them as one 2D domain.")
        return cls(P, T, name=os.path.splitext(os.path.basename(path))[0],
                   header_tokens=hdr if len(hdr) == 8 else None)

    @classmethod
    def _from_pyvista(cls, path: str) -> "EdgeMesh2D":
        import pyvista as pv
        obj = pv.read(path)
        if isinstance(obj, pv.MultiBlock):
            obj = obj.combine(merge_points=True)
        grid = obj.cast_to_unstructured_grid()
        cd = grid.cells_dict
        tri_key = int(pv.CellType.TRIANGLE)
        allowed = {tri_key, int(pv.CellType.LINE), int(pv.CellType.VERTEX), int(pv.CellType.POLY_LINE)}
        other = [k for k in cd if int(k) not in allowed]
        if other:
            raise NotImplementedError(
                f"{path}: cell types {other} present; only triangles are supported in 2D for now."
            )
        if tri_key not in cd:
            raise ValueError(f"{path}: no triangle cells found.")
        T = np.asarray(cd[tri_key], dtype=np.int64)
        P = np.asarray(grid.points, float)
        # drop points not used by triangles (merged blocks can leave strays)
        used = np.unique(T)
        if len(used) != len(P):
            remap = -np.ones(len(P), dtype=np.int64)
            remap[used] = np.arange(len(used))
            P, T = P[used], remap[T]
        return cls(P, T, name=os.path.splitext(os.path.basename(path))[0])

    def write_file(self, filename: str) -> str:
        """Write the triangulation as a .fro (single surface id 1)."""
        nn, nt = self.node_count, len(self.triangles)
        if self._header_tokens and len(self._header_tokens) == 8:
            h = list(self._header_tokens)
            h[0], h[1] = str(nt), str(nn)
        else:
            h = [str(nt), str(nn), "1", "0", "0", "10", "0", "0"]
        out = ["    " + "    ".join(h)]
        for i, p in enumerate(self.nodes, 1):
            out.append(f"{i:>9d} {_fortran_e(p[0])} {_fortran_e(p[1])} {_fortran_e(p[2])}")
        for i, t in enumerate(self.triangles, 1):
            out.append(f"{i:>10d}{t[0] + 1:>10d}{t[1] + 1:>10d}{t[2] + 1:>10d}{1:>10d}")
        d = os.path.dirname(os.path.abspath(filename))
        os.makedirs(d, exist_ok=True)
        with open(filename, "w") as f:
            f.write("\n".join(out) + "\n")
        return filename

    def write_vtk(self, filename: str) -> str:
        """Write a .vtu/.vtk with an 'edge_id' point array on the boundary (0 = interior)."""
        import pyvista as pv
        cells = np.column_stack([np.full(len(self.triangles), 3), self.triangles]).ravel()
        g = pv.UnstructuredGrid(cells, np.full(len(self.triangles), pv.CellType.TRIANGLE), self.nodes)
        lab = np.zeros(self.node_count, dtype=np.int32)
        for e in self.edges.values():
            lab[e.nodes] = e.id
        g.point_data["edge_id"] = lab
        g.save(filename)
        return filename

    # ------------------------------------------------------------ topology
    @staticmethod
    def _orient_ccw(P, T):
        a = P[T[:, 1], :2] - P[T[:, 0], :2]
        b = P[T[:, 2], :2] - P[T[:, 0], :2]
        A2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        if np.any(A2 == 0.0):
            raise ValueError(f"{int((A2 == 0).sum())} degenerate (zero-area) triangles in the mesh.")
        T = T.copy()
        neg = A2 < 0
        T[neg] = T[neg][:, [0, 2, 1]]
        return T

    def _boundary_loops(self) -> List[np.ndarray]:
        T = self.triangles
        directed = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        und = np.sort(directed, axis=1)
        _, inv, cnt = np.unique(und, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        if cnt.max() > 2:
            raise ValueError("Non-manifold mesh: an edge is shared by more than two triangles.")
        bnd = directed[cnt[inv] == 1]     # CCW triangles -> fluid on the left of a->b
        nxt: Dict[int, int] = {}
        for a, b in bnd:
            a, b = int(a), int(b)
            if a in nxt:
                raise ValueError(f"Boundary node {a} has two outgoing boundary edges "
                                 "(loops touching at a point); not supported.")
            nxt[a] = b
        loops, seen = [], set()
        for start in sorted(nxt):
            if start in seen:
                continue
            loop, cur = [], start
            while cur not in seen:
                seen.add(cur)
                loop.append(cur)
                cur = nxt[cur]
            if cur != start:
                raise ValueError("Open boundary chain found; the mesh boundary is not closed.")
            loops.append(np.array(loop, dtype=np.int64))
        # largest loop (outer boundary) first, for stable numbering
        loops.sort(key=lambda L: -len(L))
        return loops

    # ------------------------------------------------------------ labelling
    def label_from_dat(self, dat: Dat2D, endpoint_tol_frac: float = 1e-5):
        P2 = self.nodes[:, :2]
        bnodes = np.concatenate(self.loops)
        loop_of = {int(n): li for li, L in enumerate(self.loops) for n in L}
        pos_in = {int(n): k for L in self.loops for k, n in enumerate(L)}
        diag = float(np.linalg.norm(np.ptp(P2, axis=0)))
        tol = endpoint_tol_frac * diag

        def snap(xy, what):
            d = np.linalg.norm(P2[bnodes] - xy, axis=1)
            k = int(np.argmin(d))
            if d[k] > tol:
                raise ValueError(
                    f".dat does not match this mesh: {what} at ({xy[0]:.6g}, {xy[1]:.6g}) has no "
                    f"boundary node within {tol:.3g} (nearest is {d[k]:.3g} away). "
                    f"Check that '{os.path.basename(dat.path)}' and the mesh are the same design."
                )
            return int(bnodes[k])

        edges: Dict[int, Edge2D] = {}
        cover = {}
        for c in dat.curves:
            a = snap(c.xy[0], f"curve {c.id} start (point {c.point_ids[0]})")
            b = snap(c.xy[-1], f"curve {c.id} end (point {c.point_ids[-1]})")
            if loop_of[a] != loop_of[b]:
                raise ValueError(f".dat curve {c.id}: its end points lie on different boundary loops.")
            L = self.loops[loop_of[a]]
            n = len(L)
            ia, ib = pos_in[a], pos_in[b]
            if a == b:          # closed curve = whole loop
                fwd = np.concatenate([L[ia:], L[:ia], [a]])
                cand = [(fwd, False)]
            else:
                fwd = L[(ia + np.arange((ib - ia) % n + 1)) % n]       # a -> b, fluid on left
                bwd = L[(ib + np.arange((ia - ib) % n + 1)) % n]       # b -> a, fluid on left
                cand = [(fwd, False), (bwd, True)]
            # the path that follows the curve: smaller median distance to the .dat polyline
            scores = [float(np.median(_point_polyline_dist(P2[p], c.xy))) for p, _ in cand]
            path, rev = cand[int(np.argmin(scores))]
            edges[c.id] = Edge2D(id=int(c.id), name=f"{BC_NAMES.get(c.bc, 'bc%d' % c.bc)}_{c.id}",
                                 nodes=np.asarray(path, np.int64), bc=int(c.bc), loop=loop_of[a],
                                 source="dat", reversed_wrt_dat=rev)
            for u, v in zip(path[:-1], path[1:]):
                cover.setdefault((int(u), int(v)), []).append(c.id)

        # every boundary edge exactly once
        expected = {(int(L[k]), int(L[(k + 1) % len(L)])) for L in self.loops for k in range(len(L))}
        dup = {e: ids for e, ids in cover.items() if len(ids) > 1}
        missing = expected - set(cover)
        if dup or missing:
            raise ValueError(
                f".dat curves do not tile the mesh boundary: {len(missing)} boundary edges uncovered, "
                f"{len(dup)} covered twice (e.g. curves {next(iter(dup.values())) if dup else '-'})."
            )
        self.edges = dict(sorted(edges.items()))
        self.label_source = f"dat:{os.path.basename(dat.path)}"
        return self.edges

    def label_by_corners(self, corner_deg: float = 30.0):
        P2 = self.nodes[:, :2]
        edges: Dict[int, Edge2D] = {}
        eid = 1
        for li, L in enumerate(self.loops):
            prev, cur, nxt = P2[np.roll(L, 1)], P2[L], P2[np.roll(L, -1)]
            t1, t2 = cur - prev, nxt - cur
            ang = np.degrees(np.abs(np.arctan2(t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0],
                                               (t1 * t2).sum(1))))
            cuts = np.flatnonzero(ang > corner_deg)
            if len(cuts) == 0:
                edges[eid] = Edge2D(eid, f"edge_{eid}", np.concatenate([L, L[:1]]), loop=li)
                eid += 1
                continue
            n = len(L)
            for k, c0 in enumerate(cuts):
                c1 = cuts[(k + 1) % len(cuts)]
                idx = (c0 + np.arange((c1 - c0) % n + 1 if len(cuts) > 1 else n + 1)) % n
                edges[eid] = Edge2D(eid, f"edge_{eid}", L[idx].astype(np.int64), loop=li)
                eid += 1
        self.edges = edges
        self.label_source = f"corners:{corner_deg:g}deg"
        return self.edges

    # ------------------------------------------------- FroFile-style interface
    @property
    def node_count(self) -> int:
        return int(len(self.nodes))

    @property
    def boundary_triangles(self) -> np.ndarray:
        """(M,4) with surface id 1, for code that expects the FroFile layout."""
        return np.column_stack([self.triangles, np.ones(len(self.triangles), dtype=np.int64)])

    @property
    def boundary_triangle_count(self) -> int:
        return int(len(self.triangles))

    @property
    def node_connections(self) -> Dict[int, List[int]]:
        if self._node_connections is None:
            nc = {i: set() for i in range(self.node_count)}
            for a, b, c in self.triangles:
                a, b, c = int(a), int(b), int(c)
                nc[a] |= {b, c}; nc[b] |= {a, c}; nc[c] |= {a, b}
            self._node_connections = {k: sorted(v) for k, v in nc.items()}
        return self._node_connections

    def set_node_connections(self, tol=None):
        self._node_connections = None
        return self.node_connections

    def get_surface_ids(self) -> List[int]:
        return list(self.edges.keys())

    def get_surface_nodes(self, surf_id: int):
        """Same return shape as Mesh.get_surface_nodes: (gid -> local, ordered gids)."""
        e = self.edges[int(surf_id)]
        g = [int(x) for x in dict.fromkeys(e.nodes.tolist())]   # ordered, closed loops de-duplicated
        return {gid: k for k, gid in enumerate(g)}, g

    get_edge_nodes = get_surface_nodes

    def convert_node_ids_to_coordinates(self, node_ids):
        ids = [int(i) for i in node_ids]
        return {g: k for k, g in enumerate(ids)}, self.nodes[ids].copy()

    @property
    def farfield_ids(self) -> List[int]:
        return [e.id for e in self.edges.values() if e.bc in FARFIELD_BC_CODES]

    @property
    def boundary_node_ids(self) -> np.ndarray:
        return np.unique(np.concatenate(self.loops))

    def copy(self) -> "EdgeMesh2D":
        return _copy.deepcopy(self)

    def with_nodes(self, new_nodes) -> "EdgeMesh2D":
        """Copy with moved nodes; topology and labels unchanged."""
        m = self.copy()
        new_nodes = np.asarray(new_nodes, float)
        if new_nodes.shape != self.nodes.shape:
            raise ValueError(f"new_nodes shape {new_nodes.shape} != {self.nodes.shape}")
        m.nodes = new_nodes.copy()
        return m

    # ----------------------------------------------------------- diagnostics
    def triangle_areas(self, nodes=None) -> np.ndarray:
        P = self.nodes if nodes is None else np.asarray(nodes, float)
        T = self.triangles
        a = P[T[:, 1], :2] - P[T[:, 0], :2]
        b = P[T[:, 2], :2] - P[T[:, 0], :2]
        return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def summary(self) -> str:
        rows = [f"EdgeMesh2D '{self.name}': {self.node_count} nodes, {len(self.triangles)} triangles, "
                f"{len(self.loops)} boundary loop(s), edges from {self.label_source}"]
        for e in self.edges.values():
            P = self.nodes[e.nodes, :2]
            L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
            rows.append(f"  edge {e.id:>3}  {e.name:<16} bc={e.bc_name:<9} loop={e.loop} "
                        f"nodes={len(e.nodes):>5}  length={L:.4g}")
        return "\n".join(rows)

    def __repr__(self):
        return self.summary()