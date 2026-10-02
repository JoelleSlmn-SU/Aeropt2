"""
edgeModes2D.py

Laplacian eigenmodes along the T edge(s) of a 2D mesh, applied as normal
displacements to EVERY T node (no control nodes, no RBF on T).

Maths
-----
Order the T nodes along the curve and let h_i be the length of segment
(i, i+1). With linear finite elements on the curve:

    K_ii = 1/h_{i-1} + 1/h_i,   K_i,i+1 = -1/h_i        (stiffness)
    M_ii = (h_{i-1} + h_i) / 2                           (lumped mass)

and we solve the generalised problem  K phi = lambda M phi.

Using M (not the plain graph Laplacian) makes the modes functions of ARC
LENGTH, independent of how the nodes are spaced: on a curve of length L with
both ends fixed, phi_k -> sin(k pi s / L) and lambda_k -> (k pi / L)^2. The
plain graph Laplacian would instead give modes in node index, distorted by
any clustering (e.g. near a lip).

End conditions
--------------
* "fixed"  (default): Dirichlet, phi = 0 at both ends -> no step where T
  meets the neighbouring edge.
* "free": Neumann; the constant mode (a uniform normal offset) is dropped.
* closed T chains are periodic; the constant mode is dropped.

Normalisation and sign
----------------------
Each mode is scaled so that integral(phi_k^2 ds) = L/2, the value for a unit
sine. A design coefficient a_k is then the amplitude of the continuous mode
(~ its peak normal displacement) in mesh length units, independent of the
node spacing, which keeps BO bounds physically meaningful across meshes. Sign: the first lobe along the walking
direction is positive. Positive displacement points INTO THE FLUID (the
EdgeMesh2D walking direction has the fluid on the left).

Resolution guard: a mode needs roughly 8 nodes per wavelength, so
k_max ~ n_T / 4. Requests above that are refused rather than aliased.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Chain ordering
# ---------------------------------------------------------------------------
def order_chain(mesh, edge_ids: Sequence[int]):
    """
    Join the selected edges into one ordered node chain (fluid on the left).

    Returns (gids, closed). Raises if the edges are not one contiguous chain
    (e.g. two separate walls selected as T).
    """
    if not edge_ids:
        raise ValueError("No T edges given.")
    succ, pred = {}, {}
    for eid in edge_ids:
        nodes = np.asarray(mesh.edges[int(eid)].nodes, np.int64)
        for a, b in zip(nodes[:-1], nodes[1:]):
            a, b = int(a), int(b)
            if a == b:
                continue
            if a in succ and succ[a] != b:
                raise ValueError(f"T edges overlap at node {a}.")
            succ[a], pred[b] = b, a

    starts = [n for n in succ if n not in pred]
    if len(starts) > 1:
        raise ValueError(f"T edges {list(edge_ids)} form {len(starts)} separate chains; "
                         "select contiguous edges (or build one basis per chain).")
    closed = len(starts) == 0
    start = min(succ) if closed else starts[0]
    chain, cur, seen = [start], start, {start}
    while cur in succ:
        cur = succ[cur]
        if cur == start:
            break
        if cur in seen:
            raise ValueError("Chain revisits a node; T edges are not a simple curve.")
        chain.append(cur)
        seen.add(cur)
    if len(chain) != len(set(succ) | set(pred)):
        raise ValueError("T edges are not one contiguous chain.")
    return np.asarray(chain, np.int64), closed


# ---------------------------------------------------------------------------
# Geometry along the chain
# ---------------------------------------------------------------------------
def arc_length(xy, closed=False):
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if closed:
        seg = np.append(seg, np.linalg.norm(xy[0] - xy[-1]))
    s = np.concatenate([[0.0], np.cumsum(seg[: len(xy) - 1])])
    return s, seg


def chain_normals(xy, closed=False):
    """
    Unit normals pointing to the LEFT of the walking direction (= into the
    fluid for EdgeMesh2D chains). At interior nodes the two adjacent segment
    normals are averaged, weighted by segment length.
    """
    xy = np.asarray(xy, float)
    if closed:
        nxt = np.roll(xy, -1, axis=0)
        t = nxt - xy                                 # segment i: node i -> i+1
    else:
        t = np.diff(xy, axis=0)
    seg_n = np.column_stack([-t[:, 1], t[:, 0]])     # left normal * segment length
    n = np.zeros_like(xy)
    if closed:
        n += seg_n + np.roll(seg_n, 1, axis=0)
    else:
        n[:-1] += seg_n
        n[1:] += seg_n
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    if np.any(nn < 1e-300):
        raise ValueError("Zero-length normal (cusp or duplicated node on T).")
    return n / nn


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def laplacian_modes(seg, n_modes: int, ends: str = "fixed", closed: bool = False):
    """
    Modes of K phi = lambda M phi on a chain with segment lengths `seg`.

    Returns (lam (k,), phi (n_nodes, k)), phi scaled to integral(phi^2 ds) = L/2.
    """
    from scipy.linalg import eigh

    seg = np.asarray(seg, float)
    if np.any(seg <= 0):
        raise ValueError("Non-positive segment length on T (duplicate nodes).")
    n = len(seg) if closed else len(seg) + 1

    K = np.zeros((n, n))
    Md = np.zeros(n)
    nseg = len(seg)
    for i in range(nseg):
        a, b = i, (i + 1) % n
        w = 1.0 / seg[i]
        K[a, a] += w; K[b, b] += w
        K[a, b] -= w; K[b, a] -= w
        Md[a] += 0.5 * seg[i]; Md[b] += 0.5 * seg[i]

    if closed or ends == "free":
        free = np.arange(n)
        drop_const = True
    elif ends == "fixed":
        free = np.arange(1, n - 1)
        drop_const = False
    else:
        raise ValueError(f"ends must be 'fixed' or 'free', got {ends!r}")

    # Resolution guard on the COARSEST segment, not the node count: mode k has
    # wavelength ~2L/k (L for periodic/free-free pairs), and needs >= 8
    # segments per wavelength everywhere, so k <= L / (4 h_max).
    L_tot = float(seg.sum())
    k_max = max(1, int(np.floor(L_tot / (4.0 * float(seg.max())))))
    if n_modes > k_max:
        raise ValueError(f"{n_modes} modes requested but the coarsest T segment "
                         f"({seg.max():.3g} of L={L_tot:.3g}) only resolves ~{k_max} modes "
                         "(8 segments per wavelength). Refine the T edge or use fewer modes.")

    Kf = K[np.ix_(free, free)]
    Mf = np.diag(Md[free])
    lo = 1 if drop_const else 0
    lam, vec = eigh(Kf, Mf, subset_by_index=[lo, lo + n_modes - 1])

    phi = np.zeros((n, n_modes))
    phi[free] = vec
    L = float(seg.sum())
    for k in range(n_modes):
        col = phi[:, k]
        # arc-length L2 scaling: integral(phi^2 ds) = L/2, as for a unit sine.
        # (max|phi| = 1 would depend on whether a node happens to sit on the
        # peak, i.e. on the mesh.)
        col *= np.sqrt(0.5 * L / float(col @ (Md * col)))
        first = np.flatnonzero(np.abs(col) > 0.5)[0]   # first lobe positive
        if col[first] < 0:
            col *= -1.0
        phi[:, k] = col
    return lam, phi


# ---------------------------------------------------------------------------
# Basis object
# ---------------------------------------------------------------------------
@dataclass
class EdgeModes2D:
    t_edges: List[int]
    gids: np.ndarray          # (n,) ordered T node ids
    xy: np.ndarray            # (n, 2) baseline coordinates
    s: np.ndarray             # (n,) arc length
    length: float
    closed: bool
    ends: str
    normals: np.ndarray       # (n, 2) unit, into the fluid
    lam: np.ndarray           # (k,)
    phi: np.ndarray           # (n, k), integral(phi_k^2 ds) = L/2

    @property
    def n_modes(self) -> int:
        return int(self.phi.shape[1])

    @classmethod
    def from_mesh(cls, mesh, t_edges: Iterable[int], n_modes: int = 8,
                  ends: str = "fixed") -> "EdgeModes2D":
        t_edges = [int(e) for e in t_edges]
        gids, closed = order_chain(mesh, t_edges)
        xy = mesh.nodes[gids, :2]
        s, seg = arc_length(xy, closed)
        lam, phi = laplacian_modes(seg, n_modes, ends=ends, closed=closed)
        return cls(t_edges=t_edges, gids=gids, xy=xy, s=s, length=float(seg.sum()),
                   closed=closed, ends="periodic" if closed else ends,
                   normals=chain_normals(xy, closed), lam=lam, phi=phi)

    def displacement(self, coeffs) -> np.ndarray:
        """Design vector -> (n, 3) displacement of the T nodes (z = 0)."""
        a = np.asarray(coeffs, float).ravel()
        if a.size != self.n_modes:
            raise ValueError(f"expected {self.n_modes} coefficients, got {a.size}")
        amp = self.phi @ a
        d = np.zeros((len(self.gids), 3))
        d[:, :2] = amp[:, None] * self.normals
        return d

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "dim": 2, "kind": "edge_laplacian",
            "t_edges": self.t_edges, "ends": self.ends, "closed": self.closed,
            "length": self.length, "gids": self.gids.tolist(),
            "lam": self.lam.tolist(), "n_modes": self.n_modes,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_saved(cls, mesh, path_or_dict) -> "EdgeModes2D":
        """
        Rebuild from a saved basis on the same baseline mesh (modes are
        recomputed, then checked against the saved node ordering).
        """
        d = path_or_dict
        if isinstance(path_or_dict, str):
            with open(path_or_dict) as f:
                d = json.load(f)
        ends = "fixed" if d["ends"] == "periodic" else d["ends"]
        b = cls.from_mesh(mesh, d["t_edges"], int(d["n_modes"]), ends=ends)
        if b.gids.tolist() != list(d["gids"]):
            raise ValueError("Saved basis node ordering does not match this mesh "
                             "(different baseline or different edge labels).")
        return b