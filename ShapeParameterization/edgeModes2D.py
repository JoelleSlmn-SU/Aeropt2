"""
edgeModes2D.py

Eigenmodes along the T edge(s) of a 2D mesh, applied as normal displacements
to EVERY T node (no control nodes, no RBF on T).

Works for any set of T edges:
  * one or several separate chains (e.g. ramp + cowl),
  * open chains (ends meeting other edges) and closed loops (a whole body),
  * geometric corners inside a chain (sharp/blunt trailing edge, ramp kinks).

Why corners need special treatment
----------------------------------
The displacement is  d(s) = phi(s) * n(s)  with n the unit normal. At a corner
n jumps, so any phi != 0 there tears or folds the curve: the two sides are
pushed in different directions over one mesh spacing. (On a closed aerofoil
loop this is what made the cos-type periodic modes unusable: their antinode
sat on the trailing edge.) The same happens, more gently, wherever the radius
of curvature is tiny: a normal offset larger than the local radius folds the
curve.

So the T chains are cut into SMOOTH SEGMENTS at "pinned" nodes, where phi = 0:

  1. ends of open chains              (ends = "fixed" or "clamped")
  2. corners:  turning angle > corner_deg at a node that is NOT part of a
     coarsely resolved smooth arc (see _is_arc_like); above 2 * corner_deg
     it is a corner regardless
  3. tiny radius: discrete radius of curvature R_i < min_radius_frac * L_chain
     (covers small round caps / lips, and corners on fine meshes)
  4. points the USER fixes (fixed_nodes: any T node, e.g. the leading edge),
     and optionally every junction between two T edges (pin_junctions=True).
     These sit on a smooth part of the curve, so by default they are CLAMPED
     (fixed_type="clamped": position and slope held, the curve stays smooth
     through the point). fixed_type="pinned" holds the position only: the
     point becomes a hinge and the two sides can form a kink there.
     Geometric corners (rules 2, 3) are always hinges.

Each segment gets its own modes; all modes are then ranked together by
wavenumber, so `n_modes` always means "the n smoothest shapes available",
wherever they live. A segment too short/coarse to resolve a mode gets none
and simply stays fixed (e.g. the base of a blunt trailing edge).

Maths (per segment)
-------------------
Linear finite elements in arc length, with h_i the length of segment (i, i+1):

    K_ii = 1/h_{i-1} + 1/h_i,   K_i,i+1 = -1/h_i        (stiffness)
    M_ii = (h_{i-1} + h_i) / 2                           (lumped mass)

  * pinned-pinned ("fixed"):  K phi = lambda M phi, phi = 0 at both ends
        phi_k -> sin(k pi s / L),  lambda_k -> (k pi / L)^2
    C0 at the pins: the wall keeps its position but gets a slope change
    (a kink of ~ a k pi / L radians) where it meets the neighbour.
  * clamped end ("clamped"): phi = 0 AND dphi/ds = 0 there, from the
    biharmonic problem  (K M^-1 K) phi = mu M phi  (Ciarlet-Raviart mixed
    form). No kink where T meets a fixed wall: use it for supersonic walls,
    where a kink is a shock/expansion that the optimiser did not ask for.
  * free end ("free"): Neumann; the constant mode is dropped when both ends
    of the segment are free. A segment clamped at one end and free at the
    other is a cantilever (also from the biharmonic problem).
  * closed loop with no pinned node: periodic; the constant mode is dropped
    and each degenerate (cos, sin) pair is rotated to a fixed phase so the
    basis does not depend on the eigen-solver.

Using M (not the plain graph Laplacian) makes the modes functions of ARC
LENGTH, independent of how the nodes are spaced.

`lam` is reported as kappa^2 for every mode, with kappa the wavenumber
(kappa = sqrt(lambda) for Laplacian segments, mu^(1/4) for clamped ones), so
the values are comparable across segments and end conditions.

Normalisation and sign
----------------------
Each mode is scaled so that integral(phi^2 ds) over ITS SEGMENT = L_seg/2,
the value for a unit sine. A design coefficient a_k is then the amplitude of
the continuous mode (~ its peak normal displacement) in mesh length units,
independent of the node spacing. Sign: the first lobe along the walking
direction is positive. Positive displacement points INTO THE FLUID (the
EdgeMesh2D walking direction has the fluid on the left).

Resolution guard: a mode needs roughly 8 segments per wavelength, so a
segment of length L with coarsest spacing h_max carries k <= L / (4 h_max)
modes. Requests above the total are refused rather than aliased.

Reproducibility
---------------
The basis (phi, normals, node ids, baseline coordinates) is written to the
saved JSON and read back by `from_saved`, so a design vector means the same
shape on every machine, whatever LAPACK does with near-degenerate modes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BASIS_VERSION = 2
END_TYPES = ("fixed", "free", "clamped")
_PAIR_GAP = 5e-2          # periodic (cos, sin) pair: relative eigenvalue gap below this
_TIE_RTOL = 1e-8          # wavenumbers closer than this are ranked by segment order


# ---------------------------------------------------------------------------
# Chain ordering
# ---------------------------------------------------------------------------
def split_chains(mesh, edge_ids: Sequence[int]) -> List[Tuple[np.ndarray, bool]]:
    """
    Join the selected edges into ordered node chains (fluid on the left).

    Returns [(gids, closed), ...], ordered by the first T edge (in the order
    given) that touches each chain. A closed chain lists every node once and
    starts at its smallest node id (EdgeModes2D later rotates it to start at
    a pinned node, if it has one).
    """
    if not len(edge_ids):
        raise ValueError("No T edges given.")
    succ: Dict[int, int] = {}
    pred: Dict[int, int] = {}
    first_edge: Dict[int, int] = {}
    for order, eid in enumerate(edge_ids):
        nodes = np.asarray(mesh.edges[int(eid)].nodes, np.int64)
        for n in nodes:
            first_edge.setdefault(int(n), order)
        for a, b in zip(nodes[:-1], nodes[1:]):
            a, b = int(a), int(b)
            if a == b:
                continue
            if (a in succ and succ[a] != b) or (b in pred and pred[b] != a):
                raise ValueError(f"T edges overlap or branch at node {a if a in succ else b}.")
            succ[a], pred[b] = b, a

    all_nodes = set(succ) | set(pred)
    if not all_nodes:
        raise ValueError("T edges contain no segments.")
    chains, seen = [], set()
    for start in sorted(n for n in succ if n not in pred):      # open chains
        chain, cur = [start], start
        seen.add(start)
        while cur in succ:
            cur = succ[cur]
            if cur in seen:
                raise ValueError("Chain revisits a node; T edges are not simple curves.")
            chain.append(cur)
            seen.add(cur)
        chains.append((np.asarray(chain, np.int64), False))
    remaining = all_nodes - seen                                 # closed loops
    while remaining:
        start = min(remaining)
        chain, cur = [start], succ[start]
        while cur != start:
            if cur not in remaining or cur in chain[1:]:
                raise ValueError("Chain revisits a node; T edges are not simple curves.")
            chain.append(cur)
            cur = succ[cur]
        remaining -= set(chain)
        chains.append((np.asarray(chain, np.int64), True))
    chains.sort(key=lambda c: (min(first_edge[int(n)] for n in c[0]), int(c[0][0])))
    return chains


def order_chain(mesh, edge_ids: Sequence[int]):
    """
    Single-chain version kept for older callers: returns (gids, closed) and
    raises if the edges form more than one chain. New code: split_chains().
    """
    chains = split_chains(mesh, edge_ids)
    if len(chains) > 1:
        raise ValueError(f"T edges {list(edge_ids)} form {len(chains)} separate chains.")
    return chains[0]


# ---------------------------------------------------------------------------
# Geometry along a chain
# ---------------------------------------------------------------------------
def arc_length(xy, closed=False):
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if closed:
        seg = np.append(seg, np.linalg.norm(xy[0] - xy[-1]))
    s = np.concatenate([[0.0], np.cumsum(seg[: len(xy) - 1])])
    return s, seg


def chain_normals(xy, closed=False, strict=True):
    """
    Unit normals pointing to the LEFT of the walking direction (= into the
    fluid for EdgeMesh2D chains). At interior nodes the two adjacent segment
    normals are averaged, weighted by segment length.

    strict=False returns a zero vector where the average vanishes (a cusp:
    the curve doubles back on itself) instead of raising; EdgeModes2D only
    accepts that at pinned nodes, where the normal is never used.
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
    scale = float(np.abs(t).max()) if len(t) else 1.0
    bad = nn[:, 0] < 1e-9 * max(scale, 1e-300)
    if np.any(bad):
        if strict:
            raise ValueError("Zero-length normal (cusp or duplicated node on T).")
        nn = np.where(bad[:, None], 1.0, nn)
        n = np.where(bad[:, None], 0.0, n)
    return n / nn


def turning_angles(xy, closed=False):
    """Signed turning angle (degrees) at each node; 0 at the ends of an open chain."""
    xy = np.asarray(xy, float)
    ang = np.zeros(len(xy))
    if closed:
        t_in = xy - np.roll(xy, 1, axis=0)
        t_out = np.roll(xy, -1, axis=0) - xy
        sl = slice(None)
    else:
        if len(xy) < 3:
            return ang
        t = np.diff(xy, axis=0)
        t_in, t_out = t[:-1], t[1:]
        sl = slice(1, -1)
    cross = t_in[:, 0] * t_out[:, 1] - t_in[:, 1] * t_out[:, 0]
    dot = (t_in * t_out).sum(1)
    ang[sl] = np.degrees(np.arctan2(cross, dot))
    return ang


def _is_arc_like(theta, seg, closed):
    """
    True where a node looks like part of a coarsely resolved SMOOTH arc rather
    than a corner: both neighbours curve the same way, with at least half its
    discrete curvature (turning angle / mean adjacent segment length).
    A real corner has (nearly) straight neighbours, whatever the mesh spacing;
    an under-resolved leading edge does not. Curvature, not the raw angle, so
    that uneven node spacing along a smooth arc is not mistaken for a corner.
    """
    n = len(theta)
    if closed:
        h_in, h_out = np.roll(seg, 1), seg
    else:
        h_in = np.concatenate([[seg[0]], seg])[:n]
        h_out = np.concatenate([seg, [seg[-1]]])[:n]
    curv = theta / (0.5 * (h_in + h_out))             # signed
    prev, nxt = np.roll(curv, 1), np.roll(curv, -1)
    if not closed:
        prev[0], nxt[-1] = 0.0, 0.0
    same = (np.sign(prev) == np.sign(curv)) & (np.sign(nxt) == np.sign(curv))
    a = np.abs(curv)
    return same & (np.minimum(np.abs(prev), np.abs(nxt)) >= 0.5 * a) & (a > 0)


def fold_radius(seg, theta, closed):
    """
    Discrete radius of curvature at each node, R_i = min(h_in, h_out) / (2 tan(|theta_i|/2)).
    A normal offset towards the centre of curvature folds the curve at about
    this amplitude. R -> the true radius on a resolved smooth curve, and -> 0
    under refinement at a true corner. inf where the curve is straight.
    """
    n = len(theta)
    if closed:
        h_in, h_out = np.roll(seg, 1), seg
    else:
        h_in = np.concatenate([[np.inf], seg])
        h_out = np.concatenate([seg, [np.inf]])
    hmin = np.minimum(h_in, h_out)[:n]
    tn = np.tan(0.5 * np.radians(np.abs(theta)))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(tn > 0, hmin / (2.0 * tn), np.inf)
    return R


# ---------------------------------------------------------------------------
# Modes on one smooth segment
# ---------------------------------------------------------------------------
def _stiff_mass(h, periodic):
    h = np.asarray(h, float)
    m = len(h)
    n = m if periodic else m + 1
    K = np.zeros((n, n))
    Md = np.zeros(n)
    a = np.arange(m)
    b = (a + 1) % n
    w = 1.0 / h
    np.add.at(K, (a, a), w)
    np.add.at(K, (b, b), w)
    np.add.at(K, (a, b), -w)
    np.add.at(K, (b, a), -w)
    np.add.at(Md, a, 0.5 * h)
    np.add.at(Md, b, 0.5 * h)
    return K, Md


def segment_modes(h, k: int, left: str = "pinned", right: str = "pinned", periodic: bool = False):
    """
    First k modes of one smooth segment with segment lengths h.

    left/right: "pinned" (phi=0), "clamped" (phi=0, dphi/ds=0), "free".
    periodic=True: closed loop with no pinned node (left/right ignored).

    Returns (kappa (k,), phi (n_nodes, k)): wavenumbers and modes, scaled to
    integral(phi^2 ds) = L/2, first lobe positive. n_nodes = len(h)+1, or
    len(h) if periodic.
    """
    from scipy.linalg import eigh, eigh_tridiagonal

    h = np.asarray(h, float)
    if np.any(h <= 0):
        raise ValueError("Non-positive segment length on T (duplicate nodes).")
    m = len(h)
    n = m if periodic else m + 1
    L = float(h.sum())
    if k <= 0:
        return np.zeros(0), np.zeros((n, 0))

    if periodic:
        K, Md = _stiff_mass(h, True)
        d = 1.0 / np.sqrt(Md)
        S = K * d[:, None] * d[None, :]
        kk = min(k + (k % 2), n - 1)                 # whole (cos, sin) pairs
        lam, vec = eigh(S, subset_by_index=[1, kk])  # index 0 = constant mode
        vec = vec * d[:, None]
        for j in range(0, kk - 1, 2):                # fix the phase of each pair
            if abs(lam[j + 1] - lam[j]) <= _PAIR_GAP * abs(lam[j + 1]):
                va, vb = vec[:, j].copy(), vec[:, j + 1].copy()
                th = np.arctan2(vb[0], va[0])
                vec[:, j] = np.cos(th) * va + np.sin(th) * vb        # cos-like: peak at node 0
                vec[:, j + 1] = -np.sin(th) * va + np.cos(th) * vb   # sin-like: zero at node 0
                vec[0, j + 1] = 0.0
        kappa = np.sqrt(np.clip(lam[:k], 0.0, None))
        phi = vec[:, :k]
    else:
        for e in (left, right):
            if e not in ("pinned", "clamped", "free"):
                raise ValueError(f"unknown end type {e!r}")
        lo_node = 0 if left == "free" else 1
        hi_node = m if right == "free" else m - 1
        F = np.arange(lo_node, hi_node + 1)
        if len(F) == 0:
            return np.zeros(0), np.zeros((n, 0))
        Md = np.zeros(n)
        Md[:-1] += 0.5 * h
        Md[1:] += 0.5 * h
        Kd = np.zeros(n)
        Kd[:-1] += 1.0 / h
        Kd[1:] += 1.0 / h
        Ko = -1.0 / h                                 # K[i, i+1]
        phi = np.zeros((n, 0))

        if "clamped" in (left, right):
            # Biharmonic problem in mixed form: w = -phi'' lives on the interior
            # nodes plus the clamped ends (w free there enforces phi' = 0); w = 0
            # at pinned and free ends (no bending moment). phi is free on F, which
            # includes a free end. With A = M_W^-1/2 K[W,F] M_F^-1/2 the modes are
            # the eigenvectors of A^T A for its SMALLEST eigenvalues mu = kappa^4.
            #
            # A^T A is never formed: its condition number grows like n^4, which
            # would cost the low modes most of their digits on long chains. The
            # inverse (A^T A)^-1 is applied instead through the saddle system
            #     [ -I  A ] [w]   [0]
            #     [ A^T 0 ] [p] = [g]      ->  p = (A^T A)^-1 g
            # (condition ~ n^2), and the modes are its LARGEST eigenpairs.
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import splu

            W = list(range(1, m))
            if left == "clamped":
                W = [0] + W
            if right == "clamped":
                W = W + [m]
            W = np.asarray(W, dtype=np.int64)
            in_W = np.zeros(n, bool); in_W[W] = True
            in_F = np.zeros(n, bool); in_F[F] = True
            iw = -np.ones(n, np.int64)
            jf = -np.ones(n, np.int64)
            cnt = 0
            for i in range(n):                        # interleave by node -> banded system
                if in_W[i]:
                    iw[i] = cnt; cnt += 1
                if in_F[i]:
                    jf[i] = cnt; cnt += 1
            sq = 1.0 / np.sqrt(Md)
            rows, cols, vals = [], [], []
            for i in W:
                rows.append(iw[i]); cols.append(iw[i]); vals.append(-1.0)
                for j in (i - 1, i, i + 1):
                    if 0 <= j <= m and in_F[j]:
                        kij = Kd[i] if i == j else Ko[min(i, j)]
                        v = kij * sq[i] * sq[j]
                        rows += [iw[i], jf[j]]; cols += [jf[j], iw[i]]; vals += [v, v]
            lu = splu(coo_matrix((vals, (rows, cols)), shape=(cnt, cnt)).tocsc())
            rhs = np.zeros((cnt, len(F)))
            rhs[jf[F], np.arange(len(F))] = 1.0
            T = lu.solve(rhs)[jf[F]]
            T = 0.5 * (T + T.T)
            kk = min(k, len(F))
            th, vec = eigh(T, subset_by_index=[len(F) - kk, len(F) - 1])
            th, vec = th[::-1], vec[:, ::-1]          # largest 1/mu first = smallest kappa
            kappa = (1.0 / th) ** 0.25
            d = sq[F]
        else:
            drop = 1 if (left == "free" and right == "free") else 0
            kk = min(k, len(F) - drop)
            if kk <= 0:
                return np.zeros(0), np.zeros((n, 0))
            d = 1.0 / np.sqrt(Md[F])
            dd = Kd[F] * d * d
            ee = Ko[F[:-1]] * d[:-1] * d[1:]
            if len(F) == 1:
                lam, vec = np.array([dd[0]]), np.ones((1, 1))
            else:
                lam, vec = eigh_tridiagonal(dd, ee, select="i", select_range=(drop, drop + kk - 1))
            kappa = np.sqrt(np.clip(lam, 0.0, None))
        phi = np.zeros((n, kk))
        phi[F] = vec * d[:, None]

    for j in range(phi.shape[1]):
        col = phi[:, j]
        # arc-length L2 scaling: integral(phi^2 ds) = L/2, as for a unit sine.
        # (max|phi| = 1 would depend on whether a node happens to sit on the
        # peak, i.e. on the mesh.)
        col *= np.sqrt(0.5 * L / float(col @ (Md * col)))
        big = np.flatnonzero(np.abs(col) > 0.5)
        if len(big) and col[big[0]] < 0:              # first lobe positive
            col *= -1.0
        phi[:, j] = col
    return kappa, phi


def laplacian_modes(seg, n_modes: int, ends: str = "fixed", closed: bool = False):
    """
    Modes of ONE chain without corners (kept for older callers).
    Returns (lam (k,), phi (n_nodes, k)), phi scaled to integral(phi^2 ds) = L/2.
    """
    seg = np.asarray(seg, float)
    L_tot = float(seg.sum())
    k_max = max(1, int(np.floor(L_tot / (4.0 * float(seg.max())))))
    if n_modes > k_max:
        raise ValueError(f"{n_modes} modes requested but the coarsest T segment "
                         f"({seg.max():.3g} of L={L_tot:.3g}) only resolves ~{k_max} modes "
                         "(8 segments per wavelength). Refine the T edge or use fewer modes.")
    if ends not in END_TYPES:
        raise ValueError(f"ends must be one of {END_TYPES}, got {ends!r}")
    e = {"fixed": "pinned", "free": "free", "clamped": "clamped"}[ends]
    kappa, phi = segment_modes(seg, n_modes, e, e, periodic=closed)
    return kappa ** 2, phi


# ---------------------------------------------------------------------------
# Layout: chains -> pinned nodes -> smooth segments
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    chain: int
    idx: np.ndarray           # positions in the basis node arrays, walking order,
                              # both end nodes included (the same node twice on a
                              # closed chain with a single pinned node)
    h: np.ndarray             # segment lengths
    left: str                 # "pinned" | "clamped" | "free" | "periodic"
    right: str
    periodic: bool
    length: float
    k_max: int                # modes this segment resolves

    @property
    def ends_label(self) -> str:
        return "periodic" if self.periodic else f"{self.left}-{self.right}"


def _chain_segments(ci, offset, n, seg, closed, pin_type) -> List[Segment]:
    """pin_type: {local index: 'pinned' | 'clamped'}."""
    out = []

    def add(a, b, left, right):
        # local nodes a..b (b may be n == wrap to node 0 on a closed chain)
        loc = np.arange(a, b + 1)
        h = seg[a:b]
        n_free = len(loc) - (left != "free") - (right != "free")
        if n_free < 1:
            return
        out.append(Segment(ci, offset + (loc % n), h, left, right, False, float(h.sum()), 0))

    pins = sorted(pin_type)
    if closed:
        if not pins:
            out.append(Segment(ci, offset + np.arange(n), seg.copy(), "periodic", "periodic",
                               True, float(seg.sum()), 0))
            return out
        assert pins[0] == 0, "closed chain must be rotated to start at a pinned node"
        for j, a in enumerate(pins):
            b = pins[j + 1] if j + 1 < len(pins) else n
            add(a, b, pin_type[a], pin_type[b % n])
        return out
    bps = sorted(set(pins) | {0, n - 1})
    for a, b in zip(bps[:-1], bps[1:]):
        add(a, b, pin_type.get(a, "free"), pin_type.get(b, "free"))
    return out


def _k_resolved(s: Segment) -> int:
    k = int(np.floor(s.length / (4.0 * float(s.h.max()))))
    if s.periodic:
        n_dof = len(s.h) - 1
    else:
        n_dof = len(s.h) + 1 - (s.left != "free") - (s.right != "free")
        if s.left == "free" and s.right == "free":
            n_dof -= 1
    return max(0, min(k, n_dof))


# ---------------------------------------------------------------------------
# Basis object
# ---------------------------------------------------------------------------
@dataclass
class EdgeModes2D:
    t_edges: List[int]
    gids: np.ndarray          # (n,) ordered T node ids (all chains, concatenated)
    xy: np.ndarray            # (n, 2) baseline coordinates
    s: np.ndarray             # (n,) arc length from the start of each node's chain
    length: float             # total length of all T chains
    closed: bool              # True if every chain is a closed loop
    ends: str                 # end condition of open chains: fixed | free | clamped
    normals: np.ndarray       # (n, 2) unit, into the fluid (zero at a pinned cusp)
    lam: np.ndarray           # (k,) kappa^2, kappa = wavenumber of the mode
    phi: np.ndarray           # (n, k), integral(phi_k^2 ds) = L_segment/2
    # --- layout -----------------------------------------------------------
    corner_deg: float = 30.0
    min_radius_frac: float = 1e-3
    pin_junctions: bool = False
    fixed_nodes: List[int] = field(default_factory=list)   # node ids fixed by the user
    fixed_type: str = "clamped"               # clamped (smooth) | pinned (hinge)
    pinned: Optional[np.ndarray] = None       # (n,) bool, phi = 0 there
    pin_reason: Dict[int, str] = field(default_factory=dict)   # position -> why
    turn_deg: Optional[np.ndarray] = None     # (n,) signed turning angle
    radius: Optional[np.ndarray] = None       # (n,) discrete radius of curvature
    chain_of: Optional[np.ndarray] = None     # (n,) chain index
    chains: List[dict] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    mode_segment: Optional[np.ndarray] = None  # (k,) index into segments
    mode_number: Optional[np.ndarray] = None   # (k,) 1-based mode number within its segment
    k_max: int = 0                             # total number of modes resolved
    notes: List[str] = field(default_factory=list)

    @property
    def n_modes(self) -> int:
        return int(self.phi.shape[1])

    # ------------------------------------------------------------ build
    @classmethod
    def from_mesh(cls, mesh, t_edges: Iterable[int], n_modes: int = 8, ends: str = "fixed",
                  corner_deg: float = 30.0, min_radius_frac: float = 1e-3,
                  pin_junctions: bool = False, fixed_nodes: Iterable[int] = (),
                  fixed_type: str = "clamped") -> "EdgeModes2D":
        """
        n_modes = 0 returns the layout only (chains, pinned nodes, segments,
        k_max), which is what the GUI needs before the user picks n_modes.

        fixed_nodes : mesh node ids on T that must not move (user choice).
        fixed_type  : how user-fixed points and T-edge junctions are held:
                      "clamped" = position and slope (smooth, default),
                      "pinned"  = position only (hinge).
        """
        t_edges = [int(e) for e in t_edges]
        if ends not in END_TYPES:
            raise ValueError(f"ends must be one of {END_TYPES}, got {ends!r}")
        corner_deg = float(corner_deg)
        min_radius_frac = float(min_radius_frac)
        if corner_deg <= 0 or min_radius_frac < 0:
            raise ValueError("corner_deg must be > 0 and min_radius_frac >= 0")
        if fixed_type not in ("clamped", "pinned"):
            raise ValueError(f"fixed_type must be 'clamped' or 'pinned', got {fixed_type!r}")
        fixed_nodes = [int(g) for g in dict.fromkeys(int(x) for x in fixed_nodes)]
        fixed_set, fixed_seen = set(fixed_nodes), set()
        held = "position and slope" if fixed_type == "clamped" else "position only: hinge"

        junction_nodes = set()
        for e in t_edges:
            nd = mesh.edges[e].nodes
            junction_nodes |= {int(nd[0]), int(nd[-1])}

        G, XY, S, NRM, TH, RAD, PIN, CH = [], [], [], [], [], [], [], []
        reasons: Dict[int, str] = {}
        chains_meta, segments, notes = [], [], []
        offset = 0
        for ci, (g, closed) in enumerate(split_chains(mesh, t_edges)):
            xy = np.asarray(mesh.nodes[g, :2], float)
            n = len(g)
            if n < 3:
                raise ValueError(f"T chain {ci + 1} has only {n} nodes.")
            _, seg = arc_length(xy, closed)
            if np.any(seg <= 0):
                raise ValueError("Non-positive segment length on T (duplicate nodes).")
            Lc = float(seg.sum())
            theta = turning_angles(xy, closed)
            R = fold_radius(seg, theta, closed)
            # a corner turns by more than corner_deg and is not part of a coarse
            # smooth arc; beyond 2 * corner_deg at a single node it is a corner
            # whatever its neighbours do (nothing smooth is resolved there)
            is_corner = (np.abs(theta) > corner_deg) & (
                ~_is_arc_like(theta, seg, closed) | (np.abs(theta) > 2.0 * corner_deg))
            is_tight = R < min_radius_frac * Lc

            pin_type: Dict[int, str] = {}
            why: Dict[int, str] = {}
            for i in np.flatnonzero(is_corner | is_tight):
                i = int(i)
                pin_type[i] = "pinned"
                why[i] = (f"corner, turning {abs(theta[i]):.1f} deg" if is_corner[i] else
                          f"radius {R[i]:.3g} = {R[i] / Lc:.1e} of chain length "
                          f"(turning {abs(theta[i]):.1f} deg)")
            for i in range(n):
                gi = int(g[i])
                if gi in fixed_set:
                    fixed_seen.add(gi)
                    if i in pin_type:
                        why[i] += "; also fixed by user"
                    else:
                        pin_type[i] = fixed_type
                        why[i] = f"fixed by user ({held})"
                elif (pin_junctions and gi in junction_nodes and i not in pin_type
                      and (closed or 0 < i < n - 1)):
                    pin_type[i] = fixed_type
                    why[i] = f"junction between T edges ({held})"
            if not closed and ends != "free":
                for i in (0, n - 1):
                    pin_type[i] = "clamped" if ends == "clamped" else "pinned"
                    why[i] = f"chain end ({ends})"

            if closed and pin_type:                   # start the loop at a pinned node
                r = min(pin_type)
                if r:
                    g, xy, theta, R = (np.roll(a, -r, axis=0) for a in (g, xy, theta, R))
                    seg = np.roll(seg, -r)
                    pin_type = {(i - r) % n: t for i, t in pin_type.items()}
                    why = {(i - r) % n: t for i, t in why.items()}

            nrm = chain_normals(xy, closed, strict=False)
            degenerate = np.flatnonzero(np.linalg.norm(nrm, axis=1) < 0.5)
            for i in degenerate:
                if int(i) not in pin_type:
                    raise ValueError(
                        f"Zero-length normal at T node {int(g[i])} ({xy[i, 0]:.6g}, {xy[i, 1]:.6g}): "
                        "the curve doubles back there but the node is not pinned. "
                        "Lower the corner angle so it is treated as a corner.")
            s_loc = np.concatenate([[0.0], np.cumsum(seg[: n - 1])])
            segs = _chain_segments(ci, offset, n, seg, closed, pin_type)

            for i, t in why.items():
                reasons[offset + i] = t
            pinned = np.zeros(n, bool)
            pinned[list(pin_type)] = True
            free_R = np.where(pinned, np.inf, R)
            if np.isfinite(free_R).any():
                j = int(np.argmin(free_R))
                if free_R[j] < 0.05 * Lc:
                    notes.append(
                        f"chain {ci + 1}: tightest free curvature R = {free_R[j]:.3g} "
                        f"({free_R[j] / Lc:.1e} of chain length) at ({xy[j, 0]:.5g}, {xy[j, 1]:.5g}), "
                        f"turning {abs(theta[j]):.1f} deg per node. Inward normal displacement there "
                        f"folds the curve at about that amplitude.")
            chains_meta.append({"closed": bool(closed), "start": offset, "stop": offset + n,
                                "length": Lc, "n_nodes": n, "n_pinned": int(pinned.sum())})
            segments.extend(segs)
            G.append(g); XY.append(xy); S.append(s_loc); NRM.append(nrm)
            TH.append(theta); RAD.append(R); PIN.append(pinned); CH.append(np.full(n, ci))
            offset += n

        gids = np.concatenate(G)
        if len(set(gids.tolist())) != len(gids):
            raise ValueError("A node appears in two T chains.")
        missing = sorted(fixed_set - fixed_seen)
        if missing:
            raise ValueError(f"Fixed point(s) {missing} are not nodes of the T edges {t_edges}.")
        for sg in segments:
            sg.k_max = _k_resolved(sg)
        if segments and sum(sg.k_max for sg in segments) == 0:
            max(segments, key=lambda q: q.length).k_max = 1      # as before: at least one mode
        k_max = int(sum(sg.k_max for sg in segments))
        n_fixed = sum(1 for sg in segments if sg.k_max == 0)
        if n_fixed:
            notes.append(f"{n_fixed} segment(s) too short or too coarse to resolve a mode stay fixed.")

        N = len(gids)
        n_modes = int(n_modes)
        if n_modes < 0:
            raise ValueError("n_modes must be >= 0")
        if n_modes > k_max:
            raise ValueError(
                f"{n_modes} modes requested but the T edges only resolve {k_max} "
                "(8 segments per wavelength on every smooth segment). "
                "Refine the T edge or use fewer modes.")

        phi = np.zeros((N, n_modes))
        lam = np.zeros(n_modes)
        mseg = np.zeros(n_modes, np.int64)
        mnum = np.zeros(n_modes, np.int64)
        if n_modes:
            cand = []                                   # (kappa, segment, local mode, column)
            for si, sg in enumerate(segments):
                kk = min(sg.k_max, n_modes)
                if kk == 0:
                    continue
                kap, ph = segment_modes(sg.h, kk, sg.left, sg.right, sg.periodic)
                for j in range(ph.shape[1]):
                    cand.append((float(kap[j]), si, j, ph[:, j]))
            # rank by wavenumber; near-ties (equal-length segments) by segment order,
            # so the ranking does not depend on round-off
            cand.sort(key=lambda c: c[0])
            ranked, i = [], 0
            while i < len(cand):
                j = i + 1
                while j < len(cand) and cand[j][0] - cand[i][0] <= _TIE_RTOL * max(cand[j][0], 1e-300):
                    j += 1
                ranked.extend(sorted(cand[i:j], key=lambda c: (c[1], c[2])))
                i = j
            for k, (kap, si, j, col) in enumerate(ranked[:n_modes]):
                sg = segments[si]
                if sg.periodic:
                    phi[sg.idx, k] = col
                else:
                    # a closed chain with one pinned node lists that node at both ends
                    phi[sg.idx[1:-1], k] = col[1:-1]
                    for e in (0, -1):
                        if col[e] != 0.0:               # free end
                            phi[sg.idx[e], k] = col[e]
                lam[k], mseg[k], mnum[k] = kap ** 2, si, j + 1
            for si, sg in enumerate(segments):
                if sg.periodic and int((mseg == si).sum()) % 2:
                    notes.append(f"segment {si + 1} is a smooth closed loop: its modes come in "
                                 "(cos, sin) pairs and this selection keeps half of one pair, which "
                                 "favours one phase. Use one mode more or fewer.")

        return cls(t_edges=t_edges, gids=gids, xy=np.vstack(XY), s=np.concatenate(S),
                   length=float(sum(c["length"] for c in chains_meta)),
                   closed=all(c["closed"] for c in chains_meta), ends=ends,
                   normals=np.vstack(NRM), lam=lam, phi=phi,
                   corner_deg=corner_deg, min_radius_frac=min_radius_frac,
                   pin_junctions=bool(pin_junctions), fixed_nodes=fixed_nodes,
                   fixed_type=fixed_type, pinned=np.concatenate(PIN),
                   pin_reason=reasons, turn_deg=np.concatenate(TH), radius=np.concatenate(RAD),
                   chain_of=np.concatenate(CH), chains=chains_meta, segments=segments,
                   mode_segment=mseg, mode_number=mnum, k_max=k_max, notes=notes)

    # ------------------------------------------------------------ use
    def displacement(self, coeffs) -> np.ndarray:
        """Design vector -> (n, 3) displacement of the T nodes (z = 0)."""
        a = np.asarray(coeffs, float).ravel()
        if a.size != self.n_modes:
            raise ValueError(f"expected {self.n_modes} coefficients, got {a.size}")
        amp = self.phi @ a
        d = np.zeros((len(self.gids), 3))
        d[:, :2] = amp[:, None] * self.normals
        return d

    def mode_label(self, k: int) -> str:
        """Short description of mode k (0-based), e.g. 'segment 1, n=3'."""
        if self.mode_segment is None or not len(self.segments):
            return ""
        si = int(self.mode_segment[k])
        sg = self.segments[si]
        return f"segment {si + 1} ({sg.ends_label}), n={int(self.mode_number[k])}"

    def describe(self) -> List[str]:
        """Human-readable layout: chains, pinned nodes, segments, warnings."""
        out = [f"T: {len(self.chains)} chain(s), {len(self.gids)} nodes, total length {self.length:.4g}; "
               f"{len(self.segments)} smooth segment(s), up to {self.k_max} modes resolved."]
        for ci, c in enumerate(self.chains):
            out.append(f"  chain {ci + 1}: {'closed' if c['closed'] else 'open'}, {c['n_nodes']} nodes, "
                       f"length {c['length']:.4g}, {c['n_pinned']} pinned node(s)")
        for pos in sorted(self.pin_reason):
            out.append(f"    pinned ({self.xy[pos, 0]:.5g}, {self.xy[pos, 1]:.5g}): {self.pin_reason[pos]}")
        for si, sg in enumerate(self.segments):
            a, b = self.xy[sg.idx[0]], self.xy[sg.idx[-1]]
            where = ("whole loop" if sg.periodic else
                     f"({a[0]:.5g}, {a[1]:.5g}) -> ({b[0]:.5g}, {b[1]:.5g})")
            out.append(f"  segment {si + 1}: chain {sg.chain + 1}, {where}, length {sg.length:.4g}, "
                       f"{sg.ends_label}, resolves {sg.k_max} mode(s)"
                       + ("" if sg.k_max else " -> stays fixed"))
        out.extend(f"  note: {t}" for t in self.notes)
        return out

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "version": BASIS_VERSION,
            "dim": 2, "kind": "edge_modes_segmented",
            "t_edges": self.t_edges, "ends": self.ends, "closed": self.closed,
            "corner_deg": self.corner_deg, "min_radius_frac": self.min_radius_frac,
            "pin_junctions": self.pin_junctions,
            "fixed_nodes": [int(g) for g in self.fixed_nodes], "fixed_type": self.fixed_type,
            "length": self.length, "gids": self.gids.tolist(),
            "lam": self.lam.tolist(), "n_modes": self.n_modes,
            "chains": self.chains,
            "pinned_gids": self.gids[self.pinned].tolist(),
            "pin_reason": {str(int(self.gids[p])): r for p, r in self.pin_reason.items()},
            "segments": [{"chain": sg.chain, "ends": sg.ends_label, "length": sg.length,
                          "n_nodes": int(len(sg.idx)), "k_max": sg.k_max,
                          "first_gid": int(self.gids[sg.idx[0]]), "last_gid": int(self.gids[sg.idx[-1]])}
                         for sg in self.segments],
            "mode_segment": self.mode_segment.tolist(),
            "mode_number": self.mode_number.tolist(),
            # the basis itself, so a design vector is machine-independent
            "xy": self.xy.tolist(), "normals": self.normals.tolist(), "phi": self.phi.tolist(),
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_saved(cls, mesh, path_or_dict, xy_rtol: float = 1e-9) -> "EdgeModes2D":
        """
        Rebuild from a saved basis on the same baseline mesh.

        The layout is recomputed from the saved settings and checked against
        the saved node ordering and coordinates; phi and the normals are then
        TAKEN FROM THE FILE, so the shapes are identical to the ones the
        bounds were computed for.
        """
        d = path_or_dict
        if isinstance(path_or_dict, str):
            with open(path_or_dict) as f:
                d = json.load(f)
        ends = "fixed" if d.get("ends") == "periodic" else d.get("ends", "fixed")   # v1 files
        b = cls.from_mesh(mesh, d["t_edges"], int(d["n_modes"]), ends=ends,
                          corner_deg=float(d.get("corner_deg", 30.0)),
                          min_radius_frac=float(d.get("min_radius_frac", 1e-3)),
                          pin_junctions=bool(d.get("pin_junctions", False)),
                          fixed_nodes=d.get("fixed_nodes", []),
                          # files written before fixed_type existed pinned their junctions
                          fixed_type=d.get("fixed_type", "pinned"))
        if b.gids.tolist() != list(d["gids"]):
            old = "" if "version" in d else (" The file was written by an older version of "
                                             "edgeModes2D; re-save the 2D basis from the GUI.")
            raise ValueError("Saved basis node ordering does not match this mesh "
                             "(different baseline, edge labels or pinning rules)." + old)
        if "phi" not in d:
            # v1 file: only the eigenvalues were stored. If they no longer match,
            # the basis definition has changed (v1 did not pin corners) and the
            # saved coefficients/bounds would silently mean different shapes.
            lam_old = np.asarray(d.get("lam", []), float)
            if lam_old.shape != b.lam.shape or not np.allclose(lam_old, b.lam, rtol=1e-6, atol=0.0):
                raise ValueError(
                    "Saved basis was written by an older version of edgeModes2D and its modes are "
                    "not the ones built now (corners are pinned since version 2). Designs and "
                    "bounds saved with it do not carry over: re-save the 2D basis from the GUI.")
            b.notes.append("saved basis has no stored modes (older file): modes were recomputed.")
            return b
        xy = np.asarray(d["xy"], float)
        size = float(np.linalg.norm(np.ptp(b.xy, axis=0)))
        if xy.shape != b.xy.shape or np.abs(xy - b.xy).max() > xy_rtol * max(size, 1e-300):
            raise ValueError("Saved basis coordinates do not match this mesh: "
                             "it was built on a different baseline.")
        phi = np.asarray(d["phi"], float).reshape(len(b.gids), -1)
        if phi.shape != b.phi.shape:
            raise ValueError(f"Saved phi has shape {phi.shape}, expected {b.phi.shape}.")
        diff = float(np.abs(phi - b.phi).max()) if phi.size else 0.0
        if diff > 1e-6:
            b.notes.append(f"modes recomputed on this machine differ from the saved ones by up to "
                           f"{diff:.2e}; the saved ones are used.")
        b.phi = phi
        b.normals = np.asarray(d["normals"], float)
        b.lam = np.asarray(d["lam"], float)
        return b