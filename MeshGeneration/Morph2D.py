"""
Morph2D.py

Morph a 2D domain mesh (EdgeMesh2D) directly; no remeshing.

  1. T nodes move by the edge Laplacian modes (edgeModes2D), exactly.
  2. All other boundary nodes are FIXED, except U-edge nodes, which are free.
  3. The displacement is carried into the interior (and onto U) with an
     exact-interpolation compact Wendland C2 RBF whose sources are
     T (prescribed) + fixed boundary nodes (zero).

Why not MorphModel.transformT
-----------------------------
transformT uses anchors only to set its support radius; it does not hold the
fixed boundary in place (see the note in MorphMesh). For a volume/domain
morph every fixed boundary node must stay put, otherwise walls, outlets and
the far field drift. Here they are interpolation sources with zero
displacement, so they stay fixed to solver precision. We reuse the same
family of kernel, not the routine.

U edges
-------
U nodes are free: they follow the interpolated field and may move off their
original line. That is fine for "passively deforming" walls but would be
wrong for e.g. a straight outlet that should stay straight. A sliding option
can be added when we have a case that needs it.

Cost
----
Everything that does not depend on the design vector (mode basis, RBF
factorisation, source -> target matrix) is built once in __init__, so a
morph for a new design is a sparse triangular solve plus a sparse mat-vec.

Quality gate
------------
After each morph: no triangle may invert, and the smallest area ratio
A_new/A_old must stay above `min_area_ratio`. A failed design should be
reported as failed to the optimiser instead of being sent to the solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from ShapeParameterization.edgeModes2D import EdgeModes2D


def wendland_c2(r):
    """phi(r) = (1-r)^4 (4r+1) for r<1, else 0. Positive definite in up to 3D."""
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    m = r < 1.0
    rm = r[m]
    out[m] = (1.0 - rm) ** 4 * (4.0 * rm + 1.0)
    return out


def _segments_intersect(A, B, C, D):
    """Proper intersection of segments AB (n,2) with CD (m,2): (n,m) bool."""
    def orient(P, Q, R):
        return ((Q[..., 0] - P[..., 0]) * (R[..., 1] - P[..., 1])
                - (Q[..., 1] - P[..., 1]) * (R[..., 0] - P[..., 0]))
    A, B = A[:, None, :], B[:, None, :]
    C, D = C[None, :, :], D[None, :, :]
    o1, o2 = orient(A, B, C), orient(A, B, D)
    o3, o4 = orient(C, D, A), orient(C, D, B)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


@dataclass
class MorphReport:
    ok: bool
    n_inverted: int
    min_area_ratio: float
    max_disp: float
    max_fixed_disp: float
    max_abs_dz: float
    n_boundary_crossings: int = 0
    reason: str = ""

    def __str__(self):
        flag = "OK" if self.ok else f"FAILED ({self.reason})"
        return (f"[Morph2D] {flag}: inverted={self.n_inverted}, "
                f"boundary crossings={self.n_boundary_crossings}, min A_new/A_old="
                f"{self.min_area_ratio:.4f}, max|d|={self.max_disp:.4g}, "
                f"max|d| on fixed boundary={self.max_fixed_disp:.2e}, max|dz|={self.max_abs_dz:.1e}")


class Morph2D:
    def __init__(self, mesh, t_edges: Iterable[int], u_edges: Iterable[int] = (),
                 n_modes: int = 8, ends: str = "fixed",
                 support_radius: Optional[float] = None, support_frac: float = 1.0,
                 rbf_lambda: float = 1e-12, min_area_ratio: float = 0.05):
        """
        support_radius : RBF support radius in mesh units. Default
                         support_frac * (length of the T chain).
        min_area_ratio : quality gate on min(A_new / A_old).
        """
        self.mesh = mesh
        self.t_edges = [int(e) for e in t_edges]
        self.u_edges = [int(e) for e in u_edges]
        overlap = set(self.t_edges) & set(self.u_edges)
        if overlap:
            raise ValueError(f"Edges {sorted(overlap)} are both T and U.")
        unknown = (set(self.t_edges) | set(self.u_edges)) - set(mesh.edges)
        if unknown:
            raise ValueError(f"Unknown edge ids {sorted(unknown)}; mesh has {sorted(mesh.edges)}.")

        self.modes = EdgeModes2D.from_mesh(mesh, self.t_edges, n_modes=n_modes, ends=ends)
        self.min_area_ratio = float(min_area_ratio)

        N = mesh.node_count
        t_ids = self.modes.gids
        u_ids = set()
        for e in self.u_edges:
            u_ids |= set(int(x) for x in mesh.edges[e].nodes)
        u_ids -= set(int(x) for x in t_ids)           # shared end nodes belong to T
        bnd = set(int(x) for x in mesh.boundary_node_ids)
        fixed_ids = np.array(sorted(bnd - set(int(x) for x in t_ids) - u_ids), np.int64)

        self.t_ids = t_ids
        self.u_ids = np.array(sorted(u_ids), np.int64)
        self.fixed_ids = fixed_ids
        self.src_ids = np.concatenate([t_ids, fixed_ids])
        is_src = np.zeros(N, bool)
        is_src[self.src_ids] = True
        self.tgt_ids = np.flatnonzero(~is_src)

        self.R = float(support_radius) if support_radius else float(support_frac) * self.modes.length
        if self.R <= 0:
            raise ValueError("support radius must be > 0")

        P = mesh.nodes[:, :2]
        S = P[self.src_ids]
        tree_s = cKDTree(S)

        # source-source system  Phi W = d   (sparse, SPD for Wendland C2 in 2D)
        D = tree_s.sparse_distance_matrix(tree_s, self.R, output_type="coo_matrix")
        # NOTE: the coo output of sparse_distance_matrix INCLUDES the i == i
        # pairs (distance 0 -> phi = 1), so only the regularisation is added.
        A = sp.coo_matrix((wendland_c2(D.data / self.R), (D.row, D.col)), shape=(len(S), len(S)))
        A = (A + sp.identity(len(S)) * rbf_lambda).tocsc()
        diag = A.diagonal()
        if not np.allclose(diag, 1.0 + rbf_lambda):
            raise RuntimeError("RBF matrix diagonal is not phi(0); duplicate boundary nodes?")
        self._lu = splu(A)

        # only targets within R of a source can move
        tree_t = cKDTree(P[self.tgt_ids])
        B = tree_t.sparse_distance_matrix(tree_s, self.R, output_type="coo_matrix")
        self._B = sp.csr_matrix((wendland_c2(B.data / self.R), (B.row, B.col)),
                                shape=(len(self.tgt_ids), len(S)))

        self._A0 = mesh.triangle_areas()

    # ------------------------------------------------------------------
    @property
    def n_modes(self) -> int:
        return self.modes.n_modes

    def displacement_field(self, coeffs) -> np.ndarray:
        """(N, 3) displacement of every mesh node for design `coeffs`."""
        dT = self.modes.displacement(coeffs)[:, :2]
        rhs = np.zeros((len(self.src_ids), 2))
        rhs[: len(self.t_ids)] = dT
        W = np.column_stack([self._lu.solve(rhs[:, 0]), self._lu.solve(rhs[:, 1])])
        d = np.zeros((self.mesh.node_count, 3))
        d[self.tgt_ids, :2] = self._B @ W
        d[self.t_ids, :2] = dT            # exact on T
        # fixed boundary: exactly zero by construction (sources with zero data)
        return d

    def check(self, new_nodes, d) -> MorphReport:
        A1 = self.mesh.triangle_areas(new_nodes)
        ratio = A1 / self._A0
        n_inv = int((A1 <= 0.0).sum())
        rmin = float(ratio.min())
        mag = np.linalg.norm(d, axis=1)
        fixed_max = float(mag[self.fixed_ids].max()) if len(self.fixed_ids) else 0.0
        dz = float(np.abs(new_nodes[:, 2] - self.mesh.nodes[:, 2]).max())
        n_cross = self._boundary_crossings(new_nodes)
        reason = ""
        if n_inv:
            reason = f"{n_inv} inverted triangles"
        elif n_cross:
            # a wall can pass through an unmeshed hole (e.g. the body interior)
            # without inverting any triangle, so this is checked separately
            reason = f"moving boundary crosses another boundary ({n_cross} segment pairs)"
        elif rmin < self.min_area_ratio:
            reason = f"min area ratio {rmin:.3g} < {self.min_area_ratio}"
        elif dz > 0.0:
            reason = "out-of-plane displacement"
        return MorphReport(ok=not reason, n_inverted=n_inv, min_area_ratio=rmin,
                           max_disp=float(mag.max()), max_fixed_disp=fixed_max,
                           max_abs_dz=dz, n_boundary_crossings=n_cross, reason=reason)

    def _boundary_crossings(self, new_nodes) -> int:
        """Segments of the moving boundary (T, U) crossing any boundary segment."""
        if not hasattr(self, "_bseg"):
            segs = [np.column_stack([L, np.roll(L, -1)]) for L in self.mesh.loops]
            self._bseg = np.vstack(segs)
            moving = np.zeros(self.mesh.node_count, bool)
            moving[self.t_ids] = True
            moving[self.u_ids] = True
            self._mseg = self._bseg[moving[self._bseg].any(axis=1)]
        P = new_nodes[:, :2]
        a, b = self._mseg[:, 0], self._mseg[:, 1]
        c, d = self._bseg[:, 0], self._bseg[:, 1]
        hit = _segments_intersect(P[a], P[b], P[c], P[d])
        # segments sharing a node touch at that node; not a crossing
        share = ((a[:, None] == c[None]) | (a[:, None] == d[None]) |
                 (b[:, None] == c[None]) | (b[:, None] == d[None]))
        return int((hit & ~share).sum())

    def morph(self, coeffs):
        """Returns (morphed EdgeMesh2D, MorphReport). Always check report.ok."""
        d = self.displacement_field(coeffs)
        new_nodes = self.mesh.nodes + d
        rep = self.check(new_nodes, d)
        return self.mesh.with_nodes(new_nodes), rep

    def max_safe_amplitude(self, k: int, a_max: Optional[float] = None, tol: float = 1e-3):
        """
        Largest |a_k| (single mode k, others zero, both signs) that passes the
        quality gate. Bisection; returns (a_minus, a_plus) with a_minus <= 0.
        Useful for setting honest BO bounds per mode.
        """
        if a_max is None:
            a_max = self.modes.length
        out = []
        for sgn in (-1.0, 1.0):
            lo, hi = 0.0, a_max
            x = np.zeros(self.n_modes)
            x[k] = sgn * hi
            if self.check(*self._nd(x)).ok:
                out.append(sgn * hi)
                continue
            while hi - lo > tol * a_max:
                mid = 0.5 * (lo + hi)
                x[k] = sgn * mid
                if self.check(*self._nd(x)).ok:
                    lo = mid
                else:
                    hi = mid
            out.append(sgn * lo)
        return out[0], out[1]

    def _nd(self, x):
        d = self.displacement_field(x)
        return self.mesh.nodes + d, d