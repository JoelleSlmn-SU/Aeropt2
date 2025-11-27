# cad_ffd.py

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional

import numpy as np

from OCP.TopoDS import TopoDS_Shape, TopoDS_Face
from OCP.Geom import Geom_BSplineSurface
from OCP.gp import gp_Pnt


# ---------------------------------------------------------------------------
# Basic types / enums
# ---------------------------------------------------------------------------

class FaceRole(Enum):
    T = auto()  # design region
    C = auto()  # blending / buffer
    U = auto()  # unaffected / propagated region


@dataclass
class FaceInfo:
    """Store role and BSpline geometry for a CAD face."""
    face: TopoDS_Face
    role: FaceRole
    bspline: Geom_BSplineSurface
    # Optional: adjacency info, indices, etc.
    # e.g. face_index: int


@dataclass
class EmbeddedPole:
    """
    One BSpline pole embedded in the FFD lattice.

    We store enough info to:
      - find it again (which face, which i,j)
      - evaluate its FFD mapping (ξ,η,ζ)
    """
    face_id: int           # index or key back into a face map
    i: int
    j: int
    xi: float
    eta: float
    zeta: float


# ---------------------------------------------------------------------------
# FFD lattice (tri-variate B-spline volume)
# ---------------------------------------------------------------------------

class FFDVolume:
    """
    Tensor-product B-spline volume acting as an FFD lattice.

    - Stores control points P_ijk in physical space.
    - Knots and degrees in xi, eta, zeta.
    - Provides mapping:
        world -> (xi, eta, zeta) via a simple affine
        (xi, eta, zeta) -> deformed X' using deformed lattice CPs.
    """

    def __init__(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        n_ctrl: Tuple[int, int, int] = (4, 4, 3),
        degree: Tuple[int, int, int] = (3, 3, 2),
    ) -> None:
        # Store lattice definition
        self.bbox_min = np.asarray(bbox_min, dtype=float)
        self.bbox_max = np.asarray(bbox_max, dtype=float)
        self.n_ctrl = n_ctrl       # (n_xi, n_eta, n_eta)
        self.degree = degree       # (n_xi, n_eta, n_eta)

        # Lattice control points in physical space: (n_xi, n_eta, n_eta, 3)
        self.ctrl_pts = self._init_regular_grid()

        # Optional: knots, precomputed basis values, etc.
        self.knots_xi: np.ndarray = self._init_uniform_knots(n_ctrl[0], degree[0])
        self.knots_eta: np.ndarray = self._init_uniform_knots(n_ctrl[1], degree[1])
        self.knots_zeta: np.ndarray = self._init_uniform_knots(n_ctrl[2], degree[2])

        # A second copy for "original" CPs if you want to reset easily
        self.ctrl_pts_ref = self.ctrl_pts.copy()

    def _init_regular_grid(self) -> np.ndarray:
        """
        Create initial lattice CPs as a regular rectilinear grid over
        [bbox_min, bbox_max] (axis-aligned box).
        """
        n_xi, n_eta, n_zeta = self.n_ctrl

        xs = np.linspace(self.bbox_min[0], self.bbox_max[0], n_xi)
        ys = np.linspace(self.bbox_min[1], self.bbox_max[1], n_eta)
        zs = np.linspace(self.bbox_min[2], self.bbox_max[2], n_zeta)

        ctrl = np.zeros((n_xi, n_eta, n_zeta, 3), dtype=float)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    ctrl[i, j, k, :] = (x, y, z)

        return ctrl

    def _init_uniform_knots(self, n_ctrl: int, degree: int) -> np.ndarray:
        """
        Create a clamped uniform knot vector on [0,1] for a B-spline of
        given number of control points and degree.

        Length = n_ctrl + degree + 1
        - First (degree+1) knots = 0
        - Last  (degree+1) knots = 1
        - Interiors uniformly spaced in (0,1).
        """
        m = n_ctrl + degree + 1  # number of knots
        if n_ctrl <= degree:
            # Degenerate case: just all zeros/ones squashed
            return np.concatenate([
                np.zeros(degree + 1),
                np.ones(m - (degree + 1))
            ])

        knots = np.zeros(m, dtype=float)
        knots[-(degree + 1):] = 1.0

        n_int = n_ctrl - degree - 1  # number of internal distinct knots
        if n_int > 0:
            # Internal knots equally spaced between (0,1)
            internal = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
            knots[degree + 1:degree + 1 + n_int] = internal

        return knots

    # ---- Parameter mapping -------------------------------------------------

    def world_to_param(self, x: np.ndarray) -> np.ndarray:
        """
        Map world coordinates -> (ξ,η,ζ) in [0,1]^3 via simple affine
        map using axis-aligned bounding box.
        """
        x = np.asarray(x, dtype=float)
        denom = (self.bbox_max - self.bbox_min)
        denom[denom == 0.0] = 1.0  # avoid divide-by-zero if flat in a dir
        xi_eta_zeta = (x - self.bbox_min) / denom
        return xi_eta_zeta

    # ---- B-spline evaluation -----------------------------------------------

    def _basis_1d(self, t: float, knots: np.ndarray, degree: int) -> np.ndarray:
        """
        Evaluate 1D B-spline basis functions at parameter t for a
        clamped knot vector `knots` and degree.

        Returns:
            N : array of shape (n_ctrl,)
        """
        t = float(t)
        knots = np.asarray(knots, dtype=float)
        p = degree
        m = len(knots) - 1  # last index
        n_ctrl = m - p      # number of control points

        # Clamp t into param range [knots[p], knots[m-p]]
        t_min = knots[p]
        t_max = knots[m - p]
        if t < t_min:
            t = t_min
        elif t > t_max:
            t = t_max

        # Zeroth degree basis
        N = np.zeros(n_ctrl, dtype=float)
        for i in range(n_ctrl):
            if knots[i] <= t < knots[i + 1] or (t == knots[-1] and i == n_ctrl - 1):
                N[i] = 1.0

        # Cox–de Boor recursion
        for k in range(1, p + 1):
            N_prev = N.copy()
            N[:] = 0.0
            for i in range(n_ctrl):
                # left term
                denom_left = knots[i + k] - knots[i]
                if denom_left != 0.0:
                    a = (t - knots[i]) / denom_left * N_prev[i]
                else:
                    a = 0.0

                # right term
                denom_right = knots[i + k + 1] - knots[i + 1]
                if denom_right != 0.0 and i + 1 < n_ctrl:
                    b = (knots[i + k + 1] - t) / denom_right * N_prev[i + 1]
                else:
                    b = 0.0

                N[i] = a + b

        return N

    def evaluate_deformed(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        Evaluate deformed position X'(ξ,η,ζ) using current ctrl_pts.
        """
        n_xi, n_eta, n_zeta = self.n_ctrl

        B_xi = self._basis_1d(xi, self.knots_xi, self.degree[0])   # (n_xi,)
        B_eta = self._basis_1d(eta, self.knots_eta, self.degree[1])  # (n_eta,)
        B_zeta = self._basis_1d(zeta, self.knots_zeta, self.degree[2])  # (n_zeta,)

        # Triple sum: X = Σ_i Σ_j Σ_k B_xi[i] B_eta[j] B_zeta[k] * P_ijk
        # We can do this in two einsums to avoid explicit loops
        # First: combine x & y bases with ctrl_pts
        # shape: (3,) after full contraction
        tmp = np.einsum("i,ijkl->jkl", B_xi, self.ctrl_pts)      # (n_eta, n_zeta, 3)
        tmp = np.einsum("j,jkl->kl", B_eta, tmp)                 # (n_zeta, 3)
        X = np.einsum("k,kl->l", B_zeta, tmp)                    # (3,)

        return X

    # ---- Design variable interface -----------------------------------------

    def reset(self) -> None:
        """Reset lattice CPs back to reference configuration."""
        self.ctrl_pts[:] = self.ctrl_pts_ref

    def apply_displacements(self, dP: np.ndarray) -> None:
        """
        Apply a displacement array to ctrl_pts.

        Expected shapes:
          - dP.shape == (nξ, nη, nζ, 3)
          - or flattened with length nξ*nη*nζ*3
        """
        n_xi, n_eta, n_zeta = self.n_ctrl
        expected = (n_xi, n_eta, n_zeta, 3)
        if dP.shape == expected:
            self.ctrl_pts += dP
        elif dP.size == np.prod(expected):
            self.ctrl_pts += dP.reshape(expected)
        else:
            raise ValueError(
                f"dP has incompatible shape {dP.shape}, expected {expected} "
                "or a flat array of matching size."
            )


class CADFFDManager:
    """
    Orchestrates:
      - Collecting faces & roles (T/C/U)
      - Building FFDVolume around T region
      - Embedding poles of T faces
      - Applying lattice changes and (later) propagating to C and U.
    """

    def __init__(self, shape: TopoDS_Shape) -> None:
        self.shape = shape
        self.faces: Dict[int, FaceInfo] = {}
        self.ffd: Optional[FFDVolume] = None

        # Embedded poles on T faces (and later T–C edges)
        self.embedded_poles_T: List[EmbeddedPole] = []
        self.embedded_poles_TC_edges: List[EmbeddedPole] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def register_face(
        self,
        face_id: int,
        face: TopoDS_Face,
        role: FaceRole,
        bspline: Geom_BSplineSurface,
    ) -> None:
        """
        Called by higher-level code to give us faces and their roles
        plus NURBS geometry.
        """
        self.faces[face_id] = FaceInfo(
            face=face,
            role=role,
            bspline=bspline,
        )

    def build_ffd_from_T_faces(
        self,
        padding: float = 0.05,
        n_ctrl: Tuple[int, int, int] = (4, 4, 3),
        degree: Tuple[int, int, int] = (3, 3, 2),
    ) -> None:
        """
        Compute a bounding box around all T faces and create FFDVolume.
        """
        pts = []

        # Collect all BSpline poles on T faces
        for fid, finfo in self.faces.items():
            if finfo.role != FaceRole.T:
                continue

            surf = finfo.bspline
            try:
                n_u = surf.NbUPoles()
                n_v = surf.NbVPoles()
            except Exception:
                continue

            for i in range(1, n_u + 1):
                for j in range(1, n_v + 1):
                    pnt: gp_Pnt = surf.Pole(i, j)
                    pts.append([pnt.X(), pnt.Y(), pnt.Z()])

        if not pts:
            # Fallback: use a unit cube if no T poles (shouldn't happen in practice)
            bbox_min = np.array([0.0, 0.0, 0.0], dtype=float)
            bbox_max = np.array([1.0, 1.0, 1.0], dtype=float)
        else:
            pts = np.asarray(pts, dtype=float)
            bbox_min = pts.min(axis=0)
            bbox_max = pts.max(axis=0)

            # Expand bbox by padding fraction
            span = bbox_max - bbox_min
            bbox_min = bbox_min - padding * span
            bbox_max = bbox_max + padding * span

        self.ffd = FFDVolume(bbox_min, bbox_max, n_ctrl=n_ctrl, degree=degree)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_T_faces(self) -> None:
        """
        For each T face, embed all its poles into the FFD lattice.
        """
        if self.ffd is None:
            raise RuntimeError("FFDVolume not built yet.")

        self.embedded_poles_T.clear()

        for fid, finfo in self.faces.items():
            if finfo.role != FaceRole.T:
                continue

            surf = finfo.bspline
            try:
                n_u = surf.NbUPoles()
                n_v = surf.NbVPoles()
            except Exception:
                continue

            for i in range(1, n_u + 1):
                for j in range(1, n_v + 1):
                    pnt: gp_Pnt = surf.Pole(i, j)
                    xyz = np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=float)
                    xi, eta, zeta = self.ffd.world_to_param(xyz)
                    ep = EmbeddedPole(
                        face_id=fid,
                        i=i,
                        j=j,
                        xi=float(xi),
                        eta=float(eta),
                        zeta=float(zeta),
                    )
                    self.embedded_poles_T.append(ep)

    def embed_TC_shared_edges(self) -> None:
        """
        Placeholder for later: find edges shared between T and C faces
        and embed those boundary poles.
        """
        # We will fill this once adjacency logic is wired in.
        pass

    # ------------------------------------------------------------------
    # Applying a design
    # ------------------------------------------------------------------

    def apply_design(
        self,
        dP_lattice: np.ndarray,
        propagate_C: bool = True,
        propagate_U: bool = True,
    ) -> None:
        """
        Top-level call for one design:
          - apply displacements to FFD lattice CPs
          - update T poles using FFD mapping
          - (later) enforce T–C edge consistency + C/U propagation
        """
        if self.ffd is None:
            raise RuntimeError("FFDVolume not built yet.")

        # 1) deform lattice
        self.ffd.reset()
        self.ffd.apply_displacements(dP_lattice)

        # 2) update T poles
        self._update_T_poles()

        # 3) boundary recovery T -> C edges + C/U propagation
        #    (to be implemented later)
        # if propagate_C:
        #     self._recover_TC_boundaries()
        #     self._smooth_C_interior()
        #
        # if propagate_U:
        #     self._propagate_to_U()

    def _update_T_poles(self) -> None:
        """Move T-face poles using FFD mapping at embedded (ξ,η,ζ)."""
        if self.ffd is None:
            raise RuntimeError("FFDVolume not built yet.")

        for ep in self.embedded_poles_T:
            finfo = self.faces.get(ep.face_id)
            if finfo is None:
                continue

            surf = finfo.bspline
            X = self.ffd.evaluate_deformed(ep.xi, ep.eta, ep.zeta)
            p_new = gp_Pnt(float(X[0]), float(X[1]), float(X[2]))

            # Geom_BSplineSurface is 1-based indexing
            try:
                surf.SetPole(ep.i, ep.j, p_new)
            except Exception as e:
                print(f"[CAD-FFD] Failed to set pole ({ep.i},{ep.j}) on face {ep.face_id}: {e}")

    # ------------------------------------------------------------------
    # Export / retrieval
    # ------------------------------------------------------------------

    def get_deformed_shape(self) -> TopoDS_Shape:
        """
        Return the deformed TopoDS_Shape.

        In this pattern we assume that mutating the Geom_BSplineSurface
        poles mutates the underlying TopoDS_Shape geometry in place.
        """
        return self.shape