#!/usr/bin/env python3
"""
Python replacement for the legacy interpu.f workflow.

Purpose
-------
Given:
  1. baseline volume mesh: <old_base>.plt
  2. baseline global solution: <old_base>.unk
  3. morphed/new volume mesh: <new_base>.plt

produce:
  <new_base>.unk

The logic mirrors interpu.f at a high level:
  - read old/new .plt files
  - read old .unk file
  - for each new node, interpolate the old .unk field from the old volume mesh
  - if no containing old tetrahedron is found, use the nearest old volume node
  - optionally overwrite boundary nodes by direct old->new boundary transfer

Notes
-----
Fortran unformatted sequential files are read/written using scipy.io.FortranFile.
The .unk layout follows interpu.f:
  record 1: integer numberOfNodes
  record 2: variables ordered as var-major, i.e. u[var, node]

The .plt layout assumed here follows interpu.f:
  record 1: nelem, npoin, nboun
  record 2: intma(4, nelem) integer tetra connectivity, 1-based
  record 3: coord(3, npoin) real*8 coordinates
  record 4: ibsid(5, nboun) integer boundary information
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import argparse
import numpy as np
from scipy.io import FortranFile
from scipy.spatial import cKDTree

BoundaryMode = Literal["none", "same_id", "ordered", "nearest_old_boundary", "auto"]


@dataclass
class PLTMesh:
    path: Path
    nelem: int
    npoin: int
    nboun: int
    # zero-based tet connectivity, shape (nelem, 4)
    tets: np.ndarray
    # coordinates, shape (npoin, 3)
    points: np.ndarray
    # raw boundary records, shape (nboun, 5), integer, as stored in file but zero-based for node columns
    boundary: Optional[np.ndarray] = None

    @property
    def boundary_nodes(self) -> np.ndarray:
        """Unique boundary node IDs, zero-based.

        Assumes the first four columns of ibsid are node IDs and the last column is a
        boundary/surface marker. This matches the common 5-int boundary-face layout.
        """
        if self.boundary is None or self.boundary.size == 0:
            return np.array([], dtype=np.int64)

        cand = self.boundary[:, :4].reshape(-1)
        cand = cand[(cand >= 0) & (cand < self.npoin)]
        # preserve first-seen order, like a direct mesh register would
        _, idx = np.unique(cand, return_index=True)
        return cand[np.sort(idx)].astype(np.int64)


@dataclass
class CFDCase:
    base: str
    directory: Path
    mesh: Optional[PLTMesh] = None
    unk: Optional[np.ndarray] = None  # shape (npoin, num_comp)

    @property
    def plt_path(self) -> Path:
        return self.directory / f"{self.base}.plt"

    @property
    def unk_path(self) -> Path:
        return self.directory / f"{self.base}.unk"

    def read_plt(self) -> PLTMesh:
        self.mesh = read_plt(self.plt_path)
        return self.mesh

    def read_unk(self, num_comp: int) -> np.ndarray:
        self.unk = read_unk(self.unk_path, num_comp=num_comp)
        return self.unk

    def write_unk(self, path: Optional[Path] = None, num_comp_out: int = 6) -> None:
        if self.unk is None:
            raise ValueError("No .unk field stored on this CFDCase.")
        if path is None:
            path = self.unk_path
        write_unk(path, self.unk, num_comp_out=num_comp_out)


def _read_record_as_ints(ff: FortranFile) -> np.ndarray:
    """Read a Fortran record as int32, falling back to int64 if needed."""
    try:
        return ff.read_ints(np.int32)
    except Exception:
        return ff.read_ints(np.int64)


def read_plt(path: str | Path) -> PLTMesh:
    path = Path(path)
    with FortranFile(path, "r") as ff:
        header = _read_record_as_ints(ff)
        if header.size < 3:
            raise ValueError(f"{path}: first .plt record does not contain nelem,npoin,nboun")
        nelem, npoin, nboun = map(int, header[:3])

        intma = _read_record_as_ints(ff)
        if intma.size != 4 * nelem:
            raise ValueError(f"{path}: connectivity record has {intma.size} ints, expected {4*nelem}")
        # Fortran wrote ((intma(j,i), i=1,nelem), j=1,4), i.e. shape (4,nelem), Fortran order.
        tets = intma.reshape((4, nelem), order="C").T.astype(np.int64) - 1

        coords = ff.read_reals(np.float64)
        if coords.size != 3 * npoin:
            raise ValueError(f"{path}: coordinate record has {coords.size} floats, expected {3*npoin}")
        points = coords.reshape((3, npoin), order="C").T.astype(np.float64)

        boundary = None
        if nboun > 0:
            try:
                ibsid = _read_record_as_ints(ff)
                if ibsid.size == 5 * nboun:
                    boundary = ibsid.reshape((5, nboun), order="C").T.astype(np.int64)
                    # only first four columns are node IDs; convert those to zero-based
                    boundary[:, :4] -= 1
                else:
                    # Keep going: interpu.f's gtinpt1 did not read new boundary records.
                    print(f"[WARN] {path}: boundary record has {ibsid.size} ints, expected {5*nboun}; ignoring.")
            except Exception:
                print(f"[WARN] {path}: no readable boundary record; boundary direct-transfer disabled for this mesh.")

    return PLTMesh(path=path, nelem=nelem, npoin=npoin, nboun=nboun, tets=tets, points=points, boundary=boundary)


def read_unk(path: str | Path, num_comp: int) -> np.ndarray:
    path = Path(path)
    with FortranFile(path, "r") as ff:
        nrec = _read_record_as_ints(ff)
        if nrec.size < 1:
            raise ValueError(f"{path}: first .unk record does not contain node count")
        npoin = int(nrec[0])
        vals = ff.read_reals(np.float64)

    expected = npoin * num_comp
    if vals.size < expected:
        raise ValueError(f"{path}: field record has {vals.size} floats, expected at least {expected}")
    if vals.size > expected:
        print(f"[WARN] {path}: field record has {vals.size} floats; using first {expected} for {num_comp} components.")
        vals = vals[:expected]

    # Fortran wrote/read ((unkno(j,i), i=1,npoin), j=1,numComp)
    return vals.reshape((num_comp, npoin), order="C").T.copy()


def write_unk(path: str | Path, field: np.ndarray, num_comp_out: int = 6) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"field must be shape (n_nodes, n_comp), got {field.shape}")

    n_nodes, n_comp = field.shape
    out = np.zeros((n_nodes, num_comp_out), dtype=np.float64)
    n_copy = min(n_comp, num_comp_out)
    out[:, :n_copy] = field[:, :n_copy]

    # interpu.f output writes npar then ((st(i,j),j=1,npar),i=1,6)
    record = out.T.reshape(-1).astype(np.float64)
    with FortranFile(path, "w") as ff:
        ff.write_record(np.array([n_nodes], dtype=np.int32))
        ff.write_record(record)


def barycentric_weights(tet_nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return barycentric weights for point x in tetrahedron tet_nodes.

    tet_nodes shape is (4,3). The returned weights correspond to nodes [0,1,2,3].
    """
    a = tet_nodes[0]
    mat = np.column_stack((tet_nodes[1] - a, tet_nodes[2] - a, tet_nodes[3] - a))
    try:
        w123 = np.linalg.solve(mat, x - a)
    except np.linalg.LinAlgError:
        return np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float64)
    w0 = 1.0 - float(np.sum(w123))
    return np.array([w0, w123[0], w123[1], w123[2]], dtype=np.float64)


class ExistingTetSearch:
    """Search the existing old-mesh tetrahedra.

    This replaces the custom tree in interpu.f with a centroid KD-tree over the old
    tetrahedra. For each query point, it checks nearby tetrahedra and selects the
    one with the largest minimum barycentric coordinate, matching the spirit of
    trsear(). If the best minimum barycentric coordinate is non-negative, the point
    is inside that tet. Otherwise, it reports failure so the nearest-node fallback
    is used.
    """

    def __init__(self, mesh: PLTMesh, k_candidates: int = 64, inside_tol: float = 1e-10):
        self.mesh = mesh
        self.k_candidates = int(max(1, k_candidates))
        self.inside_tol = float(inside_tol)
        pts = mesh.points
        tets = mesh.tets
        self.centroids = pts[tets].mean(axis=1)
        self.tree = cKDTree(self.centroids)

    def query(self, x: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        k = min(self.k_candidates, self.mesh.nelem)
        _, idxs = self.tree.query(x, k=k)
        idxs = np.atleast_1d(idxs)

        best_nodes = None
        best_w = None
        best_score = -np.inf

        for eid in idxs:
            nodes = self.mesh.tets[int(eid)]
            tet_xyz = self.mesh.points[nodes]
            w = barycentric_weights(tet_xyz, x)
            # interpu.f clamps later; but trsear chooses max(min(weight))
            score = float(np.min(w))
            if score > best_score:
                best_score = score
                best_nodes = nodes
                best_w = w
            if score >= -self.inside_tol:
                return True, nodes, w

        return False, best_nodes, best_w


def build_boundary_transfer(old: PLTMesh, new: PLTMesh, mode: BoundaryMode) -> dict[int, int]:
    """Return mapping new_node -> old_node for direct boundary value transfer."""
    if mode == "none":
        return {}

    old_bn = old.boundary_nodes
    new_bn = new.boundary_nodes
    if old_bn.size == 0 or new_bn.size == 0:
        print("[WARN] Boundary transfer requested but boundary nodes are unavailable.")
        return {}

    if mode == "auto":
        # Prefer same IDs when legal; else ordered if lengths match; else nearest old boundary.
        if np.all(new_bn < old.npoin):
            mode = "same_id"
        elif old_bn.size == new_bn.size:
            mode = "ordered"
        else:
            mode = "nearest_old_boundary"

    if mode == "same_id":
        out = {int(j_new): int(j_new) for j_new in new_bn if int(j_new) < old.npoin}
        print(f"[INFO] Boundary transfer mode same_id: {len(out)} nodes.")
        return out

    if mode == "ordered":
        n = min(old_bn.size, new_bn.size)
        out = {int(new_bn[i]): int(old_bn[i]) for i in range(n)}
        print(f"[INFO] Boundary transfer mode ordered: {len(out)} nodes.")
        return out

    if mode == "nearest_old_boundary":
        tree = cKDTree(old.points[old_bn])
        _, loc = tree.query(new.points[new_bn], k=1)
        out = {int(j_new): int(old_bn[int(ii)]) for j_new, ii in zip(new_bn, loc)}
        print(f"[INFO] Boundary transfer mode nearest_old_boundary: {len(out)} nodes.")
        return out

    raise ValueError(f"Unknown boundary transfer mode: {mode}")


def map_solution(
    old_case: CFDCase,
    new_case: CFDCase,
    *,
    num_comp: int,
    boundary_mode: BoundaryMode = "same_id",
    k_candidates: int = 64,
    progress_every: int = 1000,
) -> np.ndarray:
    """Map old_case.unk onto new_case.mesh.

    Boundary nodes are copied directly first according to boundary_mode.
    All remaining nodes are interpolated using the old tetrahedral mesh.
    If no containing old tetrahedron is found, the nearest old volume node is used,
    matching interpu.f's at=0 fallback.
    """
    if old_case.mesh is None or new_case.mesh is None:
        raise ValueError("Both cases must have .mesh loaded.")
    if old_case.unk is None:
        raise ValueError("old_case must have .unk loaded.")

    old_mesh = old_case.mesh
    new_mesh = new_case.mesh
    old_u = np.asarray(old_case.unk, dtype=np.float64)

    if old_u.shape[0] != old_mesh.npoin:
        raise ValueError(f"old .unk node count {old_u.shape[0]} != old .plt npoin {old_mesh.npoin}")

    st = np.zeros((new_mesh.npoin, max(num_comp, old_u.shape[1])), dtype=np.float64)
    ncopy = min(st.shape[1], old_u.shape[1])

    direct = build_boundary_transfer(old_mesh, new_mesh, boundary_mode)
    done = np.zeros(new_mesh.npoin, dtype=bool)
    for j_new, j_old in direct.items():
        if 0 <= j_new < new_mesh.npoin and 0 <= j_old < old_mesh.npoin:
            st[j_new, :ncopy] = old_u[j_old, :ncopy]
            done[j_new] = True

    tet_search = ExistingTetSearch(old_mesh, k_candidates=k_candidates)
    old_node_tree = cKDTree(old_mesh.points)

    n_failed = 0
    n_interp = 0
    for ip in range(new_mesh.npoin):
        if progress_every and (ip + 1) % progress_every == 0:
            print(f"[INFO] # of points processed = {ip + 1}")

        if done[ip]:
            continue

        xr = new_mesh.points[ip]
        found, nodes, w = tet_search.query(xr)

        if (not found) or nodes is None or w is None:
            # Exact interpu.f at=0 spirit: nearest old volume node fallback.
            n_failed += 1
            _, j = old_node_tree.query(xr, k=1)
            j = int(j)
            nodes = np.array([j, j, j, j], dtype=np.int64)
            w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        else:
            # Clamp weights to [0,1], then normalise by sum, as interpu.f does.
            w = np.clip(w, 0.0, 1.0)
            at = float(np.sum(w))
            if at == 0.0:
                n_failed += 1
                _, j = old_node_tree.query(xr, k=1)
                j = int(j)
                nodes = np.array([j, j, j, j], dtype=np.int64)
                w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
            else:
                w = w / at
                n_interp += 1

        st[ip, :ncopy] = w @ old_u[nodes, :ncopy]

    print(f"[INFO] Direct boundary nodes copied : {int(done.sum())}")
    print(f"[INFO] Interpolated interior nodes  : {n_interp}")
    print(f"[INFO] Nearest-node fallback nodes : {n_failed}")
    return st[:, :max(num_comp, old_u.shape[1])]


def main() -> None:
    p = argparse.ArgumentParser(description="Python replacement for interpu.f solution transfer.")
    p.add_argument("--old-dir", required=True, help="Directory containing baseline <old-base>.plt and <old-base>.unk")
    p.add_argument("--old-base", required=True, help="Baseline file stem, e.g. corner_1")
    p.add_argument("--new-dir", required=True, help="Directory containing morphed <new-base>.plt and output <new-base>.unk")
    p.add_argument("--new-base", required=True, help="Morphed file stem, e.g. corner_2")
    p.add_argument("--turb", action="store_true", help="Read 6 components from old .unk instead of 5")
    p.add_argument("--num-comp", type=int, default=None, help="Override number of components read from old .unk")
    p.add_argument("--out", default=None, help="Optional explicit output .unk path")
    p.add_argument(
        "--boundary-mode",
        choices=["none", "same_id", "ordered", "nearest_old_boundary", "auto"],
        default="same_id",
        help="How to directly copy boundary values before interpolation.",
    )
    p.add_argument("--k-candidates", type=int, default=64, help="Number of nearby old tets checked per query point")
    args = p.parse_args()

    num_comp = args.num_comp if args.num_comp is not None else (6 if args.turb else 5)

    old = CFDCase(base=args.old_base, directory=Path(args.old_dir))
    new = CFDCase(base=args.new_base, directory=Path(args.new_dir))

    print(f"[INFO] Reading old .plt: {old.plt_path}")
    old.read_plt()
    print(f"[INFO] Reading old .unk: {old.unk_path} with num_comp={num_comp}")
    old.read_unk(num_comp=num_comp)

    print(f"[INFO] Reading new .plt: {new.plt_path}")
    new.read_plt()

    print("[INFO] Mapping solution...")
    new.unk = map_solution(
        old,
        new,
        num_comp=num_comp,
        boundary_mode=args.boundary_mode,
        k_candidates=args.k_candidates,
    )

    out_path = Path(args.out) if args.out else new.unk_path
    print(f"[INFO] Writing new .unk: {out_path}")
    new.write_unk(path=out_path, num_comp_out=num_comp)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
