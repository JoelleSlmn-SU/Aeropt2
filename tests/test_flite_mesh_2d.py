"""
Validation of the FLITE 2D solver-mesh converter (FileRW/FliteMesh2D.py).

Run from the repository root:
    python -m pytest tests/test_flite_mesh_2d.py -q
or, without pytest:
    python tests/test_flite_mesh_2d.py

These tests check the FILE (format, numbering, precision, refusals). They do
not run PrePro.exe / Solver.exe.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_edge_modes_2d import channel, edge_where, naca0012, o_mesh             # noqa: E402
from FileRW.EdgeMesh2D import EdgeMesh2D                                         # noqa: E402
from FileRW.FliteMesh2D import (FLAG_FAR_FIELD, FLAG_INVISCID_WALL, FLAG_VISCOUS_WALL,   # noqa: E402
                                check_flite2d_mesh, default_edge_flags, flag_text, flags_from_basis,
                                is_flite2d_mesh, parse_flag, read_flite2d_mesh, write_flite2d_mesh,
                                write_from_basis)
from MeshGeneration.Morph2D import Morph2D                                       # noqa: E402
from MeshGeneration.meshFile import find_dat_for                                 # noqa: E402


def _aerofoil():
    m, _ = o_mesh(naca0012(60), n_layers=20)
    return m


def test_default_flags_body_in_far_field():
    m = _aerofoil()
    f = default_edge_flags(m)
    wall = edge_where(m, lambda x, y: abs(x - 0.5) < 2 and abs(y) < 1)
    far = [e for e in m.edges if e not in wall]
    assert wall and far
    assert all(f[e] == FLAG_INVISCID_WALL for e in wall) and all(f[e] == FLAG_FAR_FIELD for e in far)
    assert all(default_edge_flags(m, viscous=True)[e] == FLAG_VISCOUS_WALL for e in wall)


def test_layout_matches_the_original_file_format():
    m = _aerofoil()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "volumes", "n_0", "case_0.dat")
        info = write_flite2d_mesh(m, path, default_edge_flags(m))
        with open(path, "rb") as fh:
            raw = fh.read()
        assert b"\r" not in raw                                          # unix newlines
        L = raw.decode().split("\n")
        assert L[0] == "1" and L[1] == "title" and L[2] == "ne\tnp\tnb\t" and L[4] == "connectivities"
        ne, npn, nb = (int(v) for v in L[3].split())
        assert (ne, npn, nb) == (len(m.triangles), m.node_count, info.n_boundary)
        assert L[5 + ne] == "coordinates" and L[6 + ne + npn] == "boundaries"
        assert len(L) == 7 + ne + npn + nb + 1 and L[-1] == ""           # nothing after the boundaries
        assert len(L[5].split()) == 5 and L[5].split()[4] == "1"         # id n1 n2 n3 1
        assert len(L[6 + ne].split()) == 3 and len(L[7 + ne + npn].split()) == 3
        assert is_flite2d_mesh(path)
        assert check_flite2d_mesh(path) == {k: 0 for k in check_flite2d_mesh(path)}


def test_round_trip_numbering_and_flags():
    m = _aerofoil()
    flags = default_edge_flags(m)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.dat")
        info = write_flite2d_mesh(m, path, flags, write_map=True)
        r = read_flite2d_mesh(path)
        order = info.node_order
        assert np.array_equal(np.sort(order), np.arange(m.node_count))                  # a permutation
        assert np.array_equal(r["nodes"], m.nodes[order, :2])                           # exact coordinates
        assert np.array_equal(order[r["triangles"]], m.triangles)                       # same elements
        B = r["boundary"]
        assert B[:, :2].max() + 1 == len(B)                                             # boundary nodes first
        n_wall = int((B[:, 2] == FLAG_INVISCID_WALL).sum())
        assert np.all(B[:n_wall, 2] == FLAG_INVISCID_WALL) and np.all(B[n_wall:, 2] == FLAG_FAR_FIELD)
        assert np.array_equal(B[:n_wall, 0], np.arange(n_wall))                         # wall nodes 1..n in order
        for (a, b, f), eid in zip(B, info.boundary_edge_id):                            # each row = its edge
            assert f == flags[int(eid)]
            assert int(order[a]) in m.edges[int(eid)].nodes and int(order[b]) in m.edges[int(eid)].nodes
        # counter-clockwise triangles, fluid on the left of every boundary edge
        T = r["triangles"]
        S = set(map(tuple, np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]).tolist()))
        assert all((int(a), int(b)) in S for a, b, _ in B)
        mp = np.load(info.map_path)
        assert np.array_equal(mp["node_order"], order) and np.array_equal(mp["boundary_flag"], B[:, 2])


def test_full_precision_keeps_close_nodes_apart():
    # the MATLAB writer kept 5 significant digits; nodes 1e-7 apart at x ~ 10 then coincide
    P = np.array([[10.0, 0.0], [10.0000001, 0.0], [10.0, 1.0], [10.0000001, 1.0]])
    m = EdgeMesh2D(P, np.array([[0, 1, 3], [0, 3, 2]]))
    m.label_by_corners(30.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.dat")
        info = write_flite2d_mesh(m, path, {e: FLAG_INVISCID_WALL for e in m.edges})
        r = read_flite2d_mesh(path)
        assert np.array_equal(r["nodes"], P[info.node_order])
        assert check_flite2d_mesh(path)["coincident_node_pairs"] == 0
        five = np.array([[float(f"{v:.5g}") for v in row] for row in P])
        assert len(np.unique(five, axis=0)) < len(P)                    # what 5 digits would have done


def test_scale():
    m = _aerofoil()
    with tempfile.TemporaryDirectory() as d:
        info = write_flite2d_mesh(m, os.path.join(d, "m.dat"), default_edge_flags(m), scale=1e-3)
        r = read_flite2d_mesh(info.path)
        assert np.allclose(r["nodes"], 1e-3 * m.nodes[info.node_order, :2], rtol=1e-15, atol=0.0)
        for bad in (0.0, -1.0, float("nan")):
            try:
                write_flite2d_mesh(m, os.path.join(d, "x.dat"), default_edge_flags(m), scale=bad)
                raise AssertionError("bad scale accepted")
            except ValueError:
                pass


def test_refuses_what_the_solver_cannot_survive():
    m = _aerofoil()
    flags = default_edge_flags(m)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.dat")

        def refused(mesh, fl):
            try:
                write_flite2d_mesh(mesh, path, fl)
            except ValueError:
                return not os.path.exists(path)          # and nothing half-written is left behind
            return False

        missing = dict(flags)
        missing.pop(next(iter(missing)))
        assert refused(m, missing)                                       # an edge without a flag
        assert refused(m, {**flags, 999: 1})                             # a flag for an edge that does not exist
        inv = m.copy()
        tri = inv.triangles[0]
        inv.nodes[tri[0], :2] = 2.0 * inv.nodes[tri[1], :2] - inv.nodes[tri[0], :2] + 5.0   # fold one node away
        assert (inv.triangle_areas() <= 0).any() and refused(inv, flags) # inverted element
        dup = m.copy()
        tri = dup.triangles[5]
        dup.nodes[tri[0]] = dup.nodes[tri[1]]
        assert refused(dup, flags)                                       # coincident nodes


def test_other_flags_pass_through_and_text_parsing():
    assert parse_flag("Far field (3)") == 3 and parse_flag(" 7 ") == 7 and parse_flag(5) == 5
    assert flag_text(1) == "Inviscid wall (1)" and flag_text(7) == "7"
    for bad in ("wall", "", "-2", "1.5"):
        try:
            parse_flag(bad)
            raise AssertionError(f"{bad!r} accepted")
        except ValueError:
            pass
    m, _, _ = channel(nx=40, ny=8)
    inlet = edge_where(m, lambda x, y: x < 0.01)
    flags = {e: (7 if e in inlet else FLAG_VISCOUS_WALL) for e in m.edges}
    with tempfile.TemporaryDirectory() as d:
        info = write_flite2d_mesh(m, os.path.join(d, "m.dat"), flags)
        B = read_flite2d_mesh(info.path)["boundary"]
        assert set(np.unique(B[:, 2]).tolist()) == {7, FLAG_VISCOUS_WALL}
        assert info.flag_counts[7] == int((B[:, 2] == 7).sum()) > 0
        assert B[0, 2] == FLAG_VISCOUS_WALL                              # single loop: starts on a wall


def test_solver_mesh_is_not_taken_for_a_geometry_file():
    m = _aerofoil()
    with tempfile.TemporaryDirectory() as d:
        mesh_path = os.path.join(d, "case.vtm")
        open(mesh_path, "w").close()
        write_flite2d_mesh(m, os.path.join(d, "case.dat"), default_edge_flags(m))
        assert find_dat_for(mesh_path) is None                           # solver mesh: ignored
        with open(os.path.join(d, "case.dat"), "w") as fh:
            fh.write("npoin nseg nvseg nlayer\n4 1 0 0 0.0\n")
        assert find_dat_for(mesh_path) == os.path.join(d, "case.dat")    # geometry file: still found


def test_every_deformation_uses_the_same_numbering():
    m = _aerofoil()
    t = edge_where(m, lambda x, y: abs(x - 0.5) < 2 and abs(y) < 1)
    mo = Morph2D(m, t, n_modes=4)
    flags = default_edge_flags(m)
    basis = {"solver_bc_flags": {str(k): v for k, v in flags.items()}, "solver_mesh_scale": 2.0}
    assert flags_from_basis(basis) == flags
    with tempfile.TemporaryDirectory() as d:
        base = write_from_basis(m, os.path.join(d, "n_0.dat"), basis, write_map=True)
        new, rep = mo.morph([0.01, -0.01, 0.005, 0.0])
        assert rep.ok
        info = write_from_basis(new, os.path.join(d, "n_1.dat"), basis)
        assert np.array_equal(info.node_order, base.node_order)          # one map serves all designs
        r0, r1 = read_flite2d_mesh(base.path), read_flite2d_mesh(info.path)
        assert np.array_equal(r0["triangles"], r1["triangles"]) and np.array_equal(r0["boundary"], r1["boundary"])
        assert np.array_equal(r1["nodes"], 2.0 * new.nodes[info.node_order, :2])
        moved = np.abs(r1["nodes"] - r0["nodes"]).max(axis=1) > 0
        assert moved.any() and not moved[r0["boundary"][r0["boundary"][:, 2] == FLAG_FAR_FIELD, 0]].any()
        try:
            write_from_basis(m, os.path.join(d, "x.dat"), {"solver_mesh_scale": 1.0})
            raise AssertionError("basis without flags accepted")
        except ValueError:
            pass


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and fn.__module__ == "__main__":
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:                                  # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
