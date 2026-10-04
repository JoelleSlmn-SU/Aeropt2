"""
Validation of the 2D edge-mode parameterisation (ShapeParameterization/edgeModes2D.py)
and the 2D morph (MeshGeneration/Morph2D.py) on synthetic geometries.

Run from the repository root:
    python -m pytest tests/test_edge_modes_2d.py -q
or, without pytest:
    python tests/test_edge_modes_2d.py

No mesh files are needed: every mesh is built here.
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FileRW.EdgeMesh2D import EdgeMesh2D                                    # noqa: E402
from MeshGeneration.Morph2D import Morph2D                                  # noqa: E402
from ShapeParameterization.edgeModes2D import EdgeModes2D, segment_modes    # noqa: E402


# ---------------------------------------------------------------------------
# geometry builders
# ---------------------------------------------------------------------------
def naca0012(n_side=80, closed_te=True, spacing="cosine"):
    """Clockwise loop (fluid on the left for a body in a fluid domain), starting at the
    trailing edge: TE -> lower surface -> LE -> upper surface (-> TE).
    closed_te=False gives a blunt base: the loop starts at the lower TE corner and
    ends at the upper one."""
    if spacing in ("cosine", "arc"):
        x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_side + 1)))
    else:
        x = np.linspace(0.0, 1.0, n_side + 1)
    a4 = -0.1036 if closed_te else -0.1015
    y = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    up = np.column_stack([x, y])            # LE -> TE
    lo = np.column_stack([x, -y])
    if closed_te:
        loop = np.vstack([[[1.0, 0.0]], lo[::-1][1:], up[1:-1]])
    else:
        loop = np.vstack([lo[::-1], up[1:]])
    if spacing == "arc":                     # equal arc-length spacing (coarse at the LE)
        # resample a fine reference curve, so the nose is not cut
        ref = naca0012(4000, closed_te=closed_te)
        R = np.vstack([ref, ref[:1]])
        sr = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(R, axis=0), axis=1))])
        n = 2 * n_side
        si = np.linspace(0.0, sr[-1], n + 1)[:-1]
        loop = np.column_stack([np.interp(si, sr, R[:, 0]), np.interp(si, sr, R[:, 1])])
    return loop


def o_mesh(inner, r_far=8.0, n_layers=40, centre=(0.5, 0.0)):
    """O-mesh between a star-shaped inner loop and a circle. Returns EdgeMesh2D, inner node ids."""
    inner = np.asarray(inner, float)
    c = np.asarray(centre, float)
    n = len(inner)
    d = inner - c
    outer = c + r_far * d / np.linalg.norm(d, axis=1, keepdims=True)
    t = np.linspace(0.0, 1.0, n_layers + 1) ** 2.0           # cluster near the body
    P = np.vstack([(1 - tj) * inner + tj * outer for tj in t])
    tri = []
    for j in range(n_layers):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            cc, dd = a + n, b + n
            tri += [[a, b, dd], [a, dd, cc]]
    m = EdgeMesh2D(P, np.asarray(tri))
    m.label_by_corners(30.0)
    return m, np.arange(n)


def channel(nx=120, ny=24, L=6.0, H=2.0, ramp=(2.0, 4.0), angle_deg=10.0):
    """Channel with a ramp on the bottom wall (two kinks of `angle_deg`)."""
    x = np.linspace(0.0, L, nx + 1)
    i0, i1 = int(round(ramp[0] / L * nx)), int(round(ramp[1] / L * nx))
    yb = np.zeros_like(x)
    yb[i0:i1 + 1] = (x[i0:i1 + 1] - x[i0]) * np.tan(np.radians(angle_deg))
    yb[i1 + 1:] = yb[i1]
    eta = np.linspace(0.0, 1.0, ny + 1)
    P = np.array([[x[i], yb[i] + (H - yb[i]) * e] for e in eta for i in range(nx + 1)])
    tri = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, cc, dd = a + 1, a + nx + 1, a + nx + 2
            tri += [[a, b, dd], [a, dd, cc]]
    m = EdgeMesh2D(P, np.asarray(tri))
    m.label_by_corners(30.0)
    return m, (x[i0], yb[i0]), (x[i1], yb[i1])


def edge_where(m, test):
    """ids of the edges whose mid node satisfies test(x, y)."""
    out = []
    for e in m.edges.values():
        p = m.nodes[e.nodes[len(e.nodes) // 2], :2]
        if test(p[0], p[1]):
            out.append(e.id)
    return out


def stub_mesh(xy, closed=True, cuts=()):
    """Minimal mesh for EdgeModes2D only: one loop/polyline, split into edges at `cuts`."""
    xy = np.asarray(xy, float)
    n = len(xy)
    nodes = np.column_stack([xy, np.zeros(n)])
    ids = np.arange(n)
    if closed:
        path = np.append(ids, 0)
    else:
        path = ids
    bps = sorted(set(cuts) | {0, len(path) - 1})
    edges = {k + 1: SimpleNamespace(id=k + 1, nodes=path[a:b + 1]) for k, (a, b) in enumerate(zip(bps[:-1], bps[1:]))}
    return SimpleNamespace(nodes=nodes, edges=edges)


def pinned_xy(b):
    return b.xy[b.pinned]


# ---------------------------------------------------------------------------
# 1. modes of one segment against the analytic solutions
# ---------------------------------------------------------------------------
def test_segment_modes_analytic():
    rng = np.random.default_rng(0)
    h = rng.uniform(0.5, 1.5, 400)
    h *= 3.0 / h.sum()
    L = 3.0
    s = np.concatenate([[0.0], np.cumsum(h)])
    kap, phi = segment_modes(h, 4, "pinned", "pinned")
    assert np.allclose(kap * L / np.pi, [1, 2, 3, 4], rtol=1e-3)
    assert np.abs(phi[:, 0] - np.sin(np.pi * s / L)).max() < 1e-4        # shape and unit amplitude
    kap, phi = segment_modes(h, 3, "clamped", "clamped")
    assert np.allclose(kap * L, [4.7300, 7.8532, 10.9956], rtol=1e-3)     # clamped-clamped beam
    assert np.all(np.abs(phi[1] / h[0]) < 0.05 * np.abs(np.diff(phi, axis=0) / h[:, None]).max(0))
    kap, _ = segment_modes(h, 3, "clamped", "pinned")
    assert np.allclose(kap * L, [3.9266, 7.0686, 10.2102], rtol=1e-3)     # clamped-pinned beam
    kap, _ = segment_modes(h, 3, "free", "pinned")
    assert np.allclose(kap * L / np.pi, [0.5, 1.5, 2.5], rtol=1e-3)
    kap, _ = segment_modes(h, 2, "free", "free")
    assert np.allclose(kap * L / np.pi, [1, 2], rtol=1e-3)                # constant mode dropped
    kap, phi = segment_modes(h, 4, periodic=True)
    assert np.allclose(kap * L / (2 * np.pi), [1, 1, 2, 2], rtol=1e-3)
    assert np.all(phi[0, 1::2] == 0.0) and np.all(phi[0, 0::2] > 0.99)    # fixed (cos, sin) phase


# ---------------------------------------------------------------------------
# 2. where chains get pinned
# ---------------------------------------------------------------------------
def test_sharp_te_is_pinned_and_le_is_not():
    b = EdgeModes2D.from_mesh(stub_mesh(naca0012(80)), [1], n_modes=6)
    assert b.closed and int(b.pinned.sum()) == 1
    assert np.allclose(pinned_xy(b), [[1.0, 0.0]])
    assert len(b.segments) == 1 and b.segments[0].ends_label == "pinned-pinned"
    assert np.all(b.phi[b.pinned] == 0.0)
    # sin(n pi s / L): odd n symmetric about the LE (thickness), even n antisymmetric (camber)
    L = b.segments[0].length
    assert np.allclose(b.lam, (np.arange(1, 7) * np.pi / L) ** 2, rtol=2e-3)


def test_coarse_leading_edge_is_not_a_corner():
    # ~1%-chord arc-length spacing (as in examples/naca0012): the LE turns by
    # more than 30 deg at one node, but its neighbours turn too -> a coarsely
    # resolved smooth arc, which must not be pinned
    b = EdgeModes2D.from_mesh(stub_mesh(naca0012(102, spacing="arc")), [1], n_modes=0)
    le = int(np.argmin(b.xy[:, 0]))
    assert 30.0 < abs(b.turn_deg[le]) < 60.0, b.turn_deg[le]
    assert not b.pinned[le] and int(b.pinned.sum()) == 1
    assert any("tightest free curvature" in t for t in b.notes)   # ... but the user is told


def test_blunt_te_base_stays_fixed():
    b = EdgeModes2D.from_mesh(stub_mesh(naca0012(80, closed_te=False)), [1], n_modes=4)
    assert int(b.pinned.sum()) == 2                               # both TE corners (adjacent nodes)
    assert np.allclose(np.sort(pinned_xy(b)[:, 0]), [1.0, 1.0])
    assert len(b.segments) == 1                                   # the 1-segment base has no free node
    assert np.all(b.phi[b.pinned] == 0.0)


def test_small_round_cap_is_pinned_by_radius():
    # flat plate with semicircular ends of radius 0.002 on a perimeter ~4: R/L = 5e-4 < 1e-3
    r, n_arc = 0.002, 6
    th = np.linspace(-np.pi / 2, np.pi / 2, n_arc + 1)
    right = np.column_stack([2.0 + r * np.cos(th), r * np.sin(th)])
    left = np.column_stack([-r * np.cos(th), -r * np.sin(th)])
    top = np.column_stack([np.linspace(2.0, 0.0, 101)[1:-1], np.full(99, r)])
    bot = np.column_stack([np.linspace(0.0, 2.0, 101)[1:-1], np.full(99, -r)])
    loop = np.vstack([right, top, left[::-1][::-1], bot])[::-1]
    b = EdgeModes2D.from_mesh(stub_mesh(loop), [1], n_modes=4)
    assert np.all(np.abs(b.turn_deg[b.pinned]) <= 30.0 + 1e-9)    # caught by the radius rule
    assert int(b.pinned.sum()) >= 2 * (n_arc - 1)
    assert np.all(b.phi[b.pinned] == 0.0)
    assert len([s for s in b.segments if s.k_max > 0]) == 2       # the two flat sides


def test_circle_is_periodic_with_fixed_phase():
    th = np.linspace(0.0, 2 * np.pi, 121)[:-1]
    loop = np.column_stack([np.cos(th), np.sin(th)])[::-1]
    b = EdgeModes2D.from_mesh(stub_mesh(loop), [1], n_modes=4)
    assert not b.pinned.any() and b.segments[0].periodic
    assert np.all(b.phi[0, 1::2] == 0.0)
    b3 = EdgeModes2D.from_mesh(stub_mesh(loop), [1], n_modes=3)
    assert any("pair" in t for t in b3.notes)


def test_junction_pinning_and_edge_order_independence():
    loop = naca0012(80)
    le = int(np.argmin(loop[:, 0]))
    mesh = stub_mesh(loop, cuts=(le,))                            # edge 1: TE->LE, edge 2: LE->TE
    b = EdgeModes2D.from_mesh(mesh, [1, 2], n_modes=6, pin_junctions=True)
    assert int(b.pinned.sum()) == 2 and len(b.segments) == 2
    assert b.segments[0].ends_label == "pinned-clamped"           # TE corner = hinge, LE = smooth
    assert sorted(b.mode_segment.tolist()) == [0, 0, 0, 1, 1, 1]  # three per surface
    assert b.mode_segment.tolist() == [0, 1, 0, 1, 0, 1]          # equal lengths: ranked by segment
    b2 = EdgeModes2D.from_mesh(mesh, [1, 2], n_modes=6)           # default: LE free, one segment
    assert int(b2.pinned.sum()) == 1 and len(b2.segments) == 1


def test_user_fixed_point_smooth_and_hinge():
    loop = naca0012(80)
    le = int(np.argmin(loop[:, 0]))
    mesh = stub_mesh(loop)
    h = np.linalg.norm(loop[le + 1] - loop[le])
    for ftype, label, smooth in (("clamped", "pinned-clamped", True), ("pinned", "pinned-pinned", False)):
        b = EdgeModes2D.from_mesh(mesh, [1], n_modes=6, fixed_nodes=[le], fixed_type=ftype)
        pos = int(np.flatnonzero(b.gids == le)[0])
        assert b.pinned[pos] and int(b.pinned.sum()) == 2 and len(b.segments) == 2
        assert b.segments[0].ends_label == label
        assert np.all(b.phi[pos] == 0.0)                              # the point does not move
        assert b.mode_segment.tolist() == [0, 1, 0, 1, 0, 1]          # three modes per surface
        slope = np.abs(b.phi[pos + 1] / h)                            # slope leaving the fixed point
        peak = np.abs(np.diff(b.phi, axis=0)).max(0) / h
        assert np.all(slope < 0.1 * peak) == smooth, (ftype, slope / peak)
    # default type is the smooth one
    assert EdgeModes2D.from_mesh(mesh, [1], n_modes=0, fixed_nodes=[le]).fixed_type == "clamped"
    # fixing a corner that is already pinned changes nothing
    te = 0
    b0 = EdgeModes2D.from_mesh(mesh, [1], n_modes=4)
    b1 = EdgeModes2D.from_mesh(mesh, [1], n_modes=4, fixed_nodes=[te])
    assert np.array_equal(b0.phi, b1.phi)
    # a node that is not on T is refused
    try:
        EdgeModes2D.from_mesh(mesh, [1], n_modes=0, fixed_nodes=[10 ** 6])
        raise AssertionError("fixed node off T was accepted")
    except ValueError:
        pass


def test_fixed_point_on_smooth_loop_and_open_free_chain():
    th = np.linspace(0.0, 2 * np.pi, 121)[:-1]
    circle = np.column_stack([np.cos(th), np.sin(th)])[::-1]
    b = EdgeModes2D.from_mesh(stub_mesh(circle), [1], n_modes=4, fixed_nodes=[30])
    assert len(b.segments) == 1 and b.segments[0].ends_label == "clamped-clamped"
    assert b.gids[0] == 30 and np.all(b.phi[0] == 0.0)                # loop now starts at the fixed point
    line = np.column_stack([np.linspace(0.0, 4.0, 161), np.zeros(161)])
    b = EdgeModes2D.from_mesh(stub_mesh(line, closed=False), [1], n_modes=4, ends="free", fixed_nodes=[80])
    assert [s.ends_label for s in b.segments] == ["free-clamped", "clamped-free"]   # two cantilevers
    assert np.allclose(np.sqrt(b.lam[:2]) * 2.0, 1.8751, rtol=2e-3)


# ---------------------------------------------------------------------------
# 3. the morph: aerofoil
# ---------------------------------------------------------------------------
def test_aerofoil_all_modes_usable_and_old_basis_is_not():
    m, _ = o_mesh(naca0012(80))
    t = edge_where(m, lambda x, y: abs(x - 0.5) < 2 and abs(y) < 1)
    mo = Morph2D(m, t, n_modes=6)
    lim = [mo.max_safe_amplitude(k) for k in range(6)]
    assert all(mo.usable_range(lim)), lim
    x = np.array([0.5 * (lo + hi) * 0.5 for lo, hi in lim])       # inside every single-mode range
    new, rep = mo.morph(0.2 * x)
    assert rep.ok, rep
    assert rep.max_fixed_disp == 0.0
    te = mo.modes.gids[mo.modes.pinned]
    assert np.allclose(new.nodes[te], m.nodes[te])                # trailing edge did not move

    # the pre-fix basis: no pinning. Its cos-type modes push the two sides of the
    # sharp trailing edge apart and have (almost) no valid range.
    old = Morph2D(m, t, n_modes=6, corner_deg=180.0, min_radius_frac=0.0)
    usable = old.usable_range([old.max_safe_amplitude(k) for k in range(6)])
    assert usable == [False, True, False, True, False, True], usable   # cos-type modes are dead


def test_aerofoil_with_fixed_leading_edge():
    m, _ = o_mesh(naca0012(80))
    t = edge_where(m, lambda x, y: abs(x - 0.5) < 2 and abs(y) < 1)
    le = int(np.argmin(np.where(np.isin(np.arange(m.node_count), np.concatenate([m.edges[e].nodes for e in t])),
                                m.nodes[:, 0], np.inf)))
    mo = Morph2D(m, t, n_modes=6, fixed_nodes=[le])
    lim = [mo.max_safe_amplitude(k) for k in range(6)]
    assert all(mo.usable_range(lim)), lim
    # single-mode limits do not add up: stay well inside them when combining all modes
    new, rep = mo.morph([0.08 * hi for _, hi in lim])
    assert rep.ok, rep
    held = mo.modes.gids[mo.modes.pinned]
    assert le in held and np.allclose(new.nodes[held], m.nodes[held])   # LE and TE did not move
    with tempfile.TemporaryDirectory() as d:                            # fixed points survive save/load
        path = os.path.join(d, "b.json")
        mo.modes.save(path)
        b2 = EdgeModes2D.from_saved(m, path)
        assert b2.fixed_nodes == [le] and b2.fixed_type == "clamped"
        assert np.array_equal(b2.phi, mo.modes.phi)


def test_limit_search_resolves_small_limits():
    m, _ = o_mesh(naca0012(80))
    t = edge_where(m, lambda x, y: abs(x - 0.5) < 2 and abs(y) < 1)
    old = Morph2D(m, t, n_modes=2, corner_deg=180.0, min_radius_frac=0.0)
    lo, hi = old.max_safe_amplitude(0)
    assert lo < 0.0 < hi                                           # tiny, but no longer reported as 0
    x = np.zeros(2)
    x[0] = 0.9 * hi
    assert old.morph(x)[1].ok
    x[0] = 1.2 * hi
    assert not old.morph(x)[1].ok


# ---------------------------------------------------------------------------
# 4. the morph: channel with a ramp (open chains, kinks, two chains, clamped ends)
# ---------------------------------------------------------------------------
def test_channel_bottom_wall_kinks():
    m, k0, k1 = channel()
    bottom = edge_where(m, lambda x, y: y < 0.5 and 0.5 < x < 5.5)
    assert len(bottom) == 1
    b = EdgeModes2D.from_mesh(m, bottom, n_modes=4)               # 10 deg kinks < 30 deg: not pinned
    assert not b.closed and int(b.pinned.sum()) == 2              # only the two chain ends
    b5 = EdgeModes2D.from_mesh(m, bottom, n_modes=6, corner_deg=5.0)
    assert int(b5.pinned.sum()) == 4 and len(b5.segments) == 3
    for kx, ky in (k0, k1):
        assert np.any(np.all(np.isclose(pinned_xy(b5), [kx, ky]), axis=1))
    mo = Morph2D(m, bottom, n_modes=6, corner_deg=5.0)
    lim = [mo.max_safe_amplitude(k) for k in range(6)]
    assert all(mo.usable_range(lim)), lim
    new, rep = mo.morph([0.25 * hi for _, hi in lim])
    assert rep.ok and rep.max_fixed_disp == 0.0
    pins = mo.modes.gids[mo.modes.pinned]
    assert np.allclose(new.nodes[pins], m.nodes[pins])


def test_channel_clamped_ends_have_no_kink():
    m, _, _ = channel()
    bottom = edge_where(m, lambda x, y: y < 0.5 and 0.5 < x < 5.5)
    for ends, small in (("fixed", False), ("clamped", True)):
        b = EdgeModes2D.from_mesh(m, bottom, n_modes=3, ends=ends)
        h0 = np.linalg.norm(b.xy[1] - b.xy[0])
        end_slope = np.abs(b.phi[1] / h0)                          # slope of each mode at the chain start
        peak_slope = np.abs(np.diff(b.phi, axis=0)).max(0) / h0
        assert np.all(end_slope < 0.1 * peak_slope) == small, (ends, end_slope / peak_slope)


def test_two_chains_and_a_corner_inside_a_chain():
    m, _, _ = channel()
    bottom = edge_where(m, lambda x, y: y < 0.5 and 0.5 < x < 5.5)
    top = edge_where(m, lambda x, y: y > 1.9)
    outlet = edge_where(m, lambda x, y: x > 5.99)
    b = EdgeModes2D.from_mesh(m, bottom + top, n_modes=6)
    assert len(b.chains) == 2 and not b.closed and len(b.segments) == 2
    assert set(b.mode_segment.tolist()) == {0, 1}                 # both walls get modes
    mo = Morph2D(m, bottom + top, n_modes=6)
    lim = [mo.max_safe_amplitude(k) for k in range(6)]
    assert all(mo.usable_range(lim)), lim
    _, rep = mo.morph([0.2 * hi for _, hi in lim])
    assert rep.ok and rep.max_fixed_disp == 0.0

    b = EdgeModes2D.from_mesh(m, bottom + outlet, n_modes=4)      # one chain, 90 deg corner inside
    assert len(b.chains) == 1 and int(b.pinned.sum()) == 3 and len(b.segments) == 2
    mo = Morph2D(m, bottom + outlet, n_modes=4)
    lim = [mo.max_safe_amplitude(k) for k in range(4)]
    assert all(mo.usable_range(lim)), lim


# ---------------------------------------------------------------------------
# 5. save / load
# ---------------------------------------------------------------------------
def test_save_load_round_trip():
    m, _ = o_mesh(naca0012(80))
    t = edge_where(m, lambda x, y: abs(x - 0.5) < 2 and abs(y) < 1)
    b = EdgeModes2D.from_mesh(m, t, n_modes=5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "edge_modes_2d.json")
        b.save(path)
        b2 = EdgeModes2D.from_saved(m, path)
        assert np.array_equal(b2.phi, b.phi) and np.array_equal(b2.normals, b.normals)
        assert np.array_equal(b2.gids, b.gids)
        mo, mo2 = Morph2D(m, t, n_modes=5), Morph2D(m, t, modes=b2)
        x = [0.01, -0.02, 0.005, 0.01, -0.003]
        assert np.array_equal(mo.displacement_field(x), mo2.displacement_field(x))
        with open(path) as f:
            dd = json.load(f)
        dd["xy"][3][0] += 1e-3                                     # a different baseline must be refused
        try:
            EdgeModes2D.from_saved(m, dd)
            raise AssertionError("moved baseline was accepted")
        except ValueError:
            pass
        v1 = {k: dd[k] for k in ("t_edges", "gids", "n_modes")}    # old file from the unpinned basis
        old = EdgeModes2D.from_mesh(m, t, n_modes=5, corner_deg=180.0, min_radius_frac=0.0)
        v1.update(ends="periodic", lam=old.lam.tolist(), gids=old.gids.tolist())
        try:
            EdgeModes2D.from_saved(m, v1)
            raise AssertionError("old unpinned basis was silently replaced")
        except ValueError:
            pass


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:                                  # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
