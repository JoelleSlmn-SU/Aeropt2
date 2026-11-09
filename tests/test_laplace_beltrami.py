"""Laplace-Beltrami modal basis: analytic eigenvalues, cache plumbing, mismatch guard."""
import os
import json

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from ShapeParameterization.modalBasis import (
    build_laplace_beltrami_basis, cotangent_laplacian, boundary_vertices,
)
from ShapeParameterization.controlNodeDisp import (
    build_t_surface_modal_cache, getDisplacements, modal_cache_mismatch, cache_basis_method,
)
from ShapeParameterization.designDisplacement import control_displacements_from_basis


def _faces(mesh):
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]


def test_sphere_eigenvalues():
    s = pv.Icosphere(radius=1.0, nsub=4).triangulate().clean()
    ev, phi, info = build_laplace_beltrami_basis(s.points, _faces(s), k_modes=8, boundary="free")
    # l(l+1): 2 (x3), 6 (x5)
    assert np.allclose(ev[:3], 2.0, rtol=2e-3)
    assert np.allclose(ev[3:8], 6.0, rtol=5e-3)
    assert info["n_components"] == 1 and info["n_boundary_nodes"] == 0


@pytest.mark.parametrize("boundary, expected", [("free", [1, 1, 2, 4]), ("clamped", [2, 5, 5])])
def test_unit_square_eigenvalues(boundary, expected):
    p = pv.Plane(i_size=1, j_size=1, i_resolution=30, j_resolution=30).triangulate().clean()
    ev, phi, _ = build_laplace_beltrami_basis(p.points, _faces(p), k_modes=len(expected), boundary=boundary)
    assert np.allclose(ev / np.pi ** 2, expected, rtol=1.5e-2)
    # unit l2 normalisation (same convention as the graph basis)
    assert np.allclose(np.linalg.norm(phi, axis=0), 1.0)
    if boundary == "clamped":
        assert np.allclose(phi[boundary_vertices(_faces(p))], 0.0)


def test_stiffness_is_psd_and_mass_is_area():
    p = pv.Plane(i_size=2, j_size=3, i_resolution=7, j_resolution=5).triangulate().clean()
    S, M, area = cotangent_laplacian(p.points, _faces(p))
    assert np.isclose(M.diagonal().sum(), 6.0)
    assert np.allclose(S @ np.ones(S.shape[0]), 0.0, atol=1e-10)   # constants in the kernel
    assert np.min(np.linalg.eigvalsh(S.toarray())) > -1e-10


def test_signs_are_deterministic():
    p = pv.Plane(i_size=1, j_size=1, i_resolution=12, j_resolution=12).triangulate().clean()
    _, a, _ = build_laplace_beltrami_basis(p.points, _faces(p), k_modes=5)
    _, b, _ = build_laplace_beltrami_basis(p.points, _faces(p), k_modes=5)
    assert np.allclose(a, b)


def _case(tmp_path, n=14):
    out = str(tmp_path)
    surf = pv.Plane(i_size=1, j_size=1, i_resolution=n, j_resolution=n).triangulate().clean()
    os.makedirs(os.path.join(out, "surfaces"))
    surf.save(os.path.join(out, "surfaces", "output.vtk"))
    pts = np.asarray(surf.points)
    cn_idx = np.arange(0, len(pts), 7)
    return out, pts[cn_idx], np.tile([0.0, 0.0, 1.0], (len(cn_idx), 1))


def test_cache_records_method_and_morph_uses_it(tmp_path):
    out, cn, nrm = _case(tmp_path)
    build_t_surface_modal_cache(out, cn, k_modes=4, graph_method="laplace_beltrami", lb_boundary="clamped")
    cache = np.load(os.path.join(out, "Control Nodes", "modal_basis_T_surface.npz"))
    assert cache_basis_method(cache) == ("laplace_beltrami", "clamped")

    basis = {"control_nodes": cn.tolist(), "control_normals": nrm.tolist(), "k_modes": 4,
             "normal_project": True, "graph_method": "laplace_beltrami", "lb_boundary": "clamped",
             "amp_alpha": 1.0, "t_patch_scale": 1.0}
    d = control_displacements_from_basis(out, basis, [1.0, 0, 0, 0], log=lambda m: None)
    assert d.shape == cn.shape and np.allclose(d[:, :2], 0.0) and np.abs(d[:, 2]).max() > 0
    # a single mode moves the CNs proportionally to that mode at their nodes
    d2 = control_displacements_from_basis(out, basis, [2.0, 0, 0, 0], log=lambda m: None)
    assert np.allclose(d2, 2 * d)


def test_mismatch_is_refused_not_rebuilt(tmp_path):
    out, cn, nrm = _case(tmp_path)
    build_t_surface_modal_cache(out, cn, k_modes=4, graph_method="mutual_knn")
    cache = np.load(os.path.join(out, "Control Nodes", "modal_basis_T_surface.npz"))
    assert modal_cache_mismatch(cache, "mutual_knn") is None
    assert "laplace_beltrami" in modal_cache_mismatch(cache, "laplace_beltrami")
    with pytest.raises(RuntimeError, match="does not match"):
        getDisplacements(out, control_nodes=cn, normals=nrm, coeffs=[1, 0, 0, 0], k_modes=4,
                         graph_method="laplace_beltrami")


def test_legacy_cache_still_accepted_for_graph(tmp_path):
    out, cn, nrm = _case(tmp_path)
    build_t_surface_modal_cache(out, cn, k_modes=4, graph_method="mutual_knn")
    p = os.path.join(out, "Control Nodes", "modal_basis_T_surface.npz")
    z = dict(np.load(p))
    z.pop("basis_method"); z.pop("lb_boundary")
    np.savez_compressed(p, **z)
    legacy = np.load(p)
    assert cache_basis_method(legacy) == (None, None)
    assert modal_cache_mismatch(legacy, "mutual_knn") is None
    assert modal_cache_mismatch(legacy, "laplace_beltrami") is not None
