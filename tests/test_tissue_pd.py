"""Projective-dynamics bake solver: same analytic checks as the XPBD solver, converged."""
import time

import numpy as np

from tissuelab import softbody as sb
from tissuelab import tissue_pd as pd


def bar(E, nu=0.45, rho=1000.0, L=0.12, a=0.02, nz=12):
    v, t = sb.box_tets(2, 2, nz, (a, a, L))
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    fixed = v[:, 2] > L - 1e-9
    s = pd.Solver(v, t, np.full(len(t), mu), np.full(len(t), lam), fixed=fixed)
    f = np.zeros_like(v)
    f[:, 2] = -9.81 * pd.tet_masses(v, t, np.full(len(t), rho))
    return v, s, f


def test_rest_is_stress_free():
    v, s, _ = bar(60e3)
    x, _, d = s.solve(v, iters=5)
    assert np.abs(x - v).max() < 1e-12


def test_hanging_bar_and_stiffness_scaling():
    exp = 1000 * 9.81 * 0.12 ** 2 / (2 * 60e3)
    out = []
    t0 = time.time()
    for E in (60e3, 240e3):
        v, s, f = bar(E)
        x, it, d = s.solve(v, f_ext=f, iters=500, tol=1e-10)
        out.append(-x[v[:, 2] < 1e-9, 2].mean())
    assert time.time() - t0 < 30
    assert abs(out[0] - exp) / exp < 0.10, (out[0], exp)          # XPBD converged to −6.7 %
    assert 3.8 < out[0] / out[1] < 4.2


def test_matches_converged_xpbd():
    v, s, f = bar(60e3)
    x, _, _ = s.solve(v, f_ext=f, iters=500, tol=1e-10)
    # Converged XPBD (dt 1.25e-4) gave 1.098 mm for this bar; PD must agree within 3 %.
    assert abs(-x[v[:, 2] < 1e-9, 2].mean() - 1.098e-3) / 1.098e-3 < 0.03


def test_fibre_membrane_is_stiffer_along_fibres():
    # A square membrane (1 cm, 0.5 mm thick) with fibres along x, stretched 10 % in x or in y.
    n = 6
    xs = np.linspace(0, 0.01, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    v = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], -1)
    idx = np.arange(len(v)).reshape(n + 1, n + 1)
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            tris += [(a, b, c), (a, c, d)]
    tris = np.array(tris)
    e1 = v[tris[:, 1]] - v[tris[:, 0]]
    u = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    fib = np.stack([u[:, 0], -u[:, 1]], -1)            # Rest-frame components of world +x
    k = len(tris)

    def pull(axis):
        fixed = np.zeros(len(v), bool)
        lo, hi = v[:, axis] < 1e-9, v[:, axis] > 0.01 - 1e-9
        fixed[lo | hi] = True
        s = pd.Solver(v, tris=tris, mu_m=np.full(k, 1e5), lam_m=np.full(k, 1e5), thick_m=np.full(k, 5e-4),
                      fibre_m=fib, kf_m=np.full(k, 1e7), fixed=fixed)
        tgt = np.full_like(v, np.nan)
        tgt[lo] = v[lo]
        tgt[hi] = v[hi] + np.eye(3)[axis] * 0.001
        x, _, _ = s.solve(v, targets=tgt, iters=300, tol=1e-12)
        # Reaction: energy grows with stiffness; use the summed internal force on the moving edge
        return s.L @ x - s._rhs(x, None)
    fx, fy = pull(0), pull(1)
    assert np.abs(fx).sum() > 5 * np.abs(fy).sum()


def test_chebyshev_acceleration_converges_faster_to_the_same_answer():
    v, s, f = bar(60e3)
    ref, _, _ = s.solve(v, f_ext=f, iters=3000, tol=1e-12)
    tip = lambda x: -x[v[:, 2] < 1e-9, 2].mean()            # noqa: E731
    plain, _, _ = s.solve(v, f_ext=f, iters=100, tol=0)
    fast, _, _ = s.solve(v, f_ext=f, iters=100, tol=0, rho=0.98)
    e_plain = abs(tip(plain) - tip(ref)) / tip(ref)
    e_fast = abs(tip(fast) - tip(ref)) / tip(ref)
    assert e_fast < 0.001 and e_fast < e_plain / 5


def membrane_solver(**kwargs):
    v = np.array([[0., 0., 0.], [0.01, 0., 0.], [0., 0.01, 0.]])
    return v, pd.Solver(v, tris=np.array([[0, 1, 2]]), mu_m=[1000.], lam_m=[1000.],
                        thick_m=[0.002], fixed=[True, False, True], **kwargs)


def test_strain_stiffening_has_no_force_below_onset_and_rises_above_it():
    v, _ = membrane_solver()
    for fast in (False, True):
        _, stiff = membrane_solver(stretch_limit_m=1.2, stretch_stiffness_m=1e6, fast=fast)
        for stretch in (1.0, 1.1, 1.2, 1.3, 1.4):
            x = v.copy()
            x[:, 0] *= stretch
            # Isolate the additional term from the same projection path.
            _, reference = membrane_solver(fast=fast)
            extra = (stiff.L @ x - stiff._rhs(x, None)) - (reference.L @ x - reference._rhs(x, None))
            # E = 0.5*k*t*A*max(stretch-limit, 0)^2, differentiated w.r.t. vertex x.
            expected = 1e6 * 0.002 * 0.00005 * max(stretch - 1.2, 0) / 0.01
            assert np.isclose(extra[1, 0], expected, atol=1e-10)
            assert np.allclose(extra.sum(0), 0, atol=1e-10)
        # Rigid rotation must not generate strain-limiting forces.
        R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        x = v @ R.T
        assert np.allclose(stiff.L @ x - stiff._rhs(x, None), 0, atol=1e-10)


def test_attachment_equilibrium_balances_tissue_force_and_allows_lag():
    for fast in (False, True):
        v, s = membrane_solver(attachment_ids=[1], attachment_weights=[10.], fast=fast)
        target = v[[1]] + [0.005, 0., 0.]
        x, _, _ = s.solve(v, attachment_targets=target, iters=2000, tol=1e-12)
        force = 10 * (target[0] - x[1])
        tissue = s.L @ x - s._rhs(x, None)
        tissue[1] -= 10 * x[1]  # remove attachment diagonal from internal tissue force
        assert v[1, 0] < x[1, 0] < target[0, 0]
        assert np.allclose(tissue[1], force, atol=1e-9)
        assert force[0] > 0
        # Releasing the attachment restores the stress-free configuration.
        _, released = membrane_solver(fast=fast)
        relaxed, _, _ = released.solve(x, iters=2000, tol=1e-12)
        assert np.allclose(relaxed, v, atol=1e-8)


def test_attachment_rejects_missing_targets_and_fixed_vertices():
    import pytest
    v, s = membrane_solver(attachment_ids=[1], attachment_weights=[10.])
    with pytest.raises(ValueError, match="attachment_targets"):
        s.solve(v)
    with pytest.raises(ValueError, match="unique free"):
        membrane_solver(attachment_ids=[0], attachment_weights=[10.])
    with pytest.raises(ValueError):
        membrane_solver(stretch_limit_m=0.9, stretch_stiffness_m=1e6)


def test_disabled_stiffening_preserves_legacy_solve():
    v, base = membrane_solver()
    _, disabled = membrane_solver(stretch_limit_m=1.2, stretch_stiffness_m=0)
    forces = np.zeros_like(v)
    forces[1, 0] = 0.01
    a, _, _ = base.solve(v, f_ext=forces, iters=100)
    b, _, _ = disabled.solve(v, f_ext=forces, iters=100)
    assert np.array_equal(a, b)


def test_stiffening_limits_extension_under_the_same_load():
    v, base = membrane_solver()
    _, stiff = membrane_solver(stretch_limit_m=1.2, stretch_stiffness_m=1e6)
    force = np.zeros_like(v)
    force[1, 0] = 0.1
    a, _, _ = base.solve(v, f_ext=force, iters=2000, tol=1e-12)
    b, _, _ = stiff.solve(v, f_ext=force, iters=2000, tol=1e-12)
    assert 1.2 < b[1, 0] / v[1, 0] < 1.25
    assert a[1, 0] - v[1, 0] > 5 * (b[1, 0] - v[1, 0])


def test_hard_limit_stops_repeated_pull_and_allows_return():
    v, s = membrane_solver(attachment_ids=[1], attachment_weights=[1000.])
    target = v[[1]] + [0.05, 0., 0.]
    x, _, _ = s.solve(v, attachment_targets=target, max_stretch=1.3, iters=30)
    assert s.limit_reached
    assert 1.299 < x[1, 0] / v[1, 0] <= 1.3
    stopped = x.copy()
    for _ in range(10):
        x, _, _ = s.solve(x, attachment_targets=target * 10, max_stretch=1.3, iters=3)
        assert s.limit_reached
        assert s._within_stretch_limit(s._deformation_gradients(x), 1.3)
    assert np.max(np.abs(x - stopped)) < 1e-6
    x, _, _ = s.solve(x, attachment_targets=v[[1]], max_stretch=1.3, iters=500)
    assert not s.limit_reached
    assert np.allclose(x, v, atol=1e-6)


def test_hard_limit_guards_volumes_and_large_inverting_steps():
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    for fast in (False, True):
        s = pd.Solver(v, tets=[[0, 1, 2, 3]], mu_t=[1.], lam_t=[1.],
                      fixed=[True, True, True, False], attachment_ids=[3],
                      attachment_weights=[100.], fast=fast)
        x, _, _ = s.solve(v, attachment_targets=[[0., 0., 100.]], max_stretch=1.4)
        assert s.limit_reached and 1.39 < x[3, 2] <= 1.4
        x, _, _ = s.solve(v, attachment_targets=[[0., 0., -100.]], max_stretch=1.4, iters=1)
        assert s.limit_reached and x[3, 2] > 0
        assert s._within_stretch_limit(s._deformation_gradients(x), 1.4)


def test_hard_limit_rejects_invalid_start_and_threshold():
    import pytest
    v, s = membrane_solver()
    for limit in (1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="max_stretch"):
            s.solve(v, max_stretch=limit)
    x = v * 2
    with pytest.raises(ValueError, match="initial state"):
        s.solve(x, max_stretch=1.5)


def test_hard_limit_measures_strain_not_world_displacement():
    v, s = membrane_solver()
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = v @ R.T + [10., 20., 30.]
    assert s._within_stretch_limit(s._deformation_gradients(moved), 1.1)
