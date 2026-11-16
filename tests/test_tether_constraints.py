import numpy as np
import pytest
from tissuelab.softbody import box_tets
from tissuelab.tissue_pd import Solver


def connection(n=1, fixed=(True, False), attachments=False):
    v = np.array([[0., 0., 0.], [1., 0., 0.]])
    kwargs = dict(attachment_ids=[0, 1], attachment_weights=[10., 10.]) if attachments else {}
    s = Solver(v, fixed=fixed, tether_nodes=np.tile([0, 1], (n, 1)),
               tether_coefficients=np.tile([-1., 1.], (n, 1)),
               tether_stiffness=np.full(n, 100. / n), tether_slack=np.full(n, 1.1), **kwargs)
    return v, s


def test_tether_slack_tension_and_rigid_motion():
    v, s = connection()
    for length in (0.5, 1., 1.1, 1.2):
        x = v.copy()
        x[1, 0] = length
        force = s.L @ x - s._rhs(x, None)
        expected = 100 * max(0, length - 1.1)
        assert force[1, 0] == pytest.approx(expected, abs=1e-12)
        assert np.allclose(force.sum(0), 0)
    rotated = np.array([[5., 5., 5.], [5., 6., 5.]])
    assert np.allclose(s.L @ rotated - s._rhs(rotated, None), 0)


def test_attachment_density_does_not_multiply_stiffness():
    v, single = connection()
    _, many = connection(n=16)
    assert np.allclose(single.L.toarray(), many.L.toarray())
    force = np.zeros_like(v)
    force[1, 0] = 10
    a, _, _ = single.solve(v, f_ext=force, iters=100)
    b, _, _ = many.solve(v, f_ext=force, iters=100)
    assert a[1, 0] == pytest.approx(1.2)
    assert np.allclose(a, b)


def test_deep_endpoint_moves_and_force_transfers():
    v, s = connection(fixed=(False, False), attachments=True)
    target = v.copy()
    target[1, 0] = 1.4
    x, _, _ = s.solve(v, attachment_targets=target, iters=1000, tol=1e-12)
    # Two 10 N/m grips in series with a 100 N/m tether, after taking up 0.1 m slack.
    tension = (0.4 - 0.1) / (1/10 + 1/100 + 1/10)
    assert x[0, 0] == pytest.approx(tension / 10, abs=1e-8)
    assert x[1, 0] == pytest.approx(1.4 - tension / 10, abs=1e-8)
    assert x[0, 0] > 0


def test_bad_tether_coefficients_are_rejected():
    with pytest.raises(ValueError, match="tether"):
        Solver(np.array([[0., 0., 0.], [1., 0., 0.]]), fixed=[True, False],
               tether_nodes=[[0, 1]], tether_coefficients=[[1., 1.]],
               tether_stiffness=[100.], tether_slack=[1.1])


def test_zero_strength_preserves_existing_solver():
    x, tet = box_tets(1, 1, 1, (1., 1., 1.))
    fixed = x[:, 2] == 0
    args = dict(tets=tet, mu_t=np.ones(len(tet)), lam_t=np.ones(len(tet)), fixed=fixed)
    base = Solver(x, **args)
    extra = Solver(x, **args, tether_nodes=[[0, len(x)-1]], tether_coefficients=[[-1., 1.]],
                   tether_stiffness=[0.], tether_slack=[2.])
    force = np.zeros_like(x)
    force[~fixed, 2] = .01
    a, _, _ = base.solve(x, f_ext=force, iters=20)
    b, _, _ = extra.solve(x, f_ext=force, iters=20)
    assert np.array_equal(a, b)


def test_embedded_endpoints_with_shared_vertices_scatter_correctly():
    x = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    s = Solver(x, fixed=[True, True, False], tether_nodes=[[0, 1, 1, 2]],
               tether_coefficients=[[-.5, -.5, .5, .5]],
               tether_stiffness=[100.], tether_slack=[1.1])
    x[2, 0] = 2.4
    force = s.L @ x - s._rhs(x, None)
    assert np.allclose(force[:, 0], [-5., 0., 5.])
    assert np.allclose(force.sum(0), 0)
