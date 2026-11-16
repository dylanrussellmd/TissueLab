"""Public synthetic fixture; no CT scan or SurgVid dependency."""
import json

import numpy as np
import pytest

from tissuelab.softbody import box_tets
from tissuelab.blender_addon.engine import Session


def write_wall(path, version=(1, 0)):
    x, t = box_tets(2, 2, 2, (.04, .04, .04))
    all_faces = t[:, [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(all_faces, 1), axis=0, return_index=True, return_counts=True)
    boundary = first[counts == 1]
    faces = all_faces[boundary]
    roles = np.all(x[faces, 2] == 0, axis=1).astype(np.int8)
    embed = np.array([np.flatnonzero((t == i).any(1))[0] for i in range(len(x))])
    weights = (t[embed] == np.arange(len(x))[:, None]).astype(float)
    data = dict(info=np.array(json.dumps({'case': 'synthetic'})), x=x, tets=t,
                tissue_names=np.array(['muscle']), tet_tissue=np.zeros(len(t), int),
                role_names=np.array(['skin', 'scan_edge']), tris=faces, tri_role=roles,
                tri_tet=boundary // 4, mu_t=np.full(len(t), 1000.), lam_t=np.full(len(t), 1000.),
                mem_tris=np.empty((0, 3), int), mu_m=[], lam_m=[], th_m=[], fib_m=np.empty((0, 2)),
                kf_m=[], fixed=x[:, 2] == 0, surface_names=np.array(['skin']),
                skin__verts=x, skin__faces=faces, skin__tet=embed, skin__w=weights,
                skin__offset=np.zeros_like(x))
    if version is not None:
        data['format_version'] = np.array(version, np.int16)
    np.savez(path, **data)


@pytest.mark.parametrize('version', [(1, 0), None])
def test_supported_and_legacy_wall_load_and_move(tmp_path, version):
    path = tmp_path/'wall.npz'
    write_wall(path, version)
    s = Session(str(path))
    assert s.format_version == ((1, 0) if version else (0, 0))
    M = np.eye(4); M[:3, 3] = [.02, .02, .04]
    assert s.add_handle('grip', M[:3, 3], .015, M) > 0
    M[2, 3] += .003
    for _ in range(20):
        s.step({'grip': M}, 3)
    assert 0 < s.stats()['max_displacement_m'] < .003
    assert s.stats()['inverted'] == 0
    s.reset()
    assert np.array_equal(s.x, s.x0)


@pytest.mark.parametrize('version', [[2, 0], [1, 1], [1], [1.0, 0.0]])
def test_unknown_format_fails_before_reading_geometry(tmp_path, version):
    path = tmp_path/'future.npz'
    np.savez(path, format_version=np.asarray(version))
    with pytest.raises(ValueError, match='Unsupported body-wall format'):
        Session(str(path))


def tether_wall(path):
    write_wall(path)
    with np.load(path) as f:
        a = dict(f)
    low, high = np.argmin(a['x'][:, 2]), np.argmax(a['x'][:, 2])
    ids = [high, low]
    tet_ids = [np.flatnonzero((a['tets'] == i).any(1))[0] for i in ids]
    a['tether_tets'] = np.array([tet_ids])
    a['tether_weights'] = (a['tets'][tet_ids] == np.array(ids)[:, None])[None].astype(float)
    a['tether_area_m2'] = np.array([.001])
    return a


def test_tether_diagnostics_distinguish_tension_disabled_and_reset(tmp_path):
    path = tmp_path/'tether.npz'
    np.savez(path, **tether_wall(path))
    for enabled in (False, True):
        s = Session(str(path), tethering=enabled)
        assert s.tether_stats()['taut'] == 0
        s.x[:, 2] *= 1.2
        stats = s.tether_stats()
        assert stats['taut'] == int(enabled)
        assert stats['max_tension_n'] == pytest.approx(1. if enabled else 0.)
        assert stats['energy_j'] == pytest.approx(.002 if enabled else 0.)
        assert not s.limit_reached
        s.reset()
        assert s.tether_stats()['max_tension_n'] == 0


@pytest.mark.parametrize('field,value', [
    ('tether_tets', [[-1, 0]]), ('tether_tets', [[99999, 0]]),
    ('tether_tets', [[0., 1.]]), ('tether_weights', np.full((1, 2, 4), .5)),
    ('tether_weights', np.full((1, 2, 4), np.nan)),
    ('tether_area_m2', [-.001]), ('tether_area_m2', [np.inf]),
    ('tether_area_m2', []),
])
def test_invalid_tether_data_rejected_even_when_disabled(tmp_path, field, value):
    path = tmp_path/'invalid.npz'
    a = tether_wall(path)
    a[field] = np.asarray(value)
    np.savez(path, **a)
    with pytest.raises(ValueError, match='invalid tether'):
        Session(str(path), tethering=False)
