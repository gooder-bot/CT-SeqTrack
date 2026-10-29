"""V40 分块 CPU 标签对冻结 v38 标量枚举的几何/计数/tie 等价。"""
import numpy as np
import pytest

from models.ct_v38.acquisition_supervision import band_grid_target as reference_target
from models.ct_v38.acquisition_supervision import build_dual_support as reference_support
from models.ct_v40.acquisition_supervision import band_grid_target, build_dual_support


def geometry(yaw=.37):
    return dict(b0_box=np.asarray([.3, -.4, .2, yaw]), box_size=np.asarray([4., 2., 1.6]),
        prior_center=np.asarray([2., .7, .1]), recovery_center=np.asarray([-4., 2., .3]),
        support_yaw=-.23, recovery_yaw=.81, crop_scale=1.25, crop_offset=2.)


@pytest.mark.parametrize('u', [(0., 0.), (.125, .875), (1., 1.)])
def test_shared_geometry_preserves_exact_float64_regions(u):
    expected, actual = reference_support(u=u, **geometry()), build_dual_support(u=u, **geometry())
    for name in ('local', 'recovery'):
        for field in ('center', 'half'):
            np.testing.assert_array_equal(actual[name][field], expected[name][field])
        assert actual[name]['yaw'] == expected[name]['yaw']
    np.testing.assert_array_equal(actual['u'], expected['u'])


def compare(points, ids, target, excluded, **kwargs):
    expected = reference_target(points, ids, target_mask=target, b0_raw_ids=excluded, **kwargs)
    actual = band_grid_target(points, ids, target_mask=target, b0_raw_ids=excluded, **kwargs)
    np.testing.assert_array_equal(actual.pop('acquisition_target'), expected.pop('acquisition_target'))
    assert actual == expected


@pytest.mark.parametrize('seed,n', [(0, 0), (1, 300), (2, 5000), (3, 17000)])
def test_chunked_grid_matches_counts_and_tie_with_duplicates_exclusion(seed, n):
    rng = np.random.default_rng(seed)
    points = rng.uniform([-20., -15., -3.], [20., 15., 3.], (n, 3))
    ids = np.arange(n, dtype=np.int64)
    if n > 20:
        ids[1:7] = ids[0]
        ids[10] = -1
        points[15, 0] = np.nan
    target = rng.random(n) < .25
    compare(points, ids, target, ids[::7], **geometry())


def test_boundary_points_and_all_background_preserve_scalar_grid_choice():
    g = geometry(0.)
    points = []
    for u in ((0., 0.), (.5, .5), (1., 1.)):
        for region in reference_support(u=u, **g).values():
            if not isinstance(region, dict):
                continue
            c, s = np.cos(region['yaw']), np.sin(region['yaw'])
            rotation = np.asarray([[c, -s], [s, c]])
            for axis in (0, 1):
                for epsilon in (-1e-6, 0., 1e-7, 2e-7):
                    delta = np.zeros(3)
                    delta[axis] = region['half'][axis] + epsilon
                    delta[:2] = delta[:2] @ rotation.T
                    points.append(region['center'] + delta)
    points = np.asarray(points)
    ids = np.arange(len(points))
    compare(points, ids, ids % 3 == 0, ids[::11], **g)
    compare(points, ids, np.zeros(len(points), bool), np.empty(0, dtype=np.int64), **g)
