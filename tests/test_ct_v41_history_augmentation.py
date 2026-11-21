import hashlib
import json

import numpy as np
import pytest

from models.ct_v41.history_augmentation import (
    DONOR_ARCHIVE, DONOR_GEOMETRY, E57_CHECKPOINT, E57_SOURCE, SCHEMA, TRAIN_DATA, HistoryAugmentation,
)


def track(scene, number):
    return f'kitti_v41/kitti_tracking/train/{scene}/{number}/Car/open3dsot_velodyne_v1'


RECEIVER = track('0000', 4)
OTHERS = [track('0001', 1), track('0002', 2)]


def payload():
    residuals = [[1., 0., 1.], [2., 0., 2.], [3., 0., 3.]]
    counts = [3, 10, 30]
    donors = [dict(track=k, endpoint=4, counts=counts, err_lwh=residuals, dt_us=[100000, 100000])
              for k in OTHERS]
    donors.append(dict(donors[0], track=RECEIVER, err_lwh=[[100., 100., 100.]] * 3))
    return dict(schema=SCHEMA, split='train', checkpoint_sha256=E57_CHECKPOINT,
                teacher_source_sha256=E57_SOURCE, source_sha256=TRAIN_DATA,
                reference_archive_sha256=DONOR_ARCHIVE, donor_geometry_sha256=DONOR_GEOMETRY,
                donor_origin='archived_frames', receivers={k: counts for k in [*OTHERS, RECEIVER]}, donors=donors)


def sample(bank, ids=(0, 1, 2), times=None):
    boxes = {i: np.array([3., 4., 5., yaw]) for i, yaw in enumerate([0., np.pi / 2, np.pi])}
    return bank.sample(RECEIVER, ids, times or {0: 0., 1: .1, 2: .2}, boxes, np.random.default_rng(71))


def test_donor_axes_anchor_coverage_and_digest(tmp_path):
    path = tmp_path / 'history.json'
    path.write_text(json.dumps(payload()), encoding='utf-8')
    bank = HistoryAugmentation.load(path)
    assert bank.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    bank.validate_receivers({k: 3 for k in [*OTHERS, RECEIVER]})
    with pytest.raises(ValueError, match='coverage'):
        bank.validate_receivers({RECEIVER: 3})
    result, reason = sample(bank)
    assert reason == 'full_density'
    np.testing.assert_allclose(result, [[4., 0., -2.], [3., 2., -1.], [0., 0., 0.]], atol=1e-12)
    assert np.array_equal(result[-1], np.zeros(3))
    np.testing.assert_array_equal(sample(bank, (1, 2))[0], result[-2:])
    assert sample(bank, (0,))[1] == 'first_frame'


def test_sparse_fallback_uses_only_other_scenes_and_matching_time():
    specification = payload()
    specification['receivers'][RECEIVER] = [3, 3, 30]
    bank = HistoryAugmentation(specification)
    assert sample(bank)[1] == 'last_density'
    result, reason = sample(bank, times={0: 0., 1: .2, 2: .3})
    assert reason == 'legacy_gap'
    assert not result.any()
    specification['donors'].pop(0)
    result, reason = sample(HistoryAugmentation(specification))
    assert reason == 'legacy_density'  # 同场景即使密度吻合，也不能充当第二条供体轨迹。
    assert not result.any()


def test_rejects_foreign_split_and_invalid_history():
    spec = payload()
    spec['receivers'][track('0017', 4)] = [0, 1, 2]
    with pytest.raises(ValueError, match='training track'):
        HistoryAugmentation(spec)
    with pytest.raises(ValueError, match='consecutive'):
        sample(HistoryAugmentation(payload()), (0, 0, 1))
    with pytest.raises(ValueError, match='timestamps'):
        sample(HistoryAugmentation(payload()), times={0: 1., 1: 0., 2: 2.})


def test_window_and_four_candidate_share_real_seed_then_diverge_by_commit():
    import torch
    from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset
    from models.ct_v41.model import JointTracker
    from tests.helpers.raw_tracking import TinySource

    class Source(TinySource):
        def get_tracklet_key(self, index):
            return RECEIVER

    spec = payload()
    spec['receivers'][RECEIVER] = [3, 10, 30, 30, 30, 30]
    bank = HistoryAugmentation(spec)
    source = Source((6,), dt=.1)
    settings = dict(v41_joint_search=True, ct_engineering_check=True, v41_attention_backend='math',
                    workers=0, batch_size=1)
    tracker = JointTracker(settings).eval()
    datasets, builders = [], []
    for mode in ('window_1448', 'four_candidates'):
        datasets.append(RawEndpointDataset(source, training=True, enable_m=True, joint_search=True,
                                           four_candidates=mode == 'four_candidates'))
        builder = BatchBuilder(dict(settings, v41_history_training=mode))
        builder.history_bank = bank
        builders.append(builder)
    with torch.no_grad():
        for t in (3, 4):
            request = EndpointRequest(3, 0, 1, 3, 5, t)
            batches = [b.prepare(ds.__getitems__([request])) for ds, b in zip(datasets, builders)]
            if t == 3:
                for key in ('point_ids', 'points', 'joint_motion_boxes', 'history_boxes', 'window_seed_offset'):
                    assert torch.equal(batches[0][key], batches[1][key])
            for builder, batch in zip(builders, batches):
                builder.commit(tracker(batch), batch)
            assert bool(builders[0].states) == (t == 3)
            assert not builders[1].states
    assert sum(builders[1].history_counts.values()) == 2 * sum(builders[0].history_counts.values())
