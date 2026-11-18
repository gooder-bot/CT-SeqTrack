"""V41 几何共享保持两套高度、原 q0 监督及最终提交的独立校验。"""
import pickle
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from models.ct_v41.attention import ValidityMask, _device_supports_bf16
from models.ct_v41.base_losses import oriented_iou_labels
from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset
from models.ct_v41.input import online_batch
from models.ct_v41.local_observation import LocalObservationReader
from models.ct_v41.losses import _supervision_targets
from models.ct_v41.model import JointTracker
from models.ct_v41.observation import B0Observation, _pooled_tokens, unique_valid_mask
from models.ct_v41.verification import (
    LocalBoxReader, _remember_final_support, build_current_pool, current_candidate_support,
    final_candidate_support, selected_commit_support,
)
from tests.helpers.raw_tracking import TinySource
from tests.test_ct_v41_recovery import config, simple_output, values
from tests.test_ct_v41_summary import measurement_output
from utils.batch_materialization import attach_materialization_cache
from utils.tracking_metrics import LocalYawBox, box_metrics, paired_box_metrics_3d


@pytest.fixture(autouse=True)
def local_cpu_rng():
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1741)
            yield
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('height,offset', [(4., 0.), (.75, .125), (1., 1.), (2., 9.)])
def test_two_vertical_conventions_share_only_the_footprint(height, offset):
    target = LocalYawBox([0., 0., 0., 0.], [2., 4., 2.])
    prediction = LocalYawBox([.125, -.125, offset, .35], [1., 3., height])
    exact, compat, distance = paired_box_metrics_3d(prediction, target, up_axis=(0, 0, 1))
    assert (exact, distance) == box_metrics(prediction, target, up_axis=(0, 0, 1), mode='geometry_exact')
    assert (compat, distance) == box_metrics(prediction, target, up_axis=(0, 0, 1), mode='benchmark_compat')


def test_exact_and_compat_keep_distinct_vertical_origins():
    target = LocalYawBox([0., 0., 0., 0.], [2., 4., 2.])
    prediction = LocalYawBox([0., 0., -1., 0.], [2., 4., 4.])
    assert paired_box_metrics_3d(prediction, target, up_axis=(0, 0, 1)) == (.5, .2, 1.)


@pytest.mark.parametrize('explicit_target_size', [True, False])
def test_shared_q0_preserves_fallback_size_clamp_and_separate_decoder_box(explicit_target_size):
    batch = values(b=3)
    batch['box_size'][1, 0] = .0002
    if explicit_target_size:
        batch['target_box_size'] = batch['box_size'].clone()
    else:
        batch.pop('target_box_size', None)
    output = simple_output(batch, q0_real=False)
    batch['target_box'] = output.decoder.q0_box.clone()
    output.decoder.q0_box[2, 0] += .125
    _, shared = _supervision_targets(batch, output, include_q0=True)
    expected = oriented_iou_labels(output.decoder.q0_box[:, None], batch['target_box'],
        batch['box_size'].clamp_min(1e-3), batch.get('target_box_size'))
    assert torch.equal(shared, expected)


def test_attention_metadata_matches_dense_outer_product_including_empty_sides():
    query = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 0]], dtype=torch.bool)
    key = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    for left, right in ((query, key), (query, query), (key[:0], key[:0])):
        expected = ValidityMask.from_allowed(left[:, :, None] & right[:, None, :])
        actual = ValidityMask.from_valid(left, right)
        for field in vars(expected):
            first, second = getattr(actual, field), getattr(expected, field)
            assert torch.equal(first, second) if torch.is_tensor(first) else first == second


def test_unique_mask_preserves_first_present_maximum_raw_id():
    maximum = torch.iinfo(torch.long).max
    ids = torch.tensor([[maximum, maximum, -1, maximum, 7, 7]])
    valid = torch.tensor([[False, True, True, True, True, True]])
    assert unique_valid_mask(valid, ids).tolist() == [[False, True, False, False, True, False]]


def test_bf16_capability_is_checked_separately_on_each_device(monkeypatch):
    visited = []
    @contextmanager
    def device(index):
        visited.append(index)
        yield
    monkeypatch.setattr(torch.cuda, 'device', device)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: visited[-1] == 0)
    _device_supports_bf16.cache_clear()
    try:
        assert [_device_supports_bf16(i) for i in (0, 1, 0, 1)] == [True, False, True, False]
        assert visited == [0, 1]
    finally:
        _device_supports_bf16.cache_clear()


def test_b0_supplies_the_exact_mixed_frame_token_count(monkeypatch):
    data = values(b=2, n=7)
    data['point_valid'][0, 0] = False
    data['point_valid'][0, 1, 1:] = False
    data['point_ids'][0, 2, 2:] = data['point_ids'][0, 2, 1].item()
    data['history_valid'][1] = False
    net = B0Observation(token_count=3).train()
    net.execution_profile = 'optimized_fp32'
    original = net.feature_pointnet.forward
    counts = []
    def checked(value, valid, **options):
        assert options['valid_count'] == int(valid.sum())
        counts.append(options['token_valid_count'])
        assert options['token_valid_count'] == int(valid.sum(-1).clamp_max(3).sum())
        return original(value, valid, **options)
    monkeypatch.setattr(net.feature_pointnet, 'forward', checked)
    net(data)
    assert counts == [9]


@pytest.mark.parametrize('token_count', [1, 3, 10])
def test_ragged_pool_keeps_token_boundaries_and_first_max_gradient(token_count):
    valid = torch.tensor([[1, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    value = torch.randint(-1, 2, (3, 4, 8)).float().requires_grad_()
    actual, token_valid = _pooled_tokens(value, valid, token_count)
    expected = torch.zeros_like(actual)
    for row, mask in enumerate(valid):
        slots = mask.nonzero().flatten()
        count = min(len(slots), token_count)
        assert token_valid[row].tolist() == [True] * count + [False] * (token_count - count)
        for token in range(count):
            members = slots[token * len(slots) // count:(token + 1) * len(slots) // count]
            expected[row, :, token] = value[row, :, members].max(-1).values
    assert torch.equal(actual, expected)
    weights = torch.randn_like(actual)
    assert torch.equal(torch.autograd.grad(actual, value, weights)[0],
                       torch.autograd.grad(expected, value, weights)[0])


def test_canonical_observation_checks_new_values_after_alias_mutation():
    raw = attach_materialization_cache(values(), enabled=True)
    raw['point_valid'].fill_(True)
    raw['point_valid'][:, :, 0] = False
    raw['points'][:, :, 0] = torch.nan
    clean = online_batch(raw)
    net = B0Observation(token_count=4).eval()
    with torch.no_grad():
        cached = net(raw)
        direct = net({k: v for k, v in raw.items() if k != '_ct_materialization_cache'})
    assert torch.equal(cached.source_tokens, direct.source_tokens)
    assert torch.equal(cached.coarse_box, direct.coarse_box)
    assert torch.equal(cached.source_valid, direct.source_valid)
    clean['points'][0, -1, 1, 0] = torch.nan
    with pytest.raises(ValueError, match='finite'):
        net(clean)


def test_equal_corner_distances_select_first_sixteen_raw_ids_with_split_gradient_ownership():
    data = values(n=24)
    data['points'].zero_()
    ids = torch.arange(23, -1, -1, dtype=torch.int32)
    data['point_ids'] = ids.expand(1, 4, 24).clone()
    features = torch.randn(1, 4, 24, 64, requires_grad=True)
    obs = SimpleNamespace(point_features=features, foreground_probability=torch.ones(1, 4, 24))
    pool = build_current_pool(obs, data, live_features=True)
    boxes = torch.zeros(1, 1, 4)
    allowed = torch.ones(1, 1, 24, dtype=torch.bool)
    verifier = LocalBoxReader()
    verified, read = verifier(pool, boxes, allowed, data['box_size'])
    assert torch.equal(read['local_point_mask'][0, 0], ids < 16)
    assert torch.autograd.grad(verified.sum(), features, allow_unused=True)[0] is None
    decoder = LocalObservationReader()
    with torch.no_grad():
        decoder.output.weight.fill_(.01)
    values_read, _ = decoder(obs, data, boxes, allowed.any(-1), pool=pool)
    grads, = torch.autograd.grad(values_read.sum(), features)
    assert grads[:, -1, ids < 16].abs().sum() > 0
    assert not grads[:, -1, ids >= 16].any()


def test_commit_validates_shared_support_independently_and_invalidates_on_edits():
    raw = attach_materialization_cache(values(), enabled=True)
    raw['points'].zero_()
    raw['point_valid'].fill_(True)
    batch = online_batch(raw)
    boxes = torch.zeros(1, 4, 4)
    obs = SimpleNamespace(foreground_probability=torch.ones_like(raw['point_valid'], dtype=torch.float32))
    output = SimpleNamespace(candidates=SimpleNamespace(boxes=boxes), observation=obs)
    support = current_candidate_support(obs, batch, boxes)
    _remember_final_support(obs, batch, boxes, support['mask'], support['count'])
    selected = torch.tensor([0])
    assert final_candidate_support(obs, raw, boxes)['mask'] is support['mask']
    proposed = selected_commit_support(output, raw, selected)
    assert proposed.any()
    proposed.zero_()
    assert selected_commit_support(output, raw, selected).any()
    boxes[:, 0, 0] = 100.
    assert not selected_commit_support(output, raw, selected).any()
    refreshed = current_candidate_support(obs, batch, boxes)
    _remember_final_support(obs, batch, boxes, refreshed['mask'], refreshed['count'])
    obs.foreground_probability.zero_()
    assert not final_candidate_support(obs, raw, boxes)['mask'].any()


def test_corrupted_second_proposal_cannot_commit_first_row_from_cached_membership():
    from tests.test_ct_v41_training import cfg
    builder = BatchBuilder(cfg())
    raw = RawEndpointDataset(TinySource((6, 6)))
    rows = [raw[EndpointRequest(0, track, 5, 1, 6, 1)] for track in (0, 1)]
    batch = attach_materialization_cache(builder.prepare(rows), enabled=True)
    prepared = online_batch(batch)
    output = measurement_output(batch)
    boxes = output.candidates.boxes
    canonical = current_candidate_support(output.observation, prepared, boxes)
    _remember_final_support(output.observation, prepared, boxes, canonical['mask'], canonical['count'])
    assert final_candidate_support(output.observation, batch, boxes)['mask'] is canonical['mask']
    before = pickle.dumps(builder.states)
    output.commit_decision.support_mask[1, 0].logical_not_()
    with pytest.raises(ValueError, match='commit support'):
        builder.commit(output, batch)
    assert pickle.dumps(builder.states) == before


def test_summary_omits_final_local_scatter_but_preserves_actual_recovery_read_and_labels():
    data = values()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = JointTracker(config()).eval()
        full = model.forward_aux(data)
        summary = model.forward_aux(data, include_local_diagnostics=False)
        assert full.verification.local_point_mask is not None
        assert summary.verification.local_point_mask is None
        assert summary.decoder.local_delta_norm is None
        assert torch.equal(full.recovery.point_mask, summary.recovery.point_mask)
        assert torch.equal(full.verification.logits, summary.verification.logits)
        assert torch.equal(full.commit_decision.support_mask, summary.commit_decision.support_mask)
        data.pop('bc_targets')
        assert torch.equal(model.compute_aux_losses(data, full)['loss_total'],
                           model.compute_aux_losses(data, summary)['loss_total'])
