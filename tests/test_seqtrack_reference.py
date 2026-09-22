"""独立原 SeqTrack 的 CPU 数学、teacher 分母和递归评测合同。"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.seqtrack_reference import ReferenceTracker, BatchBuilder, build_loaders, protocol_identity
from models.seqtrack_reference.data import (ReferenceBatchSampler, ReferenceDataset,
    ReferenceRequest, evaluation_input, normalize_frame)
from models.seqtrack_reference.protocol import reference_config
from models.seqtrack_reference._vendor import points as points_utils
from models.seqtrack_reference._vendor.misc import (get_tensor_corners_batch, create_corner_timestamps,
    get_history_frame_ids_and_masks, get_last_n_bounding_boxes,
    generate_timestamp_prev_list, resolve_kitti_hv_search_offset)
from models.seqtrack_reference._vendor import spatial
from models.seqtrack_reference._vendor.attn.Models import Seq2SeqFormer
from models.seqtrack_reference.numerics import ReferenceAdaptiveMaxPool1d, original_weighted_segmentation_loss


ROOT = Path(__file__).resolve().parents[1] / 'models' / 'seqtrack_reference'


@pytest.fixture(scope='module', autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyReferenceSource:
    def __init__(self, lengths=(4, 3, 1), invalid_first_track=False):
        self.lengths = lengths
        self.invalid_first_track = invalid_first_track

    def get_num_tracklets(self):
        return len(self.lengths)

    def get_num_frames_tracklet(self, index):
        return self.lengths[index]

    def get_tracklet_key(self, index):
        return 'ref/' + str(index)

    def get_frames(self, track, frame_ids):
        rows = []
        for frame in frame_ids:
            center = np.array([frame * .2, track * 20., 0.])
            cloud = np.array([[0., 0., 0.], [.2, .1, 0.], [-.2, -.1, 0.],
                              [0., -.2, .1], [2.3, 0., 0.], [5., 0., 0.]]) + center
            if self.invalid_first_track and track == 0:
                cloud += 100.
            rows.append(dict(pc=cloud, point_ids=np.arange(len(cloud)),
                **{'3d_bbox': np.r_[center, 0.]}, box_size=np.array([4., 2., 2.]),
                timestamp=10. + frame * .5, scene_id='reference_test'))
        return rows

    def get_frame_metadata(self, track, frame):
        return self.get_frames(track, [frame])[0]


def original_class():
    """执行保存的原类 AST；只替换 Lightning 基类与 CPU loss device。"""
    class Base(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config

        @property
        def device(self):
            return next(self.parameters()).device

    namespace = dict(torch=torch, nn=nn, F=F, Base=Base,
        points_utils=points_utils, Seq2SeqFormer=Seq2SeqFormer,
        get_tensor_corners_batch=get_tensor_corners_batch,
        create_corner_timestamps=create_corner_timestamps,
        Accuracy=lambda **kwargs: nn.Identity())
    tree = ast.parse((ROOT / '_vendor' / 'pointnet.original.txt').read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name in ('MiniPointNet', 'SegPointNet', 'FeaturePointNet')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-pointnet>', 'exec'), namespace)
    source = (ROOT / '_vendor' / 'seqtrack3d.original.txt').read_text(encoding='utf-8')
    source = source.replace('torch.tensor([0.5, 2.0]).cuda()', 'seg_logits.new_tensor([0.5, 2.0])')
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.name = 'Original'
    cls.bases = [ast.Name(id='Base', ctx=ast.Load())]
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in ('__init__', 'forward', 'compute_loss')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 '<frozen-seqtrack>', 'exec'), namespace)
    return namespace['Original']


def synthetic_batch(batch=2, points=8):
    return dict(points=torch.randn(batch, 4 * points, 5),
        candidate_bc=torch.randn(batch, 4 * points, 9),
        valid_mask=torch.ones(batch, 3, dtype=torch.long), ref_boxs=torch.randn(batch, 3, 4),
        bbox_size=torch.tensor([[2., 4., 1.5]]).expand(batch, -1),
        seg_label=torch.randint(0, 2, (batch, 4 * points)), box_label=torch.randn(batch, 4),
        box_label_prev=torch.randn(batch, 3, 4), motion_label=torch.randn(batch, 3, 4),
        motion_state_label=torch.tensor([[0, 1, 1], [1, 1, 1]])[:batch],
        prev_bc=torch.randn(batch, 3, points, 9), this_bc=torch.randn(batch, points, 9))


def test_original_forward_loss_grad_bn_and_adam_are_mathematically_equal():
    torch.manual_seed(7)
    actual = ReferenceTracker({}).train()
    expected = original_class()(reference_config()).train()
    expected.load_state_dict(actual.state_dict(), strict=True)
    batch = synthetic_batch()
    rng = torch.get_rng_state()
    actual_output = actual(batch)
    actual_losses = actual.compute_losses(batch, actual_output)
    actual_losses['loss_total'].backward()
    torch.set_rng_state(rng)
    expected_output = expected(batch)
    expected_losses = expected.compute_loss(batch, expected_output)
    expected_losses['loss_total'].backward()
    for key in expected_output:
        torch.testing.assert_close(actual_output[key], expected_output[key], rtol=0, atol=0)
    for key in expected_losses:
        # flat reduction=none 的最终 sum 与原空间 NLL mean 归约顺序不同。
        torch.testing.assert_close(actual_losses[key], expected_losses[key], rtol=2e-6, atol=2e-7)
    for name, parameter in actual.named_parameters():
        other = dict(expected.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    for name, value in actual.named_buffers():
        torch.testing.assert_close(value, dict(expected.named_buffers())[name], rtol=0, atol=0)
    for model in (actual, expected):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999), eps=1e-6,
                                     foreach=False)
        optimizer.step()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)


def test_4777_teacher_endpoint_budget_and_tail():
    sampler = ReferenceBatchSampler((4778,), batch_size=16)
    batches = list(sampler)
    rows = [r for batch in batches for r in batch]
    assert sampler.endpoint_count == 4777
    assert sampler.row_count == len(rows) == 19108
    assert len(sampler) == len(batches) == 1195 and len(batches[-1]) == 4
    assert len({(r.tracklet, r.frame, r.branch) for r in rows}) == 19108
    assert all(r.frame > 0 and r.starts_window and r.ends_window for r in rows)
    assert sampler.state_dict()['schema'] == 'seqtrack_reference.v32.teacher.v1'


def test_teacher_original_repeated_points_hint_pseudotime_and_independent_history_offsets():
    source = TinyReferenceSource((3,))
    dataset = ReferenceDataset(source, training=True)
    row = dataset[ReferenceRequest(0, 0, 0, 1, 2, 1)]
    batch = row['prepared_reference']
    points = batch['points'].reshape(4, 1024, 5)
    assert batch['valid_mask'].tolist() == [1, 0, 0]
    np.testing.assert_allclose(points[:, 0, 3], [-.1, -.1, -.1, .1])
    assert len(np.unique(points[0, :, :3], axis=0)) < 1024
    assert set(np.unique(points[0, :, 4])).issubset({0., 1.})
    assert np.all(points[-1, :, 4] == .5)
    # x=2.3 在原 1.25 倍标签框内（物体半长2），而非 production 的真实框标签。
    current_x = points[-1, :, 0]
    labels = batch['seg_label'].reshape(4, 1024)[-1]
    assert labels[np.isclose(current_x, 2.5)].all()
    np.random.seed(3)
    perturbed = dataset[ReferenceRequest(0, 0, 1, 1, 2, 1)]['prepared_reference']
    assert not np.allclose(perturbed['ref_boxs'][1], perturbed['ref_boxs'][2])
    hints = perturbed['points'][:3072, 4]
    assert np.all(np.isclose(hints, .2) | np.isclose(hints, .8))


def test_original_at_most_two_points_becomes_zeros_and_sampling_can_repeat():
    points = np.ones((2, 3), dtype=np.float32)
    empty, ids = points_utils.regularize_pc(points, 1024)
    assert ids is None and empty.shape == (1024, 3) and not empty.any()
    repeated, ids = points_utils.regularize_pc(np.arange(9).reshape(3, 3), 1024, seed=1)
    assert len(ids) == 1024 and len(np.unique(ids)) == 3
    assert repeated.shape == (1024, 3)


def test_invalid_history_resampling_is_audited_as_actual_exposure():
    torch.manual_seed(2)
    source = TinyReferenceSource((3, 3), invalid_first_track=True)
    dataset = ReferenceDataset(source, training=True)
    nominal = ReferenceRequest(0, 0, 0, 1, 2, 1)
    row = dataset[nominal]
    assert row['nominal_request'] == nominal and row['request'].tracklet == 1
    assert len(row['rejected_requests']) >= 1
    builder = BatchBuilder()
    batch = builder.prepare([row])
    builder.acquire(batch)
    builder.commit(SimpleNamespace(accepted_box=batch['box_label']), batch)
    summary = builder.exposure_summary()
    assert summary['nominal_rows'] == summary['actual_rows'] == summary['resampled_rows'] == 1
    assert summary['rejected_attempts'] >= 1
    assert summary['actual_exposure_counts'][0]['tracklet'] == 1
    assert not builder.states


def test_evaluation_uses_first_size_and_has_no_current_gt_input():
    dataset = ReferenceDataset(TinyReferenceSource((3,)), training=False)
    row = dataset[ReferenceRequest(0, 0, 4, 1, 3, 1)]
    states = {0: row['first_frame']['3d_bbox']}
    np.random.seed(4)
    before = evaluation_input(row, states)
    changed = deepcopy(row)
    changed['frames'][1]['3d_bbox'].center += 999.
    changed['frames'][1]['3d_bbox'].wlh *= 8.
    np.random.seed(4)
    after = evaluation_input(changed, states)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(before['bbox_size'], [2., 4., 2.])


def test_evaluation_input_matches_original_host_when_dimensions_are_constant():
    source = TinyReferenceSource((5,))
    dataset = ReferenceDataset(source, training=False)
    row = dataset[ReferenceRequest(0, 0, 4, 1, 5, 3)]
    sequence = [normalize_frame(frame) for frame in source.get_frames(0, list(range(5)))]
    results = [deepcopy(sequence[i]['3d_bbox']) for i in range(3)]
    results[-1].center += [.13, -.27, .05]
    from pyquaternion import Quaternion
    results[-1].orientation = Quaternion(axis=[0, 0, 1], radians=.2)
    proxy = SimpleNamespace(**{name: getattr(points_utils, name) for name in dir(points_utils) if not name.startswith('_')})
    proxy.np_to_torch_tensor = lambda data, device=None: torch.tensor(data, device=device).unsqueeze(0)
    namespace = dict(np=np, torch=torch, points_utils=proxy, geometry_utils=spatial,
        get_history_frame_ids_and_masks=get_history_frame_ids_and_masks,
        get_last_n_bounding_boxes=get_last_n_bounding_boxes,
        generate_timestamp_prev_list=generate_timestamp_prev_list,
        resolve_kitti_hv_search_offset=resolve_kitti_hv_search_offset)
    source_text = textwrap.dedent((ROOT / '_vendor' / 'evaluation.original.txt').read_text(encoding='utf-8'))
    exec(compile(source_text, '<original-evaluation-input>', 'exec'), namespace)
    host = SimpleNamespace(config=reference_config(), hist_num=3, device=torch.device('cpu'))
    np.random.seed(17)
    expected, _ = namespace['build_input_dict'](host, sequence, 3, results)
    np.random.seed(17)
    actual = evaluation_input(row, dict(enumerate(results)))
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value[0].numpy())


def test_world_offset_and_empty_fallback_match_original_getoffsetbb():
    from pyquaternion import Quaternion
    from datasets.data_classes import Box
    tracker = ReferenceTracker({}).eval()
    offset = torch.tensor([[.7, -.2, .3, .25]], dtype=torch.float32)
    tracker.forward_original = lambda batch: {'aux_estimation_boxes': offset}
    box = Box([10000., -4000., 2.], [2., 4., 1.5], Quaternion(axis=[0, 0, 1], radians=.73))
    anchor = torch.tensor([[*box.center, .73]], dtype=torch.float64)
    actual = tracker({'reference_anchor': anchor, 'reference_empty': torch.tensor([False])})
    expected = points_utils.getOffsetBB(box, offset[0].numpy().copy(), degrees=False, use_z=True, limit_box=False)
    np.testing.assert_allclose(actual.accepted_box[0, :3].numpy(), expected.center, rtol=0, atol=1e-10)
    assert actual.accepted_box[0, 3].item() == pytest.approx(expected.orientation.radians * expected.orientation.axis[-1])
    fallback = tracker({'reference_anchor': anchor, 'reference_empty': torch.tensor([True])})
    torch.testing.assert_close(fallback.accepted_box, anchor, rtol=0, atol=0)


def test_eval_complete_recursive_frames_and_shared_metric_denominator():
    from models.ct_v31.runtime import TrackingEvaluation
    source = TinyReferenceSource((4, 3, 1))
    loader = build_loaders(dict(workers=0, batch_size=2), roles=('test',), sources={'test': source})['test']
    builder, evaluator = BatchBuilder(), TrackingEvaluation()
    evaluator.add_singleton_tracks(loader.dataset)
    for rows in loader:
        batch = builder.prepare(rows)
        output = SimpleNamespace(accepted_box=batch['target_box'].clone(),
            selected_index=torch.zeros(len(rows), dtype=torch.long), selected_quality=torch.ones(len(rows)),
            evidence=SimpleNamespace(mode_valid=None))
        commits = builder.commit(output, batch, diagnostics=True)
        assert all(item == {} for item in commits)
        evaluator.add_batch(rows, batch, output, commits)
    summary = evaluator.summary()
    assert summary['prediction_frames'] == 5
    assert len(evaluator.rows) == 8 and not builder.states
    assert summary['success'] == pytest.approx(100.)


@pytest.mark.parametrize('workers', [0, 2])
def test_original_worker_rng_is_reproducible_for_fixed_worker_count(workers):
    def first_batch():
        np.random.seed(42)
        torch.manual_seed(42)
        loader = build_loaders(dict(workers=workers, batch_size=4, seed=42), roles=('train',),
                              sources={'train': TinyReferenceSource((3,))})['train']
        rows = list(loader)
        return [(r['request'], r['prepared_reference']['points']) for batch in rows for r in batch]
    one, two = first_batch(), first_batch()
    assert len(one) == len(two) == 8
    for (r1, p1), (r2, p2) in zip(one, two):
        assert r1 == r2
        np.testing.assert_array_equal(p1, p2)


def test_source_manifest_and_protocol_are_complete_and_isolated():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['vendored_files_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert len(manifest['sources']) == 11
    identity = protocol_identity()
    assert identity['canonical']['point_sample_size'] == 1024
    assert identity['canonical']['seg_weight'] == .1
    assert 'formal_workers4' in identity['random_protocol']
    assert (ROOT / 'LICENSE').read_text(encoding='utf-8').startswith('MIT License')
    for file in ROOT.rglob('*.py'):
        text = file.read_text(encoding='utf-8')
        assert 'from models.ct_v31.observation' not in text
        assert 'from models.ct_v31.model' not in text


def test_reference_and_production_source_identity_describe_same_raw_population():
    from models.ct_v31.data import RawEndpointDataset
    source = TinyReferenceSource()
    assert ReferenceDataset(source, training=True).source_sha256 == RawEndpointDataset(source).source_sha256
    source.tracklet_anno_list = [[{'t': 10. + t * .5} for t in range(n)] for n in source.lengths]
    source._anno_timestamp = lambda row: row['t']
    source.get_frame_metadata = lambda *args: (_ for _ in ()).throw(AssertionError('metadata fast path not used'))
    assert ReferenceDataset(source, training=True).source_sha256 == RawEndpointDataset(source).source_sha256


def test_real_teacher_adapter_network_loss_backward_uses_portable_label_dtypes():
    source = TinyReferenceSource((3,))
    loader = build_loaders(dict(workers=0, batch_size=2, seed=42), roles=('train',),
                          sources={'train': source})['train']
    rows = next(iter(loader))
    # 模拟 Windows 原 NumPy int 标签，防止仅手写long合成batch掩盖边界。
    for row in rows:
        for key in ('seg_label', 'motion_state_label', 'valid_mask'):
            row['prepared_reference'][key] = row['prepared_reference'][key].astype(np.int32)
    builder = BatchBuilder()
    batch = builder.prepare(rows)
    for key in ('seg_label', 'motion_state_label', 'valid_mask'):
        assert batch[key].dtype == torch.long
    model = ReferenceTracker({}).train()
    output = model(batch)
    losses = model.compute_losses(batch, output)
    assert batch['points'].shape == (2, 4096, 5)
    assert torch.isfinite(losses['loss_total'])
    losses['loss_total'].backward()
    assert model.seg_pointnet.fc.weight.grad is not None
    assert torch.isfinite(model.seg_pointnet.fc.weight.grad).all()
    builder.commit(output, batch)
    assert builder.exposure_summary()['actual_rows'] == 2


def test_training_network_size_uses_first_frame_even_when_current_gt_size_changes():
    class ChangedSizeSource(TinyReferenceSource):
        def get_frames(self, track, frame_ids):
            rows = super().get_frames(track, frame_ids)
            for frame, row in zip(frame_ids, rows):
                if frame > 0:
                    row['box_size'] = np.array([8., 5., 3.])
            return rows

    request = ReferenceRequest(0, 0, 0, 1, 2, 1)
    original = ReferenceDataset(TinyReferenceSource((3,)), training=True)[request]
    changed = ReferenceDataset(ChangedSizeSource((3,)), training=True)[request]
    # 上游原 teacher 保留当前 GT 尺寸和标签；只在网络输入边界替换 bbox_size。
    np.testing.assert_array_equal(original['prepared_reference']['bbox_size'], [2., 4., 2.])
    np.testing.assert_array_equal(changed['prepared_reference']['bbox_size'], [5., 8., 3.])
    one = BatchBuilder().prepare([original])
    two = BatchBuilder().prepare([changed])
    torch.testing.assert_close(one['bbox_size'], two['bbox_size'], rtol=0, atol=0)
    np.testing.assert_array_equal(two['bbox_size'][0].numpy(), [2., 4., 2.])
    np.testing.assert_array_equal(two['target_box_size'][0].numpy(), [8., 5., 3.])
    np.testing.assert_array_equal(two['this_bc'][0].numpy(), changed['prepared_reference']['this_bc'])


@pytest.mark.parametrize('length,count', [(1024, 128), (4096, 1), (8, 128), (7, 4), (2, 3)])
def test_deterministic_pool_preserves_original_adaptive_bins_ties_and_gradient(length, count):
    torch.manual_seed(23)
    value = torch.randint(-2, 3, (2, 3, length)).float().requires_grad_()
    expected = F.adaptive_max_pool1d(value, count)
    actual = ReferenceAdaptiveMaxPool1d(count)(value)
    gradient = torch.randn_like(actual)
    expected_gradient = torch.autograd.grad(expected, value, gradient, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, value, gradient)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_flat_unreduced_weighted_ce_preserves_noncontiguous_logits_and_denominator(dtype):
    generator = torch.Generator().manual_seed(33)
    source = torch.randn(4, 11, 4096, generator=generator, dtype=dtype, requires_grad=True)
    logits = source[:, :2]
    labels = torch.zeros(4, 4096, dtype=torch.long)
    labels[0] = 1
    labels[2, :11] = 1
    labels[3, :9] = -100
    expected = F.cross_entropy(logits, labels, weight=logits.new_tensor([.5, 2.]))
    actual = original_weighted_segmentation_loss(logits, labels)
    grad_expected = torch.autograd.grad(expected, source, retain_graph=True)[0]
    grad_actual = torch.autograd.grad(actual, source)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(grad_actual, grad_expected, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires actual CUDA runtime'))])
def test_reference_numeric_backward_preserves_strict_deterministic_setting(device):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    snapshots = []
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        for _ in range(2):
            generator = torch.Generator().manual_seed(18)
            value = torch.randn(2, 2, 1024, generator=generator).to(device).requires_grad_()
            pooled = ReferenceAdaptiveMaxPool1d(128)(value)
            overlap = ReferenceAdaptiveMaxPool1d(128)(value[..., :8])
            labels = torch.zeros(2, 128, dtype=torch.long, device=device)
            loss = original_weighted_segmentation_loss(pooled, labels) + overlap.square().mean()
            loss.backward()
            snapshots.append((loss.detach(), value.grad.detach()))
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        for one, two in zip(*snapshots):
            torch.testing.assert_close(one, two, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
