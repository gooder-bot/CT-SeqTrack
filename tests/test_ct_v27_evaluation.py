import ast
import copy
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.tracking_metrics_v27 import LocalYawBox, metric_contributions
from utils.v27_evaluation import evaluate_sequence_v27
from utils.v27_eval_reporting import summarize_endpoint_diagnostics, write_endpoint_diagnostics


class FakeHost:
    def __init__(self, apply=False):
        self.config = SimpleNamespace(up_axis=(0, 0, 1), IoU_space=3,
            export_proposal_diagnostics=True, use_ct_joint_full=True, ct_enable_b2=True)
        self.apply = apply
        self.calls = []

    def build_input_dict(self, sequence, frame_id, results, recursive_state, _ct_diagnostic_sidecar):
        assert sorted(recursive_state.predictions) == list(range(frame_id))
        assert np.array_equal(recursive_state.predictions[frame_id - 1].center, results[-1].center)
        self.calls.append(frame_id)
        # Frame 1: empty base crop but real extension; frame 2: all input empty.
        n = 0 if frame_id == 2 else 4
        acquisition = dict(global_target_count_exact=n, global_raw_point_count=n,
            base_raw_target_count=0, base_raw_point_count=0,
            base_sampled_target_count=0, base_sampled_point_count=0)
        for prefix in ('support_raw', 'support_novel', 'novel_pool', 'prepool'):
            acquisition[prefix + '_target_count'] = n
            acquisition[prefix + '_point_count'] = n
        _ct_diagnostic_sidecar['acquisition'] = acquisition
        return {'frame_id': frame_id, 'n': n, 'ct_extension_labels': torch.ones(1, 4)}, results[-1]

    def _local_prediction_to_world(self, local, anchor):
        return LocalYawBox(local.detach().cpu().numpy(), anchor.wlh)

    def evaluate_one_sample(self, batch, ref_box):
        available = bool(batch['n'])
        applied = self.apply and available
        local = torch.tensor([[0., 0., .5, .1]])
        correction = torch.tensor([[.5, 0.]]) if available else torch.zeros(1, 2)
        output = dict(observation_aux_estimation_boxes=local,
            ct_policy_candidate_valid=torch.tensor([available]),
            ct_router_bounded_residual_xy=correction,
            ct_router_applied_gate=torch.tensor([applied]),
            ct_b3_action_score=torch.tensor([.1]),
            ct_b2_extension_presence_probability=torch.tensor([0.]),
            ct_extension_selected_indices=torch.tensor([[0, 1, 2, 3]]),
            ct_extension_selected_valid_mask=torch.full((1, 4), available),
            ct_vote_mode_inlier_mask=torch.full((1, 4), available),
            ct_vote_mode_unique_count=torch.tensor([[batch['n']]]))
        final = local[0].clone()
        final[:2] += correction[0] * int(applied)
        return self._local_prediction_to_world(final, ref_box), None, output

    def _build_ct_joint_diagnostic_row(self, output, batch, target, anchor, frame_id, acquisition_diagnostics):
        return dict(selected_point_count=batch['n'], selected_target_count=batch['n'],
                    selected_target_bearing=int(batch['n'] > 0))


def sequence():
    return [dict(tracklet_key='track/1', scene_id='scene-1', timestamp=float(t),
                 **{'3d_bbox': LocalYawBox([.5, 0., .5, .1], [2., 4., 2. if t == 0 else 3.])})
            for t in range(4)]


@pytest.mark.parametrize('apply', [False, True])
def test_complete_sequence_includes_initial_and_empty_inputs(apply, tmp_path):
    host = FakeHost(apply)
    overlaps, distances, boxes = evaluate_sequence_v27(host, sequence())
    rows = host._ct_v27_sequence_endpoints
    assert host.calls == [1, 2, 3]
    assert len(rows) == len(boxes) == 4
    assert rows[0]['is_initial'] and not rows[0]['action_applied']
    assert rows[0]['final_success'] == rows[0]['final_precision'] == 1.
    assert rows[1]['structural_available'] and rows[1]['presence_score'] == 0.
    assert not rows[2]['structural_available'] and not rows[2]['action_applied']
    assert [r['action_applied'] for r in rows] == [False, apply, False, apply]
    assert all(np.array_equal(b.wlh, [2., 4., 2.]) for b in boxes)
    assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float)))
    summary = summarize_endpoint_diagnostics(rows)
    assert summary['funnel']['complete']
    assert summary['funnel']['measured_frames'] == 3
    assert summary['funnel']['n0_current_with_extension_frames'] == 2
    assert summary['funnel']['all_empty_frames'] == 1
    assert summary['funnel']['stages']['consensus_top_mode']['target_points'] == 8
    assert summary['funnel']['action_stages']['bounded']['frames'] == 2
    assert summary['funnel']['action_stages']['accepted']['frames'] == 2 * int(apply)
    assert summary['runtime']['tracking_elapsed_ms'] > 0
    assert summary['metrics']['frames'] == 4
    if not apply:
        assert summary['metrics']['actions'] == 0
        assert summary['metrics']['one_step_net_utility'] == 0.
        assert sum(r['applied_utility_gain'] for r in rows) == 0.
    write_endpoint_diagnostics(tmp_path / 'summary.json', rows)


def test_benchmark_sequence_matches_existing_geometry_and_auc():
    polygon = pytest.importorskip('shapely.geometry').Polygon
    source = ast.parse(Path('utils/metrics.py').read_text(encoding='utf-8'))
    names = {'estimateAccuracy', 'fromBoxToPoly', 'estimateOverlap'}
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(np=np, Polygon=polygon)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'legacy_metric', 'exec'), namespace)
    host, frames = FakeHost(True), sequence()
    overlaps, distances, boxes = evaluate_sequence_v27(host, frames)
    expected_iou = [namespace['estimateOverlap'](box, frame['3d_bbox'], dim=3, up_axis=(0, 0, 1)) for box, frame in zip(boxes, frames)]
    expected_distance = [namespace['estimateAccuracy'](box, frame['3d_bbox'], dim=3, up_axis=(0, 0, 1)) for box, frame in zip(boxes, frames)]
    assert np.allclose(overlaps, expected_iou)
    assert np.allclose(distances, expected_distance)
    s, p = metric_contributions(overlaps, distances)
    for values, axis, expected, compare in (
            (overlaps, torch.linspace(0, 1, 21), np.mean(s), torch.ge),
            (distances, torch.linspace(0, 2, 21), np.mean(p), torch.le)):
        curve = torch.stack([compare(torch.tensor(values, dtype=torch.float32), x).float().mean() for x in axis])
        legacy = torch.trapz(curve, axis) / axis[-1]
        assert float(legacy) == pytest.approx(expected, abs=1e-7)


def test_missing_funnel_measurements_are_not_zero_recall():
    host = FakeHost()
    evaluate_sequence_v27(host, sequence())
    rows = copy.deepcopy(host._ct_v27_sequence_endpoints)
    rows[1]['acquisition_available'] = False
    summary = summarize_endpoint_diagnostics(rows)
    assert not summary['funnel']['complete']
    assert summary['funnel']['stages']['global']['measured_frames'] == 2


def test_selection_funnel_does_not_require_optional_diagnostic_export():
    host = FakeHost(True)
    host.config.export_proposal_diagnostics = False
    evaluate_sequence_v27(host, sequence())
    assert not host._proposal_sequence_diagnostics
    report = summarize_endpoint_diagnostics(host._ct_v27_sequence_endpoints)
    assert report['funnel']['complete']
    assert report['funnel']['stages']['selected']['target_points'] == 8


def test_base_test_step_preserves_full_rows_and_csv_optional_columns(tmp_path):
    source = ast.parse(Path('models/base_model.py').read_text(encoding='utf-8'))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'BaseModelMF')
    nodes = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in {'test_step', '_write_csv_rows'}]
    for node in nodes:
        node.decorator_list = []
    ns = dict(np=np, torch=torch, time=time, csv=csv, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'base_evaluation', 'exec'), ns)
    class Metric:
        def __call__(self, *args):
            pass
        def reset(self):
            pass
        def compute(self):
            return 1.
    host = FakeHost(True)
    host.config.ct_enable_v27 = True
    host.trainer = SimpleNamespace(test_dataloaders=None)
    host.logger = SimpleNamespace(experiment=SimpleNamespace(add_scalars=lambda *args, **kwargs: None))
    host.log = lambda *args, **kwargs: None
    host.device = torch.device('cpu')
    for key in ('success', 'prec', 'success_step', 'prec_step', 'n_frames', 'runtime'):
        setattr(host, key, Metric())
    host._tracking_test_endpoints = []
    host._proposal_test_diagnostics = []
    host._b3_test_rollouts = []
    host.evaluate_one_sequence = lambda frames: evaluate_sequence_v27(host, frames)
    ns['test_step'](host, [sequence()], 9)
    rows = host._tracking_test_endpoints
    assert len(rows) == 4 and all(r['tracklet_id'] == 9 for r in rows)
    assert rows[1]['acquisition_consensus_target_count'] == 4
    assert rows[0]['is_initial'] and rows[0]['exact_final_success'] == 1.
    path = tmp_path / 'tracking.csv'
    ns['_write_csv_rows'](path, rows)
    with path.open(newline='', encoding='utf-8') as handle:
        stored = list(csv.DictReader(handle))
    assert stored[0]['acquisition_global_raw_point_count'] == ''
    assert stored[1]['acquisition_global_raw_point_count'] == '4.0'


def test_v27_main_config_plumbing_matches_strict_checkpoint_identity(monkeypatch, tmp_path):
    import argparse
    import sys
    from utils.config import load_yaml_config
    from utils.action_calibration_v27 import action_calibration_config_identity
    from utils.checkpoint_loading import load_initial_weights
    from models.ct_variant import configure_ct_variant
    source = ast.parse(Path('main.py').read_text(encoding='utf-8'))
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in {'parse_config', 'parse_limit_train_batches'}]
    ns = dict(argparse=argparse, load_yaml=load_yaml_config, EasyDict=lambda x: x)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'main_config', 'exec'), ns)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--cfg', 'cfgs/ct_seqtrack/27_full.yaml'])
    saved = ns['parse_config']()
    saved.update(ct_scene_manifest_sha256='scene-hash', ct_protocol_role='train', export_proposal_diagnostics=False)
    raw = load_yaml_config('cfgs/ct_seqtrack/27_full.yaml')
    configure_ct_variant(saved)
    configure_ct_variant(raw)
    assert action_calibration_config_identity(saved) == action_calibration_config_identity(raw)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = raw
            for name in ('seg_pointnet', 'mini_pointnet', 'motion_mlp', 'feature_pointnet', 'Transformer', 'physical_motion_encoder'):
                setattr(self, name, torch.nn.Linear(1, 1))
    model = Model()
    checkpoint = tmp_path / 'formal.ckpt'
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': {'config': saved}}, checkpoint)
    assert load_initial_weights(model, checkpoint, require_complete=True)['complete']
    model.config['ct_router_radius_max'] += 1.
    with pytest.raises(RuntimeError, match='resolved-config identity'):
        load_initial_weights(model, checkpoint, require_complete=True)
