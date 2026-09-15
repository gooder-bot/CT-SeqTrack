"""全量事务条件分母与实际闭环评测漏斗，不用零默认值冒充已测数据。"""
import json
from types import SimpleNamespace

import torch

from tests.test_ct_v30_host import construct30, batch30, full_model_runtime, sampler_runtime  # noqa: F401
from utils.v30_funnel import accumulate_funnel, flush_funnel
from utils.v30_reporting import _stage_summary


def test_full_training_counts_use_all_rows_and_exact_large_point_totals(tmp_path):
    seen = []
    host = SimpleNamespace(global_step=9, current_epoch=0,
        logger=SimpleNamespace(log_dir=str(tmp_path), experiment=SimpleNamespace(
            add_scalars=lambda *args, **kwargs: seen.append((args, kwargs)))))
    modes = SimpleNamespace(centers_xy=torch.zeros(3, 3, 2),
        valid=torch.tensor([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool),
        member_mask=torch.zeros(3, 3, 4, dtype=torch.bool))
    modes.member_mask[:2, 0, 0] = True
    data = dict(ct_extension_labels=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.]]),
        box_label=torch.zeros(3, 4), bbox_size=torch.ones(3, 3),
        b0_raw_point_count=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 20]]),
        ct_recursive_state_age=torch.tensor([0., 3., 8.]),
        velocity_label=torch.tensor([[0., 0., 0.], [2., 0., 0.], [20., 0., 0.]]),
        ct_recursive_lost_length=torch.tensor([0., 2., 9.]),
        motion_margin_global_novel_target_count=torch.tensor([5., 2., 0.]),
        motion_margin_max_reachable_target_count=torch.tensor([4., 1., 0.]),
        ct_acquisition_extension_pool_target_count=torch.tensor([2., 1., 0.]),
        ct_acquisition_sampled_target_count=torch.tensor([1., 1., 0.]),
        ct_acquisition_global_novel_point_count=torch.tensor([2.**24, 1., 1.]),
        ct_acquisition_max_reachable_point_count=torch.tensor([8., 3., 0.]),
        ct_acquisition_support_novel_point_count=torch.tensor([4., 2., 0.]),
        ct_acquisition_prepool_point_count=torch.tensor([2., 1., 0.]))
    output = dict(ct_evidence_modes=modes,
        ct_extension_selected_indices=torch.arange(4).expand(3, -1),
        ct_extension_selected_valid_mask=torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.]]),
        ct_b3_action_valid=torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0]*6], dtype=torch.bool),
        ct_b3_chosen_action_index=torch.tensor([1, -1, -1]))
    utility = dict(action_valid=output['ct_b3_action_valid'],
        action_h1_success_gain=torch.tensor([[.2, .4, 0., 0., 0., 0.], [-.1, -.2, 0., 0., 0., 0.], [0.]*6]),
        action_h1_precision_gain=torch.tensor([[.2, .4, 0., 0., 0., 0.], [-.1, -.2, 0., 0., 0., 0.], [0.]*6]))
    # 写出仅发生在flush；两个step的6行全部累计。
    for _ in range(2):
        accumulate_funnel(host, data, output, {}, utility)
    assert not seen
    summary = flush_funnel(host)
    assert summary['all/rows'] == 6
    assert summary['all/global_novel_unique_points'] == 2 * (2**24 + 2)
    assert summary['all/global_novel_conditional_denominator'] == 6
    assert summary['all/max_reachable_conditional_denominator'] == 4
    assert summary['all/max_reachable_conditional_retained'] == 4
    assert summary['all/correct_mode_utilization_denominator'] == 4
    assert summary['all/selected_correct_mode'] == 2
    assert summary['all/beneficial_action_events'] == 2
    for group in ('sparse_empty', 'sparse_1_2', 'sparse_dense', 'age_8_plus', 'speed_static', 'lost_8_plus'):
        assert summary[group + '/rows'] == 2
    saved = json.loads((tmp_path / 'ct_v30_funnel' / 'epoch_001.json').read_text())
    assert saved['counts'] == summary and len(seen) == 1
    assert all(not value.requires_grad for value in host._ct_v30_funnel_counts.values())


def test_actual_closed_loop_reports_reachable_raw_ids_without_training_grid(full_model_runtime):
    from utils.v27_evaluation import evaluate_sequence_v27
    model = construct30(full_model_runtime).eval()
    _, sequence, _ = batch30(full_model_runtime, model, batch_size=1)
    model.config.export_proposal_diagnostics = False
    model.config.export_v3_candidate_diagnostics = False
    with torch.no_grad():
        evaluate_sequence_v27(model, sequence)
    rows = model._ct_v27_sequence_endpoints[1:]
    summary = _stage_summary(rows)
    assert len(rows) == len(sequence) - 1
    for stage in summary['stages'].values():
        assert stage['measured_frames'] == len(rows)
        assert stage['unique_point_measured_frames'] == len(rows)
        assert stage['target_points'] is not None
    for row in rows:
        assert row['acquisition_max_reachable_target_count'] == row['acquisition_max_legal_novel_target_count']
        assert row['acquisition_global_novel_point_count'] >= row['acquisition_max_reachable_point_count']
        assert row['acquisition_support_novel_point_count'] >= row['acquisition_prepool_point_count'] >= row['acquisition_selected_point_count']
        assert row['acquisition_support_novel_target_count'] >= row['acquisition_prepool_target_count'] >= row['acquisition_selected_target_count']
    assert summary['stages']['max_reachable']['conditional_denominator'] == sum(
        row['acquisition_global_novel_target_count'] > 0 for row in rows)
