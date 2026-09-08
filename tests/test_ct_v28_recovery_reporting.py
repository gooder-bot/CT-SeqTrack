"""固定闭环恢复定义、完整帧分母与未知计数的报告回归。"""

import copy
import json

import pytest
import torch

from tests.test_ct_v27_evaluation import FakeHost, sequence
from utils.v27_eval_reporting import summarize_endpoint_diagnostics
from utils.v27_evaluation import evaluate_sequence_v27
from utils.v28_recovery_reporting import summarize_recovery_rows


def _rows(counts, ious, tracklet='track/1'):
    rows = [dict(scene_id='scene', tracklet_id=tracklet, frame_id=0,
                 is_initial=True, final_iou=1., final_success=1., final_precision=1.)]
    for frame, (count, iou) in enumerate(zip(counts, ious), 1):
        rows.append(dict(scene_id='scene', tracklet_id=tracklet, frame_id=frame,
                         is_initial=False, b0_raw_point_count=count,
                         final_iou=iou, final_success=iou, final_precision=1 - iou))
    return rows


def test_point_buckets_are_prediction_only_percent_and_missing_is_unknown():
    rows = _rows([0, 1, 2, 3, 10, 11, 50, 51, None], [.05, .1, .2, .3, .4, .5, .6, .7, .8])
    original = copy.deepcopy(rows)
    result = summarize_recovery_rows(list(reversed(rows)))
    assert rows == original
    assert result == summarize_recovery_rows(rows)
    assert result['coverage'] == dict(tracklets=1, all_frames=10, prediction_frames=9,
        raw_count_measured_frames=8, raw_count_unknown_frames=1, raw_count_complete=False)
    buckets = result['point_count_buckets']
    assert {key: value['frames'] for key, value in buckets.items()} == {
        '0': 1, '1': 1, '2': 1, '3-10': 2, '11-50': 2, '>50': 1, 'unknown': 1}
    assert buckets['3-10']['S'] == pytest.approx(35.)
    assert buckets['3-10']['P'] == pytest.approx(65.)
    assert buckets['unknown']['S'] == 80.
    first = result['first_failure']['tracklets'][0]
    assert first['first_lost_frame'] == 1
    assert first['first_zero_iou_frame'] is None and first['first_zero_right_censored']
    json.dumps(result, allow_nan=False)


def test_contiguous_empty_and_compatible_sparse_segments_have_explicit_censoring():
    result = summarize_recovery_rows(_rows(
        [4, 0, 0, 3, 4, 0, 1, None, 0, 0], [.8, 0., .05, .2, .5, 0., .5, .9, .1, 0.]))
    empty = result['empty_crop']
    assert empty['segments'] == 3 and empty['frames'] == 5
    assert empty['longest_observed_duration_frames'] == 2
    assert empty['post_segment_returns'] == empty['recovered_lost_segments'] == 2
    assert empty['observed_recovery_fraction'] == pytest.approx(2 / 3)
    first, second, third = empty['episodes']
    assert (first['start_frame'], first['end_frame'], first['post_segment_return_frame']) == (2, 3, 5)
    assert first['post_segment_return_delay_frames'] == 2
    assert second['post_segment_return_delay_frames'] == 1
    assert third['duration_left_censored'] and third['duration_right_censored']
    assert third['return_right_censored'] and third['return_censor_reason'] == 'sequence_end'
    assert third['observed_followup_frames'] == 0
    sparse = result['compatible_sparse']
    assert sparse['frames'] == 6 and sparse['post_segment_returns'] == 1
    assert sparse['episodes'][1]['return_censor_reason'] == 'missing_raw_count'
    assert sparse['episodes'][1]['duration_right_censored']


def test_next_segment_censors_return_and_never_lost_is_not_failed_recovery():
    rows = _rows([0, 3, 0, 3], [0., .2, 0., .5])
    rows += _rows([0, 4], [.1, .5], tracklet='never-lost')
    result = summarize_recovery_rows(rows)
    first = next(segment for segment in result['empty_crop']['episodes']
                 if segment['tracklet_id'] == 'track/1' and segment['start_frame'] == 1)
    assert first['return_censor_reason'] == 'next_segment'
    assert first['post_segment_return_frame'] is None and first['observed_followup_frames'] == 1
    assert result['empty_crop']['post_segment_returns'] == 2
    assert result['empty_crop']['lost_segments'] == 2
    assert result['empty_crop']['recovered_lost_segments'] == 1
    never = next(row for row in result['first_failure']['tracklets'] if row['tracklet_id'] == 'never-lost')
    assert never['first_lost_frame'] is None and never['first_loss_right_censored']
    assert never['observed_through_frame'] == 2


def test_missing_count_fallback_and_incomplete_or_invalid_inputs_fail_explicitly():
    rows = _rows([None, None], [0., .5])
    rows[1]['acquisition_base_raw_point_count'] = 2
    result = summarize_recovery_rows(rows)
    assert result['point_count_buckets']['2']['frames'] == 1
    assert result['empty_crop']['segments'] == 0
    assert result['point_count_buckets']['0']['S'] is None
    for broken in (rows[1:], rows + [rows[-1]], rows[:1] + rows[2:]):
        with pytest.raises(ValueError, match='contiguous'):
            summarize_recovery_rows(broken)
    rows[1]['b0_raw_point_count'] = -.1
    with pytest.raises(ValueError, match='raw crop count'):
        summarize_recovery_rows(rows)


def test_v28_evaluator_exports_original_count_separate_from_sampled_validity():
    class Host(FakeHost):
        def build_input_dict(self, *args, **kwargs):
            batch, anchor = super().build_input_dict(*args, **kwargs)
            count = (0, 2, 5)[batch['frame_id'] - 1]
            batch['b0_raw_point_count'] = torch.tensor([[4, 4, 4, count]])
            batch['b0_point_valid_mask'] = torch.ones(1, 4, 8, dtype=torch.bool)
            batch['b0_point_valid_mask'][:, -1] = count > 2
            return batch, anchor

        def evaluate_one_sample(self, *args, **kwargs):
            final, valid, output = super().evaluate_one_sample(*args, **kwargs)
            output['ct_b2_available'] = output['ct_policy_candidate_valid']
            return final, valid, output

    host = Host()
    host.config.ct_enable_v28 = True
    host.config.export_proposal_diagnostics = False
    evaluate_sequence_v27(host, sequence())
    rows = host._ct_v27_sequence_endpoints
    assert [row['b0_raw_point_count'] for row in rows] == [None, 0, 2, 5]
    assert [row['current_sampled_valid'] for row in rows] == [None, False, False, True]
    summary = summarize_endpoint_diagnostics(rows)
    assert summary['recovery']['point_count_buckets']['2']['frames'] == 1
    assert summary['recovery']['coverage']['prediction_frames'] == 3
    legacy = FakeHost()
    evaluate_sequence_v27(legacy, sequence())
    assert 'b0_raw_point_count' not in legacy._ct_v27_sequence_endpoints[1]
    assert 'recovery' not in summarize_endpoint_diagnostics(legacy._ct_v27_sequence_endpoints)
