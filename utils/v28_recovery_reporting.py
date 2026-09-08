"""v28 完整闭环端点的固定失跟、点数分桶及连续稀疏段诊断。"""

import math
from collections import defaultdict


LOST_IOU = 0.1
RETURN_IOU = 0.5
POINT_BUCKETS = ('0', '1', '2', '3-10', '11-50', '>50', 'unknown')


def _raw_count(row):
    value = row.get('b0_raw_point_count')
    if value is None or value == '':
        value = row.get('acquisition_base_raw_point_count')
    if value is None or value == '':
        return None
    count = float(value)
    if not math.isfinite(count) or count < 0 or count != int(count):
        raise ValueError('v28 raw crop count must be a finite nonnegative integer')
    return int(count)


def _metric(row, key):
    value = float(row[key])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'v28 {key} must be finite and in [0,1]')
    return value


def _bucket(count):
    if count is None:
        return 'unknown'
    if count <= 2:
        return str(count)
    return '3-10' if count <= 10 else '11-50' if count <= 50 else '>50'


def _mean(values):
    return sum(values) / len(values) if values else None


def _segments(rows, counts, threshold):
    """缺失计数中断可观测区间；不跨越未知帧拼接或归因恢复。"""
    segments = []
    index = 0
    while index < len(rows):
        if counts[index] is None or counts[index] > threshold:
            index += 1
            continue
        start = index
        while index + 1 < len(rows) and counts[index + 1] is not None and counts[index + 1] <= threshold:
            index += 1
        end = index
        duration_censored = end + 1 == len(rows) or counts[end + 1] is None
        returned = None
        observed_followup = 0
        reason = 'sequence_end'
        for followup in range(end + 1, len(rows)):
            if counts[followup] is None:
                reason = 'missing_raw_count'
                break
            if counts[followup] <= threshold:
                reason = 'next_segment'
                break
            observed_followup += 1
            if _metric(rows[followup], 'final_iou') >= RETURN_IOU:
                returned = int(rows[followup]['frame_id'])
                reason = None
                break
        segments.append(dict(
            start_frame=int(rows[start]['frame_id']), end_frame=int(rows[end]['frame_id']),
            observed_duration_frames=end - start + 1,
            duration_right_censored=duration_censored,
            duration_left_censored=start > 0 and counts[start - 1] is None,
            lost_during_segment=any(_metric(row, 'final_iou') < LOST_IOU for row in rows[start:end + 1]),
            post_segment_return_frame=returned,
            post_segment_return_delay_frames=None if returned is None else returned - int(rows[end]['frame_id']),
            observed_followup_frames=observed_followup,
            return_right_censored=returned is None, return_censor_reason=reason))
        index += 1
    return segments


def _segment_summary(segments):
    returned = [segment for segment in segments if segment['post_segment_return_frame'] is not None]
    lost = [segment for segment in segments if segment['lost_during_segment']]
    recovered = [segment for segment in lost if segment['post_segment_return_frame'] is not None]
    durations = [segment['observed_duration_frames'] for segment in segments]
    return dict(segments=len(segments), frames=sum(durations),
                longest_observed_duration_frames=max(durations, default=0),
                mean_observed_duration_frames=_mean(durations),
                duration_right_censored_segments=sum(segment['duration_right_censored'] for segment in segments),
                post_segment_returns=len(returned),
                return_right_censored_segments=len(segments) - len(returned),
                mean_observed_return_delay_frames=_mean([segment['post_segment_return_delay_frames'] for segment in returned]),
                lost_segments=len(lost), recovered_lost_segments=len(recovered),
                observed_recovery_fraction=len(recovered) / len(lost) if lost else None,
                recovery_fraction_role='observed lower bound; censored segments are not confirmed failures',
                episodes=segments)


def summarize_recovery_rows(rows):
    """只读 final 闭环输出；不筛 GT、不改变全帧 S/P 或训练/校准策略。"""
    rows = list(rows)
    if not rows:
        raise ValueError('v28 recovery report requires complete nonempty endpoints')
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row['scene_id']), str(row['tracklet_id']))].append(row)
    buckets = {name: [] for name in POINT_BUCKETS}
    tracklets, empty, sparse = [], [], []
    for (scene, tracklet), sequence in sorted(grouped.items()):
        sequence = sorted(sequence, key=lambda row: int(row['frame_id']))
        if [float(row['frame_id']) for row in sequence] != list(range(len(sequence))):
            raise ValueError('v28 recovery report requires all contiguous frames including frame0 once')
        if any(bool(float(row['is_initial'])) != (index == 0) for index, row in enumerate(sequence)):
            raise ValueError('v28 recovery report first-frame marker mismatch')
        predicted = sequence[1:]
        counts = [_raw_count(row) for row in predicted]
        for row, count in zip(predicted, counts):
            for key in ('final_iou', 'final_success', 'final_precision'):
                _metric(row, key)
            buckets[_bucket(count)].append(row)
        lost = next((int(row['frame_id']) for row in predicted if _metric(row, 'final_iou') < LOST_IOU), None)
        zero = next((int(row['frame_id']) for row in predicted if _metric(row, 'final_iou') == 0), None)
        tracklets.append(dict(scene_id=scene, tracklet_id=tracklet,
            prediction_frames=len(predicted), raw_count_unknown_frames=counts.count(None),
            first_lost_frame=lost, first_zero_iou_frame=zero,
            first_loss_right_censored=lost is None, first_zero_right_censored=zero is None,
            observed_through_frame=len(sequence) - 1))
        for threshold, destination in ((0, empty), (2, sparse)):
            destination.extend(dict(scene_id=scene, tracklet_id=tracklet, **segment)
                               for segment in _segments(predicted, counts, threshold))
    prediction_frames = sum(len(subset) for subset in buckets.values())
    return dict(schema='ct_seqtrack.recovery_diagnostics.v28',
        definitions=dict(metric_mode='benchmark_compat', output='final_closed_loop',
            lost_iou_below=LOST_IOU, return_iou_at_least=RETURN_IOU,
            first_zero_iou_exact=True, bucket_metric_scale='percent', bucket_scope='prediction_frames_only',
            raw_count='current B0 crop before slot regularization; missing is unknown',
            empty_crop='raw_count == 0', compatible_sparse='raw_count <= 2 (includes empty)',
            return_window='after segment end, before next same-kind segment, missing count, or sequence end',
            return_mean_scope='observed returns only; not censor-adjusted',
            use='reporting only; fixed thresholds do not select training data, checkpoints, or action policies'),
        coverage=dict(tracklets=len(tracklets), all_frames=len(rows), prediction_frames=prediction_frames,
                      raw_count_measured_frames=prediction_frames - len(buckets['unknown']),
                      raw_count_unknown_frames=len(buckets['unknown']),
                      raw_count_complete=not buckets['unknown']),
        first_failure=dict(lost_tracklets=sum(row['first_lost_frame'] is not None for row in tracklets),
                           zero_iou_tracklets=sum(row['first_zero_iou_frame'] is not None for row in tracklets),
                           tracklets=tracklets),
        point_count_buckets={name: dict(frames=len(subset),
            S=100 * _mean([_metric(row, 'final_success') for row in subset]) if subset else None,
            P=100 * _mean([_metric(row, 'final_precision') for row in subset]) if subset else None)
            for name, subset in buckets.items()},
        empty_crop=_segment_summary(empty), compatible_sparse=_segment_summary(sparse))
