import pytest

from tools.compare_v41_local_vote import FIXED, track_counts, summarize


def trajectory(misses, distances=None, *, start=3):
    observations = [
        dict(
            scan=start - 1,
            timestamp=0.1 * (start - 1),
            initialization=True,
            success=1.0,
            precision=1.0,
            distance=0.0,
        )
    ]
    for i, miss in enumerate(misses):
        d = 0.0 if distances is None else distances[i]
        observations.append(
            dict(
                scan=start + i,
                timestamp=0.1 * (start + i),
                initialization=False,
                success=float(d < 2),
                precision=float(d < 2),
                distance=d,
                raw=3,
                union=0 if miss else 1,
                iou=float(d < 2),
            )
        )
    return observations


def test_exclusions_use_net_reduction_and_fixed_late3_track():
    other = ("0019", 63)
    tie = track_counts(trajectory([0, 0, 0]), trajectory([0, 0, 0]))
    first = track_counts(trajectory([1, 1, 1]), trajectory([0, 0, 0]))
    second = track_counts(trajectory([0, 0, 0]), trajectory([1, 1, 1]))
    epochs = {
        58: {FIXED: first, other: second},
        59: {FIXED: first, other: tie},
        60: {FIXED: second, other: first},
    }
    result = summarize(epochs)
    assert result["max_track"] == "0020/122"
    assert result["fixed_exclusion_scope"] == "without_max"
    assert result["scopes"]["without_max"]["epochs"][58]["added"] == 3
    assert summarize({60: epochs[60]})["max_track"] == "0019/63"
    assert summarize({60: epochs[60]})["max_net_share"] is None


def test_score_denominator_and_recovery_reset_at_scan_gap():
    a = trajectory([1] * 6, [11.0] * 6)
    b = trajectory([0] * 6)
    for row in (a[-2:], b[-2:]):
        for frame in row:
            frame["scan"] += 4
    counts = track_counts(a, b)
    assert counts["frames"] == 7 and counts["predictions"] == 6
    assert counts["baseline_success"] == 1 and counts["trial_success"] == 7
    assert counts["paired_three_scan_runs"] == 1 and counts["baseline_severe_spells"] == 2
    b[-1]["raw"] = 2
    with pytest.raises(ValueError, match="unmatched"):
        track_counts(a, b)


def test_sustained_recovery_denominator_includes_censored_and_resets_at_gaps():
    a = trajectory([0] * 9, [11, 1, 1, 1, 1, 1, 11, 1, 1])
    for row in a[4:]:
        row["scan"] += 1
    counts = track_counts(a, a)
    assert counts["baseline_loss_episodes"] == 2
    assert counts["baseline_sustained_recovery_episodes"] == 1
    assert counts["baseline_unrecovered_at_end"] == 1
