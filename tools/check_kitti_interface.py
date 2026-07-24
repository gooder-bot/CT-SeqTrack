"""Dataset-free KITTI/HTV interface smoke test.

The script creates a temporary, minimal KITTI Tracking tree and checks:

* calibration, label and point-cloud loading;
* original-frame timestamps;
* multi-frame training sampler output;
* paired-history endpoint alignment;
* official all-phase KITTI-HTV selection and frozen-manifest replay;
* shuffled dynamics-time manifest generation and consumption.
"""

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset  # noqa: E402


SYNTHETIC_FRAME_IDS = [0, 1, 2, 4, 5, 8, 9, 10, 14, 15, 16, 20]


def _write_synthetic_kitti(root):
    training = root / "training"
    label_dir = training / "label_02"
    calib_dir = training / "calib"
    velo_dir = training / "velodyne" / "0000"
    label_dir.mkdir(parents=True)
    calib_dir.mkdir(parents=True)
    velo_dir.mkdir(parents=True)

    calibration = (
        "Tr_velo_cam: "
        "1 0 0 0 "
        "0 1 0 0 "
        "0 0 1 0\n"
    )
    (calib_dir / "0000.txt").write_text(
        calibration, encoding="utf-8")

    label_rows = []
    rng = np.random.default_rng(42)
    for frame_id in SYNTHETIC_FRAME_IDS:
        center_x = 0.05 * frame_id
        height, width, length = 1.5, 1.6, 3.9
        bottom_y, center_z = 0.75, 10.0
        label_rows.append(
            f"{frame_id} 1 Car 0 0 0 "
            f"0 0 50 50 {height} {width} {length} "
            f"{center_x} {bottom_y + height / 2.0} {center_z} 0\n")

        xyz = np.column_stack((
            center_x + rng.uniform(-0.8, 0.8, size=96),
            bottom_y + rng.uniform(-0.6, 0.6, size=96),
            center_z + rng.uniform(-0.5, 0.5, size=96),
        )).astype(np.float32)
        intensity = np.ones((xyz.shape[0], 1), dtype=np.float32)
        np.concatenate((xyz, intensity), axis=1).tofile(
            velo_dir / f"{frame_id:06d}.bin")
    (label_dir / "0000.txt").write_text(
        "".join(label_rows), encoding="utf-8")
    return training


def _load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return EasyDict(yaml.load(handle, Loader=yaml.FullLoader))


def _base_config(config_path, data_root):
    cfg = _load_config(config_path)
    cfg.path = str(data_root)
    cfg.preloading = False
    cfg.workers = 0
    cfg.kitti_scene_ids = [0]
    cfg.train_split = "train"
    cfg.val_split = "train"
    cfg.test_split = "train"
    return cfg


def _assert_close(actual, expected, label):
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-7):
        raise AssertionError(
            f"{label}: expected {expected}, found {actual}")


def _standard_and_sampler_checks(cfg):
    wrapped = get_dataset(
        cfg,
        type=cfg.train_type,
        split=cfg.train_split,
        protocol_role="train",
    )
    dataset = wrapped.dataset
    if dataset.get_num_tracklets() != 1:
        raise AssertionError("Synthetic KITTI must contain one tracklet")
    if dataset.get_num_frames_total() != len(SYNTHETIC_FRAME_IDS):
        raise AssertionError("Synthetic KITTI frame count mismatch")

    frames = dataset.get_frames(0, [0, 3, 5])
    expected_ids = [0, 4, 8]
    expected_times = [0.0, 0.4, 0.8]
    for frame, frame_id, timestamp in zip(
            frames, expected_ids, expected_times):
        if frame["frame_id"] != frame_id:
            raise AssertionError(
                f"Original frame id lost: {frame['frame_id']} != {frame_id}")
        _assert_close(frame["timestamp"], timestamp, "physical timestamp")
        _assert_close(
            frame["_ct_effective_timestamp"],
            timestamp,
            "true effective timestamp",
        )

    sample_index = 5 * int(cfg.num_candidates)
    sample = wrapped[sample_index]
    if sample["points"].shape != (
            (int(cfg.hist_num) + 1) * int(cfg.point_sample_size), 5):
        raise AssertionError(
            f"Unexpected training point shape: {sample['points'].shape}")
    expected_delta = np.asarray([0.3, 0.1, 0.2], dtype=np.float32)
    if not np.allclose(
            sample["delta_t"], expected_delta, rtol=0.0, atol=1e-6):
        raise AssertionError(
            f"Original KITTI gaps not preserved: {sample['delta_t']} "
            f"!= {expected_delta}")
    return dataset.virtual_rate_selection_sha256


def _paired_history_check(cfg):
    paired_cfg = copy.deepcopy(cfg)
    paired_cfg.use_m3_path_distillation = True
    paired_cfg.m3_candidate_zero_only = False
    paired_cfg.m3_view_a_offsets = [1, 2, 3]
    paired_cfg.m3_view_b_offsets = [1, 3, 5]
    paired = get_dataset(
        paired_cfg,
        type=paired_cfg.train_type,
        split=paired_cfg.train_split,
        protocol_role="train",
    )
    sample_index = 7 * int(cfg.num_candidates)
    sample = paired[sample_index]
    if set(sample) != {"view_a", "view_b"}:
        raise AssertionError("M3 paired KITTI sample is malformed")
    view_a, view_b = sample["view_a"], sample["view_b"]
    _assert_close(
        view_a["current_timestamp"],
        view_b["current_timestamp"],
        "paired current timestamp",
    )
    point_count = int(cfg.point_sample_size)
    if not np.array_equal(
            view_a["points"][-point_count:, :3],
            view_b["points"][-point_count:, :3]):
        raise AssertionError(
            "Paired KITTI views do not share current sampled XYZ")


def _preload_cache_check(cfg):
    preload_cfg = copy.deepcopy(cfg)
    preload_cfg.preloading = True
    first = get_dataset(
        preload_cfg,
        type="test",
        split=preload_cfg.test_split,
        protocol_role="test",
    ).dataset
    first_frame = first.get_frames(0, [3])[0]
    second = get_dataset(
        preload_cfg,
        type="test",
        split=preload_cfg.test_split,
        protocol_role="test",
    ).dataset
    second_frame = second.get_frames(0, [3])[0]
    if first_frame["frame_id"] != second_frame["frame_id"]:
        raise AssertionError("KITTI preload cache changed frame identity")
    if not np.array_equal(
            first_frame["pc"].points, second_frame["pc"].points):
        raise AssertionError("KITTI preload cache changed point values")


def _mixed_interval_training_check(cfg):
    mixed_cfg = copy.deepcopy(cfg)
    mixed_cfg.test_kitti_hv_interval = "all"
    mixed = get_dataset(
        mixed_cfg,
        type="test",
        split=mixed_cfg.test_split,
        protocol_role="test",
    ).dataset
    expected_tracklets = sum(
        min(len(SYNTHETIC_FRAME_IDS), interval)
        for interval in (1, 2, 3, 5, 10)
    )
    expected_frames = len(SYNTHETIC_FRAME_IDS) * 5
    if mixed.get_num_tracklets() != expected_tracklets:
        raise AssertionError(
            f"Mixed KITTI-HV tracklet count {mixed.get_num_tracklets()} "
            f"!= {expected_tracklets}")
    if mixed.get_num_frames_total() != expected_frames:
        raise AssertionError(
            f"Mixed KITTI-HV frame count {mixed.get_num_frames_total()} "
            f"!= {expected_frames}")


def _htv_and_manifest_checks(cfg, temp_root):
    cadence_manifest = temp_root / "kitti_htv_interval2.json"
    stride_cfg = copy.deepcopy(cfg)
    stride_cfg.test_kitti_hv_interval = 2
    stride_cfg.test_virtual_rate_mode = "manifest"
    stride_cfg.test_virtual_rate_manifest = str(cadence_manifest)
    stride_cfg.test_virtual_rate_manifest_allow_create = True
    stride_cfg.test_virtual_rate_manifest_require_commit_match = False

    wrapped = get_dataset(
        stride_cfg,
        type="test",
        split=stride_cfg.test_split,
        protocol_role="test",
    )
    dataset = wrapped.dataset
    expected_frame_ids = [
        [0, 2, 5, 9, 14, 16],
        [1, 4, 8, 10, 15, 20],
    ]
    found_frame_ids = [
        [int(anno["frame"]) for anno in tracklet]
        for tracklet in dataset.tracklet_anno_list
    ]
    if found_frame_ids != expected_frame_ids:
        raise AssertionError(
            f"KITTI-HTV interval-2 selection mismatch: "
            f"{found_frame_ids} != {expected_frame_ids}")
    if not cadence_manifest.is_file():
        raise AssertionError("KITTI cadence manifest was not created")

    replay_cfg = copy.deepcopy(cfg)
    replay_cfg.test_kitti_hv_interval = 2
    replay_cfg.test_virtual_rate_mode = "manifest"
    replay_cfg.test_virtual_rate_manifest = str(cadence_manifest)
    replay_cfg.test_virtual_rate_manifest_allow_create = False
    replay_cfg.test_virtual_rate_manifest_require_commit_match = False
    replay = get_dataset(
        replay_cfg,
        type="test",
        split=replay_cfg.test_split,
        protocol_role="test",
    ).dataset
    replay_ids = [
        [int(anno["frame"]) for anno in tracklet]
        for tracklet in replay.tracklet_anno_list
    ]
    if replay_ids != expected_frame_ids:
        raise AssertionError("Frozen KITTI cadence manifest replay changed")
    if (replay.virtual_rate_selection_sha256
            != dataset.virtual_rate_selection_sha256):
        raise AssertionError("KITTI cadence selection SHA changed on replay")

    wrong_interval_cfg = copy.deepcopy(replay_cfg)
    wrong_interval_cfg.test_kitti_hv_interval = 3
    try:
        get_dataset(
            wrong_interval_cfg,
            type="test",
            split=wrong_interval_cfg.test_split,
            protocol_role="test",
        )
    except ValueError as exc:
        if "kitti_hv_intervals" not in str(exc):
            raise AssertionError(
                f"Wrong KITTI interval failed for an unexpected reason: {exc}")
    else:
        raise AssertionError(
            "Interval-2 KITTI manifest was accepted as interval 3")

    dynamics_manifest = temp_root / "kitti_htv_shuffled_dt.json"
    dynamics_summary = replay.build_dynamics_time_manifest(
        dynamics_manifest, seed=42)
    shuffled_cfg = copy.deepcopy(replay_cfg)
    shuffled_cfg.test_dynamics_time_mode = "shuffled"
    shuffled_cfg.dynamics_time_manifest_test = str(dynamics_manifest)
    shuffled_cfg.test_dynamics_time_manifest_require_commit_match = False
    shuffled = get_dataset(
        shuffled_cfg,
        type="test",
        split=shuffled_cfg.test_split,
        protocol_role="test",
    ).dataset
    shuffled_frames = shuffled.get_frames(
        0, list(range(shuffled.get_num_frames_tracklet(0))))
    physical = np.asarray(
        [frame["timestamp"] for frame in shuffled_frames])
    effective = np.asarray(
        [frame["_ct_effective_timestamp"] for frame in shuffled_frames])
    if np.array_equal(physical[1:] - physical[:-1],
                      effective[1:] - effective[:-1]):
        raise AssertionError(
            "Shuffled KITTI dynamics time did not change any transition")

    return {
        "selected_frame_ids_by_phase": expected_frame_ids,
        "cadence_manifest_content_sha256":
            replay.virtual_rate_manifest_content_sha256,
        "dynamics_manifest_content_sha256":
            dynamics_summary["content_sha256"],
        "physical_timestamps": physical.tolist(),
        "effective_timestamps": effective.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default=str(ROOT / "cfgs" / "seqtrack3d_kitti.yaml"),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ct_seqtrack_kitti_") as tmp:
        temp_root = Path(tmp)
        data_root = _write_synthetic_kitti(temp_root)
        cfg = _base_config(args.cfg, data_root)
        standard_sha = _standard_and_sampler_checks(cfg)
        _paired_history_check(cfg)
        _preload_cache_check(cfg)
        _mixed_interval_training_check(cfg)
        result = _htv_and_manifest_checks(cfg, temp_root)
        result["standard_selection_sha256"] = standard_sha
        result["status"] = "PASS"
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
