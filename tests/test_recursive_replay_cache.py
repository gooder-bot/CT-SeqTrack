import copy
import json

import pytest

from utils.replay_cache import (
    RecursiveReplayCache,
    validate_replay_record,
    write_recursive_replay_cache,
)


def _manifest():
    return {
        "dataset": "nuscenes-mini-car",
        "split": "mini_train",
        "replay_config_sha256": "a" * 64,
        "commit": "0123456789abcdef",
        "b0_state_sha256": "0" * 64,
        "b1_state_sha256": "1" * 64,
        "b1_calibration_sha256": "2" * 64,
        "b0_checkpoint_sha256": "b" * 64,
        "b1_checkpoint_sha256": "c" * 64,
        "source_checkpoint_sha256": "d" * 64,
    }


def _record():
    return {
        "tracklet_key": "scene/sample/car",
        "frame_id": 7,
        "history_boxes_world": [
            [1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1],
            [0.5, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1],
        ],
        "history_valid_mask": [1, 1],
        "delta_t": [0.5, 0.5],
        "current_delta_t": 0.5,
        "anchor_world": [1.0, 2.0, 0.0, 0.1],
        "b1": {
            "mu_xy": [0.5, 0.0],
            "log_sigma_parallel_perp": [-0.5, -0.7],
            "covariance_xy": [[0.3, 0.0], [0.0, 0.2]],
            "basis_velocity_xy": [1.0, 0.0],
            "direction_xy": [1.0, 0.0],
            "velocity_xy": [1.0, 0.0],
            "feature": [0.0] * 128,
            "valid": True,
            "gap_ratio": 1.0,
            "source_id": 1,
        },
        "source": "recursive_b0_b1",
    }


def test_cache_roundtrip_and_manifest_match(tmp_path):
    write_recursive_replay_cache(tmp_path, _manifest(), [_record()])
    cache = RecursiveReplayCache(
        tmp_path, expected_manifest={"b1_checkpoint_sha256": "c" * 64})
    loaded = cache.get("scene/sample/car", 7)
    assert loaded == _record()
    loaded["b1"]["mu_xy"][0] = 99.0
    assert cache.get("scene/sample/car", 7)["b1"]["mu_xy"][0] == 0.5


def test_cache_rejects_manifest_and_content_hash_mismatch(tmp_path):
    write_recursive_replay_cache(tmp_path, _manifest(), [_record()])
    with pytest.raises(ValueError, match="manifest mismatch"):
        RecursiveReplayCache(
            tmp_path,
            expected_manifest={"replay_config_sha256": "e" * 64})
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        records_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        RecursiveReplayCache(tmp_path)


@pytest.mark.parametrize("field", [
    "replay_config_sha256", "b0_state_sha256", "b1_state_sha256",
    "b1_calibration_sha256", "source_checkpoint_sha256",
])
def test_cache_rejects_each_runtime_identity_mismatch(tmp_path, field):
    write_recursive_replay_cache(tmp_path, _manifest(), [_record()])
    with pytest.raises(ValueError, match=field):
        RecursiveReplayCache(
            tmp_path, expected_manifest={field: "f" * 64})


def test_formal_cache_rejects_v1_manifest(tmp_path):
    write_recursive_replay_cache(tmp_path, _manifest(), [_record()])
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        RecursiveReplayCache(tmp_path)


def test_cache_rejects_current_gt_fields():
    record = copy.deepcopy(_record())
    record["target"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="forbidden GT field"):
        validate_replay_record(record)
