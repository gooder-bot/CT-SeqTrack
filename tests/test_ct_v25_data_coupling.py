import copy
from pathlib import Path

import numpy as np
import pytest
from pyquaternion import Quaternion

from datasets.points_utils import regularize_pc_with_metadata
from ctseqtrack.config import configure_ct_variant
from ctseqtrack.runtime.calibration import (
    audit_action_thresholds,
    select_action_thresholds,
    validate_action_calibration,
)
from ctseqtrack.runtime.acquisition import build_preflight_artifact
from utils.config import load_yaml_config
from ctseqtrack.runtime.contracts import validate_scratch_training_contract
from utils.sampling_utils import physical_frame_point_seed
from ctseqtrack.runtime.scene_bootstrap import paired_scene_bootstrap
from ctseqtrack.data.recursive import (
    RecursiveTrackState,
    apply_training_reanchor,
    build_scene_partition_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


class DummyBox:
    def __init__(self, x=0.0):
        self.center = np.asarray([x, 0.0, 0.0], dtype=np.float64)
        self.wlh = np.asarray([2.0, 4.0, 1.5], dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=0.0)

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


class SceneDataset:
    def __init__(self, scenes=8, tracks_per_scene=2, duplicate_cross=False):
        self.rows = []
        for scene in range(scenes):
            for target in range(tracks_per_scene):
                self.rows.append((f"scene-{scene}", f"target-{target}", 4))
        self.duplicate_cross = duplicate_cross

    def get_num_tracklets(self):
        return len(self.rows)

    def get_num_frames_tracklet(self, tracklet_id):
        return self.rows[tracklet_id][2]

    def get_tracklet_key(self, tracklet_id):
        scene, target, _ = self.rows[tracklet_id]
        return f"{scene}/{target}"

    def get_partition_group_key(self, tracklet_id):
        return self.rows[tracklet_id][0]

    def get_frame_token(self, tracklet_id, frame_id):
        scene = self.rows[tracklet_id][0]
        if self.duplicate_cross and tracklet_id == 2 and frame_id == 0:
            return "scene-0/frame-0"
        return f"{scene}/frame-{frame_id}"


def _frame(frame_id):
    return {
        "3d_bbox": DummyBox(float(frame_id)),
        "timestamp": float(frame_id),
        "pc": object(),
    }


def _history(ids):
    return {
        "prev_frame_ids": list(ids),
        "prev_frames": {
            str(-(index + 1)): _frame(frame_id) for index, frame_id in enumerate(ids)
        },
    }


def test_scene_manifest_is_reproducible_disjoint_and_mini_is_5_1_1_1():
    dataset = SceneDataset()
    first = build_scene_partition_manifest(dataset, seed=42)
    second = build_scene_partition_manifest(dataset, seed=42)
    assert first == second
    assert [
        first["partitions"][name]["scene_count"]
        for name in ("train", "dev", "calibration_select", "calibration_audit")
    ] == [5, 1, 1, 1]
    scene_partitions = {}
    for row in first["tracklets"]:
        scene_partitions.setdefault(row["group_key"], set()).add(row["partition"])
    assert all(len(values) == 1 for values in scene_partitions.values())
    assert len(first["content_sha256"]) == 64


def test_scene_manifest_rejects_physical_frame_cross_scene_duplication():
    with pytest.raises(ValueError, match="multiple scenes"):
        build_scene_partition_manifest(SceneDataset(duplicate_cross=True), seed=42)


@pytest.mark.parametrize(
    "count,expected_unique",
    [(0, 0), (1, 0), (2, 0), (3, 3), (1023, 1023), (1024, 1024)],
)
def test_regularized_point_metadata_sparse_contract(count, expected_unique):
    points = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    result = regularize_pc_with_metadata(points, 1024, seed=7)
    assert result.points.shape == (1024, 3)
    assert result.source_indices.shape == (1024,)
    assert int(result.unique_valid_mask.sum()) == expected_unique
    assert result.raw_point_count == count
    assert result.unique_point_count == expected_unique
    if count == 0:
        assert np.all(result.points == 0)
        assert np.all(result.source_indices == -1)
    elif count <= 2:
        assert np.any(result.points != 0) or np.all(points == 0)


def test_b0_scale_and_b2_exact_scale_are_separate_in_sampler():
    source = (ROOT / "ctseqtrack" / "data" / "outputs.py").read_text(encoding="utf-8")
    compact = "".join(source.split())
    assert "this_points.T[:3,:],config.bb_scale).astype(int)" in compact
    assert "ct_b2_target_bb_scale" in source
    assert '"ct_base_evidence_labels":b2_base_labels' in compact


def test_b0_and_b1_sample_outputs_have_complete_support_branch_locals(
    monkeypatch, request,
):
    import importlib
    import sys
    import types

    geometry = types.ModuleType("nuscenes.utils.geometry_utils")
    geometry.points_in_box = (
        lambda _box, points, _scale=1.0: np.zeros(points.shape[1], dtype=bool)
    )
    nuscenes_utils = types.ModuleType("nuscenes.utils")
    nuscenes_utils.geometry_utils = geometry
    nuscenes = types.ModuleType("nuscenes")
    nuscenes.utils = nuscenes_utils
    monkeypatch.setitem(sys.modules, "nuscenes", nuscenes)
    monkeypatch.setitem(sys.modules, "nuscenes.utils", nuscenes_utils)
    monkeypatch.setitem(sys.modules, "nuscenes.utils.geometry_utils", geometry)
    sample_outputs = importlib.import_module("ctseqtrack.data.outputs")
    request.addfinalizer(
        lambda: sys.modules.pop("ctseqtrack.data.outputs", None)
    )

    point_count = 4
    history_count = 3
    empty_support = np.zeros((2, 3), dtype=np.float32)
    empty_mask = np.zeros((2,), dtype=np.float32)
    context = {
        "base_regularized": type(
            "Regularized", (), {"unique_valid_mask": np.ones(point_count)}
        )(),
        "baseline_search_points": np.zeros((point_count, 3), dtype=np.float32),
        "candidate_id": 0,
        "candidate_offsets": np.zeros((history_count, 3), dtype=np.float32),
        "candidate_shared_transform": np.zeros(3, dtype=np.float32),
        "candidate_shared_world_translation": np.zeros(3, dtype=np.float32),
        "candidate_trajectory_mode": "shared_se2",
        "canonical_label_boxs": [DummyBox(index) for index in range(history_count)],
        "canonical_ref_boxs": np.zeros((history_count, 4), dtype=np.float32),
        "canonical_this_box": DummyBox(1.0),
        "causal_temporal_policy": False,
        "config": type(
            "Config",
            (),
            {
                "bb_scale": 1.25,
                "bb_offset": 2.0,
                "point_sample_size": point_count,
                "degrees": False,
                "motion_threshold": 0.15,
            },
        )(),
        "coordinate_anchor": np.zeros(4, dtype=np.float32),
        "corner_timestamps": np.zeros(history_count, dtype=np.float32),
        "coordinate_anchor_box": DummyBox(),
        "ct_search_box": None,
        "ct_search_diagnostics": {"query_delta_t": 0.5},
        "ct_search_sampling": {
            "expansion_sample_count": 0,
            "expansion_available_count": 0,
        },
        "current_sampling_seed": 42,
        "current_timestamp": 1.0,
        "data": {},
        "default_time_step": 0.5,
        "delta_t_list": [0.5] * history_count,
        "effective_delta_t_list": [0.5] * history_count,
        "dynamics_time_mode": "true",
        "effective_current_timestamp": 1.0,
        "effective_local_timestamps": np.zeros(history_count, dtype=np.float32),
        "effective_relative_timestamps": np.zeros(history_count, dtype=np.float32),
        "expanded_search_points": np.empty((0, 3), dtype=np.float32),
        "ground_truth_history": [],
        "history_offsets": None,
        "main_current_value": 0.0,
        "main_timestamps": np.zeros(history_count, dtype=np.float32),
        "local_timestamps": np.zeros(history_count, dtype=np.float32),
        "motion_boxs": [DummyBox() for _ in range(history_count)],
        "ct_motion_ref_boxs": np.zeros((history_count, 4), dtype=np.float32),
        "motion_main_ref_boxs": np.zeros((history_count, 4), dtype=np.float32),
        "point_timestamps": [0.0] * history_count,
        "prev_boxs": [DummyBox() for _ in range(history_count)],
        "prev_points_list": [
            np.zeros((point_count, 3), dtype=np.float32)
            for _ in range(history_count)
        ],
        "recursive_history": [],
        "ref_boxs": [DummyBox() for _ in range(history_count)],
        "relative_timestamps": np.zeros(history_count, dtype=np.float32),
        "sample_index": 0,
        "search_v2_box": None,
        "search_v2_diagnostics": {"query_delta_t": 0.5},
        "search_v2_endpoint_xy": np.zeros(2, dtype=np.float32),
        "search_v2_expanded_points": np.empty((0, 3), dtype=np.float32),
        "search_v2_points": empty_support.copy(),
        "search_v2_point_valid_mask": empty_mask.copy(),
        "search_v2_point_source": np.zeros(2, dtype=np.int64),
        "search_v2_sampling": {
            "active": False,
            "available_count": 0,
            "extension_count": 0,
            "overlap_count": 0,
        },
        "this_box": DummyBox(1.0),
        "this_points": np.zeros((point_count, 3), dtype=np.float32),
        "trajectory_search_points": empty_support.copy(),
        "trajectory_search_point_valid_mask": empty_mask.copy(),
        "trajectory_search_point_source": np.zeros(2, dtype=np.int64),
        "trajectory_search_sampling": {
            "active": False,
            "sample_count": 0,
            "available_count": 0,
        },
        "joint_extension_sampling": None,
        "joint_extension_source": None,
        "use_ct_joint_full": False,
        "use_motion_v3": False,
        "valid_mask": [1] * history_count,
        "b2_v3_history_mode": None,
        "motion_anchor": np.zeros(4, dtype=np.float32),
        "motion_aux_contract": None,
        "num_hist": history_count,
        "point_sampling_seeds": None,
        "prev_frame_ids": None,
        "this_frame_id": 1,
        "use_search_evidence_v3": False,
        "use_trajectory_search": False,
    }
    monkeypatch.setattr(
        sample_outputs.geometry_utils,
        "points_in_box",
        lambda _box, points, _scale=1.0: np.zeros(points.shape[1], dtype=bool),
    )
    monkeypatch.setattr(
        sample_outputs,
        "canonical_dynamics_targets",
        lambda *_args, **_kwargs: (
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        sample_outputs,
        "anchor_relative_trajectory_targets",
        lambda *_args, **_kwargs: (
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(sample_outputs, "build_base_output", lambda _context: {})
    result = sample_outputs.build_sample_output(context)
    assert result == {"this_frame_id": 1, "current_sampling_seed": 42}

    def fake_joint_extensions(
        _baseline,
        _endpoint,
        _tube,
        *,
        endpoint_quota,
        tube_quota,
        seed,
    ):
        del seed
        count = endpoint_quota + tube_quota
        return (
            np.zeros((count, 3), dtype=np.float32),
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.int64),
            {
                "active": False,
                "sample_count": 0,
                "available_count": 0,
                "endpoint_available_count": 0,
                "tube_available_count": 0,
                "_pool_points": np.empty((0, 3), dtype=np.float32),
            },
        )

    monkeypatch.setattr(
        sample_outputs, "sample_joint_novel_extensions", fake_joint_extensions
    )
    context["use_ct_joint_full"] = True
    result = sample_outputs.build_sample_output(context)
    assert result["ct_extension_points"].shape == (256, 5)
    assert result["ct_acquisition_sampled_count"] == 0


def test_candidate_pool_seeds_depend_only_on_physical_frames():
    config = type("Config", (), {"seed": 42})()
    current = [physical_frame_point_seed(config, "track/0", 9) for _role in (0, 1, 2)]
    history = [
        physical_frame_point_seed(config, "track/0", 9, 7) for _gap in (1, 2, 4, 8)
    ]
    assert len(set(current)) == 1
    assert len(set(history)) == 1
    source = (ROOT / "datasets" / "sampler.py").read_text(encoding="utf-8")
    online = source.split("    def _online_raw_view", 1)[1].split(
        "    def __getitem__", 1
    )[0]
    assert "physical_frame_point_seed(" in online


def test_reanchor_covers_candidate_history_union_and_never_current_gt():
    raw = {
        "this_frame_id": 8,
        **_history([7, 6, 5]),
        "temporal_candidate_pool": {
            1: _history([7, 6, 5]),
            2: _history([6, 4, 2]),
            4: _history([4, 0, 0]),
            8: _history([0, 0, 0]),
        },
    }
    state = RecursiveTrackState(0, "track/0", DummyBox(100.0))
    for frame_id in range(1, 8):
        state.append(frame_id, DummyBox(100.0 + frame_id), frame_id)
    config = type("Config", (), {"ct_training_reanchor_policy": "periodic_past_gt"})()
    diagnostics = apply_training_reanchor(raw, state, 1, config)
    expected = [0, 2, 4, 5, 6, 7]
    assert diagnostics["reanchored_frame_ids"] == expected
    assert 8 not in state.predictions
    for frame_id in expected:
        assert state.predictions[frame_id].center[0] == frame_id
    assert state.predictions[3].center[0] == 103.0


def test_fixed_cv_state_write_uses_prediction_not_current_gt():
    source = (ROOT / "tools" / "export_ct_acquisition_preflight_rows.py").read_text(
        encoding="utf-8"
    )
    export = source.split("def export_partition", 1)[1]
    assert "predicted_box = _fixed_cv_state_box" in export
    assert "state.append(\n                        frame_id, predicted_box" in export
    assert 'raw["this_frame"]["3d_bbox"]' not in export
    assert "for horizon_index, horizon in enumerate(horizons)" in export


def _calibration_rows(partition, thresholds=None):
    rows = []
    for scene in range(3):
        for frame in range(4):
            row = {
                "tracklet_id": f"t-{scene}-{frame}",
                "partition_group_key": f"{partition}/scene-{scene}",
                "partition": partition,
                "structural_available": 1,
                "presence_score": 0.9,
                "action_score": 0.9,
                "center_gain": 0.2,
                "iou_gain": 0.05,
            }
            if thresholds is not None:
                row.update(
                    {
                        "rollout_mode": "selective",
                        "calibrated_presence_threshold": thresholds["presence"],
                        "calibrated_action_threshold": thresholds["action"],
                    }
                )
            rows.append(row)
    return rows


def test_b3_threshold_selection_and_scene_audit_are_strictly_separate():
    selection = select_action_thresholds(
        _calibration_rows("calibration_select"),
        "checkpoint",
        "config",
        "select-manifest",
        resamples=20,
        min_scenes=1,
        min_actions=1,
    )
    assert selection["passed"]
    audit = audit_action_thresholds(
        _calibration_rows("calibration_audit", selection["thresholds"]),
        selection,
        "checkpoint",
        "config",
        "select-manifest",
        "audit-manifest",
        resamples=20,
        min_scenes=1,
        min_actions=1,
    )
    assert audit["passed"]
    assert audit["safety_population"] == "calibration_audit_only"
    validate_action_calibration(
        audit, "checkpoint", "config", "audit-manifest", "select-manifest"
    )
    with pytest.raises(ValueError, match="not in calibration_audit"):
        audit_action_thresholds(
            _calibration_rows("calibration_select", selection["thresholds"]),
            selection,
            "checkpoint",
            "config",
            "select-manifest",
            "audit-manifest",
            resamples=5,
            min_scenes=1,
            min_actions=1,
        )
    overlapping = _calibration_rows("calibration_audit", selection["thresholds"])
    for row in overlapping:
        row["partition_group_key"] = selection["selection_scene_keys"][0]
    with pytest.raises(ValueError, match="share scenes"):
        audit_action_thresholds(
            overlapping,
            selection,
            "checkpoint",
            "config",
            "select-manifest",
            "audit-manifest",
            resamples=5,
            min_scenes=1,
            min_actions=1,
        )


def test_v25_four_arms_share_selected_protocol_without_legacy_configs():
    for name in ("25_b0.yaml", "25_b1.yaml", "25_full_minus_b3.yaml", "25_full.yaml"):
        config = load_yaml_config(ROOT / "cfgs" / "ct_seqtrack" / name)
        configure_ct_variant(config)
        validate_scratch_training_contract(config)
        assert config["ct_training_reanchor_policy"] == "periodic_past_gt"
        assert config["ct_b0_rng_protocol"] == "post_observation_shift_v1"
        assert config["ct_recursive_rollout_horizons"] == [1, 2, 4, 8]
        assert config["ct_partition_scheme"] == "scene_v2"
        assert config["ct_initialization_policy"] == "scratch_only"
    assert not list((ROOT / "cfgs" / "ct_seqtrack").glob("24*.yaml"))


def test_tracking_bootstrap_clusters_by_scene_not_tracklet():
    baseline = []
    method = []
    for scene in range(3):
        for tracklet in range(2):
            for frame in range(2):
                identity = {
                    "partition_group_key": f"scene-{scene}",
                    "tracklet_key": f"scene-{scene}/track-{tracklet}",
                    "frame_id": frame,
                }
                baseline.append({**identity, "final_iou": 0.4, "final_distance": 1.0})
                method.append({**identity, "final_iou": 0.6, "final_distance": 0.5})
    result = paired_scene_bootstrap(baseline, method, resamples=20, seed=4)
    assert result["sampling_unit"] == "scene"
    assert result["scene_count"] == 3
    assert result["paired_delta"]["success"]["lower_95"] > 0
    assert result["paired_delta"]["precision"]["lower_95"] > 0


def test_v25_preflight_class_counts_use_actual_candidate_loss_weights():
    def row(partition, candidate, positive):
        return {
            "partition": partition,
            "candidate_id": candidate,
            "pool_target_count": 10,
            "sampled_target_count": positive,
            "sampled_count": 10,
            "extension_pool_count": 10,
            "available": True,
            "role_satisfied": True,
            "boundary_ratio": 1.0,
        }

    rows = [row("train", 0, 2), row("train", 1, 4), row("train", 2, 8)]
    rows.extend(row("dev", candidate, 1) for candidate in (0, 1, 2))
    artifact = build_preflight_artifact(
        rows,
        {
            "acquisition": {
                "ct_protocol_version": 25,
                "ct_candidate_policy": "causal_b1_boundary",
            }
        },
        {"manifest": "identity"},
        42,
        min_target_bearing_rows=1,
        min_row_retention=0.0,
    )
    assert artifact["schema"] == "ct_seqtrack.acquisition_preflight.v4"
    assert artifact["role_weights"] == {"0": 0.5, "1": 0.3, "2": 0.2}
    counts = artifact["role_weighted_class_counts"]
    assert counts["positive_points"] == pytest.approx(3.8)
    assert counts["negative_points"] == pytest.approx(6.2)
