from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v2.action_v27 import B3UtilityUpdater, bounded_residual_xy
from utils.tracking_metrics_v27 import (
    LocalYawBox, box_metrics, metric_contributions, local_boxes_metric_gains,
)
from utils.action_calibration_v27 import (
    calibrate_actions_v27, normalize_policy, policy_mask, summarize_rows,
    shortlist_policies, validate_action_calibration_v27, validate_scene_manifest,
    install_v27_action_calibration, action_calibration_config_identity,
    sha256_file, sha256_json, CODE_FILES,
)
from utils.v27_protocol import build_scene_manifest


def test_geometry_exact_preserves_legacy_height_and_axis_as_named_alternatives():
    a = LocalYawBox([0, 0, 0, 0], [2, 2, 2])
    b = LocalYawBox([0, 0, -1, 0], [2, 2, 4])
    assert box_metrics(a, b, up_axis=(0, 0, 1))[0] == pytest.approx(.2)
    assert box_metrics(a, b, up_axis=(0, 0, 1), mode="geometry_exact")[0] == pytest.approx(.5)
    shifted = LocalYawBox([3, 4, 0, 0], [2, 2, 2])
    assert box_metrics(a, shifted, up_axis=(0, 0, 1), dim=2)[1] == 0
    assert box_metrics(a, shifted, up_axis=(0, 0, 1), dim=2, mode="geometry_exact")[1] == 5
    assert box_metrics(a, a, up_axis=(0, 0, 1))[0] == 1
    # H3 must pass z-up explicitly; legacy y-up projection degenerates for this box.
    assert box_metrics(a, a, up_axis=(0, -1, 0))[0] == 0
    with pytest.raises(TypeError):
        box_metrics(a, b)


def test_contributions_match_actual_torch_thresholds_and_first_frame():
    values = torch.linspace(0, 1, 21)
    for overlap, distance in zip(values, values * 2):
        expected_s = torch.trapz((overlap >= values).float(), x=values).item()
        expected_p = (torch.trapz((distance <= values * 2).float(), x=values * 2) / 2).item()
        success, precision = metric_contributions(float(overlap), float(distance))
        assert success == pytest.approx(expected_s, abs=1e-7)
        assert precision == pytest.approx(expected_p, abs=1e-7)
    assert metric_contributions(1, 0) == (1, 1)
    assert metric_contributions(0, 3)[0] == pytest.approx(.025)


def test_batch_labels_use_prediction_and_target_sizes_independently():
    result = local_boxes_metric_gains(
        [[0, 0, 0, 0]], [[.5, 0, 0, 0]], [[.6, 0, 0, 0]],
        [[2, 4, 2]], [[2, 4, 3]], up_axis=(0, 0, 1))
    assert result["success_gain"][0] > 0
    assert result["precision_gain"][0] > 0
    assert result["utility_gain"][0] == pytest.approx(
        (result["success_gain"][0] + result["precision_gain"][0]) / 2)


def _inputs():
    return dict(observation_box=torch.zeros(2, 4), raw_box=torch.tensor([[3., 0, 0, 0], [0., 2, 0, 0]]),
        availability=torch.ones(2), base_evidence=torch.zeros(2, 64, requires_grad=True),
        extension_evidence=torch.zeros(2, 64, requires_grad=True),
        base_presence_probability=torch.zeros(2), extension_presence_probability=torch.zeros(2),
        observation_stats=torch.zeros(2, 5), b1_sigma_parallel_perp=torch.ones(2, 2),
        query_delta_t=torch.ones(2) * .5, gap_ratio=torch.ones(2), mode_summary=torch.zeros(2, 4))


def test_updater_uses_strict_utility_policy_without_presence_gate_and_detaches_inputs():
    updater = B3UtilityUpdater(require_calibration=True)
    inputs = _inputs()
    with torch.no_grad():
        updater.expected_success_gain_head.bias.fill_(np.arctanh(.5))
        updater.expected_precision_gain_head.bias.fill_(np.arctanh(.5))
    updater.install_policy({"kind": "threshold", "threshold": .49})
    final, output = updater(**inputs)
    assert torch.all(output["ct_b3_final_gate"] == 1)
    assert torch.allclose(torch.linalg.norm(final[:, :2], dim=1), torch.full((2,), .75))
    assert output["ct_b3_mode_summary"].shape == (2, 4)
    assert output["ct_b3_bounded_action_features"].shape == (2, 4)
    output["ct_b3_action_score"].sum().backward()
    assert inputs["base_evidence"].grad is None
    assert inputs["extension_evidence"].grad is None
    updater.install_policy({"kind": "threshold", "threshold": float(output["ct_b3_action_score"][0])})
    final, output = updater(**inputs)
    assert not output["ct_b3_final_gate"].any()
    updater.install_policy({"kind": "always"})
    assert updater(**inputs)[1]["ct_b3_final_gate"].all()
    updater.install_policy({"kind": "never"})
    assert torch.equal(updater(**inputs)[0], inputs["observation_box"])


def test_invalid_candidate_or_time_never_turns_into_an_action():
    updater = B3UtilityUpdater()
    updater.install_policy({"kind": "always"})
    inputs = _inputs()
    inputs["raw_box"][0, 0] = float("nan")
    inputs["query_delta_t"][1] = float("inf")
    final, output = updater(**inputs)
    assert torch.equal(final, inputs["observation_box"])
    assert not output["ct_b3_final_gate"].any()
    with pytest.raises(ValueError):
        normalize_policy({"kind": "threshold", "threshold": float("inf")})


def _manifest(mini=True):
    if mini:
        return build_scene_manifest({"mini_train": [f"m{i}" for i in range(8)],
                                    "mini_val": ["v0", "v1"]}, "v1.0-mini")
    return build_scene_manifest({"train_track": [f"s{i}" for i in range(350)],
                                "val": [f"v{i}" for i in range(150)]}, "v1.0-trainval")


def _rows(scene):
    rows = []
    for frame, score in enumerate([0., .9, .1]):
        obs, candidate = (1., 1.) if frame == 0 else (.5, .7)
        rows.append(dict(tracklet_id=scene + "/track", scene_id=scene,
            frame_id=frame, is_initial=int(frame == 0), structural_available=int(frame > 0),
            action_score=score, presence_score=0., action_applied=0,
            observation_success=obs, observation_precision=obs,
            candidate_success=candidate, candidate_precision=candidate,
            final_success=obs, final_precision=obs))
    return rows


def test_policy_summary_uses_all_frames_and_never_deduplicates_empty_masks():
    rows = _rows("s")
    metrics = summarize_rows(rows, {"kind": "always"})
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["one_step_net_utility"] == pytest.approx(.4 / 3)
    assert metrics["S"] == pytest.approx(80)
    assert not policy_mask([.5], [True], {"kind": "threshold", "threshold": .5})[0]
    rows[1]["structural_available"] = rows[2]["structural_available"] = 0
    shortlist, all_policies = shortlist_policies(rows)
    assert shortlist == [] and len(all_policies) == 1
    assert all_policies[0]["policy"] == {"kind": "never"}
    with pytest.raises(ValueError, match="including frame0"):
        summarize_rows(rows[1:])


def test_calibration_runs_real_policy_feedback_and_only_diagnoses_locked_dev():
    manifest = _manifest()
    cal_scene, dev_scene = manifest["scenes"]["calibration"][0], manifest["scenes"]["dev"][0]
    calls = []
    def runner(role, policy):
        calls.append((role, deepcopy(policy)))
        rows = _rows(cal_scene if role == "calibration" else dev_scene)
        apply = policy_mask([r["action_score"] for r in rows], [r["structural_available"] for r in rows], policy)
        for i in (1, 2):
            rows[i]["action_applied"] = int(apply[i])
            result = .7 if apply[i] else .5
            # The second accepted action is bad on the changed recursive state.
            if i == 2 and apply[1] and apply[2]:
                result = .1
                rows[i]["candidate_success"] = rows[i]["candidate_precision"] = .1
            rows[i]["final_success"] = rows[i]["final_precision"] = result
        return rows
    artifact = calibrate_actions_v27(_rows(cal_scene), runner, checkpoint_sha256="checkpoint",
        config_sha256="config", scene_manifest=manifest, code_sha256="code")
    assert artifact["action_policy"]["kind"] == "threshold"
    assert artifact["calibration_selected_metrics"]["actions"] == 1
    assert [policy for role, policy in calls if role == "dev"] == [artifact["action_policy"], {"kind": "never"}]
    assert artifact["parameter_training_overlap"] is False
    validate_action_calibration_v27(artifact, "checkpoint", "config", code_sha256="code")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_action_calibration_v27(artifact, "other", "config", code_sha256="code")


def test_full_scene_provenance_keeps_all_350_parameter_training_scenes():
    manifest = validate_scene_manifest(_manifest(False))
    assert len(manifest["scenes"]["train"]) == 350
    assert manifest["parameter_training_overlap"] is True
    assert set(manifest["scenes"]["calibration"]) <= set(manifest["scenes"]["train"])


def test_real_runner_dispatches_full_sequence_again_for_each_policy_and_caches_only_identical_runs():
    from tools.ct_action_v27_runtime import TrackerClosedLoopRunner
    manifest = _manifest()
    scene = manifest["scenes"]["calibration"][0]
    class Dataset:
        def __init__(self):
            self.dataset = SimpleNamespace(
                get_tracklet_key=lambda index: scene + "/track",
                virtual_rate_meta=[{"scene_token": "token"}],
                nusc=SimpleNamespace(get=lambda table, token: {"name": scene}))
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return [0, 1, 2]
    class Model:
        def __init__(self):
            self.ct_joint_router = SimpleNamespace(install_policy=self.install)
            self.config = SimpleNamespace(proposal_inference_mode="observation")
            self.calls = []
        def install(self, policy):
            self.policy = deepcopy(policy)
        def evaluate_one_sequence(self, sequence):
            self.calls.append(deepcopy(self.policy))
            rows = _rows(scene)
            mask = policy_mask([r["action_score"] for r in rows],
                               [r["structural_available"] for r in rows], self.policy)
            for row, applied in zip(rows, mask):
                row["action_applied"] = int(applied)
                if applied:
                    row["final_success"] = row["candidate_success"]
                    row["final_precision"] = row["candidate_precision"]
            self._ct_v27_sequence_endpoints = rows
    runner = TrackerClosedLoopRunner.__new__(TrackerClosedLoopRunner)
    runner.config = SimpleNamespace(seed=42, category_name="Car")
    runner.device = torch.device("cpu")
    runner.datasets, runner.cache = {"calibration": Dataset()}, {}
    runner.cache_directory, runner.model = None, Model()
    first = runner("calibration", {"kind": "never"})
    second = runner("calibration", {"kind": "always"})
    runner("calibration", {"kind": "always"})
    assert len(runner.model.calls) == 2
    assert summarize_rows(second)["S"] > summarize_rows(first)["S"]


def test_model_installer_binds_checkpoint_config_scene_source_and_falls_back_with_status(tmp_path):
    import json
    manifest = _manifest()
    splits = {"mini_train": [f"m{i}" for i in range(8)], "mini_val": ["v0", "v1"]}
    checkpoint, artifact_path = tmp_path / "model.ckpt", tmp_path / "policy.json"
    checkpoint.write_bytes(b"test checkpoint identity")
    config = {"ct_enable_v27": True, "version": "v1.0-mini", "ct_partition_seed": 42,
              "checkpoint": str(checkpoint), "ct_action_calibration_path": str(artifact_path)}
    def runner(role, policy):
        rows = _rows(manifest["scenes"][role][0])
        mask = policy_mask([r["action_score"] for r in rows], [r["structural_available"] for r in rows], policy)
        for row, applied in zip(rows, mask):
            row["action_applied"] = int(applied)
            if applied:
                row["final_success"] = row["candidate_success"]
                row["final_precision"] = row["candidate_precision"]
        return rows
    artifact = calibrate_actions_v27(runner("calibration", {"kind": "never"}), runner,
        checkpoint_sha256=sha256_file(checkpoint),
        config_sha256=sha256_json(action_calibration_config_identity(config)),
        scene_manifest=manifest, code_sha256="test-source")
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    model = SimpleNamespace(ct_joint_router=B3UtilityUpdater())
    status = install_v27_action_calibration(model, config, scene_splits=splits, code_sha256="test-source")
    assert status["loaded"] and bool(model.ct_joint_router.calibrated)
    assert model.ct_joint_router.action_policy == artifact["action_policy"]
    status = install_v27_action_calibration(model, config, scene_splits=splits, code_sha256="changed-source")
    assert not status["loaded"] and "code_content_sha256" in status["reason"]
    assert model.ct_joint_router.action_policy == {"kind": "never"}
    checkpoint.write_bytes(b"different checkpoint")
    status = install_v27_action_calibration(model, config, scene_splits=splits, code_sha256="test-source")
    assert not status["loaded"] and "checkpoint_sha256" in status["reason"]
    config["ct_action_calibration_path"] = None
    assert install_v27_action_calibration(model, config)["reason"] == "missing_calibration_artifact"
    assert "models/ct_v2/evidence_v27.py" in CODE_FILES
    assert "utils/v27_input.py" in CODE_FILES
    assert "utils/v27_training.py" in CODE_FILES
