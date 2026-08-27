import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from models.ct_v2.evidence_memory import B2EvidenceAcquirer, B3SelectiveUpdater
from models.ct_v2.motion import (
    OrderedPhysicalMotionEncoder,
    acquisition_margin_pinball_loss,
    compose_b1_transaction_loss,
)
from utils.action_calibration import (
    CONSENSUS_FEATURE_SCHEMA,
    action_calibration_config_identity,
    calibrate_actions,
    sha256_json,
    validate_action_calibration,
)
from utils.config import load_yaml_config
from utils.online_contract import validate_scratch_training_contract
from utils.ct_search import (
    build_causal_history_corridor,
    diagnostic_points_in_oriented_support,
    resolve_joint_search_geometry,
    sample_bounded_novel_prepool,
)
from tools.preflight_v26_full import (
    FORMAL_CONFIGS,
    NUSCENES_TABLES,
    _load_arm,
    _validate_data_root,
)
from tools.report_ct_b2_v26 import (
    COUNTERFACTUAL_ARMS,
    COUNTERFACTUAL_METRICS,
    REQUIRED as V26_REPORT_REQUIRED,
    validate as validate_v26_report_rows,
)


ROOT = Path(__file__).resolve().parents[1]


class Box:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, wlh=(2.0, 4.0, 1.5)):
        self.center = np.asarray((x, y, 0.0), dtype=np.float64)
        self.wlh = np.asarray(wlh, dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=yaw)

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


def test_rotated_non_square_wlh_uses_local_x_length_y_width():
    box = Box(yaw=math.pi / 2.0, wlh=(2.0, 4.0, 2.0))
    rotation = box.rotation_matrix
    local = np.asarray([
        [1.5, 0.0, 0.0],   # inside length half-extent 2
        [0.0, 1.5, 0.0],   # outside width half-extent 1
    ])
    world = local @ rotation.T
    mask = diagnostic_points_in_oriented_support(world, box)
    np.testing.assert_array_equal(mask, [True, False])


def test_v26_prepool_is_novel_unique_bitmasked_and_deterministic():
    baseline = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    endpoint = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32)
    tube = np.asarray([
        [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    corridor = np.asarray([
        [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    first = sample_bounded_novel_prepool(
        baseline, endpoint, tube, corridor,
        local_quota=4, corridor_quota=3, voxel_size=0.2)
    second = sample_bounded_novel_prepool(
        baseline, endpoint, tube, corridor,
        local_quota=4, corridor_quota=3, voxel_size=0.2)
    for lhs, rhs in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(lhs, rhs)
    points, valid, source, diagnostics = first
    selected = points[valid > 0]
    assert points.shape == (7, 3)
    assert len(selected) == 4
    assert len(np.unique(selected[:, :3], axis=0)) == len(selected)
    assert not np.any(np.all(selected[:, :3] == baseline[0], axis=1))
    source_by_x = {
        float(point[0]): int(bitmask)
        for point, bitmask in zip(selected, source[valid > 0])}
    assert source_by_x[1.0] == 7
    assert source_by_x[2.0] == 1
    assert source_by_x[3.0] == 2
    assert source_by_x[4.0] == 4
    assert diagnostics["available_count"] == 4


def test_acquisition_margin_is_bounded_initially_near_two_one_and_isolated():
    encoder = OrderedPhysicalMotionEncoder(
        hidden_dim=16, step_dim=8,
        shared_kinematic_anchor=True,
        adaptive_acquisition_margin=True,
        acquisition_margin_min=(2.0, 1.0),
        acquisition_margin_max=(6.0, 3.0),
        acquisition_margin_bias=-8.0)
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                           [-1.0, 0.0, 0.0, 0.0],
                           [-2.0, 0.0, 0.0, 0.0]]])
    output = encoder(
        boxes, torch.tensor([[0.5, 0.5, 0.5]]),
        torch.ones(1, 3), torch.tensor([0.5]))
    margin = output["acquisition_margin_parallel_perp"]
    assert torch.all(margin >= torch.tensor([[2.0, 1.0]]))
    assert torch.all(margin <= torch.tensor([[6.0, 3.0]]))
    assert torch.allclose(margin, torch.tensor([[2.0, 1.0]]), atol=0.002)
    terms = acquisition_margin_pinball_loss(
        output["mu_xy"], torch.tensor([[4.0, 1.0]]),
        output["direction_xy"], margin, output["valid"], quantile=0.9)
    terms["loss_per_sample"].sum().backward()
    assert encoder.acquisition_margin_head.weight.grad is not None
    assert encoder.step_projection[0].weight.grad is None
    assert encoder.velocity_residual_head.weight.grad is None


def test_b1_transaction_reconciliation_includes_v26_margin_loss():
    reference = torch.tensor(0.0)
    losses = {
        "loss_motion_v3_prior": torch.tensor(2.0),
        "loss_ct_acquisition_margin": torch.tensor(3.0),
        "loss_motion_v3_nll": torch.tensor(5.0),
        "loss_motion_v3_aux_prior": torch.tensor(7.0),
        "loss_motion_v3_aux_nll": torch.tensor(11.0),
    }
    observed = compose_b1_transaction_loss(
        reference, losses,
        prior_weight=0.1,
        acquisition_margin_weight=0.05,
        nll_weight=0.2,
        auxiliary_prior_weight=0.3,
        auxiliary_nll_weight=0.4,
    )
    expected = 0.1 * 2 + 0.05 * 3 + 0.2 * 5 + 0.3 * 7 + 0.4 * 11
    assert observed.item() == pytest.approx(expected)


def test_corridor_uses_older_consistent_anchor_on_acceleration_violation():
    boxes = [Box(10.0), Box(0.0), Box(-1.0)]
    corridor, diagnostics = build_causal_history_corridor(
        boxes, [0.5, 0.5, 0.5], [1, 1, 1], enabled=True,
        max_speed=20.0, max_acceleration=8.0,
        max_displacement=12.0, max_length=16.0,
        width_padding=2.0, max_width=6.0)
    assert corridor is not None
    assert diagnostics["anchor_source"] == "older_consistent"
    assert diagnostics["acceleration_violation"]
    assert corridor.wlh[1] <= 16.0
    assert corridor.wlh[0] == pytest.approx(4.0)
    assert corridor.wlh[2] == pytest.approx(boxes[0].wlh[2])


def test_corridor_length_cap_still_contains_anchor_and_clipped_endpoint():
    boxes = [Box(20.0), Box(0.0), Box(-10.0)]
    corridor, diagnostics = build_causal_history_corridor(
        boxes, [0.5, 1.0, 1.0], [1, 1, 1], enabled=True,
        max_speed=20.0, max_acceleration=8.0,
        max_displacement=12.0, max_length=16.0,
        width_padding=2.0, max_width=6.0)
    assert corridor is not None
    assert diagnostics["anchor_source"] == "older_consistent"
    assert diagnostics["truncated"]
    assert diagnostics["constraint_clipped"]
    assert corridor.wlh[1] == pytest.approx(16.0)
    anchor = boxes[1].center[:2]
    endpoint = diagnostics["endpoint_center"][:2]
    for point in (anchor, endpoint):
        distance = np.linalg.norm(point - corridor.center[:2])
        assert distance <= corridor.wlh[1] / 2.0 + 1e-6


def test_b1_invalid_cv_endpoint_uses_exact_fixed_two_one_margin():
    endpoint, tube, diagnostics = resolve_joint_search_geometry(
        [Box(0.0), Box(-1.0), Box(-2.0)],
        [0.5, 0.5, 0.5], [1, 1, 1],
        prediction={"valid": False}, use_b1_prepass=True,
        use_acquisition_margin=True, fixed_margins=(2.0, 1.0),
        fallback_min_displacement=0.0)
    assert diagnostics["prior_source"] == "fallback_cv"
    assert endpoint is not None and tube is not None
    assert endpoint.wlh[0] == pytest.approx(4.0)
    assert endpoint.wlh[1] == pytest.approx(8.0)


def _v26_b2_inputs(batch=1):
    return {
        "extension_points": torch.randn(batch, 768, 5),
        "extension_valid_mask": torch.ones(batch, 768),
        "extension_source": torch.randint(1, 8, (batch, 768)),
        "current_base_features": torch.randn(batch, 1024, 64),
        "current_base_valid_mask": torch.ones(batch, 1024),
        "memory_tokens": torch.randn(batch, 36, 64),
        "memory_valid_mask": torch.ones(batch, 36),
        "observation_box": torch.randn(batch, 4),
        "observation_stats": torch.zeros(batch, 5),
        "b1_center_xy": torch.zeros(batch, 2),
        "b1_sigma_parallel_perp": torch.ones(batch, 2),
        "b1_direction_xy": torch.tensor([[1.0, 0.0]]).repeat(batch, 1),
        "b1_valid": torch.zeros(batch),
        "query_delta_t": torch.ones(batch),
        "gap_ratio": torch.ones(batch),
    }


def test_relation_selection_is_256_unique_and_b1_invalid_does_not_mask_support():
    module = B2EvidenceAcquirer(
        relation_aware_sampling=True, robust_consensus_voting=True).eval()
    inputs = _v26_b2_inputs()
    first = module(**inputs)
    second = module(**inputs)
    assert first["ct_relation_logits_prepool"].shape == (1, 768)
    assert first["ct_search_targetness_logits"].shape == (1, 256)
    assert first["ct_cross_attention_weights"].shape == (1, 4, 256, 1060)
    indices = first["ct_extension_selected_indices"][0]
    valid = first["ct_extension_selected_valid_mask"][0] > 0
    assert len(torch.unique(indices[valid])) == int(valid.sum())
    assert torch.equal(indices, second["ct_extension_selected_indices"][0])
    assert first["ct_b2_available"].item() == 1.0


def test_consensus_does_not_average_distant_modes_and_covariance_is_psd():
    votes = torch.tensor([[[-3.0, 0.0], [-2.9, 0.0],
                           [3.0, 0.0], [3.1, 0.0]]])
    weights = torch.tensor([[0.9, 0.8, 0.3, 0.2]])
    result = B2EvidenceAcquirer._consensus_vote(
        votes, weights, torch.ones(1, 4, dtype=torch.bool),
        torch.zeros(1, 2))
    assert result["center"][0, 0] < -2.5
    assert abs(float(result["center"][0, 0])) > 2.0
    eigenvalues = torch.linalg.eigvalsh(result["covariance"][0])
    assert torch.all(eigenvalues >= -1e-7)
    assert torch.isfinite(result["margin"]).all()


def test_empty_v26_pool_is_exact_observation_fallback():
    module = B2EvidenceAcquirer(
        relation_aware_sampling=True, robust_consensus_voting=True).eval()
    inputs = _v26_b2_inputs()
    inputs["extension_valid_mask"].zero_()
    output = module(**inputs)
    assert torch.equal(output["ct_b2_raw_box"], inputs["observation_box"])
    assert output["ct_b2_available"].item() == 0.0


def _rows(prefix, center_gain=0.2):
    return [{
        "tracklet_id": f"{prefix}{tracklet}",
        "structural_available": 1,
        "presence_score": 0.9,
        "action_score": 0.8,
        "center_gain": center_gain,
        "iou_gain": 0.03 if center_gain >= 0 else -0.03,
    } for tracklet in range(30) for _ in range(4)]


def test_calibration_v2_locks_on_calibration_and_promotes_on_disjoint_dev():
    artifact = calibrate_actions(
        _rows("cal"), "checkpoint", "config", "cal-manifest",
        dev_rows=_rows("dev"),
        dev_tracklet_manifest_sha256="dev-manifest",
        code_version="commit", resamples=100)
    assert artifact["schema"] == "ct_seqtrack.action_calibration.v2"
    assert artifact["passed"]
    assert artifact["consensus_feature_schema"] == CONSENSUS_FEATURE_SCHEMA
    validate_action_calibration(
        artifact, "checkpoint", "config", "cal-manifest",
        "dev-manifest", code_version="commit")
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_actions(
            _rows("same"), "checkpoint", "config", "cal-manifest",
            dev_rows=_rows("same"),
            dev_tracklet_manifest_sha256="dev-manifest", resamples=10)


def test_calibration_v2_failed_dev_installs_one_one_fail_closed_thresholds():
    artifact = calibrate_actions(
        _rows("cal"), "checkpoint", "config", "cal-manifest",
        dev_rows=_rows("dev", center_gain=-0.2),
        dev_tracklet_manifest_sha256="dev-manifest",
        code_version="commit", resamples=100)
    assert not artifact["passed"]
    assert artifact["thresholds"] == {"presence": 1.0, "action": 1.0}


def test_action_calibration_binds_resolved_mechanism_config_not_paths():
    config = load_yaml_config(
        ROOT / "cfgs" / "ct_seqtrack" / "26_full_nuscenes_full.yaml")
    baseline = sha256_json(action_calibration_config_identity(config))
    relocated = dict(config)
    relocated.update({
        "path": "/another/data/root",
        "checkpoint": "/another/checkpoint.ckpt",
        "ct_action_calibration_path": "/another/calibration.json",
        "test_split": "train_track",
        "proposal_inference_mode": "selective",
    })
    assert sha256_json(action_calibration_config_identity(relocated)) == baseline
    changed = dict(config)
    changed["ct_relation_topk"] = int(changed["ct_relation_topk"]) - 1
    assert sha256_json(action_calibration_config_identity(changed)) != baseline


def test_v26_formal_configs_are_full_nuscenes_scratch_and_unfrozen():
    expected_backends = {
        "26_b0_nuscenes_full.yaml": "gru",
        "26_b1_gru_nuscenes_full.yaml": "gru",
        "26_b1_cfc_nuscenes_full.yaml": "cfc",
        "26_full_minus_b3_nuscenes_full.yaml": "gru",
        "26_full_nuscenes_full.yaml": "gru",
    }
    for name, expected_backend in expected_backends.items():
        config = load_yaml_config(ROOT / "cfgs" / "ct_seqtrack" / name)
        assert config["version"] == "v1.0-trainval"
        assert config["seed"] == 42 and config["epoch"] == 60
        assert config["ct_initialization_policy"] == "scratch_only"
        assert not config["ct_separate_optimizers"]
        assert not config.get("b2_v3_freeze_candidate_producers", False)
        assert config["motion_v3_temporal_backend"] == expected_backend
        assert config["trainer_devices"] == 1
        assert config["ct_keep_final_window_checkpoints"] == 3
        assert config["save_top_k"] == 0
        assert config["check_val_every_n_epoch"] == 2
        validate_scratch_training_contract(config)

        one_batch = dict(config)
        one_batch["limit_train_batches"] = 1
        with pytest.raises(ValueError, match="all batches"):
            validate_scratch_training_contract(one_batch)

    strict = load_yaml_config(
        ROOT / "cfgs" / "26_seqtrack_strict_nuscenes_full.yaml")
    assert strict["ct_b0_initialization_policy"] == "scratch_only"
    assert strict["ct_formal_resume_contract"] is True
    assert strict["trainer_devices"] == 1
    assert strict["ct_keep_final_window_checkpoints"] == 3
    assert strict["save_top_k"] == 0
    assert strict["check_val_every_n_epoch"] == 2
    validate_scratch_training_contract(strict)


def test_v26_zero_step_preflight_accepts_all_registered_arms(tmp_path):
    table_root = tmp_path / "v1.0-trainval"
    table_root.mkdir()
    for name in NUSCENES_TABLES:
        (table_root / name).write_text("[]", encoding="utf-8")
    lidar_root = tmp_path / "samples" / "LIDAR_TOP"
    lidar_root.mkdir(parents=True)
    (lidar_root / "sample.pcd.bin").write_bytes(b"test")
    data_root = _validate_data_root(tmp_path, minimum_lidar_files=1)
    for arm in FORMAL_CONFIGS:
        config = _load_arm(arm, data_root)
        assert config["epoch"] == 60
        assert config["from_epoch"] == 0
        assert config["trainer_devices"] == 1
        assert config["ct_keep_final_window_checkpoints"] == 3


def test_v26_trainer_has_no_intermediate_stop_or_batch_truncation_contract():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "min_epochs=cfg.epoch, max_epochs=cfg.epoch" in source
    assert "max_steps=-1" in source
    assert "fast_dev_run=False" in source
    assert "EarlyStopping" not in source
    assert "Timer(" not in source
    assert "('loss_ct_acquisition_margin'," in (
        ROOT / "models" / "seqtrack3d.py").read_text(encoding="utf-8")


def test_v26_report_contract_requires_finite_relation_calibration_metrics():
    row = {key: "0" for key in V26_REPORT_REQUIRED}
    row.update({
        "acquisition_schema_version": "3",
        "candidate_id": "0",
        "tracklet_id": "tracklet",
        "frame_id": "1",
        "relation_auroc": "0.5",
        "relation_ap": "0.25",
        "relation_auprc": "0.25",
        "relation_ece": "0.1",
    })
    for arm in COUNTERFACTUAL_ARMS:
        for metric in COUNTERFACTUAL_METRICS:
            row[f"cf_{arm}_{metric}"] = "0"
    validate_v26_report_rows([row])
    row["relation_ece"] = "1.1"
    with pytest.raises(ValueError, match="relation_ece"):
        validate_v26_report_rows([row])
