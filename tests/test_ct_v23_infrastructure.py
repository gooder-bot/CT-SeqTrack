import copy
from pathlib import Path

import pytest
import torch

from models.ct_variant import configure_ct_variant
from utils.acquisition_metrics import acquisition_config_identity
from utils.acquisition_metrics import balanced_targetness_class_weights
from utils.acquisition_metrics import build_preflight_artifact
from utils.acquisition_metrics import validate_preflight_artifact
from utils.online_contract import (
    build_b2_method_contract,
    build_online_resume_contract,
    require_scratch_initialization,
    validate_scratch_training_contract,
    validate_b2_method_promotion,
    validate_online_resume_contract,
)
from utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
from utils.training_isolation import (
    advance_lightning_manual_transaction,
    candidate_stratified_mean,
    freeze_batchnorm_running_stats,
    update_cumulative_binary_class_balance,
)
from tools.build_ct_b2_promotion_metrics import build_metrics, success_auc


def base_config():
    return {
        "experiment_name": "scratch-e2-seed42",
        "seed": 42,
        "ct_joint_contract_version": 3,
        "ct_online_recursive_training": True,
        "use_ct_joint_full": True,
        "ct_enable_b1": True,
        "ct_enable_b2": True,
        "ct_enable_b3": False,
        "num_candidates": 4,
        "ct_recursive_candidate_views": 4,
        "ct_b0_candidate_views": 4,
        "ct_b0_candidate_weights": [
            0.5, 0.1666667, 0.1666667, 0.1666667],
        "ct_b2_candidate_views": 1,
        "ct_recursive_tracklet_slots": 16,
        "ct_recursive_rollout_horizons": [1, 2, 4, 8],
        "ct_recursive_reseed_enabled": True,
        "ct_b0_rng_shift_control": True,
        "ct_router_partition": "train",
        "ct_partition_seed": 42,
        "optimizer": "Adam",
        "lr": 1e-4,
        "ct_b0_lr": 1e-4,
        "ct_plugin_lr": 1e-4,
        "lr_decay_step": 20,
        "lr_decay_rate": 0.1,
        "ct_separate_optimizers": True,
        "ct_manual_amp_enabled": False,
        "batch_size": 16,
        "ct_auxiliary_microbatch_size": 16,
        "ct_recovery_candidate_policy": "off",
        "ct_targetness_class_weight_source": "online_canonical_preflight",
        "ct_targetness_positive_weight": 1.0,
        "ct_targetness_negative_weight": 1.0,
        "export_proposal_diagnostics": True,
        "ct_initialization_policy": "scratch_only",
        "dataset": "nuscenes",
        "train_split": "train_track",
        "path": "data/nuscenes",
    }


def test_resume_contract_is_config_driven_and_epoch_boundary_only():
    config = base_config()
    checkpoint = {
        "ct_online_resume_contract": build_online_resume_contract(config),
        "ct_epoch_boundary_complete": True,
        "ct_global_rng_state": {
            "schema": "ct_seqtrack.global_rng.v1"},
    }
    validate_online_resume_contract(checkpoint, config)
    changed = dict(config, use_ct_joint_full=False)
    with pytest.raises(ValueError, match="use_ct_joint_full"):
        validate_online_resume_contract(checkpoint, changed)
    with pytest.raises(ValueError, match="max_epochs"):
        validate_online_resume_contract(checkpoint, dict(config, epoch=5))
    mid_epoch = copy.deepcopy(checkpoint)
    mid_epoch["ct_epoch_boundary_complete"] = False
    with pytest.raises(ValueError, match="epoch boundary"):
        validate_online_resume_contract(mid_epoch, config)


def test_b0_resume_contract_accepts_joint_full_false():
    config = base_config()
    config.update({
        "experiment_name": "b0-2x2-a",
        "use_ct_joint_full": False,
        "ct_enable_b1": False,
        "ct_enable_b2": False,
        "num_candidates": 1,
        "ct_recursive_candidate_views": 1,
        "ct_b0_candidate_views": 1,
        "ct_b0_candidate_weights": [1.0],
    })
    checkpoint = {
        "ct_online_resume_contract": build_online_resume_contract(config),
        "ct_epoch_boundary_complete": True,
        "ct_global_rng_state": {
            "schema": "ct_seqtrack.global_rng.v1"},
    }
    validate_online_resume_contract(checkpoint, config)


def test_old_candidate_semantics_resume_schema_is_rejected():
    config = base_config()
    checkpoint = {
        "ct_online_resume_contract": {
            "schema": "ct_seqtrack.online_resume_contract.v4",
            "fields": build_online_resume_contract(config)["fields"],
        },
        "ct_epoch_boundary_complete": True,
        "ct_global_rng_state": {
            "schema": "ct_seqtrack.global_rng.v1"},
    }
    with pytest.raises(ValueError, match="online_resume_contract.v5"):
        validate_online_resume_contract(checkpoint, config)


def test_online_targetness_balance_matches_preflight_formula():
    assert balanced_targetness_class_weights(0, 17) == {
        "positive": 1.0, "negative": 1.0}
    weights = balanced_targetness_class_weights(20, 80)
    assert weights["positive"] == pytest.approx(2.5)
    assert weights["negative"] == pytest.approx(0.625)


def test_online_targetness_balance_accumulates_without_a_launch_gate():
    positive_count = torch.zeros((), dtype=torch.float64)
    negative_count = torch.zeros((), dtype=torch.float64)
    positive_weight = torch.ones((), dtype=torch.float64)
    negative_weight = torch.ones((), dtype=torch.float64)
    update_cumulative_binary_class_balance(
        positive_count, negative_count, positive_weight, negative_weight,
        0, 40)
    assert positive_weight.item() == 1.0
    assert negative_weight.item() == 1.0
    update_cumulative_binary_class_balance(
        positive_count, negative_count, positive_weight, negative_weight,
        10, 40)
    expected = balanced_targetness_class_weights(10, 80)
    assert positive_count.item() == 10
    assert negative_count.item() == 80
    assert positive_weight.item() == pytest.approx(expected["positive"])
    assert negative_weight.item() == pytest.approx(expected["negative"])


def test_scratch_only_rejects_init_but_not_resume_argument():
    config = base_config()
    with pytest.raises(ValueError, match="forbids --init_checkpoint"):
        require_scratch_initialization(config, "old-b0.ckpt")
    require_scratch_initialization(config, None)


def test_scratch_contract_rejects_finetune_lr_and_wrong_geometry():
    config = base_config()
    validate_scratch_training_contract(config)
    with pytest.raises(ValueError, match="ct_b0_lr"):
        validate_scratch_training_contract(dict(config, ct_b0_lr=2.5e-5))
    with pytest.raises(ValueError, match="candidate_views"):
        validate_scratch_training_contract(dict(
            config, ct_recursive_candidate_views=3))


def test_b1_and_fixed_cv_ablation_configs_are_independent_scratch_runs():
    b1 = load_yaml_config(
        ROOT / 'cfgs' / 'ct_v2' / '23_ct_b1_acquisition.yaml')
    assert b1['seed'] == 42
    assert b1['checkpoint_monitor'] == 'b1_nll/dev'
    assert b1['checkpoint_mode'] == 'min'
    assert b1['ct_initialization_policy'] == 'scratch_only'
    fixed_cv = load_yaml_config(
        ROOT / 'cfgs' / 'ct_seqtrack' / '24_full_minus_b3_cv.yaml')
    configure_ct_variant(fixed_cv)
    assert fixed_cv['seed'] == 42
    assert not fixed_cv['ct_enable_b1']
    assert fixed_cv['ct_enable_b2']
    assert not fixed_cv['ct_enable_b3']
    validate_scratch_training_contract(fixed_cv)
    for name, mode in (
            ('24_full_memory_empty.yaml', 'empty'),
            ('24_full_memory_time_misaligned.yaml', 'time_misaligned')):
        full_control = load_yaml_config(
            ROOT / 'cfgs' / 'ct_seqtrack' / name)
        configure_ct_variant(full_control)
        assert full_control['ct_enable_b3']
        assert full_control['ct_memory_mode'] == mode
        validate_scratch_training_contract(full_control)


def test_full_memory_controls_use_the_same_method_only_promotion():
    source = load_yaml_config(
        ROOT / 'cfgs' / 'ct_seqtrack' / '24_full_minus_b3.yaml')
    configure_ct_variant(source)
    promotion = {
        "schema": "ct_seqtrack.b2_evidence_promotion.v4",
        "passed": True,
        "b2_method_contract": build_b2_method_contract(source),
    }
    for name in (
            '24_full_memory_real.yaml',
            '24_full_memory_empty.yaml',
            '24_full_memory_time_misaligned.yaml'):
        config = load_yaml_config(ROOT / 'cfgs' / 'ct_seqtrack' / name)
        configure_ct_variant(config)
        validate_b2_method_promotion(promotion, config)


class _Tracker:
    def __init__(self):
        self.ready = 0
        self.completed = 0

    def increment_ready(self):
        self.ready += 1

    def increment_completed(self):
        self.completed += 1


def test_manual_dual_optimizer_has_one_hundred_logical_steps():
    tracker = _Tracker()
    manual = type("Manual", (), {"optim_step_progress": tracker})()
    epoch_loop = type("Epoch", (), {"manual_optimization": manual})()
    fit_loop = type("Fit", (), {"epoch_loop": epoch_loop})()
    trainer = type("Trainer", (), {"fit_loop": fit_loop})()
    for _ in range(100):
        advance_lightning_manual_transaction(trainer)
    assert tracker.ready == 100
    assert tracker.completed == 100


def test_three_auxiliary_microbatches_match_one_48_row_transaction():
    torch.manual_seed(9)
    inputs = torch.randn(48, 7)
    valid = (torch.rand(48, 7) > 0.35).float()
    candidate_ids = torch.arange(1, 4).repeat_interleave(16)

    full_parameter = torch.nn.Parameter(torch.tensor(0.7))
    full_values = (full_parameter * inputs).square()
    full_loss = 0.5 * candidate_stratified_mean(
        full_values, valid, candidate_ids)
    full_gradient = torch.autograd.grad(full_loss, full_parameter)[0]

    micro_parameter = torch.nn.Parameter(torch.tensor(0.7))
    micro_loss = micro_parameter.new_zeros(())
    for candidate_id in (1, 2, 3):
        rows = candidate_ids == candidate_id
        values = (micro_parameter * inputs[rows]).square()
        micro_loss = micro_loss + candidate_stratified_mean(
            values, valid[rows], candidate_ids[rows]) / 6.0
    micro_gradient = torch.autograd.grad(micro_loss, micro_parameter)[0]
    assert torch.equal(full_loss, micro_loss)
    assert torch.equal(full_gradient, micro_gradient)


def test_auxiliary_batchnorm_buffers_freeze_but_affine_gradient_flows():
    torch.manual_seed(12)
    batch_norm = torch.nn.BatchNorm1d(4)
    batch_norm.train()
    running_mean = batch_norm.running_mean.clone()
    running_variance = batch_norm.running_var.clone()
    with freeze_batchnorm_running_stats(batch_norm):
        batch_norm(torch.randn(8, 4)).square().mean().backward()
    assert torch.equal(batch_norm.running_mean, running_mean)
    assert torch.equal(batch_norm.running_var, running_variance)
    assert batch_norm.weight.grad is not None
    assert torch.isfinite(batch_norm.weight.grad).all()


def test_auxiliary_objective_has_no_plugin_gradient_path():
    b0 = torch.nn.Parameter(torch.tensor(0.7))
    plugin = torch.nn.Parameter(torch.tensor(-0.4))
    auxiliary = sum(
        weight * (b0 * scale).square()
        for weight, scale in zip(
            (1 / 6, 1 / 6, 1 / 6), (1.0, 2.0, 3.0)))
    b0_gradient, plugin_gradient = torch.autograd.grad(
        auxiliary, (b0, plugin), allow_unused=True)
    assert b0_gradient is not None and torch.isfinite(b0_gradient)
    assert plugin_gradient is None


def _row(partition, candidate, pool, sampled, positive=False,
         fallback=False):
    return {
        "partition": partition,
        "candidate_id": candidate,
        "pool_target_count": pool,
        "sampled_target_count": sampled,
        "extension_pool_count": 10,
        "available": True,
        "recovery_positive": positive,
        "recovery_fallback": fallback,
    }


def _manifest_identity(config):
    from utils.acquisition_metrics import sha256_json
    manifest = {
        "schema": "ct_seqtrack.acquisition_data_manifest.v2",
        "dataset": config["dataset"],
        "split": config["train_split"],
        "path": config["path"],
        "seed": config["seed"],
        "checkpoint_loaded": False,
        "complete": True,
        "partitions": [
            {"partition": "train", "tracklet_identity_sha256": "a",
             "complete": True, "exported_rows": 16,
             "expected_rows": 16},
            {"partition": "dev", "tracklet_identity_sha256": "b",
             "complete": True, "exported_rows": 16,
             "expected_rows": 16},
        ],
    }
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_json(manifest),
    }


def test_preflight_keeps_row_and_point_recall_distinct():
    rows = [_row("dev", 0, 100, 100)]
    rows.extend(_row("dev", 0, 1, 0) for _ in range(9))
    rows.extend([
        _row("train", 0, 2, 1),
        _row("train", 0, 0, 0),
    ])
    artifact = build_preflight_artifact(
        rows, {"config": "x"}, {"manifest": "y"}, 42,
        min_target_bearing_rows=1, min_point_retention=0.1)
    primary = next(item for item in artifact["groups"]
                   if item["partition"] == "dev"
                   and item["candidate_id"] == 0)
    assert primary["row_recall"] == pytest.approx(0.1)
    assert primary["point_recall"] == pytest.approx(100 / 109)
    assert artifact["passed"]


def test_preflight_fails_closed_on_zero_eligible_primary_rows():
    rows = [
        _row("dev", 0, 0, 0),
        _row("train", 0, 1, 1),
        _row("train", 0, 0, 0),
    ]
    artifact = build_preflight_artifact(rows, {}, {}, 42)
    assert not artifact["passed"]
    assert not artifact["criteria"][
        "dev_candidate0_target_bearing_extension_nonzero"]


def test_preflight_hash_and_sampling_contract_are_fail_closed():
    rows = [
        *[_row("dev", 0, 1, 1) for _ in range(100)],
        _row("train", 0, 1, 1),
        _row("train", 0, 0, 0),
    ]
    config = base_config()
    artifact = build_preflight_artifact(
        rows, {
            "acquisition": acquisition_config_identity(config),
        }, _manifest_identity(config), 42)
    validate_preflight_artifact(artifact, config)
    tampered = copy.deepcopy(artifact)
    tampered["groups"][0]["row_recall"] = 0.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_preflight_artifact(tampered, config)


def test_preflight_rejects_a_different_dataset_manifest():
    rows = [
        *[_row("dev", 0, 1, 1) for _ in range(100)],
        _row("train", 0, 1, 1),
        _row("train", 0, 0, 0),
    ]
    config = base_config()
    identity = _manifest_identity(config)
    identity["manifest"]["dataset"] = "kitti"
    from utils.acquisition_metrics import sha256_json
    identity["manifest_sha256"] = sha256_json(identity["manifest"])
    artifact = build_preflight_artifact(
        rows, {
            "acquisition": acquisition_config_identity(config),
        }, identity, 42)
    with pytest.raises(ValueError, match="data manifest mismatch"):
        validate_preflight_artifact(artifact, config)


def test_promotion_metrics_require_explicit_unique_dev_candidate0_rows():
    row = {
        "partition": "dev", "candidate_id": 0,
        "tracklet_id": 0, "frame_id": 1,
        "pool_target_count": 2, "sampled_target_count": 1,
        "search_valid": 1, "observation_error": 1.0,
        "raw_search_error": 0.5,
        "router_applied_gate": 1, "selective_error": 0.4,
        "observation_iou": 0.5, "raw_search_iou": 0.7,
        "selective_iou": 0.8,
    }
    raw_success = success_auc([1.0, 0.7])
    observation_success = success_auc([1.0, 0.5])
    metrics = build_metrics(
        [row], raw_success, observation_success)
    assert metrics["diagnostic_rows"] == 1
    assert metrics["diagnostic_tracklets"] == 1
    with pytest.raises(ValueError, match="only dev candidate0"):
        build_metrics(
            [dict(row, partition="train")],
            raw_success, observation_success)
    with pytest.raises(ValueError, match="duplicate"):
        build_metrics(
            [row, dict(row)], raw_success, observation_success,
            )
