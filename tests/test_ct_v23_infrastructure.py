import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from models.ct_variant import configure_ct_variant
from utils.acquisition_metrics import balanced_targetness_class_weights
from utils.online_contract import (
    build_online_resume_contract,
    require_scratch_initialization,
    validate_scratch_training_contract,
    validate_online_resume_contract,
)
from utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
from utils.training_isolation import (
    advance_lightning_manual_transaction,
    candidate_stratified_mean,
    contract_v3_action_probability,
    freeze_batchnorm_running_stats,
    partition_candidate_view_items,
    update_cumulative_binary_class_balance,
)
from tools.report_ct_b2 import (
    acquisition_stage,
    build_acquisition_metrics,
    build_metrics,
    build_reference_comparison,
    COUNTERFACTUAL_ARMS,
    COUNTERFACTUAL_METRICS,
    success_auc,
)


def base_config():
    return {
        "experiment_name": "scratch-e2-seed42",
        "seed": 42,
        "ct_joint_contract_version": 3,
        "ct_online_recursive_training": True,
        "ct_training_topology": "dual_stream",
        "ct_b0_training_protocol": "seqtrack_d86990c",
        "ct_b0_candidate_mode": "independent",
        "candidate_trajectory_mode": "independent",
        "ct_b0_steps_per_epoch": 1262,
        "ct_mechanism_stream": "online_recursive",
        "ct_mechanism_passes_per_epoch": 1,
        "ct_mechanism_b0_view": "canonical_only",
        "use_ct_joint_full": True,
        "ct_enable_b1": True,
        "ct_enable_b2": True,
        "ct_enable_b3": False,
        "num_candidates": 4,
        "ct_recursive_candidate_views": 4,
        "ct_b0_candidate_views": 4,
        "ct_b0_candidate_weights": [0.25, 0.25, 0.25, 0.25],
        "ct_b2_candidate_views": 1,
        "ct_recursive_tracklet_slots": 16,
        "ct_recursive_rollout_horizons": [1, 2, 4, 8],
        "ct_recursive_reseed_enabled": False,
        "ct_b0_rng_shift_control": False,
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
    with pytest.raises(ValueError, match="online_resume_contract.v6"):
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
        ROOT / 'cfgs' / 'ct_seqtrack' / '24_b1.yaml')
    assert b1['seed'] == 42
    assert b1['checkpoint_monitor'] == 'b1_nll/mini_val'
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
    # The two graphs are mathematically identical, but autograd accumulates
    # the 48-row and 3x16-row reductions in a different floating-point order.
    # That order may differ by a few ULPs across PyTorch/CPU builds.
    torch.testing.assert_close(
        full_gradient, micro_gradient, rtol=1e-6, atol=1e-7)


def test_candidate_views_partition_before_heterogeneous_dict_collation():
    processed = []
    context = []
    for view_id in range(4):
        for row_id in range(16):
            item = {
                'b0_view_id': np.int64(view_id),
                'points': np.full((2, 3), row_id, dtype=np.float32),
            }
            if view_id == 0:
                item['ct_base_evidence_points'] = np.full(
                    (4, 3), row_id, dtype=np.float32)
            processed.append(item)
            context.append((view_id, row_id))

    with pytest.raises(KeyError, match="ct_base_evidence_points"):
        torch.utils.data.default_collate(processed)

    canonical, auxiliary, canonical_context, auxiliary_context = (
        partition_candidate_view_items(
            processed, context, canonical_batch_size=16,
            candidate_views=4))

    canonical_batch = torch.utils.data.default_collate(canonical)
    auxiliary_batch = torch.utils.data.default_collate(auxiliary)
    assert canonical_batch['points'].shape == (16, 2, 3)
    assert canonical_batch['ct_base_evidence_points'].shape == (16, 4, 3)
    assert auxiliary_batch['points'].shape == (48, 2, 3)
    assert 'ct_base_evidence_points' not in auxiliary_batch
    assert canonical_context == [(0, row_id) for row_id in range(16)]
    assert auxiliary_context[0] == (1, 0)


def test_contract_v3_action_score_is_owned_only_by_b3():
    # B1+B2 has no utility/action head and must still collect B2 metrics.
    assert contract_v3_action_probability({}, b3_enabled=False) is None

    score = torch.tensor([0.1, 0.8])
    selected = contract_v3_action_probability(
        {'ct_b3_action_score': score}, b3_enabled=True)
    assert torch.equal(selected, score)
    assert not selected.requires_grad

    with pytest.raises(RuntimeError, match="ct_b3_action_score"):
        contract_v3_action_probability({}, b3_enabled=True)


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


def test_b2_report_requires_explicit_unique_dev_candidate0_rows():
    row = {
        "partition": "dev", "candidate_id": 0,
        "tracklet_id": 0, "frame_id": 1,
        "base_target_count": 0, "expansion_target_count": 3,
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


def test_b2_acquisition_report_localizes_every_funnel_stage():
    def row(frame_id, base, expansion, pool, sampled):
        return {
            "partition": "dev", "candidate_id": 0,
            "tracklet_id": 0, "frame_id": frame_id,
            "base_target_count": base,
            "expansion_target_count": expansion,
            "pool_target_count": pool,
            "sampled_target_count": sampled,
        }

    rows = [
        row(1, 5, 0, 0, 0),
        row(2, 0, 0, 0, 0),
        row(3, 0, 3, 0, 0),
        row(4, 0, 3, 2, 0),
        row(5, 0, 3, 2, 1),
    ]
    assert [acquisition_stage(item) for item in rows] == [
        "base_sufficient", "geometry_miss", "no_novel_target",
        "sampling_loss", "retained"]
    metrics = build_acquisition_metrics(rows)
    assert metrics["acquisition_stage_counts"] == {
        "base_sufficient": 1,
        "geometry_miss": 1,
        "no_novel_target": 1,
        "sampling_loss": 1,
        "retained": 1,
    }
    weak = metrics["acquisition_weak_recovery"]
    strict = metrics["acquisition_strict_miss"]
    for key in (
            "rows", "geometry_miss_rows", "no_novel_target_rows",
            "sampling_loss_rows", "retained_rows", "support_row_recall",
            "pool_row_recall", "sampling_row_recall",
            "sampling_point_recall", "end_to_end_row_retention"):
        assert weak[key] == strict[key]
    assert weak["rows"] == 4
    assert weak["support_row_recall"] == pytest.approx(0.75)
    assert weak["pool_row_recall"] == pytest.approx(2 / 3)
    assert weak["sampling_row_recall"] == pytest.approx(0.5)
    assert weak["sampling_point_recall"] == pytest.approx(0.25)
    assert weak["end_to_end_row_retention"] == pytest.approx(0.25)


def test_b2_acquisition_report_uses_null_for_undefined_conditionals():
    metrics = build_acquisition_metrics([{
        "partition": "dev", "candidate_id": 0,
        "tracklet_id": 0, "frame_id": 1,
        "base_target_count": 0, "expansion_target_count": 0,
        "pool_target_count": 0, "sampled_target_count": 0,
    }])
    weak = metrics["acquisition_weak_recovery"]
    assert weak["support_row_recall"] == 0.0
    assert weak["pool_row_recall"] is None
    assert weak["sampling_row_recall"] is None
    assert weak["sampling_point_recall"] is None
    assert weak["end_to_end_row_retention"] == 0.0
    assert metrics["acquisition_row_recall"] is None
    assert metrics["acquisition_point_recall"] is None


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"base_target_count": -1}, "negative base target count"),
        ({"expansion_target_count": 1, "pool_target_count": 2},
         "0 <= sampled_target_count"),
    ],
)
def test_b2_acquisition_report_rejects_invalid_counts(updates, match):
    row = {
        "partition": "dev", "candidate_id": 0,
        "tracklet_id": 0, "frame_id": 1,
        "base_target_count": 0, "expansion_target_count": 2,
        "pool_target_count": 1, "sampled_target_count": 1,
    }
    row.update(updates)
    with pytest.raises(ValueError, match=match):
        build_acquisition_metrics([row])


def test_b2_acquisition_report_requires_expansion_count():
    row = {
        "partition": "dev", "candidate_id": 0,
        "tracklet_id": 0, "frame_id": 1,
        "base_target_count": 0,
        "pool_target_count": 1, "sampled_target_count": 1,
    }
    with pytest.raises(ValueError, match="expansion_target_count"):
        build_acquisition_metrics([row])


def _v2_acquisition_row(
        frame_id, *, global_label=1, global_exact=None, base_raw=0,
        base_sampled=0, support_xy=0, support_xyz=0, pool=0, sampled=0,
        endpoint_target=0, tube_target=0, pool_source="endpoint_only",
        sampled_source="endpoint_only", cf_overrides=None,
        tracklet_key="scene/token"):
    global_exact = global_label if global_exact is None else global_exact
    row = {
        "partition": "dev",
        "candidate_id": 0,
        "tracklet_id": 0,
        "tracklet_key": tracklet_key,
        "frame_id": frame_id,
        "acquisition_schema_version": 2,
        "global_target_count_exact": global_exact,
        "global_target_count_label": global_label,
        "global_raw_point_count": max(global_label + 20, 20),
        "base_target_count": base_raw,
        "base_raw_target_count": base_raw,
        "base_sampled_target_count": base_sampled,
        "base_raw_point_count": base_raw + 4,
        "base_sampled_point_count": base_sampled + 4,
        "endpoint_raw_target_count": endpoint_target,
        "tube_raw_target_count": tube_target,
        "endpoint_raw_point_count": endpoint_target + 4,
        "tube_raw_point_count": tube_target + 4,
        "support_union_target_count": support_xyz,
        "support_union_raw_point_count": support_xyz + 5,
        "support_union_background_count": 5,
        "support_xy_target_count": support_xy,
        "support_xyz_target_count": support_xyz,
        "support_z_clip_target_count": support_xy - support_xyz,
        "endpoint_target_center_inside_xy": int(endpoint_target > 0),
        "endpoint_target_center_inside_xyz": int(endpoint_target > 0),
        "tube_target_center_inside_xy": int(tube_target > 0),
        "tube_target_center_inside_xyz": int(tube_target > 0),
        "active_endpoint_error_xy": 1.0,
        "active_endpoint_error_z": 0.25,
        "active_tube_error_xy": 0.5,
        "active_tube_error_z": 0.25,
        "active_prior_source": "b1",
        "active_support_truncated": 0,
        "active_endpoint_width": 4.0,
        "active_endpoint_length": 8.0,
        "active_endpoint_height": 2.0,
        "active_tube_width": 4.0,
        "active_tube_length": 10.0,
        "active_tube_height": 2.0,
        "expansion_target_count": support_xyz,
        "pool_target_count": pool,
        "sampled_target_count": sampled,
        "extension_pool_count": pool + 3,
        "sampled_count": sampled + 2,
        "pool_background_count": 3,
        "sampled_background_count": 2,
        "learned_motion_error": 1.0,
        "kinematic_error": 1.5,
        "learned_error_parallel": 0.75,
        "learned_error_perpendicular": -0.25,
        "learned_cv_disagreement": 0.5,
        "b1_valid": 1,
        "b1_nll": 1.0,
        "b1_mahalanobis_sq": 1.0,
        "b1_coverage_50": 1,
        "b1_coverage_80": 1,
        "b1_coverage_95": 1,
        "sigma_parallel": 1.0,
        "sigma_perpendicular": 0.5,
        "query_delta_t": 0.5,
        "gap_ratio": 1.0,
        "support_actual_length": 10.0,
        "support_actual_width": 4.0,
        "support_volume": 80.0,
        "recursive_age": -1,
        "recursive_age_valid": 0,
        "search_geometry_source_id": 1,
    }
    for prefix, total, source in (
            ("pool", pool, pool_source),
            ("sampled", sampled, sampled_source)):
        for name in ("endpoint_only", "tube_only", "overlap"):
            row[f"{prefix}_{name}_target_count"] = (
                total if name == source else 0)
    for arm in COUNTERFACTUAL_ARMS:
        xyz = support_xyz
        xy = max(support_xy, xyz)
        row.update({
            f"cf_{arm}_valid": 1,
            f"cf_{arm}_xy_target_count": xy,
            f"cf_{arm}_xyz_target_count": xyz,
            f"cf_{arm}_target_bearing": int(xyz > 0),
            f"cf_{arm}_raw_point_count": xyz + 5,
            f"cf_{arm}_background_count": 5,
            f"cf_{arm}_support_volume": 100.0,
            f"cf_{arm}_truncated": 0,
            f"cf_{arm}_endpoint_error_xy": 1.0,
            f"cf_{arm}_endpoint_error_z": 0.25,
        })
    for key, value in (cf_overrides or {}).items():
        row[key] = value
    for arm in COUNTERFACTUAL_ARMS:
        xyz_key = f"cf_{arm}_xyz_target_count"
        row[f"cf_{arm}_target_bearing"] = int(row[xyz_key] > 0)
        row[f"cf_{arm}_raw_point_count"] = row[xyz_key] + 5
        row[f"cf_{arm}_background_count"] = 5
    return row


def test_b2_v2_report_separates_observability_geometry_and_sampling():
    rows = [
        _v2_acquisition_row(1, global_label=0, global_exact=0),
        _v2_acquisition_row(2, support_xy=0, support_xyz=0,
                            global_exact=0),
        _v2_acquisition_row(3, support_xy=1, support_xyz=0,
                            cf_overrides={
                                "cf_learned_2_1_z05_xyz_target_count": 1,
                                "cf_learned_2_1_z10_xyz_target_count": 1,
                            }),
        _v2_acquisition_row(4, support_xy=1, support_xyz=1,
                            endpoint_target=1),
        _v2_acquisition_row(5, support_xy=1, support_xyz=1,
                            endpoint_target=0, tube_target=1, pool=1,
                            sampled=0, pool_source="tube_only"),
        _v2_acquisition_row(6, support_xy=1, support_xyz=1,
                            endpoint_target=1, tube_target=1, pool=1,
                            sampled=1, pool_source="overlap",
                            sampled_source="overlap"),
    ]
    metrics = build_acquisition_metrics(rows)
    assert metrics["acquisition_schema_version"] == 2
    assert metrics["observability"] == {
        "rows": 6,
        "exact_visible_rows": 4,
        "label_visible_rows": 5,
        "sensor_unobservable_rows": 1,
        "boundary_only_rows": 1,
        "label_visible_rate": pytest.approx(5 / 6),
    }
    funnel = metrics["observable_strict_funnel"]
    assert funnel["rows"] == 5
    assert funnel["xy_miss_rows"] == 1
    assert funnel["z_clip_miss_rows"] == 1
    assert funnel["xyz_geometry_miss_rows"] == 2
    assert funnel["no_novel_target_rows"] == 1
    assert funnel["sampling_loss_rows"] == 1
    assert funnel["retained_rows"] == 1
    complementarity = metrics["branch_complementarity"]
    assert complementarity["endpoint_only_rows"] == 1
    assert complementarity["tube_only_rows"] == 1
    assert complementarity["both_rows"] == 1
    assert complementarity["neither_rows"] == 2
    assert complementarity["union_recall"] == pytest.approx(3 / 5)
    z05 = metrics["counterfactual_geometry"]["learned_2_1_z05"]
    assert z05["target_bearing_recall"] == pytest.approx(4 / 5)
    assert z05["newly_recovered_vs_current_rows"] == 1
    assert metrics["strata"]["recursive_age"] is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.pop("global_target_count_label"), "lacks fields"),
        (lambda row: row.update({"support_xy_target_count": float("nan")}),
         "non-finite"),
        (lambda row: row.update({"pool_overlap_target_count": 1}),
         "do not sum"),
        (lambda row: row.update({
            "cf_learned_3_1p5_z0_xyz_target_count": 0,
            "cf_learned_3_1p5_z0_target_bearing": 0,
            "cf_learned_3_1p5_z0_raw_point_count": 5,
            "cf_learned_3_1p5_z0_background_count": 5}),
         "not monotonic"),
    ],
)
def test_b2_v2_report_fails_closed_on_invalid_rows(mutation, match):
    row = _v2_acquisition_row(
        1, support_xy=1, support_xyz=1, endpoint_target=1,
        pool=1, sampled=1)
    mutation(row)
    with pytest.raises(ValueError, match=match):
        build_acquisition_metrics([row])


def test_b2_v2_report_rejects_mixed_schema_and_duplicate_stable_keys():
    row = _v2_acquisition_row(1)
    legacy = {
        "partition": "dev", "candidate_id": 0,
        "tracklet_id": 0, "frame_id": 2,
        "base_target_count": 0, "expansion_target_count": 0,
        "pool_target_count": 0, "sampled_target_count": 0,
    }
    with pytest.raises(ValueError, match="mixed acquisition schemas"):
        build_acquisition_metrics([row, legacy])
    missing_schema = dict(row)
    missing_schema.pop("acquisition_schema_version")
    with pytest.raises(ValueError, match="without a complete schema"):
        build_acquisition_metrics([missing_schema])
    duplicate = dict(row, tracklet_id=99)
    with pytest.raises(ValueError, match="duplicate"):
        build_acquisition_metrics([row, duplicate])


def test_b2_v2_reference_comparison_reports_unpaired_keys():
    primary = [
        _v2_acquisition_row(1),
        _v2_acquisition_row(2),
    ]
    reference = [
        _v2_acquisition_row(2),
        _v2_acquisition_row(3),
    ]
    comparison = build_reference_comparison(primary, reference)
    assert comparison["paired_rows"] == 1
    assert not comparison["row_sets_identical"]
    assert comparison["primary_intersection_coverage"] == 0.5
    assert comparison["reference_intersection_coverage"] == 0.5
    assert comparison["missing_in_primary"] == [{
        "tracklet_key": "scene/token", "frame_id": 3,
        "candidate_id": 0}]
    assert comparison["missing_in_reference"] == [{
        "tracklet_key": "scene/token", "frame_id": 1,
        "candidate_id": 0}]
