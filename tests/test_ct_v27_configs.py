"""防止新正式臂的输入预算、训练起点和评测合同在配置中漂移。"""

from pathlib import Path

from utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("b0", "b1_gru", "b1_cfc", "full_minus_b3", "full")


def _config(arm, full=False):
    suffix = "_nuscenes_full" if full else ""
    return load_yaml_config(ROOT / "cfgs" / "ct_seqtrack" / f"27_{arm}{suffix}.yaml")


def test_v27_five_arms_share_b0_data_optimizer_and_scratch_contract():
    keys = (
        "candidate_trajectory_mode", "num_candidates", "ct_b0_candidate_views",
        "ct_b0_candidate_weights", "ct_b2_candidate_views", "ct_b0_steps_per_epoch",
        "ct_runtime_protocol", "ct_batch_schema", "ct_recursive_reseed_enabled",
        "ct_initialization_policy", "ct_optimizer_topology", "ct_separate_optimizers",
        "seed", "epoch", "from_epoch", "lr", "ct_b0_lr", "ct_partition_seed",
        "point_sample_size", "ct_metric_mode", "ct_exact_metric_audit", "hist_num",
    )
    reference = _config("b0")
    for arm in ARMS:
        cfg = _config(arm)
        assert {key: cfg[key] for key in keys} == {key: reference[key] for key in keys}
        assert cfg["ct_enable_v27"] and cfg["ct_enable_v26_recovery"]
        assert cfg["ct_b0_steps_per_epoch"] == 0
        assert cfg["ct_batch_schema"] == "ct_seqtrack.train.v4"
        assert cfg["epoch"] == 60 and cfg["seed"] == 42
        assert cfg["ct_initialization_policy"] == "scratch_only"
        assert cfg.get("init_checkpoint") is None


def test_v27_full_pairs_preserve_model_budget_with_role_specific_loader_settings():
    # 用户 mini 启动设 workers=4、每5轮验证；full 保留既有12/1默认值。
    allowed = {"version", "path", "train_split", "val_split", "test_split", "experiment_name",
               "workers", "check_val_every_n_epoch"}
    for arm in ARMS:
        mini, full = _config(arm), _config(arm, full=True)
        differences = {key for key in set(mini) | set(full) if mini.get(key) != full.get(key)}
        assert differences <= allowed
        assert mini["version"] == "v1.0-mini"
        assert full["version"] == "v1.0-trainval"
        assert full["train_split"] == "train_track"
        assert full["val_split"] == full["test_split"] == "val"


def test_v27_backend_comparison_and_action_policies_are_explicit():
    gru, cfc = _config("b1_gru"), _config("b1_cfc")
    assert {key for key in set(gru) | set(cfc) if gru.get(key) != cfc.get(key)} == {
        "motion_v3_temporal_backend", "experiment_name"}
    assert _config("full_minus_b3")["proposal_inference_mode"] == "bounded_always"
    assert _config("full")["proposal_inference_mode"] == "selective"
    assert _config("full")["ct_require_action_calibration"]
    assert not _config("full_minus_b3")["ct_enable_b3"]
    assert _config("full")["ct_training_state_policy"] == "observation"


def test_v27_reference_uses_common_repairs_but_no_ct_modules_or_variant():
    for suffix in ("", "_nuscenes_full"):
        cfg = load_yaml_config(ROOT / "cfgs" / f"27_seqtrack_reference{suffix}.yaml")
        assert cfg["ct_enable_v27"] and cfg["ct_v27_reference"]
        assert cfg["candidate_trajectory_mode"] == "independent"
        assert cfg["net_model"] == "seqtrack3d"
        assert "ct_variant" not in cfg
        assert "ct_runtime_protocol" not in cfg
        assert not any(cfg[key] for key in (
            "use_ct_v2", "use_ct_joint_full", "use_b1motion_v3",
            "ct_enable_b1", "ct_enable_b2", "ct_enable_b3"))
        assert cfg["ct_metric_mode"] == "benchmark_compat"
        assert cfg["ct_b0_steps_per_epoch"] == 0
        assert cfg["ct_initialization_policy"] == "scratch_only"
