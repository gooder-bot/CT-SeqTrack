"""Fail-closed checkpoint contract for CT online scratch experiments."""

from __future__ import annotations

import copy

import numpy as np


ONLINE_RESUME_SCHEMA = "ct_seqtrack.online_resume_contract.v3"


def online_candidate_state_consistent(processed, target_size):
    """Validate only the history contracts owned by the active CT arm.

    B0 intentionally has neither a B1 motion history nor a B2 Search history.
    B1-only may have the former, while B2 must expose both and keep them
    byte-identical.
    """
    motion_history = processed.get("motion_main_ref_boxs")
    search_history = processed.get("b2_v3_history_ref_boxs")
    if search_history is not None and motion_history is None:
        raise RuntimeError(
            "B2 Search history exists without its B1 motion history")
    history_consistent = bool(
        search_history is None
        or np.array_equal(motion_history, search_history))
    size_consistent = bool(np.allclose(
        np.asarray(processed["bbox_size"], dtype=np.float64),
        np.asarray(target_size, dtype=np.float64),
        rtol=0.0, atol=1e-6))
    return history_consistent and size_consistent


def _get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normal(value):
    if isinstance(value, (list, tuple)):
        return tuple(_normal(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _normal(item))
                            for key, item in value.items()))
    return value


def build_online_resume_contract(config):
    """Build the complete identity of one resumable online experiment."""
    base_lr = float(_get(config, "lr", 1e-4))
    preflight = _get(config, "ct_acquisition_preflight_manifest", {})
    method_promotion = _get(config, "ct_b2_method_promotion_manifest", {})
    fields = {
        "experiment_name": str(_get(config, "experiment_name", "")),
        "net_model": str(_get(config, "net_model", "seqtrack3d")),
        "variant": str(_get(config, "ct_variant", "legacy")),
        "prior_mode": str(_get(config, "ct_prior_mode", "legacy")),
        "time_mode": str(_get(
            config, "ct_time_mode",
            _get(config, "dynamics_time_mode", "true"))),
        "fixed_delta_t": float(_get(
            config, "dynamics_fixed_delta_t", 0.5)),
        "time_manifest": _get(config, "dynamics_time_manifest"),
        "training_state_policy": str(_get(
            config, "ct_training_state_policy", "legacy")),
        "module_isolation": str(_get(
            config, "ct_module_isolation", "legacy")),
        "seed": int(_get(config, "seed", 42) or 42),
        "joint_contract_version": int(_get(
            config, "ct_joint_contract_version", 1)),
        "online_recursive_training": bool(_get(
            config, "ct_online_recursive_training", False)),
        "use_ct_joint_full": bool(_get(
            config, "use_ct_joint_full", False)),
        "enable_b1": bool(_get(config, "ct_enable_b1", True)),
        "enable_b2": bool(_get(config, "ct_enable_b2", True)),
        "enable_b3": bool(_get(config, "ct_enable_b3", True)),
        "num_candidates": int(_get(config, "num_candidates", 1)),
        "candidate_views": int(_get(
            config, "ct_recursive_candidate_views", 1)),
        "tracklet_slots": int(_get(
            config, "ct_recursive_tracklet_slots", 1)),
        "rollout_horizons": tuple(int(value) for value in _get(
            config, "ct_recursive_rollout_horizons", [1])),
        "reseed_enabled": bool(_get(
            config, "ct_recursive_reseed_enabled", False)),
        "b0_rng_shift_control": bool(_get(
            config, "ct_b0_rng_shift_control", False)),
        "partition": str(_get(config, "ct_router_partition", "train")),
        "partition_seed": int(_get(config, "ct_partition_seed", 42)),
        "optimizer": str(_get(config, "optimizer", "Adam")).lower(),
        "base_lr": base_lr,
        "b0_lr": float(_get(config, "ct_b0_lr", base_lr)),
        "b1_lr": float(_get(
            config, "ct_b1_lr", _get(config, "ct_plugin_lr", base_lr))),
        "b2_lr": float(_get(
            config, "ct_b2_lr", _get(config, "ct_plugin_lr", base_lr))),
        "b3_lr": float(_get(
            config, "ct_b3_lr", _get(config, "ct_plugin_lr", base_lr))),
        "plugin_lr": float(_get(config, "ct_plugin_lr", base_lr)),
        "weight_decay": float(_get(config, "wd", 0.0)),
        "adam_betas": (0.5, 0.999),
        "adam_eps": 1e-6,
        "scheduler": (
            "onecycle" if str(_get(
                config, "optimizer", "Adam")).lower() == "adamonecycle"
            else "steplr"),
        "lr_decay_step": int(_get(config, "lr_decay_step", 20)),
        "lr_decay_rate": float(_get(config, "lr_decay_rate", 0.1)),
        "b0_gradient_clip": float(_get(
            config, "ct_b0_gradient_clip_val",
            _get(config, "gradient_clip_val", 0.0))),
        "plugin_gradient_clip": float(_get(
            config, "ct_plugin_gradient_clip_val",
            _get(config, "gradient_clip_val", 0.0))),
        "separate_optimizers": bool(_get(
            config, "ct_separate_optimizers", False)),
        "manual_amp_enabled": bool(_get(
            config, "ct_manual_amp_enabled", False)),
        "max_epochs": int(_get(config, "epoch", 60)),
        "limit_train_batches": _get(config, "limit_train_batches", 1.0),
        "limit_val_batches": _get(config, "limit_val_batches", 1.0),
        "check_val_every_n_epoch": int(_get(
            config, "check_val_every_n_epoch", 1)),
        "canonical_batch_size": int(_get(config, "batch_size", 1)),
        "auxiliary_microbatch_size": int(_get(
            config, "ct_auxiliary_microbatch_size", 16)),
        "memory_mode": str(_get(config, "ct_memory_mode", "real")),
        "base_evidence_mode": str(_get(
            config, "ct_base_evidence_mode", "full")),
        "recovery_candidate_policy": str(_get(
            config, "ct_recovery_candidate_policy", "off")),
        "initialization_policy": str(_get(
            config, "ct_initialization_policy",
            _get(config, "ct_b0_initialization_policy", "legacy"))),
        "targetness_class_weight_source": str(_get(
            config, "ct_targetness_class_weight_source", "legacy")),
        "targetness_positive_weight": float(_get(
            config, "ct_targetness_positive_weight", 1.0)),
        "targetness_negative_weight": float(_get(
            config, "ct_targetness_negative_weight", 1.0)),
        "acquisition_preflight_statistics_sha256": (
            preflight.get("statistics_sha256")
            if isinstance(preflight, dict) else None),
        "b2_method_source_checkpoint_sha256": (
            method_promotion.get("source_checkpoint_sha256")
            if isinstance(method_promotion, dict) else None),
    }
    return {"schema": ONLINE_RESUME_SCHEMA, "fields": fields}


def validate_online_resume_contract(checkpoint, config):
    """Validate same-run, epoch-boundary resume and return its contract."""
    observed = checkpoint.get("ct_online_resume_contract")
    if not isinstance(observed, dict):
        hyper_parameters = checkpoint.get("hyper_parameters", {})
        saved_config = hyper_parameters.get("config", hyper_parameters)
        observed = build_online_resume_contract(saved_config)
    if observed.get("schema") != ONLINE_RESUME_SCHEMA:
        raise ValueError(
            "online resume checkpoint does not carry "
            f"{ONLINE_RESUME_SCHEMA}")
    expected = build_online_resume_contract(config)
    mismatches = {}
    observed_fields = observed.get("fields", {})
    for key, expected_value in expected["fields"].items():
        observed_value = observed_fields.get(key)
        if _normal(observed_value) != _normal(expected_value):
            mismatches[key] = {
                "expected": expected_value,
                "observed": observed_value,
            }
    if mismatches:
        raise ValueError(
            "online resume contract mismatch: "
            + ", ".join(sorted(mismatches)))
    if checkpoint.get("ct_epoch_boundary_complete") is not True:
        raise ValueError(
            "online resume is supported only from an explicitly completed "
            "epoch boundary")
    rng_state = checkpoint.get("ct_global_rng_state")
    if (not isinstance(rng_state, dict)
            or rng_state.get("schema") != "ct_seqtrack.global_rng.v1"):
        raise ValueError(
            "exact online resume requires ct_seqtrack.global_rng.v1")
    return copy.deepcopy(observed)


def require_scratch_initialization(config, init_checkpoint):
    """Reject pretrained initialization for every scratch-only CT run."""
    policy = str(_get(
        config, "ct_initialization_policy",
        _get(config, "ct_b0_initialization_policy", "legacy")))
    if init_checkpoint is not None and policy == "scratch_only":
        raise ValueError(
            "scratch-only CT experiment forbids --init_checkpoint; use "
            "--checkpoint only for an epoch-boundary resume of the same run")


def validate_scratch_training_contract(config):
    """Reject launches that silently leave the matched-scratch regime.

    Epoch count and seed are intentionally not fixed here: five-epoch kill
    tests and the pre-registered 42/43/44 replications are all legal.  The
    optimizer, update topology, data geometry and epoch-0 module availability
    are invariant across those runs.
    """
    policy = str(_get(
        config, "ct_initialization_policy",
        _get(config, "ct_b0_initialization_policy", "legacy")))
    if policy != "scratch_only":
        return
    if not bool(_get(config, "ct_online_recursive_training", False)):
        raise ValueError(
            "scratch-only v23 requires online recursive training")

    errors = []

    if str(_get(config, "net_model", "seqtrack3d")) == "ctseqtrack":
        if str(_get(config, "ct_training_state_policy", "")) != "observation":
            errors.append("ct_training_state_policy must be observation")
        if str(_get(config, "ct_module_isolation", "")) != "strict":
            errors.append("ct_module_isolation must be strict")
        if (str(_get(config, "ct_time_mode", "true")) == "shuffled"
                and not _get(config, "dynamics_time_manifest")):
            errors.append(
                "ct_time_mode=shuffled requires dynamics_time_manifest")

    def require_equal(key, expected, default=None):
        observed = _get(config, key, default)
        if _normal(observed) != _normal(expected):
            errors.append(f"{key}={observed!r} (expected {expected!r})")

    base_lr = float(_get(config, "lr", 1e-4))
    b0_lr = float(_get(config, "ct_b0_lr", base_lr))
    plugin_lr = float(_get(config, "ct_plugin_lr", base_lr))
    if str(_get(config, "optimizer", "Adam")).lower() != "adam":
        errors.append("optimizer must be Adam/StepLR")
    if base_lr != 1e-4 or b0_lr != 1e-4:
        errors.append(
            "matched scratch B0 requires lr=ct_b0_lr=1e-4")
    if bool(_get(config, "ct_manual_amp_enabled", False)):
        errors.append("formal scratch training must use FP32")
    if bool(_get(config, "b2_v3_freeze_candidate_producers", False)):
        errors.append("b2_v3_freeze_candidate_producers must be false")
    if bool(_get(config, "v22_freeze_candidate_producers", False)):
        errors.append("v22_freeze_candidate_producers must be false")
    require_equal("motion_v3_warmup_epoch", 0, 0)
    require_equal("batch_size", 16, 1)
    require_equal("ct_recursive_tracklet_slots", 16, 1)
    require_equal("ct_recursive_rollout_horizons", [1, 2, 4, 8], [1])

    b1 = bool(_get(config, "ct_enable_b1", False))
    b2 = bool(_get(config, "ct_enable_b2", False))
    b3 = bool(_get(config, "ct_enable_b3", False))
    fixed_cv = str(_get(config, "ct_prior_mode", "")) == "fixed_cv"
    if b3 and not (b2 and (b1 or fixed_cv)):
        errors.append(
            "B3 scratch Full requires B2 and either learned B1 or fixed CV "
            "from epoch 0")
    if b1 or b2 or b3:
        if not bool(_get(config, "ct_separate_optimizers", False)):
            errors.append("enabled plugins require separate optimizers")
        if plugin_lr != 1e-4:
            errors.append("scratch plugins require ct_plugin_lr=1e-4")
        for module_name in ("b1", "b2", "b3"):
            if bool(_get(config, f"ct_enable_{module_name}", False)):
                module_lr = float(_get(
                    config, f"ct_{module_name}_lr", plugin_lr))
                if module_lr != 1e-4:
                    errors.append(
                        f"scratch {module_name.upper()} requires "
                        f"ct_{module_name}_lr=1e-4")

    if b2:
        require_equal("num_candidates", 4, 1)
        require_equal("ct_recursive_candidate_views", 4, 1)
        require_equal("ct_auxiliary_microbatch_size", 16, 16)
        require_equal(
            "ct_recovery_candidate_policy", "weak_miss_control", "off")
        if not bool(_get(config, "export_proposal_diagnostics", False)):
            errors.append("B2 requires export_proposal_diagnostics=true")
    else:
        require_equal("num_candidates", 1, 1)
        require_equal("ct_recursive_candidate_views", 1, 1)

    if errors:
        raise ValueError(
            "invalid scratch training contract: " + "; ".join(errors))


def build_b2_method_contract(config):
    """Fields that must match when a qualified B2 method enters Full."""
    resume = build_online_resume_contract(config)["fields"]
    # Memory and no-extension/base-evidence modes are independent controls,
    # not part of B2 acquisition promotion.  Likewise Full-B3 promotes the
    # same B2 method into Full, so the arm-level variant is deliberately not
    # part of this contract.
    keys = (
        "net_model", "prior_mode", "time_mode", "fixed_delta_t",
        "time_manifest", "training_state_policy",
        "module_isolation",
        "joint_contract_version", "enable_b1", "enable_b2",
        "num_candidates", "candidate_views", "tracklet_slots",
        "rollout_horizons", "reseed_enabled", "partition",
        "partition_seed", "optimizer", "base_lr", "b0_lr", "b1_lr",
        "b2_lr", "b3_lr", "plugin_lr",
        "weight_decay", "adam_betas", "adam_eps", "scheduler",
        "lr_decay_step", "lr_decay_rate", "b0_gradient_clip",
        "plugin_gradient_clip", "canonical_batch_size",
        "auxiliary_microbatch_size", "initialization_policy",
        "recovery_candidate_policy",
        "acquisition_preflight_statistics_sha256",
        "targetness_class_weight_source", "targetness_positive_weight",
        "targetness_negative_weight",
    )
    return {key: copy.deepcopy(resume[key]) for key in keys}


def validate_b2_method_promotion(promotion, config):
    if (not isinstance(promotion, dict)
            or promotion.get("schema")
            != "ct_seqtrack.b2_evidence_promotion.v3"
            or not bool(promotion.get("passed"))):
        raise ValueError("Full scratch requires a passed B2 method manifest v3")
    observed = promotion.get("b2_method_contract")
    expected = build_b2_method_contract(config)
    if not isinstance(observed, dict):
        raise ValueError("B2 method manifest lacks its configuration contract")
    mismatches = [key for key, value in expected.items()
                  if _normal(observed.get(key)) != _normal(value)]
    if mismatches:
        raise ValueError(
            "B2 method promotion/config mismatch: "
            + ", ".join(sorted(mismatches)))
    return copy.deepcopy(promotion)
