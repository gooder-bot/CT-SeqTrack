"""Dependency-light configuration normalization for formal CT arms."""


VARIANTS = {
    "b0": (False, False, False, False),
    "b1": (True, True, False, False),
    "full_minus_b3": (True, True, True, False),
    "full": (True, True, True, True),
}


def set_config(config, name, value):
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def get_config(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(
        config, name, default)


def configure_ct_variant(config):
    variant = str(get_config(config, "ct_variant", "full")).strip().lower()
    if variant not in VARIANTS:
        raise ValueError(
            "ct_variant must be b0, b1, full_minus_b3 or full")
    use_joint, use_b1, use_b2, use_b3 = VARIANTS[variant]
    runtime_protocol = str(get_config(
        config, "ct_runtime_protocol", "legacy")).strip().lower()
    unified_auto = runtime_protocol == "safe_seqtrack_auto_v1"
    values = {
        "ct_variant": variant,
        "use_ct_joint_full": use_joint,
        "use_b1motion_v3": use_b1,
        "ct_enable_b1": use_b1,
        "ct_enable_b2": use_b2,
        "ct_enable_b3": use_b3,
        "ct_joint_contract_version": 3,
        "ct_separate_optimizers": not unified_auto,
        "ct_optimizer_topology": (
            "unified_auto" if unified_auto
            else str(get_config(
                config, "ct_optimizer_topology", "isolated_manual"))),
        "ct_initialization_policy": "scratch_only",
        "ct_b0_initialization_policy": "scratch_only",
        "ct_training_state_policy": "observation",
        "ct_module_isolation": "strict",
        "use_motion_v3_legacy_fusion": False,
    }
    for name, value in values.items():
        set_config(config, name, value)
    if unified_auto:
        if str(get_config(
                config, "ct_observation_rng_mode", "")) != "stateless_seqtrack":
            raise ValueError(
                "safe_seqtrack_auto_v1 requires stateless_seqtrack RNG")
        if str(get_config(config, "ct_batch_schema", "")) != (
                "ct_seqtrack.train.v2"):
            raise ValueError(
                "safe_seqtrack_auto_v1 requires ct_seqtrack.train.v2")
    forbidden = (
        "use_observability_gate", "use_dynamics_encoder", "use_ct_v2",
        "use_search_evidence_v2", "use_search_evidence_v21",
        "use_motion_conditioned_search_v22",
        "use_motion_conditioned_search_v3", "use_b3_risk_router",
        "use_joint_proposal_fusion", "use_advantage_proposal_fusion",
    )
    enabled = [name for name in forbidden if bool(get_config(
        config, name, False))]
    if enabled:
        raise ValueError(
            "CTSEQTRACK formal variants forbid legacy runtime branches: "
            + ", ".join(sorted(enabled)))
    for name in forbidden:
        set_config(config, name, False)
    prior_mode = str(get_config(
        config, "ct_prior_mode", "learned_physical")).strip().lower()
    if prior_mode not in ("learned_physical", "fixed_cv"):
        raise ValueError("ct_prior_mode must be learned_physical or fixed_cv")
    set_config(config, "ct_prior_mode", prior_mode)
    if prior_mode == "fixed_cv" and variant != "b0":
        set_config(config, "use_b1motion_v3", True)
        set_config(config, "ct_enable_b1", False)
    if variant == "b0":
        # The v24 configs inherit data/training defaults from the v23 Full
        # config.  These switches belong to B1/B2 support construction and
        # must not leak into the observation-only arm.  Normalizing them here
        # keeps every B0 config safe even when its base config evolves.
        for name in (
                "use_calibrated_motion_uncertainty",
                "require_b1_calibration_artifact",
                "require_b1_calibration_passed",
                "search_v3_use_dynamic_sigma",
                "use_b1_prepass_support",
                "use_uncertainty_geometry"):
            set_config(config, name, False)
    memory_mode = str(get_config(
        config, "ct_memory_mode", "none")).strip().lower()
    if memory_mode not in ("none", "empty", "real", "time_misaligned"):
        raise ValueError(
            "ct_memory_mode must be none, empty, real or time_misaligned")
    set_config(config, "ct_memory_mode", memory_mode)
    time_mode = str(get_config(
        config, "ct_time_mode",
        get_config(config, "dynamics_time_mode", "true"))).strip().lower()
    if time_mode not in ("true", "fixed", "shuffled"):
        raise ValueError("ct_time_mode must be true, fixed or shuffled")
    set_config(config, "ct_time_mode", time_mode)
    # Dataset construction currently consumes this compatibility key.  It is
    # assigned here before dataloaders are built, not deferred to model init.
    set_config(config, "dynamics_time_mode", time_mode)
    # Candidate count is a B0 augmentation protocol, not a B2 switch.  Keep
    # the explicit YAML value so B0 control (view1) and proposed (view4) can
    # be compared without changing the module graph.
    return config
