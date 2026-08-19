import ast
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_formal_entrypoints_do_not_expose_removed_training_paths():
    main = _text("main.py")
    for removed in (
        "--init_checkpoint",
        "--use_recursive_replay_cache",
        "--recursive_replay_cache_dir",
        "--pftc_weight",
        "--motion_v3_fusion_scale",
    ):
        assert removed not in main

    assert "load_initial_weights" not in _text("utils/checkpoint_loading.py")
    assert "init_checkpoint" not in _text("ctseqtrack/runtime/provenance.py")


def test_model_and_evaluator_have_one_formal_generation():
    seqtrack = _text("models/seqtrack3d.py")
    evaluator = _text("models/base_model.py")
    for removed in (
        "models.ct_v2",
        "search_v21",
        "search_v22",
        "_build_proposal_diagnostic_row",
        "_build_b3_rollout_row",
        "recursive_replay",
    ):
        assert removed not in seqtrack
        assert removed not in evaluator
    assert "build_ct_joint_diagnostic_row" in evaluator
    formal_model = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "ctseqtrack/model").glob("*.py")
    )
    assert "requires_grad_(False)" not in formal_model
    assert "requires_grad = False" not in formal_model


def test_formal_data_path_has_only_current_contracts():
    formal_data = "\n".join(
        _text(path)
        for path in (
            "ctseqtrack/data/sample_builder.py",
            "ctseqtrack/data/inference.py",
            "ctseqtrack/data/outputs.py",
            "ctseqtrack/data/auxiliary.py",
        )
    )
    for removed in (
        "joint_contract_v2",
        "point_evidence_contract_v2",
        "legacy_canonical_prev_boxs",
    ):
        assert removed not in formal_data


def test_public_v25_yaml_has_no_historical_version_switches():
    removed = (
        "use_ct_v2:",
        "use_search_evidence_v2:",
        "use_search_evidence_v21:",
        "use_motion_conditioned_search_v22:",
        "use_motion_conditioned_search_v3:",
        "use_action_consistent_router_v3:",
        "use_recursive_replay_cache:",
        "use_dynamics_encoder:",
        "use_observability_gate:",
        "use_point_feature_tc:",
        "pftc_weight:",
        "motion_v3_fusion_scale:",
        "ct_point_evidence_contract_version:",
    )
    entries = sorted((ROOT / "cfgs/ct_seqtrack").glob("25*.yaml"))
    assert len(entries) == 16
    for path in entries + sorted((ROOT / "cfgs/ct_seqtrack/_base").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert all(key not in text for key in removed), path


def test_formal_production_files_stay_below_split_threshold():
    roots = (ROOT / "ctseqtrack", ROOT / "models", ROOT / "datasets")
    oversized = {}
    for source_root in roots:
        for path in source_root.rglob("*.py"):
            count = len(path.read_text(encoding="utf-8").splitlines())
            if count > 1200:
                oversized[str(path.relative_to(ROOT))] = count
    assert oversized == {}


def test_sample_output_context_is_bound_before_dispatch():
    builder_tree = ast.parse(_text("ctseqtrack/data/sample_builder.py"))
    builder = next(
        node
        for node in builder_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "motion_processing_mf"
    )
    call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_sample_output"
    )
    bound = {argument.arg for argument in builder.args.args}
    for node in ast.walk(builder):
        if getattr(node, "lineno", call.lineno + 1) > call.lineno:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)

    output_tree = ast.parse(_text("ctseqtrack/data/outputs.py"))
    output_builder = next(
        node
        for node in output_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_sample_output"
    )
    required = {
        node.slice.value
        for node in ast.walk(output_builder)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "context"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert required <= bound, sorted(required - bound)


def test_research_handoff_points_only_to_live_files():
    handoff = json.loads(_text("research_handoff.json"))
    assert handoff["schema"] == "ct_seqtrack.research_handoff.v25"
    missing = [path for path in handoff["primary_files"] if not (ROOT / path).is_file()]
    assert missing == []
    assert handoff["recovery"]["git_history_rewritten"] is False

    baseline = json.loads(_text("tests/fixtures/ct_v25_cleanup_baseline.json"))
    assert baseline["source_commit"].startswith("9ed2afc")
    assert baseline["baseline_local_tests"] == {"passed": 154, "skipped": 1}
    assert baseline["runtime_equivalence_fixture"]["status"] == "server_required"


def test_resume_contract_hashes_all_semantic_config_fields():
    from ctseqtrack.runtime.contracts import (
        build_online_resume_contract,
        validate_online_resume_contract,
    )

    config = {
        "experiment_name": "ct25_full_scratch_seed42",
        "ct_variant": "full",
        "ct_initialization_policy": "scratch_only",
        "seed": 42,
        "checkpoint": None,
        "log_dir": "run/original",
        "custom_semantic_knob": "alpha",
    }
    checkpoint = {
        "ct_online_resume_contract": build_online_resume_contract(config),
        "ct_epoch_boundary_complete": True,
        "ct_global_rng_state": {"schema": "ct_seqtrack.global_rng.v1"},
        "ct_recursive_state_boundary": {
            "schema": "ct_seqtrack.recursive_state_boundary.v1",
            "next_epoch_reset": True,
        },
    }
    relocated = copy.deepcopy(config)
    relocated["checkpoint"] = "moved/last.ckpt"
    relocated["log_dir"] = "run/resume"
    validate_online_resume_contract(checkpoint, relocated)

    changed = copy.deepcopy(relocated)
    changed["custom_semantic_knob"] = "beta"
    try:
        validate_online_resume_contract(checkpoint, changed)
    except ValueError as error:
        assert "resolved_training_config_sha256" in str(error)
    else:
        raise AssertionError("cross-config resume was accepted")
