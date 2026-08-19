"""Flatten the CT-SeqTrack v25 configuration inheritance graph.

The baseline snapshot is the authority: every existing 25*.yaml entry must
resolve to the exact same Python mapping before and after this rewrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctseqtrack.config import configure_ct_variant
from utils.config import load_yaml_config


CONFIG_ROOT = PROJECT_ROOT / "cfgs" / "ct_seqtrack"
BASELINE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "ct_v25_resolved_configs.json"
COMMON_PATH = CONFIG_ROOT / "_base" / "v25_common.yaml"
PROFILE_PATHS = {
    "mini": CONFIG_ROOT / "_base" / "v25_mini.yaml",
    "full": CONFIG_ROOT / "_base" / "v25_full.yaml",
}
ADAPTER_KEYS = {
    "use_ct_v2",
    "use_b1motion_v3",
    "use_ct_joint_full",
    "ct_enable_b1",
    "ct_enable_b2",
    "ct_enable_b3",
    "ct_joint_contract_version",
    "use_motion_v3_legacy_fusion",
    "use_search_evidence_v2",
    "use_search_evidence_v21",
    "use_motion_conditioned_search_v22",
    "use_motion_conditioned_search_v3",
    "use_action_consistent_router_v3",
}
REMOVED_KEYS = {
    "ct_point_evidence_contract_version",
    "ct_fusion_context_dim",
    "ct_fusion_detach_context",
    "ct_fusion_hidden_dim",
    "ct_fusion_init_alpha",
    "ct_fusion_max_alpha",
    "ct_fusion_mode",
    "ct_search_base_length",
    "ct_search_base_width",
    "ct_search_baseline_ratio",
    "ct_search_max_displacement",
    "ct_search_max_length",
    "ct_search_max_speed",
    "ct_search_max_width",
    "ct_search_min_displacement",
    "ct_search_min_expansion_points",
    "ct_search_width_per_second",
    "dynamics_displacement_weight",
    "dynamics_eps",
    "dynamics_innovation_alpha",
    "dynamics_innovation_disable_on_empty_search",
    "dynamics_innovation_radius_base",
    "dynamics_innovation_radius_max",
    "dynamics_innovation_radius_per_second",
    "dynamics_innovation_scale",
    "dynamics_innovation_warmup_epoch",
    "dynamics_long_gap_only",
    "dynamics_max_alpha",
    "dynamics_max_residual_norm",
    "dynamics_min_delta_t",
    "dynamics_motion_mode",
    "dynamics_residual_detach_stats",
    "dynamics_residual_gate_hidden_dim",
    "dynamics_residual_init_alpha",
    "dynamics_residual_scale",
    "dynamics_sparse_only",
    "dynamics_sparse_point_threshold",
    "dynamics_use_acceleration",
    "dynamics_use_query_gap",
    "dynamics_warmup_epoch",
    "dynamics_hidden_dim",
    "experiment_family",
    "m3_variant",
    "m4_fixed_delta_t",
    "m4_time_mode",
    "m4_variant",
    "motion_v3_alpha_max",
    "motion_v3_fused_weight",
    "motion_v3_fusion_scale",
    "motion_v3_gate_context_dim",
    "motion_v3_gate_hidden_dim",
    "motion_v3_gate_init_probability",
    "motion_v3_gate_weight",
    "motion_v3_help_margin",
    "motion_v3_radius_base",
    "motion_v3_radius_max",
    "motion_v3_radius_per_second",
    "motion_v3_warmup_epoch",
    "obs_gate_entropy_weight",
    "obs_gate_hidden_dim",
    "obs_gate_init_obs_bias",
    "obs_gate_min_dyn_valid",
    "obs_gate_num_stats",
    "pftc_distance_threshold",
    "pftc_min_correspondences",
    "pftc_ramp_epochs",
    "pftc_time_field",
    "pftc_time_scale",
    "pftc_time_weight_max",
    "pftc_time_weight_min",
    "pftc_time_weighting",
    "pftc_weight",
    "physical_time_adapter_hidden_dim",
    "physical_time_adapter_scale",
    "physical_time_adapter_warmup_epoch",
    "use_dynamics_encoder",
    "use_m3_path_distillation",
    "use_m4_state_filter",
    "use_m4_trajectory_tube",
    "use_observability_gate",
    "use_ordered_trajectory_encoder",
    "use_physical_time_adapter",
    "use_point_feature_tc",
    "use_recursive_replay_cache",
    "use_time_guided_search",
    "use_trajectory_adapter",
    "use_twc",
    "velocity_weight",
}


def load_baseline(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ct_seqtrack.v25_resolved_config_baseline.v1":
        raise ValueError("unexpected v25 configuration baseline schema")
    configs = {name: entry["resolved"] for name, entry in payload["configs"].items()}
    if len(configs) != 16:
        raise ValueError(f"expected 16 v25 configs, found {len(configs)}")
    return configs


def common_mapping(configs):
    iterator = iter(configs.values())
    first = next(iterator)
    common = dict(first)
    for resolved in iterator:
        common = {
            key: value
            for key, value in common.items()
            if key in resolved and resolved[key] == value
        }
    return common


def dump_yaml(path, payload, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(
        payload,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=100,
    )
    path.write_text(header + rendered, encoding="utf-8", newline="\n")


def rewrite(configs):
    public_configs = {
        name: {
            key: value
            for key, value in resolved.items()
            if key not in ADAPTER_KEYS | REMOVED_KEYS
        }
        for name, resolved in configs.items()
    }
    common = common_mapping(public_configs)
    dump_yaml(
        COMMON_PATH,
        common,
        "# Generated from the 9ed2afc v25 resolved-config baseline.\n"
        "# Paper-facing entries inherit this file directly; do not add an "
        "_base_.\n",
    )
    profiles = {
        "mini": {
            name: resolved
            for name, resolved in public_configs.items()
            if str(resolved.get("version")) == "v1.0-mini"
        },
        "full": {
            name: resolved
            for name, resolved in public_configs.items()
            if str(resolved.get("version")) != "v1.0-mini"
        },
    }
    for profile_name, members in profiles.items():
        profile = {
            key: value
            for key, value in common_mapping(members).items()
            if key not in common or common[key] != value
        }
        dump_yaml(
            PROFILE_PATHS[profile_name],
            {"_base_": "v25_common.yaml", **profile},
            f"# CT-SeqTrack v25 {profile_name} data profile.\n",
        )
    for name, resolved in public_configs.items():
        profile_name = "mini" if str(resolved.get("version")) == "v1.0-mini" else "full"
        inherited = {
            **common,
            **{
                key: value
                for key, value in common_mapping(profiles[profile_name]).items()
                if key not in common or common[key] != value
            },
        }
        override = {
            key: value
            for key, value in resolved.items()
            if key not in inherited or inherited[key] != value
        }
        payload = {
            "_base_": f"_base/v25_{profile_name}.yaml",
            **override,
        }
        dump_yaml(
            CONFIG_ROOT / name,
            payload,
            "# CT-SeqTrack v25 entry. Values not shown are defined in "
            "_base/v25_common.yaml.\n",
        )


def verify(configs):
    errors = []
    for name, expected in configs.items():
        observed = load_yaml_config(CONFIG_ROOT / name)
        leaked = sorted(ADAPTER_KEYS.intersection(observed))
        expected = configure_ct_variant(
            {key: value for key, value in expected.items() if key not in REMOVED_KEYS}
        )
        observed = configure_ct_variant(dict(observed))
        if observed != expected or leaked:
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            changed = sorted(
                key
                for key in set(expected) & set(observed)
                if expected[key] != observed[key]
                or type(expected[key]) is not type(observed[key])
            )
            errors.append(
                f"{name}: leaked={leaked}, missing={missing}, "
                f"extra={extra}, changed={changed}"
            )
    if errors:
        raise RuntimeError("v25 resolved-config parity failed:\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    configs = load_baseline(args.baseline.resolve())
    if not args.check:
        rewrite(configs)
    verify(configs)
    print(f"verified {len(configs)} exact adapted v25 runtime configurations")


if __name__ == "__main__":
    main()
