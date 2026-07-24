#!/usr/bin/env python3
"""Fail-fast checks for the runnable M3/M4 experiment contracts."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
M3_CONFIG = ROOT / "cfgs" / "seqtrack3d_nuscenes_m3_endpoint_distill_engineering.yaml"
M4_CONFIG = ROOT / "cfgs" / "seqtrack3d_nuscenes_m4_filter_tube_engineering.yaml"


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config is not a mapping: {path}")
    return payload


def require(config, key, expected=None):
    if key not in config:
        raise RuntimeError(f"Missing config key: {key}")
    value = config[key]
    if expected is not None and value != expected:
        raise RuntimeError(
            f"Unexpected {key}: expected {expected!r}, got {value!r}")
    return value


def main():
    m3 = load_yaml(M3_CONFIG)
    m4 = load_yaml(M4_CONFIG)

    require(m3, "candidate_trajectory_mode", "shared_se2")
    require(m3, "hist_num", 3)
    require(m3, "use_twc", False)
    require(m3, "use_m3_path_distillation", True)
    require(m3, "m3_irregular_supervision_weight", 0.0)
    require(m3, "m3_variant", "distill")
    require(m3, "m4_variant", "off")
    if float(require(m3, "m3_path_weight")) <= 0:
        raise RuntimeError("M3 registered path weight must be positive")
    if float(require(m3, "m3_coarse_weight")) <= 0:
        raise RuntimeError("M3 coarse endpoint weight must be positive")
    require(m3, "m3_teacher_confidence_mode", "hybrid")
    if m3["m3_view_a_offsets"][0] != 1 or m3["m3_view_b_offsets"][0] != 1:
        raise RuntimeError("M3 A/B paths must share the nearest history anchor")
    if m3["m3_view_a_offsets"] == m3["m3_view_b_offsets"]:
        raise RuntimeError("M3 irregular path must differ from canonical path")

    require(m4, "use_m3_path_distillation", False)
    require(m4, "use_twc", False)
    require(m4, "hist_num", 3)
    require(m4, "m4_variant", "filter_tube")
    if int(require(m4, "point_sample_size")) != int(m3["point_sample_size"]):
        raise RuntimeError("M3/M4 must preserve the same network point budget")
    if float(require(m4, "m4_tube_max_length")) < float(
            require(m4, "m4_tube_base_length")):
        raise RuntimeError("M4 max tube length is below its base length")
    if float(require(m4, "m4_tube_max_width")) < float(
            require(m4, "m4_tube_base_width")):
        raise RuntimeError("M4 max tube width is below its base width")

    architecture_keys = (
        "dataset", "category_name", "bb_scale", "bb_offset",
        "point_sample_size", "hist_num", "net_model", "box_aware",
        "use_dynamics_encoder", "dynamics_hidden_dim",
        "dynamics_motion_mode", "use_physical_time_adapter",
    )
    mismatches = {
        key: {"m3": m3.get(key), "m4": m4.get(key)}
        for key in architecture_keys if m3.get(key) != m4.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "M3/M4 checkpoint architecture mismatch: "
            + json.dumps(mismatches, sort_keys=True))

    required_scripts = (
        "tools/run_m3_matched_abc.sh",
        "tools/run_m3_matched_evaluation.sh",
        "tools/run_m4_matched_evaluation.sh",
        "tools/run_m_stage_pipeline.sh",
        "tools/export_m4_endpoints.py",
    )
    missing_scripts = [
        relative for relative in required_scripts
        if not (ROOT / relative).is_file()
    ]
    if missing_scripts:
        raise RuntimeError(f"Missing M-stage scripts: {missing_scripts}")

    print(json.dumps({
        "status": "PASS_M_STAGE_CONFIG_CONTRACT",
        "m3": {
            "arms": ["single_view", "paired_weight0", "endpoint_distill"],
            "path_weight": m3["m3_path_weight"],
            "coarse_weight": m3["m3_coarse_weight"],
            "confidence": m3["m3_teacher_confidence_mode"],
        },
        "m4": {
            "variants": ["off", "filter", "tube", "filter_tube"],
            "point_budget": m4["point_sample_size"],
            "clock_controls": ["fixed", "real"],
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
