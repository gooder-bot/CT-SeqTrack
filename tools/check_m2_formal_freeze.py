#!/usr/bin/env python3
"""Fail-closed E6 preflight for the frozen M1/M2 seed42 formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = Path(
    "cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_formal_true.yaml"
)
ENGINEERING_CONFIG = Path(
    "cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_engineering.yaml"
)
FREEZE_REPORT = Path("compare_results/reports/m2_e6_parameter_freeze_20260722.json")
GAP_PROTOCOL_CONFIG = Path("cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml")
BURST_PROTOCOL_CONFIG = Path("cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml")
MANIFEST_RUNNER = Path("tools/prepare_m2_formal_manifests.sh")
TRAIN_RUNNER = Path("tools/run_m2_formal_seed42_gpu2.sh")
CONTROL_RUNNER = Path("tools/run_m2_formal_time_controls_gpu3.sh")

EXPECTED_FORMAL_CONFIG_CANONICAL_SHA256 = (
    "a5eccc9179902de26387496943b22cb7a7110647cc57cd4c44975cca98e1eca9"
)
EXPECTED_FREEZE_REPORT_CANONICAL_SHA256 = (
    "3865a4ca625d994a5bbf7a4754d381dcbeb5e38e1c2008f8e823a9eddf5e261e"
)
EXPECTED_A1_SHA256 = (
    "a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82"
)
EXPECTED_ORACLE_ENDPOINTS_SHA256 = (
    "aa2e890a5fcc3e15964bb89d87dc8c7873b0c97a29f437c220b8cd00e406099b"
)
EXPECTED_ORACLE_SUMMARY_SHA256 = (
    "2ecd6e707ffee6e6551effadb7c896f974988064579174d48ad8e7686ecf367a"
)

EXPECTED_STEPS_PER_EPOCH = 1262
EXPECTED_EPOCHS = 60
EXPECTED_OPTIMIZER_STEPS = 75_720
EXPECTED_BATCH_SIZE = 16
EXPECTED_WORKERS = 12
EXPECTED_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def canonical_sha256(data: object) -> str:
    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} drifted: expected {expected!r}, got {actual!r}")


def git_state() -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    full = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()
    return {
        "commit": commit,
        "dirty_tracked": bool(tracked),
        "dirty_any": bool(full),
        "status_porcelain": full,
    }


def validate_formal_config(path: Path) -> dict[str, object]:
    cfg = load_yaml(path)
    required = {
        "dataset": "nuscenes_mf",
        "version": "v1.0-mini",
        "category_name": "Car",
        "point_sample_size": 1024,
        "train_split": "mini_train",
        "val_split": "mini_val",
        "test_split": "mini_val",
        "num_candidates": 4,
        "candidate_trajectory_mode": "shared_se2",
        "use_augmentation": False,
        "hist_num": 3,
        "use_real_time": True,
        "main_time_source": "order",
        "dynamics_time_mode": "true",
        "dynamics_fixed_delta_t": 0.5,
        "use_dynamics_encoder": True,
        "dynamics_motion_mode": "proposal_innovation",
        "use_physical_time_adapter": True,
        "physical_time_adapter_scale": 1.0,
        "physical_time_adapter_warmup_epoch": 5,
        "dynamics_innovation_alpha": 0.75,
        "dynamics_innovation_scale": 1.0,
        "dynamics_innovation_radius_base": 0.5,
        "dynamics_innovation_radius_per_second": 0.5,
        "dynamics_innovation_radius_max": 2.0,
        "dynamics_innovation_warmup_epoch": 5,
        "dynamics_innovation_disable_on_empty_search": True,
        "use_observability_gate": False,
        "use_twc": False,
        "batch_size": EXPECTED_BATCH_SIZE,
        "workers": EXPECTED_WORKERS,
        "epoch": EXPECTED_EPOCHS,
        "optimizer": "Adam",
        "lr": 0.0001,
        "max_lr": 0.001,
    }
    for key, expected in required.items():
        if key not in cfg:
            raise RuntimeError(f"Formal config is missing {key!r}")
        require_equal(cfg[key], expected, f"formal config {key}")

    engineering = load_yaml(ROOT / ENGINEERING_CONFIG)
    differences = {
        key: {"engineering": engineering.get(key), "formal": cfg.get(key)}
        for key in sorted(set(engineering) | set(cfg))
        if engineering.get(key) != cfg.get(key)
    }
    allowed_differences = {
        "batch_size",
        "workers",
        "physical_time_adapter_warmup_epoch",
        "dynamics_innovation_warmup_epoch",
    }
    unexpected = set(differences).difference(allowed_differences)
    if unexpected:
        raise RuntimeError(
            "Formal config changed unapproved engineering fields: "
            + ", ".join(sorted(unexpected))
        )
    require_equal(set(differences), allowed_differences, "formal/engineering diff set")
    return {"required_values": required, "engineering_differences": differences}


def virtual_rate_fields(path: Path) -> dict[str, object]:
    return {
        key: value
        for key, value in load_yaml(path).items()
        if "virtual_rate" in key
    }


def validate_protocol_overlays() -> dict[str, object]:
    gap = virtual_rate_fields(ROOT / GAP_PROTOCOL_CONFIG)
    burst = virtual_rate_fields(ROOT / BURST_PROTOCOL_CONFIG)
    gap_expected = {
        "virtual_rate_mode": "gap_pattern",
        "virtual_rate_gap_pattern": [1, 1, 2, 4],
        "virtual_rate_seed": 42,
        "virtual_rate_max_gap": 5,
        "virtual_rate_manifest": "",
        "virtual_rate_keep_first": True,
        "virtual_rate_keep_last": True,
        "virtual_rate_min_tracklet_len": 6,
    }
    burst_expected = {
        "virtual_rate_mode": "burst_drop",
        "virtual_rate_burst_keep_lengths": [3, 2, 3],
        "virtual_rate_burst_skip_lengths": [2, 3, 3],
        "virtual_rate_seed": 42,
        "virtual_rate_max_gap": 5,
        "virtual_rate_manifest": "",
        "virtual_rate_keep_first": True,
        "virtual_rate_keep_last": True,
        "virtual_rate_min_tracklet_len": 6,
    }
    require_equal(gap, gap_expected, "gap1124 virtual-rate overlay")
    require_equal(burst, burst_expected, "burst-drop virtual-rate overlay")
    return {"gap1124": gap, "burst_drop": burst}


def validate_freeze_report(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    require_equal(
        report.get("schema"),
        "ct_seqtrack.m2_formal_parameter_freeze",
        "freeze schema",
    )
    require_equal(report.get("schema_version"), 1, "freeze schema version")
    require_equal(report.get("decision"), "FREEZE_M2_ALPHA_RADIUS", "freeze decision")
    require_equal(
        report["inputs"]["endpoints"]["sha256"],
        EXPECTED_ORACLE_ENDPOINTS_SHA256,
        "oracle endpoints SHA256",
    )
    require_equal(
        report["inputs"]["summary"]["sha256"],
        EXPECTED_ORACLE_SUMMARY_SHA256,
        "oracle summary SHA256",
    )
    require_equal(report["primary"]["sample_count"], 1311, "freeze sample count")
    require_equal(report["primary"]["tracklet_count"], 213, "freeze tracklet count")
    if not all(report["checks"].values()):
        raise RuntimeError(f"Freeze checks are not all true: {report['checks']}")
    return {
        "decision": report["decision"],
        "parameters": report["frozen_parameters"],
        "checks": report["checks"],
    }


def validate_source_hooks() -> dict[str, bool]:
    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    provenance_source = (ROOT / "utils/run_provenance.py").read_text(encoding="utf-8")
    manifest_source = (ROOT / MANIFEST_RUNNER).read_text(encoding="utf-8")
    train_source = (ROOT / TRAIN_RUNNER).read_text(encoding="utf-8")
    control_source = (ROOT / CONTROL_RUNNER).read_text(encoding="utf-8")
    checks = {
        "main_has_init_checkpoint": "--init_checkpoint" in main_source,
        "main_init_and_resume_exclusive": (
            "--checkpoint (resume/test) and --init_checkpoint are mutually exclusive"
            in main_source
        ),
        "provenance_hashes_init_checkpoint": (
            '"init_checkpoint_sha256": sha256_file(init_checkpoint)'
            in provenance_source
        ),
        "provenance_declares_last_primary": (
            "final/last is primary" in provenance_source
        ),
        "manifest_runner_covers_three_protocols": all(
            token in manifest_source
            for token in ("STANDARD_SHUFFLE", "GAP_SHUFFLE", "BURST_SHUFFLE")
        ),
        "train_runner_uses_model_only_a1_init": "--init_checkpoint" in train_source,
        "train_runner_disables_top_k_selection": "--save_top_k 0" in train_source,
        "train_runner_checks_75720_steps": "global_step != 75720" in train_source,
        "control_runner_uses_same_checkpoint_gate": (
            "--require-same-checkpoint" in control_source
        ),
        "control_runner_has_true_fixed_shuffled": all(
            token in control_source
            for token in (
                "--dynamics-time-mode true",
                "--dynamics-time-mode fixed",
                "--dynamics-time-mode shuffled",
            )
        ),
        "control_runner_exports_a1": "--run-label A1" in control_source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Formal initialization/provenance hook missing: {checks}")
    return checks


def validate_dataset(config_path: Path, data_root: Path) -> dict[str, int]:
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    sys.path.insert(0, str(ROOT))
    from easydict import EasyDict  # pylint: disable=import-outside-toplevel
    from datasets import get_dataset  # pylint: disable=import-outside-toplevel

    cfg = EasyDict(load_yaml(config_path))
    cfg.path = str(data_root)
    cfg.preloading = False
    cfg.tiny = False
    dataset = get_dataset(
        cfg, type=cfg.train_type, split=cfg.train_split, protocol_role="train"
    )
    sample_count = len(dataset)
    steps_per_epoch = sample_count // EXPECTED_BATCH_SIZE
    total_steps = steps_per_epoch * EXPECTED_EPOCHS
    require_equal(steps_per_epoch, EXPECTED_STEPS_PER_EPOCH, "steps per epoch")
    require_equal(total_steps, EXPECTED_OPTIMIZER_STEPS, "optimizer steps")
    return {
        "train_sample_count": int(sample_count),
        "batch_size": EXPECTED_BATCH_SIZE,
        "drop_last": True,
        "steps_per_epoch": int(steps_per_epoch),
        "epochs": EXPECTED_EPOCHS,
        "optimizer_steps": int(total_steps),
    }


def run_checks(args: argparse.Namespace) -> dict[str, object]:
    config_path = ROOT / FORMAL_CONFIG
    freeze_path = ROOT / FREEZE_REPORT
    for path in (
        config_path,
        freeze_path,
        ROOT / ENGINEERING_CONFIG,
        ROOT / GAP_PROTOCOL_CONFIG,
        ROOT / BURST_PROTOCOL_CONFIG,
        ROOT / MANIFEST_RUNNER,
        ROOT / TRAIN_RUNNER,
        ROOT / CONTROL_RUNNER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    formal_data = load_yaml(config_path)
    with freeze_path.open("r", encoding="utf-8") as handle:
        freeze_data = json.load(handle)
    formal_sha = sha256_file(config_path)
    freeze_sha = sha256_file(freeze_path)
    formal_canonical_sha = canonical_sha256(formal_data)
    freeze_canonical_sha = canonical_sha256(freeze_data)
    require_equal(
        formal_canonical_sha,
        EXPECTED_FORMAL_CONFIG_CANONICAL_SHA256,
        "formal config canonical SHA256",
    )
    require_equal(
        freeze_canonical_sha,
        EXPECTED_FREEZE_REPORT_CANONICAL_SHA256,
        "freeze report canonical SHA256",
    )
    payload: dict[str, object] = {
        "schema": "ct_seqtrack.m2_formal_e6_preflight",
        "schema_version": 1,
        "decision": "PASS_E6_FORMAL_FREEZE",
        "contract": {
            "seed": EXPECTED_SEED,
            "batch_size": EXPECTED_BATCH_SIZE,
            "workers": EXPECTED_WORKERS,
            "epochs": EXPECTED_EPOCHS,
            "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
            "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "checkpoint_selection": "epoch60 last.ckpt only",
            "training_time_mode": "true",
            "fixed_shuffled_role": "same-checkpoint evaluation only",
        },
        "files": {
            "formal_config": {
                "path": FORMAL_CONFIG.as_posix(),
                "file_sha256": formal_sha,
                "canonical_sha256": formal_canonical_sha,
            },
            "freeze_report": {
                "path": FREEZE_REPORT.as_posix(),
                "file_sha256": freeze_sha,
                "canonical_sha256": freeze_canonical_sha,
            },
        },
        "formal_config": validate_formal_config(config_path),
        "freeze": validate_freeze_report(freeze_path),
        "protocol_overlays": validate_protocol_overlays(),
        "source_hooks": validate_source_hooks(),
    }

    if args.a1_checkpoint:
        checkpoint_path = Path(args.a1_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual = sha256_file(checkpoint_path)
        require_equal(actual, EXPECTED_A1_SHA256, "A1 checkpoint SHA256")
        payload["a1_checkpoint"] = {"path": str(checkpoint_path), "sha256": actual}
    elif args.require_server_inputs:
        raise RuntimeError("--require-server-inputs requires --a1-checkpoint")

    if args.data_root:
        payload["dataset_contract"] = validate_dataset(
            config_path, Path(args.data_root)
        )
    elif args.require_server_inputs:
        raise RuntimeError("--require-server-inputs requires --data-root")

    state = git_state()
    payload["git"] = state
    if args.expected_commit:
        require_equal(state["commit"], args.expected_commit, "Git commit")
    elif args.require_server_inputs:
        raise RuntimeError("--require-server-inputs requires --expected-commit")
    if args.require_clean_git and state["dirty_any"]:
        raise RuntimeError(
            "Formal E6 preflight requires a clean worktree: "
            + " | ".join(state["status_porcelain"])
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--expected-commit")
    parser.add_argument("--output")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--require-server-inputs", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        args.a1_checkpoint = None
        args.data_root = None
        args.expected_commit = None
        args.require_clean_git = False
        args.require_server_inputs = False
    payload = run_checks(args)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print("M2 formal E6 preflight: PASS")
    print(json.dumps(payload["contract"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
