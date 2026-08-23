"""Run a bounded CT-SeqTrack training transaction through ``main.py``.

This checker deliberately reuses the production dataloader, online recursive
state, Lightning hooks and the configured optimizer topology.  It does not implement a
second approximation of the training step and never writes to the protected
local ``output/`` directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_variant import configure_ct_variant  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402
from utils.online_contract import validate_scratch_training_contract  # noqa: E402


FORMAL_VARIANTS = {"b0", "b1", "full_minus_b3", "full"}


def validate_config(path: Path) -> dict:
    config = load_yaml_config(path)
    if str(config.get("net_model", "")).strip().lower() != "ctseqtrack":
        raise ValueError("bounded CT check requires net_model=ctseqtrack")
    configure_ct_variant(config)
    validate_scratch_training_contract(config)
    variant = str(config.get("ct_variant", "")).strip().lower()
    if variant not in FORMAL_VARIANTS:
        raise ValueError(f"unsupported formal CT variant: {variant!r}")
    expected = {
        "num_candidates": 4,
        "ct_recursive_candidate_views": 4,
        "ct_b0_candidate_views": 4,
        "ct_b2_candidate_views": 1,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "formal candidate contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    safe_auto = (
        str(config.get("ct_runtime_protocol", "")).strip().lower()
        == "safe_seqtrack_auto_v1")
    expected_weights = (
        [0.5, 1 / 6, 1 / 6, 1 / 6]
        if safe_auto else [0.25, 0.25, 0.25, 0.25])
    observed_weights = [
        float(value) for value
        in config.get("ct_b0_candidate_weights", [])]
    if (len(observed_weights) != len(expected_weights)
            or any(abs(observed - expected) > 1e-12
                   for observed, expected in zip(
                       observed_weights, expected_weights))):
        raise ValueError(
            "formal B0 candidate weights do not match the runtime protocol")
    return config


def build_command(args, artifact_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "main.py"),
        "--cfg",
        str(args.cfg.resolve()),
        "--batch_size",
        "16",
        "--epoch",
        "1",
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--limit_train_batches",
        str(args.steps),
        "--limit_val_batches",
        "1",
        "--check_val_every_n_epoch",
        "1",
        "--log_dir",
        str(artifact_dir),
        "--tag",
        args.tag,
    ]
    if args.path is not None:
        command.extend(("--path", args.path))
    if args.preloading:
        command.append("--preloading")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded production CT training transaction. This is a "
            "smoke check, not a 100-step/resume equivalence proof."
        )
    )
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument(
        "--steps", type=int, default=8,
        help=("bounded observation steps; use at least 8 for dual-stream "
              "plugin arms so the first scheduled mechanism transaction runs"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--tag", default="ct-bounded-train-check")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/ct_checks/bounded_train"),
    )
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be a positive integer")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if not args.cfg.is_file():
        raise FileNotFoundError(args.cfg)

    config = validate_config(args.cfg)
    artifact_dir = args.artifact_dir.resolve()
    protected_output = (ROOT / "output").resolve()
    if artifact_dir == protected_output or protected_output in artifact_dir.parents:
        raise ValueError("acceptance artifacts may not be written under output/")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(
            f"artifact directory must be new or empty: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args, artifact_dir)
    manifest = {
        "schema": "ct_seqtrack.bounded_train_check.v2",
        "variant": config["ct_variant"],
        "runtime_protocol": config.get("ct_runtime_protocol"),
        "optimizer_topology": config.get("ct_optimizer_topology"),
        "observation_rng_mode": config.get("ct_observation_rng_mode"),
        "candidate_contract": {
            "b0_views": 4,
            "b2_views": 1,
            "b0_weights": list(config["ct_b0_candidate_weights"]),
        },
        "command": command,
        "status": "launching",
    }
    manifest_path = artifact_dir / "check_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    manifest["status"] = "passed" if completed.returncode == 0 else "failed"
    manifest["returncode"] = completed.returncode
    if completed.returncode == 0:
        checkpoints = sorted(artifact_dir.rglob("last.ckpt"))
        if len(checkpoints) != 1:
            manifest["status"] = "failed"
            manifest["postcheck_error"] = (
                "expected exactly one disposable last.ckpt, found "
                f"{len(checkpoints)}")
            completed = subprocess.CompletedProcess(
                command, returncode=2)
            manifest["returncode"] = completed.returncode
        else:
            checkpoint = torch.load(
                checkpoints[0], map_location="cpu", weights_only=False)
            manifest["checkpoint"] = str(checkpoints[0])
            manifest["module_audit"] = checkpoint.get("ct_module_audit")
            manifest["b0_prefix_hashes"] = checkpoint.get(
                "ct_b0_prefix_hashes")
            manifest["observation_batch_fingerprints"] = checkpoint.get(
                "ct_observation_batch_fingerprints")
            manifest["cuda_stage_audit"] = checkpoint.get(
                "ct_cuda_stage_audit")
            if str(config.get("ct_runtime_protocol", "")) == (
                    "safe_seqtrack_auto_v1"):
                prefixes = manifest["b0_prefix_hashes"] or {}
                required = ["initial", "step_1"]
                if args.steps >= 100:
                    required.append("step_100")
                missing = [key for key in required if key not in prefixes]
                fingerprints = (
                    manifest["observation_batch_fingerprints"] or [])
                expected_fingerprints = min(int(args.steps), 100)
                if missing or len(fingerprints) < expected_fingerprints:
                    manifest["status"] = "failed"
                    manifest["postcheck_error"] = {
                        "missing_b0_prefix_hashes": missing,
                        "expected_observation_fingerprints": (
                            expected_fingerprints),
                        "observed_observation_fingerprints": len(fingerprints),
                    }
                    completed = subprocess.CompletedProcess(
                        command, returncode=2)
                    manifest["returncode"] = completed.returncode
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(f"bounded production training check passed: {artifact_dir}")


if __name__ == "__main__":
    main()
