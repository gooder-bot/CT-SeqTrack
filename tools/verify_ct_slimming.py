"""Build and verify the non-destructive CT-SeqTrack slimming baseline.

This tool never writes into ``output/`` and never mutates Git refs or history.
It records the existing ``001951a`` tree as the recovery source, snapshots the
active configuration closure, and detects accidental changes to the protected
local experiment output inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "docs" / "slimming_baseline"
BASELINE_COMMIT = "001951a3aee15fdad6e5d5e32ca02d87bf083f3a"

ACTIVE_CONFIGS = (
    "cfgs/ct_seqtrack/24_b0_candidate1_control.yaml",
    "cfgs/ct_seqtrack/24_b0.yaml",
    "cfgs/ct_seqtrack/24_b1.yaml",
    "cfgs/ct_seqtrack/24_full_minus_b3.yaml",
    "cfgs/ct_seqtrack/24_full.yaml",
    "cfgs/ct_seqtrack/24_full_cv.yaml",
    "cfgs/ct_seqtrack/24_full_minus_b3_cv.yaml",
    "cfgs/ct_seqtrack/24_b1_time_fixed.yaml",
    "cfgs/ct_seqtrack/24_b1_time_shuffled.yaml",
    "cfgs/ct_seqtrack/24_full_minus_b3_time_fixed.yaml",
    "cfgs/ct_seqtrack/24_full_minus_b3_time_shuffled.yaml",
    "cfgs/ct_seqtrack/24_full_time_fixed.yaml",
    "cfgs/ct_seqtrack/24_full_time_shuffled.yaml",
    "cfgs/ct_seqtrack/24_full_memory_real.yaml",
    "cfgs/ct_seqtrack/24_full_memory_empty.yaml",
    "cfgs/ct_seqtrack/24_full_memory_time_misaligned.yaml",
    "cfgs/ct_seqtrack/24_b0_nuscenes_full.yaml",
    "cfgs/ct_seqtrack/24_b1_nuscenes_full.yaml",
    "cfgs/ct_seqtrack/24_full_minus_b3_nuscenes_full.yaml",
    "cfgs/ct_seqtrack/24_full_nuscenes_full.yaml",
    "cfgs/ct_v2/19_b4_decoder_alignment.yaml",
    "cfgs/ct_v2/20_b4_decoder_anticollapse.yaml",
)

RUNTIME_CLOSURE = (
    "main.py",
    "models/__init__.py",
    "models/base_model.py",
    "models/seqtrack3d.py",
    "models/ctseqtrack.py",
    "models/ct_variant.py",
    "models/ct_v2/pipeline.py",
    "models/ct_v2/pipeline_contracts.py",
    "models/ct_v2/motion.py",
    "models/ct_v2/evidence_memory.py",
    "datasets/__init__.py",
    "datasets/sampler.py",
    "utils/config.py",
    "utils/recursive_state.py",
    "utils/training_isolation.py",
    "utils/online_contract.py",
    "utils/action_calibration.py",
    "utils/acquisition_metrics.py",
)

TOOL_CLOSURE = (
    "tools/calibrate_b1_uncertainty.py",
    "tools/calibrate_ct_actions.py",
    "tools/export_b1_calibration.py",
    "tools/report_ct_b1.py",
    "tools/report_ct_b2.py",
    "tools/report_ct_risk_coverage.py",
    "tools/report_ct_memory.py",
    "tools/compare_ct_module_audits.py",
    "tools/check_candidate_shared_se2.py",
    "tools/check_forward_batch.py",
    "tools/check_time_batch.py",
    "tools/check_train_steps.py",
    "tools/visualize_model_predictions.py",
    "tools/visualize_pointcloud_sample.py",
    "tools/verify_ct_slimming.py",
)

EVIDENCE_CLOSURE = (
    "README.md",
    "docs/CTSEQTRACK_B0_B3_METHOD.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/FORMAL_TOOLING.md",
    "docs/HISTORY_EVIDENCE_INDEX.md",
    "docs/SOURCE_SLIMMING_GATE.md",
    "need_to_do.md",
    "research_handoff.json",
)


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n").encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _git_paths(*args: str) -> list[str]:
    completed = subprocess.run(
        ("git", *args, "-z"), cwd=ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in completed.stdout.split(b"\0") if item
    ]


def _tracked_snapshot() -> list[dict[str, object]]:
    current = _git_paths("ls-files")
    baseline = set(_git_paths("ls-tree", "-r", "--name-only", BASELINE_COMMIT))
    rows = []
    for relative in current:
        path = ROOT / relative
        payload = path.read_bytes()
        rows.append({
            "path": relative.replace("\\", "/"),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "recoverable_from": (
                f"{BASELINE_COMMIT}:{relative.replace(os.sep, '/')}"
                if relative in baseline else None
            ),
        })
    return rows


def _output_inventory() -> dict[str, object]:
    output = ROOT / "output"
    rows = []
    total_bytes = 0
    if output.is_dir():
        for path in sorted(item for item in output.rglob("*") if item.is_file()):
            stat = path.stat()
            relative = path.relative_to(output).as_posix()
            total_bytes += stat.st_size
            rows.append(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n")
    inventory_sha256 = _sha256_bytes("".join(rows).encode("utf-8"))
    ignored = _run("git", "check-ignore", "-q", "output", check=False).returncode == 0
    return {
        "protected_path": str(output.resolve()),
        "policy": "do_not_delete_move_rename_or_write",
        "exists": output.is_dir(),
        "git_ignored": ignored,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "inventory_sha256_path_size_mtime": inventory_sha256,
    }


def _resolved_configs() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from utils.config import load_yaml_config

    result = {}
    for relative in ACTIVE_CONFIGS:
        resolved = load_yaml_config(ROOT / relative)
        payload = _json_bytes(resolved)
        result[relative] = {
            "sha256": _sha256_bytes(payload),
            "resolved": resolved,
        }
    return result


def _environment() -> dict[str, object]:
    packages = {}
    for name in (
            "torch", "pytorch_lightning", "torchmetrics", "easydict",
            "nuscenes", "pyquaternion"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "installed")
        except Exception as error:  # dependency inventory must not abort snapshot
            packages[name] = f"unavailable: {type(error).__name__}: {error}"
    gpu = _run(
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader", check=False)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi_returncode": gpu.returncode,
        "nvidia_smi": gpu.stdout.strip(),
    }


def _model_snapshot() -> dict[str, object]:
    try:
        import numpy as np
        import torch

        sys.path.insert(0, str(ROOT))
        from models import get_model
        from models.ct_variant import configure_ct_variant
        from utils.config import load_yaml_config

        result = {"status": "complete", "arms": {}}
        for name in (
                "24_b0.yaml", "24_b1.yaml",
                "24_full_minus_b3.yaml", "24_full.yaml"):
            torch.manual_seed(42)
            np.random.seed(42)
            config = SimpleNamespace(**load_yaml_config(
                ROOT / "cfgs" / "ct_seqtrack" / name))
            configure_ct_variant(config)
            model = get_model(config.net_model)(
                config=config, train_dataloader_length=1)
            state_rows = []
            digest = hashlib.sha256()
            for parameter_name, tensor in model.state_dict().items():
                cpu = tensor.detach().cpu().contiguous()
                raw = cpu.numpy().tobytes()
                row = {
                    "name": parameter_name,
                    "shape": list(cpu.shape),
                    "dtype": str(cpu.dtype),
                    "sha256": _sha256_bytes(raw),
                }
                state_rows.append(row)
                digest.update(_json_bytes(row))
            trainable = {
                parameter_name: bool(parameter.requires_grad)
                for parameter_name, parameter in model.named_parameters()
            }
            result["arms"][name] = {
                "state_dict_sha256": digest.hexdigest(),
                "state": state_rows,
                "trainable": trainable,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad),
            }
        return result
    except Exception as error:
        return {
            "status": "blocked",
            "reason": f"{type(error).__name__}: {error}",
            "required_action": (
                "Run again in the complete Lightning/nuScenes environment "
                "before physically deleting dormant source branches."
            ),
        }


def snapshot(run_tests: bool) -> None:
    head = _run("git", "rev-parse", "HEAD").stdout.strip()
    if head != BASELINE_COMMIT:
        raise RuntimeError(f"expected HEAD {BASELINE_COMMIT}, observed {head}")
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    tracked = _tracked_snapshot()
    with (BASELINE_DIR / "tracked_files.jsonl").open(
            "w", encoding="utf-8", newline="\n") as handle:
        for row in tracked:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    _write_json(BASELINE_DIR / "active_configs_resolved.json", _resolved_configs())
    _write_json(BASELINE_DIR / "output_protection.json", _output_inventory())
    _write_json(BASELINE_DIR / "environment.json", _environment())
    _write_json(BASELINE_DIR / "model_initialization.json", _model_snapshot())
    _write_json(BASELINE_DIR / "closures.json", {
        "baseline_commit": BASELINE_COMMIT,
        "active_configs": ACTIVE_CONFIGS,
        "runtime": RUNTIME_CLOSURE,
        "tools": TOOL_CLOSURE,
        "evidence": EVIDENCE_CLOSURE,
        "claim_boundary": (
            "Current B0--B3 experiments are pending; historical results are "
            "scope evidence and must not be presented as current gains."
        ),
    })
    if run_tests:
        completed = _run(sys.executable, "-m", "pytest", "-q", check=False)
        (BASELINE_DIR / "pytest_baseline.txt").write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8", newline="\n")
        if completed.returncode != 0:
            raise RuntimeError("baseline pytest failed")
    summary = {
        "baseline_commit": BASELINE_COMMIT,
        "tracked_files": len(tracked),
        "tracked_bytes": sum(int(row["bytes"]) for row in tracked),
        "all_tracked_recoverable": all(
            row["recoverable_from"] is not None for row in tracked),
        "active_config_count": len(ACTIVE_CONFIGS),
        "output": _output_inventory(),
        "git_history_policy": (
            "read-only: no commit, tag, branch, stash, bundle, reset, rebase, "
            "filter, gc, or history rewrite"
        ),
    }
    _write_json(BASELINE_DIR / "baseline_summary.json", summary)


def verify() -> None:
    head = _run("git", "rev-parse", "HEAD").stdout.strip()
    if head != BASELINE_COMMIT:
        raise RuntimeError(f"expected HEAD {BASELINE_COMMIT}, observed {head}")
    summary = json.loads(
        (BASELINE_DIR / "baseline_summary.json").read_text(encoding="utf-8"))
    expected_output = summary["output"]
    current_output = _output_inventory()
    if current_output != expected_output:
        raise RuntimeError(
            "protected output inventory changed:\n"
            + json.dumps({
                "expected": expected_output,
                "current": current_output,
            }, ensure_ascii=False, indent=2))
    baseline_configs = json.loads(
        (BASELINE_DIR / "active_configs_resolved.json").read_text(
            encoding="utf-8"))
    current_configs = _resolved_configs()
    mismatches = [
        name for name in ACTIVE_CONFIGS
        if baseline_configs[name]["sha256"] != current_configs[name]["sha256"]
    ]
    if mismatches:
        raise RuntimeError(
            "resolved configuration mismatch: " + ", ".join(mismatches))
    formal_arms = (
        "cfgs/ct_seqtrack/24_b0.yaml",
        "cfgs/ct_seqtrack/24_b1.yaml",
        "cfgs/ct_seqtrack/24_full_minus_b3.yaml",
        "cfgs/ct_seqtrack/24_full.yaml",
    )
    candidate_contract = {
        "num_candidates": 4,
        "ct_recursive_candidate_views": 4,
        "ct_b0_candidate_views": 4,
        "ct_b0_candidate_weights": [
            0.5, 0.1666667, 0.1666667, 0.1666667],
        "ct_b2_candidate_views": 1,
        "ct_recovery_candidate_policy": "off",
    }
    contract_mismatches = []
    for name in formal_arms:
        resolved = current_configs[name]["resolved"]
        for key, expected in candidate_contract.items():
            if resolved.get(key) != expected:
                contract_mismatches.append(
                    f"{name}:{key}={resolved.get(key)!r}")
    if contract_mismatches:
        raise RuntimeError(
            "formal B0=4/B2=1 candidate contract mismatch: "
            + ", ".join(contract_mismatches))
    print("slimming baseline verification passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--run-tests", action="store_true")
    subparsers.add_parser("verify")
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.run_tests)
    else:
        verify()


if __name__ == "__main__":
    main()
