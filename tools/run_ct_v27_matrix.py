"""导出 v27 六运行矩阵；只有 --execute 才顺序启动从零训练。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.config import load_yaml_config


ARMS = ("seqtrack_reference", "b0", "b1_gru", "b1_cfc", "full_minus_b3", "full")
CATEGORIES = ("Car", "Pedestrian", "Truck", "Trailer", "Bus")
LATE_EPOCHS = (58, 59, 60)
SCHEMA = "ct_seqtrack.run_matrix.v27"


def safe_output_directory(directory):
    destination = Path(directory).expanduser().resolve()
    protected = (ROOT / "output").resolve()
    if destination == protected or protected in destination.parents:
        raise ValueError("v27 matrix artifacts must not use the protected output/ directory")
    return destination


def _source_config(arm, stage):
    suffix = "_nuscenes_full" if stage == "full" else ""
    parent = ROOT / "cfgs" if arm == "seqtrack_reference" else ROOT / "cfgs/ct_seqtrack"
    return parent / f"27_{arm}{suffix}.yaml"


def _sha_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _powershell_command(argv):
    return "& " + " ".join("'" + str(part).replace("'", "''") + "'" for part in argv)


def build_matrix(stage, data_path, output_directory, python="python"):
    """纯计划构建，不读取数据/检查点，也不启动训练或评估。"""
    if stage not in ("mini", "full"):
        raise ValueError("stage must be mini or full")
    output = safe_output_directory(output_directory)
    categories = ("Car",) if stage == "mini" else CATEGORIES
    runs, configs = [], {}
    for category in categories:
        for arm in ARMS:
            source = _source_config(arm, stage)
            config = copy.deepcopy(load_yaml_config(source))
            if (not config.get("ct_enable_v27") or config.get("epoch") != 60
                    or config.get("seed") != 42 or config.get("from_epoch") != 0
                    or config.get("ct_initialization_policy") != "scratch_only"
                    or config.get("init_checkpoint")):
                raise ValueError(f"invalid scratch v27 formal config: {source}")
            name = f"ct27_{stage}_{category.lower()}_{arm}_seed42_60ep_bs16"
            config.update(category_name=category, path=str(data_path), batch_size=16)
            config_path = output / "configs" / f"{name}.yaml"
            configs[str(config_path)] = config
            run_directory = output / "runs" / name
            train = [python, "main.py", "--cfg", str(config_path), "--path", str(data_path),
                     "--category_name", category, "--seed", "42", "--epoch", "60",
                     "--batch_size", "16", "--tag", name, "--log_dir", str(run_directory)]
            followups = []
            for epoch in LATE_EPOCHS:
                checkpoint = run_directory / "formal_checkpoints" / f"epoch={epoch:03d}.ckpt"
                evaluation = [python, "main.py", "--cfg", str(config_path), "--path", str(data_path),
                              "--category_name", category, "--checkpoint", str(checkpoint), "--test",
                              "--seed", "42", "--batch_size", "16", "--log_dir",
                              str(output / "evaluations" / name / f"epoch_{epoch:03d}")]
                task = {"epoch": epoch, "checkpoint": str(checkpoint),
                        "status": "pending_training_checkpoint", "evaluation_argv": evaluation}
                if arm == "full":
                    policy = output / "calibration" / name / f"epoch_{epoch:03d}.json"
                    task["calibration_argv"] = [python, "tools/calibrate_ct_actions.py", "--v27",
                        "--config", str(config_path), "--checkpoint", str(checkpoint),
                        "--path", str(data_path), "--output", str(policy)]
                    task["calibration_artifact"] = str(policy)
                    task["evaluation_argv"] += ["--ct_action_calibration_path", str(policy)]
                    task["order"] = ["wait_for_checkpoint", "calibrate_this_checkpoint", "evaluate_official_split"]
                else:
                    task["order"] = ["wait_for_checkpoint", "evaluate_official_split"]
                followups.append(task)
            runs.append({"run_id": name, "arm": arm, "category": category,
                         "source_config": str(source.relative_to(ROOT)).replace("\\", "/"),
                         "resolved_config": str(config_path), "resolved_config_sha256": _sha_config(config),
                         "run_directory": str(run_directory), "training_status": "not_started",
                         "train_argv": train, "next_commands": followups})
    return {
        "schema": SCHEMA, "stage": stage, "working_directory": str(ROOT),
        "data_path": str(data_path), "run_count": len(runs), "categories": list(categories),
        "execution_requested": False, "evaluation_executed": False,
        "reporting": {"final_epoch": 60, "late3_epochs": list(LATE_EPOCHS),
                      "late3_aggregation": "arithmetic mean of the three independent epoch metrics",
                      "best_epoch_selection": False, "full_policy_per_checkpoint": True},
        "runs": runs,
    }, configs


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_matrix(manifest, configs, output_directory):
    output = safe_output_directory(output_directory)
    prior_manifest = output / "manifest.json"
    if prior_manifest.is_file():
        prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if prior.get("schema") != SCHEMA or any(
                run.get("training_status") != "not_started" for run in prior.get("runs", [])):
            raise ValueError("matrix output already records execution; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    for path, config in configs.items():
        target = Path(path)
        if output not in target.resolve().parents:
            raise ValueError("resolved config escaped the matrix output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")
    _write_json(output / "manifest.json", manifest)
    bash = ["#!/usr/bin/env bash", "set -euo pipefail", "# Training only; review manifest.json before execution.",
            "cd " + shlex.quote(str(ROOT))]
    powershell = ["$ErrorActionPreference = 'Stop'", "# Training only; review manifest.json before execution.",
                  "Set-Location -LiteralPath '" + str(ROOT).replace("'", "''") + "'"]
    pending = ["# Pending commands: run only after the corresponding training checkpoint exists.",
               "# Full: calibrate each checkpoint first, then evaluate with its own policy."]
    for run in manifest["runs"]:
        bash.append(shlex.join(run["train_argv"]))
        powershell += [_powershell_command(run["train_argv"]),
                       "if ($LASTEXITCODE -ne 0) { throw 'Training failed; stop this matrix.' }"]
        for task in run["next_commands"]:
            pending.append(f"\n# {run['run_id']}: epoch {task['epoch']}")
            if "calibration_argv" in task:
                pending.append(shlex.join(task["calibration_argv"]))
            pending.append(shlex.join(task["evaluation_argv"]))
    (output / "commands.sh").write_text("\n".join(bash) + "\n", encoding="utf-8")
    (output / "commands.ps1").write_text("\n".join(powershell) + "\n", encoding="utf-8")
    (output / "next_commands.txt").write_text("\n".join(pending) + "\n", encoding="utf-8")


def execute_training(manifest, output_directory):
    output = safe_output_directory(output_directory)
    if not Path(manifest["data_path"]).is_dir():
        raise ValueError("--execute requires an existing nuScenes data root")
    for run in manifest["runs"]:
        directory = Path(run["run_directory"])
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"scratch matrix refuses a nonempty run directory: {directory}")
    manifest["execution_requested"] = True
    _write_json(output / "manifest.json", manifest)
    for run in manifest["runs"]:
        run["training_status"] = "running"
        _write_json(output / "manifest.json", manifest)
        try:
            completed = subprocess.run(run["train_argv"], cwd=ROOT, check=False)
        except BaseException as error:
            run["training_status"] = "interrupted_or_launch_failed"
            run["failure_type"] = type(error).__name__
            _write_json(output / "manifest.json", manifest)
            raise
        run["exit_code"] = completed.returncode
        run["training_status"] = "process_succeeded" if completed.returncode == 0 else "failed"
        _write_json(output / "manifest.json", manifest)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, run["train_argv"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("mini", "full"))
    parser.add_argument("--path", required=True, help="nuScenes data root used by all runs")
    parser.add_argument("--output", required=True, help="new matrix artifact directory outside protected output/")
    parser.add_argument("--execute", action="store_true", help="sequential scratch training; never starts follow-up evaluation")
    parser.add_argument("--python", default="python", help="Python executable for reviewed commands")
    args = parser.parse_args(argv)
    manifest, configs = build_matrix(args.stage, args.path, args.output, args.python)
    write_matrix(manifest, configs, args.output)
    if args.execute:
        execute_training(manifest, args.output)
    print(json.dumps({"manifest": str(safe_output_directory(args.output) / "manifest.json"),
                      "runs": manifest["run_count"], "training_executed": bool(args.execute),
                      "followups_executed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
