"""v27 离线阈值拟合的真实 tracker 闭环 runner；不启动新的参数训练。"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
import numpy as np

from utils.action_calibration_v27 import (
    ROWS_SCHEMA, SCORE_DEFINITION, action_calibration_config_identity,
    calibrate_actions_v27, normalize_policy, normalize_rows, summarize_rows,
    validate_scene_manifest, sha256_file, sha256_json,
)
from utils.config import load_yaml_config


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fields = sorted({k for row in rows for k in row})
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class TrackerClosedLoopRunner:
    """每个 policy 重置序列状态并运行真实 evaluate_one_sequence。

    缓存仅复用完全相同 role/policy 的已完成闭环，不重筛旧轨迹模拟部署。
    """
    def __init__(self, config_path, checkpoint_path, *, path=None, device="auto",
                 scene_manifest_path=None, preloading=False, cache_directory=None):
        import torch
        from easydict import EasyDict
        from models import get_model
        from models.ct_variant import configure_ct_variant
        from utils.checkpoint_loading import load_initial_weights
        raw = load_yaml_config(config_path)
        configure_ct_variant(raw)
        if not raw.get("ct_enable_v27") or raw.get("ct_variant") != "full":
            raise ValueError("v27 action calibration requires a v27 Full config")
        self.config_sha256 = sha256_json(action_calibration_config_identity(raw))
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        raw.update({"preloading": preloading, "ct_action_calibration_path": None,
                    "proposal_inference_mode": "observation"})
        if path:
            raw["path"] = path
        self.config = EasyDict(raw)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.model = get_model(self.config.net_model)(self.config)
        load_initial_weights(self.model, checkpoint_path, require_complete=True)
        # Export switches must not alter strict saved-config checking during restore.
        self.config.export_proposal_diagnostics = True
        self.config.export_v3_candidate_diagnostics = True
        self.model.config.export_proposal_diagnostics = True
        self.model.config.export_v3_candidate_diagnostics = True
        self.model.to(self.device).eval()
        self.datasets, self.cache = {}, {}
        self.cache_directory = Path(cache_directory) if cache_directory else None
        self.expected_scene_manifest = None
        if scene_manifest_path:
            self.expected_scene_manifest = validate_scene_manifest(json.loads(
                Path(scene_manifest_path).read_text(encoding="utf-8")))
        self.scene_manifest = None

    def dataset(self, role):
        if role not in ("calibration", "dev"):
            raise ValueError("calibration runner cannot evaluate the official test role")
        if role not in self.datasets:
            from datasets import get_dataset
            config = copy.deepcopy(self.config)
            config.ct_protocol_role = role
            dataset = get_dataset(config, type="test", split=config.train_split, protocol_role=role)
            source = getattr(dataset, "dataset", dataset)
            manifest = validate_scene_manifest(source.ct_scene_manifest)
            if self.expected_scene_manifest and manifest != self.expected_scene_manifest:
                raise ValueError("runtime scene manifest differs from supplied manifest")
            if self.scene_manifest and manifest != self.scene_manifest:
                raise ValueError("calibration and dev must use the same scene protocol")
            self.scene_manifest = manifest
            self.datasets[role] = dataset
        return self.datasets[role]

    def __call__(self, role, policy):
        import torch
        policy = normalize_policy(policy)
        key = (role, sha256_json(policy))
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        dataset = self.dataset(role)
        source = getattr(dataset, "dataset", dataset)
        seed = int(getattr(self.config, "seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.model.ct_joint_router.install_policy(policy)
        self.model.proposal_inference_mode = "selective"
        self.model.config.proposal_inference_mode = "selective"
        rows = []
        with torch.inference_mode():
            for index in range(len(dataset)):
                sequence = dataset[index]
                tracklet_key = str(source.get_tracklet_key(index))
                meta = source.virtual_rate_meta[index]
                scene_id = str(source.nusc.get("scene", meta["scene_token"])["name"])
                self.model.evaluate_one_sequence(sequence)
                endpoints = getattr(self.model, "_ct_v27_sequence_endpoints", None)
                if endpoints is None or len(endpoints) != len(sequence):
                    raise RuntimeError("v27 host must export every frame, including initialization and empty-input fallbacks")
                for endpoint in endpoints:
                    row = dict(endpoint)
                    row.update(tracklet_id=tracklet_key, scene_id=scene_id,
                               category=str(getattr(self.config, "category_name", "unknown")),
                               tracklet_index=index, partition=role)
                    rows.append(row)
        rows = normalize_rows(rows)
        self.cache[key] = copy.deepcopy(rows)
        if self.cache_directory:
            write_rows(self.cache_directory / f"{role}_{key[1][:16]}.jsonl", rows)
            from utils.v27_eval_reporting import write_endpoint_diagnostics
            write_endpoint_diagnostics(self.cache_directory / f"{role}_{key[1][:16]}_summary.json", rows)
        print(json.dumps({"phase": "v27_closed_loop", "role": role, "policy": policy,
                          "metrics": summarize_rows(rows)}, sort_keys=True), flush=True)
        return rows


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--v27", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scene-manifest")
    parser.add_argument("--preloading", action="store_true")
    return parser


def export_main(argv=None):
    parser = _parser("Export complete v27 observation-recursive action endpoints")
    parser.add_argument("--partition", choices=("calibration", "dev"), required=True)
    args = parser.parse_args(argv)
    runner = TrackerClosedLoopRunner(args.config, args.checkpoint, path=args.path,
        device=args.device, scene_manifest_path=args.scene_manifest, preloading=args.preloading)
    rows = runner(args.partition, {"kind": "never"})
    write_rows(args.output, rows)
    from utils.v27_eval_reporting import write_endpoint_diagnostics
    write_endpoint_diagnostics(str(args.output) + '.summary.json', rows)
    manifest = {"schema": ROWS_SCHEMA, "partition": args.partition,
        "checkpoint_sha256": runner.checkpoint_sha256, "config_sha256": runner.config_sha256,
        "score_definition": SCORE_DEFINITION, "metric_mode": "benchmark_compat",
        "scene_manifest": runner.scene_manifest,
        "parameter_training_overlap": runner.scene_manifest["parameter_training_overlap"],
        "rows": len(rows), "rows_sha256": sha256_file(args.output),
        "tracklet_keys_sha256": sha256_json(sorted({r["tracklet_id"] for r in rows}))}
    _write_json(str(args.output) + ".manifest.json", manifest)


def calibrate_main(argv=None):
    args = _parser("Fit v27 policy using real calibration rollouts; diagnose locked dev policy").parse_args(argv)
    cache_directory = Path(args.output).parent / (Path(args.output).stem + "_rollouts")
    runner = TrackerClosedLoopRunner(args.config, args.checkpoint, path=args.path,
        device=args.device, scene_manifest_path=args.scene_manifest,
        preloading=args.preloading, cache_directory=cache_directory)
    calibration_rows = runner("calibration", {"kind": "never"})
    artifact = calibrate_actions_v27(calibration_rows, runner,
        checkpoint_sha256=runner.checkpoint_sha256, config_sha256=runner.config_sha256,
        scene_manifest=runner.scene_manifest)
    _write_json(args.output, artifact)
    print(json.dumps({"action_policy": artifact["action_policy"],
                      "dev_locked_metrics": artifact["dev_locked_metrics"],
                      "parameter_training_overlap": artifact["parameter_training_overlap"]}, sort_keys=True))
