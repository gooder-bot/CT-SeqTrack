"""v27 场景用途：完整 train_track 全量训练，官方 val 仅评测。"""
from __future__ import annotations

import hashlib
import json


SCHEMA = "ct_seqtrack.scene_protocol.v27"


def build_scene_manifest(scene_splits, version, seed=42):
    if str(version) not in ('v1.0-mini', 'v1.0-trainval'):
        raise ValueError('v27 supports only v1.0-mini or v1.0-trainval')
    mini = str(version) == "v1.0-mini"
    source = "mini_train" if mini else "train_track"
    evaluation = "mini_val" if mini else "val"
    scenes = sorted(set(scene_splits[source]), key=lambda name: hashlib.sha256(
        f"ct27|{int(seed)}|{name}".encode("utf-8")).hexdigest())
    expected = 8 if mini else 350
    if len(scenes) != expected:
        raise ValueError(f"v27 {source} requires {expected} scenes, got {len(scenes)}")
    test = sorted(set(scene_splits[evaluation]))
    if len(test) != (2 if mini else 150) or set(test) & set(scenes):
        raise ValueError("v27 official evaluation split count/overlap mismatch")
    roles = {
        "train": scenes[:6] if mini else scenes,
        "calibration": scenes[6:7] if mini else scenes[315:332],
        "dev": scenes[7:] if mini else scenes[332:],
        "test": test,
    }
    result = dict(schema=SCHEMA, version=str(version), split_seed=int(seed),
                  training_source=source, evaluation_source=evaluation,
                  parameter_training_overlap=not mini, scenes=roles)
    result["content_sha256"] = hashlib.sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def scene_role(config, requested_role):
    role = str(getattr(config, "ct_protocol_role", "") or requested_role)
    if role in ("val", "eval"):
        role = "test"
    if role not in ("train", "calibration", "dev", "test"):
        raise ValueError(f"Unknown v27 scene role {role!r}")
    return role


def select_scene_protocol(config, requested_role, scene_splits):
    manifest = build_scene_manifest(scene_splits, config.version,
                                    getattr(config, "ct_partition_seed", 42))
    role = scene_role(config, requested_role)
    return manifest, role, manifest["scenes"][role]
