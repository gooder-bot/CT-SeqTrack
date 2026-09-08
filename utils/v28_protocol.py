"""v28：完整训练划分，内部拟合/诊断与参数训练重叠，官方验证只评测。"""
import hashlib
import json

from utils.v27_protocol import build_scene_manifest as build_v27_scene_manifest
from utils.v27_protocol import scene_role


SCHEMA = "ct_seqtrack.scene_protocol.v28"


def build_scene_manifest(scene_splits, version, seed=42):
    # 保留 v27 已固定的内部 calibration/dev 选择，不重新挑选容易场景。
    manifest = build_v27_scene_manifest(scene_splits, version, seed)
    manifest["schema"] = SCHEMA
    manifest["scenes"]["train"] = sorted(set(scene_splits[manifest["training_source"]]))
    manifest["parameter_training_overlap"] = True
    manifest.pop("content_sha256")
    manifest["content_sha256"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return manifest


def select_scene_protocol(config, requested_role, scene_splits):
    manifest = build_scene_manifest(scene_splits, config.version,
                                    getattr(config, "ct_partition_seed", 42))
    role = scene_role(config, requested_role)
    return manifest, role, manifest["scenes"][role]
