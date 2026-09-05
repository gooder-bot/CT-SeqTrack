"""场景级隔离和原始测量身份的边界测试；不需要 nuScenes 数据。"""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from pyquaternion import Quaternion

from utils.point_identity import raw_point_ids, sampled_identity
from utils.v27_protocol import build_scene_manifest, select_scene_protocol


def _splits(mini=False):
    if mini:
        return {"mini_train": [f"scene-{i:04d}" for i in range(8)],
                "mini_val": ["scene-1000", "scene-1001"]}
    return {"train_track": [f"scene-{i:04d}" for i in range(350)],
            "val": [f"scene-{i:04d}" for i in range(500, 650)]}


def test_full_all_350_train_with_disjoint_17_18_internal_subsets():
    source = _splits()
    manifest = build_scene_manifest(source, "v1.0-trainval")
    roles = {role: set(scenes) for role, scenes in manifest["scenes"].items()}
    assert {role: len(scenes) for role, scenes in roles.items()} == {
        "train": 350, "calibration": 17, "dev": 18, "test": 150}
    assert roles["train"] == set(source["train_track"])
    assert roles["calibration"] | roles["dev"] <= roles["train"]
    assert not roles["calibration"] & roles["dev"]
    assert not roles["test"] & roles["train"]
    assert manifest["parameter_training_overlap"] is True


def test_mini_six_one_one_are_disjoint_and_official_two_are_untouched():
    source = _splits(mini=True)
    manifest = build_scene_manifest(source, "v1.0-mini")
    roles = {role: set(scenes) for role, scenes in manifest["scenes"].items()}
    assert tuple(len(roles[role]) for role in ("train", "calibration", "dev", "test")) == (6, 1, 1, 2)
    assert roles["train"] | roles["calibration"] | roles["dev"] == set(source["mini_train"])
    assert len(set.union(*roles.values())) == sum(map(len, roles.values()))
    assert roles["test"] == set(source["mini_val"])
    assert manifest["parameter_training_overlap"] is False


def test_scene_manifest_is_input_order_invariant_and_hashes_its_role_declaration():
    source = _splits()
    original = build_scene_manifest(source, "v1.0-trainval")
    reversed_source = {key: list(reversed(value)) for key, value in source.items()}
    assert original == build_scene_manifest(reversed_source, "v1.0-trainval")
    assert original["scenes"] != build_scene_manifest(source, "v1.0-trainval", seed=43)["scenes"]
    body = dict(original)
    sha = body.pop("content_sha256")
    assert sha == hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = types.SimpleNamespace(version="v1.0-trainval", ct_protocol_role="calibration")
    manifest, role, chosen = select_scene_protocol(config, "test", source)
    assert role == "calibration" and chosen == manifest["scenes"]["calibration"]


@pytest.mark.parametrize("bad_case", ("missing_train", "wrong_eval_count", "eval_overlap"))
def test_scene_protocol_rejects_incomplete_or_leaking_sources(bad_case):
    source = _splits(mini=True)
    if bad_case == "missing_train":
        source["mini_train"].pop()
    elif bad_case == "wrong_eval_count":
        source["mini_val"].pop()
    else:
        source["mini_val"][0] = source["mini_train"][0]
    with pytest.raises(ValueError):
        build_scene_manifest(source, "v1.0-mini")


@pytest.fixture
def point_runtime(monkeypatch):
    """只替换包入口；被测 PointCloud 和采样/裁剪函数使用真实源码。"""
    prior = {key: value for key, value in sys.modules.items()
             if key == "datasets" or key.startswith("datasets.")}
    package = types.ModuleType("datasets")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "datasets")]
    monkeypatch.setitem(sys.modules, "datasets", package)
    nu = types.ModuleType("nuscenes")
    nu.__path__ = []
    utilities = types.ModuleType("nuscenes.utils")
    utilities.__path__ = []
    geometry = types.ModuleType("nuscenes.utils.geometry_utils")
    nu.utils = utilities
    utilities.geometry_utils = geometry
    for key, value in (("nuscenes", nu), ("nuscenes.utils", utilities),
                       ("nuscenes.utils.geometry_utils", geometry)):
        monkeypatch.setitem(sys.modules, key, value)
    classes = importlib.import_module("datasets.data_classes")
    points = importlib.import_module("datasets.points_utils")
    yield classes, points
    for key in list(sys.modules):
        if (key == "datasets" or key.startswith("datasets.")) and key not in prior:
            sys.modules.pop(key, None)
    sys.modules.update(prior)


def test_ids_survive_se2_and_both_crops_without_coordinate_based_reidentification(point_runtime):
    classes, points = point_runtime
    pc = classes.PointCloud(np.asarray([[0., 0., 1.8, 3.2], [0., 0., .1, .1], [0., 0., 0., 0.]]),
                            point_ids=np.asarray([101, 202, 303, 404]))
    box = classes.Box([0., 0., 0.], [2., 4., 2.], Quaternion())
    rotation = Quaternion(axis=[0, 0, 1], radians=.3)
    translation = np.asarray([8., -4., 1.])
    moved_pc, moved_box = copy.deepcopy(pc), copy.deepcopy(box)
    moved_pc.rotate(rotation.rotation_matrix)
    moved_pc.translate(translation)
    moved_box.rotate(rotation)
    moved_box.translate(translation)
    np.testing.assert_array_equal(raw_point_ids(moved_pc), [101, 202, 303, 404])
    crop = points.crop_pc_oriented(moved_pc, moved_box)
    np.testing.assert_array_equal(raw_point_ids(crop), [101, 202, 303])
    centered, _ = points.cropAndCenterPC(moved_pc, moved_box)
    np.testing.assert_array_equal(raw_point_ids(centered), [101, 202, 303])
    np.testing.assert_allclose(centered.points, pc.points[:, :3], atol=1e-10)
    assert centered.point_ids[0] != centered.point_ids[1]  # coincident returns remain distinct measurements


@pytest.mark.parametrize("count", (0, 1, 2))
def test_sparse_sampling_preserves_real_evidence_and_marks_only_empty_padding(point_runtime, count):
    classes, points = point_runtime
    xyz = np.asarray([[2., 3., 4.], [5., 6., 7.]], dtype=np.float32)[:count]
    pc = classes.PointCloud(xyz.T, point_ids=np.arange(count, dtype=np.int64) + 41)
    sampled, indices = points.regularize_pc(xyz, 16, seed=42)
    ids, valid, unique = sampled_identity(pc, indices, 16)
    assert sampled.shape == (16, 3) and ids.shape == (16,)
    assert int(unique.sum()) == count
    if count == 0:
        assert np.all(ids == -1) and not valid.any() and not sampled.any()
    else:
        assert valid.all() and np.isin(ids, pc.point_ids).all()
        np.testing.assert_array_equal(sampled, xyz[indices])
        assert len(np.unique(ids)) == count
