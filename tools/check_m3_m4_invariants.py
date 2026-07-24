#!/usr/bin/env python3
"""Dataset-free invariants for the first M3/M4 engineering slice."""

import json
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the two implementation files directly. Importing ``models.*`` would
# execute models/__init__.py and pull in the full training stack, defeating
# this script's dataset-free/dependency-light contract.
PATH_DISTILLATION = load_module(
    "ct_seqtrack_path_distillation", ROOT / "models" / "path_distillation.py")
STATE_FILTER = load_module(
    "ct_seqtrack_state_filter", ROOT / "models" / "state_filter.py")

endpoint_path_terms = PATH_DISTILLATION.endpoint_path_terms
freeze_batchnorm_running_stats = PATH_DISTILLATION.freeze_batchnorm_running_stats
teacher_endpoint_confidence = PATH_DISTILLATION.teacher_endpoint_confidence
teacher_endpoint_confidence_terms = (
    PATH_DISTILLATION.teacher_endpoint_confidence_terms)
update_ema_module = PATH_DISTILLATION.update_ema_module

FixedContinuousDiscreteFilter = STATE_FILTER.FixedContinuousDiscreteFilter
build_trajectory_tube_box = STATE_FILTER.build_trajectory_tube_box
point_inside_oriented_crop = STATE_FILTER.point_inside_oriented_crop
union_point_clouds = STATE_FILTER.union_point_clouds
wrap_angle = STATE_FILTER.wrap_angle

DATA_CLASSES = load_module(
    "ct_seqtrack_data_classes", ROOT / "datasets" / "data_classes.py")
Box = DATA_CLASSES.Box
PointCloud = DATA_CLASSES.PointCloud


def check_m3_gradient_direction():
    teacher_boxes = torch.tensor(
        [[0.0, 0.0, 0.0, math.pi - 0.05],
         [1.0, 2.0, 3.0, -math.pi + 0.05]],
        requires_grad=True,
    )
    student_boxes = torch.tensor(
        [[0.4, 0.0, 0.0, -math.pi + 0.05],
         [0.5, 2.2, 3.0, math.pi - 0.05]],
        requires_grad=True,
    )
    terms = endpoint_path_terms(teacher_boxes, student_boxes)
    terms["total"].sum().backward()
    assert teacher_boxes.grad is None
    assert student_boxes.grad is not None
    assert torch.isfinite(student_boxes.grad).all()
    assert float(student_boxes.grad.abs().sum()) > 0
    assert float(terms["yaw_gap"].max()) < 0.11


def check_m3_confidence_and_ema():
    logits = torch.zeros(2, 2, 12)
    logits[:, 1, -4:] = 3.0
    output = {
        "aux_estimation_boxes": torch.zeros(2, 4),
        "estimation_boxes": torch.zeros(2, 4),
        "seg_logits": logits,
    }
    confidence = teacher_endpoint_confidence(
        output, point_sample_size=4, mode="foreground_topk", topk=2)
    assert confidence.shape == (2,)
    assert bool(torch.all((confidence > 0.9) & (confidence <= 1.0)))

    hybrid_clean = teacher_endpoint_confidence_terms(
        output,
        point_sample_size=4,
        mode="hybrid",
        topk=2,
        agreement_center_scale=1.0,
        agreement_yaw_scale=0.5,
    )
    inconsistent = dict(output)
    inconsistent["estimation_boxes"] = torch.tensor(
        [[3.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]])
    hybrid_inconsistent = teacher_endpoint_confidence_terms(
        inconsistent,
        point_sample_size=4,
        mode="hybrid",
        topk=2,
        agreement_center_scale=1.0,
        agreement_yaw_scale=0.5,
    )
    assert torch.all(
        hybrid_clean["confidence"] > hybrid_inconsistent["confidence"])
    assert torch.all(hybrid_clean["agreement"] == 1.0)
    assert torch.all(hybrid_inconsistent["agreement"] < 0.1)

    student = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    teacher = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(1.0)
        for parameter in teacher.parameters():
            parameter.zero_()
        student[1].running_mean.fill_(2.0)
        teacher[1].running_mean.zero_()
    update_ema_module(teacher, student, momentum=0.75)
    for parameter in teacher.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.25))
    assert torch.allclose(
        teacher[1].running_mean,
        torch.full_like(teacher[1].running_mean, 0.5),
    )

    protected_bn = nn.BatchNorm1d(2)
    protected_bn.train()
    before = protected_bn.running_mean.clone()
    bn_input = torch.full((4, 2), 100.0, requires_grad=True)
    with freeze_batchnorm_running_stats(protected_bn):
        protected_bn(bn_input).sum().backward()
    assert torch.equal(protected_bn.running_mean, before)
    assert protected_bn.training
    assert protected_bn.weight.grad is not None


def check_m4_filter_and_tube():
    state_filter = FixedContinuousDiscreteFilter(
        acceleration_variance=0.5,
        yaw_acceleration_variance=0.2,
        mahalanobis_gate=0.0,
    )
    state_filter.initialize(
        [0.0, 0.0, 0.0], math.pi - 0.02, 0.0,
        velocity=[2.0, 0.0, 0.0], yaw_rate=0.1,
    )
    prediction = state_filter.predict(0.5)
    assert prediction["valid"]
    assert np.allclose(prediction["mean"][0], 1.0, atol=1e-8)
    assert np.linalg.eigvalsh(prediction["covariance"]).min() > 0
    update = state_filter.update([1.1, 0.0, 0.0], -math.pi + 0.03)
    assert update["accepted"]
    assert abs(wrap_angle(state_filter.mean[6])) <= math.pi
    duplicate = state_filter.predict(0.5)
    assert not duplicate["valid"]

    template = Box(
        center=[0.0, 0.0, 0.0],
        size=[2.0, 4.0, 1.5],
        orientation=Quaternion(axis=[0, 0, 1], radians=0.0),
    )
    tube = build_trajectory_tube_box(
        template,
        prediction,
        base_length=4.0,
        base_width=2.0,
        max_length=8.0,
        max_width=5.0,
    )
    assert tube is not None
    assert 4.0 <= tube.wlh[1] <= 8.0
    assert 2.0 <= tube.wlh[0] <= 5.0
    assert np.all(np.isfinite(tube.center))
    assert point_inside_oriented_crop(template, [0.5, 0.0, 0.0])
    assert not point_inside_oriented_crop(template, [10.0, 0.0, 0.0])

    direct_filter = FixedContinuousDiscreteFilter()
    direct_filter.initialize([0.0, 0.0, 0.0], 0.0, 0.0)
    direct_filter.predict(1.0)
    direct_filter.observe_direct(
        [1.0, 0.0, 0.0], 0.1, 1.0, velocity_momentum=0.0)
    assert np.allclose(direct_filter.mean[3:6], [1.0, 0.0, 0.0])
    assert np.isclose(direct_filter.mean[7], 0.1)

    primary = PointCloud(np.asarray([
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]))
    secondary = PointCloud(np.asarray([
        [1.0, 2.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]))
    merged = union_point_clouds(primary, secondary)
    assert merged.nbr_points() == 3


def main():
    check_m3_gradient_direction()
    check_m3_confidence_and_ema()
    check_m4_filter_and_tube()
    print(json.dumps({
        "status": "PASS_M3_M4_DATASET_FREE_INVARIANTS",
        "m3": [
            "teacher_detached",
            "periodic_yaw",
            "gt_free_confidence",
            "ema_update",
            "irregular_bn_protected",
        ],
        "m4": [
            "positive_covariance",
            "invalid_dt_fallback",
            "bounded_tube",
            "tube_only_velocity_update",
            "point_union",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
