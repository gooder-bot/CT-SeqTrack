import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

from models.ct_v2.point_feature_consistency import (
    PointFeatureTemporalConsistencyLoss,
    canonicalize_points,
    chronological_frame_indices,
)
from utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]


def _observed_from_canonical(canonical, center, yaw):
    """Inverse of the project's column-vector canonical yaw transform."""
    cosine = torch.cos(torch.as_tensor(yaw))
    sine = torch.sin(torch.as_tensor(yaw))
    x_coord = cosine * canonical[..., 0] + sine * canonical[..., 1]
    y_coord = -sine * canonical[..., 0] + cosine * canonical[..., 1]
    return torch.stack((x_coord, y_coord, canonical[..., 2]), dim=-1) + center


def _loss_module(**kwargs):
    defaults = {
        "distance_threshold": 0.3,
        "min_correspondences": 1,
        "time_weighting": True,
    }
    defaults.update(kwargs)
    return PointFeatureTemporalConsistencyLoss(**defaults)


def test_canonical_coordinates_are_translation_and_yaw_invariant():
    canonical = torch.tensor([
        [0.1, 0.2, -0.1],
        [1.0, -0.4, 0.3],
        [-0.7, 0.5, 0.0],
    ])
    center = torch.tensor([3.0, -2.0, 0.7])
    yaw = torch.tensor(0.8)
    observed = _observed_from_canonical(canonical, center, yaw)
    recovered = canonicalize_points(
        observed.unsqueeze(0),
        torch.cat((center, yaw.view(1))).unsqueeze(0))
    torch.testing.assert_close(recovered.squeeze(0), canonical)

    yaw_degrees = torch.tensor(45.0)
    observed_degrees = _observed_from_canonical(
        canonical, center, torch.deg2rad(yaw_degrees))
    recovered_degrees = canonicalize_points(
        observed_degrees.unsqueeze(0),
        torch.cat((center, yaw_degrees.view(1))).unsqueeze(0),
        degrees=True)
    torch.testing.assert_close(recovered_degrees.squeeze(0), canonical)


def test_history_reorder_is_positional_and_invalid_history_is_skipped():
    assert chronological_frame_indices(4).tolist() == [2, 1, 0, 3]
    features = torch.zeros(1, 4, 3, 1)
    points = torch.zeros(1, 4, 3, 3)
    seg = torch.ones(1, 4, 3)
    boxes = torch.zeros(1, 4, 4)
    valid = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    timestamps = torch.tensor([[-0.5, -1.0, -1.5, 0.0]])
    result = _loss_module()(features, points, seg, boxes, valid, timestamps)
    # old, near, current are valid: exactly three early->late pairs.
    assert result["pftc_valid_pair_count"].item() == 3


def test_background_point_cannot_be_selected_as_correspondence():
    features = torch.tensor([[[[0.0], [0.0]], [[10.0], [0.0]]]])
    points = torch.tensor([[
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]])
    seg = torch.tensor([[[1, 0], [1, 0]]])
    boxes = torch.zeros(1, 2, 4)
    valid = torch.ones(1, 1, dtype=torch.bool)
    timestamps = torch.tensor([[-0.5, 0.0]])
    result = _loss_module()(features, points, seg, boxes, valid, timestamps)
    assert result["pftc_match_count"].item() == 1
    # SmoothL1(0, 10) = 9.5; the spatially closer background feature is 0.
    torch.testing.assert_close(
        result["loss_pftc_raw"], torch.tensor(9.5))


def test_exact_duplicate_samples_do_not_change_loss_or_match_weight():
    points = torch.tensor([[
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
    ]])
    features = torch.tensor([[
        [[0.0], [1.0], [2.0]],
        [[1.0], [3.0], [5.0]],
    ]])
    seg = torch.ones(1, 2, 3)
    boxes = torch.zeros(1, 2, 4)
    valid = torch.ones(1, 1, dtype=torch.bool)
    timestamps = torch.tensor([[-0.5, 0.0]])
    module = _loss_module()
    unique = module(features, points, seg, boxes, valid, timestamps)

    duplicate_indices = torch.tensor([0, 0, 1, 1, 2, 2])
    duplicated = module(
        features.index_select(2, duplicate_indices),
        points.index_select(2, duplicate_indices),
        seg.index_select(2, duplicate_indices),
        boxes,
        valid,
        timestamps)
    torch.testing.assert_close(
        unique["loss_pftc_raw"], duplicated["loss_pftc_raw"])
    torch.testing.assert_close(
        unique["pftc_match_count"], duplicated["pftc_match_count"])
    assert duplicated["pftc_fg_points_before"].item() == 6
    assert duplicated["pftc_fg_points_after"].item() == 3


@pytest.mark.parametrize("case", ("empty", "threshold", "minimum"))
def test_empty_and_insufficient_correspondences_return_finite_zero(case):
    points = torch.zeros(1, 2, 3, 3)
    features = torch.randn(1, 2, 3, 2, requires_grad=True)
    seg = torch.ones(1, 2, 3)
    kwargs = {}
    if case == "empty":
        seg.zero_()
    elif case == "threshold":
        points[:, 1, :, 0] = 2.0
    else:
        kwargs["min_correspondences"] = 4
    result = _loss_module(**kwargs)(
        features, points, seg, torch.zeros(1, 2, 4),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[-0.5, 0.0]]))
    assert torch.isfinite(result["loss"])
    assert result["loss"].item() == 0.0


def test_time_values_change_only_weighting_not_correspondence_topology():
    # Storage order is [near, old, current]; fixed topology becomes
    # [old, near, current] independently of timestamp magnitudes.
    features = torch.tensor([[[[1.0]], [[0.0]], [[3.0]]]])
    points = torch.zeros(1, 3, 1, 3)
    seg = torch.ones(1, 3, 1)
    boxes = torch.zeros(1, 3, 4)
    valid = torch.ones(1, 2, dtype=torch.bool)
    true_time = torch.tensor([[-0.5, -2.0, 0.0]])
    fixed_time = torch.tensor([[-0.5, -1.0, 0.0]])
    shuffled_time = torch.tensor([[-1.8, -0.3, 0.0]])
    module = _loss_module()
    outputs = [
        module(features, points, seg, boxes, valid, timestamps)
        for timestamps in (true_time, fixed_time, shuffled_time)
    ]
    for output in outputs[1:]:
        torch.testing.assert_close(
            output["loss_pftc_raw"], outputs[0]["loss_pftc_raw"])
        torch.testing.assert_close(
            output["pftc_valid_pair_count"],
            outputs[0]["pftc_valid_pair_count"])
        torch.testing.assert_close(
            output["pftc_match_count"], outputs[0]["pftc_match_count"])
        torch.testing.assert_close(
            output["pftc_match_distance"],
            outputs[0]["pftc_match_distance"])
    assert not torch.isclose(
        outputs[0]["loss_pftc_weighted"],
        outputs[1]["loss_pftc_weighted"])
    unweighted = _loss_module(time_weighting=False)(
        features, points, seg, boxes, valid, true_time)
    torch.testing.assert_close(
        unweighted["loss_pftc_weighted"],
        unweighted["loss_pftc_raw"])
    assert unweighted["pftc_time_weight_mean"].item() == 1.0
    assert unweighted["pftc_time_weight_max"].item() == 1.0


def test_gradients_flow_only_to_point_features_not_matching_inputs():
    features = torch.tensor(
        [[[[0.0]], [[2.0]]]], requires_grad=True)
    points = torch.zeros(1, 2, 1, 3, requires_grad=True)
    boxes = torch.zeros(1, 2, 4, requires_grad=True)
    result = _loss_module()(
        features, points, torch.ones(1, 2, 1), boxes,
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[-0.5, 0.0]]))
    result["loss"].backward()
    assert features.grad is not None
    assert features.grad.abs().sum().item() > 0
    assert points.grad is None
    assert boxes.grad is None


def test_cdist_runs_only_on_foreground_subsets(monkeypatch):
    batch_size, num_frames, num_points = 16, 4, 1024
    points = torch.zeros(batch_size, num_frames, num_points, 3)
    points[:, :, :8, 0] = torch.arange(8).float() * 0.02
    features = torch.randn(batch_size, num_frames, num_points, 4)
    seg = torch.zeros(batch_size, num_frames, num_points)
    seg[:, :, :8] = 1
    boxes = torch.zeros(batch_size, num_frames, 4)
    valid = torch.ones(
        batch_size, num_frames - 1, dtype=torch.bool)
    timestamps = torch.tensor(
        [[-0.5, -1.0, -1.5, 0.0]]).repeat(batch_size, 1)

    original_cdist = torch.cdist
    observed_shapes = []

    def recording_cdist(left, right, *args, **kwargs):
        observed_shapes.append((left.shape[-2], right.shape[-2]))
        return original_cdist(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", recording_cdist)
    result = _loss_module(min_correspondences=3)(
        features, points, seg, boxes, valid, timestamps)
    assert torch.isfinite(result["loss"])
    assert observed_shapes
    assert max(max(shape) for shape in observed_shapes) == 8


def _load_feature_pointnet_without_cuda_extension():
    """Load the pure FeaturePointNet class while stubbing unused PointNet++."""
    names = (
        "pointnet2",
        "pointnet2.utils",
        "pointnet2.utils.pointnet2_modules",
    )
    previous = {name: sys.modules.get(name) for name in names}
    pointnet2_module = types.ModuleType("pointnet2")
    utils_module = types.ModuleType("pointnet2.utils")
    modules_module = types.ModuleType("pointnet2.utils.pointnet2_modules")
    modules_module.PointnetSAModule = object
    sys.modules["pointnet2"] = pointnet2_module
    sys.modules["pointnet2.utils"] = utils_module
    sys.modules["pointnet2.utils.pointnet2_modules"] = modules_module
    try:
        spec = importlib.util.spec_from_file_location(
            "_feature_pointnet_pftc_test",
            ROOT / "models/backbone/pointnet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.FeaturePointNet
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_feature_pointnet_optional_output_is_point_aligned_and_inert():
    feature_pointnet = _load_feature_pointnet_without_cuda_extension()
    torch.manual_seed(42)
    model = feature_pointnet(
        input_channel=5,
        per_point_mlp1=[64, 64, 64, 128, 1024],
        per_point_mlp2=[512, 256, 128, 128],
        output_size=128)
    model.eval()
    inputs = torch.randn(2, 5, 32)
    with torch.no_grad():
        default_output = model(inputs)
        explicit_off = model(inputs, return_point_features=False)
        enabled_output, point_features = model(
            inputs, return_point_features=True)
    assert torch.equal(default_output, explicit_off)
    assert torch.equal(default_output, enabled_output)
    assert point_features.shape == (2, 64, 32)

    model.train()
    model.zero_grad(set_to_none=True)
    _, point_features = model(
        inputs, return_point_features=True)
    point_features.square().mean().backward()
    assert model.seq_per_point[0][0].weight.grad is not None
    assert model.seq_per_point[1][0].weight.grad is not None
    assert model.seq_per_point[2][0].weight.grad is None


def test_b4_configs_are_isolated_from_point_feature_path():
    base = load_yaml_config(
        ROOT / "cfgs/ct_v2/01_seqtrack3d_baseline.yaml")
    alignment = load_yaml_config(
        ROOT / "cfgs/ct_v2/19_b4_decoder_alignment.yaml")
    anticollapse = load_yaml_config(
        ROOT / "cfgs/ct_v2/20_b4_decoder_anticollapse.yaml")
    assert base["use_point_feature_tc"] is False
    assert base["pftc_distance_threshold"] == 0.3
    assert base["pftc_min_correspondences"] == 3
    assert alignment["use_b4_paired_views"] is True
    assert alignment["use_decoder_token_consistency"] is True
    assert alignment["use_point_feature_tc"] is False
    assert anticollapse["decoder_tc_variance_weight"] == 1.0
    assert anticollapse["decoder_tc_covariance_weight"] == 0.04


def test_pftc_loss_has_no_state_and_consumes_no_initialization_rng():
    torch.manual_seed(42)
    rng_before = torch.random.get_rng_state().clone()
    module = PointFeatureTemporalConsistencyLoss()
    rng_after = torch.random.get_rng_state()
    assert torch.equal(rng_before, rng_after)
    assert not module.state_dict()
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
