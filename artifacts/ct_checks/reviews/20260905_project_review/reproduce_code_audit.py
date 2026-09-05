"""只读加载源码的 CPU 最小复现，不导入 Lightning/nuScenes/PointNet CUDA。"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from utils.box_membership import axis_aligned_box_membership_mask
from utils.ct_search import bounded_novel_support_pool
spec = importlib.util.spec_from_file_location(
    "audit_evidence_memory", ROOT / "models/ct_v2/evidence_memory.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def extracted_method(name):
    tree = ast.parse((ROOT / "models/seqtrack3d.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SEQTRACK3D")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "models/seqtrack3d.py"), "exec"), ns)
    return ns[name]


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    def forward(self, boxes, dt, valid, query):
        batch = len(boxes)
        prediction = {name: torch.ones((batch, 2)) for name in (
            "mu_xy", "kinematic_prior_xy", "log_sigma_parallel_perp",
            "basis_velocity_xy", "direction_xy", "velocity_xy",
            "acquisition_margin_parallel_perp")}
        prediction.update(covariance_xy=torch.eye(2).expand(batch, 2, 2),
                          feature=torch.ones(batch, 4), valid=torch.ones(batch),
                          gap_ratio=torch.ones(batch), source_id=torch.ones(batch, dtype=torch.long))
        return prediction


torch.set_num_threads(1)
torch.manual_seed(42)
host = SimpleNamespace(use_b1motion_v3=True, config=SimpleNamespace(), physical_motion_encoder=FakeEncoder())
prediction = extracted_method("predict_motion_from_history")(
    host, torch.zeros(1, 3, 4), torch.ones(1, 3), torch.ones(1, 3), torch.ones(1))
unbatched = extracted_method("_unbatch_motion_prepass_predictions")(prediction, [1.0])[0]
prepass = {"encoder_valid": True, "public_margin_present": "acquisition_margin_parallel_perp" in prediction,
           "unbatched_valid": unbatched["valid"], "unbatched_source": unbatched["source_id"]}
assert not prepass["public_margin_present"] and not prepass["unbatched_valid"]

module = evidence.B2EvidenceAcquirer(relation_aware_sampling=True, robust_consensus_voting=True)
points = torch.zeros(1, 300, 5)
points[0, :, 2] = torch.arange(300) * 0.001  # distinct XYZ, identical XY
indices, valid, group = module._hybrid_select(points, torch.ones(1, 300, dtype=torch.bool),
    torch.arange(300, dtype=torch.float32).reshape(1, -1), torch.ones(1, 300, dtype=torch.long))
selected = indices[valid]
fps = {"selected_slots": len(selected), "unique_indices": len(torch.unique(selected)),
       "fps_first_eight": evidence._fps_indices(points[0, :, :2], 8).tolist()}
assert fps["unique_indices"] < fps["selected_slots"]

features = torch.zeros(1, 1, 2, 64)
features[0, 0, 0, 0] = 101  # front point: within length/2=2
features[0, 0, 1, 0] = 202  # side point: outside width/2=1
memory_points = torch.tensor([[[[1.8, 0.0, 0.0], [0.0, 1.8, 0.0]]]])
tokens, mask, metadata = evidence.build_box_memory_tokens(features, memory_points,
    torch.zeros(1, 1, 4), torch.tensor([[2.0, 4.0, 2.0]]), torch.ones(1, 1),
    foreground_tokens=1, context_tokens=1, return_metadata=True)
memory = {"expected_inside_feature": 101, "actual_inside_feature": int(tokens[0, 0, 0]),
          "expected_context_feature": 202, "actual_context_feature": int(tokens[0, 1, 0])}
assert memory["actual_inside_feature"] == 202 and memory["actual_context_feature"] == 101

def same_size_iou(center, size):
    overlap = torch.clamp(size - center.abs(), min=0)
    intersection = overlap.prod()
    return (intersection / (2 * size.prod() - intersection)).item()

observation = torch.tensor([1.5, 0.0])
action = torch.tensor([1.0, 0.5])  # displacement 0.707m: within normal 0.75m B3 radius
wrong_size = torch.tensor([2.0, 4.0])
right_size = wrong_size.flip(0)
iou_proxy = {"observation_xy": observation.tolist(), "action_xy": action.tolist(),
    "proxy_gain_wlh_as_xy": same_size_iou(action, wrong_size) - same_size_iou(observation, wrong_size),
    "correct_axis_aligned_gain": same_size_iou(action, right_size) - same_size_iou(observation, right_size)}
assert iou_proxy["proxy_gain_wlh_as_xy"] > 0 and iou_proxy["correct_axis_aligned_gain"] < 0

votes = torch.tensor([[[4.0, 0.0], [4.1, 0.0], [9.0, 0.0]]], requires_grad=True)
weights = torch.tensor([[0.8, 0.7, 0.1]], requires_grad=True)
consensus = evidence.B2EvidenceAcquirer._consensus_vote(votes, weights, torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 2))
consensus["center"].square().sum().backward()
gradient = {"votes_finite_nonzero": bool(torch.isfinite(votes.grad).all() and (votes.grad.abs().sum() > 0)),
            "weights_finite_nonzero": bool(torch.isfinite(weights.grad).all() and (weights.grad.abs().sum() > 0))}
assert all(gradient.values())

# Load real PointCloud and crop definitions without importing datasets/__init__.
data_spec = importlib.util.spec_from_file_location("audit_data_classes", ROOT / "datasets/data_classes.py")
data_classes = importlib.util.module_from_spec(data_spec)
data_spec.loader.exec_module(data_classes)
crop_tree = ast.parse((ROOT / "datasets/points_utils.py").read_text(encoding="utf-8"))
crop_nodes = [n for n in crop_tree.body if isinstance(n, ast.FunctionDef) and n.name in (
    "crop_pc_axis_aligned", "generate_subwindow_with_aroundboxs")]
crop_ns = {"np": np, "copy": copy, "Quaternion": Quaternion,
           "axis_aligned_box_membership_mask": axis_aligned_box_membership_mask}
exec(compile(ast.Module(body=crop_nodes, type_ignores=[]), str(ROOT / "datasets/points_utils.py"), "exec"), crop_ns)
anchor = data_classes.Box([0.1, 0.2, 0.0], [20.0, 20.0, 12.0], Quaternion(axis=[0, 0, 1], radians=0.47))
endpoint = data_classes.Box([0.3, 0.3, 0.0], [20.0, 20.0, 12.0], Quaternion(axis=[0, 0, 1], radians=1.1))
world_xyz = np.random.default_rng(42).uniform(-4.0, 4.0, (3, 100)).astype(np.float32)
world_xyz[0] += 0.1
world_xyz[1] += 0.2
cloud = data_classes.PointCloud(world_xyz)
crop = crop_ns["generate_subwindow_with_aroundboxs"]
base = crop(cloud, anchor, anchor, scale=1.0, offset=0.0).points.T
extension = crop(cloud, endpoint, anchor, scale=1.0, offset=0.0).points.T
assert len(base) == len(extension) == 100  # both arrays contain all the same physical returns
novel, sources = bounded_novel_support_pool(base, extension, extension[:0], extension[:0])
online_identity = {"same_original_points": 100, "base_points": len(base), "endpoint_points": len(extension),
    "expected_novel_count": 0, "actual_novel_count": len(novel),
    "max_local_coordinate_difference_m": float(np.max(np.abs(base - extension)))}
assert len(novel) > 0

report = {"prepass_contract": prepass, "hybrid_fps_duplicate": fps, "memory_wlh_axis_swap": memory,
          "b3_iou_label_sign_flip": iou_proxy, "robust_vote_gradient_not_disconnected": gradient,
          "online_extension_false_novel_points": online_identity}
print(json.dumps(report, ensure_ascii=False, indent=2))
(Path(__file__).parent / "reproduce_code_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
