"""仅供 v37 工程 labels 路径：共享框对几何，不混合两种指标定义。

所有真值只生成监督；缓存局限本次 loss，不进入 forward 或递归状态。
"""
import numpy as np
import torch

from models.ct_v31.losses import quality_targets
from utils.tracking_metrics import LocalYawBox, _intersection, box_metrics, metric_contributions


def paired_box_metrics(box, target):
    """一次 XY 相交产生 z-up 3D exact/compat IoU 和共同中心距离。

逐项保留 tracking_metrics.box_metrics 的 float64 运算及高度公式；
此函数不承接 y-up、二维距离或其它模式。
    """
    centers = [np.asarray(item.center, dtype=np.float64) for item in (box, target)]
    sizes = [np.asarray(item.wlh, dtype=np.float64) for item in (box, target)]
    if any(not np.isfinite(value).all() for value in centers + sizes):
        raise ValueError('box geometry must be finite')
    if any((value <= 0).any() for value in sizes):
        raise ValueError('box dimensions must be positive')
    polygons = []
    for item in (box, target):
        corners = np.asarray(item.corners(), dtype=np.float64)
        polygons.append(corners[[0, 1]][:, [2, 3, 7, 6]].T)
    intersection_area = _intersection(*polygons)
    volume_sum = np.prod(sizes[0]) + np.prod(sizes[1])
    exact_upper = min(center[2] + size[2] / 2 for center, size in zip(centers, sizes))
    exact_lower = max(center[2] - size[2] / 2 for center, size in zip(centers, sizes))
    exact_intersection = intersection_area * max(exact_upper - exact_lower, 0.0)
    exact_union = volume_sum - exact_intersection
    exact = exact_intersection / exact_union if exact_union > 0 else 0.0
    compat_upper = min(center[2] for center in centers)
    compat_lower = max(center[2] - size[2] for center, size in zip(centers, sizes))
    compat_intersection = intersection_area * max(compat_upper - compat_lower, 0.0)
    compat_union = volume_sum - compat_intersection
    compat = compat_intersection / compat_union if compat_union > 0 else 0.0
    distance = np.linalg.norm(centers[0] - centers[1])
    return float(np.clip(exact, 0, 1)), float(np.clip(compat, 0, 1)), float(distance)


@torch.no_grad()
def supervision_geometry(batch, output):
    """四个候选各算一次公共几何；返回原 targets 和 legacy q0 exact。

候选无效也保留其标签，特别是 Full fallback 的 q0 在旧质量监督中仍有效。
旧基础 loss 的 size clamp/真值 dtype 与 v37 不同时，仅 q0 走原精确公式。
    """
    boxes = output.candidates.boxes.detach()
    target = batch['target_box'].detach().to(boxes)
    size = batch['box_size'].detach().to(boxes)
    target_size = batch.get('target_box_size', size).detach().to(boxes)

    # 原字段各传输一次；CPU 上保持两条既有路径各自的 dtype 转换顺序。
    boxes_cpu = boxes.cpu()
    target_cpu = batch['target_box'].detach().cpu()
    size_cpu = batch['box_size'].detach().cpu()
    target_size_cpu = (batch['target_box_size'].detach().cpu()
                       if 'target_box_size' in batch else None)
    bxs = boxes_cpu.double().numpy()
    gts = target_cpu.to(dtype=boxes.dtype).double().numpy()
    szs = size_cpu.to(dtype=boxes.dtype).double().numpy()
    gszs = (target_size_cpu.to(dtype=boxes.dtype).double().numpy()
            if target_size_cpu is not None else szs)
    old_gts = target_cpu.double().numpy()
    old_size_cpu = size_cpu.to(dtype=target_cpu.dtype).clamp_min(1e-3)
    old_szs = old_size_cpu.double().numpy()
    old_gszs = target_size_cpu.double().numpy() if target_size_cpu is not None else old_szs

    exact = np.zeros(boxes.shape[:2], np.float32)
    utility = np.zeros((*boxes.shape[:2], 2), np.float32)
    legacy_q0 = np.zeros((len(boxes), 1), np.float32)
    for row in range(len(boxes)):
        truth = LocalYawBox(gts[row], gszs[row][[1, 0, 2]])
        for candidate in range(4):
            prediction = LocalYawBox(bxs[row, candidate], szs[row][[1, 0, 2]])
            exact_iou, compat_iou, distance = paired_box_metrics(prediction, truth)
            exact[row, candidate] = exact_iou
            utility[row, candidate] = metric_contributions(compat_iou, distance)
        same_geometry = (np.array_equal(gts[row], old_gts[row])
                         and np.array_equal(szs[row], old_szs[row])
                         and np.array_equal(gszs[row], old_gszs[row]))
        if same_geometry:
            legacy_q0[row, 0] = exact[row, 0]
        else:
            # v37 合同保证 legacy q0 与 candidates[:,0] 是同一公开框。
            legacy_q0[row, 0] = box_metrics(
                LocalYawBox(bxs[row, 0], old_szs[row][[1, 0, 2]]),
                LocalYawBox(old_gts[row], old_gszs[row][[1, 0, 2]]),
                up_axis=(0, 0, 1), mode='geometry_exact')[0]
    exact_tensor = torch.as_tensor(exact, device=boxes.device, dtype=boxes.dtype)
    quality = quality_targets(boxes, target, size, torch.ones_like(exact_tensor), exact_tensor)
    targets = dict(utility=boxes.new_tensor(utility), commit_quality=quality.detach(),
                   exact_iou=exact_tensor.detach())
    old_boxes = output.decoder.hypothesis_boxes
    q0_tensor = torch.as_tensor(legacy_q0, device=old_boxes.device, dtype=old_boxes.dtype)
    return targets, q0_tensor
