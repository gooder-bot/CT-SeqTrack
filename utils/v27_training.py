"""v27 动作监督与一次干预的 H3 影子通路。没有主递归状态写入。"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from utils.sampling_utils import stable_uint32_seed
from utils.tracking_metrics_v27 import local_boxes_metric_gains, box_metric_contributions


def _get(config, key, default):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _structural(output):
    for key in ("ct_b2_available", "ct_policy_candidate_valid"):
        if key in output:
            return output[key].detach().reshape(-1) > 0
    raise KeyError("v27 B3 requires structural availability, never a presence-threshold mask")


def _weighted_mean(error, mask):
    mask = mask.to(device=error.device, dtype=torch.bool)
    selected = torch.where(mask, error, torch.zeros_like(error))
    return selected.sum() / mask.sum().clamp_min(1)


def compute_b3_utility_loss(data, output, config):
    """输出两项条件式 gain MSE 和即时 help/harm BCE；返回未乘外层权重的 loss。"""
    predicted_s = output.get("ct_b3_expected_success_gain", output.get("ct_b3_expected_iou_gain"))
    predicted_p = output.get("ct_b3_expected_precision_gain", output.get("ct_b3_expected_center_gain"))
    if predicted_s is None or predicted_p is None:
        raise KeyError("v27 utility gain outputs are required")
    observation = output["observation_aux_estimation_boxes"].detach()[:, :4].clone()
    action = observation.clone()
    action[:, :2] += output["ct_router_bounded_residual_xy"].detach()
    target = data["box_label"].detach()[:, :4].clone().to(observation)
    prediction_wlh = data["bbox_size"].detach().to(observation)
    target_wlh = data["target_bbox_size"].detach().to(observation)
    if bool(_get(config, "degrees", False)):
        for boxes in (observation, action, target):
            boxes[:, 3] = torch.deg2rad(boxes[:, 3])
    valid = _structural(output).to(observation.device)
    if "b0_view_id" in data:
        valid &= data["b0_view_id"].to(valid.device).reshape(-1) == 0
    for tensor in (observation, action, target, prediction_wlh, target_wlh):
        valid &= torch.isfinite(tensor).all(1)
    valid &= (prediction_wlh > 0).all(1) & (target_wlh > 0).all(1)
    h1_s, h1_p = torch.zeros_like(predicted_s), torch.zeros_like(predicted_p)
    if bool(valid.any()):
        gains = local_boxes_metric_gains(observation[valid], action[valid], target[valid],
            prediction_wlh[valid], target_wlh[valid],
            up_axis=_get(config, "up_axis", (0, 0, 1)),
            mode="benchmark_compat", dim=int(_get(config, "IoU_space", 3)))
        h1_s[valid] = torch.as_tensor(gains["success_gain"], device=h1_s.device, dtype=h1_s.dtype)
        h1_p[valid] = torch.as_tensor(gains["precision_gain"], device=h1_p.device, dtype=h1_p.dtype)
    h3_mask = data.get("ct_h3_valid", torch.zeros_like(predicted_s)).detach().to(valid.device).reshape(-1) > 0
    h3_mask &= valid
    def gain_loss(predicted, h1, future_key):
        immediate = _weighted_mean((predicted - h1).square(), valid)
        future = data.get(future_key)
        if future is None:
            if bool(h3_mask.any()):
                raise KeyError(f"valid H3 rows require {future_key}")
            return immediate
        future = future.detach().to(predicted).reshape_as(predicted)
        mask = h3_mask & torch.isfinite(future)
        if not bool(mask.any()):
            return immediate
        # H3 labels contain the future-two-frame mean, never H1 again.
        delayed = _weighted_mean((predicted - torch.nan_to_num(future)).square(), mask)
        return .5 * immediate + .5 * delayed
    loss_s = gain_loss(predicted_s, h1_s, "ct_h3_success_gain")
    loss_p = gain_loss(predicted_p, h1_p, "ct_h3_precision_gain")
    utility = .5 * (h1_s + h1_p)
    helpful, harmful = (utility > 1e-6).to(predicted_s), (utility < -1e-6).to(predicted_s)
    loss_help = _weighted_mean(F.binary_cross_entropy_with_logits(
        output["ct_b3_help_logit"], helpful, reduction="none"), valid)
    loss_harm = _weighted_mean(F.binary_cross_entropy_with_logits(
        output["ct_b3_harm_logit"], harmful, reduction="none"), valid)
    total = loss_s + loss_p + .1 * loss_help + .1 * loss_harm
    return {"loss": total, "loss_success": loss_s, "loss_precision": loss_p,
            "loss_help": loss_help, "loss_harm": loss_harm,
            "help_label": helpful, "harm_label": harmful,
            "h1_success_gain": h1_s, "h1_precision_gain": h1_p,
            "valid": valid.to(predicted_s)}


def accumulate_v27_binary_rows(rows, data, output, b3_enabled):
    """诊断与实际损失对齐：selected presence、bounded S/P 和独立符号概率。"""
    labels = data['ct_extension_labels'].detach()
    point_valid = data['ct_extension_valid_mask'].detach()
    selected = output['ct_extension_selected_indices'].detach().long()
    labels = labels.gather(1, selected)
    point_valid = point_valid.gather(1, selected) * output[
        'ct_extension_selected_valid_mask'].detach().to(point_valid)
    available = output['ct_b2_available'].detach().reshape(-1) > 0
    if bool(available.any()):
        rows.setdefault('presence', []).append((
            output['ct_b2_extension_presence_probability'].detach()[available].float().cpu().numpy(),
            ((labels * point_valid).sum(1) > 0)[available].float().cpu().numpy()))
    valid = output['ct_b3_h1_valid'].detach().reshape(-1) > 0
    if not bool(valid.any()):
        return
    success = output['ct_b3_h1_success_gain_label'].detach()[valid]
    precision = output['ct_b3_h1_precision_gain_label'].detach()[valid]
    utility = .5 * (success + precision)
    score = output['ct_b3_action_score'].detach()[valid] if b3_enabled else torch.zeros_like(utility)
    rows.setdefault('bounded_utility', []).append(
        torch.stack((success, precision, score), dim=1).float().cpu().numpy())
    if b3_enabled:
        for name, target in (('help', utility > 1e-6), ('harm', utility < -1e-6)):
            rows.setdefault(name, []).append((
                torch.sigmoid(output[f'ct_b3_{name}_logit'].detach()[valid]).float().cpu().numpy(),
                target.float().cpu().numpy()))


def attach_h3_shadow_labels_v27(host, batch, output):
    """每次抽样最多两未来帧 × 两 observation 分支，不受 presence 控制。"""
    reference = output["observation_aux_estimation_boxes"]
    count = len(reference)
    for key in ("ct_h3_success_gain", "ct_h3_precision_gain", "ct_h3_valid",
                "ct_h3_scheduled", "ct_h3_shadow_valid"):
        batch[key] = reference.new_zeros(count)
    batch["ct_h3_future_exists"] = reference.new_full((count, 2), -1.)
    batch["ct_h3_failure_reason"] = ["not_scheduled"] * count
    if not bool(_get(host.config, "ct_online_recursive_training", False)) or not host.ct_enable_b3:
        return
    start = time.perf_counter()
    cuda = host.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(host.device)
        memory_before = torch.cuda.memory_allocated(host.device)
        torch.cuda.reset_peak_memory_stats(host.device)
    else:
        memory_before = 0
    forwards = 0
    structural = _structural(output)
    contexts = host._ct_online_batch_context
    if len(contexts) != count:
        raise ValueError("H3 contexts must align with canonical rows")
    for index, context in enumerate(contexts):
        raw = context["raw"]
        future_rows = raw.get("shadow_future", [])
        scheduled = bool(raw.get("shadow_scheduled", bool(future_rows)))
        batch["ct_h3_scheduled"][index] = float(scheduled)
        exists = raw.get("shadow_future_exists")
        if exists is not None:
            batch["ct_h3_future_exists"][index] = reference.new_tensor(exists).reshape(2)
        elif future_rows:
            batch["ct_h3_future_exists"][index] = reference.new_tensor([len(future_rows) >= 1, len(future_rows) >= 2])
        if not scheduled:
            continue
        if int(raw["candidate_id"]) != 0:
            raise RuntimeError("v27 H3 may sample canonical candidate0 only")
        if len(future_rows) != 2:
            batch["ct_h3_failure_reason"][index] = "terminal_incomplete_horizon" if exists is not None and not all(exists) else "missing_future_payload"
            continue
        if not bool(structural[index]):
            batch["ct_h3_failure_reason"][index] = "no_structural_evidence"
            continue
        current = reference[index].detach().clone()
        correction = output["ct_router_bounded_residual_xy"][index].detach()
        if not bool(torch.isfinite(current).all() & torch.isfinite(correction).all()):
            batch["ct_h3_failure_reason"][index] = "nonfinite_action"
            continue
        try:
            state_before = context["state"].clone()
            anchor = state_before.history_boxes([raw["prev_frame_ids"][0]], [1])[0]
            action_local = current.clone()
            action_local[:2] += correction
            observation_box = host._local_prediction_to_world(current, anchor)
            action_box = host._local_prediction_to_world(action_local, anchor)
            states = [state_before.clone(), state_before.clone()]
            for state, box in zip(states, (observation_box, action_box)):
                state.append(raw["this_frame_id"], box, raw["this_frame"].get("timestamp"))
            gains = []
            for source_future in future_rows:
                future = dict(source_future)
                future["ct_observation_only"] = True
                processed = [host._process_online_raw(future, state) for state in states]
                shadow_batch = host._move_batch_to_device(default_collate(processed), host.device)
                seed = stable_uint32_seed(int(_get(host.config, "seed", 42)),
                    raw["tracklet_key"], raw["this_frame_id"], future["this_frame_id"], "h3_shadow_observation")
                forwards += 2
                with torch.inference_mode():
                    shadow = host._shadow_forward(shadow_batch, seed)
                predictions = []
                for branch, state in enumerate(states):
                    future_anchor = state.history_boxes([future["prev_frame_ids"][0]], [1])[0]
                    world = host._local_prediction_to_world(shadow["observation_aux_estimation_boxes"][branch], future_anchor)
                    state.append(future["this_frame_id"], world, future["this_frame"].get("timestamp"))
                    predictions.append(world)
                target = future["this_frame"]["3d_bbox"]
                observation_sp, action_sp = [box_metric_contributions(
                    box, target, up_axis=_get(host.config, "up_axis", (0, 0, 1)),
                    mode="benchmark_compat", dim=int(_get(host.config, "IoU_space", 3))) for box in predictions]
                gains.append(np.asarray(action_sp) - np.asarray(observation_sp))
            delayed = np.mean(gains, axis=0)
            if not np.isfinite(delayed).all():
                raise ValueError("nonfinite future metric gain")
            batch["ct_h3_success_gain"][index] = float(delayed[0])
            batch["ct_h3_precision_gain"][index] = float(delayed[1])
            batch["ct_h3_valid"][index] = 1.
            batch["ct_h3_shadow_valid"][index] = 1.
            batch["ct_h3_failure_reason"][index] = "ok"
        except torch.cuda.OutOfMemoryError:
            raise
        except (ValueError, KeyError, IndexError, RuntimeError, FloatingPointError) as error:
            # An unsuccessful shadow is missing supervision, never a zero-return action.
            batch["ct_h3_failure_reason"][index] = f"shadow_failure:{type(error).__name__}:{str(error)[:160]}"
    if cuda:
        torch.cuda.synchronize(host.device)
    batch["ct_shadow_forward_count"] = reference.new_tensor(float(forwards))
    batch["ct_shadow_time_ms"] = reference.new_tensor((time.perf_counter() - start) * 1000.)
    batch["ct_shadow_peak_memory_mb"] = reference.new_tensor(
        max(0, torch.cuda.max_memory_allocated(host.device) - memory_before) / 1024**2 if cuda else 0.)
