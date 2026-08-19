"""Online recursive training and B3 shadow-rollout transactions."""

import time

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from datasets import points_utils
from ctseqtrack.data.sample_builder import motion_processing_mf
from ctseqtrack.data.recursive import (
    RecursiveTrackState,
    apply_training_reanchor,
    build_recursive_input_contract,
    commit_canonical_prediction,
    rotating_rollout_horizon,
)
from ctseqtrack.runtime.contracts import online_candidate_state_consistent
from utils.ct_history import (
    normalize_causal_temporal_gaps,
    select_causal_temporal_candidates,
    select_uniform_temporal_candidates,
)
from utils.metrics import estimateOverlap
from utils.sampling_utils import (
    deterministic_recovery_candidate_offset,
    stable_uint32_seed,
)


def recursive_state_for_raw(self, raw):
    if not hasattr(self, "_ct_recursive_states"):
        self._ct_recursive_states = {}
    key = (int(raw["online_epoch"]), str(raw["tracklet_key"]))
    state = self._ct_recursive_states.get(key)
    if state is None:
        state = RecursiveTrackState(
            tracklet_id=int(raw["tracklet_id"]),
            tracklet_key=str(raw["tracklet_key"]),
            first_box=raw["first_frame"]["3d_bbox"],
            timestamps={0: raw["first_frame"].get("timestamp")},
        )
        self._ct_recursive_states[key] = state
    return state


def ordered_online_history_frames(raw):
    return [
        raw["prev_frames"][key]
        for key in sorted(raw["prev_frames"], key=lambda value: abs(int(value)))
    ]


def online_rollout_horizon(self, raw):
    return rotating_rollout_horizon(
        getattr(self.config, "ct_recursive_rollout_horizons", [1, 2, 4, 8]),
        raw["online_slot"],
        raw["online_epoch"],
        getattr(self.config, "ct_recursive_tracklet_slots", 4),
    )


def prepare_online_state_group(self, raw, state):
    """Apply one causal expert boundary before all candidate views."""
    horizon = self._online_rollout_horizon(raw)
    return apply_training_reanchor(raw, state, horizon, self.config)


def online_motion_prepass(self, raw, state):
    return self._online_motion_prepass_batch([(raw, state)])[0]


@torch.no_grad()
def online_motion_prepass_batch(self, raw_state_pairs):
    """Run one vectorized B1 forward for the supplied causal histories."""
    if not (
        self.ct_joint_contract_version >= 2
        and self.use_b1_prepass_support
        and self.ct_enable_b1
    ):
        return [None] * len(raw_state_pairs)
    prepass_inputs = []
    for raw, state in raw_state_pairs:
        contract = build_recursive_input_contract(
            state,
            raw["this_frame_id"],
            len(raw["prev_frame_ids"]),
            self.config,
            candidate_id=raw["candidate_id"],
            offsets=raw["history_offsets"],
        )
        history_boxes = state.history_boxes(
            contract["history_frame_ids"], contract["history_valid_mask"].tolist()
        )
        history_frames = self._ordered_online_history_frames(raw)
        current_frame = raw["this_frame"]
        inputs = self._build_motion_prepass_inputs_contract(
            history_boxes,
            contract["history_frame_ids"],
            contract["history_valid_mask"].tolist(),
            list(contract["history_timestamps"]),
            current_frame.get("timestamp"),
            [frame.get("_ct_effective_timestamp") for frame in history_frames],
            current_frame.get("_ct_effective_timestamp"),
            current_frame.get(
                "_ct_dynamics_time_mode",
                getattr(self.config, "dynamics_time_mode", "true"),
            ),
            int(raw["this_frame_id"]),
            contract.get("history_observation_diagnostics"),
            contract.get("history_diagnostic_valid_mask"),
        )
        if inputs is None:
            raise RuntimeError(
                "online B1 prepass history length does not match hist_num"
            )
        prepass_inputs.append(inputs)
    prediction = self.predict_motion_from_history(
        np.stack([item["ref_boxs"] for item in prepass_inputs]),
        np.stack([item["delta_t"] for item in prepass_inputs]),
        np.stack([item["valid_mask"] for item in prepass_inputs]),
        np.asarray(
            [item["current_delta_t"] for item in prepass_inputs], dtype=np.float32
        ),
        np.stack(
            [
                item.get(
                    "history_observation_diagnostics",
                    np.zeros((self.hist_num, 6), dtype=np.float32),
                )
                for item in prepass_inputs
            ]
        ),
        np.stack(
            [
                item.get(
                    "history_diagnostic_valid_mask",
                    np.zeros(self.hist_num, dtype=np.float32),
                )
                for item in prepass_inputs
            ]
        ),
    )
    return self._unbatch_motion_prepass_predictions(
        prediction, [item["current_delta_t"] for item in prepass_inputs]
    )


def temporal_raw_view(raw, gap, candidate_id):
    """Materialize one role view from a grouped raw temporal carrier."""
    pool = raw.get("temporal_candidate_pool")
    if not isinstance(pool, dict) or int(gap) not in pool:
        raise KeyError(f"grouped raw carrier lacks temporal gap {gap}")
    entry = pool[int(gap)]
    view = dict(raw)
    view.update(
        {
            "candidate_id": int(candidate_id),
            "candidate_gap_frames": int(gap),
            "prev_frames": entry["prev_frames"],
            "prev_frame_ids": list(entry["prev_frame_ids"]),
            "valid_mask": list(entry["valid_mask"]),
            "history_offsets": list(entry["history_offsets"]),
            "point_sampling_seeds": np.asarray(
                entry["point_sampling_seeds"], dtype=np.int64
            ),
            "current_sampling_seed": int(entry["current_sampling_seed"]),
            "candidate_shared_transform": np.zeros(3, dtype=np.float32),
            "shadow_future": (
                list(raw.get("shadow_future", [])) if int(candidate_id) == 0 else []
            ),
        }
    )
    for key in (
        "motion_aux_prev_frames",
        "motion_aux_valid_mask",
        "motion_aux_frame_ids",
        "motion_aux_offsets",
    ):
        view.pop(key, None)
    return view


def expand_causal_temporal_groups(self, group_context):
    """Select c1/c2 from live B1 endpoints and emit three role views."""
    gaps = normalize_causal_temporal_gaps(
        getattr(self.config, "ct_temporal_candidate_gaps", [2, 4, 8])
    )
    boundary_band = float(getattr(self.config, "ct_temporal_boundary_band", 0.2))
    requests = []
    request_keys = []
    raw_views = {}
    for group_key, group in group_context.items():
        for gap in [1] + gaps:
            view = self._temporal_raw_view(group["raw"], gap, 0)
            raw_views[(group_key, gap)] = view
            requests.append((view, group["state"]))
            request_keys.append((group_key, gap))
    predictions = self._online_motion_prepass_batch(requests)
    prediction_map = dict(zip(request_keys, predictions))

    expanded = []
    for group_key, group in group_context.items():
        half_extent = np.maximum(
            0.5
            * np.asarray(group["state"].target_size[:2], dtype=np.float64)
            * float(getattr(self.config, "bb_scale", 1.0))
            + float(getattr(self.config, "bb_offset", 0.0)),
            1e-3,
        )
        ratios = {}
        available = {}
        for gap in gaps:
            prediction = prediction_map[(group_key, gap)]
            endpoint = np.asarray(prediction["mu_xy"], dtype=np.float64).reshape(2)
            ratios[gap] = float(np.max(np.abs(endpoint) / half_extent))
            history_valid = all(
                int(value) for value in raw_views[(group_key, gap)]["valid_mask"]
            )
            available[gap] = bool(history_valid and prediction.get("valid", False))
        candidate_policy = (
            str(getattr(self.config, "ct_candidate_policy", "causal_b1_boundary"))
            .strip()
            .lower()
        )
        if candidate_policy == "causal_temporal_uniform":
            raw = group["raw"]
            selected = select_uniform_temporal_candidates(
                ratios,
                available,
                seed_parts=(
                    int(getattr(self.config, "seed", 42) or 42),
                    int(raw.get("online_epoch", 0)),
                    str(raw["tracklet_key"]),
                    int(raw["this_frame_id"]),
                ),
            )
        else:
            selected = select_causal_temporal_candidates(
                ratios, available, boundary_band=boundary_band
            )

        canonical = self._temporal_raw_view(group["raw"], 1, 0)
        canonical_prediction = prediction_map[(group_key, 1)]
        canonical_ratio = float(
            np.max(
                np.abs(
                    np.asarray(canonical_prediction["mu_xy"], dtype=np.float64).reshape(
                        2
                    )
                )
                / half_extent
            )
        )
        canonical.update(
            {
                "candidate_role": 0,
                "candidate_available": 1.0,
                "candidate_boundary_ratio": canonical_ratio,
                "candidate_role_satisfied": 1.0,
            }
        )
        expanded.append(
            (canonical, group["state"], group["diagnostics"], canonical_prediction)
        )

        for role_id in (1, 2):
            role = selected[role_id]
            fallback_gap = gaps[min(role_id - 1, len(gaps) - 1)]
            gap = fallback_gap if role["gap"] is None else int(role["gap"])
            view = self._temporal_raw_view(group["raw"], gap, role_id)
            view.update(
                {
                    "candidate_role": role_id,
                    "candidate_available": float(role["available"]),
                    "candidate_boundary_ratio": float(role["boundary_ratio"]),
                    "candidate_role_satisfied": float(role["role_satisfied"]),
                }
            )
            selector = getattr(self, "_ct_selector_epoch", None)
            if isinstance(selector, dict):
                role_key = str(role_id)
                gap_key = str(gap)
                selector["gap_counts"][role_key][gap_key] = (
                    selector["gap_counts"][role_key].get(gap_key, 0) + 1
                )
                selector["available"][role_key] += int(bool(role["available"]))
                selector["satisfied"][role_key] += int(
                    bool(role["available"]) and bool(role["role_satisfied"])
                )
                identity = (
                    str(group["raw"]["tracklet_key"]),
                    int(group["raw"]["this_frame_id"]),
                    role_id,
                )
                previous = getattr(self, "_ct_selector_previous", {}).get(identity)
                if previous is not None:
                    selector["migration_comparisons"] += 1
                    selector["migrations"] += int(int(previous) != gap)
                selector["current"][identity] = gap
            expanded.append(
                (
                    view,
                    group["state"],
                    group["diagnostics"],
                    prediction_map[(group_key, gap)],
                )
            )
    return expanded


def process_online_raw(
    self, raw, state, motion_prediction=None, state_diagnostics=None
):
    payload = {
        key: value
        for key, value in raw.items()
        if key
        not in (
            "online_recursive_raw",
            "online_epoch",
            "online_batch_index",
            "online_slot",
            "shadow_future",
            "temporal_candidate_pool",
        )
    }
    contract = build_recursive_input_contract(
        state,
        raw["this_frame_id"],
        len(raw["prev_frame_ids"]),
        self.config,
        candidate_id=raw["candidate_id"],
        offsets=raw["history_offsets"],
    )
    if contract["history_frame_ids"] != list(raw["prev_frame_ids"]) or contract[
        "history_valid_mask"
    ].tolist() != list(raw["valid_mask"]):
        raise RuntimeError("raw/state recursive history contract mismatch")
    candidate_id = int(raw["candidate_id"])
    candidate_policy = (
        str(getattr(self.config, "ct_candidate_policy", "legacy_spatial"))
        .strip()
        .lower()
    )
    causal_policy = candidate_policy in (
        "causal_b1_boundary",
        "causal_temporal_uniform",
    )
    recovery_policy = (
        str(getattr(self.config, "ct_recovery_candidate_policy", "off")).strip().lower()
    )
    if recovery_policy != "off" and candidate_policy != "legacy_spatial_gt_ablation":
        raise RuntimeError(
            "GT-spatial recovery requires the explicit "
            "legacy_spatial_gt_ablation policy"
        )
    if causal_policy:
        contract["candidate_shared_transform"] = np.zeros(3, dtype=np.float32)
        contract["point_sampling_seeds"] = np.asarray(
            raw["point_sampling_seeds"], dtype=np.int64
        )
        contract["current_sampling_seed"] = int(raw["current_sampling_seed"])
    elif recovery_policy == "weak_miss_control" and candidate_id in (1, 2):
        anchor_box = state.history_boxes(
            contract["history_frame_ids"], contract["history_valid_mask"].tolist()
        )[0]
        contract["candidate_shared_transform"] = (
            deterministic_recovery_candidate_offset(
                candidate_id,
                self.config,
                anchor_box,
                raw["this_frame"]["3d_bbox"],
                state.tracklet_key,
                raw["this_frame_id"],
            )
        )
    payload["online_recursive_state"] = contract
    payload["candidate_shared_transform"] = contract["candidate_shared_transform"]
    payload["point_sampling_seeds"] = contract["point_sampling_seeds"]
    payload["current_sampling_seed"] = contract["current_sampling_seed"]
    if motion_prediction is not None:
        payload["motion_prediction"] = motion_prediction
    if "motion_aux_frame_ids" in raw:
        aux_contract = build_recursive_input_contract(
            state,
            raw["this_frame_id"],
            len(raw["motion_aux_frame_ids"]),
            self.config,
            candidate_id=raw["candidate_id"],
            offsets=raw["motion_aux_offsets"],
        )
        if aux_contract["history_frame_ids"] != list(
            raw["motion_aux_frame_ids"]
        ) or aux_contract["history_valid_mask"].tolist() != list(
            raw["motion_aux_valid_mask"]
        ):
            raise RuntimeError("raw/state auxiliary history contract mismatch")
        payload["online_motion_aux_state"] = aux_contract
    processed = motion_processing_mf(payload, self.config)
    candidate_consistent = online_candidate_state_consistent(
        processed, contract["target_size"]
    )
    if not candidate_consistent:
        raise RuntimeError("candidate crop/history/Search state contract diverged")
    state_diagnostics = state_diagnostics or {}
    current_labels = processed["seg_label"][
        -int(getattr(self.config, "point_sample_size", 1024)) :
    ]
    processed.update(
        {
            "ct_online_tracklet_id": np.int64(raw["tracklet_id"]),
            "ct_online_frame_id": np.int64(raw["this_frame_id"]),
            "ct_online_slot": np.int64(raw["online_slot"]),
            "ct_online_epoch": np.int64(raw["online_epoch"]),
            "ct_recursive_state_age": np.float32(
                state_diagnostics.get(
                    "rollout_age", int(raw["this_frame_id"]) - max(state.predictions)
                )
            ),
            "ct_recursive_rollout_horizon": np.float32(
                state_diagnostics.get("rollout_horizon", 0)
            ),
            "ct_recursive_reset_boundary": np.float32(
                bool(state_diagnostics.get("reset_boundary", False))
            ),
            "ct_recursive_state_source": np.float32(
                1.0 if state_diagnostics.get("rollout_age", 0) > 0 else 0.0
            ),
            "ct_recursive_pre_reset_anchor_error": np.float32(
                state_diagnostics.get("pre_reset_anchor_error", 0.0)
            ),
            "ct_recursive_anchor_error": np.float32(
                state_diagnostics.get("post_reset_anchor_error", 0.0)
            ),
            "ct_crop_target_points": np.float32(np.sum(current_labels > 0)),
            "ct_candidate_state_consistency": np.float32(candidate_consistent),
        }
    )
    return processed


def prepare_online_recursive_batch(self, raw_items):
    processed = []
    context = []
    group_context = {}
    for raw in raw_items:
        if not raw.get("online_recursive_raw", False):
            raise ValueError("mixed online/non-online training batch")
        group_key = (
            int(raw["online_slot"]),
            str(raw["tracklet_key"]),
            int(raw["this_frame_id"]),
        )
        if group_key in group_context:
            continue
        state = self._recursive_state_for_raw(raw)
        if self.ct_joint_contract_version >= 2:
            diagnostics = self._prepare_online_state_group(raw, state)
        else:
            diagnostics = {}
        group_context[group_key] = {
            "raw": raw,
            "state": state,
            "diagnostics": diagnostics,
            "motion_prediction": None,
        }
    causal_policy = str(
        getattr(self.config, "ct_candidate_policy", "legacy_spatial")
    ).strip().lower() in ("causal_b1_boundary", "causal_temporal_uniform")
    if causal_policy:
        if any(int(raw["candidate_id"]) != 0 for raw in raw_items):
            raise RuntimeError(
                "causal temporal batch must contain grouped candidate0 carriers"
            )
        expanded = self._expand_causal_temporal_groups(group_context)
    else:
        group_keys = list(group_context)
        prepass_predictions = self._online_motion_prepass_batch(
            [
                (group_context[key]["raw"], group_context[key]["state"])
                for key in group_keys
            ]
        )
        for key, prediction in zip(group_keys, prepass_predictions):
            group_context[key]["motion_prediction"] = prediction
        expanded = []
        for raw in raw_items:
            group_key = (
                int(raw["online_slot"]),
                str(raw["tracklet_key"]),
                int(raw["this_frame_id"]),
            )
            group = group_context[group_key]
            expanded.append(
                (raw, group["state"], group["diagnostics"], group["motion_prediction"])
            )
    for raw, state, diagnostics, prediction in expanded:
        processed.append(
            self._process_online_raw(
                raw, state, motion_prediction=prediction, state_diagnostics=diagnostics
            )
        )
        context.append({"raw": raw, "state": state})
    batch = default_collate(processed)
    batch_size = len(expanded)
    batch["ct_h3_gain"] = torch.zeros(batch_size, dtype=torch.float32)
    batch["ct_h3_center_gain"] = torch.zeros(batch_size, dtype=torch.float32)
    batch["ct_h3_iou_gain"] = torch.zeros(batch_size, dtype=torch.float32)
    batch["ct_h3_valid"] = torch.zeros(batch_size, dtype=torch.float32)
    batch["ct_shadow_forward_count"] = torch.tensor(0.0)
    batch["ct_shadow_time_ms"] = torch.tensor(0.0)
    batch["ct_shadow_peak_memory_mb"] = torch.tensor(0.0)
    self._ct_online_batch_context = context
    return self._move_batch_to_device(batch, self.device)


def local_prediction_to_world(self, local_box, anchor_box):
    values = local_box.detach().cpu().numpy().reshape(-1)[:4]
    return points_utils.getOffsetBB(
        anchor_box,
        values,
        degrees=self.config.degrees,
        use_z=self.config.use_z,
        limit_box=self.config.limit_box,
    )


def shadow_forward(self, batch, seed):
    training_flags = {module: module.training for module in self.modules()}
    previous_joint_full = self.use_ct_joint_full
    previous_motion_v3 = self.use_b1motion_v3
    previous_b2 = self.ct_enable_b2
    previous_b3 = self.ct_enable_b3
    previous_b1 = self.ct_enable_b1
    cuda_devices = (
        [
            (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
        ]
        if self.device.type == "cuda"
        else []
    )
    try:
        for module in training_flags:
            module.training = False
        # H=3 future steps are deliberately observation-only.  Disabling
        # the B1/B2/B3 action gates is insufficient because the structural
        # Joint Full branch still validates and consumes Search tensors.
        # The shadow sampler intentionally omits those tensors, so bypass
        # both plugin entry points and execute the exact B0 forward.
        self.use_ct_joint_full = False
        self.use_b1motion_v3 = False
        self.ct_enable_b2 = False
        self.ct_enable_b3 = False
        self.ct_enable_b1 = False
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            with torch.inference_mode():
                return self(batch)
    finally:
        self.use_ct_joint_full = previous_joint_full
        self.use_b1motion_v3 = previous_motion_v3
        self.ct_enable_b2 = previous_b2
        self.ct_enable_b3 = previous_b3
        self.ct_enable_b1 = previous_b1
        for module, was_training in training_flags.items():
            module.training = was_training


def attach_h3_shadow_labels(self, batch, output):
    if (
        not bool(getattr(self.config, "ct_online_recursive_training", False))
        or not self.ct_enable_b3
    ):
        return
    if self.device.type == "cuda":
        torch.cuda.synchronize(self.device)
    start = time.perf_counter()
    memory_before = (
        torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
    )
    if self.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(self.device)
    shadow_forward_count = 0
    for index, item in enumerate(self._ct_online_batch_context):
        raw = item["raw"]
        if not raw.get("shadow_future"):
            continue
        if int(raw["candidate_id"]) != 0:
            raise RuntimeError("H=3 shadow is candidate0-only")
        if float(output["ct_search_candidate_valid"][index].detach()) <= 0:
            continue
        state_before = item["state"].clone()
        anchor = state_before.history_boxes([raw["prev_frame_ids"][0]], [1])[0]
        observation_local = (
            output["observation_aux_estimation_boxes"][index].detach().clone()
        )
        search_local = observation_local.clone()
        search_local[:2] = (
            observation_local[:2]
            + output["ct_router_bounded_residual_xy"][index].detach()
        )
        observation_box = self._local_prediction_to_world(observation_local, anchor)
        search_box = self._local_prediction_to_world(search_local, anchor)
        state_o = state_before.clone()
        state_s = state_before.clone()
        timestamp = raw["this_frame"].get("timestamp")
        current_diagnostic = output.get("ct_b0_history_diagnostic")
        current_diagnostic_valid = output.get("ct_b0_history_diagnostic_valid")
        diagnostic_row = (
            None
            if current_diagnostic is None
            else current_diagnostic[index].detach().cpu().numpy()
        )
        diagnostic_valid = (
            False
            if current_diagnostic_valid is None
            else bool(current_diagnostic_valid[index].detach().item() > 0)
        )
        state_o.append(
            raw["this_frame_id"],
            observation_box,
            timestamp,
            observation_diagnostics=diagnostic_row,
            diagnostic_valid=diagnostic_valid,
        )
        state_s.append(
            raw["this_frame_id"],
            search_box,
            timestamp,
            observation_diagnostics=diagnostic_row,
            diagnostic_valid=diagnostic_valid,
        )
        target = np.asarray(raw["this_frame"]["3d_bbox"].center[:2], dtype=np.float64)
        cost_o = float(np.linalg.norm(observation_box.center[:2] - target))
        cost_s = float(np.linalg.norm(search_box.center[:2] - target))
        target_box = raw["this_frame"]["3d_bbox"]
        iou_gain = float(
            estimateOverlap(search_box, target_box, dim=self.config.IoU_space)
            - estimateOverlap(observation_box, target_box, dim=self.config.IoU_space)
        )
        horizon_count = 1

        for future_raw in raw["shadow_future"]:
            shadow_processed = [
                self._process_online_raw(future_raw, state_o),
                self._process_online_raw(future_raw, state_s),
            ]
            shadow_batch = self._move_batch_to_device(
                default_collate(shadow_processed), self.device
            )
            shadow_seed = stable_uint32_seed(
                int(getattr(self.config, "seed", 42) or 42),
                raw["tracklet_key"],
                raw["this_frame_id"],
                future_raw["this_frame_id"],
                "h3_shadow_observation",
            )
            shadow_output = self._shadow_forward(shadow_batch, shadow_seed)
            shadow_forward_count += 2
            future_boxes = []
            for branch_index, branch_state in enumerate((state_o, state_s)):
                branch_anchor = branch_state.history_boxes(
                    [future_raw["prev_frame_ids"][0]], [1]
                )[0]
                local_observation = shadow_output["observation_aux_estimation_boxes"][
                    branch_index
                ]
                world_box = self._local_prediction_to_world(
                    local_observation, branch_anchor
                )
                branch_state.append(
                    future_raw["this_frame_id"],
                    world_box,
                    future_raw["this_frame"].get("timestamp"),
                    observation_diagnostics=(
                        shadow_output["ct_b0_history_diagnostic"][branch_index]
                        .detach()
                        .cpu()
                        .numpy()
                        if "ct_b0_history_diagnostic" in shadow_output
                        else None
                    ),
                    diagnostic_valid=(
                        bool(
                            shadow_output["ct_b0_history_diagnostic_valid"][
                                branch_index
                            ]
                            .detach()
                            .item()
                            > 0
                        )
                        if "ct_b0_history_diagnostic_valid" in shadow_output
                        else False
                    ),
                )
                future_boxes.append(world_box)
            future_target = np.asarray(
                future_raw["this_frame"]["3d_bbox"].center[:2], dtype=np.float64
            )
            cost_o += float(np.linalg.norm(future_boxes[0].center[:2] - future_target))
            cost_s += float(np.linalg.norm(future_boxes[1].center[:2] - future_target))
            future_target_box = future_raw["this_frame"]["3d_bbox"]
            iou_gain += float(
                estimateOverlap(
                    future_boxes[1], future_target_box, dim=self.config.IoU_space
                )
                - estimateOverlap(
                    future_boxes[0], future_target_box, dim=self.config.IoU_space
                )
            )
            horizon_count += 1
        center_gain = float(cost_o - cost_s) / float(horizon_count)
        iou_gain /= float(horizon_count)
        batch["ct_h3_gain"][index] = center_gain
        batch["ct_h3_center_gain"][index] = center_gain
        batch["ct_h3_iou_gain"][index] = iou_gain
        batch["ct_h3_valid"][index] = 1.0

    if self.device.type == "cuda":
        torch.cuda.synchronize(self.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    memory_after = (
        torch.cuda.max_memory_allocated(self.device)
        if self.device.type == "cuda"
        else memory_before
    )
    batch["ct_shadow_forward_count"] = batch["ct_shadow_forward_count"].new_tensor(
        float(shadow_forward_count)
    )
    batch["ct_shadow_time_ms"] = batch["ct_shadow_time_ms"].new_tensor(elapsed_ms)
    batch["ct_shadow_peak_memory_mb"] = batch["ct_shadow_peak_memory_mb"].new_tensor(
        max(0, memory_after - memory_before) / (1024.0**2)
    )


def commit_online_recursive_predictions(self, output):
    if not bool(getattr(self.config, "ct_online_recursive_training", False)):
        return
    seen_slots = set()
    for index, item in enumerate(self._ct_online_batch_context):
        raw = item["raw"]
        slot = int(raw["online_slot"])
        if int(raw["candidate_id"]) != 0:
            continue
        if slot in seen_slots:
            raise RuntimeError("online batch contains duplicate canonical slot")
        seen_slots.add(slot)
        state = item["state"]
        anchor = state.history_boxes([raw["prev_frame_ids"][0]], [1])[0]
        state_policy = (
            str(getattr(self.config, "ct_training_state_policy", "observation"))
            .strip()
            .lower()
        )
        if state_policy != "observation":
            raise RuntimeError(
                "formal CT training permits only observation recursive "
                "state; B2/B3 are shadow learners"
            )
        final_box = self._local_prediction_to_world(
            output["observation_aux_estimation_boxes"][index], anchor
        )
        diagnostic_tensor = output.get("ct_b0_history_diagnostic")
        diagnostic_valid_tensor = output.get("ct_b0_history_diagnostic_valid")
        observation_diagnostic = (
            None
            if diagnostic_tensor is None
            else diagnostic_tensor[index].detach().cpu().numpy()
        )
        diagnostic_valid = (
            False
            if diagnostic_valid_tensor is None
            else bool(diagnostic_valid_tensor[index].detach().item() > 0)
        )
        commit_canonical_prediction(
            state,
            raw["candidate_id"],
            raw["this_frame_id"],
            final_box,
            raw["this_frame"].get("timestamp"),
            observation_diagnostics=observation_diagnostic,
            diagnostic_valid=diagnostic_valid,
        )
