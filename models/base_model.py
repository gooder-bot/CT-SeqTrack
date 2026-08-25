"""
baseModel.py
Created by zenn at 2021/5/9 14:40
Modified by Aron Lin at Jun 6 17:39:22 CST 2023
"""

import csv
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision, AverageMeter, TorchRuntime, TorchNumFrames
from utils.metrics import estimateOverlap, estimateAccuracy
from utils.waymo_metrics import estimateWaymoOverlap # only for waymo IOU
import torch.nn.functional as F
import numpy as np
from nuscenes.utils import geometry_utils

from datasets.misc_utils import get_history_frame_ids_and_masks,get_last_n_bounding_boxes
from datasets.misc_utils import (
    build_effective_time_fields,
    build_main_time_fields,
    build_time_fields,
    normalize_dynamics_time_mode,
)
from utils.ct_search import (
    build_ordered_trajectory_search_box,
    build_time_guided_search_box,
    combined_search_support_statistics,
    resolve_b1_search_support,
    resolve_joint_search_geometry,
    sample_padded_search_extension,
    sample_joint_novel_extensions,
    sample_source_aware_endpoint_points,
    sample_search_extension,
    stratified_search_sample,
    useful_search_coverage_need,
)
from utils.replay_cache import b2_candidate_config_sha256
from utils.recursive_state import (
    build_recursive_input_contract,
    RecursiveTrackState,
)

import time

class BaseModelMF(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config
        self.train_dataloader_length = kwargs.get('train_dataloader_length', None)

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()
        self.runtime = TorchRuntime()

        self.prec_step = TorchPrecision()
        self.success_step = TorchSuccess()
        if (bool(getattr(config, "use_ct_joint_full", False))
                and int(getattr(
                    config, "ct_joint_contract_version", 1)) >= 3):
            self.ct_observation_success = TorchSuccess()
            self.ct_raw_search_success = TorchSuccess()

        self.n_frames = TorchNumFrames()
        self._proposal_sequence_diagnostics = []
        self._proposal_test_diagnostics = []
        self._tracking_test_endpoints = []
        self._b3_sequence_rollouts = []
        self._b3_test_rollouts = []


    def configure_optimizers(self):
        # Experimental modules may contain non-trainable state; keep those
        # tensors out of optimizer parameter groups.
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.optimizer.lower() == 'adamonecycle':
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epoch,
                steps_per_epoch=self.train_dataloader_length)
            # The single-cycle learning rate needs to be explicitly updated step by step
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}} 
        else:
            raise ValueError("Invalid optimizer. Please choose from 'sgd', 'adam', or 'adamonecycle'.")



    def compute_loss(self, data, output):
        raise NotImplementedError

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        raise NotImplementedError

    def evaluate_one_sample(self, data_dict, ref_box):
        end_points = self(data_dict)

        estimation_box = end_points['aux_estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        valid_mask = end_points['valid_mask'].squeeze(0).detach().cpu().numpy()

        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)

        return candidate_box, valid_mask, end_points

    @staticmethod
    def _proposal_scalar(mapping, key, default=0.0, column=None):
        value = mapping.get(key)
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if column is not None:
                value = value.reshape(value.shape[0], -1)[0, column]
            else:
                value = value.reshape(-1)[0]
            return float(value.item())
        array = np.asarray(value)
        if column is not None:
            return float(array.reshape(array.shape[0], -1)[0, column])
        return float(array.reshape(-1)[0])

    def _build_proposal_diagnostic_row(
            self, output, data_dict, this_box, reference_box, frame_id):
        """Build one GT-labelled endpoint row for B2-v2.1 attribution."""
        target_box = points_utils.transform_box(this_box, reference_box)
        target_xy = np.asarray(target_box.center[:2], dtype=np.float64)

        def tensor_xy(key, fallback):
            value = output.get(key)
            if value is None:
                return np.asarray(fallback, dtype=np.float64)
            return value.detach().cpu().numpy().reshape(-1, 2)[0].astype(
                np.float64)

        observation_xy = output[
            "observation_aux_estimation_boxes"
        ].detach().cpu().numpy().reshape(-1, 4)[0, :2].astype(np.float64)
        final_xy = output[
            "aux_estimation_boxes"
        ].detach().cpu().numpy().reshape(-1, 4)[0, :2].astype(np.float64)
        motion_xy = tensor_xy(
            "search_v21_motion_proposal_xy", observation_xy)
        search_xy = tensor_xy("search_v21_proposal_xy", observation_xy)
        motion_valid = self._proposal_scalar(
            output, "search_v21_motion_candidate_available",
            default=self._proposal_scalar(
                output, "search_v21_motion_candidate_valid")) > 0.0
        search_valid = self._proposal_scalar(
            output, "search_v21_search_candidate_available",
            default=self._proposal_scalar(
                output, "search_v21_search_candidate_valid")) > 0.0
        margin = float(getattr(
            self.config, "advantage_help_margin", 0.05))
        observation_error = float(np.linalg.norm(observation_xy - target_xy))
        motion_error = float(np.linalg.norm(motion_xy - target_xy))
        search_error = float(np.linalg.norm(search_xy - target_xy))
        motion_helpful = bool(
            motion_valid and motion_error + margin <= observation_error)
        search_helpful = bool(
            search_valid and search_error + margin <= observation_error)
        weight_key = (
            "b3_applied_weight" if "b3_applied_weight" in output
            else "advantage_applied_weight")
        motion_weight = self._proposal_scalar(
            output, weight_key, column=0)
        search_weight = self._proposal_scalar(
            output, weight_key, column=1)

        row = {
            "frame_id": int(frame_id),
            "proposal_mode_id": int(self._proposal_scalar(
                output, "proposal_inference_mode_id", default=-1)),
            "target_x": float(target_xy[0]),
            "target_y": float(target_xy[1]),
            "observation_x": float(observation_xy[0]),
            "observation_y": float(observation_xy[1]),
            "motion_x": float(motion_xy[0]),
            "motion_y": float(motion_xy[1]),
            "search_x": float(search_xy[0]),
            "search_y": float(search_xy[1]),
            "final_x": float(final_xy[0]),
            "final_y": float(final_xy[1]),
            "observation_error": observation_error,
            "motion_error": motion_error if motion_valid else float("nan"),
            "search_error": search_error if search_valid else float("nan"),
            "final_error": float(np.linalg.norm(final_xy - target_xy)),
            "motion_valid": int(motion_valid),
            "search_valid": int(search_valid),
            "motion_helpful": int(motion_helpful),
            "search_helpful": int(search_helpful),
            "motion_help_probability": self._proposal_scalar(
                output, "advantage_help_probability", column=0),
            "search_help_probability": self._proposal_scalar(
                output, "advantage_help_probability", column=1),
            "motion_step_ratio": self._proposal_scalar(
                output, "advantage_step_ratio", column=0),
            "search_step_ratio": self._proposal_scalar(
                output, "advantage_step_ratio", column=1),
            "b3_motion_q10": self._proposal_scalar(
                output, "b3_gain_quantiles", column=0),
            "b3_motion_q50": self._proposal_scalar(
                output, "b3_gain_quantiles", column=1),
            "b3_search_q10": self._proposal_scalar(
                output, "b3_gain_quantiles", column=2),
            "b3_search_q50": self._proposal_scalar(
                output, "b3_gain_quantiles", column=3),
            "b3_selected_index": int(self._proposal_scalar(
                output, "b3_selected_index", default=0)),
            "b3_abstained": self._proposal_scalar(
                output, "b3_abstained", default=1.0),
            "motion_weight": motion_weight,
            "search_weight": search_weight,
            "search_materially_selected": int(search_weight >= 0.1),
            "correction_norm": float(np.linalg.norm(final_xy - observation_xy)),
            "geometry_valid": int(self._proposal_scalar(
                data_dict, "search_v21_geometry_valid") > 0.0),
            "available_count": self._proposal_scalar(
                data_dict, "search_v21_available_count"),
            "extension_count": self._proposal_scalar(
                data_dict, "search_v21_extension_count"),
            "overlap_count": self._proposal_scalar(
                data_dict, "search_v21_overlap_count"),
            "targetness_mean": self._proposal_scalar(
                output, "search_v21_targetness_mean"),
            "targetness_max": self._proposal_scalar(
                output, "search_v21_targetness_max"),
            "targetness_entropy": self._proposal_scalar(
                output, "search_v21_targetness_entropy"),
            "effective_sample_size": self._proposal_scalar(
                output, "search_v21_effective_sample_size"),
            "extension_weight_ratio": self._proposal_scalar(
                output, "search_v21_extension_weight_ratio"),
        }
        return row

    def _build_b3_rollout_row(
            self, output, this_box, reference_box, frame_id):
        """Build one GT-labelled CRPA training row outside model forward."""
        target_box = points_utils.transform_box(this_box, reference_box)
        target_xy = np.asarray(target_box.center[:2], dtype=np.float32)
        observation_xy = output[
            "observation_aux_estimation_boxes"
        ].detach().cpu().numpy().reshape(-1, 4)[0, :2].astype(np.float32)
        router_features = output[
            "b3_router_features"
        ].detach().cpu().numpy().reshape(1, -1)[0].astype(np.float32)
        residual = output[
            "b3_candidate_residual_xy"
        ].detach().cpu().numpy().reshape(2, 2).astype(np.float32)
        valid = output[
            "b3_candidate_valid"
        ].detach().cpu().numpy().reshape(2).astype(np.float32)
        step_cap = float(self._proposal_scalar(output, "b3_step_cap"))
        radius = float(self._proposal_scalar(output, "b3_fusion_radius"))
        target_offset = target_xy - observation_xy
        numerator = np.sum(residual * target_offset[None, :], axis=1)
        denominator = np.sum(residual * residual, axis=1) + 1e-6
        oracle_alpha = np.clip(numerator / denominator, 0.0, step_cap) * valid
        oracle_xy = observation_xy[None, :] + oracle_alpha[:, None] * residual
        observation_error = float(np.linalg.norm(observation_xy - target_xy))
        oracle_error = np.linalg.norm(
            oracle_xy - target_xy[None, :], axis=1).astype(np.float32)
        oracle_gain = np.maximum(
            observation_error - oracle_error, 0.0).astype(np.float32) * valid
        oracle_step_ratio = (
            oracle_alpha / max(step_cap, 1e-6)).astype(np.float32) * valid
        return {
            "frame_id": np.int64(frame_id),
            "router_features": router_features,
            "observation_xy": observation_xy,
            "target_xy": target_xy,
            "candidate_residual_xy": residual,
            "candidate_valid": valid,
            "step_cap": np.float32(step_cap),
            "fusion_radius": np.float32(radius),
            "oracle_gain": oracle_gain,
            "oracle_alpha": oracle_alpha.astype(np.float32),
            "oracle_step_ratio": oracle_step_ratio,
            "observation_error": np.float32(observation_error),
            "oracle_candidate_error": oracle_error,
        }

    def _build_v22_proposal_diagnostic_row(
            self, output, data_dict, this_box, reference_box, frame_id,
            previous_target_box=None):
        """Build a GT-labelled v2.2/v3 attribution row outside forward."""
        is_v3 = "search_v3_evidence_components" in output
        data_prefix = "search_v3" if is_v3 else "search_v22"
        target_box = points_utils.transform_box(this_box, reference_box)
        target_xy = np.asarray(target_box.center[:2], dtype=np.float64)
        sampled_points = data_dict.get(f"{data_prefix}_points")
        sampled_mask = data_dict.get(f"{data_prefix}_point_valid_mask")
        foreground_count = 0
        if sampled_points is not None and sampled_mask is not None:
            points_np = sampled_points.detach().cpu().numpy()[0, :, :3]
            mask_np = sampled_mask.detach().cpu().numpy().reshape(-1) > 0
            foreground = geometry_utils.points_in_box(
                target_box, points_np.T, self.config.bb_scale)
            foreground_count = int(np.sum(foreground & mask_np))
        baseline_foreground_count = 0
        baseline_points = data_dict.get("points")
        if baseline_points is not None:
            sample_size = int(getattr(
                self.config, "point_sample_size", 0))
            if sample_size > 0:
                baseline_np = baseline_points.detach().cpu().numpy()[
                    0, -sample_size:, :3]
                baseline_foreground_count = int(np.sum(
                    geometry_utils.points_in_box(
                        target_box, baseline_np.T, self.config.bb_scale)))

        def tensor_xy(key, fallback):
            value = output.get(key)
            if value is None:
                return np.asarray(fallback, dtype=np.float64)
            return value.detach().cpu().numpy().reshape(-1, 4 if value.shape[-1] == 4 else 2)[0, :2].astype(np.float64)

        observation_box4 = output[
            "observation_aux_estimation_boxes"
        ].detach().cpu().numpy().reshape(-1, 4)[0].astype(np.float64)
        observation_xy = observation_box4[:2]
        motion_xy = tensor_xy("motion_prior_xy", observation_xy)
        raw_search_xy = tensor_xy(
            "search_v3_raw_vote_xy" if is_v3 else "search_raw_vote_xy",
            observation_xy)
        refined_xy = tensor_xy(
            ("motion_search_v3_refined_xy"
             if is_v3 else "motion_search_refined_xy"), observation_xy)
        final_xy = tensor_xy("aux_estimation_boxes", observation_xy)

        def candidate_quality(candidate_xy):
            offset = observation_box4.copy()
            offset[:2] = np.asarray(candidate_xy, dtype=np.float64)
            candidate_box = points_utils.getOffsetBB(
                reference_box, offset, degrees=self.config.degrees,
                use_z=self.config.use_z, limit_box=self.config.limit_box)
            return (
                float(estimateOverlap(
                    this_box, candidate_box, dim=self.config.IoU_space,
                    up_axis=self.config.up_axis)),
                float(estimateAccuracy(
                    this_box, candidate_box, dim=self.config.IoU_space,
                    up_axis=self.config.up_axis)),
            )

        formal_diagnostics = bool(
            is_v3 and getattr(
                self.config, "use_asymmetric_dual_query", False))
        if formal_diagnostics:
            observation_iou, observation_distance = candidate_quality(
                observation_xy)
            raw_search_iou, raw_search_distance = candidate_quality(
                raw_search_xy)
            final_iou, final_distance = candidate_quality(final_xy)
        observation_error = float(np.linalg.norm(observation_xy - target_xy))
        motion_error = float(np.linalg.norm(motion_xy - target_xy))
        raw_search_error = float(np.linalg.norm(raw_search_xy - target_xy))
        refined_error = float(np.linalg.norm(refined_xy - target_xy))
        final_error = float(np.linalg.norm(final_xy - target_xy))
        official_search_is_raw = bool(self._proposal_scalar(
            output, "official_search_is_raw", default=float(is_v3)) > 0.5)
        margin = float(getattr(self.config, "advantage_help_margin", 0.05))
        motion_valid = self._proposal_scalar(
            output, "motion_prior_valid") > 0.0
        refined_valid = self._proposal_scalar(
            output,
            ("motion_search_v3_candidate_structural_valid"
             if is_v3 else "motion_search_candidate_valid")) > 0.0
        selected = int(self._proposal_scalar(
            output,
            ("router_v3_selected_candidate"
             if is_v3 else "signed_selected_candidate"),
            default=0.0))
        alpha = self._proposal_scalar(
            output,
            ("router_v3_applied_alpha"
             if is_v3 else "signed_applied_alpha"),
            default=0.0)
        motion_helpful = bool(
            motion_valid and motion_error + margin <= observation_error)
        raw_search_helpful = bool(
            refined_valid
            and raw_search_error + margin <= observation_error)
        refined_helpful = bool(
            refined_valid and refined_error + margin <= observation_error)
        intervention = selected in (1, 2) and alpha > 0
        selected_error = (
            motion_error if selected == 1
            else (raw_search_error
                  if official_search_is_raw else refined_error)
            if selected == 2
            else observation_error)
        previous_error = float("nan")
        if previous_target_box is not None:
            previous_error = float(np.linalg.norm(
                np.asarray(reference_box.center, dtype=np.float64)[:2]
                - np.asarray(
                    previous_target_box.center, dtype=np.float64)[:2]))
        sigma_mahalanobis_sq = float("nan")
        if is_v3 and motion_valid:
            direction_value = output.get("motion_prior_direction_xy")
            log_sigma_value = output.get(
                "motion_prior_log_sigma_parallel_perp")
            if direction_value is not None and log_sigma_value is not None:
                direction_xy = direction_value.detach().cpu().numpy(
                    ).reshape(-1, 2)[0]
                perpendicular_xy = np.asarray(
                    [-direction_xy[1], direction_xy[0]])
                motion_residual = target_xy - motion_xy
                aligned_error = np.asarray((
                    np.dot(motion_residual, direction_xy),
                    np.dot(motion_residual, perpendicular_xy),
                ))
                log_sigma_pp = log_sigma_value.detach().cpu().numpy(
                    ).reshape(-1, 2)[0]
                sigma_mahalanobis_sq = float(np.sum(
                    (aligned_error * np.exp(-log_sigma_pp)) ** 2))
        row = {
            "frame_id": int(frame_id),
            "b2_version": "v3" if is_v3 else "v2.2",
            "geometry_valid": int(self._proposal_scalar(
                data_dict, f"{data_prefix}_geometry_valid") > 0.0),
            "foreground_count": foreground_count,
            "baseline_foreground_count": baseline_foreground_count,
            "base_reachable": int(baseline_foreground_count >= 1),
            "prior_reachable": int(foreground_count >= 1),
            "valid_foreground": int(
                refined_valid and foreground_count >= 1),
            "motion_valid": int(motion_valid),
            "search_valid": int(refined_valid),
            "observation_error": observation_error,
            "motion_error": motion_error,
            "raw_search_error": raw_search_error,
            "legacy_clipped_error": refined_error,
            "search_error": (
                raw_search_error if is_v3 else refined_error),
            "final_error": final_error,
            "previous_error": previous_error,
            "motion_helpful": int(motion_helpful),
            "raw_search_helpful": int(raw_search_helpful),
            "legacy_clipped_helpful": int(refined_helpful),
            "search_helpful": int(
                raw_search_helpful if is_v3 else refined_helpful),
            "utility_target": int(raw_search_helpful),
            "search_materially_selected": int(selected == 2 and alpha > 0),
            "search_weight": alpha if selected == 2 else 0.0,
            "intervention": int(intervention),
            "selected_helpful": int(
                (selected == 1 and motion_helpful)
                or (selected == 2 and (
                    raw_search_helpful
                    if official_search_is_raw else refined_helpful))),
            "selected_harmful": int(
                intervention and selected_error > observation_error),
            "selected_candidate": selected,
            "selected_step_ratio": self._proposal_scalar(
                output,
                ("router_v3_selected_step_ratio"
                 if is_v3 else "signed_selected_step_ratio")),
            "step_cap": self._proposal_scalar(
                output,
                "router_v3_step_cap" if is_v3 else "signed_step_cap"),
            "gain_threshold": self._proposal_scalar(
                output,
                ("router_v3_gain_threshold"
                 if is_v3 else "signed_gain_threshold")),
            "abstained": self._proposal_scalar(
                output,
                ("router_v3_abstained"
                 if is_v3 else "signed_abstained"), default=1.0),
            "motion_q10": self._proposal_scalar(
                output,
                ("router_v3_gain_q10"
                 if is_v3 else "signed_gain_quantiles"), column=0),
            "motion_q50": self._proposal_scalar(
                output,
                ("router_v3_gain_q50"
                 if is_v3 else "signed_gain_quantiles"),
                column=0 if is_v3 else 1),
            "motion_search_q10": self._proposal_scalar(
                output,
                ("router_v3_gain_q10"
                 if is_v3 else "signed_gain_quantiles"),
                column=3 if is_v3 else 2),
            "motion_search_q50": self._proposal_scalar(
                output,
                ("router_v3_gain_q50"
                 if is_v3 else "signed_gain_quantiles"),
                column=3),
            "presence_probability": self._proposal_scalar(
                output, ("search_v3_presence_probability"
                         if is_v3 else "search_presence_probability")),
            "utility_probability": self._proposal_scalar(
                output, "search_v3_utility_probability"),
            "presence_target": int(foreground_count >= 1),
            "prior_source_id": int(self._proposal_scalar(
                data_dict, f"{data_prefix}_prior_source_id")),
            "gap_ratio": self._proposal_scalar(
                data_dict, f"{data_prefix}_gap_ratio", default=1.0),
            "support_truncated": int(self._proposal_scalar(
                data_dict, f"{data_prefix}_support_truncated") > 0.0),
            "support_requested_length": self._proposal_scalar(
                data_dict, f"{data_prefix}_support_requested_extent",
                column=0),
            "support_requested_width": self._proposal_scalar(
                data_dict, f"{data_prefix}_support_requested_extent",
                column=1),
            "support_actual_length": self._proposal_scalar(
                data_dict, f"{data_prefix}_support_actual_extent",
                column=0),
            "support_actual_width": self._proposal_scalar(
                data_dict, f"{data_prefix}_support_actual_extent",
                column=1),
            "sigma_mahalanobis_sq": sigma_mahalanobis_sq,
            "sigma_coverage_50": int(
                np.isfinite(sigma_mahalanobis_sq)
                and sigma_mahalanobis_sq <= 1.38629436112),
            "sigma_coverage_80": int(
                np.isfinite(sigma_mahalanobis_sq)
                and sigma_mahalanobis_sq <= 3.21887582487),
            "sigma_coverage_95": int(
                np.isfinite(sigma_mahalanobis_sq)
                and sigma_mahalanobis_sq <= 5.99146454711),
            "normalized_ess": self._proposal_scalar(
                output, "search_normalized_ess"),
            "raw_ess": self._proposal_scalar(output, "search_raw_ess"),
            "available_count": self._proposal_scalar(
                data_dict, f"{data_prefix}_available_count"),
            "extension_count": self._proposal_scalar(
                data_dict, f"{data_prefix}_extension_count"),
            "overlap_count": self._proposal_scalar(
                data_dict, f"{data_prefix}_overlap_count"),
            "correction_x": self._proposal_scalar(
                output,
                ("router_v3_correction_xy"
                 if is_v3 else "signed_correction_xy"), column=0),
            "correction_y": self._proposal_scalar(
                output,
                ("router_v3_correction_xy"
                 if is_v3 else "signed_correction_xy"), column=1),
        }
        if formal_diagnostics:
            row.update({
                "observation_x": float(observation_box4[0]),
                "observation_y": float(observation_box4[1]),
                "observation_z": float(observation_box4[2]),
                "observation_yaw": float(observation_box4[3]),
                "observation_iou": observation_iou,
                "observation_distance": observation_distance,
                "raw_search_iou": raw_search_iou,
                "raw_search_distance": raw_search_distance,
                "final_iou": final_iou,
                "final_distance": final_distance,
            })
        return row

    def _build_ct_joint_diagnostic_row(
            self, output, data_dict, this_box, reference_box, frame_id):
        """Export paper-facing joint-Full diagnostics; GT stays outside forward."""
        target_box = points_utils.transform_box(this_box, reference_box)
        target_xy = np.asarray(target_box.center[:2], dtype=np.float64)

        def xy(key, fallback):
            value = output.get(key)
            if value is None:
                return np.asarray(fallback, dtype=np.float64)
            return value.detach().cpu().numpy().reshape(-1, value.shape[-1])[
                0, :2].astype(np.float64)

        observation = xy("observation_aux_estimation_boxes", (0.0, 0.0))
        kinematic = xy("motion_prior_kinematic_xy", observation)
        learned = xy("motion_prior_xy", kinematic)
        raw_search = xy("ct_search_unmasked_raw_xy", observation)
        raw_obs = xy("ct_search_raw_obs_xy", raw_search)
        raw_motion = xy("ct_search_raw_motion_xy", raw_search)
        raw_alpha = xy("ct_search_raw_alpha_xy", raw_search)
        final = xy("aux_estimation_boxes", observation)
        observation_local4 = output[
            "observation_aux_estimation_boxes"
        ].detach().cpu().numpy().reshape(-1, 4)[0]
        final_local4 = output[
            "aux_estimation_boxes"
        ].detach().cpu().numpy().reshape(-1, 4)[0]
        observation_world = points_utils.getOffsetBB(
            reference_box, observation_local4, degrees=self.config.degrees,
            use_z=self.config.use_z, limit_box=self.config.limit_box)
        final_world = points_utils.getOffsetBB(
            reference_box, final_local4, degrees=self.config.degrees,
            use_z=self.config.use_z, limit_box=self.config.limit_box)
        raw_search_local4 = observation_local4.copy()
        raw_search_local4[:2] = raw_search
        raw_search_world = points_utils.getOffsetBB(
            reference_box, raw_search_local4, degrees=self.config.degrees,
            use_z=self.config.use_z, limit_box=self.config.limit_box)
        router_gate_applied = self._proposal_scalar(
            output, "ct_router_applied_gate")
        router_residual = xy(
            "ct_router_bounded_residual_xy", (0.0, 0.0))
        bounded_local4 = observation_local4.copy()
        bounded_local4[:2] = observation + router_residual
        bounded_world = points_utils.getOffsetBB(
            reference_box, bounded_local4, degrees=self.config.degrees,
            use_z=self.config.use_z, limit_box=self.config.limit_box)
        selective_local4 = observation_local4.copy()
        selective_local4[:2] = (
            observation + router_gate_applied * router_residual)
        selective_world = points_utils.getOffsetBB(
            reference_box, selective_local4, degrees=self.config.degrees,
            use_z=self.config.use_z, limit_box=self.config.limit_box)

        def foreground_count(points_key, mask_key, source_key=None):
            points = data_dict.get(points_key)
            mask = data_dict.get(mask_key)
            if points is None or mask is None:
                return 0
            points_np = points.detach().cpu().numpy()[0, :, :3]
            mask_np = mask.detach().cpu().numpy().reshape(-1) > 0
            foreground = geometry_utils.points_in_box(
                target_box, points_np.T, self.config.bb_scale)
            if source_key is not None:
                source = data_dict.get(source_key)
                if source is None:
                    return 0
                source_np = source.detach().cpu().numpy().reshape(-1) > 0
                foreground = foreground & source_np
            return int(np.sum(foreground & mask_np))

        endpoint_foreground = foreground_count(
            "search_v3_points", "search_v3_point_valid_mask")
        tube_foreground = foreground_count(
            "trajectory_search_points",
            "trajectory_search_point_valid_mask")
        endpoint_extension_foreground = foreground_count(
            "search_v3_points", "search_v3_point_valid_mask",
            "search_v3_point_source")
        tube_extension_foreground = foreground_count(
            "trajectory_search_points",
            "trajectory_search_point_valid_mask",
            "trajectory_search_point_source")
        sigma = output.get("motion_prior_log_sigma_parallel_perp")
        sigma_np = (
            np.exp(sigma.detach().cpu().numpy().reshape(-1, 2)[0])
            if sigma is not None else np.asarray((np.nan, np.nan)))
        residual_unit = output.get(
            "motion_prior_residual_unit_parallel_perp")
        residual_unit_np = (
            residual_unit.detach().cpu().numpy().reshape(-1, 2)[0]
            if residual_unit is not None else np.zeros(2, dtype=np.float32))
        b1_valid = bool(self._proposal_scalar(
            output, "motion_prior_valid") > 0.0)
        b1_nll = float("nan")
        b1_mahalanobis_sq = float("nan")
        direction_np = np.asarray((1.0, 0.0), dtype=np.float64)
        direction = output.get("motion_prior_direction_xy")
        log_sigma = output.get("motion_prior_log_sigma_parallel_perp")
        if b1_valid and direction is not None and log_sigma is not None:
            direction_np = direction.detach().cpu().numpy().reshape(-1, 2)[0]
            direction_norm = float(np.linalg.norm(direction_np))
            log_sigma_np = log_sigma.detach().cpu().numpy().reshape(-1, 2)[0]
            if (np.isfinite(direction_norm) and direction_norm > 1e-8
                    and np.isfinite(log_sigma_np).all()):
                direction_np = direction_np / direction_norm
                perpendicular_np = np.asarray(
                    (-direction_np[1], direction_np[0]), dtype=np.float64)
                learned_error_xy = target_xy - learned
                aligned_error = np.asarray((
                    np.dot(learned_error_xy, direction_np),
                    np.dot(learned_error_xy, perpendicular_np),
                ), dtype=np.float64)
                safe_log_sigma = np.clip(log_sigma_np, -4.0, 2.5)
                b1_nll = float(np.sum(0.5 * (
                    aligned_error ** 2 * np.exp(-2.0 * safe_log_sigma)
                    + 2.0 * safe_log_sigma)))
                b1_mahalanobis_sq = float(np.sum(
                    aligned_error ** 2 * np.exp(-2.0 * safe_log_sigma)))
                if not np.isfinite(b1_nll):
                    b1_nll = float("nan")
                if not np.isfinite(b1_mahalanobis_sq):
                    b1_mahalanobis_sq = float("nan")
        perpendicular_np = np.asarray(
            (-direction_np[1], direction_np[0]), dtype=np.float64)
        envelope = output.get("motion_prior_envelope_parallel_perp")
        envelope_np = (
            envelope.detach().cpu().numpy().reshape(-1, 2)[0]
            if envelope is not None else np.ones(2, dtype=np.float32))
        target_residual_xy = target_xy - kinematic
        target_residual_unit_np = np.asarray((
            np.dot(target_residual_xy, direction_np),
            np.dot(target_residual_xy, perpendicular_np),
        ), dtype=np.float64) / np.maximum(envelope_np, 1e-6)
        raw_obs_error = float(np.linalg.norm(raw_obs - target_xy))
        raw_motion_error = float(np.linalg.norm(raw_motion - target_xy))
        counterfactual_margin = float(getattr(
            self.config, 'ct_query_counterfactual_margin', 0.05))
        motion_helpful = bool(
            raw_motion_error + counterfactual_margin < raw_obs_error)
        motion_harmful = bool(
            raw_obs_error + counterfactual_margin < raw_motion_error)
        observation_error = float(np.linalg.norm(observation - target_xy))
        bounded_error = float(np.linalg.norm(
            bounded_local4[:2] - target_xy))
        observation_iou = float(estimateOverlap(
            this_box, observation_world, dim=self.config.IoU_space,
            up_axis=self.config.up_axis))
        bounded_iou = float(estimateOverlap(
            this_box, bounded_world, dim=self.config.IoU_space,
            up_axis=self.config.up_axis))
        return {
            "frame_id": int(frame_id),
            "b2_version": "ct_joint_full",
            "candidate_id": int(self._proposal_scalar(
                data_dict, "candidate_id", default=0.0)),
            "base_target_count": self._proposal_scalar(
                data_dict, "ct_acquisition_base_target_count"),
            "pool_target_count": self._proposal_scalar(
                data_dict,
                "ct_acquisition_extension_pool_target_count"),
            "sampled_target_count": self._proposal_scalar(
                data_dict, "ct_acquisition_sampled_target_count"),
            "extension_pool_count": self._proposal_scalar(
                data_dict, "ct_acquisition_extension_pool_count"),
            "sampled_count": self._proposal_scalar(
                data_dict, "ct_acquisition_sampled_count"),
            "target_in_support": int(self._proposal_scalar(
                data_dict,
                "ct_acquisition_extension_pool_target_count") > 0.0),
            "current_target_points": self._proposal_scalar(
                data_dict, "ct_acquisition_base_target_count"),
            "recursive_age": self._proposal_scalar(
                data_dict, "ct_recursive_state_age",
                default=-1.0),
            "recursive_age_valid": int(self._proposal_scalar(
                data_dict, "ct_recursive_state_age_valid",
                default=0.0) > 0.0),
            "support_actual_length": self._proposal_scalar(
                data_dict, "search_v3_support_actual_extent", column=0),
            "support_actual_width": self._proposal_scalar(
                data_dict, "search_v3_support_actual_extent", column=1),
            "support_volume": (
                self._proposal_scalar(
                    data_dict, "search_v3_support_actual_extent", column=0)
                * self._proposal_scalar(
                    data_dict, "search_v3_support_actual_extent", column=1)
                * self._proposal_scalar(
                    data_dict, "bbox_size", column=2)),
            "available": int(self._proposal_scalar(
                output, "ct_b2_available") > 0.0),
            "structural_available": int(self._proposal_scalar(
                output, "ct_b2_available") > 0.0),
            "recovery_positive": int(self._proposal_scalar(
                data_dict, "ct_recovery_positive") > 0.0),
            "recovery_fallback": int(self._proposal_scalar(
                data_dict, "ct_recovery_fallback") > 0.0),
            "query_delta_t": self._proposal_scalar(
                data_dict, "search_v3_query_delta_t"),
            "gap_ratio": self._proposal_scalar(
                data_dict, "search_v3_gap_ratio", default=1.0),
            "endpoint_foreground_count": endpoint_foreground,
            "tube_foreground_count": tube_foreground,
            "foreground_count": endpoint_foreground + tube_foreground,
            "extension_foreground_count": (
                endpoint_extension_foreground
                + tube_extension_foreground),
            "search_valid": int(self._proposal_scalar(
                output, "ct_search_candidate_valid") > 0.0),
            "search_geometry_valid": int(self._proposal_scalar(
                data_dict, "ct_search_geometry_valid") > 0.0),
            "search_new_support_valid": int(self._proposal_scalar(
                output, "ct_search_new_support_valid") > 0.0),
            "search_available": int(self._proposal_scalar(
                output, "ct_search_available") > 0.0),
            "search_geometry_source_id": int(self._proposal_scalar(
                data_dict, "search_v3_prior_source_id")),
            "search_quality_valid": int(self._proposal_scalar(
                data_dict, "ct_search_quality_valid") > 0.0),
            "search_coverage_need": int(self._proposal_scalar(
                data_dict, "ct_search_coverage_need") > 0.0),
            "search_total_point_count": self._proposal_scalar(
                data_dict, "ct_search_total_point_count"),
            "search_extension_count": self._proposal_scalar(
                data_dict, "ct_search_extension_count"),
            "search_extension_voxels": self._proposal_scalar(
                data_dict, "ct_search_extension_voxels"),
            "observation_error": observation_error,
            "observation_iou": observation_iou,
            "observation_distance": float(estimateAccuracy(
                this_box, observation_world, dim=self.config.IoU_space,
                up_axis=self.config.up_axis)),
            "kinematic_error": float(np.linalg.norm(
                kinematic - target_xy)),
            "learned_motion_error": float(np.linalg.norm(
                learned - target_xy)),
            "b1_valid": int(b1_valid and np.isfinite(b1_nll)),
            "b1_nll": b1_nll,
            "b1_mahalanobis_sq": b1_mahalanobis_sq,
            "b1_coverage_50": int(
                np.isfinite(b1_mahalanobis_sq)
                and b1_mahalanobis_sq <= 1.38629436112),
            "b1_coverage_80": int(
                np.isfinite(b1_mahalanobis_sq)
                and b1_mahalanobis_sq <= 3.21887582487),
            "b1_coverage_95": int(
                np.isfinite(b1_mahalanobis_sq)
                and b1_mahalanobis_sq <= 5.99146454711),
            "raw_search_error": float(np.linalg.norm(
                raw_search - target_xy)),
            "raw_search_iou": float(estimateOverlap(
                this_box, raw_search_world, dim=self.config.IoU_space,
                up_axis=self.config.up_axis)),
            "raw_search_distance": float(estimateAccuracy(
                this_box, raw_search_world, dim=self.config.IoU_space,
                up_axis=self.config.up_axis)),
            "selective_error": float(np.linalg.norm(
                selective_local4[:2] - target_xy)),
            "selective_iou": float(estimateOverlap(
                this_box, selective_world, dim=self.config.IoU_space,
                up_axis=self.config.up_axis)),
            "selective_distance": float(estimateAccuracy(
                this_box, selective_world, dim=self.config.IoU_space,
                up_axis=self.config.up_axis)),
            "raw_obs_error": raw_obs_error,
            "raw_motion_error": raw_motion_error,
            "raw_alpha_error": float(np.linalg.norm(
                raw_alpha - target_xy)),
            "alpha_counterfactual_uplift": (
                raw_obs_error - raw_motion_error),
            "alpha_motion_helpful": int(motion_helpful),
            "alpha_motion_harmful": int(motion_harmful),
            "alpha_ambiguous": int(not (
                motion_helpful or motion_harmful)),
            "final_error": float(np.linalg.norm(final - target_xy)),
            "final_iou": float(estimateOverlap(
                this_box, final_world, dim=self.config.IoU_space,
                up_axis=self.config.up_axis)),
            "final_distance": float(estimateAccuracy(
                this_box, final_world, dim=self.config.IoU_space,
                up_axis=self.config.up_axis)),
            "query_gate": self._proposal_scalar(
                output, "ct_query_gate_probability"),
            "query_gate_applied": self._proposal_scalar(
                output, "ct_query_gate_internal"),
            "query_shift_norm": self._proposal_scalar(
                output, "ct_query_shift_norm"),
            "router_gate": self._proposal_scalar(
                output, "ct_router_gate"),
            "action_score": self._proposal_scalar(
                output, "ct_b3_action_score"),
            "helpful_probability": self._proposal_scalar(
                output, "ct_b3_help_probability"),
            "harmful_probability": self._proposal_scalar(
                output, "ct_b3_harm_probability"),
            "expected_center_gain": self._proposal_scalar(
                output, "ct_b3_expected_center_gain"),
            "expected_iou_gain": self._proposal_scalar(
                output, "ct_b3_expected_iou_gain"),
            "center_gain": observation_error - bounded_error,
            "iou_gain": bounded_iou - observation_iou,
            "bounded_action_error": bounded_error,
            "bounded_action_iou": bounded_iou,
            "b3_calibrated": self._proposal_scalar(
                output, "ct_b3_calibrated"),
            "router_applied_gate": router_gate_applied,
            "router_evidence_valid": self._proposal_scalar(
                output, "ct_router_evidence_valid"),
            "router_radius": self._proposal_scalar(
                output, "ct_router_radius"),
            "residual_unit_parallel": float(residual_unit_np[0]),
            "residual_unit_perpendicular": float(residual_unit_np[1]),
            "target_residual_unit_parallel": float(
                target_residual_unit_np[0]),
            "target_residual_unit_perpendicular": float(
                target_residual_unit_np[1]),
            "residual_recoverable_parallel": int(
                abs(target_residual_unit_np[0]) <= 1.0),
            "residual_recoverable_perpendicular": int(
                abs(target_residual_unit_np[1]) <= 1.0),
            "residual_saturation": self._proposal_scalar(
                output, "ct_motion_residual_saturation"),
            "sigma_parallel": float(sigma_np[0]),
            "sigma_perpendicular": float(sigma_np[1]),
            "targetness_mean": self._proposal_scalar(
                output, "ct_search_targetness_mean"),
            "targetness_max": self._proposal_scalar(
                output, "ct_search_targetness_max"),
            "targetness_entropy": self._proposal_scalar(
                output, "ct_search_targetness_entropy"),
            "normalized_ess": self._proposal_scalar(
                output, "ct_search_normalized_ess"),
            "extension_mass_ratio": self._proposal_scalar(
                output, "ct_search_extension_mass_ratio"),
            "extension_vote_rms": self._proposal_scalar(
                output, "ct_search_extension_vote_rms"),
            "presence_probability": self._proposal_scalar(
                output, "ct_search_presence_probability"),
            "presence_score": self._proposal_scalar(
                output, "ct_search_presence_probability"),
            "presence_target": int(
                endpoint_extension_foreground
                + tube_extension_foreground >= 1),
        }

    @staticmethod
    def _write_csv_rows(path, rows):
        if not rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_proposal_test_diagnostics(self):
        rows = self._proposal_test_diagnostics
        if not rows or int(getattr(self, "global_rank", 0)) != 0:
            return
        logger = getattr(self, "logger", None)
        log_dir = getattr(logger, "log_dir", None)
        if log_dir is None:
            save_dir = getattr(logger, "save_dir", ".")
            log_dir = Path(save_dir) / "proposal_diagnostics"
        output_dir = Path(log_dir) / "proposal_diagnostics"
        self._write_csv_rows(output_dir / "proposal_endpoints.csv", rows)

        tracklet_rows = []
        tracklet_ids = sorted({int(row["tracklet_id"]) for row in rows})
        for tracklet_id in tracklet_ids:
            group = [
                row for row in rows
                if int(row["tracklet_id"]) == tracklet_id]

            def finite_mean(key, valid_key=None):
                values = [
                    float(row[key]) for row in group
                    if (valid_key is None or bool(row[valid_key]))
                    and np.isfinite(float(row[key]))]
                return float(np.mean(values)) if values else float("nan")

            # Joint Full exports a hard observation/Search action schema,
            # whereas the legacy v2/v3 diagnostics use soft proposal fields.
            # Keep both schemas explicit: indexing legacy-only fields here made
            # an otherwise successful Joint Full test fail at test_epoch_end.
            if "router_applied_gate" in group[0]:
                selected = [
                    row for row in group
                    if bool(row["router_applied_gate"])]
                helpful_margin = float(getattr(
                    self.config, "ct_router_help_margin", 0.05))
                tracklet_rows.append({
                    "tracklet_id": tracklet_id,
                    "endpoint_count": len(group),
                    "search_valid_rate": finite_mean("search_valid"),
                    "router_applied_rate": finite_mean(
                        "router_applied_gate"),
                    "observation_error_mean": finite_mean(
                        "observation_error"),
                    "raw_search_error_mean": finite_mean(
                        "raw_search_error", "search_valid"),
                    "final_error_mean": finite_mean("final_error"),
                    "selected_helpful_precision": (
                        float(np.mean([
                            (float(row["observation_error"])
                             - float(row["final_error"]))
                            > helpful_margin
                            for row in selected]))
                        if selected else float("nan")),
                    "selected_harm_rate": (
                        float(np.mean([
                            (float(row["final_error"])
                             - float(row["observation_error"]))
                            > helpful_margin
                            for row in selected]))
                        if selected else float("nan")),
                })
                continue

            selected = [
                row for row in group if row["search_materially_selected"]]
            tracklet_rows.append({
                "tracklet_id": tracklet_id,
                "endpoint_count": len(group),
                "geometry_valid_rate": finite_mean("geometry_valid"),
                "motion_valid_rate": finite_mean("motion_valid"),
                "search_valid_rate": finite_mean("search_valid"),
                "observation_error_mean": finite_mean("observation_error"),
                "motion_error_mean": finite_mean(
                    "motion_error", "motion_valid"),
                "search_error_mean": finite_mean(
                    "search_error", "search_valid"),
                "final_error_mean": finite_mean("final_error"),
                "search_helpful_prevalence": finite_mean(
                    "search_helpful", "search_valid"),
                "search_materially_selected_rate": finite_mean(
                    "search_materially_selected"),
                "search_selected_helpful_precision": (
                    float(np.mean([
                        row["search_helpful"] for row in selected]))
                    if selected else float("nan")),
                "search_weight_mean": finite_mean("search_weight"),
            })
            if "intervention" in group[0]:
                interventions = [
                    row for row in group if bool(row["intervention"])]
                tracklet_rows[-1].update({
                    "intervention_rate": finite_mean("intervention"),
                    "selected_helpful_precision": (
                        float(np.mean([
                            row["selected_helpful"]
                            for row in interventions]))
                        if interventions else float("nan")),
                    "selected_harm_rate": (
                        float(np.mean([
                            row["selected_harmful"]
                            for row in interventions]))
                        if interventions else float("nan")),
                })
        self._write_csv_rows(
            output_dir / "proposal_tracklets.csv", tracklet_rows)

    def _write_tracking_test_endpoints(self):
        """Persist all metric endpoints, including frame 0 and empty crops."""
        rows = self._tracking_test_endpoints
        if not rows or int(getattr(self, "global_rank", 0)) != 0:
            return
        logger = getattr(self, "logger", None)
        log_dir = getattr(logger, "log_dir", None)
        if log_dir is None:
            save_dir = getattr(logger, "save_dir", ".")
            log_dir = Path(save_dir) / "proposal_diagnostics"
        output_dir = Path(log_dir) / "proposal_diagnostics"
        self._write_csv_rows(output_dir / "tracking_endpoints.csv", rows)

    def _write_b3_test_rollouts(self):
        rows = self._b3_test_rollouts
        if not rows or int(getattr(self, "global_rank", 0)) != 0:
            return
        logger = getattr(self, "logger", None)
        log_dir = getattr(logger, "log_dir", None)
        if log_dir is None:
            save_dir = getattr(logger, "save_dir", ".")
            log_dir = Path(save_dir) / "b3_rollouts"
        output_dir = Path(log_dir) / "b3_rollouts"
        output_dir.mkdir(parents=True, exist_ok=True)
        keys = [
            "frame_id", "router_features", "observation_xy", "target_xy",
            "candidate_residual_xy", "candidate_valid", "step_cap",
            "fusion_radius", "oracle_gain", "oracle_alpha",
            "oracle_step_ratio", "observation_error",
            "oracle_candidate_error",
        ]
        arrays = {
            key: np.stack([row[key] for row in rows], axis=0)
            for key in keys
        }
        arrays["tracklet_id"] = np.asarray(
            [row["tracklet_id"] for row in rows], dtype=np.int64)
        arrays["tracklet_key"] = np.asarray(
            [row["tracklet_key"] for row in rows], dtype=np.str_)
        np.savez_compressed(output_dir / "b3_rollouts.npz", **arrays)
        manifest = {
            "schema": "ct_seqtrack.b3_crpa_rollout.v1",
            "row_count": len(rows),
            "tracklet_count": len(set(arrays["tracklet_key"].tolist())),
            "router_feature_dim": int(arrays["router_features"].shape[1]),
            "proposal_inference_mode": str(getattr(
                self.config, "proposal_inference_mode", "full")),
            "seed": int(getattr(self.config, "seed", 42) or 42),
        }
        with (output_dir / "manifest.json").open(
                "w", encoding="utf-8") as output_file:
            json.dump(manifest, output_file, indent=2, sort_keys=True)
            output_file.write("\n")

    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []

        results_bbs = []
        proposal_diagnostics = []
        b3_rollouts = []
        recursive_state = None
        for frame_id in range(len(sequence)):  # tracklet
            if frame_id == 0:
                # the first frame
                this_bb = sequence[frame_id]["3d_bbox"]
                prev_bb = sequence[frame_id]["3d_bbox"]
                results_bbs.append(this_bb)
                recursive_state = RecursiveTrackState(
                    tracklet_id=0,
                    tracklet_key=str(sequence[0].get(
                        'tracklet_key', sequence[0].get('tracklet_id', 'eval'))),
                    first_box=this_bb,
                    timestamps={0: sequence[0].get('timestamp')},
                )
                new_refboxs = [prev_bb] # Update in special cases
            else:
                this_bb = sequence[frame_id]["3d_bbox"]

                # B1 is intentionally run before point cropping.  It only
                # consumes recursive history boxes and timestamps; the
                # current annotation is never passed to the pre-pass.
                motion_prediction = None
                if (bool(getattr(
                        self.config, "use_b1_prepass_support", False))
                        and bool(getattr(
                            self, "ct_enable_b1",
                            getattr(self.config, "ct_enable_b1", True)))):
                    predictor = getattr(self, "predict_motion_prepass", None)
                    if predictor is None:
                        raise RuntimeError(
                            "B1 pre-pass support requires a motion predictor")
                    motion_prediction = predictor(
                        sequence, frame_id, results_bbs)
                build_kwargs = {}
                if motion_prediction is not None:
                    build_kwargs["motion_prediction"] = motion_prediction
                data_dict, ref_bb = self.build_input_dict(
                    sequence, frame_id, recursive_state.results_bbs,
                    recursive_state=recursive_state,
                    **build_kwargs)
                # run the tracker
                if torch.sum(data_dict['points'][:,:,:3]) == 0:
                    results_bbs.append(ref_bb)
                    print("Empty pointcloud!")
                    new_refboxs = [ref_bb]
                else:
                    (candidate_box, _, forward_output) = (
                        self.evaluate_one_sample(data_dict, ref_box=ref_bb))
                    if (bool(getattr(
                            self.config,
                            "export_proposal_diagnostics",
                            False))
                            and "search_v21_proposal_xy" in forward_output):
                        proposal_diagnostics.append(
                            self._build_proposal_diagnostic_row(
                                forward_output,
                                data_dict,
                                this_bb,
                                ref_bb,
                                frame_id,
                            ))
                    if (bool(getattr(
                            self.config,
                            "export_proposal_diagnostics",
                            False))
                            and "motion_search_refined_xy" in forward_output):
                        proposal_diagnostics.append(
                            self._build_v22_proposal_diagnostic_row(
                                forward_output,
                                data_dict,
                                this_bb,
                                ref_bb,
                                frame_id,
                                previous_target_box=sequence[
                                    frame_id - 1]["3d_bbox"],
                            ))
                    if (bool(getattr(
                            self.config,
                            "export_proposal_diagnostics",
                            False))
                            and ("ct_search_raw_xy" in forward_output
                                 or bool(getattr(
                                     self.config,
                                     "use_ct_joint_full", False)))):
                        proposal_diagnostics.append(
                            self._build_ct_joint_diagnostic_row(
                                forward_output,
                                data_dict,
                                this_bb,
                                ref_bb,
                                frame_id,
                            ))
                    if (bool(getattr(
                            self.config, "export_b3_rollouts", False))
                            and "b3_router_features" in forward_output):
                        b3_rollouts.append(self._build_b3_rollout_row(
                            forward_output, this_bb, ref_bb, frame_id))
                    results_bbs.append(candidate_box)

            
            if frame_id > 0:
                recursive_state.append(
                    frame_id, results_bbs[-1],
                    sequence[frame_id].get('timestamp'))
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)

            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)

        self._proposal_sequence_diagnostics = proposal_diagnostics
        self._b3_sequence_rollouts = b3_rollouts
        return ious, distances, results_bbs

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        start_time = time.time()
        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        epoch_number = int(getattr(self, "current_epoch", 0)) + 1
        if (bool(getattr(
                self.config, "export_v3_candidate_diagnostics", False))
                and self._proposal_sequence_diagnostics):
            if not hasattr(self, "_v3_validation_proposal_diagnostics"):
                self._v3_validation_proposal_diagnostics = []
            for row in self._proposal_sequence_diagnostics:
                row = dict(row)
                row["tracklet_id"] = int(batch_idx)
                row["epoch"] = epoch_number
                row["partition"] = "mini_val"
                self._v3_validation_proposal_diagnostics.append(row)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(sequence)

        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.success_step(torch.tensor(ious, device=self.device))
        self.prec_step(torch.tensor(distances, device=self.device))

        self.log('success/mini_val', self.success, on_epoch=True)
        self.log('precision/mini_val', self.prec, on_epoch=True)
        proposal_rows = getattr(self, '_proposal_sequence_diagnostics', [])
        b1_rows = [
            row for row in proposal_rows
            if bool(row.get('b1_valid', False))
            and np.isfinite(float(row.get('b1_nll', float('nan'))))
            and np.isfinite(float(row.get(
                'learned_motion_error', float('nan'))))
            and np.isfinite(float(row.get(
                'kinematic_error', float('nan'))))
        ]
        if b1_rows:
            b1_nll = torch.tensor(
                [float(row['b1_nll']) for row in b1_rows],
                device=self.device, dtype=torch.float32).mean()
            learned_mse = torch.tensor(
                [float(row['learned_motion_error']) ** 2
                 for row in b1_rows],
                device=self.device, dtype=torch.float32).mean()
            kinematic_mse = torch.tensor(
                [float(row['kinematic_error']) ** 2
                 for row in b1_rows],
                device=self.device, dtype=torch.float32).mean()
            self.log(
                'b1_nll/mini_val', b1_nll, on_step=False, on_epoch=True,
                batch_size=len(b1_rows))
            self.log(
                'b1_learned_motion_mse/mini_val', learned_mse,
                on_step=False, on_epoch=True, batch_size=len(b1_rows))
            self.log(
                'b1_kinematic_mse/mini_val', kinematic_mse,
                on_step=False, on_epoch=True, batch_size=len(b1_rows))
        if (hasattr(self, 'ct_observation_success') and proposal_rows
                and all('observation_iou' in row
                        and 'raw_search_iou' in row
                        for row in proposal_rows)):
            observation_ious = torch.tensor(
                [float(row['observation_iou']) for row in proposal_rows],
                device=self.device)
            raw_search_ious = torch.tensor(
                [float(row['raw_search_iou']) for row in proposal_rows],
                device=self.device)
            # Tracking initializes frame 0 from GT.  Include that endpoint so
            # observation/raw-search mini_val Success is directly comparable
            # with matched B0 and with the official sequence metric.
            frame0_iou = observation_ious.new_ones((1,))
            observation_ious = torch.cat((frame0_iou, observation_ious))
            raw_search_ious = torch.cat((frame0_iou, raw_search_ious))
            self.ct_observation_success(observation_ious)
            self.ct_raw_search_success(raw_search_ious)
            self.log(
                'success_observation/mini_val', self.ct_observation_success,
                on_epoch=True, batch_size=len(proposal_rows) + 1)
            self.log(
                'success_raw_search/mini_val', self.ct_raw_search_success,
                on_epoch=True, batch_size=len(proposal_rows) + 1)

        self.log('success/test_step', self.success_step, on_step=True, on_epoch=False)
        self.log('precision/test_step', self.prec_step, on_step=True, on_epoch=False)

        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))

        self.success_step.reset()
        self.prec_step.reset()

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/mini_val',
                                    {'success': self.success.compute(),
                                        'precision': self.prec.compute(),},
                                    global_step=self.global_step)

        self.logger.experiment.add_scalars('runtime',
                                       {'runtime':1.0/self.runtime.compute()},
                                       global_step=self.global_step)

        rows = getattr(self, "_v3_validation_proposal_diagnostics", [])
        if rows and int(getattr(self, "global_rank", 0)) == 0:
            logger = getattr(self, "logger", None)
            log_dir = getattr(logger, "log_dir", None)
            if log_dir is None:
                log_dir = getattr(logger, "save_dir", ".")
            epoch_number = int(getattr(self, "current_epoch", 0)) + 1
            self._write_csv_rows(
                Path(log_dir) / "candidate_diagnostics"
                / f"epoch_{epoch_number:02d}.csv",
                rows,
            )
        self._v3_validation_proposal_diagnostics = []


    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        start_time = time.time()
        ious, distances, result_bbs, *_= self.evaluate_one_sequence(sequence)
        test_dataset = getattr(
            getattr(self.trainer, "test_dataloaders", None), "dataset", None)
        if test_dataset is None:
            test_loaders = getattr(self.trainer, "test_dataloaders", None)
            if isinstance(test_loaders, (list, tuple)) and test_loaders:
                test_dataset = getattr(test_loaders[0], "dataset", None)
        if (test_dataset is not None
                and hasattr(test_dataset, "get_tracklet_key")):
            tracklet_key = test_dataset.get_tracklet_key(batch_idx)
        elif (test_dataset is not None
              and hasattr(test_dataset, "dataset")
              and hasattr(test_dataset.dataset, "get_tracklet_key")):
            source_index = int(batch_idx)
            if hasattr(test_dataset, "tracklet_indices"):
                source_index = int(test_dataset.tracklet_indices[batch_idx])
            tracklet_key = test_dataset.dataset.get_tracklet_key(
                source_index)
        else:
            tracklet_key = f"tracklet/{int(batch_idx)}"
        for frame_id, (overlap, distance) in enumerate(
                zip(ious, distances)):
            self._tracking_test_endpoints.append({
                "tracklet_id": int(batch_idx),
                "tracklet_key": str(tracklet_key),
                "frame_id": int(frame_id),
                "final_iou": float(overlap),
                "final_distance": float(distance),
            })
        for row in self._proposal_sequence_diagnostics:
            row = dict(row)
            row["tracklet_id"] = int(batch_idx)
            eval_partition = getattr(
                self.config, "ct_eval_partition", None)
            if eval_partition is not None:
                row["partition"] = str(eval_partition)
            source_epoch = getattr(
                self.config, "ct_source_checkpoint_epoch", None)
            if source_epoch is not None:
                row["epoch"] = int(source_epoch)
            if bool(getattr(
                    self.config, "use_asymmetric_dual_query", False)):
                row["tracklet_key"] = str(tracklet_key)
                row["dataset_split"] = str(getattr(
                    self.config, "test_split", "unknown"))
                row["candidate_config_sha256"] = (
                    b2_candidate_config_sha256(self.config))
            self._proposal_test_diagnostics.append(row)
        for row in self._b3_sequence_rollouts:
            row = dict(row)
            row["tracklet_id"] = int(batch_idx)
            row["tracklet_key"] = str(tracklet_key)
            self._b3_test_rollouts.append(row)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(sequence)

        
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))

        self.log('success/test', self.success,  on_epoch=True) 
        self.log('precision/test', self.prec,  on_epoch=True) 
        self.success_step(torch.tensor(ious, device=self.device))
        self.prec_step(torch.tensor(distances, device=self.device))
        self.n_frames(torch.tensor(n_frames, device=self.device))

        self.log('success/test_step', self.success_step, on_step=True, on_epoch=False)
        self.log('precision/test_step', self.prec_step, on_step=True, on_epoch=False)

        self.success_step.reset()
        self.prec_step.reset()


        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))
        self.logger.experiment.add_scalars('FPS', {'fps': 1.0/self.runtime.compute()}, global_step=batch_idx)

        return result_bbs

    def on_test_epoch_start(self):
        self._proposal_test_diagnostics = []
        self._tracking_test_endpoints = []
        self._b3_test_rollouts = []

    def on_test_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test/current',
                                    {'success': self.success.compute(),
                                        'precision': self.prec.compute()},
                                    global_step=self.global_step)

        self.logger.experiment.add_scalars('metrics/fps',
                                    {'runtime':1.0/self.runtime.compute(),},
                                    global_step=self.global_step)
        self.logger.experiment.add_scalars('frames',
                                    {'frame':self.n_frames.compute(),},
                                    global_step=self.global_step)
        if bool(getattr(
                self.config, "export_proposal_diagnostics", False)):
            self._write_proposal_test_diagnostics()
            self._write_tracking_test_endpoints()
        if bool(getattr(self.config, "export_b3_rollouts", False)):
            self._write_b3_test_rollouts()

    def on_validation_epoch_start(self):
        self._v3_validation_proposal_diagnostics = []

class MotionBaseModelMF(BaseModelMF):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs): # Note: There may be cases of input with empty point clouds
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        if (bool(getattr(
                self.config, "use_b1_prepass_support", False))
                and kwargs.get("motion_prediction") is None):
            predictor = getattr(self, "predict_motion_prepass", None)
            if predictor is None:
                raise RuntimeError(
                    "B1 pre-pass support requires a motion predictor")
            kwargs["motion_prediction"] = predictor(
                sequence, frame_id, results_bbs)

        recursive_state = kwargs.get('recursive_state')
        use_recursive_contract = bool(getattr(
            self.config, 'use_ct_joint_full', False)) or bool(getattr(
                self.config, 'observation_safe_bbox_size', False))
        if use_recursive_contract and recursive_state is not None:
            recursive_contract = build_recursive_input_contract(
                recursive_state, frame_id, self.hist_num, self.config,
                candidate_id=0)
            prev_frame_ids = recursive_contract['history_frame_ids']
            valid_mask = recursive_contract['history_valid_mask'].tolist()
            ref_boxs = recursive_state.history_boxes(
                prev_frame_ids, valid_mask)
            prev_sampling_seeds = recursive_contract[
                'point_sampling_seeds'].tolist()
            current_sampling_seed = int(
                recursive_contract['current_sampling_seed'])
        elif use_recursive_contract:
            prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
                frame_id, self.hist_num)
            ref_boxs = get_last_n_bounding_boxes(results_bbs, valid_mask)
            fallback_state = RecursiveTrackState(
                tracklet_id=0,
                tracklet_key=str(sequence[0].get(
                    'tracklet_key', sequence[0].get(
                        'tracklet_id', 'eval'))),
                first_box=results_bbs[0])
            for prediction_id, prediction in enumerate(results_bbs[1:], 1):
                fallback_state.append(prediction_id, prediction)
            recursive_contract = build_recursive_input_contract(
                fallback_state, frame_id, self.hist_num, self.config,
                candidate_id=0)
            prev_sampling_seeds = recursive_contract[
                'point_sampling_seeds'].tolist()
            current_sampling_seed = int(
                recursive_contract['current_sampling_seed'])
        else:
            prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
                frame_id, self.hist_num)
            ref_boxs = get_last_n_bounding_boxes(results_bbs, valid_mask)
            prev_sampling_seeds = [None] * len(prev_frame_ids)
            current_sampling_seed = 1
            recursive_contract = None
        prev_frames = [sequence[id] for id in prev_frame_ids]
        this_frame = sequence[frame_id]
        this_pc = this_frame['pc']
        prev_pcs = [frame['pc'] for frame in prev_frames]
        bbox_size = (
            recursive_contract['target_size']
            if use_recursive_contract
            else this_frame['3d_bbox'].wlh)
        num_hist = len(valid_mask)
        default_time_step = getattr(
            self.config, 'default_time_step',
            getattr(self.config, 'time_step', 0.1))
        pseudo_time_step = getattr(self.config, 'pseudo_time_step', 0.1)
        use_real_time = getattr(self.config, 'use_real_time', True)
        prev_timestamps = [frame.get('timestamp') for frame in prev_frames]
        current_timestamp = this_frame.get('timestamp')
        real_time_fields = build_time_fields(
            prev_timestamps, current_timestamp,
            frame_ids=prev_frame_ids,
            current_frame_id=frame_id,
            use_real_time=use_real_time,
            default_step=default_time_step,
            pseudo_step=pseudo_time_step)
        relative_timestamps, delta_t_list, local_timestamps, current_timestamp = (
            real_time_fields)
        dynamics_time_mode = normalize_dynamics_time_mode(
            this_frame.get(
                '_ct_dynamics_time_mode',
                getattr(self.config, 'dynamics_time_mode', 'true')))
        effective_time_fields = build_effective_time_fields(
            dynamics_time_mode,
            real_time_fields,
            effective_frame_timestamps=[
                frame.get('_ct_effective_timestamp') for frame in prev_frames
            ],
            effective_current_timestamp=this_frame.get(
                '_ct_effective_timestamp'),
            frame_ids=prev_frame_ids,
            current_frame_id=frame_id,
            default_step=float(getattr(
                self.config, 'dynamics_fixed_delta_t', default_time_step)),
            pseudo_step=pseudo_time_step,
        )
        (effective_relative_timestamps, effective_delta_t_list,
         effective_local_timestamps, effective_current_timestamp) = (
            effective_time_fields)
        main_current_value = float(
            getattr(self.config, 'main_time_current', 0.0))
        point_timestamps, corner_timestamps, main_timestamps = (
            build_main_time_fields(
                valid_mask,
                relative_timestamps,
                local_timestamps,
                num_hist,
                pseudo_step=pseudo_time_step,
                source=getattr(self.config, 'main_time_source', 'real'),
                current_value=main_current_value))

        prev_frame_pcs = []
        for i, prev_pc in enumerate(prev_pcs):
            prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(prev_pc, ref_boxs[i], ref_boxs[0],
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)
            prev_frame_pcs.append(prev_frame_pc)

        this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(
            this_pc, ref_boxs[0], ref_boxs[0],
            scale=self.config.bb_scale,
            offset=self.config.bb_offset)
        baseline_search_points = this_frame_pc.points.T
        expanded_search_points = np.empty(
            (0, baseline_search_points.shape[1]),
            dtype=baseline_search_points.dtype,
        )
        ct_search_box = None
        ct_search_diagnostics = {
            "valid": False,
            "query_delta_t": float(effective_delta_t_list[0]),
        }
        use_ct_joint_full = bool(getattr(
            self.config, "use_ct_joint_full", False))
        joint_contract_v3 = bool(
            int(getattr(
                self.config, 'ct_joint_contract_version', 1)) >= 3)
        use_trajectory_search = (
            bool(getattr(self.config, "use_trajectory_search", False))
            or use_ct_joint_full)
        if (bool(getattr(self.config, "use_time_guided_search", False))
                and use_trajectory_search):
            raise ValueError(
                "legacy time-guided search and ordered trajectory search are "
                "mutually exclusive")
        if (bool(getattr(self.config, "use_time_guided_search", False))
                or use_trajectory_search):
            if use_trajectory_search:
                ct_search_box, ct_search_diagnostics = (
                    build_ordered_trajectory_search_box(
                        ref_boxs,
                        effective_delta_t_list,
                        valid_mask=valid_mask,
                        base_length=float(getattr(
                            self.config, "trajectory_search_base_length", 4.0)),
                        base_width=float(getattr(
                            self.config, "trajectory_search_base_width", 2.0)),
                        max_length=float(getattr(
                            self.config, "ct_tube_max_length", 24.0)
                            if use_ct_joint_full else getattr(
                                self.config,
                                "trajectory_search_max_length", 20.0)),
                        max_width=float(getattr(
                            self.config, "trajectory_search_max_width", 8.0)),
                        max_speed=float(getattr(
                            self.config, "ct_motion_max_speed", 20.0)
                            if use_ct_joint_full else getattr(
                                self.config,
                                "trajectory_search_max_speed", 20.0)),
                        max_acceleration=float(getattr(
                            self.config, "ct_motion_max_acceleration", 8.0)
                            if use_ct_joint_full else getattr(
                                self.config,
                                "trajectory_search_max_acceleration", 8.0)),
                        max_displacement=float(getattr(
                            self.config, "ct_motion_max_displacement", 12.0)
                            if use_ct_joint_full else getattr(
                                self.config,
                                "trajectory_search_max_displacement", 12.0)),
                        acceleration_weight=float(getattr(
                            self.config, "ct_motion_acceleration_weight", 0.5)
                            if use_ct_joint_full else getattr(
                                self.config,
                                "trajectory_search_acceleration_weight", 0.5)),
                        sigma_parallel_scale=float(getattr(
                            self.config, "trajectory_search_sigma_parallel_scale", 2.0)),
                        sigma_perpendicular_scale=float(getattr(
                            self.config, "trajectory_search_sigma_perpendicular_scale", 2.0)),
                        min_displacement=float(getattr(
                            self.config, "trajectory_search_min_displacement", 0.2)),
                        min_delta_t=float(getattr(
                            self.config, "trajectory_search_min_delta_t", 0.75)),
                        min_gap_ratio=float(getattr(
                            self.config, "trajectory_search_min_gap_ratio", 1.5)),
                        allow_normal_cadence=(
                            True if use_ct_joint_full else bool(getattr(
                                self.config,
                                "trajectory_search_allow_normal_cadence",
                                False))),
                        require_recent_transition=use_ct_joint_full,
                    ))
            else:
                ct_search_box, ct_search_diagnostics = (
                    build_time_guided_search_box(
                        ref_boxs,
                        effective_delta_t_list,
                        valid_mask=valid_mask,
                        base_length=float(getattr(
                            self.config, "ct_search_base_length", 4.0)),
                        base_width=float(getattr(
                            self.config, "ct_search_base_width", 2.0)),
                        max_length=float(getattr(
                            self.config, "ct_search_max_length", 16.0)),
                        max_width=float(getattr(
                            self.config, "ct_search_max_width", 6.0)),
                        max_speed=float(getattr(
                            self.config, "ct_search_max_speed", 20.0)),
                        max_displacement=float(getattr(
                            self.config, "ct_search_max_displacement", 12.0)),
                        width_per_second=float(getattr(
                            self.config, "ct_search_width_per_second", 0.25)),
                        min_displacement=float(getattr(
                            self.config, "ct_search_min_displacement", 0.2)),
                    ))
            if ct_search_box is not None:
                ct_search_pc = points_utils.generate_subwindow_with_aroundboxs(
                    this_pc,
                    ct_search_box,
                    ref_boxs[0],
                    scale=1.0,
                    offset=0.0,
                )
                expanded_search_points = ct_search_pc.points.T

        use_search_evidence_v2 = bool(getattr(
            self.config, "use_search_evidence_v2", False))
        use_search_evidence_v21 = bool(getattr(
            self.config, "use_search_evidence_v21", False))
        use_search_evidence_v22 = bool(getattr(
            self.config, "use_motion_conditioned_search_v22", False))
        use_search_evidence_v3 = (
            bool(getattr(
                self.config, "use_motion_conditioned_search_v3", False))
            or use_ct_joint_full)
        if sum(map(bool, (
                use_search_evidence_v2,
                use_search_evidence_v21,
                use_search_evidence_v22,
                use_search_evidence_v3))) > 1:
            raise ValueError(
                "Search Evidence v2, v2.1, v2.2, and v3 are exclusive")
        use_endpoint_search_evidence = (
            use_search_evidence_v2
            or use_search_evidence_v21
            or use_search_evidence_v22
            or use_search_evidence_v3)
        search_config_prefix = (
            "search_v3" if use_search_evidence_v3
            else "search_v22" if use_search_evidence_v22
            else "search_v21" if use_search_evidence_v21
            else "search_v2")

        def search_config_value(name, default):
            if use_ct_joint_full:
                joint_mapping = {
                    'point_count': 'ct_endpoint_quota',
                    'extension_quota': 'ct_endpoint_quota',
                    'min_points': 'ct_search_min_points',
                    'max_length': 'ct_tube_max_length',
                    'max_width': 'ct_tube_max_width',
                    'max_speed': 'ct_motion_max_speed',
                    'max_acceleration': 'ct_motion_max_acceleration',
                    'max_displacement': 'ct_motion_max_displacement',
                    'acceleration_weight': 'ct_motion_acceleration_weight',
                }
                field = joint_mapping.get(name)
                if field is not None:
                    return getattr(self.config, field, default)
            return getattr(
                self.config, f"{search_config_prefix}_{name}", default)

        search_v2_box = None
        search_v2_diagnostics = {
            "valid": False,
            "query_delta_t": float(effective_delta_t_list[0]),
            "gap_ratio": 1.0,
            "sigma_parallel": 0.0,
            "sigma_perpendicular": 0.0,
        }
        search_v2_expanded_points = np.empty(
            (0, baseline_search_points.shape[1]),
            dtype=baseline_search_points.dtype,
        )
        search_v2_endpoint_xy = np.zeros((2,), dtype=np.float32)
        if use_endpoint_search_evidence:
            motion_prediction = kwargs.get("motion_prediction")
            use_prepass = (
                bool(getattr(self.config, "use_b1_prepass_support", False))
                if (not use_ct_joint_full or int(getattr(
                    self.config, "ct_joint_contract_version", 1)) >= 2)
                else False)
            support_kwargs = dict(
                prediction=motion_prediction,
                use_b1_prepass=use_prepass,
                use_dynamic_sigma=bool(getattr(
                    self.config, "search_v3_use_dynamic_sigma", False)),
                fixed_margins=(
                    float(getattr(
                        self.config,
                        "search_v3_fixed_margin_parallel", 2.0)),
                    float(getattr(
                        self.config,
                        "search_v3_fixed_margin_perpendicular", 1.0)),
                ),
                coverage_scale=float(getattr(
                    self.config, "search_v3_coverage_scale", 2.448)),
                standardized_residual_quantile=tuple(getattr(
                    self.config,
                    'search_v3_standardized_residual_q90_parallel_perpendicular',
                    (1.0, 1.0))),
                min_direction_speed=float(getattr(
                    self.config, "motion_v3_min_direction_speed", 0.2)),
                max_length=float(search_config_value('max_length', 24.0)),
                max_width=float(search_config_value('max_width', 10.0)),
                fallback_max_speed=float(search_config_value(
                    'max_speed', 20.0)),
                fallback_max_acceleration=float(search_config_value(
                    'max_acceleration', 8.0)),
                fallback_max_displacement=float(search_config_value(
                    'max_displacement', 12.0)),
                fallback_acceleration_weight=float(search_config_value(
                    'acceleration_weight', 0.5)),
                fallback_max_yaw_rate=float(search_config_value(
                    'max_yaw_rate', np.pi / 2.0)),
                fallback_min_displacement=float(search_config_value(
                    'min_displacement', 0.2))
                if not (use_ct_joint_full and int(getattr(
                    self.config, "ct_joint_contract_version", 1)) >= 2)
                else 0.0,
                fallback_require_recent_transition=use_ct_joint_full,
            )
            if (use_ct_joint_full and int(getattr(
                    self.config, "ct_joint_contract_version", 1)) >= 2):
                (search_v2_box,
                 ct_search_box,
                 search_v2_diagnostics) = resolve_joint_search_geometry(
                    ref_boxs,
                    effective_delta_t_list,
                    valid_mask,
                    **support_kwargs,
                )
                if ct_search_box is not None:
                    ct_search_diagnostics = dict(search_v2_diagnostics)
                    ct_search_pc = points_utils.generate_subwindow_with_aroundboxs(
                        this_pc, ct_search_box, ref_boxs[0],
                        scale=1.0, offset=0.0)
                    expanded_search_points = ct_search_pc.points.T
            else:
                search_v2_box, search_v2_diagnostics = resolve_b1_search_support(
                    ref_boxs,
                    effective_delta_t_list,
                    valid_mask,
                    **support_kwargs,
                )
            if search_v2_box is not None:
                learned_prior_support = (
                    search_v2_diagnostics.get("prior_source") == "b1")
                search_v2_pc = points_utils.generate_subwindow_with_aroundboxs(
                    this_pc,
                    search_v2_box,
                    ref_boxs[0],
                    scale=(1.0 if learned_prior_support
                           else self.config.bb_scale),
                    offset=(0.0 if learned_prior_support
                            else self.config.bb_offset),
                )
                search_v2_expanded_points = search_v2_pc.points.T
                endpoint_center = search_v2_diagnostics.get(
                    "endpoint_center")
                if endpoint_center is not None:
                    endpoint_box = copy.deepcopy(ref_boxs[0])
                    endpoint_box.center = np.asarray(
                        endpoint_center, dtype=np.float64)
                    endpoint_local = points_utils.transform_box(
                        endpoint_box, ref_boxs[0])
                    search_v2_endpoint_xy = np.asarray(
                        endpoint_local.center[:2], dtype=np.float32)
                else:
                    search_v2_local_box = points_utils.transform_box(
                        search_v2_box, ref_boxs[0])
                    search_v2_endpoint_xy = np.asarray(
                        search_v2_local_box.center[:2], dtype=np.float32)
        num_points_in_search = this_frame_pc.nbr_points()

        coordinate_anchor_box = ref_boxs[0]
        coordinate_anchor_theta = (
            coordinate_anchor_box.orientation.degrees
            * coordinate_anchor_box.orientation.axis[-1]
            if self.config.degrees
            else coordinate_anchor_box.orientation.radians
            * coordinate_anchor_box.orientation.axis[-1])
        coordinate_anchor = np.append(
            coordinate_anchor_box.center,
            coordinate_anchor_theta).astype(np.float32)
        # canonical_box = points_utils.transform_box(ref_boxs[0], ref_boxs[0])
        ref_boxs = [
            points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs
        ]

        prev_points_list = [
            points_utils.regularize_pc(
                prev_frame_pc.points.T, self.config.point_sample_size,
                seed=seed)[0]
            for prev_frame_pc, seed in zip(
                prev_frame_pcs, prev_sampling_seeds)
        ]

        trajectory_search_points = np.zeros(
            (int(getattr(self.config, "ct_tube_quota", 128)
                 if use_ct_joint_full else getattr(
                     self.config, "trajectory_search_point_count", 128)),
             baseline_search_points.shape[1]),
            dtype=np.float32,
        )
        trajectory_search_sampling = {
            "active": False,
            "sample_count": 0,
            "available_count": 0,
        }
        trajectory_search_point_valid_mask = np.zeros(
            (trajectory_search_points.shape[0],), dtype=np.float32)
        trajectory_search_point_source = np.zeros(
            (trajectory_search_points.shape[0],), dtype=np.int64)
        if use_trajectory_search:
            this_points, idx_this = points_utils.regularize_pc(
                baseline_search_points,
                self.config.point_sample_size,
                seed=current_sampling_seed,
            )
            if use_ct_joint_full:
                (trajectory_search_points,
                 trajectory_search_point_valid_mask,
                 trajectory_search_point_source,
                 trajectory_search_sampling) = (
                    sample_source_aware_endpoint_points(
                        baseline_search_points,
                        expanded_search_points,
                        sample_size=int(getattr(
                            self.config, "ct_tube_quota", 128)),
                        extension_quota=int(getattr(
                            self.config, "ct_tube_quota", 128)),
                        min_points=int(getattr(
                            self.config, "ct_search_min_points", 3)),
                        seed=current_sampling_seed,
                    ))
            else:
                trajectory_search_points, trajectory_search_sampling = (
                    sample_search_extension(
                        baseline_search_points,
                        expanded_search_points,
                        int(getattr(
                            self.config, "trajectory_search_point_count", 128)),
                        min_expansion_points=int(getattr(
                            self.config, "trajectory_search_min_points", 16)),
                        seed=current_sampling_seed,
                    ))
                trajectory_search_point_valid_mask.fill(
                    float(trajectory_search_sampling["active"]))
            ct_search_sampling = {
                "baseline_sample_count": int(self.config.point_sample_size),
                "expansion_sample_count": int(
                    trajectory_search_sampling["sample_count"]),
                "expansion_available_count": int(
                    trajectory_search_sampling["available_count"]),
            }
            ct_search_active = bool(trajectory_search_sampling["active"])
            num_points_in_search = int(len(baseline_search_points))
            if ct_search_active:
                num_points_in_search += int(
                    trajectory_search_sampling["available_count"])
        elif bool(getattr(self.config, "use_time_guided_search", False)):
            this_points, ct_search_sampling = stratified_search_sample(
                baseline_search_points,
                expanded_search_points,
                self.config.point_sample_size,
                baseline_ratio=float(getattr(
                    self.config, "ct_search_baseline_ratio", 0.75)),
                min_expansion_points=int(getattr(
                    self.config, "ct_search_min_expansion_points", 32)),
                seed=current_sampling_seed,
            )
            ct_search_active = (
                ct_search_sampling["expansion_sample_count"] > 0)
            num_points_in_search = int(len(baseline_search_points))
            if ct_search_active:
                num_points_in_search += int(
                    ct_search_sampling["expansion_available_count"])
        else:
            this_points, idx_this = points_utils.regularize_pc(
                this_frame_pc.points.T,
                self.config.point_sample_size,
                seed=current_sampling_seed)
            ct_search_sampling = {
                "baseline_sample_count": int(self.config.point_sample_size),
                "expansion_sample_count": 0,
                "expansion_available_count": 0,
            }
            ct_search_active = False
        search_v2_point_count = int(search_config_value('point_count', 128))
        search_v2_points = np.zeros(
            (search_v2_point_count, baseline_search_points.shape[1]),
            dtype=np.float32,
        )
        search_v2_point_valid_mask = np.zeros(
            (search_v2_point_count,), dtype=np.float32)
        search_v2_point_source = np.zeros(
            (search_v2_point_count,), dtype=np.int64)
        search_v2_sampling = {
            "active": False,
            "sample_count": 0,
            "available_count": 0,
            "extension_count": 0,
            "overlap_count": 0,
        }
        if use_endpoint_search_evidence and search_v2_box is not None:
            search_v2_seed = (
                int(current_sampling_seed) * 1664525 + 1013904223
            ) & 0xFFFFFFFF
            if (use_search_evidence_v21 or use_search_evidence_v22
                    or use_search_evidence_v3):
                (search_v2_points,
                 search_v2_point_valid_mask,
                 search_v2_point_source,
                 search_v2_sampling) = sample_source_aware_endpoint_points(
                    baseline_search_points,
                    search_v2_expanded_points,
                    sample_size=search_v2_point_count,
                    extension_quota=int(search_config_value(
                        'extension_quota', 64)),
                    min_points=int(search_config_value('min_points', 3)),
                    seed=search_v2_seed,
                )
            else:
                (search_v2_points,
                 search_v2_point_valid_mask,
                 search_v2_sampling) = sample_padded_search_extension(
                    baseline_search_points,
                    search_v2_expanded_points,
                    sample_size=search_v2_point_count,
                    min_expansion_points=int(search_config_value(
                        'min_points', 3)),
                    seed=search_v2_seed,
                )
        joint_extension_source = None
        joint_extension_sampling = None
        if use_ct_joint_full and joint_contract_v3:
            joint_extension_seed = (
                int(current_sampling_seed) * 22695477 + 1) & 0xFFFFFFFF
            (joint_extension_points,
             joint_extension_valid_mask,
             joint_extension_source,
             joint_extension_sampling) = sample_joint_novel_extensions(
                baseline_search_points,
                search_v2_expanded_points,
                expanded_search_points,
                endpoint_quota=int(getattr(
                    self.config, 'ct_endpoint_quota', 128)),
                tube_quota=int(getattr(
                    self.config, 'ct_tube_quota', 128)),
                seed=joint_extension_seed,
            )
            joint_extension_sampling.pop('_pool_points', None)
            endpoint_quota = int(getattr(
                self.config, 'ct_endpoint_quota', 128))
            search_v2_points = joint_extension_points[:endpoint_quota]
            search_v2_point_valid_mask = joint_extension_valid_mask[
                :endpoint_quota]
            search_v2_point_source = (
                search_v2_point_valid_mask > 0).astype(np.int64)
            trajectory_search_points = joint_extension_points[
                endpoint_quota:]
            trajectory_search_point_valid_mask = (
                joint_extension_valid_mask[endpoint_quota:])
            trajectory_search_point_source = (
                trajectory_search_point_valid_mask > 0).astype(np.int64)
        joint_support = combined_search_support_statistics(
            (search_v2_points, trajectory_search_points),
            (search_v2_point_valid_mask,
             trajectory_search_point_valid_mask),
            (search_v2_point_source, trajectory_search_point_source),
            voxel_size=float(getattr(
                self.config, 'ct_search_extension_voxel_size', 0.2)),
        )
        coverage_need, endpoint_ratio = useful_search_coverage_need(
            search_v2_diagnostics.get(
                'query_delta_t', effective_delta_t_list[0]),
            search_v2_diagnostics.get('gap_ratio', 1.0),
            search_v2_endpoint_xy,
            coordinate_anchor_box.wlh,
            len(baseline_search_points),
            min_delta_t=float(getattr(
                self.config, 'trajectory_search_min_delta_t', 0.75)),
            min_gap_ratio=float(getattr(
                self.config, 'trajectory_search_min_gap_ratio', 1.5)),
            min_endpoint_ratio=float(getattr(
                self.config, 'ct_search_endpoint_ratio', 0.6)),
            sparse_base_points=int(getattr(
                self.config, 'ct_search_sparse_base_points', 64)),
            bb_scale=float(self.config.bb_scale),
            bb_offset=float(self.config.bb_offset),
        )
        recent_history_valid = bool(
            len(valid_mask) >= 2 and int(valid_mask[0]) and int(valid_mask[1]))
        query_dt_value = float(search_v2_diagnostics.get(
            'query_delta_t', effective_delta_t_list[0]))
        time_valid = bool(np.isfinite(query_dt_value) and query_dt_value > 0.0)
        proposal_valid = bool(
            search_v2_diagnostics.get('valid', False)
            and not search_v2_diagnostics.get(
                'constraint_clipped', False)
            and np.isfinite(search_v2_endpoint_xy).all()
            and float(search_v2_diagnostics.get('displacement', 0.0))
            >= float(getattr(
                self.config, 'trajectory_search_min_displacement', 0.2)))
        point_support_valid = bool(
            joint_support['total_count'] >= int(getattr(
                self.config, 'ct_search_min_total_points', 16))
            and joint_support['extension_count'] >= int(getattr(
                self.config, 'ct_search_min_extension_points', 8))
            and joint_support['extension_voxels'] >= int(getattr(
                self.config, 'ct_search_min_extension_voxels', 4)))
        geometry_valid = bool(
            recent_history_valid and time_valid
            and search_v2_box is not None and ct_search_box is not None
            and search_v2_diagnostics.get('valid', False)
            and np.isfinite(search_v2_endpoint_xy).all())
        structural_point_valid = bool(joint_support['total_count'] >= 3)
        new_support_valid = bool(
            joint_support['extension_count'] >= 1
            and joint_support['extension_voxels'] >= 1)
        joint_contract_v2 = bool(
            int(getattr(self.config, 'ct_joint_contract_version', 1)) >= 2)
        if use_ct_joint_full and joint_contract_v3:
            search_support_valid = bool(
                geometry_valid
                and joint_extension_sampling is not None
                and joint_extension_sampling['sample_count'] > 0)
        elif (use_ct_joint_full and joint_contract_v2 and bool(getattr(
                self.config, 'ct_search_relaxed_validity', True))):
            search_support_valid = bool(
                geometry_valid and structural_point_valid
                and new_support_valid)
        else:
            search_support_valid = bool(
                recent_history_valid and time_valid and proposal_valid
                and coverage_need and point_support_valid)
        search_has_usable_points = num_points_in_search > 2
        seg_mask_prev_list = [geometry_utils.points_in_box(ref_box, prev_points.T[:3,:], 1.25).astype(float) for ref_box,prev_points in zip(ref_boxs,prev_points_list)]#应当只考虑xyz特征

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            for seg_mask_prev in seg_mask_prev_list:
                # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
                # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
                seg_mask_prev[seg_mask_prev == 0] = 0.2
                seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev_list[0].shape, fill_value=0.5)

        timestamp_prev_list = [
            np.full((self.config.point_sample_size, 1), fill_value=timestamp, dtype=np.float32)
            for timestamp in point_timestamps
        ]
        timestamp_this = np.full(
            (self.config.point_sample_size, 1), fill_value=main_current_value, dtype=np.float32)
        prev_points_list = [
        np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]],
                       axis=-1)
        for prev_points, timestamp_prev, seg_mask_prev in zip(
            prev_points_list, timestamp_prev_list, seg_mask_prev_list)
        ]

        this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)
        trajectory_timestamp = np.full(
            (trajectory_search_points.shape[0], 1),
            fill_value=main_current_value,
            dtype=np.float32,
        )
        trajectory_prior = np.full(
            (trajectory_search_points.shape[0], 1),
            fill_value=0.5,
            dtype=np.float32,
        )
        trajectory_search_points = np.concatenate(
            (trajectory_search_points, trajectory_timestamp, trajectory_prior),
            axis=-1,
        )
        search_v2_timestamp = np.full(
            (search_v2_points.shape[0], 1),
            fill_value=main_current_value,
            dtype=np.float32,
        )
        search_v2_prior = np.full(
            (search_v2_points.shape[0], 1),
            fill_value=0.5,
            dtype=np.float32,
        )
        search_v2_points = np.concatenate(
            (search_v2_points, search_v2_timestamp, search_v2_prior),
            axis=-1,
        )

        stack_points_list = prev_points_list + [this_points]
        stack_points = np.concatenate(stack_points_list, axis=0)

        ref_box_thetas = [
            ref_box.orientation.degrees * ref_box.orientation.axis[-1]
            if self.config.degrees else ref_box.orientation.radians *
            ref_box.orientation.axis[-1] for ref_box in ref_boxs
        ]
        ref_box_list = [
            np.append(ref_box.center,
                      theta).astype('float32') for ref_box, theta in zip(
                          ref_boxs, ref_box_thetas)
        ]
        ref_boxs_np = np.stack(ref_box_list, axis=0)

        current_delta_t = delta_t_list[0] if len(delta_t_list) > 0 else default_time_step
        current_delta_t_effective = (
            effective_delta_t_list[0] if len(effective_delta_t_list) > 0
            else float(getattr(self.config, 'dynamics_fixed_delta_t', default_time_step)))

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32), 
                     "ref_boxs":torch.tensor(ref_boxs_np[None, :], device=self.device, dtype=torch.float32), 
                     "valid_mask":torch.tensor(valid_mask, device=self.device, dtype=torch.float32).unsqueeze(0), 
                     "bbox_size":torch.tensor(bbox_size[None, :],device=self.device, dtype=torch.float32),
                     "timestamps": torch.tensor(main_timestamps[None, :], device=self.device, dtype=torch.float32),
                     "delta_t": torch.tensor(np.array(delta_t_list, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "delta_t_real": torch.tensor(np.array(delta_t_list, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "delta_t_effective": torch.tensor(np.array(effective_delta_t_list, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "delta_T": torch.tensor(np.array(corner_timestamps, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "timestamps_real": torch.tensor(local_timestamps[None, :], device=self.device, dtype=torch.float32),
                     "delta_T_real": torch.tensor(np.array(relative_timestamps, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "timestamps_effective": torch.tensor(np.asarray(effective_local_timestamps, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "delta_T_effective": torch.tensor(np.array(effective_relative_timestamps, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "current_timestamp": torch.tensor([current_timestamp], device=self.device, dtype=torch.float64),
                     "current_effective_timestamp": torch.tensor([effective_current_timestamp], device=self.device, dtype=torch.float64),
                     "current_delta_t": torch.tensor([current_delta_t], device=self.device, dtype=torch.float32),
                     "current_delta_t_real": torch.tensor([current_delta_t], device=self.device, dtype=torch.float32),
                     "current_delta_t_effective": torch.tensor([current_delta_t_effective], device=self.device, dtype=torch.float32),
                     "dynamics_time_mode_id": torch.tensor([
                         {'true': 0, 'fixed': 1, 'shuffled': 2}[dynamics_time_mode]
                     ], device=self.device, dtype=torch.int64),
                     "num_points_in_search": torch.tensor([num_points_in_search], device=self.device, dtype=torch.float32),
                     "search_has_usable_points": torch.tensor(
                         [search_has_usable_points],
                         device=self.device, dtype=torch.float32),
                     "ct_search_used": torch.tensor(
                         [search_support_valid
                          if use_ct_joint_full else ct_search_active],
                         device=self.device, dtype=torch.float32),
                     "ct_search_expansion_ratio": torch.tensor(
                         [ct_search_sampling["expansion_sample_count"]
                          / float(self.config.point_sample_size)],
                         device=self.device, dtype=torch.float32),
                     "ct_search_baseline_points": torch.tensor(
                         [len(baseline_search_points)],
                         device=self.device, dtype=torch.float32),
                     "ct_search_expansion_points": torch.tensor(
                         [ct_search_sampling["expansion_available_count"]],
                         device=self.device, dtype=torch.float32),
                     "ct_search_query_delta_t": torch.tensor(
                         [ct_search_diagnostics.get(
                             "query_delta_t", effective_delta_t_list[0])],
                         device=self.device, dtype=torch.float32),
                      "ct_search_predicted_displacement": torch.tensor(
                          [ct_search_diagnostics.get("displacement", 0.0)],
                          device=self.device, dtype=torch.float32),
                      "ct_search_support_valid": torch.tensor(
                          [search_support_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_geometry_valid": torch.tensor(
                          [geometry_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_structural_point_valid": torch.tensor(
                          [structural_point_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_new_support_valid": torch.tensor(
                          [new_support_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_quality_valid": torch.tensor(
                          [point_support_valid], device=self.device,
                          dtype=torch.float32),
                      "candidate_valid": torch.tensor(
                          [search_support_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_history_valid": torch.tensor(
                          [recent_history_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_time_valid": torch.tensor(
                          [time_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_proposal_valid": torch.tensor(
                          [proposal_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_point_support_valid": torch.tensor(
                          [point_support_valid], device=self.device,
                          dtype=torch.float32),
                      "ct_search_coverage_need": torch.tensor(
                          [coverage_need], device=self.device,
                          dtype=torch.float32),
                      "ct_search_endpoint_ratio": torch.tensor(
                          [endpoint_ratio], device=self.device,
                          dtype=torch.float32),
                      "ct_search_total_point_count": torch.tensor(
                          [joint_support['total_count']], device=self.device,
                          dtype=torch.float32),
                      "ct_search_extension_count": torch.tensor(
                          [joint_support['extension_count']], device=self.device,
                          dtype=torch.float32),
                      "ct_search_extension_voxels": torch.tensor(
                          [joint_support['extension_voxels']], device=self.device,
                          dtype=torch.float32),
                      }
        if use_ct_joint_full and joint_contract_v3:
            if joint_extension_source is None:
                raise RuntimeError(
                    "contract-v3 inference extension source was not built")
            extension_points = np.concatenate((
                search_v2_points, trajectory_search_points), axis=0)
            extension_valid_mask = np.concatenate((
                search_v2_point_valid_mask,
                trajectory_search_point_valid_mask), axis=0)
            data_dict.update({
                'ct_base_evidence_points': torch.tensor(
                    this_points[None, :], device=self.device,
                    dtype=torch.float32),
                'ct_base_evidence_valid_mask': torch.ones(
                    (1, self.config.point_sample_size),
                    device=self.device, dtype=torch.float32),
                'ct_extension_points': torch.tensor(
                    extension_points[None, :], device=self.device,
                    dtype=torch.float32),
                'ct_extension_valid_mask': torch.tensor(
                    extension_valid_mask[None, :], device=self.device,
                    dtype=torch.float32),
                'ct_extension_source': torch.tensor(
                    joint_extension_source[None, :], device=self.device,
                    dtype=torch.long),
            })
        if bool(getattr(self.config, "use_b1motion_v3", False)):
            # Online history is already recursive and expressed in the latest
            # predicted anchor.  Expose an explicit motion contract rather than
            # reusing legacy dynamics fields with different target semantics.
            online_motion_anchor = torch.tensor(
                coordinate_anchor[None, :],
                device=self.device, dtype=torch.float32)
            data_dict.update({
                "motion_main_ref_boxs": data_dict["ref_boxs"],
                "motion_main_delta_t": data_dict["delta_t_effective"],
                "motion_main_current_delta_t": data_dict[
                    "current_delta_t_effective"],
                "motion_main_valid_mask": data_dict["valid_mask"],
                "motion_main_anchor": online_motion_anchor,
                # The online B1 prior and B2 support are expressed in the same
                # latest recursive crop frame.  Keep both contract-v3 names as
                # references to the exact same tensor so identity conversion is
                # bitwise and the two coordinate systems cannot drift apart.
                "motion_source_anchor": online_motion_anchor,
                "coordinate_anchor": online_motion_anchor,
            })
            if use_search_evidence_v3:
                # Both branches own references to the same online recursive
                # state tensors.  No reconstruction or second clock is allowed.
                data_dict.update({
                    "b2_v3_history_ref_boxs": data_dict["ref_boxs"],
                    "b2_v3_history_delta_t":
                        data_dict["delta_t_effective"],
                    "b2_v3_history_valid_mask": data_dict["valid_mask"],
                    "b2_v3_history_mode_id": torch.tensor(
                        [2], device=self.device, dtype=torch.int64),
                    "b2_v3_history_anchor": data_dict[
                        "motion_main_anchor"],
                })
        if (use_trajectory_search or bool(getattr(
                self.config, "use_ordered_trajectory_encoder", False))):
            data_dict.update({
                "trajectory_search_points": torch.tensor(
                    trajectory_search_points[None, :],
                    device=self.device, dtype=torch.float32),
                "trajectory_search_point_valid_mask": torch.tensor(
                    trajectory_search_point_valid_mask[None, :],
                    device=self.device, dtype=torch.float32),
                "trajectory_search_point_source": torch.tensor(
                    trajectory_search_point_source[None, :],
                    device=self.device, dtype=torch.long),
                # Stable branch contract: 0=baseline, 1=endpoint, 2=tube.
                "trajectory_search_branch_source": torch.full(
                    (1, trajectory_search_points.shape[0]), 2,
                    device=self.device, dtype=torch.long),
                "trajectory_search_valid": torch.tensor(
                    [search_support_valid if use_ct_joint_full
                     else trajectory_search_sampling["active"]],
                    device=self.device, dtype=torch.float32),
                "trajectory_search_gap_ratio": torch.tensor(
                    [ct_search_diagnostics.get("gap_ratio", 1.0)],
                    device=self.device, dtype=torch.float32),
                "trajectory_search_sigma_parallel": torch.tensor(
                    [ct_search_diagnostics.get("sigma_parallel", 0.0)],
                    device=self.device, dtype=torch.float32),
                "trajectory_search_sigma_perpendicular": torch.tensor(
                    [ct_search_diagnostics.get("sigma_perpendicular", 0.0)],
                    device=self.device, dtype=torch.float32),
            })
        if use_search_evidence_v2:
            data_dict.update({
                "search_v2_points": torch.tensor(
                    search_v2_points[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v2_point_valid_mask": torch.tensor(
                    search_v2_point_valid_mask[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v2_geometry_valid": torch.tensor(
                    [search_v2_sampling["active"]],
                    device=self.device, dtype=torch.float32),
                "search_v2_endpoint_xy": torch.tensor(
                    search_v2_endpoint_xy[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v2_query_delta_t": torch.tensor(
                    [search_v2_diagnostics.get(
                        "query_delta_t", effective_delta_t_list[0])],
                    device=self.device, dtype=torch.float32),
                "search_v2_gap_ratio": torch.tensor(
                    [search_v2_diagnostics.get("gap_ratio", 1.0)],
                    device=self.device, dtype=torch.float32),
                "search_v2_sigma_parallel": torch.tensor(
                    [search_v2_diagnostics.get("sigma_parallel", 0.0)],
                    device=self.device, dtype=torch.float32),
                "search_v2_sigma_perpendicular": torch.tensor(
                    [search_v2_diagnostics.get(
                        "sigma_perpendicular", 0.0)],
                    device=self.device, dtype=torch.float32),
                "search_v2_available_count": torch.tensor(
                    [search_v2_sampling["available_count"]],
                    device=self.device, dtype=torch.float32),
            })
        if use_search_evidence_v21:
            data_dict.update({
                "search_v21_points": torch.tensor(
                    search_v2_points[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v21_point_valid_mask": torch.tensor(
                    search_v2_point_valid_mask[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v21_point_source": torch.tensor(
                    search_v2_point_source[None, :],
                    device=self.device, dtype=torch.int64),
                "search_v21_geometry_valid": torch.tensor(
                    [search_v2_sampling["active"]],
                    device=self.device, dtype=torch.float32),
                "search_v21_endpoint_xy": torch.tensor(
                    search_v2_endpoint_xy[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v21_query_delta_t": torch.tensor(
                    [search_v2_diagnostics.get(
                        "query_delta_t", effective_delta_t_list[0])],
                    device=self.device, dtype=torch.float32),
                "search_v21_gap_ratio": torch.tensor(
                    [search_v2_diagnostics.get("gap_ratio", 1.0)],
                    device=self.device, dtype=torch.float32),
                "search_v21_sigma_parallel": torch.tensor(
                    [search_v2_diagnostics.get("sigma_parallel", 0.0)],
                    device=self.device, dtype=torch.float32),
                "search_v21_sigma_perpendicular": torch.tensor(
                    [search_v2_diagnostics.get(
                        "sigma_perpendicular", 0.0)],
                    device=self.device, dtype=torch.float32),
                "search_v21_available_count": torch.tensor(
                    [search_v2_sampling["available_count"]],
                    device=self.device, dtype=torch.float32),
                "search_v21_extension_count": torch.tensor(
                    [search_v2_sampling["extension_count"]],
                    device=self.device, dtype=torch.float32),
                "search_v21_overlap_count": torch.tensor(
                    [search_v2_sampling["overlap_count"]],
                    device=self.device, dtype=torch.float32),
            })
        if use_search_evidence_v22:
            data_dict.update({
                "search_v22_points": torch.tensor(
                    search_v2_points[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v22_point_valid_mask": torch.tensor(
                    search_v2_point_valid_mask[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v22_point_source": torch.tensor(
                    search_v2_point_source[None, :],
                    device=self.device, dtype=torch.long),
                "search_v22_geometry_valid": torch.tensor(
                    [search_v2_sampling["active"]],
                    device=self.device, dtype=torch.float32),
                "search_v22_support_anchor_xy": torch.tensor(
                    search_v2_endpoint_xy[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v22_query_delta_t": torch.tensor(
                    [search_v2_diagnostics.get(
                        "query_delta_t", effective_delta_t_list[0])],
                    device=self.device, dtype=torch.float32),
                "search_v22_gap_ratio": torch.tensor(
                    [search_v2_diagnostics.get("gap_ratio", 1.0)],
                    device=self.device, dtype=torch.float32),
                "search_v22_sigma_parallel": torch.tensor(
                    [search_v2_diagnostics.get("sigma_parallel", 0.0)],
                    device=self.device, dtype=torch.float32),
                "search_v22_sigma_perpendicular": torch.tensor(
                    [search_v2_diagnostics.get(
                        "sigma_perpendicular", 0.0)],
                    device=self.device, dtype=torch.float32),
                "search_v22_available_count": torch.tensor(
                    [search_v2_sampling["available_count"]],
                    device=self.device, dtype=torch.float32),
                "search_v22_extension_count": torch.tensor(
                    [search_v2_sampling["extension_count"]],
                    device=self.device, dtype=torch.float32),
                "search_v22_overlap_count": torch.tensor(
                    [search_v2_sampling["overlap_count"]],
                    device=self.device, dtype=torch.float32),
            })
        if use_search_evidence_v3:
            data_dict.update({
                "search_v3_points": torch.tensor(
                    search_v2_points[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v3_point_valid_mask": torch.tensor(
                    search_v2_point_valid_mask[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v3_point_source": torch.tensor(
                    search_v2_point_source[None, :],
                    device=self.device, dtype=torch.long),
                "search_v3_branch_source": torch.ones(
                    (1, search_v2_points.shape[0]),
                    device=self.device, dtype=torch.long),
                "search_v3_geometry_valid": torch.tensor(
                    [geometry_valid
                     if joint_contract_v2 else search_support_valid],
                    device=self.device, dtype=torch.float32),
                "search_v3_support_valid": torch.tensor(
                    [search_support_valid],
                    device=self.device, dtype=torch.float32),
                "search_v3_total_point_count": torch.tensor(
                    [joint_support['total_count']],
                    device=self.device, dtype=torch.float32),
                "search_v3_joint_extension_count": torch.tensor(
                    [joint_support['extension_count']],
                    device=self.device, dtype=torch.float32),
                "search_v3_extension_voxels": torch.tensor(
                    [joint_support['extension_voxels']],
                    device=self.device, dtype=torch.float32),
                "search_v3_endpoint_ratio": torch.tensor(
                    [endpoint_ratio], device=self.device,
                    dtype=torch.float32),
                "search_v3_support_anchor_xy": torch.tensor(
                    search_v2_endpoint_xy[None, :],
                    device=self.device, dtype=torch.float32),
                "search_v3_query_delta_t": torch.tensor(
                    [search_v2_diagnostics.get(
                        "query_delta_t", effective_delta_t_list[0])],
                    device=self.device, dtype=torch.float32),
                "search_v3_gap_ratio": torch.tensor(
                    [search_v2_diagnostics.get("gap_ratio", 1.0)],
                    device=self.device, dtype=torch.float32),
                "search_v3_sigma_parallel": torch.tensor(
                    [search_v2_diagnostics.get("sigma_parallel", 0.0)],
                    device=self.device, dtype=torch.float32),
                "search_v3_sigma_perpendicular": torch.tensor(
                    [search_v2_diagnostics.get(
                        "sigma_perpendicular", 0.0)],
                    device=self.device, dtype=torch.float32),
                "search_v3_available_count": torch.tensor(
                    [search_v2_sampling["available_count"]],
                    device=self.device, dtype=torch.float32),
                "search_v3_extension_count": torch.tensor(
                    [search_v2_sampling["extension_count"]],
                    device=self.device, dtype=torch.float32),
                "search_v3_overlap_count": torch.tensor(
                    [search_v2_sampling["overlap_count"]],
                    device=self.device, dtype=torch.float32),
                "search_v3_prior_source_id": torch.tensor(
                    [search_v2_diagnostics.get("source_id", 0)],
                    device=self.device, dtype=torch.long),
                "search_v3_support_truncated": torch.tensor(
                    [bool(search_v2_diagnostics.get("truncated", False))],
                    device=self.device, dtype=torch.float32),
                "search_v3_support_requested_extent": torch.tensor(
                    [[search_v2_diagnostics.get("requested_length", 0.0),
                      search_v2_diagnostics.get("requested_width", 0.0)]],
                    device=self.device, dtype=torch.float32),
                "search_v3_support_actual_extent": torch.tensor(
                    [[search_v2_diagnostics.get("length", 0.0),
                      search_v2_diagnostics.get("width", 0.0)]],
                    device=self.device, dtype=torch.float32),
            })
        if getattr(self.config, 'box_aware', False):
            stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
            hist_points_list = stack_points_split[:num_hist] 
            candidate_bc_prev_list= [
                points_utils.get_point_to_box_distance(hist_points[:, :3], ref_box)
                for hist_points, ref_box in zip(hist_points_list, ref_boxs)
            ]
            candidate_bc_this = np.zeros_like(candidate_bc_prev_list[0])
            candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
            candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]
