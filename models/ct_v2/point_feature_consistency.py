"""Canonical point-feature temporal consistency for CT-SeqTrack.

The loss is deliberately training-only and parameter-free.  Point matching is
computed from detached XYZ/GT boxes, while gradients flow only through the
point-aligned features produced by the first two FeaturePointNet layers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def chronological_frame_indices(
        num_frames: int, device: torch.device | None = None) -> torch.Tensor:
    """Return the fixed CT-SeqTrack chronological order.

    Dataset tensors are stored as ``[t-1, t-2, ..., t-H, t]``.  Physical-time
    interventions must never change frame topology, so reordering is based on
    tensor position rather than timestamp values.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    history = torch.arange(
        num_frames - 2, -1, -1, dtype=torch.long, device=device)
    current = torch.tensor(
        [num_frames - 1], dtype=torch.long, device=device)
    return torch.cat((history, current), dim=0)


def canonicalize_points(
        points: torch.Tensor,
        boxes: torch.Tensor,
        degrees: bool = False) -> torch.Tensor:
    """Transform points from the common anchor frame into GT object frames.

    Args:
        points: ``[..., N, 3]`` XYZ tensor.
        boxes: ``[..., 4]`` box tensor containing center XYZ and planar yaw.
        degrees: Interpret yaw as degrees when true; radians otherwise.
    """
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [..., N, 3]")
    if boxes.shape != points.shape[:-2] + (4,):
        raise ValueError(
            "boxes must match the leading point dimensions and end in 4")

    centered = points - boxes[..., None, :3]
    yaw = boxes[..., 3]
    if degrees:
        yaw = torch.deg2rad(yaw)
    cosine = torch.cos(yaw)[..., None]
    sine = torch.sin(yaw)[..., None]
    x_coord = cosine * centered[..., 0] - sine * centered[..., 1]
    y_coord = sine * centered[..., 0] + cosine * centered[..., 1]
    return torch.stack((x_coord, y_coord, centered[..., 2]), dim=-1)


def _merge_duplicate_points(
        coordinates: torch.Tensor,
        features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge exact duplicate XYZ samples and average their features."""
    unique_coordinates, inverse, counts = torch.unique(
        coordinates, dim=0, return_inverse=True, return_counts=True)
    merged_features = features.new_zeros(
        (unique_coordinates.shape[0], features.shape[-1]))
    merged_features.index_add_(0, inverse, features)
    merged_features = merged_features / counts.to(
        device=features.device, dtype=features.dtype).unsqueeze(-1)
    return unique_coordinates, merged_features


class PointFeatureTemporalConsistencyLoss(nn.Module):
    """One-way canonical NN consistency with optional physical-time weighting."""

    def __init__(
            self,
            distance_threshold: float = 0.3,
            min_correspondences: int = 3,
            time_weighting: bool = True,
            time_scale: float = 0.5,
            time_weight_min: float = 0.5,
            time_weight_max: float = 3.0,
            degrees: bool = False):
        super().__init__()
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        if min_correspondences <= 0:
            raise ValueError("min_correspondences must be positive")
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if time_weight_min <= 0 or time_weight_max < time_weight_min:
            raise ValueError("invalid time-weight clamp")

        self.distance_threshold = float(distance_threshold)
        self.min_correspondences = int(min_correspondences)
        self.time_weighting = bool(time_weighting)
        self.time_scale = float(time_scale)
        self.time_weight_min = float(time_weight_min)
        self.time_weight_max = float(time_weight_max)
        self.degrees = bool(degrees)

    @staticmethod
    def _validate_inputs(
            point_features: torch.Tensor,
            points: torch.Tensor,
            seg_mask: torch.Tensor,
            boxes: torch.Tensor,
            history_valid_mask: torch.Tensor,
            timestamps: torch.Tensor) -> Tuple[int, int, int]:
        if point_features.ndim != 4:
            raise ValueError("point_features must have shape [B, L, N, C]")
        batch_size, num_frames, num_points, _ = point_features.shape
        if points.shape != (batch_size, num_frames, num_points, 3):
            raise ValueError("points must have shape [B, L, N, 3]")
        if seg_mask.shape != (batch_size, num_frames, num_points):
            raise ValueError("seg_mask must have shape [B, L, N]")
        if boxes.shape != (batch_size, num_frames, 4):
            raise ValueError("boxes must have shape [B, L, 4]")
        if history_valid_mask.shape != (batch_size, num_frames - 1):
            raise ValueError(
                "history_valid_mask must have shape [B, L - 1]")
        if timestamps.shape != (batch_size, num_frames):
            raise ValueError("timestamps must have shape [B, L]")
        return batch_size, num_frames, num_points

    def forward(
            self,
            point_features: torch.Tensor,
            points: torch.Tensor,
            seg_mask: torch.Tensor,
            boxes: torch.Tensor,
            history_valid_mask: torch.Tensor,
            timestamps: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, num_frames, _ = self._validate_inputs(
            point_features, points, seg_mask, boxes, history_valid_mask,
            timestamps)
        device = point_features.device
        zero = point_features.sum() * 0.0
        stat_zero = zero.detach()

        order = chronological_frame_indices(num_frames, device=device)
        point_features = point_features.index_select(1, order)
        points = points.to(device=device).index_select(1, order)
        seg_mask = seg_mask.to(device=device).index_select(1, order)
        boxes = boxes.to(device=device).index_select(1, order)
        timestamps = timestamps.to(device=device).index_select(1, order)

        frame_valid = torch.cat((
            history_valid_mask.to(device=device, dtype=torch.bool),
            torch.ones(
                batch_size, 1, device=device, dtype=torch.bool),
        ), dim=1).index_select(1, order)

        with torch.no_grad():
            canonical_points = canonicalize_points(
                points.detach().float(), boxes.detach().float(),
                degrees=self.degrees)

        sample_raw_losses: List[torch.Tensor] = []
        sample_weighted_losses: List[torch.Tensor] = []
        match_counts: List[torch.Tensor] = []
        match_distances: List[torch.Tensor] = []
        foreground_before: List[torch.Tensor] = []
        foreground_after: List[torch.Tensor] = []
        feature_stds: List[torch.Tensor] = []
        normalized_weights: List[torch.Tensor] = []
        total_valid_pairs = 0

        for batch_index in range(batch_size):
            frame_data: List[
                Tuple[torch.Tensor, torch.Tensor] | None] = []
            for frame_index in range(num_frames):
                if not bool(frame_valid[batch_index, frame_index].item()):
                    frame_data.append(None)
                    continue
                foreground = seg_mask[
                    batch_index, frame_index].detach() > 0
                before_count = int(foreground.sum().item())
                foreground_before.append(
                    zero.new_tensor(float(before_count)).detach())
                if before_count == 0:
                    foreground_after.append(stat_zero)
                    frame_data.append(None)
                    continue

                coordinates = canonical_points[
                    batch_index, frame_index, foreground]
                features = point_features[
                    batch_index, frame_index, foreground].float()
                coordinates, features = _merge_duplicate_points(
                    coordinates, features)
                foreground_after.append(
                    zero.new_tensor(float(coordinates.shape[0])).detach())
                with torch.no_grad():
                    feature_stds.append(
                        features.detach().std(
                            dim=0, unbiased=False).mean())
                frame_data.append((coordinates, features))

            pair_losses: List[torch.Tensor] = []
            raw_time_weights: List[torch.Tensor] = []
            for early_index in range(num_frames - 1):
                early_data = frame_data[early_index]
                if early_data is None:
                    continue
                for late_index in range(early_index + 1, num_frames):
                    late_data = frame_data[late_index]
                    if late_data is None:
                        continue
                    early_coordinates, early_features = early_data
                    late_coordinates, late_features = late_data
                    with torch.no_grad():
                        distances = torch.cdist(
                            early_coordinates.unsqueeze(0).float(),
                            late_coordinates.unsqueeze(0).float()).squeeze(0)
                        nearest_distance, nearest_index = distances.min(dim=1)
                        keep = nearest_distance < self.distance_threshold
                        correspondence_count = int(keep.sum().item())
                    if correspondence_count < self.min_correspondences:
                        continue

                    pair_loss = F.smooth_l1_loss(
                        early_features[keep],
                        late_features[nearest_index[keep]],
                        reduction="none").mean(dim=-1).mean()
                    pair_losses.append(pair_loss)
                    match_counts.append(
                        zero.new_tensor(
                            float(correspondence_count)).detach())
                    match_distances.append(
                        nearest_distance[keep].mean().detach())
                    with torch.no_grad():
                        delta_t = torch.abs(
                            timestamps[batch_index, late_index].float()
                            - timestamps[batch_index, early_index].float())
                        raw_time_weights.append(torch.clamp(
                            delta_t / self.time_scale,
                            min=self.time_weight_min,
                            max=self.time_weight_max))

            if not pair_losses:
                continue
            total_valid_pairs += len(pair_losses)
            pair_loss_tensor = torch.stack(pair_losses)
            raw_weight_tensor = torch.stack(raw_time_weights).to(
                device=device, dtype=pair_loss_tensor.dtype)
            normalized_weight = (
                raw_weight_tensor / raw_weight_tensor.mean().clamp_min(1e-12))
            if not self.time_weighting:
                normalized_weight = torch.ones_like(normalized_weight)
            normalized_weights.append(normalized_weight.detach())
            sample_raw_losses.append(pair_loss_tensor.mean())
            sample_weighted_losses.append(
                (normalized_weight * pair_loss_tensor).mean())

        if sample_raw_losses:
            loss_raw = torch.stack(sample_raw_losses).mean()
            loss_weighted = torch.stack(sample_weighted_losses).mean()
        else:
            loss_raw = zero
            loss_weighted = zero
        loss = loss_weighted if self.time_weighting else loss_raw

        def detached_mean(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return stat_zero
            return torch.stack(values).float().mean().detach()

        if normalized_weights:
            all_weights = torch.cat(normalized_weights)
            weight_mean = all_weights.mean().detach()
            weight_max = all_weights.max().detach()
        else:
            weight_mean = stat_zero
            weight_max = stat_zero

        return {
            "loss": loss,
            "loss_pftc_raw": loss_raw,
            "loss_pftc_weighted": loss_weighted,
            "pftc_valid_sample_ratio": zero.new_tensor(
                len(sample_raw_losses) / float(batch_size)).detach(),
            "pftc_valid_pair_count": zero.new_tensor(
                float(total_valid_pairs)).detach(),
            "pftc_valid_pairs_per_sample": zero.new_tensor(
                total_valid_pairs / float(batch_size)).detach(),
            "pftc_match_count": detached_mean(match_counts),
            "pftc_match_distance": detached_mean(match_distances),
            "pftc_fg_points_before": detached_mean(foreground_before),
            "pftc_fg_points_after": detached_mean(foreground_after),
            "pftc_time_weight_mean": weight_mean,
            "pftc_time_weight_max": weight_max,
            "pftc_fg_feature_std": detached_mean(feature_stds),
        }
