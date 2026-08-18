"""Shared causal tracking-state contract for training and inference."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch

from utils.sampling_utils import (
    deterministic_candidate_offset,
    deterministic_point_seed,
)


def box_yaw_radians(box):
    return float(box.orientation.radians * box.orientation.axis[-1])


def box_world_row(box):
    return np.concatenate((
        np.asarray(box.center, dtype=np.float64),
        np.asarray(box.wlh, dtype=np.float64),
        np.asarray([box_yaw_radians(box)], dtype=np.float64),
    ))


def stable_tracklet_partition(tracklet_key, seed=42):
    """Stable 70/15/15 split by whole tracklet, never by frame."""
    digest = hashlib.sha256(
        f"{int(seed)}::{str(tracklet_key)}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "dev"
    return "calibration"


def rotating_rollout_horizon(horizons, slot, epoch, slots):
    """Repeat a horizon cycle across slots and rotate it every epoch."""
    horizons = [int(value) for value in horizons]
    slots = int(slots)
    slot = int(slot)
    epoch = int(epoch)
    if (slots <= 0 or not horizons
            or any(value <= 0 for value in horizons)):
        raise ValueError(
            "rollout horizons must contain at least one positive value")
    if not 0 <= slot < slots or epoch < 0:
        raise ValueError("rollout slot/epoch is out of range")
    return horizons[(slot + epoch) % len(horizons)]


def build_recursive_input_contract(
        state, frame_id, hist_num, config, candidate_id=0, offsets=None):
    """Pure state-derived input contract shared by training and inference."""
    frame_id = int(frame_id)
    hist_num = int(hist_num)
    candidate_id = int(candidate_id)
    if frame_id <= 0 or hist_num <= 0:
        raise ValueError("recursive input requires frame_id>0 and hist_num>0")
    if offsets is None:
        offsets = list(range(1, hist_num + 1))
    offsets = [int(offset) for offset in offsets]
    if (len(offsets) != hist_num or any(offset <= 0 for offset in offsets)
            or any(offsets[index] >= offsets[index + 1]
                   for index in range(len(offsets) - 1))):
        raise ValueError(
            "recursive history offsets must be positive and increasing")
    frame_ids = [max(0, frame_id - offset) for offset in offsets]
    valid_mask = [int(frame_id - offset >= 0) for offset in offsets]
    state_contract = state.history_contract(frame_ids, valid_mask)
    seed_parts = (state.tracklet_key, frame_id, candidate_id)
    return {
        **state_contract,
        "history_frame_ids": frame_ids,
        "history_offsets": offsets,
        "candidate_id": candidate_id,
        "candidate_shared_transform": deterministic_candidate_offset(
            candidate_id, config, *seed_parts, 'coherent_recursive_view'),
        "point_sampling_seeds": np.asarray([
            deterministic_point_seed(
                config, *seed_parts, 'history', history_frame_id)
            for history_frame_id in frame_ids], dtype=np.int64),
        "current_sampling_seed": deterministic_point_seed(
            config, *seed_parts, 'current'),
    }


class OnlineRecursiveBatchSampler(torch.utils.data.Sampler):
    """Yield ordered tracklet slots with coherent candidate groups."""

    def __init__(self, dataset, slots=4, candidate_views=1, seed=42,
                 partition_seed=None,
                 partition="train", shadow_interval=2,
                 shadow_fraction=None, shadow_slots_per_event=1,
                 shadow_enabled=True):
        self.dataset = dataset
        self.slots = int(slots)
        self.candidate_views = int(candidate_views)
        self.seed = int(seed)
        self.partition_seed = int(
            self.seed if partition_seed is None else partition_seed)
        self.partition = str(partition)
        self.shadow_interval = max(1, int(shadow_interval))
        self.shadow_fraction = (
            None if shadow_fraction is None else float(shadow_fraction))
        self.shadow_enabled = bool(shadow_enabled)
        self.shadow_slots_per_event = int(shadow_slots_per_event)
        config = getattr(dataset, "config", None)
        self.grouped_candidate_carrier = str(getattr(
            config, "ct_candidate_policy", "legacy_spatial"
        )).strip().lower() in (
            "causal_b1_boundary", "causal_temporal_uniform")
        self.epoch = 0
        if self.slots <= 0 or self.candidate_views <= 0:
            raise ValueError("online recursive slots/views must be positive")
        if self.shadow_enabled and self.shadow_slots_per_event != 1:
            raise ValueError(
                "Joint Full requires exactly one explicit shadow slot per "
                "scheduled event")
        if self.candidate_views != int(dataset.num_candidates):
            raise ValueError(
                "online candidate views must equal dataset.num_candidates")
        self.tracklet_ids = []
        self.prediction_frames = 0
        base = dataset.dataset
        for tracklet_id in range(base.get_num_tracklets()):
            key = (
                base.get_tracklet_key(tracklet_id)
                if hasattr(base, "get_tracklet_key") else str(tracklet_id))
            if (stable_tracklet_partition(key, self.partition_seed)
                    != self.partition):
                continue
            frame_count = int(base.get_num_frames_tracklet(tracklet_id))
            if frame_count > 1:
                self.tracklet_ids.append(tracklet_id)
                self.prediction_frames += frame_count - 1
        if len(self.tracklet_ids) < self.slots:
            raise ValueError(
                "online recursive partition has fewer tracklets than slots")
        self.slot_tracklets = [[] for _ in range(self.slots)]
        self.slot_prediction_frames = [0 for _ in range(self.slots)]
        # Longest-first greedy assignment keeps the causal slots close
        # in workload, which minimizes the final tail that cannot form a full
        # complete optimizer batch.
        ordered_tracklets = sorted(
            self.tracklet_ids,
            key=lambda item: int(base.get_num_frames_tracklet(item)),
            reverse=True)
        for tracklet_id in ordered_tracklets:
            slot = min(
                range(self.slots),
                key=lambda item: self.slot_prediction_frames[item])
            self.slot_tracklets[slot].append(tracklet_id)
            self.slot_prediction_frames[slot] += (
                int(base.get_num_frames_tracklet(tracklet_id)) - 1)

    def __len__(self):
        return min(self.slot_prediction_frames)

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self):
        iterator_epoch = int(self.epoch)
        rng = np.random.default_rng(self.seed + iterator_epoch)
        queues = [
            list(rng.permutation(tracklets).astype(int))
            for tracklets in self.slot_tracklets]
        active = [
            [queue.pop(0), 1] if queue else None for queue in queues]
        batch_index = 0
        try:
            # A partial set of slots would silently change the nominal batch
            # size and H=3 overhead. Drop only the irreducible balanced tail.
            while all(item is not None for item in active):
                active_slots = [
                    slot for slot, item in enumerate(active)
                    if item is not None]
                shadow_slot = -1
                if (self.shadow_enabled
                        and batch_index % self.shadow_interval == 0):
                    eligible_shadow_slots = []
                    for slot in active_slots:
                        tracklet_id, frame_id = active[slot]
                        frame_count = int(
                            self.dataset.dataset.get_num_frames_tracklet(
                                tracklet_id))
                        if frame_id + 2 < frame_count:
                            eligible_shadow_slots.append(slot)
                    if eligible_shadow_slots:
                        shadow_event = batch_index // self.shadow_interval
                        shadow_slot = eligible_shadow_slots[
                            shadow_event % len(eligible_shadow_slots)]
                batch = []
                for slot in active_slots:
                    tracklet_id, frame_id = active[slot]
                    frame_count = int(
                        self.dataset.dataset.get_num_frames_tracklet(
                            tracklet_id))
                    can_shadow = bool(
                        slot == shadow_slot and frame_id + 2 < frame_count)
                    candidate_ids = (
                        (0,) if self.grouped_candidate_carrier
                        else range(self.candidate_views))
                    for candidate_id in candidate_ids:
                        batch.append((
                            iterator_epoch, batch_index, slot,
                            int(tracklet_id), int(frame_id), int(candidate_id),
                            bool(can_shadow and candidate_id == 0),
                        ))
                yield batch
                batch_index += 1
                next_active = list(active)
                for slot in active_slots:
                    tracklet_id, frame_id = active[slot]
                    next_frame = frame_id + 1
                    frame_count = int(
                        self.dataset.dataset.get_num_frames_tracklet(
                            tracklet_id))
                    if next_frame < frame_count:
                        next_active[slot] = [tracklet_id, next_frame]
                    elif queues[slot]:
                        next_active[slot] = [queues[slot].pop(0), 1]
                    else:
                        next_active[slot] = None
                active = next_active
        finally:
            # Preserve standalone iteration ergonomics. Lightning calls
            # ``set_epoch(current_epoch)`` before every epoch, so a resume
            # always overrides this convenience increment deterministically.
            if self.epoch == iterator_epoch:
                self.epoch = iterator_epoch + 1


@dataclass
class RecursiveTrackState:
    """Detached deployed predictions indexed by absolute tracklet frame."""

    tracklet_id: int
    tracklet_key: str
    first_box: object
    timestamps: dict[int, float | None] = field(default_factory=dict)
    predictions: dict[int, object] = field(default_factory=dict)
    # Training-only rollout metadata.  Inference never calls ``reseed_history``
    # and therefore keeps the original append-only state transition contract.
    rollout_horizon: int | None = None
    last_reseed_before_frame: int = 1
    reseed_count: int = 0

    def __post_init__(self):
        self.first_box = copy.deepcopy(self.first_box)
        self.predictions.setdefault(0, copy.deepcopy(self.first_box))

    @property
    def target_size(self):
        return np.asarray(self.first_box.wlh, dtype=np.float64).copy()

    def clone(self):
        return copy.deepcopy(self)

    def append(self, frame_id, box, timestamp=None):
        frame_id = int(frame_id)
        if frame_id <= 0:
            raise ValueError("recursive predictions may only append frame_id>0")
        expected = max(self.predictions) + 1
        if frame_id != expected:
            raise ValueError(
                f"recursive state expected frame {expected}, got {frame_id}")
        self.predictions[frame_id] = copy.deepcopy(box)
        self.timestamps[frame_id] = timestamp

    def reseed_history(
            self, frame_ids, boxes, timestamps=None, *, before_frame_id,
            rollout_horizon):
        """Replace past input state at a deterministic training boundary.

        The method may only rewrite frames that have already been observed.
        It cannot append the current frame or inspect future frames, which
        keeps expert intervention separate from current-frame supervision.
        Repeated padded frame ids (for example ``[0, 0, 0]``) are allowed.
        """
        frame_ids = [int(value) for value in frame_ids]
        boxes = list(boxes)
        before_frame_id = int(before_frame_id)
        rollout_horizon = int(rollout_horizon)
        if len(frame_ids) != len(boxes):
            raise ValueError("reseed frame ids and boxes must align")
        if timestamps is None:
            timestamps = [None] * len(frame_ids)
        else:
            timestamps = list(timestamps)
        if len(timestamps) != len(frame_ids):
            raise ValueError("reseed timestamps and frame ids must align")
        if before_frame_id <= 0 or rollout_horizon <= 0:
            raise ValueError("reseed boundary and horizon must be positive")
        if any(frame_id < 0 or frame_id >= before_frame_id
               for frame_id in frame_ids):
            raise ValueError("reseed may only use frames before the query")
        if max(self.predictions) != before_frame_id - 1:
            raise ValueError(
                "reseed requires a causally complete state up to t-1")
        for frame_id, box, timestamp in zip(frame_ids, boxes, timestamps):
            if frame_id not in self.predictions:
                raise KeyError(
                    f"reseed cannot introduce unseen frame {frame_id}")
            self.predictions[frame_id] = copy.deepcopy(box)
            self.timestamps[frame_id] = timestamp
        self.rollout_horizon = rollout_horizon
        self.last_reseed_before_frame = before_frame_id
        self.reseed_count += 1

    def rollout_age(self, frame_id):
        """Number of learner-written transitions consumed by this query."""
        frame_id = int(frame_id)
        if frame_id <= 0:
            return 0
        return max(0, frame_id - int(self.last_reseed_before_frame))

    def history_boxes(self, frame_ids, valid_mask):
        if len(frame_ids) != len(valid_mask):
            raise ValueError("history frame ids and valid mask must align")
        boxes = []
        for frame_id, valid in zip(frame_ids, valid_mask):
            frame_id = int(frame_id)
            if int(valid) and frame_id not in self.predictions:
                raise KeyError(
                    f"recursive state lacks frame {frame_id} for "
                    f"{self.tracklet_key}")
            boxes.append(copy.deepcopy(
                self.predictions.get(frame_id, self.first_box)))
        return boxes

    def history_contract(self, frame_ids, valid_mask):
        boxes = self.history_boxes(frame_ids, valid_mask)
        return {
            "history_boxes_world": np.stack(
                [box_world_row(box) for box in boxes], axis=0),
            "history_valid_mask": np.asarray(valid_mask, dtype=np.int64),
            "history_timestamps": [
                self.timestamps.get(int(frame_id)) for frame_id in frame_ids],
            "target_size": self.target_size,
            "tracklet_key": self.tracklet_key,
        }

    @property
    def results_bbs(self):
        return [self.predictions[index] for index in sorted(self.predictions)]


def commit_canonical_prediction(
        state, candidate_id, frame_id, box, timestamp=None):
    """Commit candidate 0 only; recovery views can never mutate state."""
    if int(candidate_id) != 0:
        return False
    state.append(frame_id, box, timestamp)
    return True
