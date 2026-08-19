"""Shared causal tracking-state contract for training and inference."""

from __future__ import annotations

import copy
import hashlib
import json
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
    return np.concatenate(
        (
            np.asarray(box.center, dtype=np.float64),
            np.asarray(box.wlh, dtype=np.float64),
            np.asarray([box_yaw_radians(box)], dtype=np.float64),
        )
    )


def stable_tracklet_partition(tracklet_key, seed=42):
    """Stable 70/15/15 split by whole tracklet, never by frame."""
    digest = hashlib.sha256(
        f"{int(seed)}::{str(tracklet_key)}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "dev"
    return "calibration"


SCENE_PARTITIONS = ("train", "dev", "calibration_select", "calibration_audit")


def _content_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scene_partition_counts(group_count):
    """Return deterministic 70/15/7.5/7.5 counts with no empty split."""
    group_count = int(group_count)
    if group_count < len(SCENE_PARTITIONS):
        raise ValueError(
            "scene_v2 requires at least four scenes so every partition is " "non-empty"
        )
    ratios = (0.70, 0.15, 0.075, 0.075)
    counts = [int(np.floor(group_count * ratio)) for ratio in ratios]
    for index in range(len(counts)):
        if counts[index] == 0:
            counts[index] = 1
    while sum(counts) > group_count:
        donor = max(
            range(len(counts)),
            key=lambda index: (counts[index] - 1, ratios[index], -index),
        )
        if counts[donor] <= 1:
            raise ValueError("cannot make every scene partition non-empty")
        counts[donor] -= 1
    fractions = [
        group_count * ratio - np.floor(group_count * ratio) for ratio in ratios
    ]
    while sum(counts) < group_count:
        receiver = max(
            range(len(counts)),
            key=lambda index: (fractions[index], ratios[index], -index),
        )
        counts[receiver] += 1
        fractions[receiver] = -1.0
        if all(value < 0 for value in fractions):
            fractions = [
                group_count * ratio - np.floor(group_count * ratio) for ratio in ratios
            ]
    return tuple(counts)


def build_scene_partition_manifest(dataset, seed=42):
    """Build and validate the shared scene-level v25 partition manifest.

    Tracklet identity remains the target identity.  Partition membership is
    assigned only from ``get_partition_group_key`` (a scene), so no target or
    physical frame can leak between the four protocol partitions.
    """
    if not hasattr(dataset, "get_partition_group_key"):
        raise TypeError("scene_v2 dataset must implement get_partition_group_key")
    seed = int(seed)
    groups = {}
    frame_owners = {}
    tracklets = []
    for tracklet_id in range(int(dataset.get_num_tracklets())):
        group_key = str(dataset.get_partition_group_key(tracklet_id))
        tracklet_key = str(
            dataset.get_tracklet_key(tracklet_id)
            if hasattr(dataset, "get_tracklet_key")
            else tracklet_id
        )
        frame_count = int(dataset.get_num_frames_tracklet(tracklet_id))
        if not group_key or not tracklet_key or frame_count <= 0:
            raise ValueError("scene manifest found an empty identity or tracklet")
        group = groups.setdefault(
            group_key,
            {
                "group_key": group_key,
                "tracklet_count": 0,
                "frame_count": 0,
                "unique_frame_tokens": set(),
            },
        )
        group["tracklet_count"] += 1
        group["frame_count"] += frame_count
        tracklets.append(
            {
                "tracklet_id": int(tracklet_id),
                "tracklet_key": tracklet_key,
                "group_key": group_key,
                "frame_count": frame_count,
            }
        )
        if hasattr(dataset, "get_frame_token"):
            for frame_id in range(frame_count):
                frame_token = str(dataset.get_frame_token(tracklet_id, frame_id))
                if not frame_token:
                    raise ValueError("scene manifest found an empty frame token")
                owner = frame_owners.setdefault(frame_token, group_key)
                if owner != group_key:
                    raise ValueError(
                        "physical frame token belongs to multiple scenes: "
                        f"{frame_token}"
                    )
                group["unique_frame_tokens"].add(frame_token)

    ordered_groups = sorted(
        groups,
        key=lambda key: (
            hashlib.sha256(f"{seed}::{key}".encode("utf-8")).hexdigest(),
            key,
        ),
    )
    counts = _scene_partition_counts(len(ordered_groups))
    assignments = {}
    cursor = 0
    for partition, count in zip(SCENE_PARTITIONS, counts):
        for group_key in ordered_groups[cursor : cursor + count]:
            assignments[group_key] = partition
        cursor += count
    if cursor != len(ordered_groups):
        raise RuntimeError("scene partition allocation did not cover all scenes")

    group_rows = []
    for group_key in sorted(groups):
        group = groups[group_key]
        group_rows.append(
            {
                "group_key": group_key,
                "partition": assignments[group_key],
                "tracklet_count": int(group["tracklet_count"]),
                "frame_count": int(group["frame_count"]),
                "unique_frame_count": int(len(group["unique_frame_tokens"])),
            }
        )
    tracklet_rows = []
    seen_tracklet_keys = set()
    for row in sorted(tracklets, key=lambda value: value["tracklet_key"]):
        if row["tracklet_key"] in seen_tracklet_keys:
            raise ValueError(
                "scene manifest found duplicate tracklet key: " f"{row['tracklet_key']}"
            )
        seen_tracklet_keys.add(row["tracklet_key"])
        tracklet_rows.append(
            {
                **row,
                "partition": assignments[row["group_key"]],
            }
        )
    partition_summary = {}
    for partition in SCENE_PARTITIONS:
        partition_groups = [row for row in group_rows if row["partition"] == partition]
        partition_tracklets = [
            row for row in tracklet_rows if row["partition"] == partition
        ]
        if not partition_groups or not partition_tracklets:
            raise ValueError(f"scene partition {partition!r} is empty")
        partition_summary[partition] = {
            "scene_count": len(partition_groups),
            "tracklet_count": len(partition_tracklets),
            "frame_count": sum(row["frame_count"] for row in partition_tracklets),
            "prediction_frame_count": sum(
                max(0, row["frame_count"] - 1) for row in partition_tracklets
            ),
            "content_sha256": _content_sha256(
                {
                    "partition": partition,
                    "groups": [row["group_key"] for row in partition_groups],
                    "tracklets": [
                        {
                            "tracklet_key": row["tracklet_key"],
                            "frame_count": row["frame_count"],
                        }
                        for row in partition_tracklets
                    ],
                }
            ),
        }
    content = {
        "schema": "ct_seqtrack.scene_partition_manifest.v1",
        "partition_scheme": "scene_v2",
        "seed": seed,
        "groups": group_rows,
        "tracklets": tracklet_rows,
        "partitions": partition_summary,
    }
    return {**content, "content_sha256": _content_sha256(content)}


def scene_partition_tracklet_ids(manifest, partition):
    """Resolve tracklet ids after re-validating a scene manifest digest."""
    partition = str(partition)
    if partition not in SCENE_PARTITIONS:
        raise ValueError(f"unknown scene partition {partition!r}")
    payload = dict(manifest)
    observed = payload.pop("content_sha256", None)
    if observed != _content_sha256(payload):
        raise ValueError("scene partition manifest SHA256 mismatch")
    rows = [
        row for row in payload.get("tracklets", []) if row.get("partition") == partition
    ]
    if not rows:
        raise ValueError(f"scene partition {partition!r} is empty")
    return [int(row["tracklet_id"]) for row in rows]


def scene_partition_identity_sha256(manifest, partition):
    scene_partition_tracklet_ids(manifest, partition)
    return str(manifest["partitions"][str(partition)]["content_sha256"])


def rotating_rollout_horizon(horizons, slot, epoch, slots):
    """Repeat a horizon cycle across slots and rotate it every epoch."""
    horizons = [int(value) for value in horizons]
    slots = int(slots)
    slot = int(slot)
    epoch = int(epoch)
    if slots <= 0 or not horizons or any(value <= 0 for value in horizons):
        raise ValueError("rollout horizons must contain at least one positive value")
    if not 0 <= slot < slots or epoch < 0:
        raise ValueError("rollout slot/epoch is out of range")
    return horizons[(slot + epoch) % len(horizons)]


def build_recursive_input_contract(
    state, frame_id, hist_num, config, candidate_id=0, offsets=None
):
    """Pure state-derived input contract shared by training and inference."""
    frame_id = int(frame_id)
    hist_num = int(hist_num)
    candidate_id = int(candidate_id)
    if frame_id <= 0 or hist_num <= 0:
        raise ValueError("recursive input requires frame_id>0 and hist_num>0")
    if offsets is None:
        offsets = list(range(1, hist_num + 1))
    offsets = [int(offset) for offset in offsets]
    if (
        len(offsets) != hist_num
        or any(offset <= 0 for offset in offsets)
        or any(
            offsets[index] >= offsets[index + 1] for index in range(len(offsets) - 1)
        )
    ):
        raise ValueError("recursive history offsets must be positive and increasing")
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
            candidate_id, config, *seed_parts, "coherent_recursive_view"
        ),
        "point_sampling_seeds": np.asarray(
            [
                deterministic_point_seed(
                    config, *seed_parts, "history", history_frame_id
                )
                for history_frame_id in frame_ids
            ],
            dtype=np.int64,
        ),
        "current_sampling_seed": deterministic_point_seed(
            config, *seed_parts, "current"
        ),
    }


class OnlineRecursiveBatchSampler(torch.utils.data.Sampler):
    """Yield ordered tracklet slots with coherent candidate groups."""

    def __init__(
        self,
        dataset,
        slots=4,
        candidate_views=1,
        seed=42,
        partition_seed=None,
        partition="train",
        partition_scheme="tracklet_v1",
        shadow_interval=2,
        shadow_fraction=None,
        shadow_slots_per_event=1,
        shadow_enabled=True,
    ):
        self.dataset = dataset
        self.slots = int(slots)
        self.candidate_views = int(candidate_views)
        self.seed = int(seed)
        self.partition_seed = int(
            self.seed if partition_seed is None else partition_seed
        )
        self.partition = str(partition)
        self.partition_scheme = str(partition_scheme).strip().lower()
        self.shadow_interval = max(1, int(shadow_interval))
        self.shadow_fraction = (
            None if shadow_fraction is None else float(shadow_fraction)
        )
        self.shadow_enabled = bool(shadow_enabled)
        self.shadow_slots_per_event = int(shadow_slots_per_event)
        config = getattr(dataset, "config", None)
        self.grouped_candidate_carrier = str(
            getattr(config, "ct_candidate_policy", "legacy_spatial")
        ).strip().lower() in ("causal_b1_boundary", "causal_temporal_uniform")
        self.epoch = 0
        if self.slots <= 0 or self.candidate_views <= 0:
            raise ValueError("online recursive slots/views must be positive")
        if self.shadow_enabled and self.shadow_slots_per_event != 1:
            raise ValueError(
                "Joint Full requires exactly one explicit shadow slot per "
                "scheduled event"
            )
        if self.candidate_views != int(dataset.num_candidates):
            raise ValueError("online candidate views must equal dataset.num_candidates")
        self.tracklet_ids = []
        self.prediction_frames = 0
        base = dataset.dataset
        self.partition_manifest = None
        if self.partition_scheme == "scene_v2":
            self.partition_manifest = build_scene_partition_manifest(
                base, self.partition_seed
            )
            selected_ids = set(
                scene_partition_tracklet_ids(self.partition_manifest, self.partition)
            )
        elif self.partition_scheme in ("tracklet_v1", "legacy"):
            selected_ids = None
        else:
            raise ValueError("partition_scheme must be tracklet_v1 or scene_v2")
        for tracklet_id in range(base.get_num_tracklets()):
            key = (
                base.get_tracklet_key(tracklet_id)
                if hasattr(base, "get_tracklet_key")
                else str(tracklet_id)
            )
            if selected_ids is not None:
                included = tracklet_id in selected_ids
            else:
                included = (
                    stable_tracklet_partition(key, self.partition_seed)
                    == self.partition
                )
            if not included:
                continue
            frame_count = int(base.get_num_frames_tracklet(tracklet_id))
            if frame_count > 1:
                self.tracklet_ids.append(tracklet_id)
                self.prediction_frames += frame_count - 1
        if len(self.tracklet_ids) < self.slots:
            raise ValueError(
                "online recursive partition has fewer tracklets than slots"
            )
        self.slot_tracklets = [[] for _ in range(self.slots)]
        self.slot_prediction_frames = [0 for _ in range(self.slots)]
        # Longest-first greedy assignment keeps the causal slots close
        # in workload, which minimizes the final tail that cannot form a full
        # complete optimizer batch.
        ordered_tracklets = sorted(
            self.tracklet_ids,
            key=lambda item: int(base.get_num_frames_tracklet(item)),
            reverse=True,
        )
        for tracklet_id in ordered_tracklets:
            slot = min(
                range(self.slots), key=lambda item: self.slot_prediction_frames[item]
            )
            self.slot_tracklets[slot].append(tracklet_id)
            self.slot_prediction_frames[slot] += (
                int(base.get_num_frames_tracklet(tracklet_id)) - 1
            )

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
            for tracklets in self.slot_tracklets
        ]
        active = [[queue.pop(0), 1] if queue else None for queue in queues]
        batch_index = 0
        try:
            # A partial set of slots would silently change the nominal batch
            # size and H=3 overhead. Drop only the irreducible balanced tail.
            while all(item is not None for item in active):
                active_slots = [
                    slot for slot, item in enumerate(active) if item is not None
                ]
                shadow_slot = -1
                if self.shadow_enabled and batch_index % self.shadow_interval == 0:
                    eligible_shadow_slots = []
                    for slot in active_slots:
                        tracklet_id, frame_id = active[slot]
                        frame_count = int(
                            self.dataset.dataset.get_num_frames_tracklet(tracklet_id)
                        )
                        if frame_id + 2 < frame_count:
                            eligible_shadow_slots.append(slot)
                    if eligible_shadow_slots:
                        shadow_event = batch_index // self.shadow_interval
                        shadow_slot = eligible_shadow_slots[
                            shadow_event % len(eligible_shadow_slots)
                        ]
                batch = []
                for slot in active_slots:
                    tracklet_id, frame_id = active[slot]
                    frame_count = int(
                        self.dataset.dataset.get_num_frames_tracklet(tracklet_id)
                    )
                    can_shadow = bool(
                        slot == shadow_slot and frame_id + 2 < frame_count
                    )
                    candidate_ids = (
                        (0,)
                        if self.grouped_candidate_carrier
                        else range(self.candidate_views)
                    )
                    for candidate_id in candidate_ids:
                        batch.append(
                            (
                                iterator_epoch,
                                batch_index,
                                slot,
                                int(tracklet_id),
                                int(frame_id),
                                int(candidate_id),
                                bool(can_shadow and candidate_id == 0),
                            )
                        )
                yield batch
                batch_index += 1
                next_active = list(active)
                for slot in active_slots:
                    tracklet_id, frame_id = active[slot]
                    next_frame = frame_id + 1
                    frame_count = int(
                        self.dataset.dataset.get_num_frames_tracklet(tracklet_id)
                    )
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
    observation_diagnostics: dict[int, np.ndarray] = field(default_factory=dict)
    observation_diagnostic_valid: dict[int, bool] = field(default_factory=dict)
    # Training-only rollout metadata.  Inference never calls ``reseed_history``
    # and therefore keeps the original append-only state transition contract.
    rollout_horizon: int | None = None
    last_reseed_before_frame: int = 1
    reseed_count: int = 0

    def __post_init__(self):
        self.first_box = copy.deepcopy(self.first_box)
        self.predictions.setdefault(0, copy.deepcopy(self.first_box))
        self.observation_diagnostics.setdefault(0, np.zeros(6, dtype=np.float32))
        self.observation_diagnostic_valid.setdefault(0, False)

    @property
    def target_size(self):
        return np.asarray(self.first_box.wlh, dtype=np.float64).copy()

    def clone(self):
        return copy.deepcopy(self)

    def append(
        self,
        frame_id,
        box,
        timestamp=None,
        observation_diagnostics=None,
        diagnostic_valid=False,
    ):
        frame_id = int(frame_id)
        if frame_id <= 0:
            raise ValueError("recursive predictions may only append frame_id>0")
        expected = max(self.predictions) + 1
        if frame_id != expected:
            raise ValueError(
                f"recursive state expected frame {expected}, got {frame_id}"
            )
        self.predictions[frame_id] = copy.deepcopy(box)
        self.timestamps[frame_id] = timestamp
        if observation_diagnostics is None:
            diagnostic = np.zeros(6, dtype=np.float32)
        else:
            diagnostic = np.asarray(observation_diagnostics, dtype=np.float32).reshape(
                -1
            )
            if diagnostic.size != 6 or not np.isfinite(diagnostic).all():
                raise ValueError(
                    "observation diagnostics must contain six finite values"
                )
        self.observation_diagnostics[frame_id] = diagnostic.copy()
        self.observation_diagnostic_valid[frame_id] = bool(diagnostic_valid)

    def reseed_history(
        self, frame_ids, boxes, timestamps=None, *, before_frame_id, rollout_horizon
    ):
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
        if any(frame_id < 0 or frame_id >= before_frame_id for frame_id in frame_ids):
            raise ValueError("reseed may only use frames before the query")
        if max(self.predictions) != before_frame_id - 1:
            raise ValueError("reseed requires a causally complete state up to t-1")
        for frame_id, box, timestamp in zip(frame_ids, boxes, timestamps):
            if frame_id not in self.predictions:
                raise KeyError(f"reseed cannot introduce unseen frame {frame_id}")
            self.predictions[frame_id] = copy.deepcopy(box)
            self.timestamps[frame_id] = timestamp
            # Re-anchored rows are interventions, not deployed B0 observations.
            self.observation_diagnostics[frame_id] = np.zeros(6, dtype=np.float32)
            self.observation_diagnostic_valid[frame_id] = False
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
                    f"{self.tracklet_key}"
                )
            boxes.append(copy.deepcopy(self.predictions.get(frame_id, self.first_box)))
        return boxes

    def history_contract(self, frame_ids, valid_mask):
        boxes = self.history_boxes(frame_ids, valid_mask)
        return {
            "history_boxes_world": np.stack(
                [box_world_row(box) for box in boxes], axis=0
            ),
            "history_valid_mask": np.asarray(valid_mask, dtype=np.int64),
            "history_timestamps": [
                self.timestamps.get(int(frame_id)) for frame_id in frame_ids
            ],
            "history_observation_diagnostics": np.stack(
                [
                    self.observation_diagnostics.get(
                        int(frame_id), np.zeros(6, dtype=np.float32)
                    )
                    for frame_id in frame_ids
                ],
                axis=0,
            ).astype(np.float32),
            "history_diagnostic_valid_mask": np.asarray(
                [
                    bool(valid)
                    and self.observation_diagnostic_valid.get(int(frame_id), False)
                    for frame_id, valid in zip(frame_ids, valid_mask)
                ],
                dtype=np.int64,
            ),
            "target_size": self.target_size,
            "tracklet_key": self.tracklet_key,
        }

    @property
    def results_bbs(self):
        return [self.predictions[index] for index in sorted(self.predictions)]


def _ordered_history_records(history_contract):
    frames = history_contract["prev_frames"]
    ordered_frames = [
        frames[key] for key in sorted(frames, key=lambda value: abs(int(value)))
    ]
    frame_ids = [int(value) for value in history_contract["prev_frame_ids"]]
    if len(ordered_frames) != len(frame_ids):
        raise ValueError("history frames and absolute frame ids must align")
    return list(zip(frame_ids, ordered_frames))


def candidate_history_union(raw):
    """Return every distinct physical past frame needed by c0/c1/c2."""
    query_frame = int(raw["this_frame_id"])
    contracts = [raw]
    temporal_pool = raw.get("temporal_candidate_pool")
    if temporal_pool is not None:
        contracts.extend(
            temporal_pool[key]
            for key in sorted(temporal_pool, key=lambda value: int(value))
        )
    records = {}
    for contract in contracts:
        for frame_id, frame in _ordered_history_records(contract):
            if frame_id < 0 or frame_id >= query_frame:
                raise ValueError("training reanchor may not read current or future GT")
            records.setdefault(frame_id, frame)
    return [(frame_id, records[frame_id]) for frame_id in sorted(records)]


def resolve_training_reanchor_policy(config):
    """Resolve the v25 enum while retaining the v24 boolean interface."""
    explicit = getattr(config, "ct_training_reanchor_policy", None)
    if explicit is None and isinstance(config, dict):
        explicit = config.get("ct_training_reanchor_policy")
    if explicit is None:
        legacy = getattr(config, "ct_recursive_reseed_enabled", False)
        if isinstance(config, dict):
            legacy = config.get("ct_recursive_reseed_enabled", False)
        return "periodic_past_gt" if bool(legacy) else "off"
    policy = str(explicit).strip().lower()
    if policy not in ("off", "periodic_past_gt"):
        raise ValueError("ct_training_reanchor_policy must be off or periodic_past_gt")
    return policy


def apply_training_reanchor(raw, state, horizon, config):
    """Apply the shared, past-only mixed-horizon training intervention.

    This is deliberately a data protocol function.  It is called by both the
    model training path and the checkpoint-free preflight exporter.  Inference
    never invokes it.
    """
    horizon = int(horizon)
    frame_id = int(raw["this_frame_id"])
    if horizon <= 0 or frame_id <= 0:
        raise ValueError("training reanchor requires positive horizon/frame")
    union = candidate_history_union(raw)
    canonical_id, canonical_frame = _ordered_history_records(raw)[0]
    pre_anchor = state.history_boxes([canonical_id], [1])[0]
    gt_anchor = canonical_frame["3d_bbox"]
    pre_error = float(
        np.linalg.norm(
            np.asarray(pre_anchor.center[:2], dtype=np.float64)
            - np.asarray(gt_anchor.center[:2], dtype=np.float64)
        )
    )
    enabled = resolve_training_reanchor_policy(config) == "periodic_past_gt"
    reset_boundary = bool(enabled and (frame_id - 1) % horizon == 0)
    reanchored_ids = []
    if reset_boundary and frame_id > 1:
        reanchored_ids = [frame for frame, _ in union]
        state.reseed_history(
            reanchored_ids,
            [record["3d_bbox"] for _, record in union],
            [record.get("timestamp") for _, record in union],
            before_frame_id=frame_id,
            rollout_horizon=horizon,
        )
    elif frame_id == 1:
        state.rollout_horizon = horizon
        state.last_reseed_before_frame = 1
    post_anchor = state.history_boxes([canonical_id], [1])[0]
    post_error = float(
        np.linalg.norm(
            np.asarray(post_anchor.center[:2], dtype=np.float64)
            - np.asarray(gt_anchor.center[:2], dtype=np.float64)
        )
    )
    return {
        "rollout_horizon": horizon,
        "rollout_age": state.rollout_age(frame_id),
        "reset_boundary": reset_boundary,
        "reanchored_frame_ids": reanchored_ids,
        "candidate_history_union_size": len(union),
        "pre_reset_anchor_error": pre_error,
        "post_reset_anchor_error": post_error,
    }


def commit_canonical_prediction(
    state,
    candidate_id,
    frame_id,
    box,
    timestamp=None,
    observation_diagnostics=None,
    diagnostic_valid=False,
):
    """Commit candidate 0 only; recovery views can never mutate state."""
    if int(candidate_id) != 0:
        return False
    state.append(
        frame_id,
        box,
        timestamp,
        observation_diagnostics=observation_diagnostics,
        diagnostic_valid=diagnostic_valid,
    )
    return True
