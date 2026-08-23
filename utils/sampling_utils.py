"""Deterministic sampling helpers for candidate and paired views."""

import hashlib

import numpy as np


SEQTRACK_OBSERVATION_CORE_FIELDS = frozenset({
    "points", "box_label", "ref_boxs", "box_label_prev",
    "motion_label", "motion_state_label", "bbox_size", "seg_label",
    "valid_mask", "delta_T", "num_points_in_search", "candidate_id",
    "prev_bc", "this_bc", "candidate_bc",
})


def prune_seqtrack_observation_payload(data_dict):
    """Return the B0-only tensor contract without CT mechanism fields."""
    return {
        key: value for key, value in data_dict.items()
        if key in SEQTRACK_OBSERVATION_CORE_FIELDS}


def stable_uint32_seed(base_seed, *parts):
    payload = "::".join((str(int(base_seed)), *(str(part) for part in parts)))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


class StatelessCandidateBatchSampler:
    """Epoch-addressable four-view batches independent of loader workers.

    Every batch contains the same number of rows from each candidate branch,
    but the annotation permutation is independent per branch.  This keeps the
    ordinary four-candidate population while making the configured branch
    loss weights exact inside every optimizer transaction.
    """

    def __init__(self, dataset, batch_size, candidate_views, seed, drop_last=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.candidate_views = int(candidate_views)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.candidate_views <= 0:
            raise ValueError("candidate_views must be positive")
        if self.batch_size <= 0 or self.batch_size % self.candidate_views:
            raise ValueError(
                "batch_size must be a positive multiple of candidate_views")
        if len(dataset) % self.candidate_views:
            raise ValueError(
                "candidate dataset length must be divisible by candidate_views")
        self.annotation_count = len(dataset) // self.candidate_views
        self.rows_per_view = self.batch_size // self.candidate_views
        if self.annotation_count < self.rows_per_view:
            raise ValueError("candidate dataset is smaller than one batch")

    def __len__(self):
        if self.drop_last:
            return self.annotation_count // self.rows_per_view
        return ((self.annotation_count + self.rows_per_view - 1)
                // self.rows_per_view)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)

    def __iter__(self):
        permutations = []
        for view_id in range(self.candidate_views):
            rng = np.random.default_rng(stable_uint32_seed(
                self.seed, "observation-shuffle", self.epoch, view_id))
            permutations.append(rng.permutation(self.annotation_count))
        for batch_index in range(len(self)):
            start = batch_index * self.rows_per_view
            stop = min(start + self.rows_per_view, self.annotation_count)
            if stop - start < self.rows_per_view and self.drop_last:
                return
            indices = []
            for view_id, permutation in enumerate(permutations):
                indices.extend(
                    int(annotation_id) * self.candidate_views + view_id
                    for annotation_id in permutation[start:stop])
            mix_rng = np.random.default_rng(stable_uint32_seed(
                self.seed, "observation-batch-mix", self.epoch, batch_index))
            order = mix_rng.permutation(len(indices))
            yield [indices[int(position)] for position in order]


def deterministic_candidate_offset(candidate_id, config, *seed_parts):
    if int(candidate_id) == 0:
        return np.zeros(3, dtype=np.float32)
    base_seed = int(getattr(config, "seed", 42) or 42)
    rng = np.random.default_rng(stable_uint32_seed(
        base_seed, "candidate", int(candidate_id), *seed_parts))
    offset = rng.uniform(low=-0.3, high=0.3, size=3).astype(np.float32)
    offset[2] *= 5 if config.degrees else np.deg2rad(5)
    return offset


def deterministic_recovery_candidate_offset(
        candidate_id, config, anchor_box, target_box, *seed_parts):
    """Create train-only weak-boundary and strict-miss acquisition views."""
    candidate_id = int(candidate_id)
    if candidate_id == 0:
        return np.zeros(3, dtype=np.float32)
    if candidate_id not in (1, 2):
        return deterministic_candidate_offset(
            candidate_id, config, *seed_parts)
    base_seed = int(getattr(config, "seed", 42) or 42)
    named_seed = stable_uint32_seed(
        base_seed, "recovery-role", candidate_id, *seed_parts)
    rng = np.random.default_rng(named_seed)
    axis = int(named_seed % 2)
    sign = -1.0 if int((named_seed // 2) % 2) else 1.0
    anchor_center = np.asarray(anchor_box.center, dtype=np.float64)
    target_center = np.asarray(target_box.center, dtype=np.float64)
    target_local = np.asarray(anchor_box.rotation_matrix, dtype=np.float64).T @ (
        target_center - anchor_center)
    anchor_size = np.asarray(anchor_box.wlh, dtype=np.float64)
    target_size = np.asarray(target_box.wlh, dtype=np.float64)
    crop_half_extent = (
        0.5 * anchor_size[axis] * float(getattr(config, "bb_scale", 1.0))
        + float(getattr(config, "bb_offset", 0.0)))
    target_half_extent = 0.5 * target_size[axis]
    if candidate_id == 1:
        desired_target_offset = max(
            0.0, crop_half_extent - 0.35 * target_half_extent)
    else:
        desired_target_offset = (
            crop_half_extent + target_half_extent + 0.05)
    natural = deterministic_candidate_offset(
        candidate_id, config, *seed_parts)
    transform = natural.astype(np.float64)
    transform[:2] = target_local[:2]
    transform[axis] = target_local[axis] - sign * desired_target_offset
    transform[1 - axis] += float(rng.uniform(-0.05, 0.05))
    return transform.astype(np.float32)


def deterministic_point_seed(config, *seed_parts):
    base_seed = int(getattr(config, "seed", 42) or 42)
    return stable_uint32_seed(base_seed, "points", *seed_parts)


def deterministic_candidate_retry_index(
        index, dataset_length, candidate_views, seed, epoch, attempt):
    """Choose another annotation without moving the candidate branch."""
    index = int(index)
    dataset_length = int(dataset_length)
    candidate_views = int(candidate_views)
    attempt = int(attempt)
    if candidate_views <= 0 or dataset_length <= 0:
        raise ValueError("retry dataset and candidate counts must be positive")
    if dataset_length % candidate_views:
        raise ValueError(
            "retry dataset length must be divisible by candidate views")
    annotation_count = dataset_length // candidate_views
    candidate_id = index % candidate_views
    annotation_id = index // candidate_views
    rng = np.random.default_rng(stable_uint32_seed(
        int(seed), "observation-retry", int(epoch), index,
        candidate_id, attempt))
    retry_annotation_id = int(rng.integers(0, annotation_count))
    if annotation_count > 1 and retry_annotation_id == annotation_id:
        retry_annotation_id = (retry_annotation_id + 1) % annotation_count
    return retry_annotation_id * candidate_views + candidate_id


def sample_candidate_offset(candidate_id, config):
    if int(candidate_id) == 0:
        return np.zeros(3, dtype=np.float32)
    offset = np.random.uniform(low=-0.3, high=0.3, size=3).astype(np.float32)
    offset[2] *= 5 if config.degrees else np.deg2rad(5)
    return offset


def build_shared_candidate_offset_map(candidate_id, frame_ids, config):
    return {
        int(frame_id): sample_candidate_offset(candidate_id, config)
        for frame_id in sorted(set(int(frame_id) for frame_id in frame_ids))
    }


def candidate_offsets_for_frame_ids(frame_ids, offset_map):
    return np.stack([
        np.asarray(offset_map[int(frame_id)], dtype=np.float32)
        for frame_id in frame_ids], axis=0)


def sample_point_sampling_seed():
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def build_shared_point_sampling_seed_map(frame_ids):
    return {
        int(frame_id): sample_point_sampling_seed()
        for frame_id in sorted(set(int(frame_id) for frame_id in frame_ids))
    }


def point_sampling_seeds_for_frame_ids(frame_ids, seed_map):
    return np.asarray([
        int(seed_map[int(frame_id)]) for frame_id in frame_ids],
        dtype=np.int64)

