# Created by zenn at 2021/4/27
# Modified by Aron Lin at Jun 4 20:32:36 CST 2023

import numpy as np
import torch
from easydict import EasyDict
from nuscenes.utils import geometry_utils

import datasets.points_utils as points_utils
from utils.ct_history import (
    build_alternating_aux_history_offsets,
    build_causal_temporal_history_offsets,
    build_irregular_history_offsets,
    normalize_causal_temporal_gaps,
)
from utils.candidate_utils import normalize_candidate_trajectory_mode

from datasets.misc_utils import (
    get_history_frame_ids_and_masks,
    create_history_frame_dict,
)
from utils.sampling_utils import (
    candidate_offsets_for_frame_ids,
    point_sampling_seeds_for_frame_ids,
    deterministic_candidate_offset,
    deterministic_point_seed,
    physical_frame_point_seed,
)
from ctseqtrack.data.recursive import (
    OnlineRecursiveBatchSampler,
    build_scene_partition_manifest,
    scene_partition_tracklet_ids,
    stable_tracklet_partition,
)
from ctseqtrack.data.sample_builder import (
    motion_processing_mf as build_motion_sample_mf,
)


def no_processing(data, *args):
    return data


def siamese_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    """
    first_frame = data["first_frame"]
    template_frame = data["template_frame"]
    search_frame = data["search_frame"]
    candidate_id = data["candidate_id"]
    first_pc, first_box = first_frame["pc"], first_frame["3d_bbox"]
    template_pc, template_box = template_frame["pc"], template_frame["3d_bbox"]
    search_pc, search_box = search_frame["pc"], search_frame["3d_bbox"]
    if template_transform is not None:
        template_pc, template_box = template_transform(template_pc, template_box)
        first_pc, first_box = template_transform(first_pc, first_box)
    if search_transform is not None:
        search_pc, search_box = search_transform(search_pc, search_box)
    # generating template. Merging the object from previous and the first frames.
    if candidate_id == 0:
        samplegt_offsets = np.zeros(3)
    else:
        samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        samplegt_offsets[2] = samplegt_offsets[2] * (
            5 if config.degrees else np.deg2rad(5)
        )
    template_box = points_utils.getOffsetBB(
        template_box,
        samplegt_offsets,
        limit_box=config.data_limit_box,
        degrees=config.degrees,
    )
    model_pc, model_box = points_utils.getModel(
        [first_pc, template_pc],
        [first_box, template_box],
        scale=config.model_bb_scale,
        offset=config.model_bb_offset,
    )

    assert model_pc.nbr_points() > 20, "not enough template points"

    # generating search area. Use the current gt box to select the nearby region as the search area.

    if candidate_id == 0 and config.num_candidates > 1:
        sample_offset = np.zeros(3)
    else:
        # gaussian = KalmanFiltering(bnd=[1, 1, (5 if config.degrees else np.deg2rad(5))])
        # sample_offset = gaussian.sample(1)[0]
        raise NotImplementedError(
            "Previously used pomegranate's KalmanFiltering here, now disabled. Update required."
        )
    sample_bb = points_utils.getOffsetBB(
        search_box,
        sample_offset,
        limit_box=config.data_limit_box,
        degrees=config.degrees,
    )
    search_pc_crop = points_utils.generate_subwindow(
        search_pc,
        sample_bb,
        scale=config.search_bb_scale,
        offset=config.search_bb_offset,
    )
    assert search_pc_crop.nbr_points() > 20, "not enough search points"
    search_box = points_utils.transform_box(search_box, sample_bb)
    seg_label = points_utils.get_in_box_mask(search_pc_crop, search_box).astype(int)
    search_bbox_reg = [
        search_box.center[0],
        search_box.center[1],
        search_box.center[2],
        -sample_offset[2],
    ]

    template_points, idx_t = points_utils.regularize_pc(
        model_pc.points.T, config.template_size
    )
    search_points, idx_s = points_utils.regularize_pc(
        search_pc_crop.points.T, config.search_size
    )
    seg_label = seg_label[idx_s]
    data_dict = {
        "template_points": template_points.astype("float32"),
        "search_points": search_points.astype("float32"),
        "box_label": np.array(search_bbox_reg).astype("float32"),
        "bbox_size": search_box.wlh,
        "seg_label": seg_label.astype("float32"),
    }
    if getattr(config, "box_aware", False):
        template_bc = points_utils.get_point_to_box_distance(template_points, model_box)
        search_bc = points_utils.get_point_to_box_distance(search_points, search_box)
        data_dict.update(
            {
                "points2cc_dist_t": template_bc.astype("float32"),
                "points2cc_dist_s": search_bc.astype("float32"),
            }
        )
    return data_dict


def motion_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    point_sample_size
    bb_scale
    bb_offset
    """
    prev_frame = data["prev_frame"]
    this_frame = data["this_frame"]
    candidate_id = data["candidate_id"]
    prev_pc, prev_box = prev_frame["pc"], prev_frame["3d_bbox"]
    this_pc, this_box = this_frame["pc"], this_frame["3d_bbox"]

    num_points_in_prev_box = geometry_utils.points_in_box(
        prev_box, prev_pc.points[0:3, :]
    ).sum()
    assert (
        num_points_in_prev_box > config.limit_num_points_in_prev_box
    ), "not enough target points"

    if template_transform is not None:
        prev_pc, prev_box = template_transform(prev_pc, prev_box)
    if search_transform is not None:
        this_pc, this_box = search_transform(this_pc, this_box)

    if candidate_id == 0:
        sample_offsets = np.zeros(3)
    else:
        sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        sample_offsets[2] = sample_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    ref_box = points_utils.getOffsetBB(
        prev_box,
        sample_offsets,
        limit_box=config.data_limit_box,
        degrees=config.degrees,
    )
    prev_frame_pc = points_utils.generate_subwindow(
        prev_pc, ref_box, scale=config.bb_scale, offset=config.bb_offset
    )

    this_frame_pc = points_utils.generate_subwindow(
        this_pc, ref_box, scale=config.bb_scale, offset=config.bb_offset
    )
    # assert this_frame_pc.nbr_points() > config.limit_num_this_frame_subwindow_pc, 'not enough search points'

    this_box = points_utils.transform_box(this_box, ref_box)
    prev_box = points_utils.transform_box(prev_box, ref_box)
    ref_box = points_utils.transform_box(ref_box, ref_box)
    motion_box = points_utils.transform_box(this_box, prev_box)

    prev_points, idx_prev = points_utils.regularize_pc(
        prev_frame_pc.points.T, config.point_sample_size
    )
    this_points, idx_this = points_utils.regularize_pc(
        this_frame_pc.points.T, config.point_sample_size
    )

    seg_label_this = geometry_utils.points_in_box(
        this_box, this_points.T[:3, :], 1.25
    ).astype(int)
    seg_label_prev = geometry_utils.points_in_box(
        prev_box, prev_points.T[:3, :], 1.25
    ).astype(int)
    seg_mask_prev = geometry_utils.points_in_box(
        ref_box, prev_points.T[:3, :], 1.25
    ).astype(float)
    if candidate_id != 0:
        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        seg_mask_prev[seg_mask_prev == 0] = 0.2
        seg_mask_prev[seg_mask_prev == 1] = 0.8
    seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

    timestamp_prev = np.full((config.point_sample_size, 1), fill_value=0)
    timestamp_this = np.full((config.point_sample_size, 1), fill_value=0.1)

    prev_points = np.concatenate(
        [prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1
    )
    this_points = np.concatenate(
        [this_points, timestamp_this, seg_mask_this[:, None]], axis=-1
    )

    stack_points = np.concatenate([prev_points, this_points], axis=0)
    stack_seg_label = np.hstack([seg_label_prev, seg_label_this])
    theta_this = (
        this_box.orientation.degrees * this_box.orientation.axis[-1]
        if config.degrees
        else this_box.orientation.radians * this_box.orientation.axis[-1]
    )
    box_label = np.append(this_box.center, theta_this).astype("float32")
    theta_prev = (
        prev_box.orientation.degrees * prev_box.orientation.axis[-1]
        if config.degrees
        else prev_box.orientation.radians * prev_box.orientation.axis[-1]
    )
    box_label_prev = np.append(prev_box.center, theta_prev).astype("float32")
    theta_motion = (
        motion_box.orientation.degrees * motion_box.orientation.axis[-1]
        if config.degrees
        else motion_box.orientation.radians * motion_box.orientation.axis[-1]
    )
    motion_label = np.append(motion_box.center, theta_motion).astype("float32")

    motion_state_label = (
        np.sqrt(np.sum((this_box.center - prev_box.center) ** 2))
        > config.motion_threshold
    )

    data_dict = {
        "points": stack_points.astype("float32"),
        "box_label": box_label,
        "box_label_prev": box_label_prev,
        "motion_label": motion_label,
        "motion_state_label": motion_state_label.astype("int"),
        "bbox_size": (
            data["first_frame"]["3d_bbox"].wlh
            if bool(getattr(config, "observation_safe_bbox_size", False))
            else this_box.wlh
        ),
        "seg_label": stack_seg_label.astype("int"),
    }

    if getattr(config, "box_aware", False):
        prev_bc = points_utils.get_point_to_box_distance(
            stack_points[: config.point_sample_size, :3], prev_box
        )
        this_bc = points_utils.get_point_to_box_distance(
            stack_points[config.point_sample_size :, :3], this_box
        )
        candidate_bc_prev = points_utils.get_point_to_box_distance(
            stack_points[: config.point_sample_size, :3], ref_box
        )
        candidate_bc_this = np.zeros_like(candidate_bc_prev)
        candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)

        data_dict.update(
            {
                "prev_bc": prev_bc.astype("float32"),
                "this_bc": this_bc.astype("float32"),
                "candidate_bc": candidate_bc.astype("float32"),
            }
        )
    return data_dict


def motion_processing_mf(data, config, template_transform=None, search_transform=None):
    """Compatibility entry for the canonical v25 sample builder."""
    return build_motion_sample_mf(
        data,
        config,
        template_transform=template_transform,
        search_transform=search_transform,
    )


class PointTrackingSampler(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        random_sample,
        sample_per_epoch=10000,
        processing=siamese_processing,
        config=None,
        **kwargs,
    ):
        if config is None:
            config = EasyDict(kwargs)
        self.sample_per_epoch = sample_per_epoch
        self.dataset = dataset
        self.processing = processing
        self.config = config
        self.random_sample = random_sample
        self.num_candidates = getattr(config, "num_candidates", 1)
        if getattr(self.config, "use_augmentation", False):
            print("using augmentation")
            self.transform = points_utils.apply_augmentation
        else:
            self.transform = None
        if not self.random_sample:
            num_frames_total = 0
            self.tracklet_start_ids = [num_frames_total]
            for i in range(dataset.get_num_tracklets()):
                num_frames_total += dataset.get_num_frames_tracklet(i)
                self.tracklet_start_ids.append(num_frames_total)

    def get_anno_index(self, index):
        return index // self.num_candidates

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        if self.random_sample:
            return self.sample_per_epoch * self.num_candidates
        else:
            return self.dataset.get_num_frames_total() * self.num_candidates

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            if self.random_sample:
                tracklet_id = torch.randint(
                    0, self.dataset.get_num_tracklets(), size=(1,)
                ).item()
                tracklet_annos = self.dataset.tracklet_anno_list[tracklet_id]
                frame_ids = [0] + points_utils.random_choice(
                    num_samples=2, size=len(tracklet_annos)
                ).tolist()
            else:
                for i in range(0, self.dataset.get_num_tracklets()):
                    if (
                        self.tracklet_start_ids[i]
                        <= anno_id
                        < self.tracklet_start_ids[i + 1]
                    ):
                        tracklet_id = i
                        this_frame_id = anno_id - self.tracklet_start_ids[i]
                        prev_frame_id = max(this_frame_id - 1, 0)
                        frame_ids = (0, prev_frame_id, this_frame_id)
            first_frame, template_frame, search_frame = self.dataset.get_frames(
                tracklet_id, frame_ids=frame_ids
            )
            data = {
                "first_frame": first_frame,
                "template_frame": template_frame,
                "search_frame": search_frame,
                "candidate_id": candidate_id,
            }

            return self.processing(
                data,
                self.config,
                template_transform=None,
                search_transform=self.transform,
            )
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


class TestTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, config=None, **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        return self.dataset.get_frames(index, frame_ids)


class PartitionedTestTrackingSampler(TestTrackingSampler):
    """Evaluation view over one atomic tracklet hash partition."""

    def __init__(self, dataset, config=None, partition="dev", **kwargs):
        super().__init__(dataset, config=config, **kwargs)
        self.partition = str(partition)
        seed = int(
            getattr(self.config, "ct_partition_seed", getattr(self.config, "seed", 42))
            or 42
        )
        self.partition_scheme = (
            str(getattr(self.config, "ct_partition_scheme", "tracklet_v1"))
            .strip()
            .lower()
        )
        self.partition_manifest = None
        if self.partition_scheme == "scene_v2":
            self.partition_manifest = build_scene_partition_manifest(dataset, seed)
            self.tracklet_indices = scene_partition_tracklet_ids(
                self.partition_manifest, self.partition
            )
        elif self.partition_scheme in ("tracklet_v1", "legacy"):
            self.tracklet_indices = []
            for tracklet_id in range(dataset.get_num_tracklets()):
                key = (
                    dataset.get_tracklet_key(tracklet_id)
                    if hasattr(dataset, "get_tracklet_key")
                    else str(tracklet_id)
                )
                if stable_tracklet_partition(key, seed) == self.partition:
                    self.tracklet_indices.append(tracklet_id)
        else:
            raise ValueError("ct_partition_scheme must be tracklet_v1 or scene_v2")
        if not self.tracklet_indices:
            raise ValueError(f"tracklet partition {self.partition!r} is empty")

    def __len__(self):
        return len(self.tracklet_indices)

    def __getitem__(self, index):
        tracklet_id = self.tracklet_indices[index]
        frame_count = self.dataset.get_num_frames_tracklet(tracklet_id)
        return self.dataset.get_frames(tracklet_id, list(range(frame_count)))

    def get_tracklet_key(self, index):
        tracklet_id = self.tracklet_indices[index]
        if hasattr(self.dataset, "get_tracklet_key"):
            return self.dataset.get_tracklet_key(tracklet_id)
        return str(tracklet_id)

    def get_partition_group_key(self, index):
        tracklet_id = self.tracklet_indices[index]
        if hasattr(self.dataset, "get_partition_group_key"):
            return self.dataset.get_partition_group_key(tracklet_id)
        return self.get_tracklet_key(index)


def online_recursive_collate(items):
    """Keep raw point-cloud objects out of PyTorch's tensor collator."""
    return items


class MotionTrackingSampler(PointTrackingSampler):
    def __init__(self, dataset, config=None, **kwargs):
        super().__init__(dataset, random_sample=False, config=config, **kwargs)
        self.processing = motion_processing

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:

            for i in range(0, self.dataset.get_num_tracklets()):
                if (
                    self.tracklet_start_ids[i]
                    <= anno_id
                    < self.tracklet_start_ids[i + 1]
                ):
                    tracklet_id = i
                    this_frame_id = anno_id - self.tracklet_start_ids[i]
                    prev_frame_id = max(this_frame_id - 1, 0)
                    frame_ids = (0, prev_frame_id, this_frame_id)
            first_frame, prev_frame, this_frame = self.dataset.get_frames(
                tracklet_id, frame_ids=frame_ids
            )
            data = {
                "first_frame": first_frame,
                "prev_frame": prev_frame,
                "this_frame": this_frame,
                "candidate_id": candidate_id,
            }

            return self.processing(
                data,
                self.config,
                template_transform=self.transform,
                search_transform=self.transform,
            )
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


class MotionTrackingSamplerMF(PointTrackingSampler):
    def __init__(self, dataset, config=None, **kwargs):
        super().__init__(dataset, random_sample=False, config=config, **kwargs)
        self.processing = motion_processing_mf
        self.use_b1motion_v3 = bool(getattr(self.config, "use_b1motion_v3", False))
        self.online_recursive_training = bool(
            getattr(self.config, "ct_online_recursive_training", False)
        )
        self.ct_candidate_policy = (
            str(getattr(self.config, "ct_candidate_policy", "legacy_spatial"))
            .strip()
            .lower()
        )
        self.causal_temporal_candidates = self.ct_candidate_policy in (
            "causal_b1_boundary",
            "causal_temporal_uniform",
        )
        recovery_policy = (
            str(getattr(self.config, "ct_recovery_candidate_policy", "off"))
            .strip()
            .lower()
        )
        if (
            recovery_policy != "off"
            and self.ct_candidate_policy != "legacy_spatial_gt_ablation"
        ):
            raise ValueError(
                "GT-spatial recovery requires the explicit "
                "legacy_spatial_gt_ablation policy"
            )
        self.ct_temporal_candidate_gaps = (
            normalize_causal_temporal_gaps(
                getattr(self.config, "ct_temporal_candidate_gaps", [2, 4, 8])
            )
            if self.causal_temporal_candidates
            else []
        )
        self.candidate_trajectory_mode = normalize_candidate_trajectory_mode(
            getattr(self.config, "candidate_trajectory_mode", "independent")
        )
        if self.online_recursive_training:
            if self.candidate_trajectory_mode != "shared_se2":
                raise ValueError(
                    "online recursive training requires shared_se2 candidates"
                )
            if int(self.num_candidates) != int(
                getattr(self.config, "ct_recursive_candidate_views", 1)
            ):
                raise ValueError(
                    "num_candidates must match ct_recursive_candidate_views"
                )
            if self.causal_temporal_candidates and int(self.num_candidates) != 3:
                raise ValueError(
                    "causal temporal policies require exactly three candidates"
                )
        self.trajectory_training_irregular_probability = float(
            getattr(self.config, "trajectory_training_irregular_probability", 0.0)
        )
        self.trajectory_training_query_gaps = [
            int(value)
            for value in getattr(self.config, "trajectory_training_query_gaps", [2, 4])
        ]
        self.trajectory_training_transition_gaps = [
            int(value)
            for value in getattr(
                self.config, "trajectory_training_transition_gaps", [1, 1, 2, 4]
            )
        ]
        if not 0.0 <= self.trajectory_training_irregular_probability <= 1.0:
            raise ValueError(
                "trajectory_training_irregular_probability must be in [0,1]"
            )
        if any(value <= 0 for value in self.trajectory_training_query_gaps) or any(
            value <= 0 for value in self.trajectory_training_transition_gaps
        ):
            raise ValueError("trajectory training gaps must be positive")
        if (
            self.trajectory_training_irregular_probability > 0
            and not self.trajectory_training_query_gaps
        ):
            raise ValueError("irregular trajectory training requires query gap choices")
        self.motion_v3_aux_query_gaps = [
            int(value)
            for value in getattr(self.config, "motion_v3_aux_query_gaps", [2, 4])
        ]
        self.motion_v3_aux_transition_gaps = [
            int(value)
            for value in getattr(self.config, "motion_v3_aux_transition_gaps", [1, 2])
        ]
        if self.use_b1motion_v3:
            if self.trajectory_training_irregular_probability != 0.0:
                raise ValueError(
                    "B1motion-v3 keeps the main view continuous; set legacy "
                    "trajectory_training_irregular_probability to zero"
                )
            if (
                not self.motion_v3_aux_query_gaps
                or any(value <= 0 for value in self.motion_v3_aux_query_gaps)
                or any(value <= 0 for value in self.motion_v3_aux_transition_gaps)
            ):
                raise ValueError("B1motion-v3 auxiliary gaps must be positive")

    def _locate_tracklet(self, anno_id):
        for i in range(0, self.dataset.get_num_tracklets()):
            if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                return i, anno_id - self.tracklet_start_ids[i]
        raise IndexError(f"anno_id {anno_id} is outside tracklet ranges.")

    def _sample_history_offsets(self):
        normal = list(range(1, self.dataset.hist_num + 1))
        if self.trajectory_training_irregular_probability <= 0:
            return normal
        if torch.rand(()) >= self.trajectory_training_irregular_probability:
            return normal
        query_index = torch.randint(
            0, len(self.trajectory_training_query_gaps), size=(1,)
        ).item()
        query_gap = self.trajectory_training_query_gaps[query_index]
        if len(self.trajectory_training_transition_gaps) > 1:
            start = torch.randint(
                0,
                len(self.trajectory_training_transition_gaps),
                size=(1,),
            ).item()
            transition_gaps = (
                self.trajectory_training_transition_gaps[start:]
                + self.trajectory_training_transition_gaps[:start]
            )
        else:
            transition_gaps = self.trajectory_training_transition_gaps
        return build_irregular_history_offsets(
            self.dataset.hist_num, query_gap, transition_gaps
        )

    def _motion_v3_aux_offsets(self, sample_index):
        return build_alternating_aux_history_offsets(
            self.dataset.hist_num,
            sample_index,
            query_gaps=self.motion_v3_aux_query_gaps,
            transition_gaps=self.motion_v3_aux_transition_gaps,
        )

    def _build_view(
        self,
        tracklet_id,
        this_frame_id,
        first_frame,
        this_frame,
        candidate_id,
        offsets,
        candidate_offset_map=None,
        point_sampling_seed_map=None,
        current_sampling_seed=None,
        candidate_shared_transform=None,
        motion_aux_offsets=None,
        sample_index=0,
    ):
        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
            this_frame_id, self.dataset.hist_num, offsets=offsets
        )
        prev_frames_tuple = self.dataset.get_frames(
            tracklet_id, frame_ids=prev_frame_ids
        )
        prev_frames_dict = create_history_frame_dict(prev_frames_tuple)
        tracklet_key = (
            self.dataset.get_tracklet_key(tracklet_id)
            if hasattr(self.dataset, "get_tracklet_key")
            else str(tracklet_id)
        )
        data = {
            "first_frame": first_frame,
            "prev_frames": prev_frames_dict,
            "this_frame": this_frame,
            "candidate_id": candidate_id,
            "valid_mask": valid_mask,
            "prev_frame_ids": prev_frame_ids,
            "this_frame_id": this_frame_id,
            "history_offsets": offsets,
            "sample_index": int(sample_index),
            "tracklet_id": int(tracklet_id),
            "tracklet_key": tracklet_key,
        }
        if motion_aux_offsets is not None:
            motion_aux_frame_ids, motion_aux_valid_mask = (
                get_history_frame_ids_and_masks(
                    this_frame_id,
                    self.dataset.hist_num,
                    offsets=motion_aux_offsets,
                )
            )
            motion_aux_frames = self.dataset.get_frames(
                tracklet_id, frame_ids=motion_aux_frame_ids
            )
            data.update(
                {
                    "motion_aux_prev_frames": create_history_frame_dict(
                        motion_aux_frames
                    ),
                    "motion_aux_valid_mask": motion_aux_valid_mask,
                    "motion_aux_frame_ids": motion_aux_frame_ids,
                    "motion_aux_offsets": motion_aux_offsets,
                }
            )
        if candidate_offset_map is not None:
            data["candidate_offsets"] = candidate_offsets_for_frame_ids(
                prev_frame_ids, candidate_offset_map
            )
        if candidate_shared_transform is not None:
            data["candidate_shared_transform"] = np.asarray(
                candidate_shared_transform, dtype=np.float32
            )
        if point_sampling_seed_map is not None:
            data["point_sampling_seeds"] = point_sampling_seeds_for_frame_ids(
                prev_frame_ids, point_sampling_seed_map
            )
        if current_sampling_seed is not None:
            data["current_sampling_seed"] = current_sampling_seed
        return self.processing(
            data,
            self.config,
            template_transform=self.transform,
            search_transform=self.transform,
        )

    def _online_raw_view(
        self,
        epoch,
        batch_index,
        slot,
        tracklet_id,
        this_frame_id,
        candidate_id,
        build_shadow=False,
    ):
        offsets = list(range(1, self.dataset.hist_num + 1))
        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
            this_frame_id, self.dataset.hist_num, offsets=offsets
        )
        first_frame, this_frame = self.dataset.get_frames(
            tracklet_id, frame_ids=(0, this_frame_id)
        )
        tracklet_key = (
            self.dataset.get_tracklet_key(tracklet_id)
            if hasattr(self.dataset, "get_tracklet_key")
            else str(tracklet_id)
        )
        physical_seed_scope = (
            str(
                getattr(
                    self.config, "ct_candidate_sampling_seed_scope", "candidate_role"
                )
            )
            .strip()
            .lower()
            == "physical_frame"
        )

        def history_point_seed(frame_id, *legacy_parts):
            if physical_seed_scope:
                return physical_frame_point_seed(
                    self.config, tracklet_key, this_frame_id, frame_id
                )
            return deterministic_point_seed(
                self.config, *legacy_parts, "history", int(frame_id)
            )

        def current_point_seed(*legacy_parts):
            if physical_seed_scope:
                return physical_frame_point_seed(
                    self.config, tracklet_key, this_frame_id
                )
            return deterministic_point_seed(self.config, *legacy_parts, "current")

        temporal_pool = None
        if self.causal_temporal_candidates:
            if int(candidate_id) != 0:
                raise ValueError(
                    "causal temporal online sampling uses one candidate0 carrier"
                )
            histories = {1: offsets}
            histories.update(
                {
                    gap: build_causal_temporal_history_offsets(
                        self.dataset.hist_num, gap
                    )
                    for gap in self.ct_temporal_candidate_gaps
                }
            )
            history_contracts = {}
            unique_frame_ids = []
            for gap, history_offsets in histories.items():
                frame_ids, history_valid = get_history_frame_ids_and_masks(
                    this_frame_id, self.dataset.hist_num, offsets=history_offsets
                )
                history_contracts[gap] = (
                    list(history_offsets),
                    list(frame_ids),
                    list(history_valid),
                )
                unique_frame_ids.extend(frame_ids)
            unique_frame_ids = sorted(set(int(value) for value in unique_frame_ids))
            unique_frames = self.dataset.get_frames(
                tracklet_id, frame_ids=unique_frame_ids
            )
            frame_map = dict(zip(unique_frame_ids, unique_frames))
            temporal_pool = {}
            for gap, (
                history_offsets,
                frame_ids,
                history_valid,
            ) in history_contracts.items():
                gap_seed_parts = (
                    tracklet_key,
                    int(this_frame_id),
                    "temporal_gap",
                    int(gap),
                )
                temporal_pool[int(gap)] = {
                    "prev_frames": create_history_frame_dict(
                        [frame_map[int(frame_id)] for frame_id in frame_ids]
                    ),
                    "prev_frame_ids": frame_ids,
                    "valid_mask": history_valid,
                    "history_offsets": history_offsets,
                    "point_sampling_seeds": np.asarray(
                        [
                            history_point_seed(int(frame_id), *gap_seed_parts)
                            for frame_id in frame_ids
                        ],
                        dtype=np.int64,
                    ),
                    "current_sampling_seed": current_point_seed(*gap_seed_parts),
                }
            normal = temporal_pool[1]
            prev_frames = [
                normal["prev_frames"][str(-(index + 1))]
                for index in range(self.dataset.hist_num)
            ]
        else:
            prev_frames = self.dataset.get_frames(tracklet_id, frame_ids=prev_frame_ids)
        seed_parts = (tracklet_key, int(this_frame_id), int(candidate_id))
        raw = {
            "online_recursive_raw": True,
            "online_epoch": int(epoch),
            "online_batch_index": int(batch_index),
            "online_slot": int(slot),
            "first_frame": first_frame,
            "prev_frames": create_history_frame_dict(prev_frames),
            "this_frame": this_frame,
            "candidate_id": int(candidate_id),
            "valid_mask": list(valid_mask),
            "prev_frame_ids": list(prev_frame_ids),
            "this_frame_id": int(this_frame_id),
            "history_offsets": offsets,
            "sample_index": int(this_frame_id),
            "tracklet_id": int(tracklet_id),
            "tracklet_key": tracklet_key,
            "candidate_shared_transform": (
                np.zeros(3, dtype=np.float32)
                if self.causal_temporal_candidates
                else deterministic_candidate_offset(
                    candidate_id, self.config, *seed_parts, "coherent_recursive_view"
                )
            ),
            "point_sampling_seeds": np.asarray(
                [
                    history_point_seed(frame_id, *seed_parts)
                    for frame_id in prev_frame_ids
                ],
                dtype=np.int64,
            ),
            "current_sampling_seed": current_point_seed(*seed_parts),
            "shadow_future": [],
        }
        if temporal_pool is not None:
            raw["temporal_candidate_pool"] = temporal_pool
            raw["point_sampling_seeds"] = temporal_pool[1]["point_sampling_seeds"]
            raw["current_sampling_seed"] = temporal_pool[1]["current_sampling_seed"]
        if self.use_b1motion_v3 and not self.causal_temporal_candidates:
            motion_aux_offsets = self._motion_v3_aux_offsets(this_frame_id)
            motion_aux_frame_ids, motion_aux_valid_mask = (
                get_history_frame_ids_and_masks(
                    this_frame_id, self.dataset.hist_num, offsets=motion_aux_offsets
                )
            )
            motion_aux_frames = self.dataset.get_frames(
                tracklet_id, frame_ids=motion_aux_frame_ids
            )
            raw.update(
                {
                    "motion_aux_prev_frames": create_history_frame_dict(
                        motion_aux_frames
                    ),
                    "motion_aux_valid_mask": list(motion_aux_valid_mask),
                    "motion_aux_frame_ids": list(motion_aux_frame_ids),
                    "motion_aux_offsets": list(motion_aux_offsets),
                }
            )
        if build_shadow:
            for future_id in (this_frame_id + 1, this_frame_id + 2):
                future_raw = self._online_raw_view(
                    epoch,
                    batch_index,
                    slot,
                    tracklet_id,
                    future_id,
                    candidate_id=0,
                    build_shadow=False,
                )
                future_raw["ct_observation_only"] = True
                raw["shadow_future"].append(future_raw)
        return raw

    def __getitem__(self, index):
        if self.online_recursive_training:
            if not isinstance(index, tuple) or len(index) != 7:
                raise TypeError(
                    "online recursive sampler requires structured batch indices"
                )
            return self._online_raw_view(*index)
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            tracklet_id, this_frame_id = self._locate_tracklet(anno_id)
            frame_ids = (0, this_frame_id)
            first_frame, this_frame = self.dataset.get_frames(
                tracklet_id, frame_ids=frame_ids
            )
            if self.use_b1motion_v3:
                offsets = list(range(1, self.dataset.hist_num + 1))
                motion_aux_offsets = self._motion_v3_aux_offsets(index)
            else:
                offsets = self._sample_history_offsets()
                motion_aux_offsets = None
            return self._build_view(
                tracklet_id,
                this_frame_id,
                first_frame,
                this_frame,
                candidate_id,
                offsets,
                motion_aux_offsets=motion_aux_offsets,
                sample_index=anno_id,
            )
        except AssertionError:
            # return 1
            return self[torch.randint(0, len(self), size=(1,)).item()]
