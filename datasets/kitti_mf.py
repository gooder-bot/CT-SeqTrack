"""KITTI Tracking multi-frame dataset for CT-SeqTrack.

The loader follows the Open3DSOT split convention over KITTI Tracking's
labelled training sequences:

* train: scenes 0000-0016
* valid/val: scenes 0017-0018
* test: scenes 0019-0020

``path`` may point either to the KITTI root containing ``training/`` or
directly to the tracking ``training`` directory.
"""

import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from datasets import base_dataset, points_utils
from datasets.data_classes import Box, PointCloud
from datasets.temporal_protocol import TemporalProtocolMixin


_KITTI_LABEL_FIELDS = (
    "frame",
    "track_id",
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
    "height",
    "width",
    "length",
    "x",
    "y",
    "z",
    "rotation_y",
)

_INTEGER_FIELDS = {"frame", "track_id", "occluded"}


class KITTIMFDataset(TemporalProtocolMixin, base_dataset.BaseDataset):
    """KITTI Tracking adapter with CT-SeqTrack temporal contracts."""

    CACHE_SCHEMA_VERSION = 2

    def __init__(
            self,
            path,
            split,
            category_name="Car",
            version="kitti_tracking",
            **kwargs):
        super().__init__(path, split, category_name, **kwargs)
        self.version = str(version)
        self.hist_num = int(kwargs.get("hist_num", 1))
        if self.hist_num <= 0:
            raise ValueError("hist_num must be positive for KITTIMFDataset")
        supported_categories = {
            "car", "van", "pedestrian", "cyclist", "all"}
        if str(self.category_name).strip().lower() not in supported_categories:
            raise ValueError(
                "KITTI category_name must be one of Car, Van, Pedestrian, "
                "Cyclist, or All")
        self.preload_offset = float(kwargs.get("preload_offset", -1))
        self.frame_period = float(kwargs.get(
            "frame_period", kwargs.get("default_time_step", 0.1)))
        if not np.isfinite(self.frame_period) or self.frame_period <= 0:
            raise ValueError("KITTI frame_period must be finite and positive")
        self.kitti_hv_intervals = self._parse_kitti_hv_intervals(
            kwargs.get("kitti_hv_interval", 1))
        self.allow_missing_pointcloud = self._parse_bool(
            kwargs.get("allow_missing_pointcloud", False))

        self.data_root = self._resolve_data_root(path)
        self.KITTI_Folder = str(self.data_root)
        self.KITTI_velo = self.data_root / "velodyne"
        self.KITTI_label = self.data_root / "label_02"
        self.KITTI_calib = self.data_root / "calib"
        self._validate_layout()

        scene_ids = kwargs.get("scene_ids")
        self.scene_list = self._build_scene_list(split, scene_ids=scene_ids)
        self.velos = defaultdict(dict)
        self.calibs = {}

        self._configure_temporal_protocol(
            kwargs,
            dataset_name="kitti_mf",
            version=self.version,
            default_delta_t=self.frame_period,
        )
        self.tracklet_anno_list, self.tracklet_len_list = (
            self._build_tracklet_anno())
        self._initialize_temporal_protocol()
        if self.preloading:
            self.training_samples = self._load_data()

    @staticmethod
    def _resolve_data_root(path):
        root = Path(path).expanduser().resolve()
        if (root / "label_02").is_dir():
            return root
        if (root / "training" / "label_02").is_dir():
            return root / "training"
        return root

    def _validate_layout(self):
        missing = [
            str(path) for path in (
                self.KITTI_label, self.KITTI_velo, self.KITTI_calib)
            if not path.is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "KITTI Tracking layout is incomplete. Expected label_02, "
                "velodyne and calib under the configured path (or its "
                f"training/ child). Missing: {missing}")

    @classmethod
    def _build_scene_list(cls, split, scene_ids=None):
        if scene_ids is not None:
            if isinstance(scene_ids, str):
                values = scene_ids.replace(",", " ").split()
            else:
                values = list(scene_ids)
            parsed = sorted(set(int(value) for value in values))
            if not parsed:
                raise ValueError("scene_ids must not be empty")
            if any(scene_id < 0 or scene_id > 20 for scene_id in parsed):
                raise ValueError("KITTI Tracking scene_ids must be in [0, 20]")
            return [f"{scene_id:04d}" for scene_id in parsed]

        normalized = str(split).strip().lower().replace("-", "_")
        tiny = "tiny" in normalized
        if normalized.startswith("train"):
            scene_names = [0] if tiny else list(range(0, 17))
        elif normalized.startswith(("valid", "val")):
            scene_names = [18] if tiny else list(range(17, 19))
        elif normalized.startswith("test"):
            scene_names = [19] if tiny else list(range(19, 21))
        elif normalized in ("full", "all"):
            scene_names = list(range(21))
        else:
            raise ValueError(
                f"Unsupported KITTI split {split!r}; use train, valid/val, "
                "test, full, or provide scene_ids")
        return [f"{scene_name:04d}" for scene_name in scene_names]

    @staticmethod
    def _parse_label_value(field, value):
        if field in _INTEGER_FIELDS:
            return int(float(value))
        if field == "type":
            return str(value)
        return float(value)

    @classmethod
    def _read_label_file(cls, label_path, scene):
        annotations = []
        with label_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                values = raw_line.split()
                if not values:
                    continue
                if len(values) < len(_KITTI_LABEL_FIELDS):
                    raise ValueError(
                        f"Malformed KITTI label row {label_path}:"
                        f"{line_number}; expected at least "
                        f"{len(_KITTI_LABEL_FIELDS)} values, got {len(values)}")
                annotation = {
                    field: cls._parse_label_value(field, value)
                    for field, value in zip(_KITTI_LABEL_FIELDS, values)
                }
                annotation["scene"] = str(scene)
                annotations.append(annotation)
        return annotations

    def _category_matches(self, annotation_type):
        requested = str(self.category_name).strip().lower()
        annotation_type = str(annotation_type).strip().lower()
        if requested == "all":
            return annotation_type in {
                "car", "van", "pedestrian", "cyclist"}
        return annotation_type == requested

    @staticmethod
    def _parse_kitti_hv_intervals(value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "all":
                return (1, 2, 3, 5, 10)
            values = normalized.replace(",", " ").split()
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = [value]
        intervals = tuple(sorted(set(int(item) for item in values)))
        if not intervals or any(interval <= 0 for interval in intervals):
            raise ValueError(
                "kitti_hv_interval must be a positive integer, a list of "
                "positive integers, or 'all'")
        return intervals

    def _build_tracklet_anno(self):
        tracklets = []
        lengths = []
        missing_labels = []
        for scene in self.scene_list:
            label_path = self.KITTI_label / f"{scene}.txt"
            if not label_path.is_file():
                missing_labels.append(str(label_path))
                continue
            annotations = [
                annotation
                for annotation in self._read_label_file(label_path, scene)
                if self._category_matches(annotation["type"])
                and int(annotation["track_id"]) >= 0
            ]
            by_track = defaultdict(list)
            for annotation in annotations:
                by_track[int(annotation["track_id"])].append(annotation)
            for track_id in sorted(by_track):
                tracklet = sorted(
                    by_track[track_id], key=lambda item: int(item["frame"]))
                frames = [int(item["frame"]) for item in tracklet]
                if len(frames) != len(set(frames)):
                    raise ValueError(
                        f"Duplicate frame in KITTI scene={scene}, "
                        f"track_id={track_id}")
                # Match the official HVTrack preprocessing: interval N
                # produces N phase-aligned sub-tracklets tracklet[i::N].
                for interval in self.kitti_hv_intervals:
                    for phase in range(min(len(tracklet), interval)):
                        phase_tracklet = []
                        for annotation in tracklet[phase::interval]:
                            annotation = dict(annotation)
                            annotation["_ct_kitti_hv_interval"] = interval
                            annotation["_ct_kitti_hv_phase"] = phase
                            phase_tracklet.append(annotation)
                        if phase_tracklet:
                            tracklets.append(phase_tracklet)
                            lengths.append(len(phase_tracklet))

        if missing_labels:
            raise FileNotFoundError(
                "Configured KITTI split is missing label files. Point path at "
                "the labelled KITTI Tracking training directory or choose an "
                f"available scene_ids subset. First missing files: "
                f"{missing_labels[:3]}")
        if not tracklets:
            raise RuntimeError(
                f"No KITTI tracklets found for split={self.split!r}, "
                f"category={self.category_name!r}")
        return tracklets, lengths

    def _tracklet_identity(self, source_idx, tracklet=None):
        if not tracklet:
            tracklet = self.tracklet_anno_list[int(source_idx)]
        first = tracklet[0]
        scene_id = str(first["scene"])
        track_id = int(first["track_id"])
        interval = int(first.get("_ct_kitti_hv_interval", 1))
        phase = int(first.get("_ct_kitti_hv_phase", 0))
        tracklet_key = (
            f"kitti_mf/{self.version}/{self.split}/"
            f"{scene_id}/{track_id}/{self.category_name}/"
            f"interval/{interval}/phase/{phase}")
        return {
            "tracklet_key": tracklet_key,
            "scene_id": scene_id,
            "track_id": track_id,
            "kitti_hv_interval": interval,
            "kitti_hv_phase": phase,
        }

    def _manifest_header(self):
        header = super()._manifest_header()
        header["kitti_hv_intervals"] = list(self.kitti_hv_intervals)
        return header

    @staticmethod
    def _anno_frame_token(anno):
        return f"{anno['scene']}/{int(anno['frame']):06d}"

    def _anno_timestamp(self, anno):
        return float(int(anno["frame"])) * self.frame_period

    def _preload_cache_path(self):
        category = self._safe_tag(self.category_name)
        split = self._safe_tag(self.split)
        version = self._safe_tag(self.version)
        protocol = self._virtual_rate_cache_tag()
        if protocol:
            protocol = f"_{protocol}"
        htv = "-".join(str(value) for value in self.kitti_hv_intervals)
        selection = self.virtual_rate_selection_sha256[:8]
        return self.data_root / (
            f"preload_kitti_mf_v{self.CACHE_SCHEMA_VERSION}_{category}_"
            f"{split}_{version}_htv{htv}_{self.preload_offset:g}{protocol}_"
            f"{selection}.dat")

    def _load_data(self):
        print("preloading KITTI data into memory")
        preload_data_path = self._preload_cache_path()
        if preload_data_path.is_file():
            print(f"loading from saved file {preload_data_path}")
            with preload_data_path.open("rb") as handle:
                training_samples = pickle.load(handle)
            if len(training_samples) != len(self.tracklet_anno_list):
                raise ValueError(
                    f"Stale KITTI preload cache: {preload_data_path}")
            return training_samples

        print("reading KITTI frames from annotations")
        training_samples = [
            [self._get_frame_from_anno(anno) for anno in tracklet]
            for tracklet in self.tracklet_anno_list
        ]
        temporary_path = preload_data_path.with_suffix(
            preload_data_path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(
                training_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, preload_data_path)
        print(f"saved loaded KITTI data to {preload_data_path}")
        return training_samples

    def get_num_scenes(self):
        return len(self.scene_list)

    def get_num_tracklets(self):
        return len(self.tracklet_anno_list)

    def get_num_frames_total(self):
        return sum(self.tracklet_len_list)

    def get_num_frames_tracklet(self, tracklet_id):
        return self.tracklet_len_list[int(tracklet_id)]

    def get_frames(self, seq_id, frame_ids):
        seq_id = int(seq_id)
        frame_ids = [int(frame_id) for frame_id in frame_ids]
        if self.preloading:
            frames = [
                self.training_samples[seq_id][frame_id]
                for frame_id in frame_ids
            ]
        else:
            tracklet = self.tracklet_anno_list[seq_id]
            frames = [
                self._get_frame_from_anno(tracklet[frame_id])
                for frame_id in frame_ids
            ]
        return self._enrich_frames_with_effective_time(
            seq_id, frame_ids, frames)

    def _calibration_for_scene(self, scene_id):
        scene_id = str(scene_id)
        if scene_id in self.calibs:
            return self.calibs[scene_id]
        calib_path = self.KITTI_calib / f"{scene_id}.txt"
        if not calib_path.is_file():
            raise FileNotFoundError(
                f"Missing KITTI calibration file: {calib_path}")
        calibration = self._read_calib_file(calib_path)
        transform = None
        for key in ("Tr_velo_cam", "Tr_velo_to_cam"):
            if key in calibration:
                transform = calibration[key]
                break
        if transform is None:
            raise KeyError(
                f"{calib_path} does not define Tr_velo_cam or "
                "Tr_velo_to_cam")
        self.calibs[scene_id] = transform
        return transform

    def _pointcloud_for_frame(self, scene_id, frame_id):
        scene_id = str(scene_id)
        frame_id = int(frame_id)
        if frame_id in self.velos[scene_id]:
            return self.velos[scene_id][frame_id]
        velodyne_path = (
            self.KITTI_velo / scene_id / f"{frame_id:06d}.bin")
        if not velodyne_path.is_file():
            if self.allow_missing_pointcloud:
                pointcloud = PointCloud(
                    np.zeros((3, 1), dtype=np.float32))
                self.velos[scene_id][frame_id] = pointcloud
                return pointcloud
            raise FileNotFoundError(
                f"Missing KITTI point cloud: {velodyne_path}")
        raw = np.fromfile(velodyne_path, dtype=np.float32)
        if raw.size == 0 or raw.size % 4 != 0:
            raise ValueError(
                f"Malformed KITTI point cloud {velodyne_path}: "
                f"{raw.size} float32 values")
        pointcloud = PointCloud(raw.reshape(-1, 4).T)
        self.velos[scene_id][frame_id] = pointcloud
        return pointcloud

    def _get_frame_from_anno(self, anno):
        scene_id = str(anno["scene"])
        frame_id = int(anno["frame"])
        velo_to_cam = self._calibration_for_scene(scene_id)
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3, :4] = velo_to_cam

        box_center_cam = np.array(
            [
                float(anno["x"]),
                float(anno["y"]) - float(anno["height"]) / 2.0,
                float(anno["z"]),
                1.0,
            ],
            dtype=np.float64,
        )
        box_center_velo = np.linalg.solve(
            homogeneous, box_center_cam)[:3]
        size = [
            float(anno["width"]),
            float(anno["length"]),
            float(anno["height"]),
        ]
        orientation = (
            Quaternion(axis=[0, 0, -1], radians=float(anno["rotation_y"]))
            * Quaternion(axis=[0, 0, -1], degrees=90)
        )
        box = Box(
            box_center_velo,
            size,
            orientation,
            name=str(anno["type"]),
        )
        pointcloud = self._pointcloud_for_frame(scene_id, frame_id)
        if self.preload_offset > 0:
            pointcloud = points_utils.crop_pc_axis_aligned(
                pointcloud, box, offset=self.preload_offset)

        return {
            "pc": pointcloud,
            "3d_bbox": box,
            "meta": dict(anno),
            # Preserve the original KITTI frame number. After HTV/stride
            # filtering this is deliberately not the filtered list index.
            "timestamp": self._anno_timestamp(anno),
            "frame_id": frame_id,
        }

    @staticmethod
    def _read_calib_file(filepath):
        data = {}
        with Path(filepath).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                values = raw_line.split()
                if not values:
                    continue
                key = values[0].rstrip(":")
                try:
                    numbers = np.asarray(
                        [float(value) for value in values[1:]],
                        dtype=np.float64,
                    )
                except ValueError:
                    continue
                if numbers.size == 12:
                    data[key] = numbers.reshape(3, 4)
                else:
                    data[key] = numbers
        return data
