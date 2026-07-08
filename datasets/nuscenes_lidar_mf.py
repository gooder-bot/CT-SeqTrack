import os
import json

import numpy as np
import pickle
import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.splits import create_splits_scenes

from pyquaternion import Quaternion

from datasets import points_utils, base_dataset
from datasets.data_classes import PointCloud

from datasets.misc_utils import get_history_frame_ids_and_masks

# import vis_tool as vt

general_to_tracking_class = {"animal": "void / ignore",
                             "human.pedestrian.personal_mobility": "void / ignore",
                             "human.pedestrian.stroller": "void / ignore",
                             "human.pedestrian.wheelchair": "void / ignore",
                             "movable_object.barrier": "void / ignore",
                             "movable_object.debris": "void / ignore",
                             "movable_object.pushable_pullable": "void / ignore",
                             "movable_object.trafficcone": "void / ignore",
                             "static_object.bicycle_rack": "void / ignore",
                             "vehicle.emergency.ambulance": "void / ignore",
                             "vehicle.emergency.police": "void / ignore",
                             "vehicle.construction": "void / ignore",
                             "vehicle.bicycle": "bicycle",
                             "vehicle.bus.bendy": "bus",
                             "vehicle.bus.rigid": "bus",
                             "vehicle.car": "car",
                             "vehicle.motorcycle": "motorcycle",
                             "human.pedestrian.adult": "pedestrian",
                             "human.pedestrian.child": "pedestrian",
                             "human.pedestrian.construction_worker": "pedestrian",
                             "human.pedestrian.police_officer": "pedestrian",
                             "vehicle.trailer": "trailer",
                             "vehicle.truck": "truck", }

tracking_to_general_class = {
    'void / ignore': ['animal', 'human.pedestrian.personal_mobility', 'human.pedestrian.stroller',
                      'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
                      'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack',
                      'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.construction'],
    'bicycle': ['vehicle.bicycle'],
    'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
    'car': ['vehicle.car'],
    'motorcycle': ['vehicle.motorcycle'],
    'pedestrian': ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                   'human.pedestrian.police_officer'],
    'trailer': ['vehicle.trailer'],
    'truck': ['vehicle.truck']}


class NuScenesMFDataset(base_dataset.BaseDataset):
    def __init__(self, path, split, category_name="Car", version='v1.0-trainval', **kwargs):
        super().__init__(path, split, category_name, **kwargs)
        self.nusc = NuScenes(version=version, dataroot=path, verbose=False)
        self.version = version
        self.key_frame_only = kwargs.get('key_frame_only', False)
        self.min_points = kwargs.get('min_points', False)
        self.preload_offset = kwargs.get('preload_offset', -1)
        self.hist_num = kwargs.get('hist_num', 1) # Supports numbers between 0-N
        self.virtual_rate_mode = self._normalize_virtual_rate_mode(
            kwargs.get('virtual_rate_mode', 'none'))
        self.virtual_rate_gap_pattern = self._parse_int_list(
            kwargs.get('virtual_rate_gap_pattern', [1, 1, 2, 4]),
            default=[1, 1, 2, 4])
        self.virtual_rate_stride = int(kwargs.get('virtual_rate_stride', 2))
        self.virtual_rate_drop_every = int(kwargs.get('virtual_rate_drop_every', 5))
        self.virtual_rate_drop_prob = float(kwargs.get('virtual_rate_drop_prob', 0.0))
        self.virtual_rate_seed = int(kwargs.get('virtual_rate_seed', 42))
        self.virtual_rate_max_gap = int(kwargs.get('virtual_rate_max_gap', 5))
        self.virtual_rate_keep_first = self._parse_bool(
            kwargs.get('virtual_rate_keep_first', True))
        self.virtual_rate_keep_last = self._parse_bool(
            kwargs.get('virtual_rate_keep_last', True))
        self.virtual_rate_min_tracklet_len = int(
            kwargs.get('virtual_rate_min_tracklet_len', 0))
        self.virtual_rate_manifest = str(kwargs.get('virtual_rate_manifest', '') or '')
        if self.virtual_rate_mode == 'none' and self.virtual_rate_manifest:
            self.virtual_rate_mode = 'manifest'
        self.virtual_rate_burst_keep_lengths = self._parse_int_list(
            kwargs.get('virtual_rate_burst_keep_lengths', [3, 2, 3]),
            default=[3, 2, 3])
        self.virtual_rate_burst_skip_lengths = self._parse_int_list(
            kwargs.get('virtual_rate_burst_skip_lengths', [2, 3, 3]),
            default=[2, 3, 3])

        self.track_instances = self.filter_instance(split, category_name.lower(), self.min_points)
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()
        self.virtual_rate_meta = []
        self.virtual_rate_summary = self._build_virtual_rate_summary(
            original_lengths=self.tracklet_len_list,
            filtered_lengths=self.tracklet_len_list)
        self._apply_virtual_rate()
        if self.preloading:
            self.training_samples = self._load_data()

    def filter_instance(self, split, category_name=None, min_points=-1):
        """
        This function is used to filter the tracklets.

        split: the dataset split
        category_name:
        min_points: the minimum number of points in the first bbox
        """
        if category_name is not None:
            general_classes = tracking_to_general_class[category_name]
        instances = []
        scene_splits = nuscenes.utils.splits.create_splits_scenes()
        for instance in self.nusc.instance:
            anno = self.nusc.get('sample_annotation', instance['first_annotation_token'])
            sample = self.nusc.get('sample', anno['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            instance_category = self.nusc.get('category', instance['category_token'])['name']
            if scene['name'] in scene_splits[split] and anno['num_lidar_pts'] >= min_points and \
                    (category_name is None or category_name is not None and instance_category in general_classes):
                instances.append(instance)
        return instances

    def _build_tracklet_anno(self):
        list_of_tracklet_anno = []
        list_of_tracklet_len = []
        for instance in self.track_instances:
            track_anno = []
            curr_anno_token = instance['first_annotation_token']

            while curr_anno_token != '':

                ann_record = self.nusc.get('sample_annotation', curr_anno_token)
                sample = self.nusc.get('sample', ann_record['sample_token'])
                sample_data_lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

                curr_anno_token = ann_record['next']
                if self.key_frame_only and not sample_data_lidar['is_key_frame']:
                    continue
                track_anno.append({"sample_data_lidar": sample_data_lidar, "box_anno": ann_record})

            list_of_tracklet_anno.append(track_anno) 
            list_of_tracklet_len.append(len(track_anno))
        return list_of_tracklet_anno, list_of_tracklet_len

    @staticmethod
    def _normalize_virtual_rate_mode(mode):
        mode = str(mode or 'none').strip().lower().replace('-', '_')
        aliases = {
            '': 'none',
            'off': 'none',
            'false': 'none',
            'no': 'none',
            'gap': 'gap_pattern',
            'gap_pattern_manifest': 'gap_pattern',
            'periodic': 'periodic_drop',
            'periodicdrop': 'periodic_drop',
            'random': 'random_drop',
            'random_drop_manifest': 'random_drop',
            'randomdrop': 'random_drop',
            'burst': 'burst_drop',
            'burstdrop': 'burst_drop',
        }
        return aliases.get(mode, mode)

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, str):
            return value.strip().lower() not in ('0', 'false', 'no', 'off', '')
        return bool(value)

    @staticmethod
    def _parse_int_list(value, default=None):
        if value is None:
            return list(default or [])
        if isinstance(value, str):
            cleaned = value.replace('[', '').replace(']', '').replace(',', ' ')
            values = [item for item in cleaned.split() if item]
        else:
            values = list(value)
        parsed = [int(item) for item in values]
        return parsed if parsed else list(default or [])

    @staticmethod
    def _safe_tag(value):
        allowed = []
        for char in str(value):
            if char.isalnum() or char in ('_', '-'):
                allowed.append(char)
            else:
                allowed.append('_')
        return ''.join(allowed).strip('_') or 'none'

    def _pattern_tag(self):
        return ''.join(str(gap) for gap in self.virtual_rate_gap_pattern)

    def _virtual_rate_cache_tag(self):
        mode = self.virtual_rate_mode
        if mode == 'none':
            return ''
        if mode == 'gap_pattern':
            return f"vr_gap{self._pattern_tag()}"
        if mode == 'periodic_drop':
            return f"vr_drop{self.virtual_rate_drop_every}"
        if mode == 'burst_drop':
            keep = ''.join(str(x) for x in self.virtual_rate_burst_keep_lengths)
            skip = ''.join(str(x) for x in self.virtual_rate_burst_skip_lengths)
            return f"vr_burst_k{keep}_s{skip}"
        if mode == 'random_drop':
            prob = int(round(self.virtual_rate_drop_prob * 100))
            return f"vr_rand{prob}_seed{self.virtual_rate_seed}_max{self.virtual_rate_max_gap}"
        if mode == 'stride':
            return f"vr_stride{self.virtual_rate_stride}"
        if self.virtual_rate_manifest:
            manifest_name = os.path.splitext(os.path.basename(self.virtual_rate_manifest))[0]
            return f"vr_manifest_{self._safe_tag(manifest_name)}"
        return f"vr_{self._safe_tag(mode)}"

    def _build_virtual_rate_summary(self, original_lengths, filtered_lengths):
        original_lengths = list(original_lengths)
        filtered_lengths = list(filtered_lengths)
        original_frames = int(sum(original_lengths))
        filtered_frames = int(sum(filtered_lengths))
        dropped = original_frames - filtered_frames
        ratio = float(dropped / original_frames) if original_frames > 0 else 0.0
        return {
            "mode": self.virtual_rate_mode,
            "tracklets_before": len(original_lengths),
            "tracklets_after": len(filtered_lengths),
            "frames_before": original_frames,
            "frames_after": filtered_frames,
            "dropped_frame_ratio": ratio,
            "min_tracklet_len": int(min(filtered_lengths)) if filtered_lengths else 0,
            "mean_tracklet_len": float(np.mean(filtered_lengths)) if filtered_lengths else 0.0,
            "cache_tag": self._virtual_rate_cache_tag() or "none",
        }

    def _load_manifest_keep_indices(self):
        if not self.virtual_rate_manifest or not os.path.isfile(self.virtual_rate_manifest):
            return None
        with open(self.virtual_rate_manifest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        entries = manifest.get('tracklets', manifest)
        by_source = {}
        for list_idx, entry in enumerate(entries):
            if isinstance(entry, dict):
                source_idx = int(entry.get('source_tracklet', list_idx))
                keep_indices = entry.get('keep_indices', entry.get('keep', []))
            else:
                source_idx = list_idx
                keep_indices = entry
            by_source[source_idx] = [int(idx) for idx in keep_indices]
        print(f'loaded virtual-rate manifest {self.virtual_rate_manifest}')
        return by_source

    def _save_virtual_rate_manifest(self, meta):
        if not self.virtual_rate_manifest:
            return
        if os.path.isfile(self.virtual_rate_manifest):
            return
        parent = os.path.dirname(self.virtual_rate_manifest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        manifest = {
            "dataset": "nuscenes_mf",
            "version": self.version,
            "split": self.split,
            "category_name": self.category_name,
            "mode": self.virtual_rate_mode,
            "gap_pattern": self.virtual_rate_gap_pattern,
            "drop_prob": self.virtual_rate_drop_prob,
            "drop_every": self.virtual_rate_drop_every,
            "seed": self.virtual_rate_seed,
            "max_gap": self.virtual_rate_max_gap,
            "tracklets": meta,
        }
        with open(self.virtual_rate_manifest, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        print(f'saved virtual-rate manifest {self.virtual_rate_manifest}')

    def _validate_keep_indices(self, keep_indices, original_len):
        keep = sorted(set(int(idx) for idx in keep_indices
                          if 0 <= int(idx) < int(original_len)))
        if original_len <= 0:
            return []
        if self.virtual_rate_keep_first and 0 not in keep:
            keep.insert(0, 0)
        if self.virtual_rate_keep_last and (original_len - 1) not in keep:
            keep.append(original_len - 1)
        keep = sorted(set(keep))
        min_len = max(0, int(self.virtual_rate_min_tracklet_len))
        if min_len > 0 and len(keep) < min_len:
            filler = np.linspace(0, original_len - 1, min(original_len, min_len))
            keep = sorted(set(keep + [int(round(idx)) for idx in filler]))
        return keep

    def _gap_pattern_keep_indices(self, original_len):
        keep = [0]
        current = 0
        pattern = [max(1, int(gap)) for gap in self.virtual_rate_gap_pattern]
        pattern_idx = 0
        while pattern and current + pattern[pattern_idx % len(pattern)] < original_len:
            current += pattern[pattern_idx % len(pattern)]
            keep.append(current)
            pattern_idx += 1
        return keep

    def _periodic_drop_keep_indices(self, original_len):
        drop_every = max(2, int(self.virtual_rate_drop_every))
        return [idx for idx in range(original_len) if (idx + 1) % drop_every != 0]

    def _burst_drop_keep_indices(self, original_len):
        keep = []
        idx = 0
        stage = 0
        keep_lengths = [max(1, int(x)) for x in self.virtual_rate_burst_keep_lengths]
        skip_lengths = [max(1, int(x)) for x in self.virtual_rate_burst_skip_lengths]
        while idx < original_len:
            keep_len = keep_lengths[stage % len(keep_lengths)]
            for offset in range(keep_len):
                if idx + offset < original_len:
                    keep.append(idx + offset)
            idx += keep_len
            skip_len = skip_lengths[stage % len(skip_lengths)]
            idx += skip_len
            stage += 1
        return keep

    def _random_drop_keep_indices(self, original_len, source_tracklet):
        rng = np.random.default_rng(self.virtual_rate_seed + int(source_tracklet) * 1009)
        drop_prob = min(max(float(self.virtual_rate_drop_prob), 0.0), 0.95)
        max_gap = max(1, int(self.virtual_rate_max_gap))
        keep = [0]
        last_kept = 0
        for idx in range(1, max(original_len - 1, 1)):
            must_keep = (idx - last_kept) >= max_gap
            if must_keep or rng.random() >= drop_prob:
                keep.append(idx)
                last_kept = idx
        if original_len > 1:
            keep.append(original_len - 1)
        return keep

    def _stride_keep_indices(self, original_len):
        stride = max(1, int(self.virtual_rate_stride))
        return list(range(0, original_len, stride))

    def _build_keep_indices(self, original_len, source_tracklet):
        mode = self.virtual_rate_mode
        if mode == 'gap_pattern':
            keep = self._gap_pattern_keep_indices(original_len)
        elif mode == 'periodic_drop':
            keep = self._periodic_drop_keep_indices(original_len)
        elif mode == 'burst_drop':
            keep = self._burst_drop_keep_indices(original_len)
        elif mode == 'random_drop':
            keep = self._random_drop_keep_indices(original_len, source_tracklet)
        elif mode == 'stride':
            keep = self._stride_keep_indices(original_len)
        else:
            keep = list(range(original_len))
        return self._validate_keep_indices(keep, original_len)

    def _apply_virtual_rate(self):
        manifest_keep = self._load_manifest_keep_indices()
        if self.virtual_rate_mode == 'none' and manifest_keep is None:
            return

        original_lengths = list(self.tracklet_len_list)
        new_tracklets = []
        new_lengths = []
        meta = []

        for source_idx, tracklet in enumerate(self.tracklet_anno_list):
            if manifest_keep is not None and source_idx in manifest_keep:
                keep_indices = self._validate_keep_indices(
                    manifest_keep[source_idx], len(tracklet))
            else:
                keep_indices = self._build_keep_indices(len(tracklet), source_idx)
            if len(keep_indices) < max(1, int(self.virtual_rate_min_tracklet_len)):
                continue
            new_tracklets.append([tracklet[idx] for idx in keep_indices])
            new_lengths.append(len(keep_indices))
            meta.append({
                "source_tracklet": source_idx,
                "original_len": len(tracklet),
                "kept_len": len(keep_indices),
                "keep_indices": keep_indices,
            })

        if not new_tracklets:
            raise RuntimeError(
                f"virtual_rate_mode={self.virtual_rate_mode} removed all tracklets. "
                "Lower virtual_rate_min_tracklet_len or use a milder protocol.")

        self.tracklet_anno_list = new_tracklets
        self.tracklet_len_list = new_lengths
        self.virtual_rate_meta = meta
        self.virtual_rate_summary = self._build_virtual_rate_summary(
            original_lengths=original_lengths,
            filtered_lengths=new_lengths)
        self._save_virtual_rate_manifest(meta)

        summary = self.virtual_rate_summary
        print(
            "virtual-rate "
            f"mode={summary['mode']} "
            f"tracklets={summary['tracklets_after']}/{summary['tracklets_before']} "
            f"frames={summary['frames_after']}/{summary['frames_before']} "
            f"drop={summary['dropped_frame_ratio']:.3f} "
            f"tag={summary['cache_tag']}"
        )

    def _load_data(self):
        print('preloading data into memory')
        cache_suffix = self._virtual_rate_cache_tag()
        if cache_suffix:
            cache_suffix = f"_{cache_suffix}"
        preload_data_path = os.path.join(
            self.path,
            f"preload_nuscenes_{self.category_name}_{self.split}_{self.version}_"
            f"{self.preload_offset}_{self.min_points}{cache_suffix}.dat")
        if os.path.isfile(preload_data_path):
            print(f'loading from saved file {preload_data_path}.')
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
        else:
            print('reading from annos')
            training_samples = []
            for i in range(len(self.tracklet_anno_list)):
                frames = []
                for anno in self.tracklet_anno_list[i]:
                    frames.append(self._get_frame_from_anno_data(anno))
                training_samples.append(frames)
            with open(preload_data_path, 'wb') as f:
                print(f'saving loaded data to {preload_data_path}')
                pickle.dump(training_samples, f)
        return training_samples

    def get_num_tracklets(self):
        return len(self.tracklet_anno_list)

    def get_num_frames_total(self):
        return sum(self.tracklet_len_list)

    def get_num_frames_tracklet(self, tracklet_id):
        return self.tracklet_len_list[tracklet_id]

    def get_frames(self, seq_id, frame_ids):
        if self.preloading:
            frames = [self.training_samples[seq_id][f_id] for f_id in frame_ids]
        else:
            seq_annos = self.tracklet_anno_list[seq_id]
            frames = [self._get_frame_from_anno_data(seq_annos[f_id]) for f_id in frame_ids]

        return frames

    def _get_frame_from_anno_data(self, anno):
        sample_data_lidar = anno['sample_data_lidar']
        box_anno = anno['box_anno']
        bb = Box(box_anno['translation'], box_anno['size'], Quaternion(box_anno['rotation']),
                 name=box_anno['category_name'], token=box_anno['token'])
        pcl_path = os.path.join(self.path, sample_data_lidar['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        cs_record = self.nusc.get('calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        poserecord = self.nusc.get('ego_pose', sample_data_lidar['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        pc = PointCloud(points=pc.points)
        return {
            "pc": pc,
            "3d_bbox": bb,
            "meta": anno,
            "timestamp": sample_data_lidar['timestamp'] * 1e-6,
            "frame_id": sample_data_lidar['token'],
        }
