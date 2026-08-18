import os
import json
import subprocess
from pathlib import Path

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
from datasets.misc_utils import normalize_dynamics_time_mode
from datasets.protocol_utils import (
    canonical_sha256,
    file_sha256,
    payload_with_content_sha256,
    verify_content_sha256,
)

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
        self.virtual_rate_manifest_strict = self._parse_bool(
            kwargs.get('virtual_rate_manifest_strict', True))
        self.virtual_rate_manifest_allow_create = self._parse_bool(
            kwargs.get('virtual_rate_manifest_allow_create', False))
        self.virtual_rate_manifest_require_commit_match = self._parse_bool(
            kwargs.get('virtual_rate_manifest_require_commit_match', False))
        self.protocol_role = str(kwargs.get('protocol_role', 'eval'))
        self.virtual_rate_manifest_content_sha256 = ''
        self.virtual_rate_manifest_file_sha256 = ''
        if self.virtual_rate_mode == 'none' and self.virtual_rate_manifest:
            self.virtual_rate_mode = 'manifest'
        self.virtual_rate_burst_keep_lengths = self._parse_int_list(
            kwargs.get('virtual_rate_burst_keep_lengths', [3, 2, 3]),
            default=[3, 2, 3])
        self.virtual_rate_burst_skip_lengths = self._parse_int_list(
            kwargs.get('virtual_rate_burst_skip_lengths', [2, 3, 3]),
            default=[2, 3, 3])
        self.dynamics_time_mode = normalize_dynamics_time_mode(
            kwargs.get('dynamics_time_mode', 'true'))
        self.dynamics_fixed_delta_t = float(
            kwargs.get('dynamics_fixed_delta_t', 0.5))
        if self.dynamics_fixed_delta_t <= 0:
            raise ValueError('dynamics_fixed_delta_t must be positive.')
        self.dynamics_time_manifest = str(
            kwargs.get('dynamics_time_manifest', '') or '')
        self.dynamics_time_manifest_strict = self._parse_bool(
            kwargs.get('dynamics_time_manifest_strict', True))
        self.dynamics_time_manifest_require_commit_match = self._parse_bool(
            kwargs.get('dynamics_time_manifest_require_commit_match', False))
        self.dynamics_time_manifest_content_sha256 = ''
        self.dynamics_time_manifest_file_sha256 = ''
        self._shuffled_effective_timestamps = {}

        self.track_instances = self.filter_instance(split, category_name.lower(), self.min_points)
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()
        self.virtual_rate_meta = []
        self.virtual_rate_summary = self._build_virtual_rate_summary(
            original_lengths=self.tracklet_len_list,
            filtered_lengths=self.tracklet_len_list)
        self._apply_virtual_rate()
        self._prepare_dynamics_time()
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
        if self.virtual_rate_manifest:
            manifest_name = os.path.splitext(os.path.basename(self.virtual_rate_manifest))[0]
            digest = ''
            if os.path.isfile(self.virtual_rate_manifest):
                digest = f"_{file_sha256(self.virtual_rate_manifest)[:8]}"
            return f"vr_manifest_{self._safe_tag(manifest_name)}{digest}"
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
        return f"vr_{self._safe_tag(mode)}"

    @staticmethod
    def _git_state():
        root = Path(__file__).resolve().parents[1]
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True
            ).strip()
            dirty = bool(subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'],
                cwd=str(root), text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = 'unknown', True
        return {'commit': commit, 'dirty_tracked': dirty}

    def _virtual_rate_protocol(self):
        return {
            'mode': self.virtual_rate_mode,
            'gap_pattern': list(self.virtual_rate_gap_pattern),
            'stride': self.virtual_rate_stride,
            'drop_every': self.virtual_rate_drop_every,
            'drop_prob': self.virtual_rate_drop_prob,
            'seed': self.virtual_rate_seed,
            'max_gap': self.virtual_rate_max_gap,
            'keep_first': self.virtual_rate_keep_first,
            'keep_last': self.virtual_rate_keep_last,
            'min_tracklet_len': self.virtual_rate_min_tracklet_len,
            'burst_keep_lengths': list(self.virtual_rate_burst_keep_lengths),
            'burst_skip_lengths': list(self.virtual_rate_burst_skip_lengths),
        }

    def _tracklet_identity(self, source_idx, tracklet=None):
        instance = self.track_instances[int(source_idx)]
        instance_token = str(instance.get('token', ''))
        if tracklet:
            sample_token = tracklet[0]['box_anno']['sample_token']
        else:
            first_anno = self.nusc.get(
                'sample_annotation', instance['first_annotation_token'])
            sample_token = first_anno['sample_token']
        sample = self.nusc.get('sample', sample_token)
        scene_token = str(sample['scene_token'])
        tracklet_key = (
            f"nuscenes_mf/{self.version}/{self.split}/"
            f"{scene_token}/{instance_token}")
        return {
            'tracklet_key': tracklet_key,
            'scene_token': scene_token,
            'instance_token': instance_token,
        }

    def _validate_manifest_header(self, manifest, label):
        verify_content_sha256(manifest, label)
        expected = {
            'dataset': 'nuscenes_mf',
            'version': self.version,
            'split': self.split,
            'category_name': self.category_name,
            'protocol_role': self.protocol_role,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"{label} {key} mismatch: expected={value!r}, "
                    f"found={manifest.get(key)!r}")
        code = manifest.get('code', {})
        if self.virtual_rate_manifest_require_commit_match:
            current = self._git_state()['commit']
            if code.get('commit') != current:
                raise ValueError(
                    f"{label} commit mismatch: expected current {current}, "
                    f"found {code.get('commit')}")

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
        if not self.virtual_rate_manifest:
            return None
        if not os.path.isfile(self.virtual_rate_manifest):
            if self.virtual_rate_manifest_allow_create:
                return None
            raise FileNotFoundError(
                f"Frozen virtual-rate manifest does not exist: "
                f"{self.virtual_rate_manifest}. Build it explicitly with "
                "tools/build_virtual_rate_manifest.py.")
        with open(self.virtual_rate_manifest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        self.virtual_rate_manifest_file_sha256 = file_sha256(
            self.virtual_rate_manifest)

        schema_version = int(manifest.get('schema_version', 1)) if isinstance(manifest, dict) else 1
        if schema_version >= 2:
            if manifest.get('schema') != 'ct_seqtrack.virtual_rate_manifest':
                raise ValueError('Unsupported virtual-rate manifest schema.')
            self._validate_manifest_header(manifest, 'virtual-rate manifest')
            self.virtual_rate_manifest_content_sha256 = manifest['content_sha256']

            recorded_protocol = manifest.get('protocol', {})
            expected_protocol = self._virtual_rate_protocol()
            for key, expected in expected_protocol.items():
                # ``mode: manifest`` is an explicit request to replay the file's
                # recorded cadence; every other protocol field remains strict.
                if key == 'mode' and expected == 'manifest':
                    continue
                if recorded_protocol.get(key) != expected:
                    raise ValueError(
                        f"virtual-rate manifest protocol.{key} mismatch: "
                        f"expected={expected!r}, found={recorded_protocol.get(key)!r}")

            entries = manifest.get('tracklets', [])
            by_key = {}
            for entry in entries:
                key = str(entry.get('tracklet_key', ''))
                if not key or key in by_key:
                    raise ValueError(
                        'virtual-rate manifest contains a missing or duplicate tracklet_key')
                by_key[key] = entry
            selection = [
                {
                    'tracklet_key': entry['tracklet_key'],
                    'included': bool(entry.get('included', True)),
                    'keep_indices': [int(idx) for idx in entry.get('keep_indices', [])],
                }
                for entry in entries
            ]
            if canonical_sha256(selection) != manifest.get('selection_sha256'):
                raise ValueError(
                    'virtual-rate manifest selection_sha256 mismatch')

            expected_keys = {
                self._tracklet_identity(idx, tracklet)['tracklet_key']
                for idx, tracklet in enumerate(self.tracklet_anno_list)
            }
            found_keys = set(by_key)
            if expected_keys != found_keys:
                missing = sorted(expected_keys - found_keys)[:3]
                extra = sorted(found_keys - expected_keys)[:3]
                raise ValueError(
                    'virtual-rate manifest tracklet set mismatch: '
                    f'missing={missing}, extra={extra}')
            print(
                f"loaded virtual-rate manifest {self.virtual_rate_manifest} "
                f"content_sha256={self.virtual_rate_manifest_content_sha256}")
            return by_key

        if self.virtual_rate_manifest_strict:
            raise ValueError(
                'Legacy index-keyed virtual-rate manifest rejected in strict mode. '
                'Rebuild it with schema_version=2, or set '
                'virtual_rate_manifest_strict=false only for legacy reproduction.')

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
        frame_count = int(sum(
            int(entry['kept_len']) for entry in meta if entry['included']))
        endpoint_count = int(sum(
            max(int(entry['kept_len']) - 1, 0)
            for entry in meta if entry['included']))
        selection = [
            {
                'tracklet_key': entry['tracklet_key'],
                'included': entry['included'],
                'keep_indices': entry['keep_indices'],
            }
            for entry in meta
        ]
        manifest = {
            "schema": "ct_seqtrack.virtual_rate_manifest",
            "schema_version": 2,
            "dataset": "nuscenes_mf",
            "version": self.version,
            "split": self.split,
            "category_name": self.category_name,
            "protocol_role": self.protocol_role,
            "protocol": self._virtual_rate_protocol(),
            "tracklet_count_input": len(meta),
            "tracklet_count_included": sum(bool(entry['included']) for entry in meta),
            "frame_count": frame_count,
            "endpoint_count": endpoint_count,
            "selection_sha256": canonical_sha256(selection),
            "code": self._git_state(),
            "tracklets": meta,
        }
        manifest = payload_with_content_sha256(manifest)
        with open(self.virtual_rate_manifest, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write('\n')
        self.virtual_rate_manifest_content_sha256 = manifest['content_sha256']
        self.virtual_rate_manifest_file_sha256 = file_sha256(
            self.virtual_rate_manifest)
        print(
            f"saved virtual-rate manifest {self.virtual_rate_manifest} "
            f"content_sha256={self.virtual_rate_manifest_content_sha256}")

    def _strict_manifest_keep_indices(self, entry, original_len, identity):
        if int(entry.get('original_len', -1)) != int(original_len):
            raise ValueError(
                f"virtual-rate manifest length mismatch for {identity['tracklet_key']}: "
                f"expected={original_len}, found={entry.get('original_len')}")
        for key in ('scene_token', 'instance_token'):
            if str(entry.get(key, '')) != identity[key]:
                raise ValueError(
                    f"virtual-rate manifest {key} mismatch for "
                    f"{identity['tracklet_key']}")
        raw = entry.get('keep_indices', [])
        keep = [int(idx) for idx in raw]
        if keep != sorted(set(keep)):
            raise ValueError(
                f"virtual-rate manifest keep_indices must be sorted and unique: "
                f"{identity['tracklet_key']}")
        if any(idx < 0 or idx >= original_len for idx in keep):
            raise ValueError(
                f"virtual-rate manifest keep_indices out of range: "
                f"{identity['tracklet_key']}")
        included = bool(entry.get('included', True))
        if included != (len(keep) >= max(1, self.virtual_rate_min_tracklet_len)):
            raise ValueError(
                f"virtual-rate manifest included flag is inconsistent for "
                f"{identity['tracklet_key']}")
        if included and self._validate_keep_indices(keep, original_len) != keep:
            raise ValueError(
                f"virtual-rate manifest keep_indices violate the frozen protocol for "
                f"{identity['tracklet_key']}")
        return keep, included

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

        original_lengths = list(self.tracklet_len_list)
        new_tracklets = []
        new_lengths = []
        meta = []
        manifest_meta = []

        for source_idx, tracklet in enumerate(self.tracklet_anno_list):
            identity = self._tracklet_identity(source_idx, tracklet)
            if manifest_keep is not None and identity['tracklet_key'] in manifest_keep:
                entry = manifest_keep[identity['tracklet_key']]
                keep_indices, included = self._strict_manifest_keep_indices(
                    entry, len(tracklet), identity)
            elif manifest_keep is not None and source_idx in manifest_keep:
                keep_indices = self._validate_keep_indices(
                    manifest_keep[source_idx], len(tracklet))
                included = len(keep_indices) >= max(
                    1, int(self.virtual_rate_min_tracklet_len))
            else:
                keep_indices = self._build_keep_indices(len(tracklet), source_idx)
                included = len(keep_indices) >= max(
                    1, int(self.virtual_rate_min_tracklet_len))

            record = {
                **identity,
                "source_tracklet": source_idx,
                "original_len": len(tracklet),
                "kept_len": len(keep_indices),
                "included": bool(included),
                "keep_indices": keep_indices,
            }
            manifest_meta.append(record)
            if not included:
                continue
            new_tracklets.append([tracklet[idx] for idx in keep_indices])
            new_lengths.append(len(keep_indices))
            meta.append(record)

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
        selection = [
            {
                'tracklet_key': entry['tracklet_key'],
                'included': entry['included'],
                'keep_indices': entry['keep_indices'],
            }
            for entry in manifest_meta
        ]
        self.virtual_rate_selection_sha256 = canonical_sha256(selection)
        self._save_virtual_rate_manifest(manifest_meta)

        summary = self.virtual_rate_summary
        if self.virtual_rate_mode != 'none' or self.virtual_rate_manifest:
            print(
                "virtual-rate "
                f"mode={summary['mode']} "
                f"tracklets={summary['tracklets_after']}/{summary['tracklets_before']} "
                f"frames={summary['frames_after']}/{summary['frames_before']} "
                f"drop={summary['dropped_frame_ratio']:.3f} "
                f"tag={summary['cache_tag']}"
            )

    @staticmethod
    def _anno_frame_token(anno):
        return str(anno['sample_data_lidar']['token'])

    @staticmethod
    def _anno_timestamp(anno):
        return float(anno['sample_data_lidar']['timestamp']) * 1e-6

    def get_tracklet_key(self, tracklet_id):
        return str(self.virtual_rate_meta[int(tracklet_id)]['tracklet_key'])

    def get_partition_group_key(self, tracklet_id):
        """Scene-level identity used only for v25 data partitioning."""
        return str(self.virtual_rate_meta[int(tracklet_id)]['scene_token'])

    def get_frame_token(self, tracklet_id, frame_id):
        anno = self.tracklet_anno_list[int(tracklet_id)][int(frame_id)]
        return self._anno_frame_token(anno)

    def get_endpoint_key(self, tracklet_id, frame_id):
        tracklet_id = int(tracklet_id)
        frame_id = int(frame_id)
        anno = self.tracklet_anno_list[tracklet_id][frame_id]
        return f"{self.get_tracklet_key(tracklet_id)}/frame/{self._anno_frame_token(anno)}"

    def _endpoint_records(self):
        records = []
        for tracklet_id, annos in enumerate(self.tracklet_anno_list):
            previous_timestamp = None
            for frame_id, anno in enumerate(annos):
                timestamp = self._anno_timestamp(anno)
                incoming = None
                if previous_timestamp is not None:
                    incoming = timestamp - previous_timestamp
                    if not np.isfinite(incoming) or incoming <= 0:
                        raise ValueError(
                            f"Non-positive physical timestamp gap at "
                            f"{self.get_endpoint_key(tracklet_id, frame_id)}")
                records.append({
                    'endpoint_key': self.get_endpoint_key(tracklet_id, frame_id),
                    'tracklet_key': self.get_tracklet_key(tracklet_id),
                    'frame_token': self._anno_frame_token(anno),
                    'frame_index': frame_id,
                    'timestamp_real': timestamp,
                    'real_incoming_delta_t': incoming,
                })
                previous_timestamp = timestamp
        return records

    @staticmethod
    def _deranged_permutation(size, seed):
        size = int(size)
        if size <= 1:
            return np.arange(size, dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        base = np.arange(size, dtype=np.int64)
        for _ in range(256):
            permutation = rng.permutation(size)
            if np.all(permutation != base):
                return permutation
        # Deterministic fallback that is a derangement for every size > 1.
        return np.roll(base, 1)

    def build_dynamics_time_manifest(self, output_path, seed=42):
        records = self._endpoint_records()
        transitions = [record for record in records if record['frame_index'] > 0]
        permutation = self._deranged_permutation(len(transitions), seed)

        effective_by_endpoint = {}
        mapping_rows = []
        for target_index, source_index in enumerate(permutation.tolist()):
            target = transitions[target_index]
            source = transitions[source_index]
            gap = float(source['real_incoming_delta_t'])
            effective_by_endpoint[target['endpoint_key']] = {
                'source_endpoint_key': source['endpoint_key'],
                'effective_incoming_delta_t': gap,
            }
            mapping_rows.append({
                'target_endpoint_key': target['endpoint_key'],
                'source_endpoint_key': source['endpoint_key'],
            })

        entries = []
        cumulative_by_tracklet = {}
        for record in records:
            tracklet_key = record['tracklet_key']
            if record['frame_index'] == 0:
                cumulative_by_tracklet[tracklet_key] = 0.0
                mapped = {
                    'source_endpoint_key': None,
                    'effective_incoming_delta_t': None,
                }
            else:
                mapped = effective_by_endpoint[record['endpoint_key']]
                cumulative_by_tracklet[tracklet_key] += float(
                    mapped['effective_incoming_delta_t'])
            entries.append({
                **record,
                **mapped,
                'effective_timestamp': cumulative_by_tracklet[tracklet_key],
            })

        payload = {
            'schema': 'ct_seqtrack.dynamics_time_manifest',
            'schema_version': 1,
            'dataset': 'nuscenes_mf',
            'version': self.version,
            'split': self.split,
            'category_name': self.category_name,
            'protocol_role': self.protocol_role,
            'hist_num': int(self.hist_num),
            'mode': 'shuffled',
            'seed': int(seed),
            'endpoint_count': len(entries),
            'transition_count': len(transitions),
            'virtual_rate_selection_sha256': self.virtual_rate_selection_sha256,
            'virtual_rate_manifest_content_sha256': (
                self.virtual_rate_manifest_content_sha256 or None),
            'permutation_sha256': canonical_sha256(mapping_rows),
            'code': self._git_state(),
            'entries': entries,
        }
        payload = payload_with_content_sha256(payload)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        return {
            'path': str(output),
            'content_sha256': payload['content_sha256'],
            'file_sha256': file_sha256(output),
            'permutation_sha256': payload['permutation_sha256'],
            'endpoint_count': len(entries),
            'transition_count': len(transitions),
        }

    def _load_dynamics_time_manifest(self):
        path = self.dynamics_time_manifest
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                'dynamics_time_mode=shuffled requires an existing offline '
                f'dynamics_time_manifest; not found: {path!r}')
        with open(path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        if manifest.get('schema') != 'ct_seqtrack.dynamics_time_manifest':
            raise ValueError('Unsupported dynamics-time manifest schema.')
        verify_content_sha256(manifest, 'dynamics-time manifest')
        expected = {
            'dataset': 'nuscenes_mf',
            'version': self.version,
            'split': self.split,
            'category_name': self.category_name,
            'protocol_role': self.protocol_role,
            'hist_num': int(self.hist_num),
            'mode': 'shuffled',
            'virtual_rate_selection_sha256': self.virtual_rate_selection_sha256,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"dynamics-time manifest {key} mismatch: expected={value!r}, "
                    f"found={manifest.get(key)!r}")
        if self.dynamics_time_manifest_require_commit_match:
            current = self._git_state()['commit']
            if manifest.get('code', {}).get('commit') != current:
                raise ValueError(
                    'dynamics-time manifest commit does not match current code: '
                    f"manifest={manifest.get('code', {}).get('commit')}, current={current}")

        entries = manifest.get('entries', [])
        by_endpoint = {}
        mapping_rows = []
        source_keys = []
        for entry in entries:
            key = str(entry.get('endpoint_key', ''))
            if not key or key in by_endpoint:
                raise ValueError(
                    'dynamics-time manifest contains a missing or duplicate endpoint_key')
            by_endpoint[key] = entry
            if int(entry.get('frame_index', -1)) > 0:
                source_key = str(entry.get('source_endpoint_key', ''))
                source_keys.append(source_key)
                mapping_rows.append({
                    'target_endpoint_key': key,
                    'source_endpoint_key': source_key,
                })

        expected_records = self._endpoint_records()
        expected_by_key = {
            record['endpoint_key']: record for record in expected_records
        }
        expected_keys = {record['endpoint_key'] for record in expected_records}
        if set(by_endpoint) != expected_keys:
            missing = sorted(expected_keys - set(by_endpoint))[:3]
            extra = sorted(set(by_endpoint) - expected_keys)[:3]
            raise ValueError(
                'dynamics-time manifest endpoint set mismatch: '
                f'missing={missing}, extra={extra}')
        transition_keys = {
            record['endpoint_key'] for record in expected_records
            if record['frame_index'] > 0
        }
        if set(source_keys) != transition_keys or len(source_keys) != len(set(source_keys)):
            raise ValueError(
                'dynamics-time manifest is not a one-to-one split permutation')
        actual_mapping_sha = canonical_sha256(mapping_rows)
        if actual_mapping_sha != manifest.get('permutation_sha256'):
            raise ValueError(
                'dynamics-time manifest permutation_sha256 mismatch')

        cumulative_by_tracklet = {}
        for record in expected_records:
            key = record['endpoint_key']
            entry = by_endpoint[key]
            tracklet_key = record['tracklet_key']
            if record['frame_index'] == 0:
                expected_timestamp = 0.0
                cumulative_by_tracklet[tracklet_key] = 0.0
                if entry.get('source_endpoint_key') is not None:
                    raise ValueError(
                        'First-frame dynamics-time entry must not have a source endpoint')
            else:
                source_key = str(entry['source_endpoint_key'])
                expected_gap = float(expected_by_key[source_key]['real_incoming_delta_t'])
                found_gap = float(entry['effective_incoming_delta_t'])
                if not np.isclose(found_gap, expected_gap, rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"dynamics-time mapped gap mismatch at {key}")
                cumulative_by_tracklet[tracklet_key] += expected_gap
                expected_timestamp = cumulative_by_tracklet[tracklet_key]
            if not np.isclose(
                    float(entry['effective_timestamp']), expected_timestamp,
                    rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"dynamics-time cumulative timestamp mismatch at {key}")

        self.dynamics_time_manifest_content_sha256 = manifest['content_sha256']
        self.dynamics_time_manifest_file_sha256 = file_sha256(path)
        self.dynamics_time_permutation_sha256 = actual_mapping_sha
        self._shuffled_effective_timestamps = {
            key: float(entry['effective_timestamp']) for key, entry in by_endpoint.items()
        }

    def _prepare_dynamics_time(self):
        self.dynamics_time_permutation_sha256 = ''
        if self.dynamics_time_mode == 'shuffled':
            self._load_dynamics_time_manifest()
        elif self.dynamics_time_manifest:
            raise ValueError(
                'dynamics_time_manifest is only valid when '
                'dynamics_time_mode=shuffled; clear it for true/fixed controls.')
        self.dynamics_time_summary = {
            'mode': self.dynamics_time_mode,
            'fixed_delta_t': self.dynamics_fixed_delta_t,
            'manifest': self.dynamics_time_manifest,
            'manifest_content_sha256': self.dynamics_time_manifest_content_sha256,
            'manifest_file_sha256': self.dynamics_time_manifest_file_sha256,
            'permutation_sha256': self.dynamics_time_permutation_sha256,
        }

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

        enriched = []
        for frame, frame_id in zip(frames, frame_ids):
            frame = dict(frame)
            endpoint_key = self.get_endpoint_key(seq_id, frame_id)
            if self.dynamics_time_mode == 'true':
                effective_timestamp = frame.get('timestamp')
            elif self.dynamics_time_mode == 'fixed':
                effective_timestamp = float(frame_id) * self.dynamics_fixed_delta_t
            else:
                effective_timestamp = self._shuffled_effective_timestamps[endpoint_key]
            frame['_ct_endpoint_key'] = endpoint_key
            frame['_ct_dynamics_time_mode'] = self.dynamics_time_mode
            frame['_ct_effective_timestamp'] = float(effective_timestamp)
            enriched.append(frame)
        return enriched

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
