"""预检通过真实 get_dataset 入口；仅 nuScenes 数据对象与采样数据壳替换为元数据。"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.preflight_ct_v28 import inspect_protocol
from utils.config import load_yaml_config
from utils.recursive_state import OnlineRecursiveBatchSampler


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def metadata_dataset_entry(monkeypatch):
    """运行 datasets/__init__.py，确保未省略真正入口所读取的配置字段。"""
    package = ModuleType('datasets')
    package.__path__ = [str(ROOT / 'datasets')]
    monkeypatch.setitem(sys.modules, 'datasets', package)
    constructor_calls = []

    class NuScenesMetadata:
        def __init__(self, **kwargs):
            constructor_calls.append(kwargs)
            self.ct_scene_names = kwargs['ct_scene_names']
            self.ct_scene_manifest = kwargs['ct_scene_manifest']
            # 不等长且少于16条轨迹，让完整覆盖检查同时经过部分槽尾部。
            self.lengths = [11, 8, 6, 4, 3, 1]

        def get_num_tracklets(self):
            return len(self.lengths)

        def get_num_frames_tracklet(self, index):
            return self.lengths[index]

        def get_num_frames_total(self):
            return sum(self.lengths)

    class MetadataSampler:
        def __init__(self, dataset, config):
            self.dataset = dataset
            self.num_candidates = config.num_candidates

        def __len__(self):
            return self.dataset.get_num_frames_total() * self.num_candidates

    children = {
        'sampler': dict(MotionTrackingSamplerMF=MetadataSampler,
                        TestTrackingSampler=MetadataSampler,
                        OnlineRecursiveBatchSampler=OnlineRecursiveBatchSampler),
        'nuscenes_lidar_mf': dict(NuScenesMFDataset=NuScenesMetadata),
        'kitti_mf': {}, 'waymo_data_mf': {},
    }
    for name, attributes in children.items():
        child = ModuleType(f'datasets.{name}')
        child.__dict__.update(attributes)
        monkeypatch.setattr(package, name, child, raising=False)
        monkeypatch.setitem(sys.modules, child.__name__, child)

    # protocol_utils 同样执行生产文件，恢复原模块状态由 monkeypatch 负责。
    protocol_spec = importlib.util.spec_from_file_location(
        'datasets.protocol_utils', ROOT / 'datasets/protocol_utils.py')
    protocol = importlib.util.module_from_spec(protocol_spec)
    monkeypatch.setitem(sys.modules, protocol_spec.name, protocol)
    protocol_spec.loader.exec_module(protocol)

    splits = {'train_track': [f'train{i:03}' for i in range(350)],
              'val': [f'val{i:03}' for i in range(150)]}
    nu = ModuleType('nuscenes')
    nu.__path__ = []
    nu_utils = ModuleType('nuscenes.utils')
    nu_utils.__path__ = []
    nu_splits = ModuleType('nuscenes.utils.splits')
    nu_splits.create_splits_scenes = lambda: splits
    for module in (nu, nu_utils, nu_splits):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    entry_spec = importlib.util.spec_from_file_location(
        '_ct_v28_test_dataset_entry', ROOT / 'datasets/__init__.py')
    entry = importlib.util.module_from_spec(entry_spec)
    entry_spec.loader.exec_module(entry)
    monkeypatch.setattr(package, 'get_dataset', entry.get_dataset, raising=False)
    return splits, constructor_calls


@pytest.mark.parametrize('preloading', ('missing', False, True))
def test_full_preflight_real_dataset_entry_defaults_preloading_and_preserves_yaml(
        metadata_dataset_entry, preloading):
    splits, constructor_calls = metadata_dataset_entry
    config = load_yaml_config(ROOT / 'cfgs/ct_seqtrack/28_full_nuscenes_full.yaml')
    assert 'preloading' not in config
    if preloading != 'missing':
        config['preloading'] = preloading
    cfg = SimpleNamespace(**config)

    report = inspect_protocol(cfg, splits, load_datasets=True)

    expected_preloading = False if preloading == 'missing' else preloading
    assert cfg.preloading is expected_preloading
    assert report['status'] == 'passed' and report['actual_datasets_verified']
    assert len(constructor_calls) == 3  # observation、mechanism、官方 val。
    assert [row['preloading'] for row in constructor_calls] == [expected_preloading] * 3
    assert [len(row['ct_scene_names']) for row in constructor_calls] == [350, 350, 150]
    assert [row['min_points'] for row in constructor_calls] == [-1, -1, 1]
    assert [row['preload_offset'] for row in constructor_calls] == [10, 10, -1]
    assert all(row['version'] == 'v1.0-trainval' for row in constructor_calls)
    assert report['mechanism']['coverage']['missing_endpoints'] == 0
    assert report['observation']['update_count_check']['expected_updates_per_epoch'] is None
    assert 'single full-data diagnostic' in report['initial_formal_run']
    assert cfg.ct_deterministic_algorithms and not cfg.ct_deterministic_warn_only
