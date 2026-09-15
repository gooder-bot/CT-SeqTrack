"""只读抽取真实方法，检查索引复杂度与原始帧加载次数；不访问 nuScenes。"""
import ast
import bisect
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def extract(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), 'exec'), namespace)


scope = dict(np=np)
extract('datasets/misc_utils.py', {'get_history_frame_ids_and_masks', 'create_history_frame_dict'}, scope)
scope['deterministic_candidate_offset'] = lambda *args, **kwargs: np.zeros(3)
scope['deterministic_point_seed'] = lambda *args, **kwargs: 0
extract('datasets/sampler.py', {'_locate_tracklet', '_first_and_current_frames', '_online_raw_view'}, scope)


class Dataset:
    hist_num = 3

    def __init__(self, tracks=10000):
        self.tracks = tracks
        self.reads = []
        self.metadata = []

    def get_num_tracklets(self):
        return self.tracks

    def get_num_frames_tracklet(self, index):
        return 40

    def get_tracklet_key(self, index):
        return 'test/track'

    def get_frames(self, index, frame_ids):
        self.reads.extend(frame_ids)
        return [dict(pc='FULL_POINT_CLOUD', frame_id=i, timestamp=i*.5,
                     **{'3d_bbox': 'BOX'}) for i in frame_ids]

    def get_frame_metadata(self, index, frame_id):
        self.metadata.append(frame_id)
        return dict(frame_id=frame_id, timestamp=frame_id*.5, **{'3d_bbox': 'BOX'})


class Sampler:
    _first_and_current_frames = scope['_first_and_current_frames']
    _online_raw_view = scope['_online_raw_view']
    _locate_tracklet = scope['_locate_tracklet']
    use_b1motion_v3 = True

    def __init__(self, tracks=10000, perf=True):
        self.dataset = Dataset(tracks)
        self.tracklet_start_ids = list(range(0, tracks*40+1, 40))
        self.config = SimpleNamespace(ct_enable_v29=True,
            ct_runtime_optimization='equivalent_v1' if perf else '')

    def _motion_v3_aux_offsets(self, frame):
        return [2, 4, 6]


results = {'scope': 'synthetic point IO call counts; actual methods AST-extracted unchanged; CPU timings only'}
io_rows = []
for perf in (False, True):
    for shadow in (False, True):
        sampler = Sampler(perf=perf)
        raw = sampler._online_raw_view(0, 0, 0, 0, 10, 0, shadow)
        io_rows.append(dict(perf=perf, shadow=shadow, full_cloud_reads=len(sampler.dataset.reads),
            unique_frame_tokens=len(set(sampler.dataset.reads)), frame_ids=sampler.dataset.reads,
            metadata_reads=len(sampler.dataset.metadata), auxiliary_clouds=3*(1+len(raw['shadow_future']))))
results['io_calls_per_mechanism_row'] = io_rows

def fast(sampler, index):
    if index < 0 or index >= sampler.tracklet_start_ids[-1]:
        raise IndexError(index)
    i = bisect.bisect_right(sampler.tracklet_start_ids, index)-1
    return i, index-sampler.tracklet_start_ids[i]

lookup_rows = []
for tracks in (274, 10000):
    sampler = Sampler(tracks)
    indexes = np.random.default_rng(42).integers(0, tracks*40, 4000).tolist()
    expected = [sampler._locate_tracklet(index) for index in indexes]
    assert expected == [fast(sampler, index) for index in indexes]
    timings = {}
    for name, fn in [('linear', sampler._locate_tracklet), ('bisect', lambda i: fast(sampler, i))]:
        samples = []
        for _ in range(3):
            before = time.perf_counter()
            for index in indexes:
                fn(index)
            samples.append((time.perf_counter()-before)*1000)
        timings[name+'_median_ms_4000_queries'] = statistics.median(samples)
    lookup_rows.append(dict(synthetic_tracklets=tracks, equality=True, **timings))
results['lookup'] = lookup_rows
out = Path(__file__).with_name('data_path_probe.json')
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=False, indent=2))
