"""有层级的配对 wall/stream 计时；不把 CPU 等待误称为 CUDA 算子时间。"""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import time
import torch


class StageRecorder:
    def __init__(self, device):
        self.device = torch.device(device)
        self.records = []
        self.stack = []

    @contextmanager
    def stage(self, name):
        record = dict(name=name, children=0., start=time.perf_counter())
        if self.device.type == 'cuda':
            record['begin'] = torch.cuda.Event(enable_timing=True)
            record['end'] = torch.cuda.Event(enable_timing=True)
            record['begin'].record()
        self.stack.append(record)
        try:
            with torch.profiler.record_function('v39::'+name):
                yield
        finally:
            if self.device.type == 'cuda':
                record['end'].record()
            elapsed = time.perf_counter()-record['start']
            record['wall_ms'] = 1000.*elapsed
            record['self_wall_ms'] = 1000.*(elapsed-record['children'])
            self.stack.pop()
            if self.stack:
                self.stack[-1]['children'] += elapsed
            self.records.append(record)

    def wrap(self, owner, method, name):
        original = getattr(owner, method)
        @wraps(original)
        def invoke(*args, **kwargs):
            with self.stage(name):
                return original(*args, **kwargs)
        setattr(owner, method, invoke)
        return original

    def clear(self):
        self.records.clear()

    def summary(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        groups = defaultdict(list)
        for record in self.records:
            clean = {k: record[k] for k in ('wall_ms','self_wall_ms')}
            if self.device.type == 'cuda':
                clean['cuda_stream_span_ms'] = record['begin'].elapsed_time(record['end'])
            groups[record['name']].append(clean)
        def distribution(values):
            array = torch.tensor(values, dtype=torch.float64)
            return dict(total=float(array.sum()), mean=float(array.mean()),
                        median=float(array.median()), p95=float(torch.quantile(array,.95)))
        return {name: dict(calls=len(values), **{key: distribution([v[key] for v in values])
                    for key in values[0]}) for name,values in groups.items()}


def attach_stages(recorder, model, builder, service):
    for owner, method, name in (
        (service,'fetch_many','raw_decode_cache'), (service,'tensors','raw_transfer_local'),
        (service,'cached_tensors','raw_device_cache_lookup'), (builder,'_raw','raw_lookup_local'),
        (builder,'begin','begin'), (builder,'prepare_current_G','prepare_G'),
        (builder,'prepare_search','prepare_search'), (builder,'materialize_B0','materialize_B0'),
        (builder,'commit','commit'), (model,'plan_prior','B1'), (model,'_prefix','prefix_route_search'),
        (model.g,'encode','G_encode'), (model.g,'refine','G_refine'),
        (model.g,'verify','G_verify'), (model.g,'coarse_proposals','coarse_search'),
        (model.observation,'forward','B0_observation'), (model.decoder,'forward','B0_decoder')):
        recorder.wrap(owner,method,name)
