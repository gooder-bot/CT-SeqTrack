"""profile 的 CPU 计时范围、工作量身份与无副作用 CLI 验收。"""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import os
import gc
import weakref

import numpy as np
import pytest
import torch

from models.ct_v39.profiling import StageRecorder, attach_stages
from tools.benchmark_v39 import workload, release_profile_memory


ROOT=Path(__file__).resolve().parents[1]


def test_benchmark_cli_help_and_invalid_counts_do_not_need_gpu(tmp_path):
    script=ROOT/'tools/benchmark_v39.py'
    environment=dict(os.environ,PYTHONIOENCODING='utf-8')
    help_result=subprocess.run([sys.executable,str(script),'--help'],cwd=ROOT,
                               text=True,capture_output=True,encoding='utf-8',env=environment)
    assert help_result.returncode==0
    assert '--inference-mode' in help_result.stdout and '--warmup' in help_result.stdout
    output=tmp_path/'must_not_exist'
    invalid=subprocess.run([sys.executable,str(script),'--steps','0','--output',str(output)],
                           cwd=ROOT,text=True,capture_output=True,encoding='utf-8',env=environment)
    assert invalid.returncode!=0 and 'steps must be' in invalid.stderr
    assert not output.exists()


def test_nested_cpu_stage_self_time_does_not_double_count():
    recorder=StageRecorder('cpu')
    with recorder.stage('outer'):
        with recorder.stage('child'):
            torch.arange(100).sum()
    result=recorder.summary()
    outer,child=result['outer'],result['child']
    assert outer['calls']==child['calls']==1
    assert outer['wall_ms']['total']>=child['wall_ms']['total']
    assert outer['self_wall_ms']['total']+child['self_wall_ms']['total']==pytest.approx(outer['wall_ms']['total'])
    assert 'cuda_stream_span_ms' not in outer


def test_profile_cleanup_releases_wrapper_cycle_before_emptying_cuda_cache(monkeypatch):
    """真实计时 wrapper 的 bound method 会保活 module，empty_cache 前应已释放。"""
    enabled = gc.isenabled()
    gc.disable()
    try:
        owner = torch.nn.Linear(2, 2)
        recorder = StageRecorder('cpu')
        recorder.wrap(owner, 'forward', 'linear')
        reference = weakref.ref(owner)
        del owner, recorder
        assert reference() is not None
        observed = []
        monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: observed.append(reference() is None))
        release_profile_memory()
        assert observed == [True]
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


def test_workload_hash_distinguishes_crop_work_from_endpoint_identity():
    from tests.test_ct_v39_data_contracts import request
    tx=SimpleNamespace(rows=[dict(request=request(),tracklet_key='tiny')],
                       diagnostics_cpu=dict(diagnostic_raw_point_count=np.array([80]),
                                            diagnostic_sampled_point_count=np.array([8])))
    committed=[dict(full_executed=True,policy_full=False,search_triggered=False)]
    first=workload(tx,committed)
    tx.diagnostics_cpu['diagnostic_sampled_point_count']=np.array([3])
    second=workload(tx,committed)
    assert first['endpoint_sha256']==second['endpoint_sha256']
    assert first['workload_sha256']!=second['workload_sha256']
    assert first['full_executed']==[True] and first['policy_full']==[False]


def test_stage_attachment_covers_real_cpu_transaction_and_device_cache_hits():
    from models.ct_v39.raw_service import RawFrameService
    from models.ct_v39.data import BatchBuilder, RawEndpointDataset
    from models.ct_v39.model import V39Model
    from tests.test_ct_v39_data_contracts import config, MetadataSource, request
    previous=torch.get_num_threads();torch.set_num_threads(1)
    cfg=config();service=RawFrameService(cfg)
    try:
        builder=BatchBuilder(cfg,service);raw=RawEndpointDataset(MetadataSource(),service)
        model=V39Model(cfg).eval();recorder=StageRecorder('cpu')
        attach_stages(recorder,model,builder,service)
        with torch.no_grad(),recorder.stage('endpoint_total'):
            tx=builder.begin([raw[request()]],'cpu',training=False)
            output=model.forward_transaction(tx,builder)
            builder.commit(output,tx)
        summary=recorder.summary()
        for name in ('begin','commit','B1','B0_observation','B0_decoder','G_verify',
                     'prefix_route_search','raw_lookup_local','raw_device_cache_lookup'):
            assert summary[name]['calls']>=1,name
        assert not recorder.stack
    finally:
        service.close();torch.set_num_threads(previous)
