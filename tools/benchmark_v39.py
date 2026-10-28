#!/usr/bin/env python
"""v39 工程 profile：同初始参数/端点的两种 E 执行配方端到端对比。

不写 output、不改正式配置；必须显式给 artifacts/ct_checks 下的新目录。
CUDA event 是含 CPU 发射间隙的 stream span；算子 CUDA 时间以 trace 为准。
plain warmup+steps 后只额外连续一批做 trace。BF16 可改变递归状态和后续
crop；本工具记录工作量差异，不把结果称为同张量算子的因果收益。
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from models.ct_v39.config import load_config, normalize_config
from models.ct_v39.raw_service import RawFrameService
from models.ct_v39.data import BatchBuilder, build_loaders
from models.ct_v39.model import V39Model
from models.ct_v39.profiling import StageRecorder, attach_stages
from utils.bn_policy import running_batch_norm


def parameter_hash(model):
    digest = hashlib.sha256()
    for name,value in model.named_parameters():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def release_profile_memory():
    """先释放方法 wrapper/MethodType 自引用，再清空 CUDA allocator 缓存。

    empty_cache 不会释放循环垃圾中仍被持有的张量；两配方之间必须先 GC，
    防止 reference 的参数和暂存张量污染 optimized 的显存峰值。
    """
    gc.collect()
    torch.cuda.empty_cache()


def workload(tx, committed):
    """计时外复用 commit 的 CPU packet，不增加逐字段 D2H。"""
    endpoints=[(r['request'].epoch,r['tracklet_key'],r['request'].branch,
                r['request'].window_start,r['request'].frame) for r in tx.rows]
    result=dict(endpoints=endpoints,full_executed=[bool(r['full_executed']) for r in committed],
                policy_full=[bool(r['policy_full']) for r in committed],
                search=[bool(r['search_triggered']) for r in committed])
    for name in ('diagnostic_raw_point_count','diagnostic_crop_point_count',
                 'diagnostic_sampled_point_count','diagnostic_current_G_point_count'):
        value=tx.diagnostics_cpu.get(name)
        result[name]=None if value is None else np.asarray(value).tolist()
    result['endpoint_sha256']=hashlib.sha256(json.dumps(endpoints,separators=(',',':')).encode()).hexdigest()
    result['workload_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def run(config, destination, *, optimized, steps, warmup, training, checkpoint=None, mode='accuracy'):
    if steps<1 or warmup<0:
        raise ValueError('profile requires steps >= 1 and warmup >= 0')
    torch.manual_seed(config.seed);np.random.seed(config.seed)
    overrides = dict(config, ct_engineering_check=True, log_dir=str(destination),
        v39_point_packing=optimized, v39_feature_bf16=optimized,
        v39_attention_backend='fa2_varlen' if optimized else 'math_reference',
        execution_profile='optimized_bf16' if optimized else 'optimized_fp32',
        v31_evaluate_late3=False, test=mode=='adaptive', checkpoint=checkpoint if mode=='adaptive' else None,
        v39_inference_mode=mode)
    cfg = normalize_config(overrides)
    device = torch.device('cuda:0')
    service=RawFrameService(cfg);builder=BatchBuilder(cfg,service)
    try:
        loaders=build_loaders(cfg,roles=('train' if training else 'val',),raw_service=service)
        loader=loaders['train' if training else 'val']
        if hasattr(loader.batch_sampler,'set_epoch'):
            loader.batch_sampler.set_epoch(11)
        required=warmup+steps+1
        if len(loader)<required:
            raise ValueError(f'profile needs {required} batches including one trace; loader provides {len(loader)}')
        model=V39Model(cfg).to(device).train(training)
        if checkpoint:
            payload=torch.load(checkpoint,map_location='cpu')
            state=payload['state_dict']
            for prefix in ('tracker.','model.'):
                if any(key.startswith(prefix) for key in state):
                    state={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)};break
            model.load_state_dict(state,strict=True)
        initial_hash=parameter_hash(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr,betas=(.5,.999),eps=1e-6,
                                   weight_decay=cfg.wd,foreach=False,fused=False) if training else None
        destination.mkdir(parents=True,exist_ok=False)
        iterator=iter(loader)
        def execute(recorder=None):
            scope=lambda name: recorder.stage(name) if recorder is not None else nullcontext()
            with scope('endpoint_total'):
                with scope('metadata_loader'):
                    rows=next(iterator)
                with torch.set_grad_enabled(training):
                    tx=builder.begin(rows,device,training=training)
                    use_running=training and (len(rows)<cfg.batch_size or
                        any(getattr(row['request'],'drain',False) for row in rows))
                    with running_batch_norm(model.observation,enabled=use_running):
                        output=model.forward_transaction(tx,builder)
                    if training:
                        with scope('supervision_loss'):
                            losses=model.compute_losses(tx,output)
                        with scope('backward'):
                            losses['loss_total'].backward()
                        with scope('optimizer'):
                            optimizer.step();optimizer.zero_grad(set_to_none=True)
                    committed=builder.commit(output,tx,diagnostics=False)
            return tx,output,committed

        end_to_end=[];workloads=[]
        torch.cuda.reset_peak_memory_stats(device)
        # plain pass 没有 profiler、record_function wrapper 或 CUDA event 干扰。
        for index in range(warmup+steps):
            torch.cuda.synchronize(device);start=time.perf_counter()
            tx,output,committed=execute()
            torch.cuda.synchronize(device);elapsed=(time.perf_counter()-start)*1000.
            if index>=warmup:
                end_to_end.append(elapsed);workloads.append(workload(tx,committed))
        plain_peak_allocated=torch.cuda.max_memory_allocated(device)
        plain_peak_reserved=torch.cuda.max_memory_reserved(device)
        plain_cache=dict(service.stats)
        plain_kernel_calls=sum(m.kernel_calls for m in model.modules() if hasattr(m,'kernel_calls'))
        # 延续同一个窗口/优化器，仅再运行一批用于归因；不混入 plain 速度统计。
        recorder=StageRecorder(device);attach_stages(recorder,model,builder,service)
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as profile:
            tx,output,committed=execute(recorder)
            torch.cuda.synchronize(device);profile.step()
        trace_workload=workload(tx,committed)
        profile.export_chrome_trace(str(destination/'trace.json'))
        (destination/'operators.txt').write_text(profile.key_averages().table(sort_by='self_cuda_time_total',row_limit=60),encoding='utf-8')
        backends=sorted({str(module.last_backend) for module in model.modules() if hasattr(module,'last_backend')})
        if optimized and plain_kernel_calls<1:
            raise RuntimeError('optimized profile did not demonstrate an actual FA2 call')
        count=sum(len(w['endpoints']) for w in workloads)
        full=sum(sum(w['full_executed']) for w in workloads);searched=sum(sum(w['search']) for w in workloads)
        result=dict(schema='ct_v39_profile_v2',optimized=optimized,training=training,mode=mode,steps=steps,warmup=warmup,
            initial_parameter_sha256=initial_hash,attention_backends=backends,stages=recorder.summary(),
            batch_wall_ms=dict(median=float(np.median(end_to_end)),p95=float(np.quantile(end_to_end,.95))),
            samples=count,b0_call_rate=full/max(count,1),search_rate=searched/max(count,1),
            peak_allocated_bytes=plain_peak_allocated,peak_reserved_bytes=plain_peak_reserved,
            cache=plain_cache,cache_after_trace=dict(service.stats),plain_fa2_kernel_calls=plain_kernel_calls,
            workloads=workloads,trace_workload=trace_workload,plain_batch_times_ms=end_to_end,
            torch=torch.__version__,gpu=torch.cuda.get_device_name(device),
            sharing='record nvidia-smi snapshot; results are not exclusive unless independently verified',
            timing_note='batch_wall/peak_memory are plain warmup+steps with profiler disabled; stages/operators are one subsequent trace batch; stream spans include launch gaps',
            comparison_note='same initialization/endpoints; BF16/dropout/update changes may change recursive state/crop work; not fixed-tensor causal speedup')
        (destination/'profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    finally:
        builder.abort();service.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg',default='cfgs/ct_seqtrack/39_full_a_mini.yaml')
    parser.add_argument('--path');parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=12);parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--training',action='store_true');parser.add_argument('--checkpoint')
    parser.add_argument('--inference-mode',choices=('accuracy','adaptive'),default='accuracy')
    args=parser.parse_args();destination=Path(args.output).resolve()
    if args.steps<1 or args.warmup<0:parser.error('steps must be >= 1 and warmup must be >= 0')
    allowed=(ROOT/'artifacts/ct_checks').resolve()
    if allowed not in destination.parents or destination.exists():
        parser.error('output must be a new subdirectory of artifacts/ct_checks')
    cfg=load_config(args.cfg)
    if args.path: cfg.path=args.path
    if args.inference_mode=='adaptive' and (args.training or not args.checkpoint):
        parser.error('adaptive profile requires an evaluated checkpoint and cannot use --training')
    from tools.register_v39 import verify_registration
    verify_registration()
    from models.ct_v39.entry import apply_device_selection, configure_numerics
    apply_device_selection(cfg);configure_numerics()
    if not torch.cuda.is_available(): parser.error('paired FA2 profiling requires CUDA')
    destination.mkdir(parents=True)
    results={}
    for name,optimized in (('reference',False),('optimized',True)):
        results[name]=run(cfg,destination/name,optimized=optimized,steps=args.steps,warmup=args.warmup,
                          training=args.training,checkpoint=args.checkpoint,mode=args.inference_mode)
        release_profile_memory()
    if results['reference']['initial_parameter_sha256'] != results['optimized']['initial_parameter_sha256']:
        raise RuntimeError('paired profiles did not start with identical parameters')
    endpoint_match=([w['endpoint_sha256'] for w in results['reference']['workloads']]
                    ==[w['endpoint_sha256'] for w in results['optimized']['workloads']])
    if not endpoint_match:raise RuntimeError('paired profiles used different endpoint schedules')
    results['comparison']=dict(endpoint_schedule_matches=True,
        workload_hashes_match=([w['workload_sha256'] for w in results['reference']['workloads']]
                              ==[w['workload_sha256'] for w in results['optimized']['workloads']]),
        interpretation='end-to-end execution recipes; numerical changes can change crop/search work')
    (destination/'paired.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(destination/'paired.json'))


if __name__=='__main__': main()
