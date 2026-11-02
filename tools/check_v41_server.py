"""用户上传后显式运行的 V41 KITTI CUDA 生命周期验收；不保存工程权重。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new_json(path, payload):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def output_directory(value):
    allowed = (ROOT / 'artifacts/ct_checks').resolve()
    destination = (Path(value).expanduser().resolve() if value else allowed /
        ('v41_server_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + uuid.uuid4().hex[:8]))
    require(destination != allowed and allowed in destination.parents,
            '--output-dir must be a new child directory of this project artifacts/ct_checks')
    # 连空的旧目录也不复用，失败报告与旧实验都不会被覆盖。
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def rng_equal(left, right):
    import numpy as np
    import torch
    a, b = left['numpy'], right['numpy']
    cuda_a, cuda_b = left['cuda'], right['cuda']
    return (left['python'] == right['python'] and a[0] == b[0]
        and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
        and torch.equal(left['torch'], right['torch'])
        and ((cuda_a is None and cuda_b is None) or
             (cuda_a is not None and cuda_b is not None and len(cuda_a) == len(cuda_b)
              and all(torch.equal(x, y) for x, y in zip(cuda_a, cuda_b)))))


def upstream_snapshot(tracker):
    prefixes = ('observation.', 'decoder.', 'prior.', 'multistep.')
    return dict(modes={name: module.training for name, module in tracker.named_modules()},
        buffers={name: value.detach().clone() for name, value in tracker.named_buffers()
                 if name.startswith(prefixes)})


def check_upstream_snapshot(tracker, snapshot):
    import torch
    require(snapshot['modes'] == {name: module.training for name, module in tracker.named_modules()},
            'auxiliary forward failed to restore exact module training modes')
    current = dict(tracker.named_buffers())
    require(all(name in current and torch.equal(value, current[name])
                for name, value in snapshot['buffers'].items()),
            'auxiliary forward changed upstream BN/other buffers')


def gradient_summary(named_parameters, gradients):
    import torch
    groups = {name: dict(parameters=0, connected=0, nonzero=0, squared_norm=0.)
              for name in ('observation', 'decoder', 'prior', 'multistep', 'verifier', 'recovery_head')}
    for (name, parameter), gradient in zip(named_parameters, gradients):
        group = groups[name.split('.', 1)[0]]
        group['parameters'] += parameter.numel()
        if gradient is None:
            continue
        require(bool(torch.isfinite(gradient).all()), 'nonfinite gradient: ' + name)
        norm = float(gradient.detach().double().square().sum().cpu())
        group['connected'] += 1
        group['nonzero'] += int(norm > 0.)
        group['squared_norm'] += norm
    return groups


class ActualFlashAudit:
    """检查实际调用的扩展，不能仅用配置字符串冒充 FA2 验收。"""
    def __enter__(self):
        import models.ct_v41.attention as attention
        self.module, self.original = attention, attention._flash_varlen
        self.calls = 0
        self.dropout_values = set()
        def getter():
            actual = self.original()
            def call(*args, **kwargs):
                import torch
                q, k, v = args[:3]
                require(all(item.is_cuda and item.dtype == torch.bfloat16 for item in (q, k, v)),
                        'actual FA2 must receive CUDA BF16 QKV')
                require(kwargs.get('deterministic') is True and kwargs.get('causal') is False,
                        'actual FA2 deterministic/noncausal settings differ')
                result = actual(*args, **kwargs)
                require(result.is_cuda and result.dtype == torch.bfloat16,
                        'actual FA2 output must remain CUDA BF16')
                self.calls += 1
                self.dropout_values.add(float(kwargs.get('dropout_p', 0.)))
                return result
            return call
        self.module._flash_varlen = getter
        return self

    def __exit__(self, *unused):
        self.module._flash_varlen = self.original


def check_output(batch, output):
    import torch
    from models.ct_v41.verification import current_candidate_support
    b = len(output.selected_index)
    require(output.candidates.boxes.shape == (b, 4, 4), 'expected four position candidates')
    require(output.candidates.valid.shape == (b, 4), 'candidate validity must have four slots')
    require(output.verification.logits.shape == (b, 4, 3), 'expected four S/P/T candidate outputs')
    require(output.recovery is not None, 'Full must return recovery output')
    require(bool(torch.isfinite(output.candidates.boxes).all()), 'candidate boxes must be finite')
    measurement = output.candidates.measurement_candidate
    require(not bool(measurement[:, 1:3].any()), 'propagation cannot gain measurement provenance')
    require(torch.equal(measurement[:, 0], output.candidates.raw_valid[:, 0]), 'q0 provenance mismatch')
    require(torch.equal(measurement[:, 3], output.recovery.valid), 'R provenance mismatch')
    row = torch.arange(b, device=output.selected_index.device)
    selected = output.selected_index
    require(bool(output.candidates.valid[row, selected].all()), 'selected candidate is invalid')
    support = current_candidate_support(output.observation, batch, output.candidates.boxes)
    expected_support = support['mask'][row, selected] & measurement[row, selected][:, None]
    require(torch.equal(expected_support, output.commit_decision.support_mask),
            'commit support differs from final selected geometry')
    require(torch.equal(output.accepted_box, output.candidates.boxes[row, selected]), 'accepted box mismatch')
    expected_quality = torch.where(selected == 0, output.legacy_q0_quality,
                                   torch.zeros_like(output.legacy_q0_quality))
    require(torch.equal(output.selected_quality, expected_quality), 'R/propagation reused q0 quality')
    return dict(rows=b, R_valid=int(output.recovery.valid.sum().item()),
        R_selected=int((selected == 3).sum().item()),
        trusted_eligible=int(output.commit_decision.eligible.sum().item()),
        candidate_slots=4, R_read_points=int(output.recovery.point_mask.sum().item()))


def run_smoke(args, report):
    # 源码登记在任何模型导入、CUDA 初始化、数据读取之前验证。
    from models.ct_v41.registration import verify_registration
    registration = verify_registration()
    report.update(source_sha256=registration['source']['sha256'],
        registration_sha256=hashlib.sha256((ROOT / 'cfgs/ct_seqtrack/41_registration.json').read_bytes()).hexdigest(),
        verified_engineering_file=registration['engineering_files'].get('tools/check_v41_server.py'))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.devices)
    from models.ct_v41.config import load_config
    from models.ct_v41.entry import configure_numerics, initialization_identity
    config = load_config(ROOT / 'cfgs/ct_seqtrack/41_full_kitti.yaml', dict(
        path=str(Path(args.path).expanduser().resolve()), devices=args.devices,
        ct_engineering_check=True, workers=0, log_dir=None,
        v31_evaluate_late3=False, v41_attention_backend='fa2_varlen'))
    configure_numerics(config)
    import numpy as np
    import torch
    from models.ct_v41.attention import validate_fa_runtime
    from models.ct_v41.auxiliary import isolated_auxiliary_rng
    from models.ct_v41.data import build_loaders, stable_seed
    from models.ct_v41.runtime import capture_rng_state
    from models.ctseqtrackv41 import CTSEQTRACKV41
    require(torch.cuda.is_available(), 'this explicit server check requires CUDA; CPU/math fallback is forbidden')
    torch.cuda.set_device(0)
    report['fa_runtime'] = validate_fa_runtime()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    loaders = build_loaders(config, roles=('train', 'auxiliary'))
    for role, loader in loaders.items():
        loader.batch_sampler.set_epoch(0)
        loader.generator.manual_seed(stable_seed(config.seed, role, 0))
        require(len(loader) >= args.steps, role + ' loader has fewer batches than --steps')
    main_sampler, auxiliary_sampler = (loaders[role].batch_sampler for role in ('train', 'auxiliary'))
    require(tuple(main_sampler.lengths) == tuple(auxiliary_sampler.lengths)
            and main_sampler.source_sha256 == auxiliary_sampler.source_sha256,
            'main and auxiliary must use the same training trajectories and source metadata')
    host = CTSEQTRACKV41(config, loaders=loaders).cuda().train()
    host._require_lightning()
    tracker = host.tracker
    optimizer = host.configure_optimizers()['optimizer']
    require(sum(p.numel() for p in tracker.recovery_head.parameters()) == 195, 'R must have exactly 195 parameters')
    optimizer_parameters = [p for group in optimizer.param_groups for p in group['params']]
    require(len(optimizer_parameters) == len({id(p) for p in optimizer_parameters})
            and {id(p) for p in optimizer_parameters} == {id(p) for p in tracker.parameters()},
            'optimizer must include each tracker parameter once')
    report.update(engineering_config=dict(config), initialization=initialization_identity(tracker),
        physical_gpu=args.devices, logical_cuda_device=0,
        loader_plans={role: loader.batch_sampler.state_dict() for role, loader in loaders.items()},
        steps=[], aux_isolation_checks=[])
    main_iterator, auxiliary_iterator = iter(loaders['train']), iter(loaders['auxiliary'])
    named_parameters = list(tracker.named_parameters())
    # 主 M 应执行；辅助流不得执行。hook 记录实际调用，而非推断配置。
    mtp_calls = []
    handle = tracker.multistep.register_forward_hook(lambda *_: mtp_calls.append(1))
    torch.cuda.reset_peak_memory_stats()
    try:
        with ActualFlashAudit() as flash:
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                main_batch = auxiliary_batch = main_output = auxiliary_output = None
                main_losses = auxiliary_losses = combined = auxiliary_grads = None
                torch.cuda.synchronize()
                started = time.perf_counter()
                main_rows = next(main_iterator)
                main_before = flash.calls
                main_mtp_before = len(mtp_calls)
                main_batch, main_output = host._forward_raw(main_rows, host.train_builder, training=True)
                host._pending_train = (main_output, main_batch, len(main_rows))
                main_losses = tracker.compute_losses(main_batch, main_output)
                require(len(mtp_calls) == main_mtp_before + 1, 'main Full loss must execute M once')
                main_kernel_calls = flash.calls - main_before
                require(main_kernel_calls > 0, 'main forward did not execute actual FA2')
                before = upstream_snapshot(tracker)
                rng_before, mtp_before = capture_rng_state(), len(mtp_calls)
                auxiliary_before = flash.calls
                with isolated_auxiliary_rng(config.seed, 0, step):
                    auxiliary_rows = next(auxiliary_iterator)
                    auxiliary_batch, auxiliary_output = host._forward_auxiliary_raw(auxiliary_rows)
                    host._pending_auxiliary = (auxiliary_output, auxiliary_batch, len(auxiliary_rows))
                    auxiliary_losses = tracker.compute_aux_losses(auxiliary_batch, auxiliary_output)
                    host._pending_auxiliary_diagnostics = host._auxiliary_diagnostic_records(
                        auxiliary_rows, auxiliary_batch, auxiliary_output, auxiliary_losses)
                    host._auxiliary_fetches += 1
                    require(bool(torch.isfinite(auxiliary_losses['loss_total'])), 'nonfinite auxiliary loss')
                    auxiliary_grads = torch.autograd.grad(auxiliary_losses['loss_total'],
                        [parameter for _, parameter in named_parameters], retain_graph=True, allow_unused=True)
                    isolated_gradients = gradient_summary(named_parameters, auxiliary_grads)
                    require(all(isolated_gradients[name]['connected'] == 0
                                for name in ('observation', 'decoder', 'prior', 'multistep')),
                            'auxiliary gradient reached an upstream parameter')
                    require(isolated_gradients['verifier']['connected'] > 0,
                            'auxiliary loss is disconnected from B3')
                    require(isolated_gradients['recovery_head']['connected'] > 0,
                            'auxiliary XYZ loss is disconnected from R')
                    del auxiliary_grads
                    auxiliary_grads = None
                require(rng_equal(rng_before, capture_rng_state()), 'auxiliary execution changed main RNG')
                check_upstream_snapshot(tracker, before)
                require(len(mtp_calls) == mtp_before, 'M executed in the auxiliary stream')
                auxiliary_kernel_calls = flash.calls - auxiliary_before
                require(auxiliary_kernel_calls > 0, 'auxiliary forward did not execute actual FA2')
                summaries = dict(main=check_output(main_batch, main_output),
                                 auxiliary=check_output(auxiliary_batch, auxiliary_output))
                combined = main_losses['loss_total'] + config.v41_aux_loss_weight * auxiliary_losses['loss_total']
                require(combined.ndim == 0 and bool(torch.isfinite(combined)), 'combined loss is not finite scalar')
                combined.backward()
                gradients = gradient_summary(named_parameters, [parameter.grad for _, parameter in named_parameters])
                require(sum(item['nonzero'] for item in gradients.values()) > 0, 'combined loss produced no finite nonzero gradient')
                host.on_before_optimizer_step(optimizer)
                # 每个主/辅助 batch 对只有这一次 Adam；随后两套状态各自提交。
                optimizer.step()
                host.on_train_batch_end(None, main_rows, step)
                require(host._epoch_steps == step + 1 and host._auxiliary_batches == step + 1,
                        'expected one Adam/main commit and one auxiliary commit per smoke pair')
                require(host._pending_train is None and host._pending_auxiliary is None
                        and host.train_builder._pending is None and host.auxiliary_builder._pending is None,
                        'one of the two transactions was not committed/cleaned')
                adam_steps = [int(value['step'].item()) for value in optimizer.state.values() if 'step' in value]
                require(adam_steps and max(adam_steps) == step + 1, 'unexpected extra/missing Adam step')
                torch.cuda.synchronize()
                report['steps'].append(dict(index=step, seconds=time.perf_counter()-started,
                    main_loss=float(main_losses['loss_total'].detach().cpu()),
                    auxiliary_loss=float(auxiliary_losses['loss_total'].detach().cpu()),
                    combined_loss=float(combined.detach().cpu()),
                    actual_fa_calls=dict(main=main_kernel_calls, auxiliary=auxiliary_kernel_calls),
                    batch_keys=dict(main=sorted(main_batch), auxiliary=sorted(auxiliary_batch)),
                    loss_keys=dict(main=sorted(main_losses), auxiliary=sorted(auxiliary_losses)),
                    output_keys=sorted(vars(main_output)),
                    outputs=summaries, combined_gradients=gradients,
                    allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                report['aux_isolation_checks'].append(dict(index=step, BN_buffers_equal=True,
                    modes_equal=True, RNG_equal=True, upstream_gradients_none=True,
                    M_not_executed=True, gradients=isolated_gradients))
                optimizer.zero_grad(set_to_none=True)
                main_batch = auxiliary_batch = main_output = auxiliary_output = None
                main_losses = auxiliary_losses = combined = None
                del before
            report['actual_flash'] = dict(calls=flash.calls, dropout_values=sorted(flash.dropout_values))
        report.update(status='passed_lifecycle_smoke_only', optimizer_steps=host._epoch_steps,
            main_committed_rows=host._epoch_rows, auxiliary_committed_rows=host._auxiliary_rows,
            main_M_forward_calls=len(mtp_calls), saved_checkpoints=0,
            auxiliary_diagnostics=host.auxiliary_diagnostics.state_dict(),
            remaining_main_states=len(host.train_builder.states),
            remaining_auxiliary_states=len(host.auxiliary_builder.states),
            complete_epoch_validated=False, formal_scores_validated=False)
    finally:
        handle.remove()
        host.clear_pending_materialization()
        host.train_builder.reset()
        host.auxiliary_builder.reset()
        optimizer.zero_grad(set_to_none=True)
        # 工程模型/优化器随进程丢弃；不得存checkpoint或转为正式初始化。


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--path', required=True, help='real KITTI training root')
    parser.add_argument('--devices', required=True, type=int, help='one physical CUDA GPU index')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output-dir', help='new directory under this project artifacts/ct_checks')
    args = parser.parse_args(argv)
    if args.devices < 0 or args.steps < 1:
        parser.error('--devices must be nonnegative and --steps must be positive')
    args.path = str(Path(args.path).expanduser().resolve())
    destination = output_directory(args.output_dir)
    os.chdir(ROOT)
    report = dict(schema='ct_seqtrack.v41.server_lifecycle_smoke.v1', status='started',
        run_type='engineering_scratch_discard_weights', started_utc=datetime.now(timezone.utc).isoformat(),
        requested_steps=args.steps, output_dir=str(destination),
        acceptance_scope='real KITTI main+aux forward/backward/single-Adam/dual-commit',
        pending=['server contract tests', 'complete-epoch main+aux coverage and resume acceptance'],
        interpretation=['Smoke pairs use one auxiliary batch per main batch, not the formal sparse insertion rate.',
            'Timing and memory include cold kernels and gradient-isolation instrumentation; not a formal speed benchmark.',
            'No checkpoint is saved; no tracking gain, full-epoch stability or formal-score conclusion.'])
    try:
        run_smoke(args, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), saved_checkpoints=0)
        write_new_json(destination / 'report.json', report)
        print(json.dumps(dict(status='failed', report=str(destination / 'report.json'),
                              error_type=type(error).__name__), ensure_ascii=False))
        return 1
    write_new_json(destination / 'report.json', report)
    print(json.dumps(dict(status=report['status'], source_sha256=report['source_sha256'],
        steps=len(report['steps']), optimizer_steps=report['optimizer_steps'],
        saved_checkpoints=0, report=str(destination / 'report.json')), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
