"""v35/v36/v37 配对等价性、吞吐及 BF16 工程检查；不保存训练权重。"""
from __future__ import annotations

import argparse
from dataclasses import is_dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', type=Path, required=True)
    parser.add_argument('--path')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--mode', choices=('check', 'benchmark', 'bf16', 'all'), default='check')
    parser.add_argument('--candidate_profile', choices=('optimized_fp32', 'optimized_bf16'),
                        default='optimized_fp32', help='benchmark/all 也可显式检查 BF16；不更改正式配方')
    parser.add_argument('--baseline_profile', choices=('legacy', 'optimized_fp32'),
                        help='v37 BF16 默认对比 optimized_fp32；其余默认 legacy')
    parser.add_argument('--inference', action='store_true', help='另测闭环推理吞吐（同样预热/计时/重复）')
    parser.add_argument('--output', type=Path, help='新的 artifacts/ct_checks/.../report.json')
    parser.add_argument('--check_steps', type=int, default=8)
    parser.add_argument('--check_epochs', nargs='+', type=int, default=[0, 10],
                        help='零基课程 epoch；默认同时检查初始课程和成熟递推课程')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trace_steps', type=int, default=5)
    parser.add_argument('--weights', type=Path, help='仅 check/bf16 的只读历史权重；不作 resume')
    parser.add_argument('--run_manifest', type=Path, help='--weights 所属原 run_manifest.json')
    args = parser.parse_args(argv)
    if min(args.check_steps, args.steps, args.repeats) < 1 or min(args.warmup, args.trace_steps) < 0:
        parser.error('steps/repeats must be positive; warmup/trace_steps must be nonnegative')
    if any(value < 0 for value in args.check_epochs) or len(set(args.check_epochs)) != len(args.check_epochs):
        parser.error('check_epochs must contain distinct nonnegative epoch indices')
    if bool(args.weights) != bool(args.run_manifest):
        parser.error('--weights and --run_manifest must be supplied together')
    if args.weights and args.mode in ('benchmark', 'all'):
        parser.error('historical weights are for correctness/BF16 comparison only, not throughput runs')
    return args


def module_set(config):
    version = model_version(config)
    package = 'models.ct_v' + version
    modules = {name: importlib.import_module(package + '.' + name)
               for name in ('config', 'data', 'runtime', 'candidate_records')}
    host = importlib.import_module('models.ctseqtrackv' + version)
    modules['host'] = getattr(host, 'CTSEQTRACKV' + version.upper())
    return modules


def model_version(config):
    versions = {'ctseqtrackv35': '31', 'ctseqtrackv36': '36', 'ctseqtrackv37': '37'}
    if config['net_model'] not in versions:
        raise ValueError('execution checks support v35/v36/v37 only')
    return versions[config['net_model']]


def paired_baseline(config, candidate_profile, requested=None):
    baseline = requested or ('optimized_fp32' if config['net_model'] == 'ctseqtrackv37'
                             and candidate_profile == 'optimized_bf16' else 'legacy')
    if baseline not in ('legacy', 'optimized_fp32') or baseline == candidate_profile:
        raise ValueError('baseline and candidate must be distinct supported execution profiles')
    return baseline


def load_engineering_config(path, data_root=None):
    from utils.config import load_yaml_config
    raw = load_yaml_config(path)
    name = raw.get('net_model')
    api = importlib.import_module('models.ct_v' + model_version(raw) + '.config')
    overrides = dict(ct_engineering_check=True, checkpoint=None, init_checkpoint=None,
                     test=False, log_dir=None, execution_profile='optimized_fp32')
    if data_root is not None:
        overrides['path'] = data_root
    return api.load_config(path, overrides)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze(value):
    """只读 CPU 快照；私有执行缓存不属于模型/数据/日志协议。"""
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()
                if not (isinstance(key, str) and key.startswith('_ct_'))}
    if isinstance(value, (tuple, list)):
        return type(value)(freeze(item) for item in value)
    if is_dataclass(value):
        # 动态附加的 loss_statistics 同样保留。
        return {key: freeze(item) for key, item in vars(value).items()}
    if hasattr(value, '__dict__') and not isinstance(value, type):
        return {key: freeze(item) for key, item in vars(value).items()}
    return value


def assert_identical(expected, actual, path='root'):
    """按元素存储位比较，含零的符号和 NaN payload；合法输出另查有限值。"""
    import numpy as np
    import struct
    import torch
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(path + ': tensor dtype/shape mismatch')
        # size=1 的 expand 可能 stride=0 且被判定 contiguous，必须显式拷成标准布局。
        left = expected.detach().reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
        right = actual.detach().reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
        if not torch.equal(left, right):
            raise AssertionError(path + ': tensor values differ')
    elif isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(path + ': array dtype/shape mismatch')
        if expected.tobytes(order='C') != actual.tobytes(order='C'):
            raise AssertionError(path + ': array values differ')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or list(expected) != list(actual):
            raise AssertionError(path + ': fields/order differ')
        for key in expected:
            assert_identical(expected[key], actual[key], path + '/' + str(key))
    elif isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(path + ': sequence mismatch')
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_identical(left, right, path + '/' + str(index))
    elif isinstance(expected, (float, np.floating)):
        if type(expected) is not type(actual):
            raise AssertionError(path + ': scalar dtype differs')
        left = expected.tobytes() if isinstance(expected, np.floating) else struct.pack('>d', expected)
        right = actual.tobytes() if isinstance(actual, np.floating) else struct.pack('>d', actual)
        if left != right:
            raise AssertionError(path + ': scalar bits differ')
    elif expected != actual:
        raise AssertionError(path + ': scalar differs: ' + repr(expected) + ' vs ' + repr(actual))


def finite_training(model, losses, output):
    import torch
    for name, value in losses.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('nonfinite loss: ' + name)
    if not bool(torch.isfinite(output.accepted_box).all()):
        raise FloatingPointError('nonfinite accepted box')
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError('nonfinite parameter: ' + name)
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError('nonfinite gradient: ' + name)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encode_report(report):
    """数值诊断自身失败时仍能写出合法 failed JSON，不把 NaN 静默变成成绩。"""
    try:
        return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', report['status']
    except (TypeError, ValueError) as error:
        failure = dict(schema='ct_seqtrack.execution_check.v1', status='failed',
            error='report serialization failed: ' + str(error), original_error=report.get('error'),
            saved_training_weights=False)
        return json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'failed'


def verified_engineering_weights(config, checkpoint_path, manifest_path):
    """验证原 run 的来源，只取模型状态；绝不伪造正式恢复 payload。"""
    import torch
    from utils.config import load_yaml_config
    checkpoint_path, manifest_path = Path(checkpoint_path).resolve(), Path(manifest_path).resolve()
    if manifest_path.name != 'run_manifest.json' or manifest_path.parent not in checkpoint_path.parents:
        raise ValueError('weights must belong to the supplied original run directory')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    original_cfg = manifest_path.parent / 'resolved_config.yaml'
    original = module_set(config)['config'].normalize_config(load_yaml_config(original_cfg))
    if module_set(config)['config'].config_identity(original) != manifest.get('config_sha256'):
        raise ValueError('original resolved config does not match its manifest')
    for key in ('net_model', 'seed', 'hist_num', 'point_sample_size', 'v31_temporal_backend'):
        if original[key] != config[key]:
            raise ValueError('engineering weights/config mismatch: ' + key)
    version = model_version(config)
    arm_key = 'v' + version + '_arm'
    if original[arm_key] != config[arm_key] or manifest.get('model') != config.net_model:
        raise ValueError('engineering weights must match model and arm')
    source = manifest.get('source', {})
    file_hashes = source.get('files')
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError('original source manifest is required')
    digest = hashlib.sha256(json.dumps(file_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != source.get('sha256'):
        raise ValueError('original source manifest checksum mismatch')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    runtime_key = 'ct_v' + ('35' if version == '31' else version) + '_runtime'
    payload = saved.get(runtime_key, {})
    if payload.get('config_sha256') != manifest['config_sha256'] or payload.get('epoch_complete') is not True:
        raise ValueError('weights require original config identity and a complete epoch')
    identity = importlib.import_module('models.ct_v' + version + '.identity')
    if (payload.get('schema') != identity.runtime_schema(original)
            or payload.get('model_schema') != identity.model_schema(original)
            or manifest.get('schema') != identity.model_schema(original)):
        raise ValueError('original checkpoint/runtime/model schema mismatch')
    epoch = payload.get('completed_epoch')
    if type(epoch) is not int or epoch < 1:
        raise ValueError('original checkpoint requires a positive completed epoch')
    if version in ('36', '37') and payload.get('source') != source:
        raise ValueError('original checkpoint/source mismatch')
    if version in ('36', '37'):
        run_id = manifest.get('run_id')
        if not isinstance(run_id, str) or not run_id or payload.get('run_id') != run_id:
            raise ValueError('original checkpoint/run_id mismatch')
    state = saved.get('state_dict')
    if not isinstance(state, dict):
        raise ValueError('checkpoint has no model state_dict')
    return state, dict(checkpoint_sha256=sha256_file(checkpoint_path),
        manifest_sha256=sha256_file(manifest_path), source_sha256=digest,
        purpose='read_only_engineering_weights_not_resume',
        completed_epoch=payload.get('completed_epoch'))


def _make_pair(config, device, candidate_profile, weights=None, epoch_index=0, baseline_profile=None):
    api = module_set(config)
    result = []
    for profile in (paired_baseline(config, candidate_profile, baseline_profile), candidate_profile):
        cfg = api['config'].normalize_config(dict(config, execution_profile=profile, log_dir=None))
        seed_all(cfg.seed)
        model = api['host'](cfg).to(device).train()
        if weights is not None:
            model.load_state_dict(weights, strict=True)
        # 无 Lightning 的本地合成检查仍使用原 host 事务，无训练日志替代实现。
        if not hasattr(model, 'current_epoch'):
            model.current_epoch = epoch_index
        model.on_train_epoch_start()
        if model.training_diagnostics is not None:
            model.training_diagnostics.reset(epoch_index)
        optimizer = model.configure_optimizers()['optimizer']
        result.append((model, optimizer))
    assert_identical(freeze(result[0][0].state_dict()), freeze(result[1][0].state_dict()), 'initial_parameters')
    return result


def _step(model, optimizer, rows, api):
    import torch
    optimizer.zero_grad(set_to_none=True)
    try:
        batch, output = model._forward_raw(rows, model.train_builder, training=True)
        if model.training_diagnostics is not None:
            output.loss_statistics = {}
        losses = model.tracker.compute_losses(batch, output)
        if model.training_diagnostics is not None:
            model.training_diagnostics.add_batch(rows, batch, output, losses)
        candidates = api['candidate_records'].candidate_records(batch, output)
        losses['loss_total'].backward()
        finite_training(model, losses, output)
        before = dict(prepared=freeze(batch), output=freeze(output), losses=freeze(losses),
            gradients={name: freeze(parameter.grad) for name, parameter in model.named_parameters()},
            candidates=freeze(candidates))
        if hasattr(getattr(output, 'verification', None), 'measurement_score'):
            before['output']['verification']['measurement_score'] = freeze(output.verification.measurement_score)
        optimizer.step()
        finite_training(model, losses, output)
        model._pending_train = (output, batch, len(rows))
        model.on_train_batch_end(None, rows, 0)
        before.update(parameters=freeze(model.state_dict()), optimizer=freeze(optimizer.state_dict()),
            state=freeze(model.train_builder.states), rng=freeze(api['runtime'].capture_rng_state()),
            host_steps=model._epoch_steps,
            diagnostics=(freeze(model.training_diagnostics.summary())
                         if model.training_diagnostics is not None else None))
        return before
    except BaseException:
        model.train_builder.reset()
        model._pending_train = None
        raise


def _numerical_delta(left, right):
    import torch
    def equal(a, b):
        try:
            assert_identical(a, b)
            return True
        except AssertionError:
            return False
    def finite_delta(a, b, valid=None):
        mask = torch.isfinite(a) & torch.isfinite(b)
        if valid is not None:
            mask = mask & valid
        return float((a[mask] - b[mask]).abs().max()) if bool(mask.any()) else None
    box_a, box_b = left['output']['accepted_box'], right['output']['accepted_box']
    selected_a, selected_b = left['output']['selected_index'], right['output']['selected_index']
    report = dict(max_abs_accepted_box=float((box_a - box_b).abs().max()),
        max_abs_center_m=float((box_a[:, :3] - box_b[:, :3]).abs().max()),
        max_abs_yaw_rad=float(torch.atan2((box_a[:, 3] - box_b[:, 3]).sin(),
                                         (box_a[:, 3] - box_b[:, 3]).cos()).abs().max()),
        output_choice_flips=int((selected_a != selected_b).sum()), endpoints=len(box_a),
        loss_abs_differences={name: float((value - right['losses'][name]).abs())
                              for name, value in left['losses'].items()})
    decision_a, decision_b = left['output'].get('decision'), right['output'].get('decision')
    if left['output'].get('commit_quality') is not None:
        a, b = left['output']['verification'], right['output']['verification']
        qa, qb = left['output']['commit_quality'], right['output']['commit_quality']
        report['max_abs_commit_quality'] = finite_delta(qa, qb)
        report['commit_quality_threshold_flips'] = int(((qa >= .5) != (qb >= .5)).sum())
        report['max_abs_output_utility'] = finite_delta(
            a['logits'][..., :2].sigmoid().mean(-1), b['logits'][..., :2].sigmoid().mean(-1))
        report['verification_support_flips'] = int((a['support_mask'] != b['support_mask']).sum())
        if a.get('commit_support_mask') is not None and b.get('commit_support_mask') is not None:
            ca, cb = a['commit_support_mask'], b['commit_support_mask']
            report['commit_support_flips'] = int((ca != cb).sum())
            row = torch.arange(len(selected_a))
            eligible_a = (ca.sum(-1)[row, selected_a] >= 3) & (qa[row, selected_a] >= .5)
            eligible_b = (cb.sum(-1)[row, selected_b] >= 3) & (qb[row, selected_b] >= .5)
            report['selected_update_eligibility_flips'] = int((eligible_a != eligible_b).sum())
        report['fallback_flips'] = int((left['output']['q0_is_fallback'] != right['output']['q0_is_fallback']).sum())
        valid_a, valid_b = left['output']['candidates']['valid'], right['output']['candidates']['valid']
    elif decision_a is not None:
        report.update(measurement_choice_flips=int((decision_a['measurement_index'] != decision_b['measurement_index']).sum()),
            memory_eligibility_flips=int((decision_a['memory_eligible'] != decision_b['memory_eligible']).sum()))
        a, b = left['output']['verification'], right['output']['verification']
        report['measurement_threshold_flips'] = int(((a['measurement_score'] >= .5) != (b['measurement_score'] >= .5)).sum())
        report['max_abs_measurement_score'] = finite_delta(a['measurement_score'], b['measurement_score'])
        report['verification_support_flips'] = int((a['support_mask'] != b['support_mask']).sum())
        valid_a, valid_b = left['output']['candidates']['valid'], right['output']['candidates']['valid']
    else:
        a, b = left['output']['decoder']['quality_logits'], right['output']['decoder']['quality_logits']
        valid_a = left['output']['decoder']['hypothesis_valid']
        valid_b = right['output']['decoder']['hypothesis_valid']
        report['max_abs_quality_logit'] = finite_delta(a, b, valid_a & valid_b)
        report['quality_threshold_flips'] = int(((a.sigmoid() >= .5) != (b.sigmoid() >= .5)).sum())
    report['candidate_validity_flips'] = int((valid_a != valid_b).sum())
    for field in ('point_indices', 'point_valid', 'members'):
        a, b = left['output']['evidence'][field], right['output']['evidence'][field]
        report['evidence_' + field + '_changes'] = int((a != b).sum())
    report['rng_equal'] = equal(left['rng'], right['rng'])
    report['host_step_counts_equal'] = left['host_steps'] == right['host_steps']
    if not report['host_step_counts_equal']:
        raise AssertionError('mixed precision must not skip or add a host/Adam transaction')
    step_a = {key: values.get('step') for key, values in left['optimizer']['state'].items()}
    step_b = {key: values.get('step') for key, values in right['optimizer']['state'].items()}
    report['per_parameter_adam_steps_equal'] = equal(step_a, step_b)
    report['memory_state_equal'] = equal(
        {key: value.get('memory') for key, value in left['state'].items()},
        {key: value.get('memory') for key, value in right['state'].items()})
    # 实际递归状态的变化另列，不把门限翻转当作已经证明掉分。
    try:
        assert_identical(left['state'], right['state'])
        report['recursive_state_equal'] = True
    except AssertionError:
        report['recursive_state_equal'] = False
    return report


def run_equivalence(config, device, *, steps=8, candidate_profile='optimized_fp32',
                    sources=None, weights=None, epoch_indices=(0, 10), baseline_profile=None):
    """sources 仅供合成合同测试；CLI 始终读取原数据集，不减少正式点数。"""
    import torch
    from utils.feature_precision import feature_autocast
    baseline_profile = paired_baseline(config, candidate_profile, baseline_profile)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if candidate_profile == 'optimized_bf16':
        with feature_autocast(torch.zeros(1, device=device), enabled=True):
            pass
    if candidate_profile not in ('optimized_fp32', 'optimized_bf16'):
        raise ValueError('candidate profile must be optimized_fp32 or optimized_bf16')
    if not epoch_indices or any(type(index) is not int or index < 0 for index in epoch_indices):
        raise ValueError('epoch_indices must contain nonnegative integers')
    if len(epoch_indices) > 1:
        phases = [run_equivalence(config, device, steps=steps, candidate_profile=candidate_profile,
                                 sources=sources, weights=weights, epoch_indices=(index,),
                                 baseline_profile=baseline_profile)
                  for index in epoch_indices]
        return dict(status=phases[-1]['status'], candidate_profile=candidate_profile,
            baseline_profile=baseline_profile, phases=phases,
            training_steps=sum(phase['training_steps'] for phase in phases),
            evaluation_steps=sum(phase['evaluation_steps'] for phase in phases),
            bitwise_equal=all(phase['bitwise_equal'] for phase in phases),
            evaluation=phases[-1]['evaluation'], saved_training_weights=False,
            actual_lightning_logger_checked=False, scheduler_checked=False,
            scope='paired_initial_and_mature_curriculum_not_final_score_validation')
    epoch_index = epoch_indices[0]
    api = module_set(config)
    loader_config = api['config'].normalize_config(dict(config, execution_profile=baseline_profile, log_dir=None))
    seed_all(config.seed)
    loaders = api['data'].build_loaders(loader_config, roles=('train', 'val'), sources=sources)
    loader = loaders['train']
    loader.batch_sampler.set_epoch(epoch_index)
    loader.generator.manual_seed(api['data'].stable_seed(config.seed, 'train', epoch_index))
    models = _make_pair(config, device, candidate_profile, weights, epoch_index, baseline_profile)
    comparison, processed = [], 0
    coverage = dict(branches={}, max_depth=0, nonempty_postcommit_state_steps=0, tail_steps=0, drain_steps=0)
    iterator = iter(loader)
    for step in range(steps):
        try:
            rows = next(iterator)
        except StopIteration:
            break
        rng = api['runtime'].capture_rng_state()
        legacy = _step(*models[0], rows, api)
        api['runtime'].restore_rng_state(rng)
        optimized = _step(*models[1], rows, api)
        if candidate_profile == 'optimized_fp32':
            assert_identical(legacy, optimized, 'train_step_' + str(step))
        else:
            comparison.append(_numerical_delta(legacy, optimized))
        for row in rows:
            request = row['request']
            branch = str(request.branch)
            coverage['branches'][branch] = coverage['branches'].get(branch, 0) + 1
            coverage['max_depth'] = max(coverage['max_depth'], request.frame - request.window_start)
        coverage['nonempty_postcommit_state_steps'] += int(bool(legacy['state']))
        coverage['tail_steps'] += int(len(rows) < config.batch_size)
        coverage['drain_steps'] += int(any(getattr(row['request'], 'drain', False) for row in rows))
        processed += 1
        del legacy, optimized
    if not processed:
        raise ValueError('dataset produced no training batch')
    # 用两个已更新模型独立递推，检查原 collector 的候选、逐帧指标及状态。
    for model, _ in models:
        model.eval()
        model.evaluation_builder.reset()
        model.evaluation = api['runtime'].TrackingEvaluation(model.config)
        model.evaluation.add_singleton_tracks(loaders['val'].dataset)
    eval_steps = 0
    evaluation_differences = []
    with torch.no_grad():
        for index, rows in enumerate(loaders['val']):
            if index >= steps:
                break
            rng = api['runtime'].capture_rng_state()
            snapshots = []
            for model, optimizer in models:
                api['runtime'].restore_rng_state(rng)
                captured = {}
                def capture(module, inputs, output):
                    captured['output'] = freeze(output)
                    if hasattr(getattr(output, 'verification', None), 'measurement_score'):
                        captured['output']['verification']['measurement_score'] = freeze(output.verification.measurement_score)
                handle = model.tracker.register_forward_hook(capture)
                try:
                    model._evaluation_step(rows)
                finally:
                    handle.remove()
                captured.update(state=freeze(model.evaluation_builder.states), losses={},
                    rng=freeze(api['runtime'].capture_rng_state()), optimizer=freeze(optimizer.state_dict()),
                    host_steps=model._epoch_steps)
                snapshots.append(captured)
            if candidate_profile == 'optimized_fp32':
                assert_identical(snapshots[0], snapshots[1], 'evaluation_output_' + str(index))
                assert_identical(freeze(models[0][0].evaluation.rows), freeze(models[1][0].evaluation.rows),
                                 'evaluation_rows_' + str(index))
                assert_identical(freeze(models[0][0].evaluation_builder.states),
                                 freeze(models[1][0].evaluation_builder.states), 'evaluation_state')
            else:
                delta = _numerical_delta(*snapshots)
                delta['endpoints_by_identity'] = [dict(tracklet=row['tracklet_key'], frame=row['request'].frame)
                                                   for row in rows]
                evaluation_differences.append(delta)
            eval_steps += 1
    summaries = [model.evaluation.summary() for model, _ in models]
    if candidate_profile == 'optimized_fp32':
        assert_identical(freeze(summaries[0]), freeze(summaries[1]), 'evaluation_metrics')
    result = dict(status='passed', candidate_profile=candidate_profile,
        baseline_profile=baseline_profile, training_steps=processed,
        epoch_index=epoch_index, training_coverage=coverage,
        requested_steps=steps, evaluation_steps=eval_steps, bitwise_equal=candidate_profile == 'optimized_fp32',
        evaluation={baseline_profile: summaries[0], 'candidate': summaries[1]}, bf16_differences=comparison,
        bf16_evaluation_differences=evaluation_differences,
        scope='paired_short_closed_loop_not_final_score_validation', saved_training_weights=False,
        actual_lightning_logger_checked=False, scheduler_checked=False,
        logger_validation='real Lightning CSV/TensorBoard validation belongs to benchmark mode')
    if candidate_profile == 'optimized_bf16':
        result['status'] = 'engineering_passed_score_unverified'
    for model, _ in models:
        model.train_builder.reset()
        model.evaluation_builder.reset()
    return result


def main(argv=None):
    args = parse_args(argv)
    from tools.check_v35_batch import report_destination
    from models.ct_v31.entry import configure_numerics, source_identity
    configure_numerics()
    import torch
    destination = report_destination(args.output, version='execution')
    if destination.parent.exists() and any(destination.parent.iterdir()):
        raise FileExistsError('execution report requires a new empty directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='ct_seqtrack.execution_check.v1', status='running', source=source_identity(),
        tools_sha256={name: sha256_file(ROOT / name) for name in
            ('tools/check_execution_performance.py', 'tools/execution_benchmark.py', 'tools/check_v35_batch.py')
            if (ROOT / name).exists()}, environment=dict(python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, device=args.device, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            torch_num_threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
        arguments={key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}, saved_training_weights=False)
    for package in ('numpy', 'pytorch-lightning'):
        try:
            report['environment'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['environment'][package] = None
    if args.device == 'cuda' and torch.cuda.is_available():
        report['environment']['gpu_name'] = torch.cuda.get_device_name()
    try:
        config = load_engineering_config(args.cfg, args.path)
        profile = 'optimized_bf16' if args.mode == 'bf16' else args.candidate_profile
        config = module_set(config)['config'].normalize_config(dict(config, execution_profile=profile))
        report['config'] = dict(config)
        state = None
        if args.weights:
            state, report['historical_weights'] = verified_engineering_weights(config, args.weights, args.run_manifest)
        if args.mode in ('check', 'all', 'bf16'):
            report['correctness'] = run_equivalence(config, args.device, steps=args.check_steps,
                                                  candidate_profile=profile, weights=state,
                                                  epoch_indices=tuple(args.check_epochs),
                                                  baseline_profile=args.baseline_profile)
        if args.mode in ('benchmark', 'all'):
            from tools.execution_benchmark import run_benchmark
            report['performance'] = run_benchmark(config, args.device, destination.parent / 'benchmark',
                warmup=args.warmup, steps=args.steps, repeats=args.repeats, trace_steps=args.trace_steps,
                baseline_profile=args.baseline_profile, inference=args.inference)
        report['status'] = ('engineering_passed_score_unverified' if profile == 'optimized_bf16' else 'passed')
        return_code = 0
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        return_code = 1
    content, status = encode_report(report)
    report['status'] = status
    if status == 'failed':
        return_code = 1
    destination.write_text(content, encoding='utf-8')
    print(json.dumps(dict(status=report['status'], report=str(destination)), ensure_ascii=False))
    return return_code


if __name__ == '__main__':
    raise SystemExit(main())
