"""v37 工程事务对照；不改变正式 execution profile，不保存模型权重。"""
import hashlib
import json
from pathlib import Path
import struct
from contextlib import nullcontext


def course_host(host_type, epoch):
    """短窗使用指定课程；Trainer 仍只运行一轮，优化器仍从零开始。"""
    class EngineeringCourseHost(host_type):
        @property
        def current_epoch(self):
            return epoch
    return EngineeringCourseHost


def snapshot_digest(value):
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            data = item.detach().cpu().reshape(-1).clone(memory_format=torch.contiguous_format)
            digest.update(data.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif isinstance(item, (float, np.floating)):
            digest.update(type(item).__name__.encode())
            digest.update(item.tobytes() if isinstance(item, np.floating) else struct.pack('>d', item))
        else:
            digest.update(repr((type(item).__name__, item)).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def bundle_identity(root, *, normalize_newlines=False):
    root = Path(root).resolve()
    paths = [root / 'main.py', root / 'requirement.txt']
    for name in ('models', 'datasets', 'utils', 'tools'):
        paths.extend((root / name).rglob('*.py'))
    files = {path.relative_to(root).as_posix(): hashlib.sha256(
                 path.read_bytes().replace(b'\r\n', b'\n') if normalize_newlines else path.read_bytes()).hexdigest()
             for path in sorted(paths)}
    return dict(sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=files)


def coverage_add(coverage, rows, packet):
    for row in rows:
        request = row['request']
        branch = str(request.branch)
        coverage['branches'][branch] = coverage['branches'].get(branch, 0) + 1
        coverage['max_depth'] = max(coverage['max_depth'], request.frame - request.window_start)
    coverage['nonempty_postcommit_state_steps'] += int(bool(packet['state']))
    coverage['tail_steps'] += int(len(rows) < coverage['batch_size'])
    coverage['drain_steps'] += int(any(getattr(row['request'], 'drain', False) for row in rows))


def new_coverage(batch_size):
    return dict(batch_size=batch_size, branches={}, max_depth=0,
                nonempty_postcommit_state_steps=0, tail_steps=0, drain_steps=0)


def make_model(config, device, epoch):
    from tools.check_execution_performance import module_set, seed_all
    api = module_set(config)
    seed_all(config.seed)
    model = course_host(api['host'], epoch)(config).to(device).train()
    model.on_train_epoch_start()
    return model, model.configure_optimizers()['optimizer']


def train_loader(config, epoch, sources=None):
    from tools.check_execution_performance import module_set, seed_all
    api = module_set(config)
    seed_all(config.seed)
    loader = api['data'].build_loaders(config, roles=('train',), sources=sources)['train']
    loader.batch_sampler.set_epoch(epoch)
    loader.generator.manual_seed(api['data'].stable_seed(config.seed, 'train', epoch))
    return loader


def replay_disabled(config, device, *, steps=8, epochs=(0, 10), sources=None):
    """也由原仓库子进程运行；只依赖原工程工具，证明所有开关关闭的行为。"""
    from tools.check_execution_performance import _step, module_set
    api = module_set(config)
    try:
        from utils.execution_lab import execution_scope
    except ModuleNotFoundError as error:
        if error.name != 'utils.execution_lab':
            raise
        execution_scope = lambda: nullcontext()
    phases = []
    for epoch in epochs:
        loader = train_loader(config, epoch, sources)
        model, optimizer = make_model(config, device, epoch)
        hashes, coverage = [], new_coverage(config.batch_size)
        for index, rows in enumerate(loader):
            if index >= steps:
                break
            with execution_scope():
                packet = _step(model, optimizer, rows, api)
            hashes.append(snapshot_digest(packet))
            coverage_add(coverage, rows, packet)
        model.train_builder.reset()
        model.clear_pending_materialization()
        if len(hashes) != steps:
            raise RuntimeError('disabled replay did not complete requested steps')
        phases.append(dict(epoch_index=epoch, snapshot_sha256=hashes, coverage=coverage))
    return dict(status='passed', phases=phases, saved_training_weights=False)


def numerical_report(left, right):
    """固定原 atol/rtol，仅报告；不可将容差通过当作闭环等价。"""
    import numpy as np
    import torch
    result = dict(atol=1e-5, rtol=1e-4, changed_fields=0, tolerance_failures=0,
                  max_abs=0., representative_changes=[], representative_failures=[])
    def structural_change(path):
        result['changed_fields'] += 1
        result['tolerance_failures'] += 1
        for key in ('representative_changes', 'representative_failures'):
            if len(result[key]) < 12:
                result[key].append(path)
    def visit(a, b, path):
        if path.startswith('transaction/gradients/') and (a is None or b is None):
            if (a is None) != (b is None):
                structural_change(path)
            return
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                if path.startswith(('transaction/state/', 'transaction/optimizer/state/')):
                    structural_change(path)
                    return
                raise AssertionError('tensor interface changed: ' + path)
            if a.is_floating_point():
                finite_a, finite_b = torch.isfinite(a), torch.isfinite(b)
                if not torch.equal(finite_a, finite_b):
                    raise FloatingPointError('finite mask changed: ' + path)
                av, bv = a[finite_a].double(), b[finite_b].double()
                delta = float((av - bv).abs().max()) if av.numel() else 0.
                equal = snapshot_digest(a) == snapshot_digest(b)
                close = (torch.allclose(av, bv, atol=1e-5, rtol=1e-4)
                         and snapshot_digest(a[~finite_a]) == snapshot_digest(b[~finite_b]))
            else:
                equal = close = torch.equal(a, b)
                delta = 0.
        elif isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
                if path.startswith('transaction/state/'):
                    structural_change(path)
                    return
                raise AssertionError('array interface changed: ' + path)
            equal = a.tobytes() == b.tobytes()
            if np.issubdtype(a.dtype, np.floating):
                finite_a, finite_b = np.isfinite(a), np.isfinite(b)
                if not np.array_equal(finite_a, finite_b):
                    raise FloatingPointError('finite mask changed: ' + path)
                av, bv = a[finite_a].astype(np.float64), b[finite_b].astype(np.float64)
                delta = float(np.max(np.abs(av - bv))) if av.size else 0.
                close = (np.allclose(av, bv, atol=1e-5, rtol=1e-4)
                         and a[~finite_a].tobytes() == b[~finite_b].tobytes())
            else:
                close, delta = equal, 0.
        elif isinstance(a, dict):
            if not isinstance(b, dict) or list(a) != list(b):
                if path == 'transaction' or path.startswith(('transaction/output', 'transaction/losses',
                            'transaction/gradients', 'transaction/parameters')):
                    raise AssertionError('fixed dictionary interface changed: ' + path)
                # 闭环状态或Adam的活跃参数可因选择分叉变化，保留该差异。
                equal = close = False
                delta = 0.
            else:
                for key in a:
                    visit(a[key], b[key], path + '/' + str(key))
                return
        elif isinstance(a, (tuple, list)):
            if type(a) is type(b) and len(a) == len(b):
                for index, (av, bv) in enumerate(zip(a, b)):
                    visit(av, bv, path + '/' + str(index))
                return
            equal = close = False
            delta = 0.
        else:
            equal = close = snapshot_digest(a) == snapshot_digest(b)
            delta = 0.
        result['max_abs'] = max(result['max_abs'], delta)
        if not equal:
            result['changed_fields'] += 1
            if len(result['representative_changes']) < 12:
                result['representative_changes'].append(path)
        if not close:
            result['tolerance_failures'] += 1
            if len(result['representative_failures']) < 12:
                result['representative_failures'].append(path)
    visit(left, right, 'transaction')
    return result


def run_pair(config, device, baseline_options=(), candidate_options=(), *, steps=8,
             epochs=(0, 10), sources=None, numerical=False):
    from tools.check_execution_performance import _step, module_set, assert_identical, freeze
    from utils.execution_lab import execution_scope
    api = module_set(config)
    phases = []
    first_divergence = None
    for epoch in epochs:
        loader = train_loader(config, epoch, sources)
        pairs = [make_model(config, device, epoch) for _ in range(2)]
        assert_identical(freeze(pairs[0][0].state_dict()), freeze(pairs[1][0].state_dict()), 'initial_state')
        coverage, differences, processed = new_coverage(config.batch_size), [], 0
        for index, rows in enumerate(loader):
            if index >= steps:
                break
            rng = api['runtime'].capture_rng_state()
            packets = []
            for (model, optimizer), options in zip(pairs, (baseline_options, candidate_options)):
                api['runtime'].restore_rng_state(rng)
                with execution_scope(options, engineering=True):
                    packets.append(_step(model, optimizer, rows, api))
            if numerical:
                delta = numerical_report(*packets)
                prepared_equal = snapshot_digest(packets[0]['prepared']) == snapshot_digest(packets[1]['prepared'])
                if first_divergence is None and delta['changed_fields']:
                    first_divergence = dict(epoch_index=epoch, step=index,
                                            fields=delta['representative_changes'])
                assert_identical(packets[0]['host_steps'], packets[1]['host_steps'], 'transaction_contract/host_steps')
                delta['rng_equal'] = snapshot_digest(packets[0]['rng']) == snapshot_digest(packets[1]['rng'])
                gradient_masks = [{n: g is None for n, g in p['gradients'].items()} for p in packets]
                adam_steps = [{n: state.get('step') for n, state in p['optimizer']['state'].items()} for p in packets]
                delta['gradient_connectivity_equal'] = gradient_masks[0] == gradient_masks[1]
                delta['adam_step_coverage_equal'] = snapshot_digest(adam_steps[0]) == snapshot_digest(adam_steps[1])
                if prepared_equal:
                    assert_identical(packets[0]['rng'], packets[1]['rng'], 'transaction_contract/rng')
                    assert_identical(*gradient_masks, path='gradient_connectivity')
                    assert_identical(*adam_steps, path='Adam_step_coverage')
                # 参数接口、可训练性与梯度连通性不是浮点近似允许改变的内容。
                assert_identical({n: (tuple(p.shape), p.requires_grad) for n, p in pairs[0][0].named_parameters()},
                                 {n: (tuple(p.shape), p.requires_grad) for n, p in pairs[1][0].named_parameters()})
                delta['prepared_equal'] = prepared_equal
                differences.append(delta)
            else:
                assert_identical(*packets, path=f'epoch{epoch}/step{index}')
            coverage_add(coverage, rows, packets[0])
            processed += 1
            del packets
        for model, _ in pairs:
            model.train_builder.reset()
            model.clear_pending_materialization()
        if processed != steps:
            raise RuntimeError('paired check did not complete requested steps')
        phases.append(dict(epoch_index=epoch, steps=processed, coverage=coverage, numerical_differences=differences))
    return dict(status='contract_passed_numerics_changed_score_unverified' if numerical else 'passed',
                bitwise_equal=not numerical, phases=phases, first_divergence=first_divergence,
                saved_training_weights=False, actual_lightning_logger_checked=False,
                formal_score_verified=False)
