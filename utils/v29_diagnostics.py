"""v29 运行诊断抽样；训练 loss、端点和递归动作从不抽样。"""

import torch
from collections.abc import Mapping
from contextlib import contextmanager

from utils.sampling_utils import stable_uint32_seed
from utils.v29_performance import diagnostics_sampled, scalar_items_to_python, _get


def assert_h3_diagnostic_contract(config):
    if (not bool(_get(config, 'ct_enable_v29', False))
            or _get(config, 'ct_b3_target_contract', None) != 'instantaneous_sp_gain_v1'):
        raise RuntimeError('H3 sampling requires v29 H1-only utility supervision')


class H1OnlyLabels(Mapping):
    """v29 perf 的 B3 loss 不能读取 H3 载荷；未来误接会在实际事务中报错。"""
    def __init__(self, source):
        self.source = source

    def __getitem__(self, key):
        if str(key).startswith(('ct_h3_', 'ct_shadow_')):
            raise RuntimeError('diagnostic H3 cannot enter B3 loss')
        return self.source[key]

    def __iter__(self):
        return (key for key in self.source if not key.startswith(('ct_h3_', 'ct_shadow_')))

    def __len__(self):
        return sum(1 for _ in self)


@contextmanager
def assert_h3_state_unchanged(host, output):
    """仅实际抽中事件断言 policy 输出及主递归状态未被影子通路修改。"""
    from utils.v29_profiling import snapshot, first_difference
    keys = ('ct_final_box', 'aux_estimation_boxes', 'observation_aux_estimation_boxes',
            'ct_router_applied_gate', 'ct_router_bounded_residual_xy', 'ct_b3_action_score',
            'ct_policy_candidate_valid', 'ct_b2_available', 'ct_v29_behavior_final_boxes')
    def state():
        return snapshot(dict(policy={key: output[key] for key in keys if key in output},
            contexts=[context['state'] for context in getattr(host, '_ct_online_batch_context', [])],
            recursive=getattr(host, '_ct_recursive_states', {})))
    before = state()
    yield
    difference = first_difference(before, state())
    if difference:
        raise RuntimeError('H3 diagnostic mutated accepted policy or recursive state: ' + difference)


def sample_training_diagnostics(host, *, scalar=False):
    """仅前五步和固定间隔；尾批由 epoch-end 的 pending 队列统一处理。"""
    if not getattr(host, 'training', True) or not diagnostics_sampled(host.config):
        return True
    batch_index = getattr(host, '_ct_perf_batch_index', None)
    if batch_index is None:
        return True
    step = int(getattr(host, 'global_step', 0)) + 1
    key = 'ct_scalar_log_every_n_steps' if scalar else 'ct_diagnostic_every_n_steps'
    interval = int(getattr(host.config, key, 50 if scalar else 100))
    if interval < 1:
        raise ValueError('diagnostic interval must be positive')
    return step <= 5 or step % interval == 0


def _detached(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_detached(item) for item in value)
    return value


def diagnostic_call(host, key, callback, *args, scalar=False):
    """每类只保留最后一次事务的 detached 诊断载荷，不保留计算图或整段历史。"""
    if not host.training or not diagnostics_sampled(host.config):
        return callback(*args)
    pending = getattr(host, '_ct_perf_pending_diagnostics', None)
    if pending is None:
        pending = host._ct_perf_pending_diagnostics = {}
    entry = dict(callback=callback, args=_detached(args),
                 step=int(host.global_step), written=False)
    pending[key] = entry
    if sample_training_diagnostics(host, scalar=scalar):
        result = callback(*entry['args'])
        entry['written'] = True
        return result
    return None


def flush_pending_diagnostics(host):
    """只写未写出的最后事务；on_train_epoch_end 调用，不推测尾步、不重复写。"""
    pending = getattr(host, '_ct_perf_pending_diagnostics', {})
    for entry in pending.values():
        if not entry['written']:
            entry['callback'](*entry['args'])
            entry['written'] = True
    host._ct_perf_pending_diagnostics = {}


def keep_h3_event(host, raw):
    """原定事件的稳定子集；不使用 Python/NumPy/Torch 的可变 RNG。"""
    if not getattr(host, 'training', True) or not diagnostics_sampled(host.config):
        return True
    ratio = float(getattr(host.config, 'ct_h3_diagnostic_keep_ratio', .1))
    if not 0. <= ratio <= 1.:
        raise ValueError('H3 diagnostic keep ratio must be in [0,1]')
    value = stable_uint32_seed(int(getattr(host.config, 'seed', 42)),
        'ct_v29_h3_sampling_v1', int(raw.get('online_epoch', 0)),
        str(raw['tracklet_key']), int(raw['this_frame_id']),
        str(raw.get('shadow_event', 'h3_single_intervention_b0_fallback')))
    return value < ratio * 2**32


def accumulate_core_losses(host, losses, rows):
    """所有事务的核心 loss 全量累计；只保留 detached 的加权和。"""
    if not host.training or not diagnostics_sampled(host.config):
        return
    names = tuple(key for key in losses if key.startswith('loss'))
    if not names:
        return
    values = torch.stack([losses[key].detach().reshape(()) for key in names])
    population = 'mechanism' if getattr(host, '_ct_mechanism_transaction', False) else 'observation'
    store = getattr(host, '_ct_perf_epoch_losses', None)
    if store is None:
        store = host._ct_perf_epoch_losses = {}
    key = (population, names)
    if key not in store:
        store[key] = [values * int(rows), int(rows), 1]
    else:
        store[key][0].add_(values * int(rows))
        store[key][1] += int(rows)
        store[key][2] += 1


def flush_core_losses(host):
    store = getattr(host, '_ct_perf_epoch_losses', {})
    values = {}
    totals_by_metric = {}
    populations = {}
    for (population, names), (total, rows, transactions) in store.items():
        for index, name in enumerate(names):
            key = population + '/' + name
            if key not in totals_by_metric:
                totals_by_metric[key] = [total[index], rows]
            else:
                totals_by_metric[key][0] = totals_by_metric[key][0] + total[index]
                totals_by_metric[key][1] += rows
        counts = populations.setdefault(population, [0, 0])
        counts[0] += rows
        counts[1] += transactions
    values.update({key: total / max(rows, 1)
                   for key, (total, rows) in totals_by_metric.items()})
    for population, (rows, transactions) in populations.items():
        values[population + '/rows'] = rows
        values[population + '/transactions'] = transactions
    converted = scalar_items_to_python(values)
    if values:
        host.logger.experiment.add_scalars('ct_epoch_core_loss',
            converted, global_step=host.global_step)
    flush_pending_diagnostics(host)
    counts = getattr(host, '_ct_perf_h3_counts', {})
    if counts:
        metrics = dict(counts)
        valid_events = int(counts.get('valid_events', 0))
        if valid_events:
            metrics['valid_success_gain_mean'] = counts['success_gain_sum'] / valid_events
            metrics['valid_precision_gain_mean'] = counts['precision_gain_sum'] / valid_events
        host.logger.experiment.add_scalars('ct_epoch_h3_sampling', metrics,
                                           global_step=host.global_step)
    host._ct_perf_last_epoch_summary = dict(
        schema='ct_seqtrack.diagnostic_sampling.v1',
        epoch=int(getattr(host, 'current_epoch', 0)) + 1,
        core_loss_population='all_training_transactions', core_losses=converted,
        h3_event_counts=dict(counts),
        h3_keep_ratio=float(getattr(host.config, 'ct_h3_diagnostic_keep_ratio', .1)),
        metric_population='sampled_first5_periodic_and_epoch_end_flush',
        metric_interval=int(getattr(host.config, 'ct_diagnostic_every_n_steps', 100)),
        binary_rows={name: sum(len(entry[0]) for entry in entries)
                     for name, entries in getattr(host, '_ct_epoch_binary_rows', {}).items()
                     if name in ('presence', 'help', 'harm')})
    host._ct_perf_epoch_losses = {}
    host._ct_perf_h3_counts = {}


@torch.no_grad()
def relation_rank_metrics(logits, prepool_labels, prepool_valid, reference):
    """原 AP/AUROC/ECE 算式；整数 cumsum 路径保持。"""
    device, dtype = reference.device, reference.dtype
    relation_probability = torch.sigmoid(
        logits)
    relation_select = prepool_valid > 0
    relation_scores_flat = relation_probability[relation_select]
    relation_targets_flat = prepool_labels[relation_select] > 0.5
    relation_auroc = reference.new_tensor(0.5)
    relation_auprc = reference.new_zeros(())
    relation_ece = reference.new_zeros(())
    positive_scores = relation_scores_flat[relation_targets_flat]
    negative_scores = relation_scores_flat[~relation_targets_flat]
    if positive_scores.numel() and negative_scores.numel():
        ascending = torch.argsort(relation_scores_flat, stable=True)
        ranks = torch.empty_like(ascending, dtype=dtype)
        ranks[ascending] = torch.arange(
            1, ascending.numel() + 1, device=device, dtype=dtype)
        positive_count = positive_scores.numel()
        negative_count = negative_scores.numel()
        relation_auroc = (
            ranks[relation_targets_flat].sum()
            - 0.5 * positive_count * (positive_count + 1)) / float(
                positive_count * negative_count)
    if positive_scores.numel():
        order = torch.argsort(relation_scores_flat, descending=True)
        ordered_target = relation_targets_flat[order].to(dtype)
        # 二值计数用整数累计，避开 CUDA 浮点 cumsum 的确定性限制。
        precision_at_k = torch.cumsum(
            ordered_target, dim=0, dtype=torch.int64).to(dtype) / torch.arange(
                1, ordered_target.numel() + 1,
                device=device, dtype=dtype)
        relation_auprc = (
            precision_at_k * ordered_target).sum() / torch.clamp(
                ordered_target.sum(), min=1.0)
    if relation_scores_flat.numel():
        ece_terms = []
        for bin_index in range(10):
            lower = bin_index / 10.0
            upper = (bin_index + 1) / 10.0
            in_bin = (
                (relation_scores_flat >= lower)
                & (relation_scores_flat < upper
                   if bin_index < 9 else relation_scores_flat <= upper))
            if bool(in_bin.any()):
                ece_terms.append(
                    in_bin.to(dtype).mean()
                    * torch.abs(
                        relation_scores_flat[in_bin].mean()
                        - relation_targets_flat[in_bin].to(dtype).mean()))
        if ece_terms:
            relation_ece = torch.stack(ece_terms).sum()
    
    relation_metrics = dict(ct_relation_auroc=relation_auroc, ct_relation_ap=relation_auprc,
                            ct_relation_auprc=relation_auprc, ct_relation_ece=relation_ece)
    return relation_metrics


def write_relation_diagnostics(host, logits, labels, valid, reference, step):
    metrics = relation_rank_metrics(logits, labels, valid, reference)
    metrics = {'sampled_' + key: value for key, value in metrics.items()}
    metrics['sampled_relation_points'] = (valid > 0).sum()
    metrics['sampled_relation_rows'] = len(valid)
    host.logger.experiment.add_scalars('ct_relation_sampled',
        scalar_items_to_python(metrics), global_step=step)


def write_training_scalars(host, losses, step, population):
    host.logger.experiment.add_scalars('loss_' + population,
        scalar_items_to_python(losses), global_step=step)


def write_gate_histogram(host, gate, step):
    if hasattr(host.logger.experiment, 'add_histogram'):
        host.logger.experiment.add_histogram('ct/router_probability_histogram',
            gate, global_step=step)
