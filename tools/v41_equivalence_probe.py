"""V41 配对测速器复用的短程真实数据一致性检查。"""
from collections import Counter
import hashlib
import json
from unittest.mock import patch

import numpy as np
import torch


RTOL, ATOL = 1e-4, 1e-5


def snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    return value


def compare(actual, expected, group, stats, *, exact=False, record=None):
    """历史门禁在容差外中止；轨迹记录器保留各组第一处逐值差异并继续。"""
    def difference(detail, within_tolerance=False):
        if record is not None:
            record(group, detail, within_tolerance)
        elif exact or not within_tolerance:
            raise AssertionError(f'{group}: {detail}')

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return difference('different mapping keys')
        for key in expected:
            compare(actual[key], expected[key], f'{group}/{key}', stats, exact=exact, record=record)
    elif isinstance(expected, (list, tuple)):
        if type(expected) != type(actual) or len(actual) != len(expected):
            return difference('different sequence type or length')
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{group}/{index}', stats, exact=exact, record=record)
    elif torch.is_tensor(expected):
        floating = expected.is_floating_point()
        stats['float_tensors' if floating else 'discrete_tensors'] += 1
        if not torch.is_tensor(actual) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            return difference('different tensor type, shape or dtype')
        if floating and record is not None and (torch.isnan(actual).any() or torch.isnan(expected).any()):
            raise AssertionError(f'{group}: NaN in trajectory probe')
        if not torch.equal(actual, expected):
            within = floating and torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)
            detail = 'tensor values differ'
            if floating and actual.numel():
                # 同号无穷（例如被屏蔽的 logits）不应污染有限误差的数值。
                changed = actual != expected
                delta = (actual[changed] - expected[changed]).abs().max().item()
                detail += f', max_abs={delta:.7g}'
            difference(detail, within)
    elif isinstance(expected, (float, np.floating)):
        if actual != expected:
            if record is not None and (np.isnan(expected) or (isinstance(actual, (float, np.floating)) and np.isnan(actual))):
                raise AssertionError(f'{group}: NaN in trajectory probe')
            within = isinstance(actual, (int, float, np.number)) and np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
            difference(f'{actual} != {expected}', bool(within))
    elif expected is None and actual is not None:
        difference('unexpected value in place of None')
    elif actual != expected:
        difference(f'{actual} != {expected}')


def closed_loop(builder, historical):
    retired = ('previous_quality', 'previous_innovation') if historical else ()
    return {key: {k: v for k, v in vars(state).items() if k not in retired}
            for key, state in builder.states.items()}


class EquivalenceProbe:
    def __init__(self, args, done):
        self.args, self.done = args, done
        self.trace = None
        self.report = {}
        self.observations = {}
        self.checked = []
        self.stats = Counter()
        self.continuations = 0
        self.workload = hashlib.sha256()
        pair = getattr(args, 'revision_pair', 'v3-v4')
        self.historical = pair == 'v2-v3'
        self.exact = pair == 'v4-v4'
        self.compared_groups = []
        self.differences = {}
        self.first_difference = None

    def record_requests(self, role, rows):
        requests = [vars(row['request']) for row in rows]
        self.workload.update(json.dumps([role, requests], separators=(',', ':'), sort_keys=True).encode() + b'\n')
        return requests

    def compare_step(self, index, observed, expected):
        if observed.keys() != expected.keys():
            raise AssertionError(f'step {index}: different observation groups')

        def check(name, value, reference, strict=False):
            strict = strict or self.exact
            if name not in self.compared_groups:
                self.compared_groups.append(name)
            def record(path, detail, within_tolerance):
                item = dict(step_index=index, path=path, detail=detail, within_tolerance=within_tolerance)
                if self.first_difference is None:
                    self.first_difference = item
                group = self.differences.setdefault(name, dict(first=item, differing_leaves=0, outside_tolerance=0))
                group['differing_leaves'] += 1
                group['outside_tolerance'] += int(not within_tolerance)
            compare(value, reference, f'step_{index}/{name}', self.stats,
                    exact=strict, record=None if self.historical or strict else record)

        for role in ('main', 'aux'):
            if role not in observed:
                continue
            request_key = 'main_requests' if role == 'main' else 'auxiliary_requests'
            check(request_key, observed[request_key], expected[request_key], strict=True)
            request_rows = observed[request_key]
            if role == 'aux':
                request_rows = [item['request'] for item in request_rows]
            new_windows = [i for i, r in enumerate(request_rows) if r['frame'] == r['window_start']]
            for field, value in observed[role].items():
                if field not in expected[role]:
                    raise AssertionError(f'{role}: unexpected field {field}')
                # 每个窗口首行来自固定数据和 seed，后续递归裁点可受前面预测影响。
                reference = expected[role][field]
                if (not self.historical and new_windows
                        and field in ('points', 'point_ids', 'point_valid', 'history_boxes', 'history_valid')):
                    check(role + '/fresh_inputs/' + field, value[new_windows], reference[new_windows], strict=True)
                check(role + '/' + field, value, reference)
            if observed[role].keys() != expected[role].keys():
                raise AssertionError(f'{role}: missing observation field')
        for name in ('optimization_loss', 'common_gradients_before_adam', 'BN_after_step', 'state_after_commit', 'committed'):
            check(name, observed[name], expected[name], strict=(name == 'committed'))

    def incomplete(self, message):
        (self.args.destination / 'equivalence-failure.json').write_text(
            json.dumps(dict(status='not_verified', detail=message, steps_compared=self.checked),
                       ensure_ascii=False, indent=2), encoding='utf-8')
        raise RuntimeError(message)

    def selected(self, model, index):
        if len(model._auxiliary_insertions) < 2:
            self.incomplete('closed loop needs two scheduled auxiliary batches, found '
                            + str(len(model._auxiliary_insertions)))
        return index == 0 or index in model._auxiliary_insertions[:2]

    def install(self, stack, model):
        tracker = model.tracker

        def with_loss(fn, role):
            def run(batch, output):
                losses = dict(fn(batch, output))
                if self.historical and role == 'main' and self.args.reference:
                    losses['loss_total'] = losses['loss_total'] - .05 * losses['loss_acquisition']
                if self.selected(model, model._epoch_steps):
                    common_losses = dict(losses)
                    if self.historical and role == 'main' and self.args.reference:
                        del common_losses['loss_acquisition']
                    self.observations[role] = snapshot(dict(points=batch['points'],
                        point_ids=batch['point_ids'], point_valid=batch['point_valid'],
                        history_boxes=batch['history_boxes'], history_valid=batch['history_valid'],
                        losses=common_losses, candidates=vars(output.candidates), recovery=vars(output.recovery),
                        selected=output.selected_index, accepted=output.accepted_box,
                        logits=output.verification.logits, commit=vars(output.commit_decision)))
                    if not self.historical:
                        self.observations[role]['candidate_labels'] = snapshot(output.supervision_targets)
                return losses
            return run

        stack.enter_context(patch.object(tracker, 'compute_losses', with_loss(tracker.compute_losses, 'main')))
        stack.enter_context(patch.object(tracker, 'compute_aux_losses', with_loss(tracker.compute_aux_losses, 'aux')))
        auxiliary_forward = model._forward_auxiliary_raw

        def auxiliary(rows):
            requests = [row['request'] for row in rows]
            self.record_requests('aux', rows)
            # 初始 epoch 的主窗长度为 1。第二个自然辅助批才会真正消费上一端点的状态。
            states = model.auxiliary_builder.states
            reused = [not r.starts_window and r.state_key in states
                      and states[r.state_key].next_frame == r.frame for r in requests]
            self.observations['auxiliary_requests'] = [dict(request=vars(r), reused=live)
                                                       for r, live in zip(requests, reused)]
            self.continuations += sum(reused)
            return auxiliary_forward(rows)
        stack.enter_context(patch.object(model, '_forward_auxiliary_raw', auxiliary))
        training_step = model.training_step

        def optimization_loss(rows, batch_idx):
            main_requests = self.record_requests('main', rows)
            loss = training_step(rows, batch_idx)
            if self.selected(model, batch_idx):
                self.observations['optimization_loss'] = snapshot(loss)
                self.observations['main_requests'] = main_requests
            return loss
        stack.enter_context(patch.object(model, 'training_step', optimization_loss))
        before_step = model.on_before_optimizer_step

        def gradients(optimizer):
            before_step(optimizer)
            if self.selected(model, model._epoch_steps):
                self.observations['common_gradients_before_adam'] = snapshot({
                    name: value.grad for name, value in tracker.named_parameters()
                    if not self.historical or not name.startswith('prior.acquisition_head.')})
        stack.enter_context(patch.object(model, 'on_before_optimizer_step', gradients))

    def after_advance(self, trainer):
        model = trainer.lightning_module
        index = model._epoch_steps - 1
        if not self.selected(model, index):
            return
        observed = self.observations
        self.observations = {}
        expected_groups = {'main', 'common_gradients_before_adam', 'optimization_loss', 'main_requests'}
        if index in model._auxiliary_by_main_step:
            expected_groups.update(('aux', 'auxiliary_requests'))
        if observed.keys() != expected_groups:
            raise RuntimeError(f'missing scheduled equivalence observation at batch {index}')
        observed['BN_after_step'] = snapshot({name: value for name, value in model.tracker.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))})
        if not observed['BN_after_step']:
            raise RuntimeError('batch normalization buffers were not observed')
        observed['state_after_commit'] = snapshot(dict(main=closed_loop(model.train_builder, self.historical),
                                                auxiliary=closed_loop(model.auxiliary_builder, self.historical)))
        observed['committed'] = (int(model._epoch_rows), int(model._auxiliary_rows),
                                 int(model._auxiliary_batches), int(trainer.global_step))
        path = self.args.initial_state.parent / f'equivalence-step-{index}.pt'
        if self.args.reference:
            torch.save(observed, path)
        else:
            expected = torch.load(path, map_location='cpu', weights_only=False)
            try:
                self.compare_step(index, observed, expected)
            except AssertionError as error:
                failure = dict(status='failed', step_index=index, detail=str(error))
                (self.args.destination / 'equivalence-failure.json').write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
                raise
        self.checked.append(index)
        if index == model._auxiliary_insertions[1]:
            if not self.continuations:
                self.incomplete('the second auxiliary batch supplied no live predecessor')
            status = 'observations_exact' if self.exact else ('passed' if self.historical else 'trajectory_compared')
            self.report = dict(status='reference_recorded' if self.args.reference else status,
                reference_objective=('original main total - 0.05 * loss_acquisition, normal auxiliary loss'
                                     if self.historical else 'unmodified main and auxiliary losses'),
                steps_compared=self.checked, first_aux_step=model._auxiliary_insertions[0],
                continued_auxiliary_endpoints=self.continuations,
                training_source_sha256=model._loaders['train'].batch_sampler.source_sha256,
                observed_workload_sha256=self.workload.hexdigest(), first_observed_difference=self.first_difference,
                compared_groups=self.compared_groups, differing_groups=self.differences,
                checked_tensors=dict(self.stats), rtol=0. if self.exact else RTOL, atol=0. if self.exact else ATOL)
            raise self.done()
