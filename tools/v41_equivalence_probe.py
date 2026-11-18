"""V41 配对测速器复用的短程真实数据一致性检查。"""
from collections import Counter
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


def compare(actual, expected, group, stats):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f'{group}: different keys {actual.keys() ^ expected.keys()}')
        for key in expected:
            compare(actual[key], expected[key], f'{group}/{key}', stats)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f'{group}: different sequence lengths')
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{group}/{index}', stats)
    elif torch.is_tensor(expected):
        floating = expected.is_floating_point()
        try:
            torch.testing.assert_close(actual, expected, rtol=RTOL if floating else 0.,
                                       atol=ATOL if floating else 0., equal_nan=False)
        except (AssertionError, ValueError, TypeError) as error:
            raise AssertionError(f'{group}: {error}') from error
        stats['float_tensors' if floating else 'discrete_tensors'] += 1
    elif isinstance(expected, (float, np.floating)):
        if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL):
            raise AssertionError(f'{group}: {actual} != {expected}')
    elif actual != expected:
        raise AssertionError(f'{group}: {actual} != {expected}')


def closed_loop(builder):
    # 这两个旧字段仅供已退役 acquisition，接受/可信控制字段全部参加比较。
    return {key: {k: v for k, v in vars(state).items() if k not in ('previous_quality', 'previous_innovation')}
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
                if role == 'main' and self.args.reference:
                    losses['loss_total'] = losses['loss_total'] - .05 * losses['loss_acquisition']
                if self.selected(model, model._epoch_steps):
                    common_losses = dict(losses)
                    if role == 'main' and self.args.reference:
                        del common_losses['loss_acquisition']
                    self.observations[role] = snapshot(dict(losses=common_losses, points=batch['points'],
                        point_ids=batch['point_ids'], point_valid=batch['point_valid'],
                        history_boxes=batch['history_boxes'], history_valid=batch['history_valid'],
                        candidates=vars(output.candidates), recovery=vars(output.recovery),
                        selected=output.selected_index, accepted=output.accepted_box,
                        logits=output.verification.logits, commit=vars(output.commit_decision)))
                return losses
            return run

        stack.enter_context(patch.object(tracker, 'compute_losses', with_loss(tracker.compute_losses, 'main')))
        stack.enter_context(patch.object(tracker, 'compute_aux_losses', with_loss(tracker.compute_aux_losses, 'aux')))
        auxiliary_forward = model._forward_auxiliary_raw

        def auxiliary(rows):
            requests = [row['request'] for row in rows]
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
            loss = training_step(rows, batch_idx)
            if self.selected(model, batch_idx):
                self.observations['optimization_loss'] = snapshot(loss)
                self.observations['main_requests'] = [vars(row['request']) for row in rows]
            return loss
        stack.enter_context(patch.object(model, 'training_step', optimization_loss))
        before_step = model.on_before_optimizer_step

        def gradients(optimizer):
            before_step(optimizer)
            if self.selected(model, model._epoch_steps):
                self.observations['common_gradients_before_adam'] = snapshot({
                    name: value.grad for name, value in tracker.named_parameters()
                    if not name.startswith('prior.acquisition_head.')})
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
        observed['state_after_commit'] = snapshot(dict(main=closed_loop(model.train_builder),
                                                        auxiliary=closed_loop(model.auxiliary_builder)))
        observed['committed'] = (int(model._epoch_rows), int(model._auxiliary_rows),
                                 int(model._auxiliary_batches), int(trainer.global_step))
        path = self.args.initial_state.parent / f'equivalence-step-{index}.pt'
        if self.args.reference:
            torch.save(observed, path)
        else:
            expected = torch.load(path, map_location='cpu', weights_only=False)
            try:
                compare(observed, expected, f'step_{index}', self.stats)
            except AssertionError as error:
                failure = dict(status='failed', step_index=index, detail=str(error))
                (self.args.destination / 'equivalence-failure.json').write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
                raise
        self.checked.append(index)
        if index == model._auxiliary_insertions[1]:
            if not self.continuations:
                self.incomplete('the second auxiliary batch supplied no live predecessor')
            self.report = dict(status='reference_recorded' if self.args.reference else 'passed',
                reference_objective='original main total - 0.05 * loss_acquisition, normal auxiliary loss',
                steps_compared=self.checked, first_aux_step=model._auxiliary_insertions[0],
                continued_auxiliary_endpoints=self.continuations,
                training_source_sha256=model._loaders['train'].batch_sampler.source_sha256,
                checked_tensors=dict(self.stats), rtol=RTOL, atol=ATOL)
            raise self.done()
