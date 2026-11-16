"""V41 完整 epoch 保存与评估隔离；固定适配 Lightning 2.0.2。"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from pytorch_lightning.callbacks import Checkpoint

from .data import stable_seed
from .runtime import capture_rng_state, restore_rng_state


EVALUATION_RNG_POLICY = 'isolated_split_seed_v2'


def epoch_report(module, epoch):
    """只持有 CPU 标量/摘要；不将预测张量挂在 callback 上。"""
    if not hasattr(module, '_epoch_reports'):
        module._epoch_reports = {}
    return module._epoch_reports.setdefault(int(epoch), dict(
        epoch=int(epoch), evaluation={}, timing={}, checkpoint={}, losses={}))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


@contextmanager
def isolated_evaluation_rng(config, role, loader=None, module=None):
    """涵盖 iterator/worker 初始化；退出及异常均还原训练 RNG 和每层模式。"""
    if role not in ('test', 'dev'):
        raise ValueError('evaluation RNG role must be test or dev')
    state = capture_rng_state()
    modes = [(child, child.training) for child in module.modules()] if module is not None else []
    generator = getattr(loader, 'generator', None)
    generator_state = generator.get_state() if generator is not None else None
    try:
        seed = stable_seed(int(config.seed), 'evaluation', role)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if generator is not None:
            generator.manual_seed(stable_seed(int(config.seed), 'val' if role == 'dev' else 'test'))
        yield
    finally:
        restore_rng_state(state)
        if generator is not None:
            generator.set_state(generator_state)
        # 直接还原每层原值，避免 train() 递归覆盖原本独立的 BN 模式。
        for child, training in modes:
            child.training = training


def install_evaluation_rng(trainer, module, config):
    """包住整个 run，而非在已经创建 iterator 的 validation_start 才进入。"""
    for loop, standalone in ((trainer.fit_loop.epoch_loop.val_loop, False),
                             (trainer.test_loop, True)):
        if getattr(loop, '_ct_v41_rng_installed', False):
            continue
        original = loop.run

        def wrap(run, is_test):
            @wraps(run)
            def isolated_run(*args, **kwargs):
                role = config.eval_split
                started = time.perf_counter()
                # factory 可能尚未被 setup_data 请求；先局部隔离构造，再由外层
                # scope 从固定种子开始迭代。host 的 val_dataloader 同样局部隔离。
                with isolated_evaluation_rng(config, role, module=module):
                    loader = module._loader('val' if role == 'dev' else 'test')
                try:
                    with isolated_evaluation_rng(config, role, loader=loader, module=module):
                        return run(*args, **kwargs)
                finally:
                    if not trainer.sanity_checking:
                        epoch = ((config.eval_checkpoint_epoch or module._completed_epoch) if is_test
                                 else int(trainer.current_epoch) + 1)
                        report = epoch_report(module, epoch)
                        timing = report['timing']
                        timing['eval_seconds'] = timing.get('eval_seconds', 0.) + time.perf_counter() - started
            return isolated_run

        loop.run = wrap(original, standalone)
        loop._ct_v41_rng_installed = True


class EpochCheckpoint(Checkpoint):
    """仅在主辅双流 epoch 完整后保存；同一 epoch 只有一份完整权重。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.best_epoch = None
        self.best_precision = None
        self.history = {}
        self.paths = {}
        self.completed_epoch = 0

    @property
    def state_key(self):
        return 'ct_seqtrack.v41.EpochCheckpoint.v1'

    def state_dict(self):
        return deepcopy(dict(best_epoch=self.best_epoch, best_precision=self.best_precision,
            history=self.history, paths=self.paths, completed_epoch=self.completed_epoch,
            split=self.config.eval_split, selection=self.config.report_checkpoint))

    def load_state_dict(self, state_dict):
        if (state_dict['split'] != self.config.eval_split
                or state_dict['selection'] != self.config.report_checkpoint):
            raise ValueError('epoch checkpoint evaluation/selection policy mismatch')
        self.best_epoch = state_dict['best_epoch']
        self.best_precision = state_dict['best_precision']
        self.history = {int(k): deepcopy(v) for k, v in state_dict['history'].items()}
        self.paths = {int(k): str(v) for k, v in state_dict['paths'].items()}
        self.completed_epoch = int(state_dict['completed_epoch'])

    def _result(self, epoch):
        if epoch is None:
            return None
        return dict(epoch=epoch, checkpoint=self.paths.get(epoch),
                    evaluation=deepcopy(self.history.get(epoch)))

    def results(self):
        last = self._result(self.completed_epoch) if self.completed_epoch else None
        best = self._result(self.best_epoch)
        available = sorted(self.history)[-int(self.config.report_last_k):]
        return dict(schema='ct_seqtrack.v41.epoch_results.v1', split=self.config.eval_split,
            selection=self.config.report_checkpoint,
            selected_on_test=self.config.eval_split == 'test' and self.config.report_checkpoint == 'best_precision',
            best=best, last=last, main=best if self.config.report_checkpoint == 'best_precision' else last,
            lateK=dict(k=int(self.config.report_last_k), available_epochs=available,
                count=len(available), complete=len(available) == int(self.config.report_last_k),
                definition='last_k_evaluated_epochs', rows=[self._result(e) for e in available],
                success=(sum(self.history[e]['success'] for e in available) / len(available)) if available else None,
                precision=(sum(self.history[e]['precision'] for e in available) / len(available)) if available else None),
            evaluated_epochs=sorted(self.history), checkpoint_paths={str(k): v for k, v in sorted(self.paths.items())})

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch) + 1
        if (not pl_module._epoch_complete or not pl_module._auxiliary_complete
                or pl_module._completed_epoch != epoch
                or pl_module._pending_train is not None or pl_module._pending_auxiliary is not None
                or pl_module.train_builder.states or pl_module.auxiliary_builder.states):
            raise RuntimeError('cannot save an incomplete main/auxiliary epoch')
        report = epoch_report(pl_module, epoch)
        timing = report['timing']
        started = report.pop('_started_at', None)
        if started is not None:
            timing['train_seconds'] = max(0., time.perf_counter() - started - timing.get('eval_seconds', 0.))
        timing.setdefault('eval_seconds', 0.)
        summary = report['evaluation'].get(self.config.eval_split)
        due = epoch % int(self.config.check_val_every_n_epoch) == 0 or epoch == int(trainer.max_epochs)
        if due and summary is None:
            raise RuntimeError('completed epoch is missing its scheduled evaluation')
        improved = False
        previous_state = self.state_dict()
        if summary is not None:
            if (summary.get('epoch') != epoch or summary.get('checkpoint_epoch') != epoch
                    or summary.get('split') != self.config.eval_split
                    or summary.get('complete_coverage') is not True):
                raise RuntimeError('checkpoint selection requires this epoch and complete split coverage')
            if not all(math.isfinite(float(summary[key])) for key in ('success', 'precision')):
                raise ValueError('checkpoint selection requires finite Success and Precision')
            score = float(summary['precision'])
            self.history[epoch] = deepcopy(summary)
            improved = self.best_precision is None or score > self.best_precision
            if improved:  # 相同 Precision 保留更早轮，不另用 Success 隐式破同分。
                self.best_epoch, self.best_precision = epoch, score
        self.completed_epoch = epoch
        periodic = epoch % int(self.config.checkpoint_every_n_epochs) == 0
        save = periodic or epoch == int(trainer.max_epochs) or improved
        root = Path(trainer.default_root_dir)
        path = root / 'formal_checkpoints' / f'epoch={epoch:03d}.ckpt'
        if save:
            self.paths[epoch] = str(path)
        report['checkpoint'] = dict(path=str(path) if save else None, saved=False,
            is_best=improved, best_epoch=self.best_epoch, best_path=self.paths.get(self.best_epoch))
        save_started = time.perf_counter()
        try:
            if save:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError('checkpoint already exists; resume from the latest completed epoch: ' + str(path))
                temporary = path.with_name('.' + path.name + '.tmp')
                trainer.save_checkpoint(str(temporary), weights_only=False)
                os.replace(temporary, path)
                report['checkpoint']['saved'] = True
        except BaseException:
            self.load_state_dict(previous_state)
            raise
        finally:
            timing['save_seconds'] = time.perf_counter() - save_started if save else 0.
        atomic_json(root / 'results.json', self.results())
