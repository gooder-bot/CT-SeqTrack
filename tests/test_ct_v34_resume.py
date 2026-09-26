"""真实 v34 网络的 CPU host 合同：完整小 epoch 恢复，不代替 Lightning Trainer。

手动驱动宿主 hooks、Adam 和 epoch scheduler，保留训练态 dropout、
BatchBuilder 的 accepted 提交与 pending RNG 恢复；工程预算不冒充正式训练。
"""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from models.ct_v31.config import load_config, normalize_config
from models.ct_v31.data import build_loaders
from models.ct_v31.runtime import capture_rng_state
from models.ctseqtrackv31 import CTSEQTRACKV31
from tests.test_ct_v31_runtime import TinySource


class CPUContractHost(CTSEQTRACKV31):
    """只替代 Trainer 提供的 epoch/log，不替代训练或状态管理方法。"""

    _contract_epoch = 0

    @property
    def current_epoch(self):
        return self._contract_epoch

    def log(self, *args, **kwargs):
        pass


def _build(short):
    name = 'context' if short == 3 else 'context_w4'
    config = normalize_config(dict(
        load_config(f'cfgs/ct_seqtrack/34_b0_{name}_mini.yaml'),
        ct_engineering_check=True, batch_size=4, point_sample_size=8, workers=0,
        epoch=2, v31_curriculum_epochs=1, v32_reserve_windows=0,
        # 工程小 epoch 恰好跨过一次衰减，检查 scheduler 状态也实际生效。
        lr_milestones=[1], log_dir=None))
    loaders = build_loaders(config, roles=('train',),
                            sources={'train': TinySource(lengths=(6, 5))})
    host = CPUContractHost(config, loaders=loaders).train()
    configured = host.configure_optimizers()
    return host, configured['optimizer'], configured['lr_scheduler']['scheduler']


def _epoch(host, optimizer, scheduler, epoch):
    host._contract_epoch = epoch
    loader = host._loader('train')
    host.on_train_epoch_start()
    assert host._pending_rng is None and host._resume_sampler is None
    observations = []
    for index, rows in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        loss = host.training_step(rows, index)
        output, _, _ = host._pending_train
        observations.append((loss.detach().clone(), output.accepted_box.detach().clone()))
        loss.backward()
        optimizer.step()
        # 与宿主契约一致，唯一 accepted 事务必须在本批 Adam 之后提交。
        host.on_train_batch_end(loss, rows, index)
    host.on_train_epoch_end()
    scheduler.step()
    assert host._epoch_complete and host._completed_epoch == epoch + 1
    assert host._epoch_rows == 36  # 仅 TinySource 的 (5+4) 个端点 × 四分支。
    assert host._epoch_steps == len(loader)
    assert not host.train_builder.states and host._pending_train is None
    return observations


def _assert_identical(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_identical(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('short', [3, 4])
def test_real_v34_host_epoch_resume_is_bitwise_equal_with_dropout_and_accepted_state(short):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        random.seed(342)
        np.random.seed(342)
        torch.manual_seed(342)
        host, optimizer, scheduler = _build(short)
        assert any(isinstance(layer, torch.nn.Dropout) and layer.training and layer.p > 0
                   for layer in host.tracker.modules())
        _epoch(host, optimizer, scheduler, 0)
        checkpoint = dict(state_dict=deepcopy(host.state_dict()),
                          optimizer_states=[deepcopy(optimizer.state_dict())],
                          lr_schedulers=[deepcopy(scheduler.state_dict())])
        host.on_save_checkpoint(checkpoint)
        payload = checkpoint['ct_v34_runtime']
        assert payload['epoch_complete'] and payload['completed_epoch'] == 1
        assert payload['rows'] == 36 and payload['sampler']['short_window'] == short
        assert optimizer.param_groups[0]['lr'] == pytest.approx(5e-6)

        uninterrupted = _epoch(host, optimizer, scheduler, 1)
        expected = dict(model=deepcopy(host.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
                        scheduler=deepcopy(scheduler.state_dict()), rng=capture_rng_state())
        # 新建网络主动消耗 RNG；只有宿主 pending RNG 恢复能对齐后续 dropout。
        resumed, resumed_optimizer, resumed_scheduler = _build(short)
        resumed.load_state_dict(checkpoint['state_dict'], strict=True)
        resumed_optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        resumed_scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
        resumed.on_load_checkpoint(checkpoint)
        assert resumed._pending_rng is not None and resumed._resume_sampler is not None
        restored = _epoch(resumed, resumed_optimizer, resumed_scheduler, 1)
        _assert_identical(uninterrupted, restored)
        actual = dict(model=resumed.state_dict(), optimizer=resumed_optimizer.state_dict(),
                      scheduler=resumed_scheduler.state_dict(), rng=capture_rng_state())
        _assert_identical(expected, actual)
    finally:
        torch.set_num_threads(previous_threads)
