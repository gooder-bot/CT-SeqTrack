"""v35真实CPU网络的完整epoch恢复；dropout、Adam、课程与accepted同时核对。"""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from models.ct_v31.config import load_config, normalize_config
from models.ct_v31.data import build_loaders
from models.ct_v31.runtime import capture_rng_state
from tests.test_ct_v31_runtime import TinySource
from tests.test_ct_v34_resume import CPUContractHost, _epoch, _assert_identical


def _build_v35(seed):
    cfg = normalize_config(dict(
        load_config('cfgs/ct_seqtrack/35_b0_w_quarter_lr_mini.yaml'),
        seed=seed, ct_engineering_check=True, batch_size=4, point_sample_size=8,
        workers=0, epoch=2, v31_curriculum_epochs=1, v32_reserve_windows=0,
        lr_milestones=[1], log_dir=None))
    loaders = build_loaders(cfg, roles=('train',),
                            sources={'train': TinySource(lengths=(6, 5))})
    host = CPUContractHost(cfg, loaders=loaders).train()
    configured = host.configure_optimizers()
    return host, configured['optimizer'], configured['lr_scheduler']['scheduler']


@pytest.mark.parametrize('seed', [42, 52])
def test_real_v35_epoch_resume_preserves_model_optimizer_rng_and_accepted(seed):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        host, optimizer, scheduler = _build_v35(seed)
        _epoch(host, optimizer, scheduler, 0)
        checkpoint = dict(state_dict=deepcopy(host.state_dict()),
                          optimizer_states=[deepcopy(optimizer.state_dict())],
                          lr_schedulers=[deepcopy(scheduler.state_dict())])
        host.on_save_checkpoint(checkpoint)
        payload = checkpoint['ct_v35_runtime']
        assert payload['epoch_complete'] and payload['completed_epoch'] == 1
        assert payload['sampler']['short_window'] == 4
        assert payload['sampler']['initial_seed_policy'] == 'initial_exact_other_perturbed_v1'
        assert optimizer.param_groups[0]['lr'] == pytest.approx(2.5e-6)
        uninterrupted = _epoch(host, optimizer, scheduler, 1)
        expected = dict(model=deepcopy(host.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
                        scheduler=deepcopy(scheduler.state_dict()), rng=capture_rng_state())
        resumed, resumed_optimizer, resumed_scheduler = _build_v35(seed)
        resumed.load_state_dict(checkpoint['state_dict'], strict=True)
        resumed_optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        resumed_scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
        resumed.on_load_checkpoint(checkpoint)
        restored = _epoch(resumed, resumed_optimizer, resumed_scheduler, 1)
        _assert_identical(uninterrupted, restored)
        actual = dict(model=resumed.state_dict(), optimizer=resumed_optimizer.state_dict(),
                      scheduler=resumed_scheduler.state_dict(), rng=capture_rng_state())
        _assert_identical(expected, actual)
    finally:
        torch.set_num_threads(previous)
