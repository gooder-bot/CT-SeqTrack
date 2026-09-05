"""真实 host 事务/Adam 的 CPU 隔离检查；小网络代替不可用的 CUDA backbone。

这些测试不冒充真实 nuScenes 更新等价；验证实际 host 路由、加权、BN/RNG 边界及 Adam 状态。
"""
import ast
import contextlib
import copy
import hashlib
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.ct_v2.motion import OrderedPhysicalMotionEncoder
from utils.config import load_yaml_config
from utils.dual_stream import DualStreamLoader
from utils.sampling_utils import StatelessCandidateBatchSampler
from utils.training_isolation import (
    assert_training_transaction_equal, capture_global_rng_state,
    freeze_batchnorm_running_stats, isolated_constructor_rng,
    partition_named_parameter_groups, restore_global_rng_state,
    weighted_candidate_sum,
)


ROOT = Path(__file__).resolve().parents[1]
HOST_TREE = ast.parse((ROOT / "models/seqtrack3d.py").read_text(encoding="utf-8-sig"))


def _host_method(name, observation_prefix=False):
    definition = copy.deepcopy(next(item for cls in HOST_TREE.body if isinstance(cls, ast.ClassDef)
        for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == name))
    definition.decorator_list = []
    if observation_prefix:
        cutoff = next(i for i, item in enumerate(definition.body) if isinstance(item, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "required_motion"
                              for target in item.targets))
        definition.body = definition.body[:cutoff] + [ast.Return(value=ast.Name(id="observation_output", ctx=ast.Load()))]
    namespace = dict(globals())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                 "models/seqtrack3d.py", "exec"), namespace)
    return namespace[name]


class TransactionHost(nn.Module):
    training_step = _host_method("training_step")
    configure_optimizers = _host_method("configure_optimizers")
    _ct_candidate_weighted_observation_loss = _host_method("_ct_candidate_weighted_observation_loss")
    _ct_module_parameter_hash = _host_method("_ct_module_parameter_hash")
    _ct_optimizer_state_hash = _host_method("_ct_optimizer_state_hash")
    _observation_prefix = _host_method("_forward_safe_mechanism", observation_prefix=True)
    prefixes = {"physical_motion_encoder": "b1", "ct_joint_search_refiner": "b2", "ct_joint_router": "b3"}

    def __init__(self, arm):
        super().__init__()
        self.config = SimpleNamespace(optimizer="Adam", lr=1e-4, wd=0., lr_decay_step=20, lr_decay_rate=.1)
        self.b0 = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Dropout(.4), nn.Linear(8, 2))
        enabled = {"b1": arm != "b0", "b2": arm in ("full_minus_b3", "full"), "b3": arm == "full"}
        for prefix, group in self.prefixes.items():
            if enabled[group]:
                with isolated_constructor_rng(42, group):
                    setattr(self, prefix, nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8),
                        nn.Dropout(.3), nn.Linear(8, 2)))
        self.ct_enable_b1, self.ct_enable_b2, self.ct_enable_b3 = (enabled[name] for name in ("b1", "b2", "b3"))
        self.ct_enable_v27 = True
        self.ct_unified_auto = True
        self.ct_separate_optimizers = False
        self.ct_initialization_policy = "scratch_only"
        self.use_ct_joint_full = self.ct_enable_b1
        self.use_b1motion_v3 = self.ct_enable_b1
        self.use_motion_cls = False
        self.ct_joint_contract_version = 3
        self.ct_manual_amp_enabled = False
        self.ct_b0_candidate_views = 4
        self.ct_b0_candidate_weights = (.5, 1 / 6, 1 / 6, 1 / 6)
        self.device = torch.device("cpu")
        self.global_step = 0
        self.logger = SimpleNamespace(experiment=SimpleNamespace(add_scalars=lambda *a, **k: None))
        self.seg_acc = lambda *a, **k: (torch.tensor(0.), torch.tensor(0.))

    @staticmethod
    def _ct_any_plugin_parameter(name):
        return name.split(".", 1)[0] in TransactionHost.prefixes

    @staticmethod
    def _ct_plugin_group(name):
        return TransactionHost.prefixes[name.split(".", 1)[0]]

    @staticmethod
    def _slice_batch_rows(batch, rows):
        return {key: value[rows] for key, value in batch.items()}

    def _ct_record_cuda_stage(self, *args):
        pass

    def _ct_record_observation_batch_fingerprint(self, *args):
        pass

    def _accumulate_joint_binary_rows(self, *args):
        pass

    def log(self, *args, **kwargs):
        pass

    def is_paired_batch(self, *args):
        return False

    def forward(self, batch):
        prediction = self.b0(batch["x"])
        return {"aux_estimation_boxes": prediction, "seg_logits": prediction.unsqueeze(-1)}

    def _forward_safe_mechanism(self, batch):
        result = self._observation_prefix(batch)  # actual v27 host eval/no-grad/restore section
        random.random()
        np.random.random(3)
        torch.rand(5)
        for prefix, group in self.prefixes.items():
            if getattr(self, "ct_enable_" + group):
                result[group] = getattr(self, prefix)(result["aux_estimation_boxes"].detach())
        return result

    def compute_loss(self, batch, output):
        base = (output["aux_estimation_boxes"] - batch["target"]).square().mean()
        losses = {"loss_total": base, "loss_b0_transaction": base}
        for group in ("b1", "b2", "b3"):
            if group in output:
                losses["loss_" + group + "_transaction"] = (output[group] - batch["target"]).square().mean()
        return losses


def _two_step_snapshot(arm):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    host = TransactionHost(arm).train()
    optimizer = host.configure_optimizers()["optimizer"]
    initial = {name: tensor.detach().clone() for name, tensor in host.b0.state_dict().items()}
    initial_rng = capture_global_rng_state()
    snapshots = []
    for step in range(2):
        batch = {"x": torch.randn(16, 4), "target": torch.randn(16, 2),
                 "candidate_id": torch.arange(4).repeat(4), "seg_label": torch.zeros(16, 1, dtype=torch.long)}
        mechanism = {key: value.clone() for key, value in batch.items()} if arm != "b0" else None
        optimizer.zero_grad(set_to_none=True)
        loss = host.training_step({"ct_stream_schema": "ct_seqtrack.train.v4",
                                  "observation": batch, "mechanism": mechanism}, step)
        loss.backward()
        for group in optimizer.param_groups:
            assert all(parameter.requires_grad for parameter in group["params"])
            grads = [parameter.grad for parameter in group["params"] if parameter.grad is not None]
            assert grads and all(torch.isfinite(grad).all() for grad in grads)
            assert any(torch.count_nonzero(grad) for grad in grads), group["name"]
        gradients = {name: parameter.grad.detach().clone() for name, parameter in host.b0.named_parameters()}
        optimizer.step()
        snapshots.append({"parameters_and_bn": copy.deepcopy(host.b0.state_dict()), "gradients": gradients,
                          "adam": {name: copy.deepcopy(optimizer.state[parameter])
                                   for name, parameter in host.b0.named_parameters()},
                          "rng": capture_global_rng_state()})
        host.global_step += 1
    return initial, initial_rng, snapshots


def test_five_arm_host_transactions_keep_b0_weights_gradients_adam_bn_and_rng_exact():
    baseline = _two_step_snapshot("b0")
    for arm in ("b1_gru", "b1_cfc", "full_minus_b3", "full"):
        actual = _two_step_snapshot(arm)
        assert_training_transaction_equal(baseline, actual, arm)


def test_actual_gru_cfc_named_common_encoder_decoder_and_margin_initialization_match():
    def build(backend):
        with isolated_constructor_rng(42, "b1.motion"):
            return OrderedPhysicalMotionEncoder(enable_v27=True, initialization_seed=42,
                temporal_backend=backend, adaptive_acquisition_margin=True)
    torch.manual_seed(101)
    before = torch.get_rng_state().clone()
    gru, cfc = build("gru"), build("cfc")
    assert torch.equal(before, torch.get_rng_state())
    common = [name for name in gru.state_dict() if not name.startswith("gru.")]
    assert common and all(name in cfc.state_dict() for name in common)
    for name in common:
        assert torch.equal(gru.state_dict()[name], cfc.state_dict()[name]), name
    assert all(parameter.requires_grad for module in (gru, cfc) for parameter in module.parameters())


class GlobalRandomLoader:
    def __init__(self, count, private_generator=None):
        self.count, self.private_generator = count, private_generator

    def __len__(self):
        return self.count

    def __iter__(self):
        for _ in range(self.count):
            result = (random.random(), np.random.rand(), torch.rand(2))
            if self.private_generator is not None:
                torch.rand(3, generator=self.private_generator)
            yield result
        random.random()
        np.random.rand()
        torch.rand(3)  # iterator finalization must also restore globals


@pytest.mark.parametrize('observation_steps,mechanism_steps', [(4, 2), (2, 5)])
def test_mechanism_fetch_and_exhaustion_preserve_global_rng_but_advance_private_generator(observation_steps, mechanism_steps):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    start = capture_global_rng_state()
    expected = list(DualStreamLoader(GlobalRandomLoader(observation_steps), None,
                                    schema='ct_seqtrack.train.v4', isolate_mechanism_rng=True))
    expected_state = capture_global_rng_state()
    restore_global_rng_state(start)
    private = torch.Generator().manual_seed(999)
    before = private.get_state().clone()
    actual = list(DualStreamLoader(GlobalRandomLoader(observation_steps), GlobalRandomLoader(mechanism_steps, private),
                                  schema='ct_seqtrack.train.v4', isolate_mechanism_rng=True))
    assert_training_transaction_equal([row["observation"] for row in expected], [row["observation"] for row in actual])
    assert_training_transaction_equal(expected_state, capture_global_rng_state())
    assert not torch.equal(before, private.get_state())


@pytest.mark.parametrize("frames", (16, 17, 101))
def test_reference_and_ct_observation_have_equal_four_candidate_epoch_budget(frames):
    reference = load_yaml_config(ROOT / "cfgs/27_seqtrack_reference.yaml")
    baseline = load_yaml_config(ROOT / "cfgs/ct_seqtrack/27_b0.yaml")
    for key in ("batch_size", "num_candidates", "point_sample_size", "hist_num", "train_type", "train_split",
                "bb_scale", "bb_offset", "main_time_source", "empty_box_limit", "limit_num_points_in_prev_box"):
        assert reference[key] == baseline[key], key
    dataset = list(range(frames * reference["num_candidates"]))
    ordinary = torch.utils.data.DataLoader(dataset, batch_size=16, drop_last=True)
    stateless = StatelessCandidateBatchSampler(dataset, batch_size=16, candidate_views=4, seed=42)
    assert len(ordinary) == len(stateless) == (4 * frames) // 16
