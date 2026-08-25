from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_variant import configure_ct_variant
from utils.config import load_yaml_config
from utils.dual_stream import DualStreamLoader
from utils.online_contract import (
    build_online_resume_contract,
    validate_online_resume_contract,
    validate_scratch_training_contract,
)
from utils.sampling_utils import (
    StatelessCandidateBatchSampler,
    deterministic_candidate_offset,
    deterministic_candidate_retry_index,
    deterministic_point_seed,
    prune_seqtrack_observation_payload,
)
from utils.training_isolation import (
    freeze_batchnorm_running_stats,
    partition_named_parameter_groups,
    weighted_candidate_sum,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v25_configs_are_safe_unified_scratch_arms():
    expected = {
        "25_b0.yaml": "b0",
        "25_b1.yaml": "b1",
        "25_full_minus_b3.yaml": "full_minus_b3",
        "25_full.yaml": "full",
    }
    for name, variant in expected.items():
        config = load_yaml_config(ROOT / "cfgs" / "ct_seqtrack" / name)
        configure_ct_variant(config)
        assert config["ct_variant"] == variant
        assert config["ct_runtime_protocol"] == "safe_seqtrack_auto_v1"
        assert config["ct_optimizer_topology"] == "unified_auto"
        assert config["ct_observation_rng_mode"] == "stateless_seqtrack"
        assert config["ct_batch_schema"] == "ct_seqtrack.train.v2"
        assert config["ct_candidate_policy"] == "b2_raw"
        assert config["ct_b0_candidate_weights"] == pytest.approx(
            [0.5, 1 / 6, 1 / 6, 1 / 6])
        assert config["ct_b0_candidate_views"] == 4
        assert config["ct_b2_candidate_views"] == 1
        assert config["ct_cuda_stage_audit"] is True
        assert config["ct_observation_fingerprint_steps"] == 100
        assert config["ct_separate_optimizers"] is False
        assert config["motion_v3_temporal_backend"] == "gru"
        assert config["motion_v3_cfc_backbone_units"] == 105
        assert config["motion_v3_beta_nll_beta"] == pytest.approx(0.5)
        assert config["motion_v3_tail_direction_weight"] == pytest.approx(
            0.25)
        assert config["motion_v3_tail_direction_margin"] == pytest.approx(0.9)
        assert config["motion_v3_log_sigma_min"] == pytest.approx(
            -2.302585092994046)
        assert config["motion_v3_log_sigma_max"] == pytest.approx(2.5)
        assert config["motion_v3_aux_nll_weight"] == pytest.approx(0.05)
        assert [config[f"ct_{name}_lr"] for name in ("b0", "b1", "b2", "b3")] == [
            1e-4, 1e-4, 1e-4, 1e-4]
        assert config["ct_initialization_policy"] == "scratch_only"
        assert (config["batch_size"], config["epoch"], config["seed"],
                config["check_val_every_n_epoch"]) == (16, 60, 42, 1)
        validate_scratch_training_contract(config)


def test_v25_full_nuscenes_configs_inherit_only_split_changes():
    for stem in ("b0", "b1", "full_minus_b3", "full"):
        config = load_yaml_config(
            ROOT / "cfgs" / "ct_seqtrack" /
            f"25_{stem}_nuscenes_full.yaml")
        configure_ct_variant(config)
        assert config["train_split"] == "train_track"
        assert config["val_split"] == "val"
        assert config["version"] == "v1.0-trainval"
        assert config["ct_b0_steps_per_epoch"] == 0
        assert config["ct_runtime_protocol"] == "safe_seqtrack_auto_v1"
        validate_scratch_training_contract(config)


class _CandidateDataset:
    def __init__(self, size):
        self.size = int(size)
        self.epochs = []

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epochs.append(int(epoch))


def test_stateless_candidate_batches_are_balanced_and_epoch_addressable():
    dataset = _CandidateDataset(40)
    sampler = StatelessCandidateBatchSampler(
        dataset, batch_size=8, candidate_views=4, seed=42)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    replay = list(sampler)
    sampler.set_epoch(4)
    changed = list(sampler)

    assert first == replay
    assert first != changed
    assert len(first) == 5
    assert dataset.epochs == [3, 3, 4]
    for batch in first:
        assert Counter(index % 4 for index in batch) == {
            0: 2, 1: 2, 2: 2, 3: 2}
    assert sorted(index for batch in first for index in batch) == list(range(40))


def test_stateless_candidate_and_point_roles_are_identity_stable():
    config = SimpleNamespace(seed=42, degrees=False)
    assert np.array_equal(
        deterministic_candidate_offset(0, config, "e", 1),
        np.zeros(3, dtype=np.float32))
    first = deterministic_candidate_offset(2, config, "epoch", 3, "frame", 9)
    replay = deterministic_candidate_offset(2, config, "epoch", 3, "frame", 9)
    changed = deterministic_candidate_offset(2, config, "epoch", 4, "frame", 9)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, changed)
    assert deterministic_point_seed(config, "history", 7) == (
        deterministic_point_seed(config, "history", 7))
    assert deterministic_point_seed(config, "history", 7) != (
        deterministic_point_seed(config, "current", 7))


def test_stateless_retry_changes_annotation_but_preserves_candidate_branch():
    initial = [5 * 4 + candidate_id for candidate_id in range(4)]
    first = [
        deterministic_candidate_retry_index(
            index, dataset_length=64, candidate_views=4,
            seed=42, epoch=3, attempt=0)
        for index in initial]
    replay = [
        deterministic_candidate_retry_index(
            index, dataset_length=64, candidate_views=4,
            seed=42, epoch=3, attempt=0)
        for index in initial]

    assert first == replay
    assert [index % 4 for index in first] == [0, 1, 2, 3]
    assert all(index // 4 != 5 for index in first)


def test_b0_candidate_loss_is_half_canonical_half_auxiliary():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    branches = [parameter * value for value in (1.0, 2.0, 3.0, 4.0)]
    weights = (0.5, 1 / 6, 1 / 6, 1 / 6)
    loss = weighted_candidate_sum(branches, weights)
    expected = 0.5 * 1.0 + (2.0 + 3.0 + 4.0) / 6.0
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert parameter.grad.item() == pytest.approx(expected)


def test_unified_optimizer_has_ordered_disjoint_named_groups():
    modules = {
        "core": torch.nn.Linear(2, 2),
        "physical_motion_encoder": torch.nn.Linear(2, 2),
        "ct_joint_search_refiner": torch.nn.Linear(2, 2),
        "ct_joint_router": torch.nn.Linear(2, 2),
    }
    named = [
        (f"{module_name}.{parameter_name}", parameter)
        for module_name, module in modules.items()
        for parameter_name, parameter in module.named_parameters()]
    prefixes = {
        "physical_motion_encoder": "b1",
        "ct_joint_search_refiner": "b2",
        "ct_joint_router": "b3",
    }
    is_plugin = lambda name: name.split(".", 1)[0] in prefixes
    plugin_group = lambda name: prefixes[name.split(".", 1)[0]]
    grouped = partition_named_parameter_groups(
        named, is_plugin, plugin_group,
        {"b1": True, "b2": True, "b3": True})
    optimizer = torch.optim.Adam([
        {"params": [parameter for _, parameter in group],
         "lr": 1e-4, "name": name}
        for name, group in grouped.items()], lr=1e-4)
    assert isinstance(optimizer, torch.optim.Adam)
    assert [group["name"] for group in optimizer.param_groups] == [
        "b0", "b1", "b2", "b3"]
    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {
        id(parameter) for _, parameter in named}


class _Loader:
    def __init__(self, values):
        self.values = list(values)
        self.iterator_count = 0

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        self.iterator_count += 1
        return iter(self.values)


def test_v2_envelope_is_common_and_mechanism_iterator_is_lazy():
    observation = _Loader(range(6))
    mechanism = _Loader(["m0", "m1"])
    loader = DualStreamLoader(
        observation, mechanism, schema="ct_seqtrack.train.v2")
    iterator = iter(loader)
    assert mechanism.iterator_count == 0
    first = next(iterator)
    assert first == {
        "ct_stream_schema": "ct_seqtrack.train.v2",
        "observation": 0,
        "mechanism": None,
    }
    assert mechanism.iterator_count == 0
    rows = [first, *list(iterator)]
    assert mechanism.iterator_count == 1
    assert [row["mechanism"] for row in rows
            if row["mechanism"] is not None] == ["m0", "m1"]

    b0_rows = list(DualStreamLoader(
        _Loader([1, 2]), None, schema="ct_seqtrack.train.v2"))
    assert all(row["mechanism"] is None for row in b0_rows)


def test_observation_payload_prunes_ct_mechanism_fields():
    payload = {
        "points": np.zeros((4, 5), dtype=np.float32),
        "box_label": np.zeros(4, dtype=np.float32),
        "candidate_id": np.int64(0),
        "ct_extension_points": np.ones((256, 5), dtype=np.float32),
        "motion_main_ref_boxs": np.ones((3, 4), dtype=np.float32),
        "ct_search_geometry_valid": np.float32(1.0),
        "current_delta_t_real": np.float32(0.1),
        "point_sampling_seeds": np.ones(3, dtype=np.int64),
    }
    pruned = prune_seqtrack_observation_payload(payload)
    assert set(pruned) == {"points", "box_label", "candidate_id"}


def test_v25_resume_contract_rejects_v24_protocol():
    v25 = load_yaml_config(
        ROOT / "cfgs" / "ct_seqtrack" / "25_b0.yaml")
    configure_ct_variant(v25)
    contract = build_online_resume_contract(v25)
    assert contract["schema"] == "ct_seqtrack.online_resume_contract.v8"
    checkpoint = {
        "ct_online_resume_contract": contract,
        "ct_epoch_boundary_complete": True,
        "ct_global_rng_state": {"schema": "ct_seqtrack.global_rng.v1"},
    }
    validate_online_resume_contract(checkpoint, v25)

    v24 = load_yaml_config(
        ROOT / "cfgs" / "ct_seqtrack" / "24_b0.yaml")
    configure_ct_variant(v24)
    with pytest.raises(ValueError, match="online_resume_contract.v6"):
        validate_online_resume_contract(checkpoint, v24)


def test_bounded_train_tool_accepts_v24_and_v25_candidate_weights():
    source = (ROOT / "tools" / "check_train_steps.py").read_text(
        encoding="utf-8")
    assert "[0.5, 1 / 6, 1 / 6, 1 / 6]" in source
    assert "[0.25, 0.25, 0.25, 0.25]" in source
    assert '"ct_seqtrack.bounded_train_check.v2"' in source
    compare_source = (
        ROOT / "tools" / "compare_ct_module_audits.py").read_text(
            encoding="utf-8")
    assert 'required = ("initial", "step_1", "step_100")' in compare_source
    assert "first-100 observation fingerprints mismatch" in compare_source
    assert "ct_b0_optimizer_state_hashes" in compare_source
    assert "lacks a finite nonzero gradient" in compare_source


def test_v25_cuda_audit_covers_registered_training_phases():
    source = (ROOT / "models" / "seqtrack3d.py").read_text(encoding="utf-8")
    for stage in ("batch_transfer", "forward", "loss", "backward", "step"):
        assert f"_ct_record_cuda_stage('{stage}'" in source
    assert "ct_cuda_stage_audit" in source
    assert "ct_observation_batch_fingerprints" in source
    assert "ct_b0_optimizer_state_hashes" in source
    assert "max_gradient_norm" in source


def test_mechanism_shadow_freezes_only_b0_batchnorm():
    module = torch.nn.Module()
    module.b0 = torch.nn.BatchNorm1d(2)
    module.ct_joint_search_refiner = torch.nn.BatchNorm1d(2)
    module.train()
    with freeze_batchnorm_running_stats(
            module, excluded_prefixes=("ct_joint_search_refiner",)):
        assert module.b0.training is False
        assert module.ct_joint_search_refiner.training is True
    assert module.b0.training is True
    assert module.ct_joint_search_refiner.training is True


def test_default_b2_raw_keeps_existing_b3_availability_contract():
    source = (ROOT / "models" / "seqtrack3d.py").read_text(encoding="utf-8")
    assert "router_availability = evidence_contract.structural_available" in source
    assert "availability=router_availability" in source
