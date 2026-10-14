"""工程去重的权限、瞬时缓存、局部读取及混合候选提交合同（CPU）。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from models.ct_v37 import verification
from models.ct_v37.data import BatchBuilder, RawEndpointDataset
from models.ct_v37.decoder import SharedHypothesisDecoder
from models.ct_v37.model import JointTracker
from models.ct_v37.verification import FinalBoxVerifier, build_current_pool
from tests.test_ct_v31_runtime import TinySource, request, fake_prior, fake_output
from tests.test_ct_v37_data import cfg
from tests.test_ct_v37_network import batch, config, _decoder_inputs
from tools.check_execution_performance import assert_identical, freeze
from utils.execution_lab import execution_scope


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(1410)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('empty', [False, True])
def test_pool_shared_once_and_never_cached_between_forwards(monkeypatch, empty):
    model = JointTracker(config()).eval()
    values = batch()
    if empty:
        values['point_valid'].zero_()
        values['extension_valid'].zero_()
    else:
        values['point_ids'][:, -1, 1] = values['point_ids'][:, -1, 0]
    calls = []
    original = verification._build_current_pool
    def record(*args):
        pool = original(*args)
        calls.append(pool)
        return pool
    monkeypatch.setattr(verification, '_build_current_pool', record)
    with torch.no_grad():
        expected = model(values)
        assert len(calls) == 2
        calls.clear()
        with execution_scope(('pool',), engineering=True):
            actual = model(values)
            assert len(calls) == 1
            assert '_raw_id_order' in calls[0]
            assert_identical(freeze(expected), freeze(actual))
            values['point_ids'].add_(10000)
            model(values)
            assert len(calls) == 2
            assert calls[0] is not calls[1]
            assert not torch.equal(calls[0]['ids'], calls[1]['ids'])
    assert not any('pool' in key for key in model.state_dict())


def test_shared_live_pool_preserves_decoder_gradients_but_verifier_is_detached():
    values, observation, prior, evidence = _decoder_inputs()
    evidence.point_features = evidence.point_features.detach().requires_grad_()
    decoder = SharedHypothesisDecoder(dropout=0.).eval()
    verifier = FinalBoxVerifier().eval()
    torch.nn.init.normal_(decoder.local_reader.output.weight, std=.03)
    torch.nn.init.normal_(verifier.output.weight, std=.02)
    results = []
    for options in ((), ('pool',)):
        with execution_scope(options, engineering=True):
            pool = build_current_pool(observation, evidence, values) if options else None
            candidates = decoder(observation, evidence, values, prior, _current_pool=pool)
            output = verifier(candidates, observation, evidence, values, prior, _current_pool=pool)
            upstream = (observation.point_features, evidence.point_features, candidates.boxes,
                        evidence.centers_xyz, prior.feature)
            gradients = torch.autograd.grad(output.logits.sum(), upstream,
                retain_graph=True, allow_unused=True)
            assert all(value is None for value in gradients)
            gradients = torch.autograd.grad(candidates.features.square().sum(),
                (observation.point_features, evidence.point_features), retain_graph=True)
            assert all(value.abs().sum() > 0 for value in gradients)
            results.append((freeze(candidates), freeze(output), gradients))
    assert_identical(results[0], results[1])


@pytest.mark.parametrize('options', [('pool',), ('geometry_gather',),
                                    ('pool', 'geometry_gather')])
def test_two_training_steps_preserve_loss_gradients_adam_bn_and_rng(options):
    # prior_read 不列入严格通过组：两步 CPU Adam 已观察到 reader bias 梯度差异。
    # 工程入口保留该候选完整逐位验收，并记录拒绝原因，不能借此测试批准它。
    old = JointTracker(config()).train()
    new = deepcopy(old)
    values = batch()
    optimizers = [torch.optim.Adam(model.parameters(), lr=1.5e-4) for model in (old, new)]
    for _ in range(2):
        rng = torch.get_rng_state()
        snapshots = []
        for model, optimizer, flags in zip((old, new), optimizers, ((), options)):
            torch.set_rng_state(rng)
            optimizer.zero_grad(set_to_none=True)
            with execution_scope(flags, engineering=True):
                output = model(values)
                losses = model.compute_losses(values, output)
                losses['loss_total'].backward()
                gradients = freeze({name: p.grad for name, p in model.named_parameters()})
                optimizer.step()
            snapshots.append(freeze(dict(output=output, losses=losses, gradients=gradients,
                model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state())))
        assert_identical(*snapshots)


@pytest.mark.parametrize('training', [False, True])
def test_prior_read_removed_only_from_decoder_and_preserves_outputs_rng(monkeypatch, training):
    from models.ct_v37 import decoder as decoder_module
    values, observation, prior, evidence = _decoder_inputs()
    decoder = SharedHypothesisDecoder(dropout=.2).train(training)
    verifier = FinalBoxVerifier().train(training)
    torch.nn.init.normal_(decoder.local_reader.output.weight, std=.03)
    torch.nn.init.normal_(decoder.pose_head.weight, std=.03)
    seen = []
    original = verification.read_lwh_local
    def record(reader, pool, boxes, allowed, size):
        seen.append((reader, boxes.shape[1]))
        return original(reader, pool, boxes, allowed, size)
    monkeypatch.setattr(decoder_module, 'read_lwh_local', record)
    monkeypatch.setattr(verification, 'read_lwh_local', record)
    rng = torch.get_rng_state()
    expected = decoder(observation, evidence, values, prior)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    with execution_scope(('prior_read',), engineering=True):
        actual = decoder(observation, evidence, values, prior)
        verifier(actual, observation, evidence, values, prior)
    assert seen == [(decoder.local_reader, 4), (decoder.local_reader, 3), (verifier.local_reader, 4)]
    assert_identical(freeze(expected), freeze(actual))
    assert torch.equal(expected_rng, torch.get_rng_state())
    # CUDA GEMM 的形状变化仍须经过工程入口的逐位梯度/Adam 验收。


def mixed_transaction():
    builder = BatchBuilder(cfg())
    raw = RawEndpointDataset(TinySource(lengths=(10,) * 5))
    rows = [raw[request(1, track=i)] for i in range(5)]
    values = builder.prepare(rows)
    builder.acquire(values, fake_prior(values))
    output = fake_output(values)
    selected = torch.tensor([0, 1, 2, 3, 0])
    fallback = torch.tensor([False, False, False, False, True])
    boxes = values['fallback_box'][:, None].repeat(1, 4, 1)
    output.candidates = SimpleNamespace(boxes=boxes, valid=torch.ones(5, 4, dtype=torch.bool))
    output.selected_index = selected
    output.q0_is_fallback = fallback
    output.accepted_box = boxes[torch.arange(5), selected]
    output.selected_quality = torch.tensor([.9, 0., .9, .9, 0.])
    output.commit_quality = output.selected_quality[:, None].repeat(1, 4)
    output.evidence.point_valid = torch.zeros(5, 256, dtype=torch.bool)
    output.evidence.point_indices = torch.full((5, 256), -1, dtype=torch.long)
    output.evidence.members = torch.zeros(5, 2, 256, dtype=torch.bool)
    output.evidence.mode_valid = torch.ones(5, 2, dtype=torch.bool)
    values['extension_ids'][:, :2] = torch.tensor([4, 5])
    values['extension_valid'][:, :2] = True
    output.evidence.point_valid[:, :2] = True
    output.evidence.point_indices[:, :2] = torch.tensor([0, 1])
    output.evidence.members[:, 0, 0] = True
    output.evidence.members[:, 1, 1] = True
    return builder, values, output


@pytest.mark.parametrize('diagnostics', [False, True])
def test_batched_commit_indices_preserve_mixed_permissions_state_and_diagnostics(diagnostics):
    first, values, output = mixed_transaction()
    second = deepcopy(first)
    original_valid = values['extension_valid'].clone()
    expected = first.commit(output, values, diagnostics=diagnostics)
    with execution_scope(('commit_indices',), engineering=True):
        actual = second.commit(output, values, diagnostics=diagnostics)
    assert_identical(freeze(expected), freeze(actual))
    assert_identical(freeze(first.states), freeze(second.states))
    assert torch.equal(values['extension_valid'], original_valid)
    if diagnostics:
        assert [r['commit_source'] for r in actual] == ['q0', 'prior', 'mode0', 'mode1', 'q0']
        assert actual[1]['propagation_only'] and actual[4]['propagation_only']


@pytest.mark.parametrize('failure', ['no_members', 'bad_slot', 'invalid_candidate'])
def test_batched_commit_still_rejects_invalid_mode(failure):
    builder, values, output = mixed_transaction()
    if failure == 'no_members':
        output.evidence.members[2, 0].zero_()
    elif failure == 'bad_slot':
        output.evidence.point_indices[2, 0] = -1
    else:
        output.candidates.valid[2, 2] = False
    with execution_scope(('commit_indices',), engineering=True), pytest.raises(ValueError):
        builder.commit(output, values)

