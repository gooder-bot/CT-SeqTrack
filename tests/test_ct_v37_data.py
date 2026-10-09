"""v37 原 W 排程、选中框写入权限及旧 trusted 相邻运动边。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v37.config import normalize_config, as_v35_config
from models.ct_v37.data import BatchBuilder, build_loaders, RawEndpointDataset
from models.ct_v31.data import BatchBuilder as LegacyBuilder, build_loaders as legacy_loaders
from tests.test_ct_v31_runtime import TinySource, request, fake_prior, fake_output
from tests.test_ct_v34_resume import _assert_identical


def cfg(arm='full'):
    return normalize_config(dict(v37_arm=arm, ct_engineering_check=True, point_sample_size=8,
        workers=0, batch_size=4, v32_reserve_windows=0, epoch=11))


def test_v37_sampler_and_batches_are_exactly_v35_w():
    config = cfg()
    new = build_loaders(config, roles=('train',), sources={'train': TinySource()})['train']
    old = legacy_loaders(as_v35_config(config, arm='full'), roles=('train',),
                        sources={'train': TinySource()})['train']
    for epoch in (0, 9, 10, 59):
        new.batch_sampler.set_epoch(epoch)
        old.batch_sampler.set_epoch(epoch)
        assert new.batch_sampler.state_dict() == old.batch_sampler.state_dict()
        assert list(new.batch_sampler) == list(old.batch_sampler)


@pytest.mark.parametrize('arm', ['b0', 'b1'])
def test_baseline_prepare_acquire_commit_are_unmodified(arm):
    config = cfg(arm)
    new, old = BatchBuilder(config), LegacyBuilder(as_v35_config(config, arm=arm))
    raw = RawEndpointDataset(TinySource())
    for frame in (1, 2, 3):
        row = raw[request(frame)]
        a, b = new.prepare([row]), old.prepare([deepcopy(row)])
        _assert_identical(a, b)
        new.acquire(a, fake_prior(a)); old.acquire(b, fake_prior(b))
        _assert_identical(a, b)
        out = fake_output(a)
        left = new.commit(out, a, diagnostics=True)
        right = old.commit(deepcopy(out), b, diagnostics=True)
        assert left == right
        astate, bstate = new.states[row['request'].state_key], old.states[row['request'].state_key]
        _assert_identical({k: v for k, v in vars(astate).items() if k != 'memory'},
                          {k: v for k, v in vars(bstate).items() if k != 'memory'})
        _assert_identical(vars(astate.memory), vars(bstate.memory))


def prepared(arm='full'):
    builder = BatchBuilder(cfg(arm))
    row = RawEndpointDataset(TinySource())[request(1)]
    batch = builder.prepare([row]); builder.acquire(batch, fake_prior(batch))
    return builder, row, batch


def output_for(batch, selected=0, fallback=False):
    out = fake_output(batch)
    boxes = batch['fallback_box'][:, None].repeat(1, 4, 1)
    out.candidates = SimpleNamespace(boxes=boxes, valid=torch.ones(1, 4, dtype=torch.bool))
    out.accepted_box = boxes[:, selected]
    out.selected_index[:] = selected
    out.q0_is_fallback = torch.tensor([fallback])
    if selected == 1 or (selected == 0 and fallback):
        out.selected_quality.zero_()
    out.commit_quality = out.selected_quality[:, None].repeat(1, 4)
    out.evidence.point_valid = torch.zeros(1, 256, dtype=torch.bool)
    out.evidence.point_indices = torch.full((1, 256), -1, dtype=torch.long)
    out.evidence.members = torch.zeros(1, 2, 256, dtype=torch.bool)
    out.evidence.mode_valid = torch.ones(1, 2, dtype=torch.bool)
    return out


@pytest.mark.parametrize('selected,fallback', [(1, False), (0, True)])
def test_prior_and_fallback_propagate_without_trusted_or_memory_write(selected, fallback):
    builder, row, batch = prepared()
    state = builder.states[row['request'].state_key]
    previous_trusted = deepcopy(state.trusted)
    out = output_for(batch, selected=selected, fallback=fallback)
    record = builder.commit(out, batch, diagnostics=True)[0]
    assert record['propagation_only'] and not record['memory_write'] and not record['supported']
    assert not record['strong'] and record['commit_quality'] == 0
    assert state.next_frame == 2 and not state.memory.recent
    _assert_identical(state.trusted, previous_trusted)
    with pytest.raises(RuntimeError, match='duplicate commit'):
        builder.commit(out, batch)


def test_shadow_uses_only_current_b0_and_never_extension_for_write():
    builder, row, batch = prepared('b1_b2')
    out = output_for(batch)
    seen = {}
    state = builder.states[row['request'].state_key]
    original = state.memory.update
    def record(points, ids, probabilities, *args):
        seen['ids'] = ids.copy()
        return original(points, ids, probabilities, *args)
    state.memory.update = record
    report = builder.commit(out, batch, diagnostics=True)[0]
    assert set(seen['ids']).issubset(set(batch['point_ids'][0, -1][batch['point_valid'][0, -1]].tolist()))
    assert report['commit_source'] == 'q0'


def test_mode_permission_is_own_selected_members_and_original_batch_unchanged():
    builder, row, batch = prepared()
    # raw 4/5 are outside B0. Two selected slots, only the first belongs to mode0.
    batch['extension_ids'][0, :2] = torch.tensor([4, 5])
    batch['extension_valid'][0, :2] = True
    out = output_for(batch, selected=2)
    out.evidence.point_valid[0, :2] = True
    out.evidence.point_indices[0, :2] = torch.tensor([0, 1])
    out.evidence.members[0, 0, 0] = True
    out.evidence.members[0, 1, 1] = True
    out.evidence.identity_logits.fill_(-3.)  # BG retained in memory input, independent of own-box FG.
    seen = {}
    def record(points, ids, probabilities, *args):
        seen['ids'] = ids.copy()
        return False
    builder.states[row['request'].state_key].memory.update = record
    original = batch['extension_valid'].clone()
    builder.commit(out, batch)
    assert 4 in seen['ids'] and 5 not in seen['ids']
    assert torch.equal(batch['extension_valid'], original)


def test_shadow_rejects_nonq0_and_propagation_quality_is_fail_closed():
    builder, _, batch = prepared('b1_b2')
    with pytest.raises(ValueError, match='commit q0'):
        builder.commit(output_for(batch, selected=2), batch)
    builder, _, batch = prepared()
    out = output_for(batch, selected=1)
    out.selected_quality[:] = .9
    with pytest.raises(ValueError, match='commit quality'):
        builder.commit(out, batch)
