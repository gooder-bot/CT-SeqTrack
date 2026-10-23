"""事务规范化缓存的别名、失效及旧输入梯度路径。"""
import pytest
import torch

from models.ct_v38.model import canonicalize_batch
from models.ct_v38.observation import B0Observation
from tests.helpers.synthetic_batch import make_batch
from utils.batch_materialization import (attach_materialization_cache,
    clear_materialization_cache, materialization_cache)


def cached_batch():
    return attach_materialization_cache(make_batch(1, 8), enabled=True)


def test_original_and_canonical_aliases_reuse_validated_tensors(monkeypatch):
    import models.ct_v38.observation as observation
    original = observation.unique_valid_mask
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(observation, 'unique_valid_mask', counted)
    batch = cached_batch()
    batch['point_ids'][..., 1] = batch['point_ids'][..., 0]
    batch['points'][..., 1, :] = float('nan')
    first = canonicalize_batch(batch)
    second = canonicalize_batch(batch)
    third = canonicalize_batch(first)
    assert len(calls) == 1
    for key in ('points', 'point_valid'):
        assert first[key] is second[key] is third[key]
    assert B0Observation.measurement_mask(first) is first['point_valid']
    assert B0Observation.measurement_mask(batch) is first['point_valid']
    assert len(calls) == 1
    assert torch.count_nonzero(first['points'][..., 1, :]) == 0


@pytest.mark.parametrize('key', ('points', 'point_valid', 'point_ids', 'history_valid'))
def test_original_in_place_change_invalidates_both_aliases(key):
    batch = cached_batch()
    first = canonicalize_batch(batch)
    if key == 'points':
        batch[key][0, -1, 0, 0] += 1
    elif key == 'point_ids':
        batch[key][..., 1] = batch[key][..., 0]
    else:
        batch[key].reshape(-1)[0] = False
    cache = materialization_cache(batch)
    assert cache.canonical_values(first) is None
    actual = canonicalize_batch(batch)
    expected = canonicalize_batch({k: v for k, v in batch.items() if torch.is_tensor(v)})
    assert actual['points'] is not first['points']
    assert torch.equal(actual['points'], expected['points'])
    assert torch.equal(actual['point_valid'], expected['point_valid'])


def test_canonical_mutation_invalidates_original_and_rechecks_finite_values():
    batch = cached_batch()
    first = canonicalize_batch(batch)
    first['points'][0, -1, 0, 0] = float('nan')
    assert materialization_cache(batch).canonical_values(batch) is None
    with pytest.raises(ValueError, match='finite'):
        canonicalize_batch(first)
    fresh = canonicalize_batch(batch)
    assert torch.isfinite(fresh['points']).all()




def test_mask_only_cache_detects_mutated_returned_mask():
    batch = cached_batch()
    mask = B0Observation.measurement_mask(batch)
    mask.zero_()
    assert B0Observation.measurement_mask(batch).all()


def test_clear_invalidates_copied_batch_sidecar_and_new_transactions():
    batch = cached_batch()
    prepared = canonicalize_batch(batch)
    cache = materialization_cache(batch)
    clear_materialization_cache(batch)
    assert materialization_cache(batch) is None
    assert cache.point is None
    assert materialization_cache(prepared).canonical_values(prepared) is None
    attach_materialization_cache(batch, enabled=True)
    assert materialization_cache(batch) is not cache
    attach_materialization_cache(batch, enabled=False)
    assert materialization_cache(batch) is None


def test_input_gradients_and_inference_tensors_use_uncached_path():
    batch = cached_batch()
    batch['points'].requires_grad_()
    first, second = canonicalize_batch(batch), canonicalize_batch(batch)
    assert first['points'] is not second['points']
    assert materialization_cache(batch).point is None
    (first['points'].sum() + second['points'].sum()).backward()
    assert torch.equal(batch['points'].grad, torch.full_like(batch['points'], 2.))
    with torch.inference_mode():
        inference = cached_batch()
        first, second = canonicalize_batch(inference), canonicalize_batch(inference)
        assert first['points'] is not second['points']
        assert materialization_cache(inference).point is None
        assert torch.equal(first['points'], second['points'])
