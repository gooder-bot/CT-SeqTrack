"""四分支完整覆盖与因果队列迁移合同。"""
from copy import deepcopy
import pytest
from models.ct_v38.data import ReadyQueueBatchSampler

@pytest.mark.parametrize('epoch',[0,4,9,59])
def test_four_branch_full_coverage_and_causal_ready_queue(epoch):
    lengths = [1, 2, 4, 11, 19]
    sampler = ReadyQueueBatchSampler(lengths, 7)
    sampler.set_epoch(epoch)
    rows = [row for batch in sampler for row in batch]
    assert len(rows) == 4 * sum(n - 1 for n in lengths)
    assert len({(r.tracklet, r.branch, r.frame) for r in rows}) == len(rows)
    previous = {}
    for batch in sampler:
        assert len({r.state_key for r in batch}) == len(batch)
        for r in batch:
            assert r.frame == previous.get(r.state_key, r.window_start - 1) + 1
            previous[r.state_key] = r.frame
    assert len(list(sampler)) == len(sampler)
    assert sum(r.branch == 0 for r in rows) * 4 == len(rows)
    assert max(r.window_end - r.window_start for r in rows if r.branch == 3) <= 8
    assert sampler.state_dict() == deepcopy(sampler).state_dict()


def test_curriculum_lengths_and_offset_windows():
    sampler = ReadyQueueBatchSampler([30], 4)
    assert sampler.window_length(8) == 1
    sampler.set_epoch(9)
    rows = [r for batch in sampler for r in batch]
    assert sampler.window_length(8) == 8
    starts1 = {r.window_start for r in rows if r.branch == 1}
    starts2 = {r.window_start for r in rows if r.branch == 2}
    assert starts1 != starts2
    assert len(rows) == 116

