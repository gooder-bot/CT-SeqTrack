"""新五组固定主模型、预算重建与真实宿主候选读取。"""
import copy
import json

import pytest
import torch

from tools import compare_v37 as compare


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def summary_fixture(monkeypatch, full_passed=False):
    runs = {label: dict(manifest=dict(run_id=label, common_parameter_hash='a' * 64,
        prior_parameter_hash=None if label == 'b0' else 'b' * 64, initial_parameter_hash='c' * 64),
        sampler_plans={e: str(e) for e in range(1, 61)},
        rows={60: {}}, mechanism_diagnostics={}) for label in (*compare.LABELS, 'baseline', 'old_w_quarter')}
    def loaded(*args, **kwargs):
        assert kwargs['candidate_labels'] == compare.LABELS
        assert ('b1_minus_strong_b0', 'b1', 'strong_b0') in kwargs['comparisons']
        assert ('full_bf16_minus_full', 'full_bf16', 'full') in kwargs['comparisons']
        return dict(candidates={label: dict(passed=full_passed if label == 'full' else True) for label in compare.LABELS},
            status='passed', selected='full_bf16', passing_rank=['full_bf16'], criterion={}, limitations=[])
    monkeypatch.setattr(compare.base, 'compare_loaded', loaded)
    monkeypatch.setattr(compare.original, 'tail_diagnostics', lambda run: {})
    monkeypatch.setattr(compare.original, 'explanatory_slices', lambda *args: {})
    monkeypatch.setattr(compare.original, 'tail_counts', lambda run: dict(final60=0, late3=0))
    return runs, dict(source=dict(sha256='a' * 64), reused={})


@pytest.mark.parametrize('passed', [False, True])
def test_bf16_or_control_winner_never_replaces_registered_full(monkeypatch, passed):
    runs, registration = summary_fixture(monkeypatch, passed)
    result = compare.summarize_loaded(runs, registration)
    assert result['status'] == ('passed' if passed else 'failed')
    assert result['selected'] == ('full' if passed else None)
    assert result['registered_main_model'] == 'full' and 'passing_rank' not in result


def test_changed_sampling_plan_is_not_a_precision_control(monkeypatch):
    runs, registration = summary_fixture(monkeypatch)
    runs['full_bf16']['sampler_plans'][11] = 'forged'
    with pytest.raises(compare.EvidenceError, match='sampler plans'):
        compare.summarize_loaded(runs, registration)


@pytest.mark.parametrize('label,field,message', [('b1', 'common_parameter_hash', 'B0/q0'),
    ('b1_b2', 'prior_parameter_hash', 'B1'), ('full_bf16', 'initial_parameter_hash', 'FP32/BF16')])
def test_comparison_rejects_mismatched_initialization(monkeypatch, label, field, message):
    runs, registration = summary_fixture(monkeypatch)
    runs[label]['manifest'][field] = 'd' * 64
    with pytest.raises(compare.EvidenceError, match=message):
        compare.summarize_loaded(runs, registration)


def test_forged_sampler_checksum_does_not_hide_wrong_plan():
    from models.ct_v31.data import ReadyQueueBatchSampler
    from models.ct_v37.config import normalize_config
    sampler = ReadyQueueBatchSampler([4778], batch_size=16, short_window=4,
        source_sha256='a' * 64, initial_seed_policy='initial_exact_other_perturbed_v1')
    sampler.set_epoch(10)
    value = sampler.state_dict()
    value['plan_sha256'] = 'b' * 64
    value['manifest_sha256'] = compare.original.digest({k: v for k, v in value.items() if k != 'manifest_sha256'})
    with pytest.raises(compare.EvidenceError, match='does not reconstruct'):
        compare.validate_sampler(value, normalize_config(), 11, 'full')


def real_rows(arm, empty=False):
    import numpy as np
    from models.ct_v37.data import build_loaders
    from models.ctseqtrackv37 import CTSEQTRACKV37
    from tests.test_ct_v37_runtime import config
    from tests.test_ct_v31_runtime import TinySource
    class EmptySource(TinySource):
        def get_frames(self, index, frame_ids):
            result = super().get_frames(index, frame_ids)
            for row in result:
                row['pc'], row['point_ids'] = np.empty((0, 3)), np.empty(0, np.int64)
            return result
    cfg = config(v37_arm=arm)
    source = (EmptySource if empty else TinySource)(lengths=(3, 2))
    loaders = build_loaders(cfg, roles=('test',), sources={'test': source})
    host = CTSEQTRACKV37(cfg, loaders=loaders).eval()
    host.log = lambda *args, **kwargs: None
    host.on_test_epoch_start()
    with torch.no_grad():
        for index, rows in enumerate(host._loader('test')):
            host.test_step(rows, index)
    host.on_test_epoch_end()
    return [row for row in host.evaluation.rows if not row['initialization']]


@pytest.mark.parametrize('arm', ['b0', 'b1', 'b1_b2', 'full'])
def test_real_host_candidate_records_pass_independent_reader(arm):
    for row in real_rows(arm):
        compare.validate_candidates(row, arm, arm)


def test_tampered_selected_box_distance_is_rejected():
    row = copy.deepcopy(real_rows('full')[0])
    row['distance'] += 3
    with pytest.raises(compare.EvidenceError, match='distance'):
        compare.validate_candidates(row, 'full', 'full')


@pytest.mark.parametrize('arm', ['b1_b2', 'full'])
def test_all_empty_q0_alias_is_legal_only_for_shadow_output(arm):
    for row in real_rows(arm, empty=True):
        record = compare.validate_candidates(row, arm, arm)
        assert record['q0_is_fallback'] and not record['valid'][0]
        assert record['selected_index'] == (0 if arm == 'b1_b2' else 1)
        assert not row['memory_write'] and not row['supported']


def test_tampered_commit_count_cannot_borrow_scoring_support():
    row = copy.deepcopy(real_rows('full')[0])
    row['geometric_fg_count'] += 3
    with pytest.raises(compare.EvidenceError, match='commit own-box'):
        compare.validate_candidates(row, 'full', 'full')


def test_cli_missing_evidence_never_writes_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, 'compare_runs', lambda **kwargs: compare.require(False, 'missing evidence'))
    args = []
    for name in ('registered-reference', 'baseline', 'old-b0', 'old-w-quarter', 'strong-b0', 'strong-b1',
                 'b0', 'b1', 'b1-b2', 'full', 'full-bf16'):
        args += ['--' + name, str(tmp_path / name)]
    assert compare.main(args) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'invalid_evidence'
    assert not list(tmp_path.iterdir())
