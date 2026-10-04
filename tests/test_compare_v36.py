"""v36 比较器反例：不混用 output/measurement，不以控制组替换主模型。"""
from copy import deepcopy
import json

import pytest
import torch

from models.ct_v36.candidate_records import candidate_records
from models.ct_v36.sampler import ReadyQueueBatchSampler
from tests.test_ct_v34_identity import MINI_TRAIN_LENGTHS
from tests.test_ct_v36_diagnostics import logged_output
from tools import compare_v36 as compare


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def record_row(*, wrong=False):
    batch, output = logged_output(wrong=wrong)
    record = candidate_records(batch, output)[0]
    action = record['action']
    selected, measured = action['output_index'], action['measurement_index']
    candidate = record['candidates'][measured]
    own = candidate['support_raw_ids']
    written = own[:8]
    pool = record['support_pool']
    labels = {raw: label for raw, label in zip(pool['raw_ids'], pool['target_labels'])}
    true_fg = sum(labels[raw] > .5 for raw in written)
    return dict(diagnostic_candidates=record, **{key: action[key] for key in
        ('output_index', 'measurement_index', 'measurement_accepted', 'output_measurement_disagreement')},
        selected_index=selected, selected_quality=record['candidates'][selected]['utility'],
        distance=record['candidates'][selected]['geometry']['center_error_m'],
        measurement_world=candidate['box_world'], measurement_fg_ids=own,
        memory_write=True, memory_write_eligible=True, memory_world=candidate['box_world'],
        memory_source_index=measured, memory_written_fg_ids=written, memory_fg_count=len(written),
        memory_true_fg_count=true_fg, memory_wrong_write=true_fg / len(written) < .5)


@pytest.mark.parametrize('wrong', [False, True])
def test_output_prior_still_commits_independent_highest_score_measurement(wrong):
    row = record_row(wrong=wrong)
    record = compare.validate_candidates(row, 'fixture', 'A')
    assert record['action']['output_index'] == 1
    assert record['action']['measurement_index'] == (2 if wrong else 0)
    assert record['action']['measurement_correct'] is (not wrong)


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r.update(measurement_index=1), 'independent highest'),
    (lambda r: r.update(memory_source_index=1), 'memory source'),
    (lambda r: r.update(memory_world=[0., 0., 0., 0.]), 'own box'),
    (lambda r: r.update(memory_written_fg_ids=[987654], memory_fg_count=1), 'measurement subset'),
    (lambda r: r['diagnostic_candidates']['candidates'][0].update(support_raw_ids=[0, 0]), 'unique'),
    (lambda r: r['diagnostic_candidates']['candidates'][0].update(measurement_score=.99), 'measurement_score'),
    (lambda r: r['diagnostic_candidates']['action'].update(measurement_correct=False), 'truth flags'),
])
def test_reject_tampered_action_and_memory(mutation, match):
    row = record_row()
    mutation(row)
    with pytest.raises(compare.EvidenceError, match=match):
        compare.validate_candidates(row, 'fixture', 'A')


def test_own_box_support_is_recomputed_not_merely_declared():
    row = record_row()
    candidate = row['diagnostic_candidates']['candidates'][0]
    raw = candidate['support_raw_ids'][0]
    pool = row['diagnostic_candidates']['support_pool']
    for i, identity in enumerate(pool['raw_ids']):
        if identity == raw:
            pool['xyz_anchor_relative'][i] = [500., 500., 500.]
    with pytest.raises(compare.EvidenceError, match='own-box predicted FG'):
        compare.validate_candidates(row, 'fixture', 'A')


def sampler(arm='A', epoch=11):
    value = ReadyQueueBatchSampler(MINI_TRAIN_LENGTHS, source_sha256='a' * 64,
        annotation_sha256='b' * 64, continuous=arm != 'F')
    value.set_epoch(epoch - 1)
    return value.state_dict()


def test_reconstruct_continuous_and_reset_plans_preserving_endpoint_order():
    a, f = sampler('A'), sampler('F')
    assert a['plan_sha256'] != f['plan_sha256']
    assert a['endpoint_order_sha256'] == f['endpoint_order_sha256']
    for arm, value in [('A', a), ('F', f)]:
        assert compare.validate_sampler(value, {'v36_arm': arm}, 11, arm) == 'a' * 64


def test_forged_sampler_checksum_cannot_hide_wrong_plan():
    value = sampler()
    value['plan_sha256'] = 'c' * 64
    value['manifest_sha256'] = compare.original.digest({k: v for k, v in value.items() if k != 'manifest_sha256'})
    with pytest.raises(compare.EvidenceError, match='does not reconstruct'):
        compare.validate_sampler(value, {'v36_arm': 'A'}, 11, 'A')


def test_sampler_rejects_wrong_arm_even_when_original_checksum_valid():
    with pytest.raises(compare.EvidenceError, match='continuous identity'):
        compare.validate_sampler(sampler('F'), {'v36_arm': 'A'}, 11, 'A')


def summary_fixture(monkeypatch, *, a_passed=False):
    runs = {label: dict(manifest=dict(run_id=label), annotation_sha256='a' * 64,
        endpoint_order={e: str(e) for e in range(1, 61)}, rows={60: {}},
        mechanism_diagnostics={}, measurement_followups={}, validation_peaks={})
        for label in (*compare.LABELS, 'baseline', 'old_w_quarter')}
    def loaded(*args, **kwargs):
        assert kwargs['candidate_labels'] == compare.LABELS
        assert kwargs['config_differences'] == ('v36_arm',)
        return dict(candidates={a: dict(passed=a_passed if a == 'A' else True) for a in compare.LABELS},
            status='passed', selected='F', passing_rank=['F', 'C'], criterion={}, limitations=[])
    monkeypatch.setattr(compare.base, 'compare_loaded', loaded)
    monkeypatch.setattr(compare.original, 'tail_diagnostics', lambda r: {})
    monkeypatch.setattr(compare.original, 'explanatory_slices', lambda *args: {})
    monkeypatch.setattr(compare.original, 'tail_counts', lambda r: dict(final60=0, late3=0))
    return runs, dict(source=dict(sha256='c' * 64), reused={})


def test_control_winner_never_replaces_failed_A(monkeypatch):
    runs, registration = summary_fixture(monkeypatch)
    result = compare.summarize_loaded(runs, registration)
    assert result['status'] == 'failed' and result['selected'] is None
    assert result['registered_main_model'] == 'A' and 'passing_rank' not in result


def test_A_passes_independently_from_control_ranking(monkeypatch):
    runs, registration = summary_fixture(monkeypatch, a_passed=True)
    result = compare.summarize_loaded(runs, registration)
    assert result['status'] == 'passed' and result['selected'] == 'A'


def test_comparison_rejects_different_endpoint_order(monkeypatch):
    runs, registration = summary_fixture(monkeypatch)
    runs['F']['endpoint_order'][12] = 'forged'
    with pytest.raises(compare.EvidenceError, match='endpoint order'):
        compare.summarize_loaded(runs, registration)


def test_validation_peak_preserves_paired_metric_and_does_not_choose_checkpoint(tmp_path):
    directory = tmp_path / 'validation'
    directory.mkdir()
    for epoch in range(5, 61, 5):
        score = dict(schema=compare.SCHEMA, checkpoint_epoch=epoch, metric_mode='benchmark_compat',
            **compare.original.COUNTS, complete_coverage=True,
            success=90 if epoch == 10 else 30, precision=95 if epoch == 15 else 40)
        (directory / f'epoch={epoch:03d}.json').write_text(json.dumps(score), encoding='utf-8')
    result = compare.validation_peaks(tmp_path, lambda relative: compare.original.read_object(tmp_path / relative),
        {'check_val_every_n_epoch': 5})
    assert result['best_success'] == dict(epoch=10, success=90, precision=40)
    assert result['best_precision'] == dict(epoch=15, success=30, precision=95)
    assert result['formal_gate'] is False
    (directory / 'epoch=025.json').unlink()
    with pytest.raises(compare.EvidenceError, match='coverage'):
        compare.validation_peaks(tmp_path, lambda relative: None, {'check_val_every_n_epoch': 5})


def test_historical_frames_are_pinned_even_with_same_manifest():
    run = dict(manifest=dict(config_sha256='a', source={'sha256': 'b'}),
        provenance={f'evaluation/epoch={e:03d}/frames.jsonl': str(e) for e in compare.original.EPOCHS})
    pin = dict(config_sha256='a', manifest_source={'sha256': 'b'},
        frames_sha256={str(e): str(e) for e in compare.original.EPOCHS})
    compare._pin_reused(run, pin, 'strong_b0')
    run['provenance']['evaluation/epoch=060/frames.jsonl'] = 'different run'
    with pytest.raises(compare.EvidenceError, match='fingerprints'):
        compare._pin_reused(run, pin, 'strong_b0')


def test_cli_missing_evidence_returns_two_without_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, 'compare_runs', lambda **kwargs: compare.require(False, 'missing evidence'))
    args = []
    for name in ('registered-reference', 'baseline', 'old-b0', 'old-w-quarter', 'strong-b0', 'strong-b1', 'a', 'c', 'd', 'e', 'f'):
        args += ['--' + name, str(tmp_path / name)]
    assert compare.main(args) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'invalid_evidence'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('arm', compare.LABELS)
def test_real_host_frame_records_pass_reader_contract(arm):
    from models.ct_v36.data import build_loaders
    from models.ctseqtrackv36 import CTSEQTRACKV36
    from tests.test_ct_v36_runtime import config
    from tests.test_ct_v31_runtime import TinySource
    cfg = config(v36_arm=arm)
    loaders = build_loaders(cfg, roles=('test',), sources={'test': TinySource(lengths=(3, 2))})
    host = CTSEQTRACKV36(cfg, loaders=loaders).eval()
    host.log = lambda *args, **kwargs: None
    host.on_test_epoch_start()
    with torch.no_grad():
        for index, rows in enumerate(host._loader('test')):
            host.test_step(rows, index)
    host.on_test_epoch_end()
    for row in host.evaluation.rows:
        if not row['initialization']:
            compare.base._check_diagnostics(row, arm)
            compare.validate_context(row, arm)
            compare.validate_candidates(row, arm, arm)
