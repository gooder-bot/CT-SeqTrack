"""三候选验收拒绝无证据写入、别名与选择政策漂移。"""
from copy import deepcopy

import pytest

from tools.compare_v38 import EvidenceError, validate_candidates, reliability_plot_data, write_reliability_plot


def record(*, arm='full', selected=0, real=True):
    geometry = [dict(exact_iou=.8, success=.825, precision=.925) for _ in range(3)]
    valid = [real, True, True]
    utility = [.7, .6, .5] if arm == 'full' else None
    if arm == 'full' and selected != 0:
        utility[selected] = .8
    eligible = real and selected == 0
    candidate = dict(schema='ct_seqtrack.v38.candidate_records.v1',
        names=['q0', 'prior', 'trusted_propagation'],
        boxes_anchor_relative=[[0., 0., 0., 0.], [1., 0., 0., 0.], [2., 0., 0., 0.]],
        valid=valid, raw_valid=valid, alias_of=[-1 if real else 1, -1, -1],
        q0_is_fallback=not real, selected_index=selected,
        selected_source=['q0', 'prior', 'trusted_propagation'][selected],
        utility=utility, selected_utility=utility[selected] if utility else None,
        support=dict(raw_ids=[1, 2, 3], count=3, target_count=3, target_purity=1.),
        geometry=geometry, trusted_valid=real, trusted_target=True if real else None,
        trusted_score=.8 if arm == 'full' and real else None, legacy_q0_quality=.8,
        trusted_eligible=eligible)
    return dict(diagnostic_candidates=candidate, selected_index=selected,
        trusted_write=eligible, supported=eligible, geometric_fg_count=3 if arm == 'b1' or selected == 0 else 0,
        memory_write=False, success=.825, precision=.925)


@pytest.mark.parametrize('arm,selected,real', [('full', 0, True), ('full', 1, True),
    ('full', 2, True), ('b1', 0, True), ('b1', 1, False)])
def test_valid_source_and_permission_cases(arm, selected, real):
    validate_candidates(record(arm=arm, selected=selected, real=real), 'test', arm)


def test_prior_cannot_write_even_with_high_output_utility():
    row = record(selected=1)
    row['trusted_write'] = True
    with pytest.raises(EvidenceError, match='trusted commit'):
        validate_candidates(row, 'test', 'full')


def test_wrong_rank_duplicate_raw_ids_and_memory_write_rejected():
    for field in ('rank', 'ids', 'memory'):
        row = record()
        if field == 'rank':
            row['diagnostic_candidates']['utility'] = [.5, .8, .4]
        elif field == 'ids':
            row['diagnostic_candidates']['support']['raw_ids'] = [1, 1, 3]
        else:
            row['memory_write'] = True
        with pytest.raises(EvidenceError):
            validate_candidates(row, field, 'full')


def test_duplicate_propagation_and_zero_gain_tie_are_deterministic():
    row = record()
    c = row['diagnostic_candidates']
    c['boxes_anchor_relative'][2] = list(c['boxes_anchor_relative'][1])
    c['valid'] = [True, True, False]
    c['raw_valid'] = [True, True, True]
    c['alias_of'][2] = 1
    c['utility'] = [.7, .7, .99]
    validate_candidates(row, 'same', 'full')
    c['alias_of'][2] = -1
    with pytest.raises(EvidenceError, match='dedup'):
        validate_candidates(row, 'bad_alias', 'full')


def test_b1_preserves_actual_world_geometry_commit_at_canonical_boundary():
    row = record(arm='b1')
    row.update(geometric_fg_count=2, trusted_write=False, supported=False)
    validate_candidates(row, 'b1_boundary', 'b1')
    row['trusted_write'] = True
    with pytest.raises(EvidenceError, match='trusted commit'):
        validate_candidates(row, 'false_b1_write', 'b1')


def test_dedup_reconstructs_fp32_subtraction_at_fixed_threshold():
    import torch
    row = record()
    candidate = row['diagnostic_candidates']
    candidate['boxes_anchor_relative'][0][0] = -float(torch.tensor(1e-4))
    candidate['boxes_anchor_relative'][1][0] = float(torch.tensor(3e-12))
    candidate['valid'] = [True, False, True]
    candidate['raw_valid'] = [True, True, True]
    candidate['alias_of'] = [-1, 0, -1]
    validate_candidates(row, 'fp32_dedup_boundary', 'full')


def reliability_fixture():
    from models.ct_v38.runtime import reliability_summary
    records = [dict(trusted_score=.15, trusted_valid=True, trusted_target=False) for _ in range(10)]
    records += [dict(trusted_score=.55, trusted_valid=True, trusted_target=(i < 2)) for i in range(4)]
    records += [dict(trusted_score=.85, trusted_valid=True, trusted_target=True) for _ in range(5)]
    return dict(schema='ct_seqtrack.v38.comparison.v1', status='failed', synthetic_fixture=True,
        mechanism_diagnostics=dict(full_standard={str(epoch): dict(trusted_reliability=reliability_summary(
            records if epoch != 59 else [])) for epoch in (58, 59, 60)}))


def test_reliability_keeps_empty_bins_missing_and_rejects_fabricated_zero():
    report = reliability_fixture()
    data = reliability_plot_data(report)
    assert data['epochs']['59']['brier'] is None
    assert data['epochs']['58']['bins'][0]['mean_probability'] is None
    report['mechanism_diagnostics']['full_standard']['58']['trusted_reliability']['bins'][0]['event_frequency'] = 0.
    with pytest.raises(EvidenceError, match='empty bins'):
        reliability_plot_data(report)


def test_reliability_figure_and_source_data_are_new_artifacts_only(tmp_path, monkeypatch):
    import json
    from tools import compare_v38
    monkeypatch.setattr(compare_v38, 'ROOT', tmp_path)
    destination = tmp_path / 'artifacts/ct_checks/reliability.png'
    result = write_reliability_plot(destination, reliability_fixture())
    assert destination.read_bytes().startswith(b'\x89PNG')
    data = json.loads(destination.with_suffix('.data.json').read_text(encoding='utf-8'))
    assert data['synthetic_fixture'] is True
    assert data['epochs']['59']['count'] == 0
    assert result['source_data'] == str(destination.with_suffix('.data.json'))
    with pytest.raises(EvidenceError, match='already exists'):
        write_reliability_plot(destination, reliability_fixture())
    with pytest.raises(EvidenceError, match='artifacts/ct_checks'):
        write_reliability_plot(tmp_path / 'outside.png', reliability_fixture())


@pytest.mark.parametrize('arm,empty', [('b1', False), ('full', False), ('b1', True), ('full', True)])
def test_real_host_candidate_output_passes_independent_comparator(arm, empty):
    import numpy as np
    import torch
    from models.ct_v38.data import build_loaders
    from models.ctseqtrackv38 import CTSEQTRACKV38
    from tests.test_ct_v38_runtime import config
    from tests.test_ct_v31_runtime import TinySource
    class EmptySource(TinySource):
        def get_frames(self, index, frame_ids):
            frames = super().get_frames(index, frame_ids)
            for frame in frames:
                frame['pc'], frame['point_ids'] = np.empty((0, 3)), np.empty(0, np.int64)
            return frames
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cfg = config(v38_arm=arm)
        source = (EmptySource if empty else TinySource)(lengths=(3, 2))
        host = CTSEQTRACKV38(cfg, loaders=build_loaders(cfg, roles=('test',), sources={'test': source})).eval()
        host.log = lambda *args, **kwargs: None
        host.on_test_epoch_start()
        with torch.no_grad():
            for index, rows in enumerate(host._loader('test')):
                host.test_step(rows, index)
        host.on_test_epoch_end()
        for row in host.evaluation.rows:
            if not row['initialization']:
                validate_candidates(row, 'real_host', arm)
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)
