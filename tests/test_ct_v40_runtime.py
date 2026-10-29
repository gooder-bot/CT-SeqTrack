"""V40 恢复身份、被动诊断与会话日志合同；仅编写，待服务器执行。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from models.ct_v40.config import normalize_config, config_identity
from models.ct_v40.data import ReadyQueueBatchSampler, BatchBuilder, RawEndpointDataset
from models.ct_v40.runtime import resume_payload, validate_resume_payload, TrackingEvaluation


def config(arm='b1', **updates):
    values = dict(v40_arm=arm, ct_engineering_check=True, workers=0, batch_size=4,
                  v32_reserve_windows=0, epoch=11, v40_attention_backend='math')
    values.update(updates)
    return normalize_config(values)


def payload_for(cfg, monkeypatch):
    import models.ct_v40.entry as entry
    import models.ct_v40.runtime as runtime
    source = dict(sha256='unit-source', files={}, inventory_policy='unit')
    monkeypatch.setattr(entry, 'source_identity', lambda: source)
    monkeypatch.setattr(runtime, 'capture_rng_state', lambda: {'unit_rng_marker': True})
    sampler = ReadyQueueBatchSampler((3, 2), batch_size=4, seed=42, training=True,
        short_window=4, long_window=8, curriculum_epochs=10, reserve_windows=0,
        source_sha256='unit-dataset', initial_seed_policy=ReadyQueueBatchSampler.INITIAL_SEED_POLICY)
    sampler.set_epoch(9)
    return resume_payload(cfg, completed_epoch=10, complete=True, rows=sampler.row_count,
                          steps=len(sampler), sampler=sampler.state_dict(), run_id='v40-unit-run')


def test_same_run_complete_boundary_can_evaluate_without_future_labels(monkeypatch):
    cfg = config('b1_mtp')
    payload = payload_for(cfg, monkeypatch)
    assert validate_resume_payload(payload, cfg) is payload
    evaluation = config('b1_mtp', test=True, checkpoint='epoch=010.ckpt', eval_checkpoint_epoch=10)
    assert config_identity(evaluation) == config_identity(cfg)
    assert validate_resume_payload(payload, evaluation, training=False) is payload
    assert payload['completed_epoch'] == payload['sampler']['epoch'] + 1


@pytest.mark.parametrize('field,value,message', [
    ('epoch_complete', False, 'complete epoch'), ('completed_epoch', 9, 'counters'),
    ('optimizer_steps', 999, 'counters'), ('rows', 999, 'counters'),
    ('run_id', '', 'run_id'), ('source', {}, 'source identity'),
    ('schema', 'ct_seqtrack.v38.epoch_boundary.v1', 'schema'),
    ('rng', None, 'RNG')])
def test_incomplete_or_mismatched_resume_is_rejected(monkeypatch, field, value, message):
    cfg = config()
    payload = payload_for(cfg, monkeypatch)
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        validate_resume_payload(payload, cfg)


def test_sampler_checksum_and_cross_arm_m_identity_are_not_bypassed(monkeypatch):
    cfg = config()
    payload = payload_for(cfg, monkeypatch)
    changed = deepcopy(payload)
    changed['sampler']['rows'] += 1
    with pytest.raises(ValueError, match='checksum'):
        validate_resume_payload(changed, cfg)
    for arm in ('b0', 'b1_mtp', 'full'):
        with pytest.raises(ValueError, match='identity'):
            validate_resume_payload(payload, config(arm))


def _complete_output(batch, *, delta=0., trusted=.9):
    from tests.test_ct_v40_data import output_for
    out = output_for(batch, delta=delta, trusted=trusted)
    out.observation.coarse_box = out.accepted_box.clone()
    out.observation.sequence_valid = batch['point_valid'].flatten(1).any(-1)
    out.observation.segmentation_logits = batch['points'].new_zeros((*batch['point_valid'].shape, 2))
    out.observation.bc_prediction = batch['bc_targets'].clone()
    out.decoder = SimpleNamespace(q0_box=out.accepted_box.clone(),
                                  history_boxes=batch['history_target_boxes'].clone())
    out.prior.valid = torch.zeros(len(batch['points']), dtype=torch.bool)
    out.candidates.raw_valid = torch.ones_like(out.candidates.valid)
    out.candidates.alias_of = torch.full_like(out.candidates.valid, -1, dtype=torch.long)
    out.commit_decision.legacy_q0_quality = out.legacy_q0_quality
    out.verification = None
    return out


def test_b0_training_diagnostics_do_not_require_acquisition_or_m_labels():
    from tests.test_ct_v40_data import MetadataSource, request
    from models.ct_v40.training_diagnostics import TrainingDiagnostics
    builder = BatchBuilder(config('b0'))
    row = RawEndpointDataset(MetadataSource())[request(1, branch=3)]
    batch = builder.prepare([row])
    builder.supervise_prior(batch, None, training=True)
    output = _complete_output(batch)
    output.candidates.raw_valid[:, 1] = False
    output.candidates.valid[:, 1] = False
    diagnostics = TrainingDiagnostics(seed=42, optimized=True)
    diagnostics.reset(9)
    diagnostics.add_batch([row], batch, output, {'loss_total': torch.zeros(())})
    result = diagnostics.summary()
    assert result['totals']['endpoints'] == 1
    assert 'acquisition_demand_endpoints' not in result['batch_loss_valid_counts']
    assert not any(name.startswith('future_') for name in batch)


def test_search_coverage_flows_to_rows_and_summary_without_causal_claim():
    from tests.test_ct_v40_data import MetadataSource, request
    cfg = config('full')
    raw, builder = RawEndpointDataset(MetadataSource()), BatchBuilder(cfg)
    evaluation = TrackingEvaluation(cfg)
    for frame, delta, trusted in ((1, 20., .1), (2, 0., .9)):
        row = raw[request(frame)]
        batch = builder.prepare([row])
        output = _complete_output(batch, delta=delta, trusted=trusted)
        committed = builder.commit(output, batch, diagnostics=True)
        evaluation.add_batch([row], batch, output, committed)
    last = evaluation.rows[-1]
    assert last['search_reason'] == 'accepted_untrusted'
    assert last['diagnostic_search_backup_active']
    assert last['diagnostic_search_main_target_count'] == 0
    assert last['diagnostic_search_sampled_backup_target_count'] >= 3
    summary = evaluation.summary()['search']
    assert summary['triggered_frames'] == 1
    assert summary['backup_added_target_frames'] == 1
    assert summary['correct_trusted_writes_with_backup_points'] == 1
    assert 'not proof of causal' in summary['interpretation']


def test_replaying_an_epoch_preserves_previous_stream_in_new_session(tmp_path):
    from models.ctseqtrackv40 import CTSEQTRACKV40
    # 注入小 CPU tracker，仅检验host文件会话；不构造/运行FA或网络。
    cfg = normalize_config(dict(v40_arm='b1', log_dir=str(tmp_path)))
    first = CTSEQTRACKV40(cfg, tracker=torch.nn.Linear(1, 1))
    first._completed_epoch = 10
    first._evaluation_start('test')
    first.teardown('test')
    paths = list(tmp_path.glob('frame_streams/test/**/epoch=010.jsonl'))
    assert len(paths) == 1
    paths[0].write_text('preserved partial prior session\n', encoding='utf-8')
    second = CTSEQTRACKV40(cfg, tracker=torch.nn.Linear(1, 1))
    second._completed_epoch = 10
    second._evaluation_start('test')
    second.teardown('test')
    assert len(list(tmp_path.glob('frame_streams/test/**/epoch=010.jsonl'))) == 2
    assert paths[0].read_text(encoding='utf-8') == 'preserved partial prior session\n'
