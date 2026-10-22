"""v38 配置、源码、精度、run 与完整 W epoch 的恢复身份。"""
from copy import deepcopy

import pytest

from models.ct_v38.config import normalize_config
from models.ct_v38.data import build_loaders
from models.ct_v38.runtime import resume_payload, validate_resume_payload, loss_episode_summary
from tests.test_ct_v31_runtime import TinySource


def config(**updates):
    value = dict(v38_arm='b1', ct_engineering_check=True, workers=0, point_sample_size=8,
                 batch_size=4, v32_reserve_windows=0, epoch=11)
    value.update(updates)
    return normalize_config(value)


def payload_for(cfg):
    sampler = build_loaders(cfg, roles=('train',), sources={'train': TinySource((3, 2))})['train'].batch_sampler
    sampler.set_epoch(9)
    return resume_payload(cfg, completed_epoch=10, complete=True, rows=sampler.row_count,
                          steps=len(sampler), sampler=sampler.state_dict(), run_id='v38-unit-run')


def test_exact_complete_epoch_resume_and_eval_identity():
    cfg = config(); payload = payload_for(cfg)
    assert validate_resume_payload(payload, cfg) is payload
    assert validate_resume_payload(payload, cfg, training=False) is payload
    assert payload['sampler']['schema'] == 'ct_seqtrack.v32.ready_queue.v2'
    assert payload['sampler']['short_window'] == 4


@pytest.mark.parametrize('field,value,message', [
    ('run_id', '', 'run_id'), ('source', {}, 'source identity'),
    ('epoch_complete', False, 'complete epoch'), ('optimizer_steps', 999, 'counters'),
    ('config_sha256', 'wrong', 'identity'), ('schema', 'ct_seqtrack.v36', 'schema')])
def test_restore_rejects_wrong_identity_or_incomplete_transaction(field, value, message):
    cfg = config(); payload = payload_for(cfg); payload[field] = value
    with pytest.raises(ValueError, match=message):
        validate_resume_payload(payload, cfg)


def test_sampler_checksum_and_cross_arm_precision_are_not_bypassed():
    cfg = config(); payload = payload_for(cfg)
    changed = deepcopy(payload); changed['sampler']['rows'] += 1
    with pytest.raises(ValueError, match='checksum'):
        validate_resume_payload(changed, cfg)
    for other in (config(v38_arm='full'), config(v38_training_recipe='b1_hold')):
        with pytest.raises(ValueError, match='identity'):
            validate_resume_payload(payload, other)


def test_loss_recovery_keeps_intermediate_frames_and_right_censoring():
    rows = [dict(tracklet='t', frame=i, iou=v, timestamp=float(i))
            for i, v in enumerate((.8, .05, .3, .6, .01, .2))]
    result = loss_episode_summary(rows)
    assert result['total_lost_frames'] == 4
    assert result['recovered']['episodes'] == result['unrecovered']['episodes'] == 1
