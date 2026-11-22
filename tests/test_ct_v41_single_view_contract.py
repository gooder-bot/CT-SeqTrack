import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from models.ct_v41.auxiliary import FullTrackAuxiliarySampler, insertion_indices, insertion_identity
from models.ct_v41.config import config_identity, normalize_config, recipe_identity
from models.ct_v41.data import KNOWN_GT_BOX, PERTURBED_BOX, PREDICTED_BOX, RawEndpointDataset, ReadyQueueBatchSampler
from models.ct_v41.fixed_replay import FixedReplayBank
from models.ct_v41.history_training import HistoryTrainingAudit
from models.ct_v41.runtime import resume_payload, validate_resume_payload
from models.ctseqtrackv41 import CTSEQTRACKV41
from tests.test_ct_v41_fixed_replay import TrainKITTI
from tests.test_ct_v41_worker_transport import kitti


def test_single_view_freezes_its_own_e2_and_rejects_paired_catalog(tmp_path, kitti, monkeypatch, request):
    root = kitti.KITTI_velo
    monkeypatch.setattr('datasets.kitti_v41.get_v41_raw_dataset', lambda cfg, role: TrainKITTI(root, role))
    dataset = RawEndpointDataset(TrainKITTI(root), training=True, joint_search=True, enable_m=True)
    cfg = normalize_config(dict(v41_joint_search=True, v41_history_training='window_1448',
        v41_fixed_replay_bank=str(tmp_path / 'pair.sqlite'), ct_engineering_check=True,
        batch_size=1, workers=0, v41_attention_backend='math'))
    host = CTSEQTRACKV41(cfg)
    host._completed_epoch = 2
    def open_bank(path):
        bank = FixedReplayBank(path, cfg.seed)
        request.addfinalizer(bank.close)
        bank.ensure_catalog(host, dataset)
        return bank
    paired = open_bank(cfg.v41_fixed_replay_bank)
    host.config.v41_history_training = 'four_candidates'
    assert open_bank(paired.path).manifest_sha256 == paired.manifest_sha256
    host.config.v41_history_training = 'single_view_cgp'
    with pytest.raises(ValueError, match='mismatch'):
        open_bank(paired.path)
    single = open_bank(tmp_path / 'single.sqlite')
    assert single.recipe_sha256 == config_identity(host.config) != paired.recipe_sha256
    assert single.donor_weights_sha256 == paired.donor_weights_sha256
    assert single.manifest_sha256 != paired.manifest_sha256
    assert recipe_identity(host.config)['replay'] == 'own_e2_natural_same_endpoint_branch0_v1'
    assert open_bank(single.path).manifest_sha256 == single.manifest_sha256


def test_single_view_checkpoint_binds_physical_coverage_and_sampler_mode():
    cfg = normalize_config(dict(v41_history_training='single_view_cgp', v41_joint_search=True,
        ct_engineering_check=True, batch_size=2, workers=0, v41_attention_backend='math'))
    lengths = [5] * 8
    train = ReadyQueueBatchSampler(lengths, cfg.batch_size, seed=cfg.seed, single_view_cgp=True,
        source_sha256='training', initial_seed_policy=cfg.v35_seed_policy).state_dict()
    aux = FullTrackAuxiliarySampler(lengths, cfg.batch_size, seed=cfg.seed, source_sha256='training').state_dict()
    indices = insertion_indices(train['batches'], aux['batches'])
    audit = dict(schema='ct_seqtrack.v41.auxiliary_audit.v1', sampler=aux, complete=True,
        rows=aux['rows'], batches=aux['batches'], main_batches=train['batches'],
        loss_weight=cfg.v41_aux_loss_weight, insertion_indices=list(indices),
        insertion_plan_sha256=insertion_identity(indices))
    payload = resume_payload(cfg, completed_epoch=1, complete=True, rows=train['rows'],
        steps=train['batches'], sampler=train, run_id='single', auxiliary=audit)
    payload.update(history_bank_sha256='a' * 64, joint_replay={
        'schema': 'ct_seqtrack.fixed_joint_e2.v1', 'completed_epoch': 1,
        'manifest_sha256': None, 'donor_weights_sha256': None})
    validate_resume_payload(payload, cfg)
    for changes, message in (({'rows': train['rows'] * 4}, 'once per epoch'),
                             ({'history_training': None}, 'view count')):
        tampered = dict(train, **changes)
        content = {k: v for k, v in tampered.items() if k != 'manifest_sha256'}
        tampered['manifest_sha256'] = hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with pytest.raises(ValueError, match=message):
            validate_resume_payload(dict(payload, sampler=tampered, rows=tampered['rows']), cfg)


def test_single_view_audit_counts_physical_seed_and_excludes_natural_replay():
    row = dict(request=SimpleNamespace(branch=0, starts_window=True, frame=2))
    first = dict(request=SimpleNamespace(branch=0, starts_window=True, frame=1))
    rows = [first, row, row, dict(row, _joint_replay_state=object())]
    provenance = torch.tensor([[KNOWN_GT_BOX], [KNOWN_GT_BOX], [PERTURBED_BOX], [PREDICTED_BOX]])
    batch = dict(history_box_provenance=provenance, joint_motion_boxes=torch.zeros(4, 3, 4),
        joint_motion_times=torch.zeros(4, 3), joint_motion_pairs=torch.zeros(4, 2, dtype=torch.bool),
        joint_motion_valid=torch.zeros(4, 3, dtype=torch.bool), current_dt=torch.ones(4))
    audit = HistoryTrainingAudit(epoch=3)
    audit.prepared(rows, batch, 'main')
    report = audit.report(None, SimpleNamespace(history_counts={}))
    assert report['ordinary_reset_seeds'] == {'first_prediction_gt': 1, 'clean_seed': 1, 'perturbed_seed': 1}
    assert report['endpoints'] == {'reset': 3, 'natural_replay': 1}
