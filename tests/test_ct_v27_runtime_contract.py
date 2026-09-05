import pytest
import torch

from utils.v27_protocol import build_scene_manifest
from utils.action_calibration_v27 import validate_scene_manifest
from utils.v27_quality import observation_quality
from utils.online_contract import build_online_resume_contract


def test_quality_counts_only_unique_observation_predictions_and_empty_rows():
    logits = torch.zeros(2, 2, 8, requires_grad=True)
    with torch.no_grad():
        logits[0, :, -2] = torch.tensor([0., 2.])
        logits[0, :, -1] = torch.tensor([20., -20.])
    unique = torch.zeros(2, 4, 2)
    unique[0, -1, 0] = 1
    counts = torch.zeros(2, 4)
    counts[0, -1] = 1
    quality = observation_quality(logits, unique, counts)
    assert not quality.requires_grad
    assert quality[0, 0] == 1 and quality[0, 3] == 1
    torch.testing.assert_close(quality[0, 1], torch.sigmoid(torch.tensor(2.)))
    torch.testing.assert_close(quality[1], torch.zeros(4))


def test_unknown_nuscenes_version_cannot_be_silently_treated_as_full():
    splits = {'train_track': [f't{i}' for i in range(350)], 'val': [f'v{i}' for i in range(150)]}
    with pytest.raises(ValueError, match='supports only'):
        build_scene_manifest(splits, 'v1.0-test')
    manifest = build_scene_manifest(splits, 'v1.0-trainval')
    manifest['version'] = 'v1.0-test'
    with pytest.raises(ValueError, match='supports only'):
        validate_scene_manifest(manifest)


def test_resume_identity_records_v27_scene_and_metric_contract():
    config = dict(ct_enable_v27=True, ct_scene_manifest_sha256='350train150val',
                  ct_metric_mode='benchmark_compat', ct_runtime_protocol='safe_seqtrack_auto_v1')
    contract = build_online_resume_contract(config)
    fields = contract.get('fields', contract.get('identity', contract))
    assert fields['enable_v27'] is True
    assert fields['scene_manifest_sha256'] == '350train150val'
    assert fields['metric_mode'] == 'benchmark_compat'


def test_host_multiple_mechanism_ticks_keep_order_and_endpoint_weighting():
    from tests.test_ct_v27_isolation import TransactionHost

    class Host(TransactionHost):
        def training_step(self, batch, batch_idx):
            if isinstance(batch, dict) and 'ct_stream_schema' in batch:
                return super().training_step(batch, batch_idx)
            if isinstance(batch, list):
                self.visited.extend(row['frame'] for row in batch)
                return self.plugin_parameter * batch[0]['gain']
            return self.observation_parameter * 7

    host = Host('full').train()
    host._ct_optimizer_names = ['b0', 'b1', 'b2', 'b3']
    host.observation_parameter = torch.nn.Parameter(torch.tensor(1.))
    host.plugin_parameter = torch.nn.Parameter(torch.tensor(1.))
    host.visited = []
    loss = host.training_step(dict(ct_stream_schema='ct_seqtrack.train.v4',
        observation={}, mechanism={'ct_mechanism_sequence': [
            [dict(frame=1, gain=1.), dict(frame=2, gain=1.)],
            [dict(frame=3, gain=4.)]]}), 0)
    loss.backward()
    assert host.visited == [1, 2, 3]
    assert host.observation_parameter.grad == 7
    assert host.plugin_parameter.grad == 2  # (2*1 + 1*4) / 3
