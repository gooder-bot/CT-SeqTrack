"""真实导出hook的元信息修复，不引入Lightning/nuScenes或改变前向输入。"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Subset


def production_functions():
    tree = ast.parse(Path('models/base_model.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseModelMF')
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == '_ct_v28_export_metadata']
    functions += [n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name in ('validation_step', 'test_step')]
    scope = dict(np=np, torch=torch, time=SimpleNamespace(time=lambda: 0.))
    exec(compile(ast.Module(body=copy.deepcopy(functions), type_ignores=[]),
                 'production_export_hooks', 'exec'), scope)
    return scope


def source_dataset(role='test', mini=False):
    names = ['scene-train', 'scene-val-a', 'scene-val-b']
    source = SimpleNamespace(
        ct_scene_role=role,
        virtual_rate_meta=[dict(tracklet_key=f'track/{i}', scene_token=f'token/{i}')
                           for i in range(3)],
        ct_scene_manifest=dict(
            training_source='mini_train' if mini else 'train_track',
            evaluation_source='mini_val' if mini else 'val',
            parameter_training_overlap=True,
            scenes=dict(train=names[:1], calibration=names[:1], dev=[], test=names[1:])),
    )
    source.get_tracklet_key = lambda index: source.virtual_rate_meta[index]['tracklet_key']
    source.nusc = SimpleNamespace(get=lambda kind, token: {'name': names[int(token.split('/')[1])]})
    return source


@pytest.mark.parametrize('mini', [False, True])
@pytest.mark.parametrize('as_list', [False, True])
def test_real_source_mapping_official_val_has_no_training_overlap(mini, as_list):
    source = source_dataset(mini=mini)
    # DataLoader -> torch Subset -> partitioned sampler -> source dataset.
    partitioned = SimpleNamespace(dataset=source, tracklet_indices=[2, 1])
    loader = SimpleNamespace(dataset=Subset(partitioned, [1, 0]))
    before = copy.deepcopy(source.virtual_rate_meta)
    metadata = production_functions()['_ct_v28_export_metadata']([loader] if as_list else loader, 0)
    assert metadata == dict(scene_id='scene-val-a', tracklet_key='track/1',
                            source_tracklet_index=1, partition='test',
                            dataset_split='mini_val' if mini else 'val',
                            parameter_training_overlap=False,
                            metadata_status='ok', metadata_error='')
    assert source.virtual_rate_meta == before


def test_training_internal_role_reports_actual_scene_overlap():
    source = source_dataset(role='train')
    sampler = SimpleNamespace(dataset=source, tracklet_indices=[0], partition='calibration')
    metadata = production_functions()['_ct_v28_export_metadata'](SimpleNamespace(dataset=sampler), 0)
    assert metadata['partition'] == 'calibration'
    assert metadata['parameter_training_overlap'] is True
    assert metadata['dataset_split'] == 'train_track'


@pytest.mark.parametrize('fault', ['missing_loader', 'missing_scene', 'wrong_role', 'bad_index', 'bad_key'])
def test_unreliable_metadata_stays_unknown_with_row_and_log_error(fault):
    source = source_dataset()
    loader, index = SimpleNamespace(dataset=SimpleNamespace(dataset=source)), 1
    if fault == 'missing_loader':
        loader = None
    elif fault == 'missing_scene':
        source.virtual_rate_meta[1].pop('scene_token')
    elif fault == 'wrong_role':
        source.ct_scene_role = 'calibration'
    elif fault == 'bad_index':
        index = 99
    else:
        source.get_tracklet_key = lambda index: 'wrong'
    with pytest.warns(RuntimeWarning, match='v28 export metadata unavailable'):
        metadata = production_functions()['_ct_v28_export_metadata'](loader, index)
    assert metadata['metadata_status'] == 'unavailable' and metadata['metadata_error']
    assert metadata['scene_id'] == metadata['tracklet_key'] == 'unknown'
    assert metadata['parameter_training_overlap'] is None


class Metric:
    def __init__(self):
        self.values = []
    def __call__(self, *values):
        self.values.append([v.detach().clone() if torch.is_tensor(v) else v for v in values])
    def reset(self):
        pass
    def compute(self):
        return 1.


def export_host(loader, sequence):
    host = SimpleNamespace(
        config=SimpleNamespace(ct_enable_v27=True, ct_enable_v28=True,
                               export_v3_candidate_diagnostics=True, version='v1.0-trainval'),
        trainer=SimpleNamespace(val_dataloaders=[loader], test_dataloaders=loader),
        current_epoch=59, global_step=75720, device=torch.device('cpu'),
        logger=SimpleNamespace(experiment=SimpleNamespace(add_scalars=lambda *args, **kwargs: None)),
        _tracking_test_endpoints=[], _proposal_test_diagnostics=[], _b3_test_rollouts=[],
        _proposal_sequence_diagnostics=[dict(frame_id=1, structural_available=True, utility_gain=.125)],
        _b3_sequence_rollouts=[],
        _ct_v27_sequence_endpoints=[dict(frame_id=i, is_initial=i == 0, final_success=.7,
                                       final_precision=.8, scene_id='unknown') for i in range(2)],
        logged=[],
    )
    host.log = lambda name, value, **kwargs: host.logged.append(name)
    for key in ('success', 'prec', 'success_step', 'prec_step', 'n_frames', 'runtime'):
        setattr(host, key, Metric())
    def evaluate(frames):
        assert frames is sequence
        assert 'tracklet_key' not in frames[0] and 'scene_id' not in frames[0]
        host.forward_sample = torch.rand(4)
        return [.8, .6], [.1, .3], ['box0', 'box1']
    host.evaluate_one_sequence = evaluate
    return host


@pytest.mark.parametrize('hook', ['validation_step', 'test_step'])
def test_export_hook_preserves_forward_input_rng_metrics_and_monitor_tags(hook):
    source = source_dataset()
    loader = SimpleNamespace(dataset=SimpleNamespace(dataset=source, tracklet_indices=[2]))
    sequence = [dict(frame_id=i, points=torch.tensor([i, 1.])) for i in range(2)]
    before = copy.deepcopy(sequence)
    scope = production_functions()
    current, control = export_host(loader, sequence), export_host(loader, sequence)
    torch.manual_seed(42)
    scope[hook](current, [sequence], 0)
    current_rng = torch.get_rng_state().clone()
    # 相同生产hook，唯一取消项是导出元信息；前向和指标必须逐位一致。
    scope['_ct_v28_export_metadata'] = lambda *args: {}
    torch.manual_seed(42)
    scope[hook](control, [sequence], 0)
    assert torch.equal(current_rng, torch.get_rng_state())
    assert torch.equal(current.forward_sample, control.forward_sample)
    assert current.logged == control.logged
    assert ('success/dev' if hook == 'validation_step' else 'success/test') in current.logged
    for key in ('success', 'prec', 'success_step', 'prec_step', 'n_frames', 'runtime'):
        a, b = getattr(current, key).values, getattr(control, key).values
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            assert all(torch.equal(x, y) for x, y in zip(av, bv))
    for frame, saved in zip(sequence, before):
        assert frame.keys() == saved.keys()
        assert torch.equal(frame['points'], saved['points'])
    rows_key = '_ct_v27_validation_endpoints' if hook == 'validation_step' else '_tracking_test_endpoints'
    rows, baseline = getattr(current, rows_key), getattr(control, rows_key)
    assert len(rows) == len(baseline) == 2
    metadata_keys = set(production_functions()['_ct_v28_export_metadata'](loader, 0))
    for row, original in zip(rows, baseline):
        assert row['scene_id'] == 'scene-val-b' and row['tracklet_key'] == 'track/2'
        assert row['partition'] == 'test' and row['parameter_training_overlap'] is False
        assert {k: v for k, v in row.items() if k not in metadata_keys} == {
            k: v for k, v in original.items() if k not in metadata_keys}
    proposal_key = '_v3_validation_proposal_diagnostics' if hook == 'validation_step' else '_proposal_test_diagnostics'
    assert getattr(current, proposal_key)[0]['scene_id'] == 'scene-val-b'
    assert current._ct_v27_sequence_endpoints[0]['scene_id'] == 'unknown'
