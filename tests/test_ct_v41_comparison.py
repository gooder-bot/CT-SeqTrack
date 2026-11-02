"""结果汇总必须拒绝丢帧、不同数据、未完成辅助流和混合run。运行留给服务器。"""
import json

import pytest

from tools.compare_ct_v41 import compare


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _fixture(tmp_path):
    full, baseline = tmp_path / 'full', tmp_path / 'baseline'
    shared = dict(dataset_manifest={'protocol': 'same'}, input_metadata={'content_sha256': 'same'})
    recipe = dict(epoch=60, batch_size=16, seed=42, lr=.00015, lr_milestones=[20, 40],
        lr_decay_rate=.2, v41_arm='full', dataset='kitti_mf', category_name='Car')
    _write(full / 'run_manifest.json', dict(model='ctseqtrackv41', run_type='formal', actual_recipe=recipe, **shared))
    _write(full / 'training_budget.json', dict(completed_epoch=60, epoch_complete=True, auxiliary={'complete': True}))
    _write(baseline / 'fixed_comparison.json', {'protocol': 'fixed_final60_and_mean58_59_60'})
    recipe = dict(batch_size=16, epoch=60, seed=42, lr=.0001, lr_decay_step=20, lr_decay_rate=.1, category_name='Car')
    for epoch in (58, 59, 60):
        bd = baseline / f'epoch={epoch:03d}'
        _write(bd / 'run_manifest.json', dict(recipe=recipe, source_sha256='source', mode='test',
            checkpoint=f'/baseline/train/fixed_checkpoints/epoch={epoch:03d}.ckpt', **shared))
        for directory, result_name, score in ((full / 'evaluation' / f'epoch={epoch:03d}', 'metrics.json', .6),
                (bd / 'evaluation/test' / f'epoch={epoch:03d}', 'summary.json', .4)):
            rows = [dict(scene_id='0019', track_id=2, raw_frame_id=17, initialization=True, success=1., precision=1.),
                    dict(scene_id='0019', track_id=2, raw_frame_id=18, initialization=False, success=score, precision=score)]
            _write(directory / result_name, {'success': 50*(1+score), 'precision': 50*(1+score),
                                             'checkpoint_epoch': epoch,
                                             'checkpoint': f'/full/run/formal_checkpoints/epoch={epoch:03d}.ckpt'})
            (directory / 'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    return full, baseline


def test_comparison_uses_common_frames_and_fixed_final_late3(tmp_path):
    result = compare(*_fixture(tmp_path))
    assert result['checkpoint_epochs'] == [58, 59, 60]
    assert result['final60']['full']['success'] == pytest.approx(80.)
    assert result['late3']['delta']['success'] == pytest.approx(10.)


@pytest.mark.parametrize('problem', ['missing_frame', 'wrong_input', 'mixed_run', 'aux_incomplete', 'baseline_changed'])
def test_comparison_rejects_noncomparable_results(tmp_path, problem):
    full, baseline = _fixture(tmp_path)
    manifest = baseline / 'epoch=059/run_manifest.json'
    if problem == 'missing_frame':
        path = baseline / 'epoch=059/evaluation/test/epoch=059/frames.jsonl'
        path.write_text(path.read_text(encoding='utf-8').splitlines()[0], encoding='utf-8')
    elif problem == 'aux_incomplete':
        _write(full / 'training_budget.json', dict(completed_epoch=60, epoch_complete=True, auxiliary={'complete': False}))
    else:
        row = json.loads(manifest.read_text(encoding='utf-8'))
        if problem == 'wrong_input':
            row['input_metadata']['content_sha256'] = 'changed'
        elif problem == 'baseline_changed':
            row['source_sha256'] = 'changed'
        else:
            row['checkpoint'] = '/other/train/fixed_checkpoints/epoch=059.ckpt'
        _write(manifest, row)
    with pytest.raises(ValueError):
        compare(full, baseline)
