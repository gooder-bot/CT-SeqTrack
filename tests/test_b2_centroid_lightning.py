"""新配方经过真实 Lightning 入口、checkpoint 和独立评测的最小生命周期。"""
import json

import pytest
import torch

pytest.importorskip('pytorch_lightning')

from models.ct_v37.config import normalize_config
from models.ct_v37.data import build_loaders
from models.ct_v37.entry import run
from tests.test_ct_v31_runtime import TinySource
from tests.test_ct_v37_lightning import artifact_root, cpu_threads, engineering_config
from utils.b2_centroid_recipes import recipe_identity
from utils.execution_lab import OPTIONS, active_options, execution_scope


def test_centroid_entry_checkpoint_and_independent_evaluation(artifact_root):
    cfg = engineering_config(artifact_root / 'centroid_entry',
        v37_centroid_recipe='full_mass_detach')
    sources = {role: TinySource((3, 2)) for role in ('train', 'val', 'test')}
    with execution_scope(OPTIONS, engineering=True):
        root = run(cfg, loaders=build_loaders(cfg, roles=tuple(sources), sources=sources))
        assert active_options() == OPTIONS
    manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['centroid_recipe_identity'] == recipe_identity(cfg)
    assert 'training_execution_recipe' not in manifest
    checkpoint = root / 'formal_checkpoints/epoch=001.ckpt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    runtime = saved['ct_v37_runtime']
    assert runtime['epoch_complete'] and runtime['completed_epoch'] == 1
    assert runtime['rows'] == 12 and runtime['optimizer_steps'] == 3
    assert saved['lr_schedulers'][0]['last_epoch'] == 1
    evaluation = normalize_config(dict(cfg, test=True, checkpoint=str(checkpoint),
        log_dir=str(artifact_root / 'centroid_standalone')))
    test_loaders = build_loaders(evaluation, roles=('test',), sources={'test': TinySource((3, 2))})
    standalone = run(evaluation, loaders=test_loaders)
    first = json.loads((root / 'results.json').read_text(encoding='utf-8'))['final']
    second = json.loads((standalone / 'results.json').read_text(encoding='utf-8'))['final']
    assert first == second
    relative = 'evaluation/epoch=001/frames.jsonl'
    assert (root / relative).read_bytes() == (standalone / relative).read_bytes()
