"""新 B1 配方的真实 PL2.0.2 两轮验证与独立评测；无真实数据依赖。"""
import json
from pathlib import Path
import tempfile

import pytest
import torch

pl = pytest.importorskip('pytorch_lightning')

from models.ct_v37.config import load_config, normalize_config
from models.ct_v37.data import build_loaders
from models.ct_v37.entry import run
from tests.test_ct_v31_runtime import TinySource
from utils.execution_lab import active_options
from utils.b1_training_recipes import recipe_identity


def test_factorized_b1_validates_every_epoch_and_standalone_test_matches():
    artifact_root = Path(__file__).resolve().parents[1] / 'artifacts/ct_checks'
    artifact_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='b1_recipe_pl202_', dir=artifact_root))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = load_config('cfgs/ct_seqtrack/37_b1_dedup_factorized_mini.yaml',
            dict(ct_engineering_check=True, accelerator='cpu', epoch=2, workers=0,
                 point_sample_size=8, batch_size=2, v32_reserve_windows=0,
                 log_dir=str(root / 'train')))
        sources = {role: TinySource((3, 2)) for role in ('train', 'val', 'test')}
        trained = run(cfg, loaders=build_loaders(cfg, roles=tuple(sources), sources=sources))
        assert active_options() == ()
        for epoch in (1, 2):
            record = json.loads((trained / 'validation' / f'epoch={epoch:03d}.json').read_text())
            assert record['complete_coverage'] is True
        manifest = json.loads((trained / 'run_manifest.json').read_text())
        assert manifest['training_execution_recipe'] == recipe_identity(cfg)
        checkpoint = trained / 'formal_checkpoints/epoch=002.ckpt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['ct_v37_runtime']['epoch_complete'] is True
        assert saved['ct_v37_runtime']['completed_epoch'] == 2
        assert saved['lr_schedulers'][0]['last_epoch'] == 2
        evaluation = normalize_config(dict(cfg, test=True, checkpoint=str(checkpoint),
            log_dir=str(root / 'independent_test')))
        independent = run(evaluation, loaders=build_loaders(evaluation, roles=('test',),
            sources={'test': TinySource((3, 2))}))
        first = json.loads((trained / 'results.json').read_text())['final']
        second = json.loads((independent / 'results.json').read_text())['final']
        assert first == second
        relative = Path('evaluation/epoch=002/frames.jsonl')
        assert (trained / relative).read_bytes() == (independent / relative).read_bytes()
        assert active_options() == ()
    finally:
        torch.set_num_threads(previous)
