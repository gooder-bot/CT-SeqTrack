"""从本轮两组之一已完成的 E2 权重预生成共享自然回放库。训练入口也能自动生成。"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run', required=True, type=Path, help='本次对照一个正式组的运行目录')
    parser.add_argument('--checkpoint', type=Path, help='同 run 完整 E2 checkpoint，缺省使用 formal_checkpoints')
    parser.add_argument('--path', type=Path, help='同内容的 KITTI training 数据路径')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import yaml
    from datasets.kitti_v41 import get_v41_raw_dataset
    from models.ct_v41.config import normalize_config, recipe_scope
    from models.ct_v41.data import RawEndpointDataset
    from models.ct_v41.entry import configure_numerics
    from models.ct_v41.fixed_replay import FixedReplayBank
    from models.ct_v41.identity import runtime_key
    from models.ct_v41.runtime import validate_resume_payload
    from models.ctseqtrackv41 import CTSEQTRACKV41

    run = args.run
    checkpoint = args.checkpoint or run / 'formal_checkpoints' / 'epoch=002.ckpt'
    resolved = yaml.safe_load((run / 'resolved_config.yaml').read_text(encoding='utf-8'))
    if args.path is not None:
        resolved['path'] = str(args.path)
    cfg = normalize_config(dict(resolved, test=True, log_dir=None, checkpoint=str(checkpoint)))
    if cfg.v41_history_training == 'legacy' or not cfg.v41_joint_search or cfg.ct_engineering_check:
        raise ValueError('expected a formal history-window comparison run')
    configure_numerics(cfg)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = validate_resume_payload(saved.get(runtime_key(cfg)), cfg, training=False)
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    if (identity['completed_epoch'] != 2 or identity['run_id'] != manifest['run_id']
            or identity['config_sha256'] != manifest['config_sha256']):
        raise ValueError('expected a complete E2 checkpoint of the specified run')
    with recipe_scope(cfg):
        host = CTSEQTRACKV41(cfg)
        host.load_state_dict(saved['state_dict'], strict=True)
        del saved
        host._completed_epoch = 2
        host.cuda().eval()
        dataset = RawEndpointDataset(get_v41_raw_dataset(cfg, 'train'), training=True, joint_search=True)
        bank = FixedReplayBank(args.output, cfg.seed)
        bank.ensure_catalog(host, dataset)
        print(json.dumps(dict(path=str(args.output), catalog_sha256=bank.manifest_sha256,
                              E2_weights_sha256=bank.donor_weights_sha256,
                              events={k: len(v) for k, v in bank.groups.items()})))


if __name__ == '__main__':
    main()
