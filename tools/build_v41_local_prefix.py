"""从原四槽 uniform E60 的合法首框递推，保存训练集每次预测之前的旧状态。"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile
import zlib

from build_v41_q0_replay import E60_RECEIPT, file_sha256, prefix_state, track_role


SCHEMA = 'ct_seqtrack.uniform_e60_ledger.v1'


def build(args, connection):
    source_path = args.donor_source.resolve(strict=True)
    sys.path.insert(0, str(source_path))
    import pytorch_lightning as pl
    import torch
    import yaml
    from datasets.kitti_v41 import get_v41_raw_dataset
    from models.ct_v41 import data, entry
    from models.ct_v41.config import normalize_config, recipe_scope
    from models.ct_v41.provenance import source_identity
    from models.ctseqtrackv41 import CTSEQTRACKV41

    run = args.donor_run
    manifest = json.loads((run / 'run_manifest.json').read_bytes())
    checkpoint = run / 'formal_checkpoints/epoch=060.ckpt'
    checksum = file_sha256(checkpoint)
    evaluator = source_identity()['sha256']
    if (Path(data.__file__).resolve().parents[2] != source_path
            or evaluator != E60_RECEIPT['evaluator_source'] or checksum != E60_RECEIPT['checkpoint_sha256']
            or manifest['source']['sha256'] != evaluator or manifest['run_id'] != E60_RECEIPT['run_id']
            or manifest['config_sha256'] != E60_RECEIPT['config_sha256']):
        raise ValueError('expected the archived v3 uniform E60 run and evaluator')
    resolved = yaml.safe_load((run / 'resolved_config.yaml').read_bytes())
    resolved.update(path=str(args.path.resolve(strict=True)), workers=0, test=True,
                    log_dir=None, checkpoint=str(checkpoint), eval_checkpoint_epoch=60)
    if args.attention_backend:
        resolved.update(ct_engineering_check=True, v41_attention_backend=args.attention_backend)
    cfg = normalize_config(resolved)
    entry.configure_numerics()
    pl.seed_everything(cfg.seed, workers=True)
    with recipe_scope(cfg), torch.no_grad():
        source = get_v41_raw_dataset(cfg, 'auxiliary')
        ds = data.RawEndpointDataset(source, training=False)
        if source.v41_source_role != 'train' or source.v41_data_role != 'auxiliary':
            raise ValueError('uniform ledger requires complete uncut training scans')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        identity = saved['ct_v41_runtime']
        if (identity['sampler']['source_sha256'] != ds.source_sha256 or identity['completed_epoch'] != 60
                or not identity['epoch_complete'] or identity['run_id'] != manifest['run_id']):
            raise ValueError('uniform checkpoint and training split do not match')
        host = CTSEQTRACKV41(cfg).to(args.device).eval()
        host.load_state_dict(saved['state_dict'], strict=True)
        del saved
        builder = host.evaluation_builder
        builder.detailed_diagnostics = False
        tracks = [(source.get_tracklet_key(i), n) for i, n in enumerate(ds.lengths)]
        for i in range(len(tracks)):
            track_role(source, i)
        connection.execute('CREATE TABLE states(track TEXT, frame INTEGER, state BLOB NOT NULL, '
                           'PRIMARY KEY(track,frame)) WITHOUT ROWID')
        connection.execute('CREATE TABLE metadata(value TEXT NOT NULL)')
        sampler = data.ReadyQueueBatchSampler(ds.lengths, cfg.batch_size, seed=cfg.seed, training=False)
        sampler.set_epoch(60)
        visited = 0
        for requests in sampler:
            batch, output = host._forward_raw(ds.__getitems__(requests), builder, training=False)
            try:
                records = []
                for ctx in builder._pending:
                    req = ctx['row']['request']
                    value = prefix_state(ctx['state'], req.frame)
                    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
                    records.append((ctx['row']['tracklet_key'], req.frame, zlib.compress(raw, 1)))
                connection.executemany('INSERT INTO states VALUES(?,?,?)', records)
                builder.commit(output, batch, diagnostics=False)
            finally:
                host._release_materialization(batch, builder)
            visited += len(requests)
            if visited % 1024 < len(requests):
                print(f'{visited}/{len(ds)} train endpoints', flush=True)
            if args.check_batches and visited >= cfg.batch_size * args.check_batches:
                break
        complete = visited == len(ds) and not builder.states
        if not complete and not args.check_batches:
            raise RuntimeError('incomplete autonomous uniform rollout')
        receipt = dict(schema=SCHEMA, split='train', source_sha256=ds.source_sha256, tracks=tracks,
                       evaluator_sha256=evaluator, checkpoint_sha256=checksum, run_id=manifest['run_id'],
                       donor_epoch=60, center_weighting=cfg.v41_q0_center_weighting,
                       config_sha256=manifest['config_sha256'],
                       attention_backend=cfg.v41_attention_backend, device=args.device,
                       torch_version=str(torch.__version__), endpoints=visited, complete=complete)
        connection.execute('INSERT INTO metadata VALUES(?)', (json.dumps(receipt, separators=(',', ':')),))
        return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor-source', required=True, type=Path, help='原 b819e76 源码目录')
    p.add_argument('--donor-run', required=True, type=Path, help='v3 uniform 正式运行目录')
    p.add_argument('--path', required=True, type=Path, help='KITTI training 数据根目录')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    p.add_argument('--attention-backend', choices=('math', 'fa2_varlen'))
    p.add_argument('--check-batches', type=int, default=0, help='只做工程检查；不完整文件不能进入正式训练')
    args = p.parse_args()
    if args.check_batches < 0 or args.output.exists():
        p.error('expected a new output file and nonnegative check-batches')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, prefix='.local-ledger-', suffix='.sqlite', delete=False) as f:
        temporary = Path(f.name)
    try:
        conn = sqlite3.connect(temporary)
        try:
            with conn:
                receipt = build(args, conn)
        finally:
            conn.close()
        os.link(temporary, args.output)
        print(json.dumps({k:v for k,v in receipt.items() if k != 'tracks'}), flush=True)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
