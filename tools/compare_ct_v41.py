"""两组KITTI固定final60/late3只读比较；先核对每一帧的共同评价分母。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EPOCHS = (58, 59, 60)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_frames(path):
    rows = [_read_line(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    indexed = {}
    for row in rows:
        key = (f"{int(row['scene_id']):04d}", int(row['track_id']), int(row['raw_frame_id']))
        if key in indexed:
            raise ValueError('duplicate frame: ' + repr(key))
        if key[0] not in ('0019', '0020'):
            raise ValueError('comparison requires test scenes 0019/0020')
        if not all(isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])
                   and 0 <= row[metric] <= 1 for metric in ('success', 'precision')):
            raise ValueError('frame S/P must be finite 0..1 contributions')
        indexed[key] = row
    if not indexed:
        raise ValueError('empty evaluation')
    return indexed


def _read_line(line):
    return json.loads(line)


def scores(rows):
    return {metric: 100 * math.fsum(row[metric] for row in rows.values()) / len(rows)
            for metric in ('success', 'precision')}


def compare(full_run, baseline_evaluation):
    full_run, baseline_evaluation = Path(full_run), Path(baseline_evaluation)
    full_manifest = _read(full_run / 'run_manifest.json')
    if full_manifest.get('model') != 'ctseqtrackv41':
        raise ValueError('Full run is not V41')
    budget = _read(full_run / 'training_budget.json')
    if budget.get('completed_epoch') != 60 or budget.get('epoch_complete') is not True:
        raise ValueError('Full run must complete 60 epochs including auxiliary coverage')
    if not isinstance(budget.get('auxiliary'), dict) or budget['auxiliary'].get('complete') is not True:
        raise ValueError('Full run is missing complete auxiliary coverage')
    baseline_summary = _read(baseline_evaluation / 'fixed_comparison.json')
    if baseline_summary.get('protocol') != 'fixed_final60_and_mean58_59_60':
        raise ValueError('baseline must use its fixed epoch evaluator')
    output, canonical, baseline_checkpoint_parent, full_checkpoint_parent = {}, None, None, None
    for epoch in EPOCHS:
        full_dir = full_run / 'evaluation' / f'epoch={epoch:03d}'
        base_root = baseline_evaluation / f'epoch={epoch:03d}'
        base_dir = base_root / 'evaluation/test' / f'epoch={epoch:03d}'
        base_manifest = _read(base_root / 'run_manifest.json')
        recipe = base_manifest['recipe']
        if any(recipe.get(k) != v for k, v in dict(batch_size=16, epoch=60, seed=42, lr=.0001,
                lr_decay_step=20, lr_decay_rate=.1, category_name='Car').items()):
            raise ValueError('baseline recipe differs from registered original recipe')
        if base_manifest.get('mode') != 'test':
            raise ValueError('baseline frames are not from test evaluation')
        checkpoint = Path(base_manifest['checkpoint'])
        if checkpoint.name != f'epoch={epoch:03d}.ckpt':
            raise ValueError('baseline checkpoint epoch mismatch')
        if baseline_checkpoint_parent is not None and checkpoint.parent != baseline_checkpoint_parent:
            raise ValueError('baseline checkpoints must belong to the same run')
        baseline_checkpoint_parent = checkpoint.parent
        expected_source = full_manifest.get('seqtrack_baseline_source_sha256')
        if not expected_source or base_manifest['source_sha256'] != expected_source:
            raise ValueError('two-arm baseline source identity mismatch')
        for field in ('dataset_manifest', 'input_metadata'):
            if not full_manifest.get(field) or full_manifest[field] != base_manifest.get(field):
                raise ValueError('two-arm dataset identity mismatch: ' + field)
        a, b = read_frames(full_dir / 'frames.jsonl'), read_frames(base_dir / 'frames.jsonl')
        if set(a) != set(b) or (canonical is not None and set(a) != canonical):
            raise ValueError('arms/checkpoints have different test frame inventories')
        canonical = set(a)
        if any(bool(a[k]['initialization']) != bool(b[k]['initialization']) for k in a):
            raise ValueError('initialization accounting differs between arms')
        sa, sb = scores(a), scores(b)
        for directory, observed, filename in ((full_dir, sa, 'metrics.json'), (base_dir, sb, 'summary.json')):
            reported = _read(directory / filename)
            if filename == 'metrics.json' and reported.get('checkpoint_epoch') != epoch:
                raise ValueError('Full checkpoint epoch differs from evaluation directory')
            if filename == 'metrics.json':
                checkpoint = Path(reported['checkpoint'])
                if checkpoint.name != f'epoch={epoch:03d}.ckpt':
                    raise ValueError('Full checkpoint filename differs from evaluation epoch')
                if full_checkpoint_parent is not None and checkpoint.parent != full_checkpoint_parent:
                    raise ValueError('Full checkpoints must belong to the same run')
                full_checkpoint_parent = checkpoint.parent
            if any(abs(float(reported[k]) - observed[k]) > .0002 for k in observed):
                raise ValueError('aggregate S/P differs from per-frame contributions')
        output[str(epoch)] = dict(full=sa, seqtrack=sb,
            delta={k: sa[k] - sb[k] for k in sa}, frames=len(a),
            initialization_frames=sum(bool(row['initialization']) for row in a.values()))
    late = {arm: {metric: sum(output[str(epoch)][arm][metric] for epoch in EPOCHS) / 3
                  for metric in ('success', 'precision')} for arm in ('full', 'seqtrack')}
    late['delta'] = {metric: late['full'][metric] - late['seqtrack'][metric] for metric in ('success', 'precision')}
    return dict(schema='ct_seqtrack.v41.two_arm_results.v1', checkpoint_epochs=list(EPOCHS),
        final60=output['60'], late3=late, epochs=output,
        limitations=['one seed; late3 is not independent replication',
                     'two systems with different original training recipes; no module-causal attribution'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--full-run', required=True)
    parser.add_argument('--seqtrack-evaluation', required=True)
    parser.add_argument('--output', help='Optional new JSON file under artifacts/ct_checks')
    args = parser.parse_args(argv)
    result = compare(args.full_run, args.seqtrack_evaluation)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output).resolve()
        allowed = (Path(__file__).resolve().parents[1] / 'artifacts/ct_checks').resolve()
        if allowed not in destination.parents or destination.exists():
            raise ValueError('comparison output must be a new file under artifacts/ct_checks')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(content + '\n')
    print(content)


if __name__ == '__main__':
    main()
