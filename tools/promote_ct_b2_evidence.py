"""Attach a fail-closed B2 promotion record to a contract-v3 checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


SCHEMA = 'ct_seqtrack.b2_evidence_promotion.v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(metrics):
    required = (
        'acquisition_recall',
        'raw_helpful_precision',
        'raw_harmful_rate',
        'matched_b0_success_delta',
        'real_vs_empty_memory_success_delta',
        'real_vs_shuffled_memory_success_delta',
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(
            'B2 promotion metrics are missing: ' + ', '.join(missing))
    criteria = {
        'acquisition_recall_ge_0.50': float(
            metrics['acquisition_recall']) >= 0.50,
        'raw_helpful_precision_ge_0.75': float(
            metrics['raw_helpful_precision']) >= 0.75,
        'raw_harmful_rate_le_0.05': float(
            metrics['raw_harmful_rate']) <= 0.05,
        'not_below_matched_b0': float(
            metrics['matched_b0_success_delta']) >= 0.0,
        'real_memory_beats_empty': float(
            metrics['real_vs_empty_memory_success_delta']) > 0.0,
        'real_memory_beats_shuffled': float(
            metrics['real_vs_shuffled_memory_success_delta']) > 0.0,
    }
    return criteria, all(criteria.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    metrics_path = Path(args.metrics).resolve()
    output_path = Path(args.output).resolve()
    if output_path == checkpoint_path:
        raise ValueError('promotion output must not overwrite its checkpoint')
    if output_path.exists():
        raise FileExistsError(output_path)
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    criteria, passed = evaluate(metrics)
    if not passed:
        failed = sorted(key for key, value in criteria.items() if not value)
        raise RuntimeError('B2 promotion failed: ' + ', '.join(failed))
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint['ct_b2_promotion'] = {
        'schema': SCHEMA,
        'passed': True,
        'criteria': criteria,
        'metrics': {key: float(value) for key, value in metrics.items()},
        'source_checkpoint_sha256': sha256_file(checkpoint_path),
        'source_metrics_sha256': sha256_file(metrics_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(json.dumps(checkpoint['ct_b2_promotion'], sort_keys=True))


if __name__ == '__main__':
    main()
