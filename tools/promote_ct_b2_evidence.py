"""Attach a fail-closed B2 promotion record to a contract-v3 checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.online_contract import build_b2_method_contract
from utils.acquisition_metrics import validate_preflight_artifact


SCHEMA = 'ct_seqtrack.b2_evidence_promotion.v4'
PREFLIGHT_SCHEMA = 'ct_seqtrack.acquisition_preflight.v3'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(metrics):
    required = (
        'acquisition_row_recall',
        'acquisition_eligible_rows',
        'raw_helpful_precision',
        'raw_harmful_rate',
        'raw_action_count',
        'raw_action_rate',
        'raw_center_gain',
        'raw_iou_gain',
        'raw_oracle_center_headroom',
        'raw_oracle_iou_headroom',
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(
            'B2 promotion metrics are missing: ' + ', '.join(missing))
    values = {}
    for key in required:
        try:
            values[key] = float(metrics[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'B2 promotion metric {key} is not numeric') from exc
        if not math.isfinite(values[key]):
            raise ValueError(
                f'B2 promotion metric {key} must be finite')
    for key in (
            'acquisition_row_recall', 'raw_helpful_precision',
            'raw_harmful_rate', 'raw_action_rate'):
        if not 0.0 <= values[key] <= 1.0:
            raise ValueError(
                f'B2 promotion rate {key} must be in [0, 1]')
    for key in ('acquisition_eligible_rows', 'raw_action_count'):
        if values[key] < 0.0:
            raise ValueError(
                f'B2 promotion count {key} must be non-negative')
    criteria = {
        'acquisition_eligible_rows_ge_100': values[
            'acquisition_eligible_rows'] >= 100.0,
        'dev_candidate0_acquisition_row_recall_ge_0.50': values[
            'acquisition_row_recall'] >= 0.50,
        'raw_helpful_precision_ge_0.75': values[
            'raw_helpful_precision'] >= 0.75,
        'raw_harmful_rate_le_0.05': values[
            'raw_harmful_rate'] <= 0.05,
        'raw_action_count_ge_100': values['raw_action_count'] >= 100.0,
        'raw_action_rate_nonzero': values['raw_action_rate'] > 0.0,
        'raw_center_gain_positive': values['raw_center_gain'] > 0.0,
        'raw_iou_gain_positive': values['raw_iou_gain'] > 0.0,
        'raw_oracle_center_headroom_positive': values[
            'raw_oracle_center_headroom'] > 0.0,
        'raw_oracle_iou_headroom_positive': values[
            'raw_oracle_iou_headroom'] > 0.0,
    }
    return criteria, all(criteria.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--manifest-output',
        help='Optional method-only JSON manifest for scratch Full startup')
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    metrics_path = Path(args.metrics).resolve()
    preflight_path = Path(args.preflight).resolve()
    output_path = Path(args.output).resolve()
    if output_path == checkpoint_path:
        raise ValueError('promotion output must not overwrite its checkpoint')
    if output_path.exists():
        raise FileExistsError(output_path)
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    if (preflight.get('schema') != PREFLIGHT_SCHEMA
            or not bool(preflight.get('passed'))
            or not preflight.get('statistics_sha256')):
        raise RuntimeError(
            'B2 promotion requires a passed causal acquisition preflight v3')
    manifest = preflight.get('data_manifest', {}).get('manifest', {})
    dev_identity = next((
        item for item in manifest.get('partitions', [])
        if isinstance(item, dict) and item.get('partition') == 'dev'), None)
    if (not isinstance(dev_identity, dict)
            or int(metrics.get('diagnostic_tracklets', -1))
            != int(dev_identity.get('tracklet_count', -2))
            or int(metrics.get('diagnostic_rows', -1))
            != int(dev_identity.get('prediction_frames', -2))):
        raise RuntimeError(
            'promotion diagnostics do not cover the complete dev partition')
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if metrics.get('source_checkpoint_sha256') != actual_checkpoint_sha256:
        raise RuntimeError(
            'promotion metrics are not bound to the selected checkpoint')
    for provenance_key in ('candidate_diagnostics_sha256',):
        if not metrics.get(provenance_key):
            raise RuntimeError(
                f'promotion metrics lack {provenance_key}')
    checkpoint_epoch = checkpoint.get('epoch')
    if (checkpoint_epoch is None
            or int(metrics.get('source_checkpoint_epoch', -1))
            != int(checkpoint_epoch) + 1):
        raise RuntimeError(
            'promotion metrics epoch does not match selected checkpoint')
    source_config = checkpoint.get('hyper_parameters', {}).get(
        'config', checkpoint.get('hyper_parameters', {}))
    validate_preflight_artifact(preflight, source_config)

    def source_value(key, default=None):
        if isinstance(source_config, dict):
            return source_config.get(key, default)
        return getattr(source_config, key, default)

    if (str(source_value('ct_memory_mode', 'none')) != 'none'
            or str(source_value('ct_base_evidence_mode', 'full')) != 'full'):
        raise RuntimeError(
            'B2 method promotion requires memory-none/full-base evidence')
    if int(source_value('epoch', -1)) != 60:
        raise RuntimeError(
            'formal B2/Full promotion requires an independently launched '
            '60-epoch scratch run')
    if bool(source_value('ct_enable_b3', False)):
        raise RuntimeError(
            'B2 method promotion must come from the Full-B3 arm')
    criteria, passed = evaluate(metrics)
    if not passed:
        failed = sorted(key for key, value in criteria.items() if not value)
        raise RuntimeError('B2/Full promotion failed: ' + ', '.join(failed))
    learned_or_fixed_prior = (
        bool(source_value('ct_enable_b1', False))
        or str(source_value('ct_prior_mode', '')) == 'fixed_cv')
    if args.manifest_output and (
            not learned_or_fixed_prior
            or not bool(source_value('ct_enable_b2', False))
            or bool(source_value('ct_enable_b3', False))
            or int(source_value('seed', -1)) != 42):
        raise RuntimeError(
            'Full startup manifest requires the 60-epoch scratch seed42 '
            'B1-or-fixed-CV+B2, B3-off method')
    embedded_preflight = checkpoint.get('ct_acquisition_preflight')
    if (not isinstance(embedded_preflight, dict)
            or embedded_preflight.get('statistics_sha256')
            != preflight.get('statistics_sha256')):
        raise RuntimeError(
            'promotion preflight does not match the training checkpoint')
    promotion = {
        'schema': SCHEMA,
        'passed': True,
        'criteria': criteria,
        # Keep provenance labels such as population="dev_candidate0" while
        # evaluate() above validates every numeric promotion input.
        'metrics': dict(metrics),
        'source_checkpoint_sha256': actual_checkpoint_sha256,
        'source_metrics_sha256': sha256_file(metrics_path),
        'source_preflight_sha256': sha256_file(preflight_path),
        'preflight_statistics_sha256': preflight['statistics_sha256'],
        'b2_method_contract': build_b2_method_contract(
            checkpoint.get('hyper_parameters', {}).get(
                'config', checkpoint.get('hyper_parameters', {}))),
    }
    checkpoint['ct_b2_promotion'] = promotion
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    if args.manifest_output:
        manifest_path = Path(args.manifest_output).resolve()
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(promotion, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
    print(json.dumps(promotion, sort_keys=True))


if __name__ == '__main__':
    main()
