"""核验三臂同协议完整 mini 报告；final60 双升门与 late3 报告。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.action_calibration import sha256_file

ARMS = ('b0', 'full_cfc', 'full_gru')
EPOCHS = (58, 59, 60)


def summarize_mini(reports):
    """输入为各臂三个生产 endpoint summary，禁止用最佳 epoch 替换。"""
    if set(reports) != set(ARMS):
        raise ValueError('mini summary requires all three preregistered arms')
    rows, common = {}, None
    for arm in ARMS:
        values = list(reports[arm])
        if len(values) != 3:
            raise ValueError('each arm requires epochs 58, 59 and 60')
        indexed = {}
        for report in values:
            if report.get('schema') != 'ct_seqtrack.endpoint_diagnostics.v30' or report.get('metric_mode') != 'benchmark_compat':
                raise ValueError('requires official v30 endpoint summaries, not bare S/P')
            identity, metrics = report.get('identity', {}), report.get('metrics', {})
            expected = dict(arm=arm, dataset='nuscenes_mf', dataset_version='v1.0-mini',
                category='Car', seed=42, ablation='none', coordinate_mode='global', frame_stride=1,
                roles=['test'], scenes=['scene-0103', 'scene-0916'],
                official_checkpoint_evaluation=True, formal_config_validated=True,
                population_complete=True, all_endpoint_metadata_valid=True)
            if any(identity.get(k) != v for k, v in expected.items()):
                wrong = [k for k, v in expected.items() if identity.get(k) != v]
                raise ValueError('mini report identity mismatch: ' + ', '.join(wrong))
            if arm != 'b0' and identity.get('calibrated_policy_loaded') is not True:
                raise ValueError('Full mini reports require the checkpoint-bound calibrated policy')
            epoch = identity.get('checkpoint_epoch')
            if type(epoch) is not int or epoch not in EPOCHS or epoch in indexed:
                raise ValueError('mini checkpoint epochs must be exactly 58, 59, 60')
            for key in ('dataset_manifest_sha256', 'endpoint_population_sha256', 'checkpoint_sha256',
                        'comparison_protocol_sha256', 'code_content_sha256'):
                value = identity.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('missing/invalid official identity hash: ' + key)
            comparison = {k: identity[k] for k in ('dataset_manifest_sha256', 'endpoint_population_sha256',
                'comparison_protocol_sha256', 'code_content_sha256', 'frames', 'tracklets')}
            if common is None:
                common = comparison
            elif common != comparison:
                raise ValueError('mini reports do not cover the same protocol and endpoint population')
            if metrics.get('frames') != identity.get('frames') or metrics.get('tracklets') != identity.get('tracklets'):
                raise ValueError('metric denominator differs from the verified endpoint population')
            if metrics.get('frames', 0) <= 0 or metrics.get('tracklets', 0) <= 0:
                raise ValueError('mini report population must be nonempty')
            for key in ('S', 'P'):
                if type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 100:
                    raise ValueError('official S/P must be finite percentages')
            indexed[epoch] = report
        rows[arm] = dict(final60={key: indexed[60]['metrics'][key] for key in ('S', 'P')},
            late3={key: sum(indexed[e]['metrics'][key] for e in EPOCHS) / 3 for key in ('S', 'P')},
            epochs={str(e): dict(S=indexed[e]['metrics']['S'], P=indexed[e]['metrics']['P'],
                checkpoint_sha256=indexed[e]['identity']['checkpoint_sha256']) for e in EPOCHS})
    qualifying = []
    for arm in ARMS[1:]:
        rows[arm]['final60_delta_vs_b0'] = {k: rows[arm]['final60'][k] - rows['b0']['final60'][k] for k in ('S', 'P')}
        rows[arm]['late3_delta_vs_b0'] = {k: rows[arm]['late3'][k] - rows['b0']['late3'][k] for k in ('S', 'P')}
        rows[arm]['final60_both_above_b0'] = all(x > 0 for x in rows[arm]['final60_delta_vs_b0'].values())
        if rows[arm]['final60_both_above_b0']:
            qualifying.append(arm)
    return dict(schema='ct_seqtrack.mini_summary.v30', comparison_identity=common, arms=rows,
        promotion=dict(passed=bool(qualifying), qualifying_arms=qualifying,
            condition='at_least_one_preregistered_full_final60_S_and_P_strictly_above_same_version_b0',
            late3_role='report_only', allowed_next_datasets=['nuscenes_full', 'kitti'] if qualifying else [],
            allowed_next_arms=list(ARMS) if qualifying else [], launches_training=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument('--' + arm.replace('_', '-'), nargs=3, type=Path, required=True,
                            metavar=('E58_REPORT', 'E59_REPORT', 'E60_REPORT'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.output.resolve()
    protected = (ROOT / 'output').resolve()
    if destination == protected or protected in destination.parents:
        raise ValueError('summary must not modify the protected output directory')
    reports = {arm: [json.loads(p.read_text(encoding='utf-8')) for p in getattr(args, arm)] for arm in ARMS}
    summary = summarize_mini(reports)
    summary['sources'] = {arm: [dict(path=str(p), sha256=sha256_file(p)) for p in getattr(args, arm)] for arm in ARMS}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(summary['promotion'], sort_keys=True))
    return summary


if __name__ == '__main__':
    main()
