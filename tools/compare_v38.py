"""v38 三组只读验收：固定 R/C、final60/late3及预登记Full，不挑最佳轮。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from tools import compare_v35_b0 as original
from tools.compare_v37 import validate_sampler
from models.ct_v38.config import (REGISTRATION_FILE, REGISTRATION_SCHEMA, RECIPES,
    normalize_config, config_identity, recipe_identity)
from models.ct_v38.identity import SCHEMA
from utils.execution_profile import validate_execution_identity

base = original.base
require, EvidenceError = original.require, original.EvidenceError
LABELS = tuple(RECIPES)
SOURCES = ['q0', 'prior', 'trusted_propagation']
REPORT_SCHEMA = 'ct_seqtrack.v38.comparison.v1'
CANDIDATE_SCHEMA = 'ct_seqtrack.v38.candidate_records.v1'


def load_registration():
    value = original.read_object(ROOT / 'cfgs/ct_seqtrack' / REGISTRATION_FILE)
    require(value.get('schema') == REGISTRATION_SCHEMA, 'v38 registration schema mismatch')
    source = value.get('source', {})
    require(isinstance(source.get('files'), dict) and source['files'], 'missing registered source')
    require(source.get('sha256') == original.digest(source['files']), 'source checksum mismatch')
    for path, checksum in source['files'].items():
        original._sha256(checksum, 'source.' + path)
    for path in ('models/ct_v31/data.py', 'utils/tracking_metrics.py'):
        require(source['files'].get(path) == hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                'metric/sampler reconstruction source differs: ' + path)
    rows = value.get('runs', [])
    require(len(rows) == 3 and {row.get('label') for row in rows} == set(LABELS), 'three v38 runs required')
    for row in rows:
        require(row.get('arm') == RECIPES[row['label']]['arm'], 'registered arm mismatch')
        require(row.get('execution_profile') == 'optimized_fp32', 'v38 requires FP32')
        original._sha256(row.get('config_sha256'), row['label'] + '.config')
    require(value.get('registered_main_model') == 'full_standard', 'fixed Full main model required')
    require(value.get('policy_activation') == 'active_from_epoch1', 'no shadow curriculum is registered')
    expected = {name: list(original.REGISTERED_FRAME_SHA256[name]) for name in ('reference', 'baseline')}
    require(value.get('reference_frames_sha256') == expected, 'fixed R/C fingerprints mismatch')
    return value


def _array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), label + ': nonfinite/wrong shape')
    return result


def validate_candidates(row, label, arm):
    record = row.get('diagnostic_candidates')
    require(isinstance(record, dict) and record.get('schema') == CANDIDATE_SCHEMA, label + ': missing candidate schema')
    require(record.get('names') == SOURCES, label + ': candidate order mismatch')
    valid, raw = record.get('valid'), record.get('raw_valid')
    require(all(isinstance(v, list) and len(v) == 3 and all(type(x) is bool for x in v)
                for v in (valid, raw)), label + ': invalid validity masks')
    boxes = _array(record.get('boxes_anchor_relative'), (3, 4), label + '.boxes')
    require(raw[1:] == [True, True] and any(valid), label + ': finite propagation candidates required')
    require(record.get('q0_is_fallback') is (not raw[0]), label + ': q0 provenance mismatch')
    expected_valid, aliases = list(raw), [-1, -1, -1]
    # 模型使用 FP32；不得用 NumPy float64 在边界重新定义已登记去重。
    float_boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if not raw[0]:
        aliases[0] = 1
    for later in range(1, 3):
        for earlier in range(later):
            delta = float_boxes[later] - float_boxes[earlier]
            angle = torch.atan2(delta[3].sin(), delta[3].cos()).abs()
            if expected_valid[later] and expected_valid[earlier] and bool(delta[:3].abs().amax() <= 1e-4) and bool(angle <= 1e-5):
                expected_valid[later], aliases[later] = False, earlier
    require(valid == expected_valid and record.get('alias_of') == aliases, label + ': candidate dedup mismatch')
    selected = record.get('selected_index')
    require(type(selected) is int and 0 <= selected < 3 and valid[selected], label + ': invalid selection')
    require(row.get('selected_index') == selected and record.get('selected_source') == SOURCES[selected], label + ': source mismatch')
    if arm == 'full':
        utility = _array(record.get('utility'), (3,), label + '.utility')
        require(bool(((utility >= 0) & (utility <= 1)).all()), label + ': utility outside [0,1]')
        require(selected == int(np.where(valid, utility, -np.inf).argmax()), label + ': selection differs from strict gain/tie rule')
        original.close(record.get('selected_utility'), utility[selected], label + '.selected_utility', 2e-7)
    else:
        require(record.get('utility') is None and selected == (0 if raw[0] else 1), label + ': B1 policy changed')
    support = record.get('support', {})
    ids = support.get('raw_ids')
    require(isinstance(ids, list) and all(type(v) is int and v >= 0 for v in ids)
            and len(ids) == len(set(ids)) == support.get('count'), label + ': support raw IDs/count invalid')
    count, target_count = support['count'], support.get('target_count')
    require(type(target_count) is int and 0 <= target_count <= count, label + ': target support invalid')
    purity = target_count / count if count else 0.
    original.close(support.get('target_purity'), purity, label + '.purity', 1e-7)
    geometry = record.get('geometry')
    require(isinstance(geometry, list) and len(geometry) == 3, label + ': missing candidate geometry')
    real_q0 = raw[0]
    target = real_q0 and geometry[0]['exact_iou'] >= .5 and count >= 3 and purity >= .5
    require(record.get('trusted_valid') is real_q0 and record.get('trusted_target') == (target if real_q0 else None),
            label + ': trusted supervision target mismatch')
    score = record.get('trusted_score') if arm == 'full' else record.get('legacy_q0_quality')
    if real_q0:
        require(0 <= base.number(score, label + '.trusted_score') <= 1, label + ': trusted score invalid')
    eligible = selected == 0 and real_q0 and count >= 3 and score >= .5
    require(record.get('trusted_eligible') is eligible, label + ': trusted gate mismatch')
    actual_count = row.get('geometric_fg_count')
    require(type(actual_count) is int and actual_count >= 0, label + ': missing actual geometry support')
    if arm == 'full':
        require(actual_count == (count if selected == 0 and real_q0 else 0),
                label + ': Full actual support differs from canonical q0 support')
    # B1 保留原 world-space 几何计算，以真实提交计数核验；canonical 台账不改旧策略。
    actual_eligible = eligible if arm == 'full' else selected == 0 and real_q0 and actual_count >= 3 and score >= .5
    require(row.get('trusted_write') is actual_eligible and row.get('supported') is actual_eligible,
            label + ': trusted commit mismatch')
    require(row.get('memory_write') is False, label + ': v38 must never write identity memory')
    for metric in original.METRICS:
        original.close(geometry[selected].get(metric), row[metric], label + '.selected_' + metric, 2e-7)


def _metrics_identity(metrics, epoch, label):
    original.equal_fields(metrics, dict(schema=SCHEMA, checkpoint_epoch=epoch,
        metric_mode='benchmark_compat', **original.COUNTS), label + '.metrics')
    require(metrics.get('complete_coverage') is True, label + ': incomplete coverage')


def read_run(directory, spec, registration):
    root, arm, label = Path(directory).resolve(), spec['arm'], spec['label']
    source, provenance = registration['source'], {}
    def read(relative, yaml_file=False):
        path = root / relative
        value = original.read_object(path, yaml_file=yaml_file)
        provenance[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return value
    manifest = read('run_manifest.json')
    original.equal_fields(manifest, dict(schema=SCHEMA, model='ctseqtrackv38', seed=42, arm=arm,
        temporal_backend='cfc', evaluation_rng_policy='per_checkpoint_seed_v1',
        policy_activation='active_from_epoch1'), label + '.manifest')
    require(manifest.get('enabled') == dict(B1=True, B2=False, B3=arm == 'full'), label + ': enabled modules mismatch')
    require(manifest.get('source') == source and manifest.get('calibration_required') is False, label + ': source/calibration mismatch')
    run_id = manifest.get('run_id')
    require(isinstance(run_id, str) and len(run_id) == 32 and all(c in '0123456789abcdef' for c in run_id), label + ': run ID')
    expected_registration = dict(schema=REGISTRATION_SCHEMA, label=label,
        sha256=hashlib.sha256((ROOT / 'cfgs/ct_seqtrack' / REGISTRATION_FILE).read_bytes()).hexdigest())
    require(manifest.get('registration') == expected_registration, label + ': registration fingerprint mismatch')
    for field in ('common_parameter_hash', 'prior_parameter_hash', 'base_initial_parameter_hash', 'initial_parameter_hash'):
        original._sha256(manifest.get(field), label + '.' + field)
    config = read('resolved_config.yaml', True)
    require(dict(normalize_config(config)) == config, label + ': incomplete normalized config')
    require(config['v38_arm'] == arm and config['v38_training_recipe'] == label
            and not config['test'] and not config['ct_engineering_check'], label + ': formal recipe mismatch')
    require(config_identity(config) == manifest.get('config_sha256') == spec['config_sha256'], label + ': config mismatch')
    require(manifest.get('training_execution_recipe') == spec['training_execution_recipe'] == recipe_identity(config), label + ': execution recipe mismatch')
    validate_execution_identity(manifest.get('execution_identity'), config, source)
    for path in sorted((root / 'resume_manifests').glob('*.json')):
        original.equal_fields(read(path.relative_to(root).as_posix()), dict(schema='ct_seqtrack.v38.resume.v1',
            run_id=run_id, config_sha256=manifest['config_sha256'], source=source), label + '.resume')
    budget = read('training_budget.json')
    original.equal_fields(budget, dict(schema=SCHEMA, **original.BUDGET), label + '.budget')
    require(budget.get('epoch_complete') is True, label + ': epoch60 incomplete')
    training_source = validate_sampler(budget.get('sampler'), config, 60, label + '.sampler')
    require(len(list((root / 'training_audits').glob('epoch=*.json'))) == 60, label + ': sixty audits required')
    plans = {}
    for epoch in range(1, 61):
        audit = read(f'training_audits/epoch={epoch:03d}.json')
        original.equal_fields(audit, dict(schema='ct_seqtrack.v38.training_audit.v1', model_schema=SCHEMA,
            run_id=run_id, config_sha256=manifest['config_sha256'], completed_epoch=epoch,
            rows=19108, optimizer_steps=1195), label + '.audit')
        require(audit.get('epoch_complete') is True and validate_sampler(audit.get('sampler'), config, epoch, label) == training_source,
                label + ': incomplete audit/data changed')
        plans[epoch] = audit['sampler']['plan_sha256']
    require(audit['sampler'] == budget['sampler'], label + ': final audit/budget mismatch')
    results = read('results.json')
    require(results.get('checkpoint_epochs') == original.EPOCHS, label + ': final window must be 58/59/60')
    require(results.get('policy_activation') == 'active_from_epoch1', label + ': no shadow policy allowed')
    rows_by_epoch, scores, diagnostics = {}, {}, {}
    for epoch in original.EPOCHS:
        relative = f'evaluation/epoch={epoch:03d}'
        path = root / relative / 'frames.jsonl'
        rows = original.read_frames(path)
        provenance[relative + '/frames.jsonl'] = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = read(relative + '/metrics.json')
        _metrics_identity(metrics, epoch, label)
        for key, row in rows.items():
            if key[1]:
                validate_candidates(row, label + f'.{epoch}.{key}', arm)
        from models.ct_v38.runtime import reliability_summary
        require(metrics.get('trusted_reliability') == reliability_summary(
            [row['diagnostic_candidates'] for key, row in rows.items() if key[1]]),
            label + ': reliability summary differs from original candidate records')
        scores[epoch] = original.score(rows.values())
        for metric in original.METRICS:
            original.close(metrics.get(metric), scores[epoch][metric], label + '.' + metric, 2e-6)
        diagnostics[str(epoch)] = {key: metrics.get(key) for key in
            ('selection', 'trusted', 'trusted_reliability', 'motion_slices', 'diagnostics')}
        rows_by_epoch[epoch] = rows
    late3 = {m: math.fsum(scores[e][m] for e in original.EPOCHS) / 3 for m in original.METRICS}
    _metrics_identity(results.get('final', {}), 60, label)
    for metric in original.METRICS:
        original.close(results['final'].get(metric), scores[60][metric], label + '.final.' + metric, 2e-6)
        original.close(results.get('late3', {}).get(metric), late3[metric], label + '.late3.' + metric, 2e-6)
    return dict(directory=str(root), config=config, manifest=manifest, source=training_source,
        sampler_plans=plans, rows=rows_by_epoch, scores=scores, late3=late3, provenance=provenance,
        mechanism_diagnostics=diagnostics, timing={key: results.get(key) for key in
            ('fit_wall_seconds', 'train_wall_seconds', 'validation_wall_seconds', 'evaluation_wall_seconds')})


def compare_runs(*, registered_reference, baseline, **paths):
    registration = load_registration()
    require(set(paths) == set(LABELS) and len({str(Path(p).resolve()) for p in paths.values()}) == 3,
            'three distinct complete v38 runs required')
    runs = dict(reference=base.read_run(registered_reference, 'reference'), baseline=base.read_run(baseline, 'baseline'))
    for name in ('reference', 'baseline'):
        actual = tuple(runs[name]['provenance'][f'evaluation/epoch={epoch:03d}/frames.jsonl'] for epoch in original.EPOCHS)
        require(actual == original.REGISTERED_FRAME_SHA256[name], 'fixed R/C original evidence mismatch: ' + name)
    specs = {row['label']: row for row in registration['runs']}
    for label in LABELS:
        runs[label] = read_run(paths[label], specs[label], registration)
    require(len({runs[x]['manifest']['run_id'] for x in LABELS}) == 3, 'three new run IDs required')
    require(all(runs[x]['sampler_plans'] == runs['b1_standard']['sampler_plans'] for x in LABELS), 'sampler plans differ')
    for key in ('common_parameter_hash', 'prior_parameter_hash', 'base_initial_parameter_hash'):
        require(len({runs[x]['manifest'][key] for x in LABELS}) == 1, 'common initialization differs: ' + key)
    require(runs['b1_standard']['manifest']['initial_parameter_hash'] == runs['b1_hold']['manifest']['initial_parameter_hash'], 'B1 initialization differs')
    comparisons = [('full_minus_b1', 'full_standard', 'b1_standard'), ('hold_minus_standard', 'b1_hold', 'b1_standard')]
    report = base.compare_loaded(runs, candidate_labels=LABELS,
        config_differences=('v38_arm', 'v38_training_recipe', 'lr_milestones', 'devices'), comparisons=comparisons)
    passed = report['candidates']['full_standard']['passed']
    report.update(schema=REPORT_SCHEMA, status='passed' if passed else 'failed',
        registered_main_model='full_standard', selected='full_standard' if passed else None,
        source_sha256=registration['source']['sha256'], policy_activation='active_from_epoch1',
        mechanism_diagnostics={label: runs[label]['mechanism_diagnostics'] for label in LABELS},
        timing={label: runs[label]['timing'] for label in LABELS},
        center_tail_diagnostics={label: original.tail_diagnostics(run) for label, run in runs.items()})
    report.pop('passing_rank', None)
    report['criterion']['ranking'] = 'Full standard fixed; hold/control/best epochs never replace it'
    report['limitations'].extend(['三组均单seed42；late3不是独立重复。',
        'Full差值包含选择与可信更新及其闭环训练影响，不能拆成两个因果收益。',
        '并发运行墙钟不能证明提速；train wall排除逐轮验证但仍包含I/O等宿主开销。'])
    return report


def reliability_plot_data(report):
    """保留每轮固定分箱；空箱必须为 None，不跨轮合并相关样本。"""
    require(report.get('status') in ('passed', 'failed'), 'reliability plot requires valid comparison evidence')
    full = report.get('mechanism_diagnostics', {}).get('full_standard', {})
    epochs = {}
    for epoch in original.EPOCHS:
        summary = full.get(str(epoch), {}).get('trusted_reliability')
        require(isinstance(summary, dict), f'epoch {epoch}: missing trusted reliability')
        bins = summary.get('bins')
        require(isinstance(bins, list) and len(bins) == 10, f'epoch {epoch}: ten fixed bins required')
        total = 0
        for index, item in enumerate(bins):
            require(item.get('lower') == index / 10 and item.get('upper') == (index + 1) / 10,
                    f'epoch {epoch}: bin edges changed')
            count = item.get('count')
            require(type(count) is int and count >= 0, f'epoch {epoch}: invalid bin count')
            total += count
            if not count:
                require(item.get('mean_probability') is None and item.get('event_frequency') is None,
                        f'epoch {epoch}: empty bins must remain missing')
            else:
                for field in ('mean_probability', 'event_frequency'):
                    require(0 <= base.number(item.get(field), field) <= 1,
                            f'epoch {epoch}: invalid {field}')
        require(total == summary.get('count') and total == summary.get('positive', -1) + summary.get('negative', -1),
                f'epoch {epoch}: reliability counts disagree')
        if total:
            require(0 <= base.number(summary.get('brier'), 'brier') <= 1, f'epoch {epoch}: invalid Brier')
        else:
            require(summary.get('brier') is None, f'epoch {epoch}: empty Brier must remain missing')
        epochs[str(epoch)] = summary
    return dict(schema='ct_seqtrack.v38.reliability_plot.v1', epochs=epochs,
        report_schema=report.get('schema'), report_without_plot_sha256=original.digest(report),
        source_sha256=report.get('source_sha256'), synthetic_fixture=bool(report.get('synthetic_fixture', False)),
        transformations=['Full standard only; one panel per epoch 58/59/60',
            'fixed ten bins; scatter occupied bins at their mean T and empirical target frequency',
            'empty bins remain None and are not plotted; no connecting lines or interpolation',
            'labels show frame count per occupied bin; no pooling or independent-replicate intervals'],
        interpretation='Descriptive reliability only; correlated frames and epochs are not independent repetitions.')


def write_reliability_plot(destination, report):
    """导出独立图片与分箱 JSON；只允许新建 artifacts/ct_checks 产物。"""
    destination = Path(destination).resolve()
    allowed = (ROOT / 'artifacts/ct_checks').resolve()
    require(allowed in destination.parents, 'reliability output must be under artifacts/ct_checks')
    require(destination.suffix.lower() in ('.png', '.pdf', '.svg'), 'reliability output requires PNG/PDF/SVG')
    data_path = destination.with_suffix('.data.json')
    require(not destination.exists() and not data_path.exists(), 'reliability output/data already exists')
    data = reliability_plot_data(report)
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with matplotlib.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        figure = Figure(figsize=(10.5, 4.8), dpi=180)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3, sharex=True, sharey=True)
        figure.subplots_adjust(left=.075, right=.98, top=.77, bottom=.29, wspace=.25)
        for axis, epoch in zip(axes, original.EPOCHS):
            summary = data['epochs'][str(epoch)]
            occupied = [item for item in summary['bins'] if item['count']]
            axis.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='.5', label='Identity reference')
            if occupied:
                axis.scatter([item['mean_probability'] for item in occupied],
                    [item['event_frequency'] for item in occupied], marker='o', s=32, color='#0072B2', zorder=3, clip_on=False)
                for item in occupied:
                    axis.annotate(str(item['count']), (item['mean_probability'], item['event_frequency']),
                        xytext=(0, 7 if item['event_frequency'] < .9 else -13), textcoords='offset points',
                        ha='center', fontsize=8)
            else:
                axis.text(.5, .3, 'No real q0 scores', ha='center', fontsize=9, transform=axis.transAxes)
            brier = 'missing' if summary['brier'] is None else f"{summary['brier']:.4f}"
            axis.set(title=f"Epoch {epoch}: n={summary['count']}, Brier={brier}",
                xlabel='Mean trusted score T', xlim=(0, 1), ylim=(0, 1),
                xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1])
            axis.grid(alpha=.2)
        axes[0].set_ylabel('Empirical trusted-target frequency')
        axes[0].legend(loc='lower right', fontsize=8, frameon=False)
        title = 'Full standard: trusted q0 reliability'
        if data['synthetic_fixture']:
            title += ' (synthetic fixture; not experiment results)'
        figure.suptitle(title, y=.95, fontsize=13)
        figure.text(.075, .075, 'Target: exact IoU >= 0.5, unique predicted-FG support >= 3, target purity >= 0.5.\n'
            'Numbers label bin frame counts; empty bins are omitted. Correlated frames/epochs; no independence claim.\n'
            'All real q0 scores are included, even when q0 is not selected. Fixed threshold T = 0.5 is not fitted here.', fontsize=9)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            figure.savefig(stream, format=destination.suffix[1:].lower(), dpi=180)
    data.update(figure=str(destination), dimensions_inches=[10.5, 4.8], dpi=180,
                matplotlib_version=matplotlib.__version__)
    with data_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return dict(figure=str(destination), source_data=str(data_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('registered_reference', 'baseline', *LABELS):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reliability-output', type=Path,
        help='可选：在 artifacts/ct_checks 下新建 Full 末三轮可靠性图及 .data.json')
    args = vars(parser.parse_args(argv))
    destination = args.pop('output')
    reliability_destination = args.pop('reliability_output')
    try:
        report = compare_runs(**args)
        if reliability_destination is not None:
            report['reliability_plot'] = write_reliability_plot(reliability_destination, report)
        status = 0 if report['status'] == 'passed' else 1
    except (EvidenceError, ValueError, OSError, KeyError, TypeError, ImportError) as error:
        report, status = dict(schema=REPORT_SCHEMA, status='invalid', error=str(error)), 2
    if destination is not None:
        base.write_report(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
