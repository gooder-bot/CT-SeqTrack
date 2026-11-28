"""比较 11/27 四候选基线与支持点中心修正组的固定 KITTI Car test 台账。"""
import argparse
import json
from math import fsum
from pathlib import Path
from statistics import fmean
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_v41_training_arms import EPOCHS, METRICS, read, read_split, require
from tools.compare_v41_q0 import budget


BASELINE = Path(__file__).resolve().parents[1] / 'output' / '20261127-v41-uniform-local-vote-kitti-lr1e-4-step20-gamma01'
BASELINE_RUN_ID = '991887c517a848338a837f75f1065bd2'
SCENES = ('0019', '0020')
FIXED_TRACKS = (('0020', 122), ('0020', 123))


def training_run(root, *, upgraded):
    manifest = read(root / 'run_manifest.json')
    recipe = manifest['actual_recipe']
    implementation = manifest['training_execution_recipe']['implementation_id']
    expected_id = 'v41_uniform_local_vote_support_center_v1' if upgraded else 'v41_uniform_local_vote_v1'
    require(manifest['run_type'] == 'formal' and manifest['model'] == 'ctseqtrackv41'
            and 'source_run_id' not in manifest and recipe['v41_arm'] == 'full'
            and recipe['dataset'] == 'kitti_mf' and recipe['category_name'] == 'Car'
            and recipe['v41_local_vote'] is True and recipe['epoch'] == 60
            and recipe.get('v41_support_center', False) is upgraded and implementation == expected_id,
            f'expected {"upgraded" if upgraded else "baseline"} four-slot scratch60 training run')
    if not upgraded:
        require(manifest['run_id'] == BASELINE_RUN_ID, 'expected the 11/27 four-slot baseline run')
    budget = read(root / 'training_budget.json')
    require(budget['completed_epoch'] == 60 and budget['epoch_complete'] is True
            and budget['auxiliary']['complete'] is True, 'incomplete main/auxiliary training epoch')
    return manifest


def inventory(tracks):
    return {key: tuple((r['raw_frame_id'], r['timestamp'], r['raw']) for r in rows)
            for key, rows in tracks.items()}


def paired_counts(baseline, trial):
    reference = inventory(baseline)
    require(reference == inventory(trial), 'unmatched physical track, scan, timestamp or raw target count')
    require({s for s, _ in reference} == set(SCENES) and len(reference) == 120
            and sum(map(len, reference.values())) == 6424, 'expected complete KITTI Car test coverage')
    counts = {}
    for key, rows in baseline.items():
        count = dict(frames=len(rows))
        for side, values in (('baseline', rows), ('trial', trial[key])):
            for metric in METRICS:
                count[f'{side}_{metric}'] = fsum(r[metric] for r in values)
            count[f'{side}_over10m'] = sum(r['distance'] > 10. for r in values[1:])
        counts[key] = count
    return reference, counts


def score(counts, keys):
    frames = sum(counts[k]['frames'] for k in keys)
    result = dict(frames=frames, tracks=len(keys), predictions=frames - len(keys))
    for metric in (*METRICS, 'over10m'):
        for side in ('baseline', 'trial'):
            field = f'{side}_{metric}'
            total = sum(counts[k][field] for k in keys)
            result[field] = total if metric == 'over10m' else 100 * total / frames
        result[f'delta_{metric}'] = result[f'trial_{metric}'] - result[f'baseline_{metric}']
    return result


def period_scores(epochs, retained):
    scores = {e: score(counts, retained) for e, counts in epochs.items()}
    mean = {field: fmean(s[field] for s in scores.values()) for field in next(iter(scores.values()))}
    return dict(epochs=scores, epoch_mean=mean)


def summarize(epochs):
    keys = sorted(next(iter(epochs.values())))
    require(all(k in keys for k in FIXED_TRACKS), 'expected fixed sensitivity tracks in test')
    gains = {k: fsum(counts[k]['trial_success'] - counts[k]['baseline_success']
                     for counts in epochs.values()) for k in keys}
    largest = max(keys, key=gains.get)
    groups = {'all': keys, 'without_max': [k for k in keys if k != largest]}
    for key in FIXED_TRACKS:
        groups[f'without_{key[0]}_{key[1]}'] = [k for k in keys if k != key]
    groups.update({f'scene_{scene}': [k for k in keys if k[0] == scene] for scene in SCENES})
    return dict(max_track=f'{largest[0]}/{largest[1]}', max_track_net_success_sum=gains[largest],
                scopes={name: period_scores(epochs, retained) for name, retained in groups.items()},
                per_track={f'{k[0]}/{k[1]}': period_scores(epochs, [k]) for k in keys})


def training_evidence(baseline_root, baseline, trial_root, trial):
    # files 的键是相对于 training 根目录的标签/标定路径，跨机器挂载不影响比较。
    metadata = baseline['input_metadata']
    for key in ('files', 'content_sha256'):
        require(metadata[key] and metadata[key] == trial['input_metadata'][key], f'incompatible input_metadata: {key}')
    for key in ('dataset_manifest', 'history_bank_sha256', 'q0_replay_manifest_sha256', 'local_prefix_sha256'):
        require(baseline[key] == trial[key], f'incompatible input: {key}')
    for key, value in baseline['actual_recipe'].items():
        if key != 'v41_support_center':
            require(trial['actual_recipe'].get(key) == value, f'incompatible training recipe: {key}')
    first_plan, first_total = budget(baseline_root, baseline)
    second_plan, second_total = budget(trial_root, trial)
    require(first_total == second_total, 'unequal main/auxiliary exposures or optimizer updates')
    # 每轮比对主、辅采样与插入计划。辅助批梯度由同一次主路 Adam 更新应用。
    require(first_plan == second_plan, 'unequal main/auxiliary sampling or insertion schedules')
    actual = dict(first_total, auxiliary_batches=sum(plan[-1] for plan in first_plan))
    return dict(input_metadata_sha256=metadata['content_sha256'], matched_training_epochs=len(first_plan),
                baseline=actual, trial=dict(second_total, auxiliary_batches=sum(plan[-1] for plan in second_plan)))


def compare(baseline_root, trial_root):
    baseline = training_run(baseline_root, upgraded=False)
    trial = training_run(trial_root, upgraded=True)
    training = training_evidence(baseline_root, baseline, trial_root, trial)
    epochs, previous = {}, None
    for epoch in EPOCHS:
        physical, epochs[epoch] = paired_counts(read_split(baseline_root, 'test', epoch, run=baseline),
                                                read_split(trial_root, 'test', epoch, run=trial))
        require(previous is None or physical == previous, 'frame inventory changed across fixed epochs')
        previous = physical
    return dict(baseline_run_id=baseline['run_id'], trial_run_id=trial['run_id'], matched_training=training,
                definitions=dict(delta='trial minus baseline; S/P in percentage points',
                    denominator='S/P include each legal initialization; over10m counts predictions only',
                    max_track='largest sum of paired per-frame trial minus baseline Success; ties by scene/track',
                    late3='mean of E58, E59 and E60; same largest-contributing track excluded across these epochs'),
                final60=summarize({60: epochs[60]}), late3=summarize(epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE, help='11/27 四候选基线训练目录')
    parser.add_argument('--run', type=Path, required=True, help='支持点中心修正训练目录')
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.run), ensure_ascii=False, allow_nan=False, indent=2))
