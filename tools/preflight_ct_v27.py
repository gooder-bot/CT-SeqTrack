"""检查 v27 官方场景集合及实际 observation/mechanism sampler 覆盖。"""
from pathlib import Path
import argparse
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect_scheduler_coverage(scheduler):
    """实际遍历索引调度器，检查每条轨迹的每个非初始化 endpoint 仅出现一次。"""
    base = scheduler.dataset.dataset
    expected = {index: int(base.get_num_frames_tracklet(index)) - 1
                for index in range(base.get_num_tracklets())
                if base.get_num_frames_tracklet(index) > 1}
    next_frame = {index: 1 for index in expected}
    original_epoch = int(scheduler.epoch)
    declared_batches = len(scheduler)
    batches = endpoints = partial_batches = shadow_rows = 0
    smallest_batch = scheduler.slots
    largest_batch = 0
    try:
        for batch_index, batch in enumerate(scheduler):
            if not batch or len(batch) % scheduler.candidate_views:
                raise RuntimeError('mechanism iterator emitted an empty/incomplete candidate group')
            slots = {}
            for epoch, emitted_index, slot, tracklet, frame, candidate, shadow in batch:
                if epoch != original_epoch or emitted_index != batch_index:
                    raise RuntimeError('mechanism epoch/batch identity mismatch')
                if slot not in range(scheduler.slots) or candidate not in range(scheduler.candidate_views):
                    raise RuntimeError('mechanism slot/candidate identity out of range')
                if candidate in slots.setdefault(slot, {}):
                    raise RuntimeError('mechanism iterator repeated a slot/candidate')
                slots[slot][candidate] = (tracklet, frame)
                if candidate == 0:
                    if tracklet not in expected or frame != next_frame[tracklet]:
                        raise RuntimeError('mechanism endpoint missing, duplicated, or temporally out of order')
                    next_frame[tracklet] += 1
                    endpoints += 1
                shadow_rows += int(shadow)
            if any(set(rows) != set(range(scheduler.candidate_views))
                   or len(set(rows.values())) != 1 for rows in slots.values()):
                raise RuntimeError('mechanism candidates do not share a causal endpoint')
            active = len(slots)
            smallest_batch, largest_batch = min(smallest_batch, active), max(largest_batch, active)
            partial_batches += int(active < scheduler.slots)
            batches += 1
    finally:
        scheduler.set_epoch(original_epoch)
    missing = sum(expected[index] - (next_frame[index] - 1) for index in expected)
    if batches != declared_batches or missing != 0:
        raise RuntimeError(f'mechanism traversal incomplete: emitted {endpoints}/{sum(expected.values())} endpoints, '
                           f'{batches}/{declared_batches} batches; missing={missing}')
    return dict(expected_endpoints=sum(expected.values()), visited_endpoints=endpoints,
                missing_endpoints=missing, complete_tracklets=len(expected),
                declared_batches=declared_batches, visited_batches=batches,
                partial_slot_batches=partial_batches, minimum_active_slots=smallest_batch,
                maximum_active_slots=largest_batch, shadow_rows=shadow_rows,
                slot_prediction_frames=list(scheduler.slot_prediction_frames),
                full_epoch_coverage=bool(scheduler.full_epoch_coverage))


def inspect_protocol(config, scene_splits, *, load_datasets=False):
    from utils.v27_protocol import build_scene_manifest
    from models.ct_variant import configure_ct_variant
    manifest = build_scene_manifest(scene_splits, config.version, config.ct_partition_seed)
    result = dict(schema='ct_seqtrack.preflight.v27', scene_manifest=manifest,
                  status='scene_manifest_verified', actual_datasets_verified=False)
    if not load_datasets:
        return result
    from datasets import get_dataset
    from datasets.sampler import OnlineRecursiveBatchSampler
    observation_config = copy.deepcopy(config)
    if str(config.net_model).lower() == 'ctseqtrack':
        observation_config.ct_variant = 'b0'
        configure_ct_variant(observation_config)
    observation_config.ct_online_recursive_training = False
    observation = get_dataset(observation_config, type=config.train_type,
                              split=config.train_split, protocol_role='train')
    if set(observation.dataset.ct_scene_names) != set(manifest['scenes']['train']):
        raise RuntimeError('observation sampler scene selection mismatch')
    result['observation'] = dict(scenes=len(observation.dataset.ct_scene_names),
        tracklets=observation.dataset.get_num_tracklets(), samples=len(observation),
        updates_per_epoch=len(observation) // int(config.batch_size))
    if bool(getattr(config, 'ct_enable_b1', False)):
        mechanism_config = copy.deepcopy(config)
        mechanism_config.num_candidates = 1
        mechanism_config.ct_recursive_candidate_views = 1
        mechanism_config.ct_b0_candidate_views = 1
        mechanism_config.ct_b0_candidate_weights = [1.]
        mechanism_config.candidate_trajectory_mode = 'shared_se2'
        mechanism = get_dataset(mechanism_config, type=config.train_type,
                                split=config.train_split, protocol_role='train')
        scheduler = OnlineRecursiveBatchSampler(mechanism,
            slots=int(config.ct_recursive_tracklet_slots), candidate_views=1,
            seed=int(config.seed), partition='train',
            shadow_enabled=bool(getattr(config, 'ct_enable_b3', False)),
            shadow_interval=int(getattr(config, 'ct_router_shadow_interval', 2)),
            shadow_slots_per_event=int(getattr(config, 'ct_router_shadow_slots_per_event', 1)))
        expected = {i for i in range(mechanism.dataset.get_num_tracklets())
                    if mechanism.dataset.get_num_frames_tracklet(i) > 1}
        actual = set(scheduler.tracklet_ids)
        if actual != expected or set(mechanism.dataset.ct_scene_names) != set(manifest['scenes']['train']):
            raise RuntimeError('mechanism stream still filters training tracklets/scenes')
        result['mechanism'] = dict(scenes=len(mechanism.dataset.ct_scene_names),
            eligible_tracklets=len(expected), selected_tracklets=len(actual), batches=len(scheduler),
            coverage=inspect_scheduler_coverage(scheduler))
        observation_steps = result['observation']['updates_per_epoch']
        if observation_steps <= 0:
            raise RuntimeError('the observation dataset cannot form one complete batch of the registered size')
        result['mechanism'].update(
            observation_budget_compatible=True,
            multiple_ticks_required=len(scheduler) > observation_steps,
            max_ticks_per_observation=(len(scheduler) + observation_steps - 1) // observation_steps,
            tick_loss_weighting='endpoint_count_within_observation_transaction')
    evaluation = get_dataset(config, type='test', split=config.test_split, protocol_role='test')
    if set(evaluation.dataset.ct_scene_names) != set(manifest['scenes']['test']):
        raise RuntimeError('official evaluation scene selection mismatch')
    result['evaluation'] = dict(scenes=len(evaluation.dataset.ct_scene_names),
        tracklets=evaluation.dataset.get_num_tracklets(), frames=evaluation.dataset.get_num_frames_total())
    result.update(status='passed', actual_datasets_verified=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, type=Path)
    parser.add_argument('--path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    from easydict import EasyDict
    from nuscenes.utils.splits import create_splits_scenes
    from models.ct_variant import configure_ct_variant
    from utils.config import load_yaml_config
    config = EasyDict(load_yaml_config(args.cfg))
    if not config.get('ct_enable_v27'):
        raise ValueError('preflight requires a 27_* config')
    if str(config.net_model).lower() == 'ctseqtrack':
        configure_ct_variant(config)
    if args.path:
        config.path = args.path
    target = args.output.resolve()
    if target.is_relative_to((ROOT / 'output').resolve()):
        raise ValueError('preflight artifacts belong in artifacts/ct_checks')
    result = inspect_protocol(config, create_splits_scenes(), load_datasets=not args.manifest_only)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"{result['status']}: {target}")


if __name__ == '__main__':
    main()
