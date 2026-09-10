"""检查 v28 官方场景集合及实际 observation/mechanism sampler 覆盖。"""
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


def inspect_v29_observation_batch(observation, sampler, config, *, device='auto', host=None):
    """用生产 collate/roll-in 接口执行一个真实 batch；不构造优化器或更新参数。"""
    import torch
    from torch.utils.data import DataLoader
    from utils.v29_rollin import observation_collate, prepare_observation_batch

    if host is None:
        from models import get_model
        target = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        host = get_model(config.net_model)(config).to(torch.device(target)).eval()
    loader = DataLoader(observation, batch_sampler=sampler, num_workers=0,
                        collate_fn=observation_collate, pin_memory=False)
    payload = next(iter(loader))
    if set(payload) != {'ct_v29_observation_items'}:
        raise RuntimeError('v29 preflight did not use the production observation collate')
    items = payload['ct_v29_observation_items']
    with torch.no_grad():
        batch = prepare_observation_batch(host, items)
        # 预检只检查 observation host；插件完整训练事务由 main.py 的100步检查覆盖。
        output = host(batch)
        losses = host.compute_loss(batch, output)
    def finite_tensors(value, name):
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError('non-finite v29 preflight tensor: ' + name)
            return 1
        if isinstance(value, dict):
            return sum(finite_tensors(item, name + '.' + str(key)) for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return sum(finite_tensors(item, name + '.' + str(index)) for index, item in enumerate(value))
        return 0
    checked = sum(finite_tensors(value, name) for value, name in
                  ((batch, 'batch'), (output, 'output'), (losses, 'losses')))
    if not losses or checked == 0:
        raise RuntimeError('v29 preflight must check actual tensors and observation losses')
    return dict(status='passed', rows=len(items),
                teacher_rows=sum(item['mode'] == 'teacher' for item in items),
                rollin_rows=sum(item['mode'] == 'rollin' for item in items),
                checked_finite_tensors=checked,
                losses={key: float(value.detach().cpu()) for key, value in losses.items()
                        if torch.is_tensor(value) and value.numel() == 1},
                rollin_diagnostics=getattr(host, '_ct_v29_rollin_diagnostics', {}),
                optimizer_steps=0, engineering_only=True)


def inspect_protocol(config, scene_splits, *, load_datasets=False, device='auto'):
    from utils.v28_protocol import build_scene_manifest
    from models.ct_variant import configure_ct_variant
    from utils.online_contract import validate_scratch_training_contract, validate_v28_observation_updates
    if not getattr(config, 'ct_enable_v28', False):
        raise ValueError('preflight requires a v28 config')
    # main.py 的 argparse 会补此字段；独立 YAML 预检也必须提供。
    # 缺省按需读取，避免完整数据预检意外构造整套内存缓存；保留显式 YAML 值。
    if not hasattr(config, 'preloading'):
        config.preloading = False
    configure_ct_variant(config)
    validate_scratch_training_contract(config)
    manifest = build_scene_manifest(scene_splits, config.version, config.ct_partition_seed)
    version = 'v29' if getattr(config, 'ct_enable_v29', False) else 'v28'
    result = dict(schema='ct_seqtrack.preflight.' + version, scene_manifest=manifest,
                  status='scene_manifest_verified', actual_datasets_verified=False,
                  numeric_contract='strict_deterministic_fp32_no_tf32',
                  initial_formal_run=('scratch-only; supports a single full-data diagnostic; '
                                      'score recovery is assessed separately'))
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
    from utils.sampling_utils import StatelessObservationBatchSampler
    sampler = StatelessObservationBatchSampler(observation, int(config.batch_size), int(config.seed))
    result['observation']['update_count_check'] = validate_v28_observation_updates(
        config, sample_count=len(observation), batch_size=int(config.batch_size),
        drop_last=sampler.drop_last, observed_updates=len(sampler))
    batches = list(sampler)
    indices = [index for batch in batches for index in batch]
    if len(indices) != len(sampler) * int(config.batch_size) or len(set(indices)) != len(indices):
        raise RuntimeError('v28 observation population repeats or misses registered batch rows')
    result['observation'].update(sampling_contract=getattr(
        config, 'ct_b0_sampling_contract', 'seqtrack_original_slots_v1'),
        candidate_balancing=False, visited_rows=len(indices),
        dropped_final_rows=len(observation) - len(indices),
        loss_reduction='reference_batch')
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
    if getattr(config, 'ct_enable_v29', False):
        result['observation']['real_batch'] = inspect_v29_observation_batch(
            observation, sampler, observation_config, device=device)
    result.update(status='passed', actual_datasets_verified=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, type=Path)
    parser.add_argument('--path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--device', default='auto', help='v29 real observation batch device')
    args = parser.parse_args()
    from easydict import EasyDict
    from nuscenes.utils.splits import create_splits_scenes
    from models.ct_variant import configure_ct_variant
    from utils.config import load_yaml_config
    config = EasyDict(load_yaml_config(args.cfg))
    if not config.get('ct_enable_v28'):
        raise ValueError('preflight requires a 28_* config')
    if str(config.net_model).lower() == 'ctseqtrack':
        configure_ct_variant(config)
    if args.path:
        config.path = args.path
    from utils.online_contract import configure_v28_numerics, capture_v28_runtime_environment
    configure_v28_numerics(config)
    target = args.output.resolve()
    if target.is_relative_to((ROOT / 'output').resolve()):
        raise ValueError('preflight artifacts belong in artifacts/ct_checks')
    result = inspect_protocol(config, create_splits_scenes(), load_datasets=not args.manifest_only,
                              device=args.device)
    result['runtime_environment'] = capture_v28_runtime_environment()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"{result['status']}: {target}")


if __name__ == '__main__':
    main()
