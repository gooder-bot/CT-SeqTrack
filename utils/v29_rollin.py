"""v29 数据 worker 只提供窗口；host 在当前 Adam 更新前生成自身历史。"""

import copy
import time

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from utils.sampling_utils import stable_uint32_seed, sample_candidate_offset
from utils.b0_sampling import isolated_observation_rng
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state


def observation_collate(items):
    return {'ct_v29_observation_items': items}


def observation_item(sampler, index):
    """原请求 candidate 决定模式；teacher 的 actual index 仍可跨 candidate 重抽。"""
    if not isinstance(index, (int, np.integer)) or not 0 <= index < len(sampler):
        raise IndexError(index)
    index = int(index)
    candidate = int(sampler.get_candidate_index(index))
    if candidate == 0:
        return {'mode': 'teacher', 'sample': sampler._getitem_v28_observation(index)}
    anno_id = sampler.get_anno_index(index)
    tracklet, endpoint = sampler._locate_tracklet(anno_id)
    start = max(0, endpoint - 4)
    frames = sampler.dataset.get_frames(tracklet, frame_ids=list(range(start, endpoint + 1)))
    first = sampler.dataset.get_frames(tracklet, frame_ids=[0])[0]
    key = (sampler.dataset.get_tracklet_key(tracklet)
           if hasattr(sampler.dataset, 'get_tracklet_key') else str(tracklet))
    return dict(mode='rollin', index=index, candidate=candidate,
                epoch=int(sampler.epoch), start=start, endpoint=endpoint,
                tracklet_id=tracklet, tracklet_key=str(key), frames=frames, first=first)


def _initial_box(item, config):
    from datasets import points_utils
    box = copy.deepcopy(item['frames'][0]['3d_bbox'])
    box.wlh = np.asarray(item['first']['3d_bbox'].wlh).copy()
    seed = stable_uint32_seed(config.seed, 'v29-rollin-start', item['epoch'], item['index'])
    with isolated_observation_rng(seed):
        offset = sample_candidate_offset(item['candidate'], config)
        return points_utils.getOffsetBB(box, offset, limit_box=config.data_limit_box,
                                       degrees=config.degrees)


def process_query(item, predictions, frame_id, config):
    """同一 anchor 的输入/监督，绝对 ID 和局部窗口有效性分别表示。"""
    from datasets.misc_utils import create_history_frame_dict
    from datasets.sampler import motion_processing_mf
    from utils.recursive_state import box_world_row

    start = item['start']
    count = int(config.hist_num)
    ids = [max(start, frame_id - offset) for offset in range(1, count + 1)]
    valid = [int(frame_id - offset >= start) for offset in range(1, count + 1)]
    histories = [item['frames'][i - start] for i in ids]
    contract = dict(
        history_boxes_world=np.stack([box_world_row(predictions[i]) for i in ids]),
        history_valid_mask=np.asarray(valid, dtype=np.int64),
        history_timestamps=[frame.get('timestamp') for frame in histories],
        target_size=np.asarray(item['first']['3d_bbox'].wlh).copy(),
        history_quality=np.zeros((count, 4), dtype=np.float32))
    seed = stable_uint32_seed(config.seed, 'v29-rollin-query', item['epoch'],
                             item['index'], frame_id)
    payload = dict(first_frame=item['first'], this_frame=item['frames'][frame_id - start],
                   prev_frames=create_history_frame_dict(histories), candidate_id=0,
                   valid_mask=valid, prev_frame_ids=ids, this_frame_id=frame_id,
                   history_offsets=list(range(1, count + 1)), sample_index=item['index'],
                   tracklet_key=item['tracklet_key'], tracklet_id=item['tracklet_id'],
                   online_recursive_state=contract,
                   candidate_shared_transform=np.zeros(3, dtype=np.float32),
                   point_sampling_seeds=np.asarray([stable_uint32_seed(seed, 'history', i)
                                                    for i in range(count)], dtype=np.int64),
                   current_sampling_seed=stable_uint32_seed(seed, 'current'),
                   ct_observation_only=True, is_initial_query=(frame_id == start + 1),
                   history_reference_reliable=False)
    with isolated_observation_rng(seed):
        result = motion_processing_mf(payload, config)
    result.update(candidate_id=np.int64(item['candidate']),
                  ct_observation_original_index=np.int64(item['index']),
                  ct_observation_actual_index=np.int64(item['index']),
                  ct_observation_retry_count=np.int64(0),
                  ct_observation_retry_reason=np.int64(0))
    return result, predictions[ids[0]]


def prepare_observation_batch(host, items):
    """最多三波 no-grad B0 前向；不使用插件、不提交任何全局递归状态。"""
    from models.ct_variant import configure_ct_variant

    config = copy.deepcopy(host.config)
    config.ct_variant = 'b0'
    configure_ct_variant(config)
    config.candidate_trajectory_mode = 'shared_se2'
    config.ct_observation_payload_mode = 'seqtrack_core'
    config.num_candidates = config.ct_b0_candidate_views = config.ct_recursive_candidate_views = 1
    config.ct_b0_candidate_weights = [1.]
    results = [item['sample'] if item['mode'] == 'teacher' else None for item in items]
    rows = [i for i, item in enumerate(items) if item['mode'] == 'rollin']
    predictions = {i: {items[i]['start']: _initial_box(items[i], config)} for i in rows}
    routing_names = ('use_ct_joint_full', 'use_b1motion_v3', 'ct_enable_b1', 'ct_enable_b2', 'ct_enable_b3')
    routing = {key: getattr(host, key) for key in routing_names}
    flags = [(module, module.training) for module in host.modules()]
    buffers = {name: value.detach().clone() for name, value in host.named_buffers()}
    rng = capture_global_rng_state()
    old_observation = getattr(host, '_ct_observation_only_forward', False)
    started = time.perf_counter()
    forwards = 0
    try:
        host.eval()
        host._ct_observation_only_forward = True
        for key in routing_names:
            setattr(host, key, False)
        with torch.no_grad():
            for step in range(1, 4):
                active = [i for i in rows if items[i]['start'] + step < items[i]['endpoint']]
                if not active:
                    continue
                rendered = [process_query(items[i], predictions[i], items[i]['start'] + step, config)
                            for i in active]
                batch = host._move_batch_to_device(default_collate([r[0] for r in rendered]), host.device)
                output = host(batch)['observation_aux_estimation_boxes']
                forwards += len(active)
                for j, i in enumerate(active):
                    world = host._local_prediction_to_world(output[j], rendered[j][1])
                    world.wlh = np.asarray(items[i]['first']['3d_bbox'].wlh).copy()
                    predictions[i][items[i]['start'] + step] = world
            for i in rows:
                results[i] = process_query(items[i], predictions[i], items[i]['endpoint'], config)[0]
    finally:
        with torch.no_grad():
            for name, value in host.named_buffers():
                if not torch.equal(value, buffers[name]):
                    value.copy_(buffers[name])
        for module, flag in flags:
            module.training = flag
        for key, value in routing.items():
            setattr(host, key, value)
        host._ct_observation_only_forward = old_observation
        restore_global_rng_state(rng)
    host._ct_v29_rollin_diagnostics = dict(
        sample_forwards=forwards, elapsed_ms=1000 * (time.perf_counter() - started),
        rows=len(rows), teacher_rows=len(items) - len(rows))
    return host._move_batch_to_device(default_collate(results), host.device)
