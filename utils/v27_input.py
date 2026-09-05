"""v27 evaluation assembly uses the same causal sampler as training."""

import copy

import torch
from torch.utils.data._utils.collate import default_collate

from utils.recursive_state import RecursiveTrackState, build_recursive_input_contract


def build_v27_eval_input(host, sequence, frame_id, results_bbs,
                         recursive_state=None, motion_prediction=None,
                         diagnostic_sidecar=None):
    """Assemble one canonical endpoint without an additional B0 forward.

    Current GT is only consumed by sampler labels and diagnostic sidecars.
    Recursive inputs come exclusively from the supplied deployed state.
    """
    from datasets.misc_utils import create_history_frame_dict
    from datasets.sampler import motion_processing_mf

    frame_id = int(frame_id)
    if frame_id <= 0 or frame_id >= len(sequence):
        raise ValueError('v27 evaluation input requires a non-initial endpoint')
    config = copy.deepcopy(host.config)
    if not bool(getattr(config, 'ct_enable_v27', False)):
        raise ValueError('v27 input assembly requires ct_enable_v27')
    config.candidate_trajectory_mode = 'shared_se2'
    config.num_candidates = 1
    config.ct_recursive_candidate_views = 1
    config.ct_b0_candidate_views = 1
    config.ct_b0_candidate_weights = [1.]
    config.ct_observation_payload_mode = 'legacy'
    state = recursive_state
    if state is None:
        if len(results_bbs) != frame_id:
            raise ValueError('v27 evaluation needs every deployed prediction before the query')
        first = sequence[0]
        state = RecursiveTrackState(
            tracklet_id=0, tracklet_key=str(first.get('tracklet_key', first.get('tracklet_id', 'eval'))),
            first_box=results_bbs[0], timestamps={0: first.get('timestamp')})
        for index in range(1, frame_id):
            state.append(index, results_bbs[index], sequence[index].get('timestamp'))
    hist_num = int(getattr(config, 'hist_num', 3))
    contract = build_recursive_input_contract(
        state, frame_id, hist_num, config, candidate_id=0,
        offsets=list(range(1, hist_num + 1)), epoch=0)
    history_ids = contract['history_frame_ids']
    history_frames = [sequence[index] for index in history_ids]
    if (motion_prediction is None
            and bool(getattr(config, 'use_b1_prepass_support', False))
            and bool(getattr(host, 'ct_enable_b1', getattr(config, 'ct_enable_b1', True)))):
        motion_prediction = host.predict_motion_prepass(
            sequence, frame_id, state.results_bbs, recursive_state=state)
    payload = dict(
        first_frame=sequence[0], prev_frames=create_history_frame_dict(history_frames),
        this_frame=sequence[frame_id], candidate_id=0,
        valid_mask=contract['history_valid_mask'].tolist(),
        prev_frame_ids=history_ids, this_frame_id=frame_id,
        history_offsets=contract['history_offsets'], sample_index=frame_id,
        tracklet_key=state.tracklet_key, tracklet_id=state.tracklet_id,
        online_recursive_state=contract,
        candidate_shared_transform=contract['candidate_shared_transform'],
        point_sampling_seeds=contract['point_sampling_seeds'],
        current_sampling_seed=contract['current_sampling_seed'],
        ct_observation_only=False, _ct_inference=True)
    if motion_prediction is not None:
        payload['motion_prediction'] = motion_prediction
    if diagnostic_sidecar is not None:
        payload['_ct_diagnostic_sidecar'] = diagnostic_sidecar
    processed = motion_processing_mf(payload, config)
    batch = default_collate([processed])
    device = getattr(host, 'device', torch.device('cpu'))

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    ref_box = state.history_boxes(history_ids, contract['history_valid_mask'])[0]
    return move(batch), ref_box

