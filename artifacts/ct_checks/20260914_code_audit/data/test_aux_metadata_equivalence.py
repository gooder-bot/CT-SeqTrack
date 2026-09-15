"""用真实 sampler 验证辅助历史无需点云；仅测试夹具数据。"""
import copy
from pathlib import Path

import numpy as np

from tests.test_ct_v27_input_flow import sampler_runtime, _case
from utils.config import load_yaml_config
from models.ct_variant import configure_ct_variant


def test_v29_auxiliary_metadata_only_is_identical(sampler_runtime):
    sampler, _, _, _, payload, _, _ = _case(sampler_runtime)
    root = Path(__file__).resolve().parents[4]
    config = sampler_runtime[3](load_yaml_config(root / 'cfgs/ct_seqtrack/29_full_cfc_nuscenes_full_perf.yaml'))
    configure_ct_variant(config)
    config.candidate_trajectory_mode = 'shared_se2'
    payload['is_initial_query'] = False
    payload['history_reference_reliable'] = False
    original = sampler.motion_processing_mf(copy.deepcopy(payload), config)
    altered = copy.deepcopy(payload)
    altered['motion_aux_prev_frames'] = {
        key: {field: value for field, value in frame.items() if field != 'pc'}
        for key, frame in altered['motion_aux_prev_frames'].items()}
    metadata_only = sampler.motion_processing_mf(altered, config)
    assert original.keys() == metadata_only.keys()
    for key in original:
        np.testing.assert_array_equal(original[key], metadata_only[key], err_msg=key)
