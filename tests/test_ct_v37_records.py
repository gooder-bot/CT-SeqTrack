"""真实原帧前向的被动候选台账：传输等价、同态基线与权限字段。"""
import pytest
import torch

from models.ct_v37.candidate_records import candidate_records
from models.ct_v37.config import normalize_config
from models.ct_v37.data import build_loaders
from models.ctseqtrackv37 import CTSEQTRACKV37
from tests.test_ct_v31_runtime import TinySource


@pytest.mark.parametrize('arm', ['b0', 'b1', 'b1_b2', 'full'])
def test_real_candidates_packed_equal_and_passive_summary_uses_same_denominator(arm):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = normalize_config(dict(v37_arm=arm, ct_engineering_check=True, point_sample_size=8,
            batch_size=2, workers=0, v32_reserve_windows=0, epoch=1))
        loaders = build_loaders(config, roles=('test',), sources={'test': TinySource((3, 2))})
        host = CTSEQTRACKV37(config, loaders=loaders).eval()
        rows = next(iter(loaders['test']))
        with torch.no_grad():
            batch, output = host._forward_raw(rows, host.evaluation_builder, training=False)
            rng = torch.get_rng_state().clone()
            expected = candidate_records(batch, output, optimized=False)
            actual = candidate_records(batch, output, optimized=True)
            assert actual == expected
            assert torch.equal(rng, torch.get_rng_state())
            commits = host.evaluation_builder.commit(output, batch, diagnostics=True)
            host.evaluation.add_batch(rows, batch, output, commits)
        summary = host.evaluation.summary()
        assert summary['same_state_q0']['interpretation'] == 'passive_one_step_not_rerollout'
        if arm != 'full':
            assert summary['same_state_q0']['success'] == summary['success']
            assert summary['same_state_q0']['precision'] == summary['precision']
        for record in actual:
            assert record['selected_geometry']['center_error_m'] >= 0
            assert record['passive']['local_reader']['coordinate_scale'] == 'first_frame_lwh'
            assert len(record['passive']['coarse_box_anchor_relative']) == 4
            if arm in ('b1_b2', 'full'):
                assert len(record['commit_support']) == 4
                assert record['commit_support'][1]['count'] == 0
                assert record['commit_support'][0]['count'] <= record['final_support'][0]['count']
                assert len(record['passive']['verification_metadata'][0]) == 35
                assert 'candidates_physical_context_norm' in record['passive']
    finally:
        torch.set_num_threads(previous)
