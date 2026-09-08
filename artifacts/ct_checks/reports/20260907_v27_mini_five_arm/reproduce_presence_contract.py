"""Execute real B2, B3 and v27 evaluation code with a synthetic B0 observation.

No model/config/output files are modified. This is a contract reproduction, not
a nuScenes score or a real-batch training test.
"""
from pathlib import Path
import json
import math
import runpy
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from models.ct_v2.action_v27 import B3UtilityUpdater
from utils.tracking_metrics_v27 import LocalYawBox
from utils.v27_evaluation import evaluate_sequence_v27

torch.set_num_threads(1)
# Reuse the repository's existing synthetic identity-valid B2 fixture.
fixture = runpy.run_path(str(ROOT/'tests/test_ct_v27_evidence.py'))
model = fixture['_module']()
inputs = fixture['_inputs'](extension_count=4, base_count=4)
with torch.no_grad():
    model.extension_presence_head[-1].weight.zero_()
    model.extension_presence_head[-1].bias.fill_(math.log(.1/.9))
    b2 = model(**inputs)

observation = inputs['observation_box']
updater = B3UtilityUpdater(require_calibration=True).eval()
updater.install_policy({'kind': 'always'})
with torch.no_grad():
    final, b3 = updater(observation_box=observation, raw_box=b2['ct_b2_raw_box'],
        availability=b2['ct_b2_available'], base_evidence=b2['ct_b2_base_evidence'],
        extension_evidence=b2['ct_b2_extension_evidence'],
        base_presence_probability=b2['ct_b2_base_presence_probability'],
        extension_presence_probability=b2['ct_b2_extension_presence_probability'],
        observation_stats=inputs['observation_stats'],
        b1_sigma_parallel_perp=inputs['b1_sigma_parallel_perp'],
        query_delta_t=inputs['query_delta_t'], gap_ratio=inputs['gap_ratio'])


class SyntheticObservationHost:
    """B0/data stub only; B2/B3 outputs and the evaluator above are real code."""
    device = torch.device('cpu')
    config = SimpleNamespace(up_axis=(0,0,1), IoU_space=3, ct_enable_b2=True,
        export_proposal_diagnostics=False, export_v3_candidate_diagnostics=False)

    def build_input_dict(self, sequence, frame_id, results, recursive_state, _ct_diagnostic_sidecar):
        return {}, results[-1]

    def _local_prediction_to_world(self, row, anchor):
        return LocalYawBox(row.numpy(), anchor.wlh)

    def evaluate_one_sample(self, batch, ref_box):
        output = dict(b2, **b3)
        output['observation_aux_estimation_boxes'] = observation
        # Same current host assignment as models/seqtrack3d.py:3002-3003,3049.
        output['ct_policy_candidate_valid'] = b2['ct_search_candidate_valid']
        return self._local_prediction_to_world(final[0], ref_box), None, output


sequence = [dict(tracklet_id=0, tracklet_key='synthetic/0', timestamp=i*.5,
    scene_id='synthetic', **{'3d_bbox': LocalYawBox([0.,0.,.3,.4],[1.8,4.2,1.6])})
    for i in range(2)]
report = {
    'scope': 'real_B2_B3_and_public_evaluator_with_synthetic_B0_no_nuScenes',
    'ct_b2_available': b2['ct_b2_available'].item(),
    'ct_search_candidate_valid': b2['ct_search_candidate_valid'].item(),
    'selected_point_count': b2['ct_search_extension_selected_count'].item(),
    'extension_presence_probability': b2['ct_b2_extension_presence_probability'].item(),
    'calibrated_always_b3_applied': b3['ct_router_applied_gate'].item(),
    'bounded_always_current_host_applies': bool(b2['ct_search_candidate_valid'].item()),
}
try:
    evaluate_sequence_v27(SyntheticObservationHost(), sequence)
except RuntimeError as exc:
    report['evaluation_exception'] = str(exc)
else:
    report['evaluation_exception'] = None
assert report['ct_b2_available'] == 1
assert report['ct_search_candidate_valid'] == 0
assert report['calibrated_always_b3_applied'] == 1
assert report['evaluation_exception'] == 'v27 action applied without a finite structural candidate'
(Path(__file__).parent/'presence_contract_reproduction.json').write_text(
    json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
