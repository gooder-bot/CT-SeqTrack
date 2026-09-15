"""Read-only synthetic probes of production v29 B1/B2 functions.

Run from repository root: python -B artifacts/ct_checks/20260914_code_audit/b1b2/probe_b1b2.py
No model weights, production files, or historical artifacts are changed.
"""
import json
import inspect
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from pyquaternion import Quaternion
from utils.ct_search import resolve_joint_search_geometry, build_causal_history_corridor
from utils.acquisition_v29 import support_membership, acquisition_margin_grid_target_v29
from models.ct_v2.evidence_memory import B2EvidenceAcquirer
from models.ct_v2.motion import acquisition_margin_target_loss


class Box:
    def __init__(self, center=(0., 0., 0.), wlh=(2., 4., 2.)):
        self.center = np.asarray(center, dtype=np.float64)
        self.wlh = np.asarray(wlh, dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=0.)

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


def geometry_probe():
    boxes = [Box(), Box(), Box()]
    points = np.asarray([[0., 3.5, 0.], [0., 4.01, 0.], [4.2, 0., 0.]])
    report = {}
    for label, margins in [('initial', (2.04, 1.02)), ('maximum', (6., 3.))]:
        prediction = dict(mu_xy=np.zeros(2), velocity_xy=np.zeros(2),
                          direction_xy=np.asarray([1., 0.]), valid=True, source_id=1,
                          acquisition_margin_parallel_perp=np.asarray(margins),
                          current_delta_t=.5, gap_ratio=1.)
        endpoint, tube, _ = resolve_joint_search_geometry(
            boxes, [.5]*3, [1]*3, prediction=prediction,
            use_b1_prepass=True, use_acquisition_margin=True,
            fixed_margins=(2., 1.), enable_v27=True, first_frame_size=boxes[0].wlh,
            enable_v29=True, b0_crop_box=boxes[0])
        report[label] = dict(wlh=endpoint.wlh.tolist(),
            membership=(support_membership(points, endpoint) | support_membership(points, tube)).tolist())
        if label == 'initial':
            target = acquisition_margin_grid_target_v29(points[:1], np.array([100]),
                np.array([True]), np.array([], dtype=np.int64), endpoint_box=endpoint,
                tube_box=tube, actual_margins=margins, b0_crop_box=boxes[0])
            report['reachable_margin_target'] = target['target_margin'].tolist()
    corridor, diagnostic = build_causal_history_corridor(
        boxes, [.5]*3, [1]*3, enabled=True, first_frame_size=boxes[0].wlh,
        enable_v27=True, enable_v29=True, b0_crop_box=boxes[0])
    report['stationary_corridor'] = dict(exists=corridor is not None, reason=diagnostic['reason'])
    report['b0_half_xy'] = (.5 * boxes[0].wlh[[1,0]] * 1.25 + 2.).tolist()
    return report


def consensus(votes, weights):
    votes = torch.tensor([votes], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([weights], dtype=torch.float64, requires_grad=True)
    n = votes.shape[1]
    output = B2EvidenceAcquirer._consensus_vote(votes, weights,
        torch.ones(1,n,dtype=torch.bool), torch.zeros(1,2,dtype=torch.float64),
        identity_margin=torch.zeros(1,n,dtype=torch.float64),
        point_ids=torch.arange(n).reshape(1,n))
    return votes, weights, output


def isolated_candidate_consensus(votes, weights, *, remove_count_factor=False,
                                  mass_seed=False):
    """Exact production function copy, changed only at the two proposed lines.

    Dynamic compilation is confined to this diagnostic process; no source module
    or existing class method is modified. The seed change ranks the same raw
    candidates by targetness mass inside the already registered 1 m inlier ball.
    """
    source = textwrap.dedent(inspect.getsource(B2EvidenceAcquirer._consensus_vote))
    source = source.replace('@staticmethod\n', '', 1)
    if remove_count_factor:
        source = source.replace('normalized_mass * inlier_ratio', 'normalized_mass', 1)
    if mass_seed:
        original = ('seed_order = torch.argsort(\n'
                    '            point_weights, descending=True, stable=True)')
        replacement = ('seed_mass = ((torch.cdist(points, points) <= 1.0)\n'
                       '             * point_weights.unsqueeze(0)).sum(dim=1)\n'
                       '        seed_order = torch.argsort(\n'
                       '            seed_mass, descending=True, stable=True)')
        if original not in source:
            raise RuntimeError('Production seed implementation changed; inspect before rerunning')
        source = source.replace(original, replacement, 1)
    namespace = {'torch': torch}
    exec(compile(source, '<isolated_candidate_consensus>', 'exec'), namespace)
    v = torch.tensor([votes], dtype=torch.float64)
    w = torch.tensor([weights], dtype=torch.float64)
    n = v.shape[1]
    result = namespace['_consensus_vote'](v, w, torch.ones(1,n,dtype=torch.bool),
        torch.zeros(1,2,dtype=torch.float64),
        identity_margin=torch.zeros(1,n,dtype=torch.float64),
        point_ids=torch.arange(n).reshape(1,n))
    return dict(center=result['center'].tolist(),
                selected_mode_count=result['mode_unique_count'].tolist())


def consensus_probe():
    # All five foreground votes are correct and more confident. The background
    # cluster has lower TOTAL confidence mass, but a larger raw point count.
    votes, weights, output = consensus([[0.,0.]]*5 + [[5.,0.]]*100, [.9]*5 + [.02]*100)
    loss = torch.nn.functional.smooth_l1_loss(output['center'], torch.zeros(1,2,dtype=torch.float64))
    loss.backward()
    report = dict(center=output['center'].detach().tolist(),
        target_mass=4.5, background_mass=2.,
        selected_mode_count=output['mode_unique_count'].tolist(),
        target_vote_gradient_norm=votes.grad[0,:5].norm().item(),
        background_vote_gradient_norm=votes.grad[0,5:].norm().item(),
        targetness_gradient_norm=weights.grad.norm().item())
    # Three isolated high-score false positives consume all hypothesis seeds,
    # before a 20-point internally consistent foreground cluster is considered.
    _, _, output = consensus([[5.,0.],[7.,0.],[9.,0.]]+[[0.,0.]]*20,
                              [.99,.98,.97]+[.9]*20)
    report['seed_starvation'] = dict(center=output['center'].detach().tolist(),
        selected_mode_count=output['mode_unique_count'].tolist(),
        foreground_mass=18., false_positive_mass=2.94)
    report['candidate_remove_count_factor'] = isolated_candidate_consensus(
        [[0.,0.]]*5 + [[5.,0.]]*100, [.9]*5 + [.02]*100,
        remove_count_factor=True)
    report['candidate_local_mass_seed'] = isolated_candidate_consensus(
        [[5.,0.],[7.,0.],[9.,0.]]+[[0.,0.]]*20, [.99,.98,.97]+[.9]*20,
        mass_seed=True)
    return report


def margin_prevalence_probe():
    target = torch.tensor([[2.,1.]]*91 + [[4.,2.75]]*9)
    prediction = torch.tensor([[3.,2.]], requires_grad=True).expand(100,2)
    terms = acquisition_margin_target_loss(prediction, target, torch.ones(100), quantile=.9)
    gradient = torch.autograd.grad(terms['loss_per_sample'].mean(), prediction)[0].sum(0)
    result = dict(need_fraction=.09, gradient_at_margin_3_2=gradient.tolist())
    result['loss_by_constant_margin'] = {
        str(p): acquisition_margin_target_loss(torch.tensor([p]*100), target,
                    torch.ones(100), quantile=.9)['loss_per_sample'].mean().item()
        for p in [(2.,1.), (3.,2.), (4.,2.75)]}
    return result


if __name__ == '__main__':
    report = dict(geometry=geometry_probe(), consensus=consensus_probe(),
                  margin_prevalence=margin_prevalence_probe())
    Path(__file__).with_name('probe_results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
