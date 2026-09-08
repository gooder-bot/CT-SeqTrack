"""Read-only audit of historical versus v27 B0 input builders on real sampler fixtures."""
import ast
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

from tests.test_ct_v27_full_model import full_model_runtime, _construct, _training_batch
from tests.test_ct_v27_input_flow import sampler_runtime
from utils.v27_input import build_v27_eval_input


def test_astra_old_new_eval_geometry(full_model_runtime):
    import models.base_model as base
    from datasets import points_utils

    model = _construct(full_model_runtime, 'b0').eval()
    _, sequence, state = _training_batch(full_model_runtime, model)
    # Nontrivial absolute anchor, heterogeneous history yaw, including wrap.
    yaws = [2.7, 2.9, 3.11, -3.10, -2.9, -2.7, -2.5, -2.3]
    translation = np.asarray([124.375, -237.8125, 1.123])
    for i, frame in enumerate(sequence):
        frame['pc'].points += translation[:, None]
        frame['3d_bbox'].center += translation
    for i, prediction in state.predictions.items():
        prediction.center += translation
        prediction.orientation = Quaternion(axis=[0, 0, 1], radians=yaws[i])
    state.first_box = copy.deepcopy(state.predictions[0])
    # Reposition points near each predicted box so history remains dense.
    rng = np.random.default_rng(4721)
    for i, frame in enumerate(sequence):
        anchor = state.predictions.get(i, state.predictions[7])
        local = rng.uniform([-4., -3., -.8], [4., 3., .8], (3000, 3)).T
        frame['pc'].points = anchor.rotation_matrix @ local + anchor.center[:, None]

    source = subprocess.check_output(
        ['git', 'show', 'a43a8ee^:models/base_model.py'], text=True, encoding='utf8')
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MotionBaseModelMF')
    node = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'build_input_dict'][-1]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = base.__dict__.copy()
    exec(compile(module, 'a43a8ee_parent_base_model.py', 'exec'), namespace)
    old_builder = namespace['build_input_dict']

    records = []
    for frame_id in (1, 2, 3, 8):
        truncated = copy.deepcopy(state)
        truncated.predictions = {i: p for i, p in truncated.predictions.items() if i < frame_id}
        truncated.timestamps = {i: p for i, p in truncated.timestamps.items() if i < frame_id}
        model.config.ct_enable_v27 = False
        old, old_ref = old_builder(model, sequence, frame_id, truncated.results_bbs,
                                   recursive_state=truncated)
        model.config.ct_enable_v27 = True
        new, new_ref = build_v27_eval_input(model, sequence, frame_id, truncated.results_bbs,
                                           recursive_state=truncated)
        differences = {}
        for key in ('ref_boxs', 'bbox_size', 'candidate_bc', 'valid_mask', 'timestamps',
                    'delta_t_real', 'delta_t_effective', 'delta_T'):
            a, b = old[key].to(torch.float64), new[key].to(torch.float64)
            differences[key] = dict(max_abs=float((a-b).abs().max()), shape=list(a.shape))
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        a, b = old['points'], new['points']
        point_channel_differences = (a-b).abs().amax(dim=(0,1)).tolist()
        torch.testing.assert_close(a[:,:,:4], b[:,:,:4], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(old_ref.center, new_ref.center)
        records.append(dict(frame_id=frame_id, compared_fields=differences,
            points_channel_max_abs=point_channel_differences,
            history_prior_old=torch.unique(a[:,:3072,4]).tolist(),
            history_prior_new=torch.unique(b[:,:3072,4]).tolist(),
            raw_point_counts=new['b0_raw_point_count'][0].tolist()))

    # Canonical-coordinate restoration changes float roundoff, not crop membership.
    crop_cases = []
    for i in (0, 5, 6, 7):
        pc = sequence[i]['pc']
        support = state.predictions[i]
        anchor = state.predictions[7]
        old_crop = points_utils.generate_subwindow_with_aroundboxs(
            pc, support, anchor, 1.25, 2., canonicalize=False)
        new_crop = points_utils.generate_subwindow_with_aroundboxs(
            pc, support, anchor, 1.25, 2., canonicalize=True)
        assert np.array_equal(old_crop.point_ids, new_crop.point_ids)
        np.testing.assert_allclose(old_crop.points, new_crop.points, atol=1e-10, rtol=1e-10)
        crop_cases.append(dict(frame_id=i, count=len(new_crop.point_ids),
            point_ids_equal=True, xyz_max_abs=float(np.max(np.abs(old_crop.points-new_crop.points)))))

    Path(__file__).with_name('astra_eval_geometry_reproduction.json').write_text(
        json.dumps(dict(historical_commit='a43a8ee^', new_builder='working tree v27',
                        real_builders=True, synthetic_data=True, dense_inputs=records,
                        canonicalization=crop_cases), indent=2), encoding='utf8')
