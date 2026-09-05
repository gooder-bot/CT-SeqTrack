from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.tracking_metrics_v27 import LocalYawBox, metric_contributions
from utils.v27_training import (compute_b3_utility_loss, attach_h3_shadow_labels_v27,
                               accumulate_v27_binary_rows)


def _loss_case():
    data = {"box_label": torch.tensor([[1., 0, 0, 0], [0., 0, 0, 0]]),
            "bbox_size": torch.tensor([[2., 4, 2], [2., 4, 2]]),
            "target_bbox_size": torch.tensor([[2., 4, 3], [2., 4, 3]]),
            "b0_view_id": torch.zeros(2), "ct_h3_valid": torch.tensor([0., 1.]),
            "ct_h3_success_gain": torch.tensor([float("nan"), .4]),
            "ct_h3_precision_gain": torch.tensor([float("nan"), .2])}
    output = {"observation_aux_estimation_boxes": torch.zeros(2, 4, requires_grad=True),
              "ct_router_bounded_residual_xy": torch.tensor([[.5, 0], [.5, 0]], requires_grad=True),
              "ct_b2_available": torch.ones(2),
              "ct_b3_expected_success_gain": torch.zeros(2, requires_grad=True),
              "ct_b3_expected_precision_gain": torch.zeros(2, requires_grad=True),
              "ct_b3_help_logit": torch.zeros(2, requires_grad=True),
              "ct_b3_harm_logit": torch.zeros(2, requires_grad=True)}
    return data, output


def test_utility_loss_uses_real_action_labels_independent_h3_masks_and_no_absence_override():
    data, output = _loss_case()
    config = {"degrees": False, "up_axis": [0, 0, 1], "IoU_space": 3}
    losses = compute_b3_utility_loss(data, output, config)
    assert losses["help_label"].tolist() == [1., 0.]
    assert losses["harm_label"].tolist() == [0., 1.]
    expected_s = .5 * losses["h1_success_gain"].square().mean() + .5 * .4**2
    expected_p = .5 * losses["h1_precision_gain"].square().mean() + .5 * .2**2
    assert torch.allclose(losses["loss_success"], expected_s)
    assert torch.allclose(losses["loss_precision"], expected_p)
    losses["loss"].backward()
    assert output["observation_aux_estimation_boxes"].grad is None
    assert output["ct_router_bounded_residual_xy"].grad is None
    assert output["ct_b3_expected_success_gain"].grad is not None
    data["ct_h3_valid"].zero_()
    terminal = compute_b3_utility_loss(data, output, config)
    assert torch.allclose(terminal["loss_success"], terminal["h1_success_gain"].square().mean())


def test_epoch_diagnostics_use_selected_presence_and_actual_bounded_utility():
    # All target points may disappear at selection; an empty selected target set
    # does not imply that a zero displacement harmed the tracker.
    data = {'ct_extension_labels': torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
            'ct_extension_valid_mask': torch.ones(2, 3)}
    output = {'ct_extension_selected_indices': torch.tensor([[1, 2], [0, 1]]),
              'ct_extension_selected_valid_mask': torch.ones(2, 2),
              'ct_b2_available': torch.ones(2),
              'ct_b2_extension_presence_probability': torch.tensor([.1, .9]),
              'ct_b3_h1_valid': torch.ones(2),
              'ct_b3_h1_success_gain_label': torch.tensor([0., -.2]),
              'ct_b3_h1_precision_gain_label': torch.tensor([0., -.4]),
              'ct_b3_action_score': torch.tensor([-.8, -.3]),
              'ct_b3_help_logit': torch.tensor([-2., -3.]),
              'ct_b3_harm_logit': torch.tensor([-1., 2.])}
    rows = {}
    accumulate_v27_binary_rows(rows, data, output, True)
    assert rows['presence'][0][1].tolist() == [0., 1.]
    assert rows['help'][0][1].tolist() == [0., 0.]
    assert rows['harm'][0][1].tolist() == [0., 1.]
    np.testing.assert_allclose(rows['harm'][0][0], torch.sigmoid(output['ct_b3_harm_logit']))
    np.testing.assert_allclose(rows['bounded_utility'][0][:, 2], [-.8, -.3])
    assert 'alpha' not in rows  # A signed gain score is not a binary probability.


class _State:
    def __init__(self):
        self.boxes = {0: LocalYawBox([0, 0, 0, 0], [2, 4, 2])}
    def clone(self):
        return deepcopy(self)
    def history_boxes(self, frame_ids, mask):
        return [self.boxes[index] for index in frame_ids]
    def append(self, frame_id, box, timestamp=None):
        self.boxes[frame_id] = box


def _shadow_case():
    raw = {"candidate_id": 0, "this_frame_id": 1, "prev_frame_ids": [0],
           "tracklet_key": "track", "this_frame": {"timestamp": .5},
           "shadow_scheduled": True, "shadow_future_exists": [True, True]}
    raw["shadow_future"] = [dict(this_frame_id=i, prev_frame_ids=[i - 1],
        this_frame={"timestamp": i * .5, "3d_bbox": LocalYawBox([.5, 0, 0, 0], [2, 4, 2])})
        for i in (2, 3)]
    state = _State()
    class Host:
        device = torch.device("cpu")
        ct_enable_b3 = True
        config = SimpleNamespace(ct_online_recursive_training=True, seed=42,
                                 up_axis=(0, 0, 1), IoU_space=3)
        def __init__(self):
            self.calls = 0
            self._ct_online_batch_context = [{"raw": raw, "state": state}]
        def _local_prediction_to_world(self, local, anchor):
            return LocalYawBox([*(anchor.center + local[:3].detach().numpy()), float(local[3])], anchor.wlh)
        def _process_online_raw(self, future, branch_state):
            assert future["ct_observation_only"] is True
            return {"dummy": torch.zeros(1)}
        def _move_batch_to_device(self, batch, device):
            return batch
        def _shadow_forward(self, batch, seed):
            self.calls += 1
            return {"observation_aux_estimation_boxes": torch.zeros(2, 4)}
    output = {"observation_aux_estimation_boxes": torch.zeros(1, 4),
              "ct_router_bounded_residual_xy": torch.tensor([[.5, 0]]),
              "ct_b2_available": torch.ones(1),
              # Learned presence veto must not select H3 supervision.
              "ct_search_candidate_valid": torch.zeros(1)}
    return Host(), raw, state, output


def test_h3_uses_two_future_eval_branches_and_never_writes_main_state():
    host, raw, state, output = _shadow_case()
    batch = {}
    attach_h3_shadow_labels_v27(host, batch, output)
    assert batch["ct_h3_failure_reason"] == ["ok"]
    assert batch["ct_h3_valid"].item() == 1
    assert batch["ct_h3_scheduled"].item() == 1
    assert batch["ct_h3_future_exists"].tolist() == [[1., 1.]]
    assert batch["ct_shadow_forward_count"].item() == 4
    assert host.calls == 2
    assert set(state.boxes) == {0}
    expected_p = metric_contributions(1, 0)[1] - metric_contributions(1, .5)[1]
    assert batch["ct_h3_precision_gain"].item() == pytest.approx(expected_p)
    assert batch["ct_h3_success_gain"].item() > 0


def test_h3_terminal_unsampled_and_failed_execution_have_distinct_diagnostics():
    host, raw, state, output = _shadow_case()
    raw["shadow_future"] = raw["shadow_future"][:1]
    raw["shadow_future_exists"] = [True, False]
    batch = {}
    attach_h3_shadow_labels_v27(host, batch, output)
    assert host.calls == 0 and batch["ct_h3_valid"].item() == 0
    assert batch["ct_h3_failure_reason"] == ["terminal_incomplete_horizon"]
    raw["shadow_scheduled"] = False
    attach_h3_shadow_labels_v27(host, batch, output)
    assert batch["ct_h3_failure_reason"] == ["not_scheduled"]
    host, raw, state, output = _shadow_case()
    def fail(batch, seed):
        raise ValueError("test input failure")
    host._shadow_forward = fail
    attach_h3_shadow_labels_v27(host, batch, output)
    assert batch["ct_h3_valid"].item() == 0
    assert batch["ct_h3_failure_reason"][0].startswith("shadow_failure:ValueError")
    assert set(state.boxes) == {0}
