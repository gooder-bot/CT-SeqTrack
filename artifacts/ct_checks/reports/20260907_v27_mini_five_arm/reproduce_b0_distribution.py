"""CPU evidence using the repository's real sampler/model; no model edits."""
from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from tests.test_ct_v27_input_flow import sampler_runtime, _case
from tests.test_ct_v27_full_model import full_model_runtime, _construct, _training_batch
from utils.sampling_utils import prune_seqtrack_observation_payload
from utils.v27_input import build_v27_eval_input


def main():
    output = {"schema": "ct_seqtrack.b0_distribution_reproduction.v1"}
    patch = pytest.MonkeyPatch()
    sampler_generator = sampler_runtime.__wrapped__(patch)
    sampler_context = next(sampler_generator)
    model_generator = full_model_runtime.__wrapped__(sampler_context, patch)
    runtime = next(model_generator)
    try:
        model = _construct(runtime, "b0").train()
        batch, _, _ = _training_batch(runtime, model, batch_size=4)
        batch["candidate_id"] = torch.arange(4)
        shared = prune_seqtrack_observation_payload(batch)
        output["actual_observation_route"] = {}
        for arm in ("b0", "b1_cfc", "b1_gru", "full_minus_b3", "full"):
            host = _construct(runtime, arm).train()
            observation = copy.deepcopy(shared)
            # Exact production observation transaction routing in training_step.
            host.use_ct_joint_full = host.use_b1motion_v3 = False
            host.ct_enable_b1 = host.ct_enable_b2 = host.ct_enable_b3 = False
            result = host(observation)
            loss = host.compute_loss(observation, result)
            branches = []
            for view in range(4):
                mask = observation["candidate_id"] == view
                b = host._slice_batch_rows(observation, mask)
                o = host._slice_batch_rows(result, mask)
                item = host.compute_loss(b, o)
                branches.append({"view": view,
                    "auxiliary_shortcut_taken": host._b0_auxiliary_batch(b),
                    "total": float(item["loss_total"]),
                    "transaction": float(item["loss_b0_transaction"]),
                    "box_cloud": float(item["loss_bc"]),
                    "extra_box_cloud_coefficient": float((
                        item["loss_b0_transaction"] - item["loss_total"]) / item["loss_bc"])})
            weighted = host._ct_candidate_weighted_observation_loss(observation, result, loss)
            output["actual_observation_route"][arm] = {
                "branches": branches, "actual_weighted_loss": float(weighted)}

        host = _construct(runtime, "b0").train()
        observation = copy.deepcopy(shared)
        result = host(observation)
        host.config.bc_weight = 0.
        without = host.compute_loss(observation, result)
        host.config.bc_weight = 1.
        with_bc = host.compute_loss(observation, result)
        delta = with_bc["loss_b0_transaction"] - without["loss_b0_transaction"]
        grad = torch.autograd.grad(delta, result["pred_bc"], retain_graph=True)[0]
        expected = torch.autograd.grad(with_bc["loss_bc"], result["pred_bc"], retain_graph=True)[0]
        output["box_cloud_gradient_check"] = {
            "bc_loss": float(with_bc["loss_bc"]),
            "total_delta_when_weight_0_to_1": float(with_bc["loss_total"] - without["loss_total"]),
            "transaction_delta_when_weight_0_to_1": float(delta),
            "gradient_norm_ratio": float(grad.norm() / expected.norm()),
            "max_error_vs_twice_expected_gradient": float((grad - 2 * expected).abs().max())}

        sampler, config, sequence, state, payload, host, _ = _case(sampler_context, variant="b0")
        evaluation, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs, recursive_state=state)
        points = evaluation["points"][0].numpy().reshape(4, 1024, 5)
        output["predicted_history_prior"] = {
            "frame": 8, "prediction_history_error_m": .2,
            "actual_unique_values": [np.unique(x[:, 4]).tolist() for x in points[:3]],
            "original_seqtrack_expected_unique_values": [[.2, .8]] * 3}
        empty = copy.deepcopy(payload)
        empty["this_frame"]["pc"] = sampler_context[1].PointCloud(np.zeros((3, 0)))
        empty.pop("online_recursive_state", None)
        empty.pop("online_motion_aux_state", None)
        train = sampler.motion_processing_mf(empty, config)
        current = train["points"].reshape(4, 1024, 5)[-1]
        output["empty_current_training_sample"] = {
            "valid": float(train["ct_current_observation_valid"]),
            "point_valid_sum": float(train["b0_valid_mask"][-1].sum()),
            "unique_current_input_features": np.unique(current, axis=0).tolist(),
            "gt_box_target": train["box_label"].tolist()}
    finally:
        for generator in (model_generator, sampler_generator):
            try:
                next(generator)
            except StopIteration:
                pass
        patch.undo()
    path = Path(__file__).with_name("b0_distribution_reproduction.json")
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
