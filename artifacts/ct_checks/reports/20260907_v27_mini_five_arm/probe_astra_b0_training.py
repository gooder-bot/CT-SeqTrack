"""Read-only controlled B0 training and crop probes for the astra commit audit."""
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


def main():
    result = {"schema": "ct_seqtrack.astra_b0_training_probe.v1"}
    patch = pytest.MonkeyPatch()
    sg = sampler_runtime.__wrapped__(patch)
    context = next(sg)
    mg = full_model_runtime.__wrapped__(context, patch)
    runtime = next(mg)
    try:
        model = _construct(runtime, "b0").train()
        batch, _, _ = _training_batch(runtime, model, batch_size=4)
        batch["candidate_id"] = torch.arange(4)
        batch = prune_seqtrack_observation_payload(batch)
        assert bool((batch["b0_valid_mask"] == 1).all())
        initial = copy.deepcopy(model.state_dict())
        snapshots = []
        for enabled in (False, True):
            model.load_state_dict(initial)
            model.ct_enable_v27 = enabled
            model.config.ct_enable_v27 = enabled
            model.zero_grad(set_to_none=True)
            torch.manual_seed(314159)
            output = model(batch)
            losses = model.compute_loss(batch, output)
            weighted = model._ct_candidate_weighted_observation_loss(batch, output, losses)
            weighted.backward()
            snapshots.append({
                "outputs": {k: v.detach().clone() for k, v in output.items() if torch.is_tensor(v)},
                "losses": {k: float(v.detach()) for k, v in losses.items()
                           if torch.is_tensor(v) and v.numel() == 1},
                "weighted": float(weighted.detach()),
                "grads": {k: p.grad.detach().clone() for k, p in model.named_parameters()
                          if p.grad is not None},
            })
        old, new = snapshots
        shared = sorted(set(old["outputs"]) & set(new["outputs"]))
        forward_delta = {k: float((old["outputs"][k].float() - new["outputs"][k].float()).abs().max())
                         for k in shared if old["outputs"][k].numel()}
        grad_delta = {k: float((old["grads"][k] - new["grads"][k]).abs().max()) for k in old["grads"]}
        result["dense_same_batch_toggle"] = {
            "scope": "same current B0 real forward/loss, flag off vs on, identical state and CPU RNG; not full parent-commit training replay",
            "all_point_slots_valid": True,
            "common_forward_tensor_count": len(shared),
            "common_forward_max_abs_delta": max(forward_delta.values()),
            "observation_box_max_abs_delta": forward_delta["observation_aux_estimation_boxes"],
            "weighted_loss_off": old["weighted"], "weighted_loss_on": new["weighted"],
            "losses_off": {k: old["losses"][k] for k in ("loss_seg", "loss_bc", "loss_b0_transaction")},
            "losses_on": {k: new["losses"][k] for k in ("loss_seg", "loss_bc", "loss_b0_transaction")},
            "gradient_tensor_count": len(grad_delta),
            "gradient_tensors_bitwise_changed": sum(v != 0 for v in grad_delta.values()),
            "gradient_max_abs_delta": max(grad_delta.values()),
            "gradient_delta_l2_relative": float(torch.cat([(new["grads"][k] - old["grads"][k]).flatten()
                for k in grad_delta]).norm() / torch.cat([old["grads"][k].flatten() for k in grad_delta]).norm()),
        }
        sampler, config, sequence, state, payload, host, _ = _case(context, "b0")
        payload.pop("online_recursive_state", None)
        payload.pop("online_motion_aux_state", None)
        payload.pop("motion_prediction", None)
        payload["ct_observation_only"] = True
        empty_history = copy.deepcopy(payload)
        for frame in empty_history["prev_frames"].values():
            frame["pc"] = context[1].PointCloud(np.empty((3, 0)))
        outcomes = {}
        for enabled in (False, True):
            cfg = copy.deepcopy(config)
            cfg.ct_enable_v27 = enabled
            try:
                row = sampler.motion_processing_mf(copy.deepcopy(empty_history), cfg)
                outcomes[str(enabled)] = {"accepted": True,
                    "box_label": row["box_label"].tolist(),
                    "valid_slots_per_frame": row.get("b0_valid_mask", np.empty(0)).sum(-1).tolist()}
            except AssertionError as error:
                outcomes[str(enabled)] = {"accepted": False, "reason": str(error)}
        result["zero_history_training_sample"] = outcomes

        points_utils = sampler.points_utils
        crop_rows = []
        for idx in (0, 4, 7):
            pc = sequence[idx]["pc"]
            support, anchor = sequence[idx]["3d_bbox"], sequence[7]["3d_bbox"]
            crops = [points_utils.generate_subwindow_with_aroundboxs(pc, support, anchor,
                scale=config.bb_scale, offset=config.bb_offset, canonicalize=flag)
                for flag in (False, True)]
            crop_rows.append({"frame": idx, "ids_equal": bool(np.array_equal(crops[0].point_ids, crops[1].point_ids)),
                "point_count": crops[0].nbr_points(), "coordinate_max_abs_delta": float(np.max(
                    np.abs(crops[0].points - crops[1].points)))})
        result["crop_direct_anchor_vs_roundtrip"] = crop_rows
    finally:
        for generator in (mg, sg):
            try:
                next(generator)
            except StopIteration:
                pass
        patch.undo()
    path = Path(__file__).with_name("astra_b0_training_probe.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
