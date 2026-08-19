"""SeqTrack3D B0 with the formal CT-SeqTrack v25 composition.

The public class and all state-dict module names are retained.  Historical
CT experiment branches live only in Git history; this file exposes the single
B0 -> B1 -> B2 -> B3 contract used by the v25 configurations.
"""

import torch
from torchmetrics import Accuracy

from models import base_model
from ctseqtrack.model.base_losses import compute_v25_loss
from ctseqtrack.model.builder import initialize_v25
from ctseqtrack.model.observation import (
    build_observability_stats,
    encode_point_time,
    forward_v25,
)
from ctseqtrack.model.prepass import (
    build_motion_prepass_inputs,
    empty_motion_prepass_prediction,
    predict_motion_from_history,
    predict_motion_prepass,
    predict_motion_prepass_contract,
    unbatch_motion_prepass_predictions,
)
from ctseqtrack.runtime.checkpointing import (
    load_v25_checkpoint,
    save_v25_checkpoint,
)
from ctseqtrack.runtime.diagnostics import (
    accumulate_joint_binary_rows,
    binary_curve_metrics_numpy,
    on_train_epoch_end,
    on_train_epoch_start,
    parameter_group_sha256,
    record_parameter_hash,
)
from ctseqtrack.runtime.online import (
    attach_h3_shadow_labels,
    commit_online_recursive_predictions,
    expand_causal_temporal_groups,
    local_prediction_to_world,
    online_motion_prepass,
    online_motion_prepass_batch,
    online_rollout_horizon,
    ordered_online_history_frames,
    prepare_online_recursive_batch,
    prepare_online_state_group,
    process_online_raw,
    recursive_state_for_raw,
    shadow_forward,
    temporal_raw_view,
)
from ctseqtrack.runtime.optimization import (
    auxiliary_microbatch_gradients,
    configure_isolated_optimizers,
    ct_training_step,
    ensure_ct_scalers,
    isolated_optimizer_step,
    record_acquisition_supply,
)


def _build_binary_segmentation_accuracy():
    """Create the metric across supported torchmetrics APIs."""
    try:
        return Accuracy(task="multiclass", num_classes=2, average="none")
    except (TypeError, AssertionError):
        return Accuracy(num_classes=2, average="none")


class SEQTRACK3D(base_model.MotionBaseModelMF):
    """Formal v25 tracker while preserving the original external class."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        initialize_v25(self, config, _build_binary_segmentation_accuracy)

    def on_fit_start(self):
        for module_name, enabled in (
            ("physical_motion_encoder", self.ct_enable_b1),
            ("ct_joint_search_refiner", self.ct_enable_b2),
            ("ct_joint_router", self.ct_enable_b3),
        ):
            module = getattr(self, module_name, None)
            if enabled and (
                module is None
                or any(not parameter.requires_grad for parameter in module.parameters())
            ):
                raise RuntimeError(
                    f"enabled v25 module is not fully trainable: {module_name}"
                )

    def on_save_checkpoint(self, checkpoint):
        save_v25_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint):
        load_v25_checkpoint(self, checkpoint)

    def on_train_epoch_start(self):
        on_train_epoch_start(self)

    def on_train_epoch_end(self):
        on_train_epoch_end(self)

    @staticmethod
    def _binary_curve_metrics_numpy(scores, targets):
        return binary_curve_metrics_numpy(scores, targets)

    def _accumulate_joint_binary_rows(self, data, output):
        return accumulate_joint_binary_rows(self, data, output)

    def encode_point_time(self, points):
        return encode_point_time(self, points)

    @staticmethod
    def is_paired_batch(batch):
        return isinstance(batch, dict) and "view_a" in batch and "view_b" in batch

    def build_observability_stats(self, input_dict, seg_logits, chunk_size):
        return build_observability_stats(self, input_dict, seg_logits, chunk_size)

    @torch.no_grad()
    def predict_motion_from_history(
        self, ref_boxs, delta_t, valid_mask, current_delta_t
    ):
        return predict_motion_from_history(
            self, ref_boxs, delta_t, valid_mask, current_delta_t
        )

    def _build_motion_prepass_inputs_contract(
        self,
        history_boxes,
        history_ids,
        valid_mask,
        history_timestamps,
        current_timestamp,
        effective_history_timestamps,
        effective_current_timestamp,
        dynamics_time_mode_value,
        current_frame_id,
    ):
        return build_motion_prepass_inputs(
            self,
            history_boxes,
            history_ids,
            valid_mask,
            history_timestamps,
            current_timestamp,
            effective_history_timestamps,
            effective_current_timestamp,
            dynamics_time_mode_value,
            current_frame_id,
        )

    def _empty_motion_prepass_prediction(self):
        return empty_motion_prepass_prediction(self)

    @staticmethod
    def _unbatch_motion_prepass_predictions(tensor_prediction, current_delta_t):
        return unbatch_motion_prepass_predictions(tensor_prediction, current_delta_t)

    @torch.no_grad()
    def _predict_motion_prepass_contract(
        self,
        history_boxes,
        history_ids,
        valid_mask,
        history_timestamps,
        current_timestamp,
        effective_history_timestamps,
        effective_current_timestamp,
        dynamics_time_mode_value,
        current_frame_id,
    ):
        return predict_motion_prepass_contract(
            self,
            history_boxes,
            history_ids,
            valid_mask,
            history_timestamps,
            current_timestamp,
            effective_history_timestamps,
            effective_current_timestamp,
            dynamics_time_mode_value,
            current_frame_id,
        )

    @torch.no_grad()
    def predict_motion_prepass(self, sequence, frame_id, results_bbs):
        return predict_motion_prepass(self, sequence, frame_id, results_bbs)

    def _ct_plugin_parameter(self, name):
        return (
            (self.ct_enable_b1 and name.startswith("physical_motion_encoder."))
            or (self.ct_enable_b2 and name.startswith("ct_joint_search_refiner."))
            or (self.ct_enable_b3 and name.startswith("ct_joint_router."))
        )

    @staticmethod
    def _ct_any_plugin_parameter(name):
        return name.startswith(
            ("physical_motion_encoder.", "ct_joint_search_refiner.", "ct_joint_router.")
        )

    @staticmethod
    def _ct_plugin_group(name):
        if name.startswith("physical_motion_encoder."):
            return "b1"
        if name.startswith("ct_joint_search_refiner."):
            return "b2"
        if name.startswith("ct_joint_router."):
            return "b3"
        raise ValueError(f"not a CT plugin parameter: {name}")

    def _build_isolated_optimizer(self, parameters, learning_rate):
        optimizer_name = self.config.optimizer.lower()
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                parameters, lr=learning_rate, momentum=0.9, weight_decay=self.config.wd
            )
        if optimizer_name in ("adam", "adamonecycle"):
            return torch.optim.Adam(
                parameters,
                lr=learning_rate,
                weight_decay=self.config.wd,
                betas=(0.5, 0.999),
                eps=1e-6,
            )
        raise ValueError("Invalid optimizer. Choose 'sgd', 'adam', or 'adamonecycle'.")

    def _ct_parameter_group_sha256(self, group_name):
        return parameter_group_sha256(self, group_name)

    def _ct_record_parameter_hash(self, event):
        return record_parameter_hash(self, event)

    def configure_optimizers(self):
        return configure_isolated_optimizers(self)

    def forward(self, input_dict):
        return forward_v25(self, input_dict)

    def compute_loss(self, data, output):
        return compute_v25_loss(self, data, output)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if (
            isinstance(batch, list)
            and batch
            and isinstance(batch[0], dict)
            and batch[0].get("online_recursive_raw", False)
        ):
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @staticmethod
    def _move_batch_to_device(value, device):
        if torch.is_tensor(value):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, dict):
            return {
                key: SEQTRACK3D._move_batch_to_device(item, device)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return type(value)(
                SEQTRACK3D._move_batch_to_device(item, device) for item in value
            )
        return value

    def _recursive_state_for_raw(self, raw):
        return recursive_state_for_raw(self, raw)

    @staticmethod
    def _ordered_online_history_frames(raw):
        return ordered_online_history_frames(raw)

    def _online_rollout_horizon(self, raw):
        return online_rollout_horizon(self, raw)

    def _prepare_online_state_group(self, raw, state):
        return prepare_online_state_group(self, raw, state)

    def _online_motion_prepass(self, raw, state):
        return online_motion_prepass(self, raw, state)

    @torch.no_grad()
    def _online_motion_prepass_batch(self, raw_state_pairs):
        return online_motion_prepass_batch(self, raw_state_pairs)

    @staticmethod
    def _temporal_raw_view(raw, gap, candidate_id):
        return temporal_raw_view(raw, gap, candidate_id)

    def _expand_causal_temporal_groups(self, group_context):
        return expand_causal_temporal_groups(self, group_context)

    def _process_online_raw(
        self, raw, state, motion_prediction=None, state_diagnostics=None
    ):
        return process_online_raw(
            self,
            raw,
            state,
            motion_prediction=motion_prediction,
            state_diagnostics=state_diagnostics,
        )

    def _prepare_online_recursive_batch(self, raw_items):
        return prepare_online_recursive_batch(self, raw_items)

    def _local_prediction_to_world(self, local_box, anchor_box):
        return local_prediction_to_world(self, local_box, anchor_box)

    def _shadow_forward(self, batch, seed):
        return shadow_forward(self, batch, seed)

    def _attach_h3_shadow_labels(self, batch, output):
        return attach_h3_shadow_labels(self, batch, output)

    def _commit_online_recursive_predictions(self, output):
        return commit_online_recursive_predictions(self, output)

    def _ensure_ct_scalers(self):
        return ensure_ct_scalers(self)

    @staticmethod
    def _slice_batch_rows(batch, row_mask):
        batch_size = int(row_mask.numel())
        return {
            key: (
                value[row_mask]
                if (
                    torch.is_tensor(value)
                    and value.dim() > 0
                    and int(value.shape[0]) == batch_size
                )
                else value
            )
            for key, value in batch.items()
        }

    @staticmethod
    def _assign_parameter_gradients(parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient

    def _ct_record_acquisition_supply(self, loss_dict, population):
        return record_acquisition_supply(self, loss_dict, population)

    def _ct_isolated_optimizer_step(self, loss_dict, auxiliary_gradients=None):
        return isolated_optimizer_step(
            self, loss_dict, auxiliary_gradients=auxiliary_gradients
        )

    def _ct_auxiliary_microbatch_gradients(self, auxiliary_batch):
        return auxiliary_microbatch_gradients(self, auxiliary_batch)

    def training_step(self, batch, batch_idx):
        return ct_training_step(self, batch, batch_idx)


__all__ = ["SEQTRACK3D"]
