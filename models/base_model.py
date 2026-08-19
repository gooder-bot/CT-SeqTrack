"""Shared training/evaluation shell for the formal CT-SeqTrack v25 model."""

import csv
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from easydict import EasyDict

from datasets import points_utils
from utils.metrics import (
    TorchNumFrames,
    TorchPrecision,
    TorchRuntime,
    TorchSuccess,
    estimateAccuracy,
    estimateOverlap,
)
from ctseqtrack.data.inference import build_v25_input_dict
from ctseqtrack.data.recursive import RecursiveTrackState
from ctseqtrack.runtime.evaluation import build_ct_joint_diagnostic_row


class BaseModelMF(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config
        self.train_dataloader_length = kwargs.get("train_dataloader_length", None)

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()
        self.runtime = TorchRuntime()

        self.prec_step = TorchPrecision()
        self.success_step = TorchSuccess()
        if (
            bool(getattr(config, "use_ct_joint_full", False))
            and int(getattr(config, "ct_joint_contract_version", 1)) >= 3
        ):
            self.ct_observation_success = TorchSuccess()
            self.ct_raw_search_success = TorchSuccess()

        self.n_frames = TorchNumFrames()
        self._proposal_sequence_diagnostics = []
        self._proposal_test_diagnostics = []
        self._tracking_test_endpoints = []

    def configure_optimizers(self):
        # Experimental modules may contain non-trainable state; keep those
        # tensors out of optimizer parameter groups.
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                trainable_parameters,
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.wd,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_decay_step,
                gamma=self.config.lr_decay_rate,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.config.lr,
                weight_decay=self.config.wd,
                betas=(0.5, 0.999),
                eps=1e-06,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_decay_step,
                gamma=self.config.lr_decay_rate,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.optimizer.lower() == "adamonecycle":
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.config.lr,
                weight_decay=self.config.wd,
                betas=(0.5, 0.999),
                eps=1e-06,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epoch,
                steps_per_epoch=self.train_dataloader_length,
            )
            # The single-cycle learning rate needs to be explicitly updated step by step
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            raise ValueError(
                "Invalid optimizer. Please choose from 'sgd', 'adam', or 'adamonecycle'."
            )

    def compute_loss(self, data, output):
        raise NotImplementedError

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        raise NotImplementedError

    def evaluate_one_sample(self, data_dict, ref_box):
        end_points = self(data_dict)

        estimation_box = end_points["aux_estimation_boxes"]
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        valid_mask = end_points["valid_mask"].squeeze(0).detach().cpu().numpy()

        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(
            ref_box,
            estimation_box_cpu,
            degrees=self.config.degrees,
            use_z=self.config.use_z,
            limit_box=self.config.limit_box,
        )

        return candidate_box, valid_mask, end_points

    @staticmethod
    def _proposal_scalar(mapping, key, default=0.0, column=None):
        value = mapping.get(key)
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if column is not None:
                value = value.reshape(value.shape[0], -1)[0, column]
            else:
                value = value.reshape(-1)[0]
            return float(value.item())
        array = np.asarray(value)
        if column is not None:
            return float(array.reshape(array.shape[0], -1)[0, column])
        return float(array.reshape(-1)[0])

    def _build_ct_joint_diagnostic_row(
        self,
        output,
        data_dict,
        this_box,
        reference_box,
        frame_id,
        previous_ground_truth_box=None,
        older_ground_truth_box=None,
        previous_ground_truth_delta_t=None,
    ):
        return build_ct_joint_diagnostic_row(
            self,
            output,
            data_dict,
            this_box,
            reference_box,
            frame_id,
            previous_ground_truth_box=previous_ground_truth_box,
            older_ground_truth_box=older_ground_truth_box,
            previous_ground_truth_delta_t=previous_ground_truth_delta_t,
        )

    @staticmethod
    def _write_csv_rows(path, rows):
        if not rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _diagnostic_output_dir(self):
        logger = getattr(self, "logger", None)
        log_dir = getattr(logger, "log_dir", None)
        if log_dir is None:
            log_dir = Path(getattr(logger, "save_dir", ".")) / "proposal_diagnostics"
        return Path(log_dir) / "proposal_diagnostics"

    def _write_proposal_test_diagnostics(self):
        rows = self._proposal_test_diagnostics
        if not rows or int(getattr(self, "global_rank", 0)) != 0:
            return
        output_dir = self._diagnostic_output_dir()
        self._write_csv_rows(output_dir / "proposal_endpoints.csv", rows)

        tracklet_rows = []
        for tracklet_id in sorted({int(row["tracklet_id"]) for row in rows}):
            group = [row for row in rows if int(row["tracklet_id"]) == tracklet_id]

            def finite_mean(key, valid_key=None):
                values = [
                    float(row[key])
                    for row in group
                    if (valid_key is None or bool(row[valid_key]))
                    and np.isfinite(float(row[key]))
                ]
                return float(np.mean(values)) if values else float("nan")

            selected = [row for row in group if bool(row["router_applied_gate"])]
            helpful_margin = float(getattr(self.config, "ct_router_help_margin", 0.05))
            tracklet_rows.append(
                {
                    "tracklet_id": tracklet_id,
                    "endpoint_count": len(group),
                    "search_valid_rate": finite_mean("search_valid"),
                    "router_applied_rate": finite_mean("router_applied_gate"),
                    "observation_error_mean": finite_mean("observation_error"),
                    "raw_search_error_mean": finite_mean(
                        "raw_search_error", "search_valid"
                    ),
                    "final_error_mean": finite_mean("final_error"),
                    "selected_helpful_precision": (
                        float(
                            np.mean(
                                [
                                    float(row["observation_error"])
                                    - float(row["final_error"])
                                    > helpful_margin
                                    for row in selected
                                ]
                            )
                        )
                        if selected
                        else float("nan")
                    ),
                    "selected_harm_rate": (
                        float(
                            np.mean(
                                [
                                    float(row["final_error"])
                                    - float(row["observation_error"])
                                    > helpful_margin
                                    for row in selected
                                ]
                            )
                        )
                        if selected
                        else float("nan")
                    ),
                }
            )
        self._write_csv_rows(output_dir / "proposal_tracklets.csv", tracklet_rows)

    def _write_tracking_test_endpoints(self):
        rows = self._tracking_test_endpoints
        if not rows or int(getattr(self, "global_rank", 0)) != 0:
            return
        self._write_csv_rows(
            self._diagnostic_output_dir() / "tracking_endpoints.csv", rows
        )

    def evaluate_one_sequence(self, sequence):
        ious = []
        distances = []
        results_bbs = []
        proposal_diagnostics = []
        recursive_state = None
        for frame_id in range(len(sequence)):
            this_bb = sequence[frame_id]["3d_bbox"]
            observation_diagnostic = None
            observation_diagnostic_valid = False
            if frame_id == 0:
                results_bbs.append(this_bb)
                recursive_state = RecursiveTrackState(
                    tracklet_id=0,
                    tracklet_key=str(
                        sequence[0].get(
                            "tracklet_key", sequence[0].get("tracklet_id", "eval")
                        )
                    ),
                    first_box=this_bb,
                    timestamps={0: sequence[0].get("timestamp")},
                )
            else:
                motion_prediction = None
                if bool(getattr(self.config, "use_b1_prepass_support", False)) and bool(
                    getattr(
                        self, "ct_enable_b1", getattr(self.config, "ct_enable_b1", True)
                    )
                ):
                    predictor = getattr(self, "predict_motion_prepass", None)
                    if predictor is None:
                        raise RuntimeError(
                            "B1 pre-pass support requires a motion predictor"
                        )
                    motion_prediction = predictor(
                        sequence,
                        frame_id,
                        results_bbs,
                        recursive_state=recursive_state,
                    )
                build_kwargs = {}
                if motion_prediction is not None:
                    build_kwargs["motion_prediction"] = motion_prediction
                data_dict, ref_bb = self.build_input_dict(
                    sequence,
                    frame_id,
                    recursive_state.results_bbs,
                    recursive_state=recursive_state,
                    **build_kwargs,
                )
                if torch.sum(data_dict["points"][:, :, :3]) == 0:
                    results_bbs.append(ref_bb)
                    print("Empty pointcloud!")
                else:
                    candidate_box, _, forward_output = self.evaluate_one_sample(
                        data_dict, ref_box=ref_bb
                    )
                    if bool(
                        getattr(self.config, "export_proposal_diagnostics", False)
                    ) and (
                        "ct_search_raw_xy" in forward_output
                        or "motion_prior_xy" in forward_output
                        or bool(getattr(self.config, "use_ct_joint_full", False))
                    ):
                        proposal_diagnostics.append(
                            self._build_ct_joint_diagnostic_row(
                                forward_output,
                                data_dict,
                                this_bb,
                                ref_bb,
                                frame_id,
                                previous_ground_truth_box=sequence[frame_id - 1][
                                    "3d_bbox"
                                ],
                                older_ground_truth_box=(
                                    sequence[frame_id - 2]["3d_bbox"]
                                    if frame_id >= 2
                                    else None
                                ),
                                previous_ground_truth_delta_t=(
                                    (
                                        sequence[frame_id - 1].get("timestamp"),
                                        sequence[frame_id - 2].get("timestamp"),
                                    )
                                    if frame_id >= 2
                                    and sequence[frame_id - 1].get("timestamp")
                                    is not None
                                    and sequence[frame_id - 2].get("timestamp")
                                    is not None
                                    else None
                                ),
                            )
                        )
                    results_bbs.append(candidate_box)
                    diagnostic_tensor = forward_output.get("ct_b0_history_diagnostic")
                    diagnostic_valid_tensor = forward_output.get(
                        "ct_b0_history_diagnostic_valid"
                    )
                    if diagnostic_tensor is not None:
                        observation_diagnostic = (
                            diagnostic_tensor[0].detach().cpu().numpy()
                        )
                    if diagnostic_valid_tensor is not None:
                        observation_diagnostic_valid = bool(
                            diagnostic_valid_tensor[0].detach().item() > 0
                        )

            if frame_id > 0:
                recursive_state.append(
                    frame_id,
                    results_bbs[-1],
                    sequence[frame_id].get("timestamp"),
                    observation_diagnostics=observation_diagnostic,
                    diagnostic_valid=observation_diagnostic_valid,
                )
            ious.append(
                estimateOverlap(
                    this_bb,
                    results_bbs[-1],
                    dim=self.config.IoU_space,
                    up_axis=self.config.up_axis,
                )
            )
            distances.append(
                estimateAccuracy(
                    this_bb,
                    results_bbs[-1],
                    dim=self.config.IoU_space,
                    up_axis=self.config.up_axis,
                )
            )

        self._proposal_sequence_diagnostics = proposal_diagnostics
        return ious, distances, results_bbs

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        start_time = time.time()
        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        epoch_number = int(getattr(self, "current_epoch", 0)) + 1
        if (
            bool(getattr(self.config, "export_v3_candidate_diagnostics", False))
            and self._proposal_sequence_diagnostics
        ):
            if not hasattr(self, "_v3_validation_proposal_diagnostics"):
                self._v3_validation_proposal_diagnostics = []
            for row in self._proposal_sequence_diagnostics:
                row = dict(row)
                row["tracklet_id"] = int(batch_idx)
                row["epoch"] = epoch_number
                row["partition"] = "dev"
                self._v3_validation_proposal_diagnostics.append(row)
        end_time = time.time()
        runtime = end_time - start_time
        n_frames = len(sequence)

        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.success_step(torch.tensor(ious, device=self.device))
        self.prec_step(torch.tensor(distances, device=self.device))

        self.log("success/test", self.success, on_epoch=True)
        self.log("precision/test", self.prec, on_epoch=True)
        proposal_rows = getattr(self, "_proposal_sequence_diagnostics", [])
        b1_rows = [
            row
            for row in proposal_rows
            if bool(row.get("b1_valid", False))
            and np.isfinite(float(row.get("b1_nll", float("nan"))))
            and np.isfinite(float(row.get("learned_motion_error", float("nan"))))
            and np.isfinite(float(row.get("kinematic_error", float("nan"))))
            and np.isfinite(float(row.get("b1_endpoint_error", float("nan"))))
            and np.isfinite(float(row.get("b1_anchor_drift_error", float("nan"))))
        ]
        if b1_rows:
            b1_nll = torch.tensor(
                [float(row["b1_nll"]) for row in b1_rows],
                device=self.device,
                dtype=torch.float32,
            ).mean()
            learned_mse = torch.tensor(
                [float(row["learned_motion_error"]) ** 2 for row in b1_rows],
                device=self.device,
                dtype=torch.float32,
            ).mean()
            kinematic_mse = torch.tensor(
                [float(row["kinematic_error"]) ** 2 for row in b1_rows],
                device=self.device,
                dtype=torch.float32,
            ).mean()
            endpoint_mse = torch.tensor(
                [float(row["b1_endpoint_error"]) ** 2 for row in b1_rows],
                device=self.device,
                dtype=torch.float32,
            ).mean()
            anchor_drift_mse = torch.tensor(
                [float(row["b1_anchor_drift_error"]) ** 2 for row in b1_rows],
                device=self.device,
                dtype=torch.float32,
            ).mean()
            self.log(
                "b1_nll/dev",
                b1_nll,
                on_step=False,
                on_epoch=True,
                batch_size=len(b1_rows),
            )
            self.log(
                "b1_learned_motion_mse/dev",
                learned_mse,
                on_step=False,
                on_epoch=True,
                batch_size=len(b1_rows),
            )
            self.log(
                "b1_physical_rmse/dev",
                torch.sqrt(learned_mse),
                on_step=False,
                on_epoch=True,
                batch_size=len(b1_rows),
            )
            self.log(
                "b1_endpoint_rmse/dev",
                torch.sqrt(endpoint_mse),
                on_step=False,
                on_epoch=True,
                batch_size=len(b1_rows),
            )
            self.log(
                "b1_anchor_drift_rmse/dev",
                torch.sqrt(anchor_drift_mse),
                on_step=False,
                on_epoch=True,
                batch_size=len(b1_rows),
            )
            self.log(
                "b1_kinematic_mse/dev",
                kinematic_mse,
                on_step=False,
                on_epoch=True,
                batch_size=len(b1_rows),
            )
        if (
            hasattr(self, "ct_observation_success")
            and proposal_rows
            and all(
                "observation_iou" in row and "raw_search_iou" in row
                for row in proposal_rows
            )
        ):
            observation_ious = torch.tensor(
                [float(row["observation_iou"]) for row in proposal_rows],
                device=self.device,
            )
            raw_search_ious = torch.tensor(
                [float(row["raw_search_iou"]) for row in proposal_rows],
                device=self.device,
            )
            # Tracking initializes frame 0 from GT.  Include that endpoint so
            # observation/raw-search dev Success is directly comparable with
            # matched B0 success/test and with the official sequence metric.
            frame0_iou = observation_ious.new_ones((1,))
            observation_ious = torch.cat((frame0_iou, observation_ious))
            raw_search_ious = torch.cat((frame0_iou, raw_search_ious))
            self.ct_observation_success(observation_ious)
            self.ct_raw_search_success(raw_search_ious)
            self.log(
                "success_observation/dev",
                self.ct_observation_success,
                on_epoch=True,
                batch_size=len(proposal_rows) + 1,
            )
            self.log(
                "success_raw_search/dev",
                self.ct_raw_search_success,
                on_epoch=True,
                batch_size=len(proposal_rows) + 1,
            )

        self.log("success/test_step", self.success_step, on_step=True, on_epoch=False)
        self.log("precision/test_step", self.prec_step, on_step=True, on_epoch=False)

        self.runtime(
            torch.tensor(runtime, device=self.device),
            torch.tensor(n_frames, device=self.device),
        )

        self.success_step.reset()
        self.prec_step.reset()

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalars(
            "metrics/test",
            {
                "success": self.success.compute(),
                "precision": self.prec.compute(),
            },
            global_step=self.global_step,
        )

        self.logger.experiment.add_scalars(
            "runtime",
            {"runtime": 1.0 / self.runtime.compute()},
            global_step=self.global_step,
        )

        rows = getattr(self, "_v3_validation_proposal_diagnostics", [])
        if rows and int(getattr(self, "global_rank", 0)) == 0:
            logger = getattr(self, "logger", None)
            log_dir = getattr(logger, "log_dir", None)
            if log_dir is None:
                log_dir = getattr(logger, "save_dir", ".")
            epoch_number = int(getattr(self, "current_epoch", 0)) + 1
            self._write_csv_rows(
                Path(log_dir)
                / "candidate_diagnostics"
                / f"epoch_{epoch_number:02d}.csv",
                rows,
            )
        self._v3_validation_proposal_diagnostics = []

    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        start_time = time.time()
        ious, distances, result_bbs, *_ = self.evaluate_one_sequence(sequence)
        test_dataset = getattr(
            getattr(self.trainer, "test_dataloaders", None), "dataset", None
        )
        if test_dataset is None:
            test_loaders = getattr(self.trainer, "test_dataloaders", None)
            if isinstance(test_loaders, (list, tuple)) and test_loaders:
                test_dataset = getattr(test_loaders[0], "dataset", None)
        if test_dataset is not None and hasattr(test_dataset, "get_tracklet_key"):
            tracklet_key = test_dataset.get_tracklet_key(batch_idx)
        elif (
            test_dataset is not None
            and hasattr(test_dataset, "dataset")
            and hasattr(test_dataset.dataset, "get_tracklet_key")
        ):
            source_index = int(batch_idx)
            if hasattr(test_dataset, "tracklet_indices"):
                source_index = int(test_dataset.tracklet_indices[batch_idx])
            tracklet_key = test_dataset.dataset.get_tracklet_key(source_index)
        else:
            tracklet_key = f"tracklet/{int(batch_idx)}"
        if test_dataset is not None and hasattr(
            test_dataset, "get_partition_group_key"
        ):
            partition_group_key = test_dataset.get_partition_group_key(batch_idx)
        elif (
            test_dataset is not None
            and hasattr(test_dataset, "dataset")
            and hasattr(test_dataset.dataset, "get_partition_group_key")
        ):
            source_index = int(batch_idx)
            if hasattr(test_dataset, "tracklet_indices"):
                source_index = int(test_dataset.tracklet_indices[batch_idx])
            partition_group_key = test_dataset.dataset.get_partition_group_key(
                source_index
            )
        else:
            partition_group_key = str(tracklet_key)
        for frame_id, (overlap, distance) in enumerate(zip(ious, distances)):
            self._tracking_test_endpoints.append(
                {
                    "tracklet_id": int(batch_idx),
                    "tracklet_key": str(tracklet_key),
                    "partition_group_key": str(partition_group_key),
                    "frame_id": int(frame_id),
                    "final_iou": float(overlap),
                    "final_distance": float(distance),
                }
            )
        for row in self._proposal_sequence_diagnostics:
            row = dict(row)
            row["tracklet_id"] = int(batch_idx)
            row["partition_group_key"] = str(partition_group_key)
            row["rollout_mode"] = str(
                getattr(self.config, "proposal_inference_mode", "full")
            )
            calibration = getattr(
                self,
                "_ct_action_calibration",
                getattr(self, "_ct_action_threshold_selection", None),
            )
            if isinstance(calibration, dict):
                thresholds = calibration.get("thresholds", {})
                row["calibrated_presence_threshold"] = float(
                    thresholds.get("presence", np.nan)
                )
                row["calibrated_action_threshold"] = float(
                    thresholds.get("action", np.nan)
                )
            eval_partition = getattr(self.config, "ct_eval_partition", None)
            if eval_partition is not None:
                row["partition"] = str(eval_partition)
            source_epoch = getattr(self.config, "ct_source_checkpoint_epoch", None)
            if source_epoch is not None:
                row["epoch"] = int(source_epoch)
            self._proposal_test_diagnostics.append(row)
        end_time = time.time()
        runtime = end_time - start_time
        n_frames = len(sequence)

        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))

        self.log("success/test", self.success, on_epoch=True)
        self.log("precision/test", self.prec, on_epoch=True)
        self.success_step(torch.tensor(ious, device=self.device))
        self.prec_step(torch.tensor(distances, device=self.device))
        self.n_frames(torch.tensor(n_frames, device=self.device))

        self.log("success/test_step", self.success_step, on_step=True, on_epoch=False)
        self.log("precision/test_step", self.prec_step, on_step=True, on_epoch=False)

        self.success_step.reset()
        self.prec_step.reset()

        self.runtime(
            torch.tensor(runtime, device=self.device),
            torch.tensor(n_frames, device=self.device),
        )
        self.logger.experiment.add_scalars(
            "FPS", {"fps": 1.0 / self.runtime.compute()}, global_step=batch_idx
        )

        return result_bbs

    def on_test_epoch_start(self):
        self._proposal_test_diagnostics = []
        self._tracking_test_endpoints = []

    def on_test_epoch_end(self):
        self.logger.experiment.add_scalars(
            "metrics/test/current",
            {"success": self.success.compute(), "precision": self.prec.compute()},
            global_step=self.global_step,
        )

        self.logger.experiment.add_scalars(
            "metrics/fps",
            {
                "runtime": 1.0 / self.runtime.compute(),
            },
            global_step=self.global_step,
        )
        self.logger.experiment.add_scalars(
            "frames",
            {
                "frame": self.n_frames.compute(),
            },
            global_step=self.global_step,
        )
        if bool(getattr(self.config, "export_proposal_diagnostics", False)):
            self._write_proposal_test_diagnostics()
            self._write_tracking_test_endpoints()

    def on_validation_epoch_start(self):
        self._v3_validation_proposal_diagnostics = []


class MotionBaseModelMF(BaseModelMF):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        return build_v25_input_dict(self, sequence, frame_id, results_bbs, **kwargs)
