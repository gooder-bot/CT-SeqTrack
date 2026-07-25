""" 
baseModel.py
Created by zenn at 2021/5/9 14:40
Modified by Aron Lin at Jun 6 17:39:22 CST 2023
"""

import torch
import torch.nn as nn
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision, AverageMeter, TorchRuntime, TorchNumFrames
from utils.metrics import estimateOverlap, estimateAccuracy
from utils.waymo_metrics import estimateWaymoOverlap # only for waymo IOU
import torch.nn.functional as F
import numpy as np
from nuscenes.utils import geometry_utils

from datasets.misc_utils import get_history_frame_ids_and_masks,get_last_n_bounding_boxes
from datasets.misc_utils import (
    build_effective_time_fields,
    build_main_time_fields,
    build_time_fields,
    normalize_timestamp,
    normalize_dynamics_time_mode,
)
from models.state_filter import (
    FixedContinuousDiscreteFilter,
    box_yaw,
    build_trajectory_tube_box,
    point_inside_oriented_crop,
    union_point_clouds,
)
from utils.ct_search import (
    build_time_guided_search_box,
    stratified_search_sample,
)

import time

class BaseModelMF(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config
        self.train_dataloader_length = kwargs.get('train_dataloader_length', None)

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()
        self.runtime = TorchRuntime()

        self.prec_step = TorchPrecision()
        self.success_step = TorchSuccess()

        self.n_frames = TorchNumFrames()


    def configure_optimizers(self):
        # M3 keeps a frozen EMA teacher as a registered submodule so that its
        # state is checkpointed.  Filtering here prevents frozen teacher
        # tensors from entering optimizer parameter groups.
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.optimizer.lower() == 'adamonecycle':
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epoch,
                steps_per_epoch=self.train_dataloader_length)
            # The single-cycle learning rate needs to be explicitly updated step by step
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}} 
        else:
            raise ValueError("Invalid optimizer. Please choose from 'sgd', 'adam', or 'adamonecycle'.")



    def compute_loss(self, data, output):
        raise NotImplementedError

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        raise NotImplementedError

    def evaluate_one_sample(self, data_dict, ref_box):
        end_points = self(data_dict)

        estimation_box = end_points['aux_estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        valid_mask = end_points['valid_mask'].squeeze(0).detach().cpu().numpy()

        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)

        return candidate_box,valid_mask

    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []

        results_bbs = []
        m4_filter_enabled = bool(
            getattr(self.config, "use_m4_state_filter", False))
        m4_tube_enabled = bool(
            getattr(self.config, "use_m4_trajectory_tube", False))
        m4_enabled = m4_filter_enabled or m4_tube_enabled
        m4_filter = self._build_m4_filter() if m4_enabled else None
        m4_diagnostics = []
        for frame_id in range(len(sequence)):  # tracklet
            if frame_id == 0:
                # the first frame
                this_bb = sequence[frame_id]["3d_bbox"]
                prev_bb = sequence[frame_id]["3d_bbox"]
                results_bbs.append(this_bb)
                new_refboxs = [prev_bb] # Update in special cases
                if m4_filter is not None:
                    initial_timestamp = self._m4_timestamp(sequence, frame_id)
                    m4_filter.initialize(
                        this_bb.center,
                        box_yaw(this_bb),
                        initial_timestamp,
                    )
                    m4_diagnostics.append({
                        "frame_id": frame_id,
                        "initialized": True,
                        "reason": "first_frame_initialization",
                    })
            else:
                this_bb = sequence[frame_id]["3d_bbox"]
                current_timestamp = (
                    self._m4_timestamp(sequence, frame_id)
                    if m4_filter is not None else None)
                m4_prediction = (
                    m4_filter.predict(current_timestamp)
                    if m4_filter is not None else None
                )

                # construct input dict
                if m4_enabled:
                    data_dict, ref_bb = self.build_input_dict(
                        sequence,
                        frame_id,
                        results_bbs,
                        m4_prediction=m4_prediction,
                    )
                else:
                    data_dict, ref_bb = self.build_input_dict(
                        sequence, frame_id, results_bbs)
                # run the tracker
                if torch.sum(data_dict['points'][:,:,:3]) == 0:
                    if (m4_filter_enabled
                            and m4_prediction is not None
                            and m4_prediction.get("valid", False)
                            and bool(getattr(
                                self.config, "m4_use_filtered_output", True))):
                        results_bbs.append(m4_filter.box_from_state(ref_bb))
                    else:
                        results_bbs.append(ref_bb)
                    if (m4_filter is not None
                            and not bool(
                                m4_prediction
                                and m4_prediction.get("valid", False))):
                        m4_filter.initialize(
                            ref_bb.center,
                            box_yaw(ref_bb),
                            current_timestamp,
                        )
                    print("Empty pointcloud!")
                    new_refboxs = [ref_bb]
                    m4_diagnostics.append({
                        "frame_id": frame_id,
                        "initialized": True,
                        "prediction_valid": bool(
                            m4_prediction and m4_prediction.get("valid", False)),
                        "measurement_accepted": False,
                        "reason": "empty_pointcloud",
                    })
                else:
                    candidate_box,*_ = self.evaluate_one_sample(data_dict, ref_box=ref_bb)
                    if m4_filter_enabled:
                        if (m4_prediction is not None
                                and m4_prediction.get("valid", False)):
                            update = m4_filter.update(
                                candidate_box.center, box_yaw(candidate_box))
                        else:
                            m4_filter.initialize(
                                candidate_box.center,
                                box_yaw(candidate_box),
                                current_timestamp,
                            )
                            update = {
                                "accepted": True,
                                "reason": "invalid_delta_t_reinitialized",
                                "mahalanobis": 0.0,
                            }
                        if bool(getattr(
                                self.config, "m4_use_filtered_output", True)):
                            candidate_box = m4_filter.box_from_state(candidate_box)
                        results_bbs.append(candidate_box)
                        m4_diagnostics.append({
                            "frame_id": frame_id,
                            "initialized": True,
                            "prediction_valid": bool(
                                m4_prediction and m4_prediction.get("valid", False)),
                            "measurement_accepted": bool(update["accepted"]),
                            "reason": update.get("reason", ""),
                            "mahalanobis": float(update.get("mahalanobis", 0.0)),
                        })
                    elif m4_tube_enabled:
                        m4_filter.observe_direct(
                            candidate_box.center,
                            box_yaw(candidate_box),
                            current_timestamp,
                            velocity_momentum=float(getattr(
                                self.config,
                                "m4_tube_velocity_momentum",
                                0.5,
                            )),
                        )
                        results_bbs.append(candidate_box)
                        m4_diagnostics.append({
                            "frame_id": frame_id,
                            "initialized": True,
                            "prediction_valid": bool(
                                m4_prediction and m4_prediction.get("valid", False)),
                            "measurement_accepted": True,
                            "reason": "tube_only_direct_observation",
                        })
                    else:
                        results_bbs.append(candidate_box)

                if m4_enabled:
                    diagnostic = m4_diagnostics[-1]
                    for key in (
                            "m4_num_points_search_baseline",
                            "m4_num_points_search_tube",
                            "m4_num_points_search_union",
                            "m4_tube_width",
                            "m4_tube_length"):
                        if key in data_dict:
                            diagnostic[key] = float(
                                data_dict[key].detach().cpu().reshape(-1)[0])
                    diagnostic.update(getattr(
                        self, "_m4_last_input_diagnostics", {}))

            
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)

            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)

        self._m4_sequence_diagnostics = m4_diagnostics
        return ious, distances, results_bbs

    def _m4_timestamp(self, sequence, frame_id):
        time_mode = str(getattr(
            self.config, "m4_time_mode", "fixed")).strip().lower()
        if time_mode not in ("fixed", "real"):
            raise ValueError("m4_time_mode must be 'fixed' or 'real'")
        default_step = float(getattr(
            self.config, "m4_fixed_delta_t",
            getattr(
                self.config, "default_time_step",
                getattr(self.config, "time_step", 0.5))))
        if default_step <= 0:
            raise ValueError("m4_fixed_delta_t must be positive")
        timestamp = None
        if time_mode == "real":
            timestamp = normalize_timestamp(
                sequence[frame_id].get("timestamp"))
        if timestamp is None:
            timestamp = float(frame_id) * default_step
        return float(timestamp)

    def _build_m4_filter(self):
        return FixedContinuousDiscreteFilter(
            acceleration_variance=float(getattr(
                self.config, "m4_acceleration_variance", 2.0)),
            yaw_acceleration_variance=float(getattr(
                self.config, "m4_yaw_acceleration_variance", 0.5)),
            measurement_position_variance=float(getattr(
                self.config, "m4_measurement_position_variance", 0.25)),
            measurement_yaw_variance=float(getattr(
                self.config, "m4_measurement_yaw_variance", 0.09)),
            initial_position_variance=float(getattr(
                self.config, "m4_initial_position_variance", 0.25)),
            initial_velocity_variance=float(getattr(
                self.config, "m4_initial_velocity_variance", 4.0)),
            initial_yaw_variance=float(getattr(
                self.config, "m4_initial_yaw_variance", 0.09)),
            initial_yaw_rate_variance=float(getattr(
                self.config, "m4_initial_yaw_rate_variance", 1.0)),
            mahalanobis_gate=float(getattr(
                self.config, "m4_mahalanobis_gate", 0.0)),
            max_delta_t=float(getattr(
                self.config, "m4_max_delta_t", 5.0)),
            covariance_jitter=float(getattr(
                self.config, "m4_covariance_jitter", 1e-8)),
        )

    def _log_m4_diagnostics(self):
        if not self._m4_sequence_diagnostics:
            return
        diagnostics = self._m4_sequence_diagnostics
        for source_key, log_key in (
                ("prediction_valid", "m4/prediction_valid_ratio"),
                ("measurement_accepted", "m4/measurement_accept_ratio"),
                ("m4_oracle_target_center_in_baseline",
                 "m4/oracle_center_recall_baseline"),
                ("m4_oracle_target_center_in_tube",
                 "m4/oracle_center_recall_tube"),
                ("m4_oracle_target_center_in_union",
                 "m4/oracle_center_recall_union"),
                ("m4_num_points_search_baseline",
                 "m4/search_points_baseline"),
                ("m4_num_points_search_tube",
                 "m4/search_points_tube"),
                ("m4_num_points_search_union",
                 "m4/search_points_union"),
                ("m4_tube_width", "m4/tube_width"),
                ("m4_tube_length", "m4/tube_length")):
            values = [item[source_key]
                      for item in diagnostics if source_key in item]
            if values:
                value = torch.tensor(
                    values, device=self.device, dtype=torch.float32).mean()
                self.log(
                    log_key,
                    value,
                    on_step=False,
                    on_epoch=True,
                    batch_size=len(values),
                )

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        start_time = time.time()
        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(sequence)

        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.success_step(torch.tensor(ious, device=self.device))
        self.prec_step(torch.tensor(distances, device=self.device))

        self.log('success/test', self.success, on_epoch=True)
        self.log('precision/test', self.prec, on_epoch=True)

        self.log('success/test_step', self.success_step, on_step=True, on_epoch=False)
        self.log('precision/test_step', self.prec_step, on_step=True, on_epoch=False)
        self._log_m4_diagnostics()

        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))

        self.success_step.reset()
        self.prec_step.reset()

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test',
                                    {'success': self.success.compute(),
                                        'precision': self.prec.compute(),},
                                    global_step=self.global_step)

        self.logger.experiment.add_scalars('runtime',
                                       {'runtime':1.0/self.runtime.compute()},
                                       global_step=self.global_step)


    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        start_time = time.time()
        ious, distances, result_bbs, *_= self.evaluate_one_sequence(sequence)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(sequence)

        
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))

        self.log('success/test', self.success,  on_epoch=True) 
        self.log('precision/test', self.prec,  on_epoch=True) 
        self.success_step(torch.tensor(ious, device=self.device))
        self.prec_step(torch.tensor(distances, device=self.device))
        self.n_frames(torch.tensor(n_frames, device=self.device))

        self.log('success/test_step', self.success_step, on_step=True, on_epoch=False)
        self.log('precision/test_step', self.prec_step, on_step=True, on_epoch=False)
        self._log_m4_diagnostics()

        self.success_step.reset()
        self.prec_step.reset()


        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))
        self.logger.experiment.add_scalars('FPS', {'fps': 1.0/self.runtime.compute()}, global_step=batch_idx)

        return result_bbs

    def on_test_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test/current',
                                    {'success': self.success.compute(),
                                        'precision': self.prec.compute()},
                                    global_step=self.global_step)

        self.logger.experiment.add_scalars('metrics/fps',
                                    {'runtime':1.0/self.runtime.compute(),},
                                    global_step=self.global_step)
        self.logger.experiment.add_scalars('frames',
                                    {'frame':self.n_frames.compute(),},
                                    global_step=self.global_step)

class MotionBaseModelMF(BaseModelMF):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs): # Note: There may be cases of input with empty point clouds
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(frame_id,self.hist_num)
        prev_frames = [sequence[id] for id in prev_frame_ids]
        this_frame = sequence[frame_id]
        this_pc = this_frame['pc']
        bbox_size = this_frame['3d_bbox'].wlh
        prev_pcs = [frame['pc'] for frame in prev_frames]
        ref_boxs = get_last_n_bounding_boxes(results_bbs,valid_mask)
        num_hist = len(valid_mask)
        default_time_step = getattr(
            self.config, 'default_time_step',
            getattr(self.config, 'time_step', 0.1))
        pseudo_time_step = getattr(self.config, 'pseudo_time_step', 0.1)
        use_real_time = getattr(self.config, 'use_real_time', True)
        prev_timestamps = [frame.get('timestamp') for frame in prev_frames]
        current_timestamp = this_frame.get('timestamp')
        real_time_fields = build_time_fields(
            prev_timestamps, current_timestamp,
            frame_ids=prev_frame_ids,
            current_frame_id=frame_id,
            use_real_time=use_real_time,
            default_step=default_time_step,
            pseudo_step=pseudo_time_step)
        relative_timestamps, delta_t_list, local_timestamps, current_timestamp = (
            real_time_fields)
        dynamics_time_mode = normalize_dynamics_time_mode(
            this_frame.get(
                '_ct_dynamics_time_mode',
                getattr(self.config, 'dynamics_time_mode', 'true')))
        effective_time_fields = build_effective_time_fields(
            dynamics_time_mode,
            real_time_fields,
            effective_frame_timestamps=[
                frame.get('_ct_effective_timestamp') for frame in prev_frames
            ],
            effective_current_timestamp=this_frame.get(
                '_ct_effective_timestamp'),
            frame_ids=prev_frame_ids,
            current_frame_id=frame_id,
            default_step=float(getattr(
                self.config, 'dynamics_fixed_delta_t', default_time_step)),
            pseudo_step=pseudo_time_step,
        )
        (effective_relative_timestamps, effective_delta_t_list,
         effective_local_timestamps, effective_current_timestamp) = (
            effective_time_fields)
        main_current_value = float(
            getattr(self.config, 'main_time_current', 0.0))
        point_timestamps, corner_timestamps, main_timestamps = (
            build_main_time_fields(
                valid_mask,
                relative_timestamps,
                local_timestamps,
                num_hist,
                pseudo_step=pseudo_time_step,
                source=getattr(self.config, 'main_time_source', 'real'),
                current_value=main_current_value))

        prev_frame_pcs = []
        for i, prev_pc in enumerate(prev_pcs):
            prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(prev_pc, ref_boxs[i], ref_boxs[0],
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)
            prev_frame_pcs.append(prev_frame_pc)

        this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(
            this_pc, ref_boxs[0], ref_boxs[0],
            scale=self.config.bb_scale,
            offset=self.config.bb_offset)
        baseline_search_points = this_frame_pc.points.T
        expanded_search_points = np.empty(
            (0, baseline_search_points.shape[1]),
            dtype=baseline_search_points.dtype,
        )
        ct_search_box = None
        ct_search_diagnostics = {
            "valid": False,
            "query_delta_t": float(effective_delta_t_list[0]),
        }
        if bool(getattr(self.config, "use_time_guided_search", False)):
            ct_search_box, ct_search_diagnostics = (
                build_time_guided_search_box(
                    ref_boxs,
                    effective_delta_t_list,
                    valid_mask=valid_mask,
                    base_length=float(getattr(
                        self.config, "ct_search_base_length", 4.0)),
                    base_width=float(getattr(
                        self.config, "ct_search_base_width", 2.0)),
                    max_length=float(getattr(
                        self.config, "ct_search_max_length", 16.0)),
                    max_width=float(getattr(
                        self.config, "ct_search_max_width", 6.0)),
                    max_speed=float(getattr(
                        self.config, "ct_search_max_speed", 20.0)),
                    max_displacement=float(getattr(
                        self.config, "ct_search_max_displacement", 12.0)),
                    width_per_second=float(getattr(
                        self.config, "ct_search_width_per_second", 0.25)),
                    min_displacement=float(getattr(
                        self.config, "ct_search_min_displacement", 0.2)),
                ))
            if ct_search_box is not None:
                ct_search_pc = points_utils.generate_subwindow_with_aroundboxs(
                    this_pc,
                    ct_search_box,
                    ref_boxs[0],
                    scale=1.0,
                    offset=0.0,
                )
                expanded_search_points = ct_search_pc.points.T
        num_points_in_search_baseline = this_frame_pc.nbr_points()
        num_points_in_search_tube = 0
        tube_box = None
        m4_diagnostics_enabled = bool(
            getattr(self.config, "use_m4_state_filter", False)
            or getattr(self.config, "use_m4_trajectory_tube", False))
        target_center = None
        target_center_in_baseline = False
        target_center_in_tube = False
        if m4_diagnostics_enabled:
            target_center = np.asarray(
                this_frame["3d_bbox"].center, dtype=np.float64)
            target_center_in_baseline = point_inside_oriented_crop(
                ref_boxs[0],
                target_center,
                scale=self.config.bb_scale,
                offset=self.config.bb_offset,
            )
        m4_prediction = kwargs.get("m4_prediction")
        if (bool(getattr(self.config, "use_m4_trajectory_tube", False))
                and m4_prediction is not None
                and m4_prediction.get("valid", False)):
            tube_box = build_trajectory_tube_box(
                ref_boxs[0],
                m4_prediction,
                base_length=float(getattr(
                    self.config, "m4_tube_base_length", 4.0)),
                base_width=float(getattr(
                    self.config, "m4_tube_base_width", 2.0)),
                sigma_parallel_scale=float(getattr(
                    self.config, "m4_tube_sigma_parallel_scale", 2.0)),
                sigma_perpendicular_scale=float(getattr(
                    self.config, "m4_tube_sigma_perpendicular_scale", 2.0)),
                max_length=float(getattr(
                    self.config, "m4_tube_max_length", 20.0)),
                max_width=float(getattr(
                    self.config, "m4_tube_max_width", 8.0)),
                min_speed=float(getattr(
                    self.config, "m4_tube_min_speed", 0.2)),
            )
            tube_pc = points_utils.generate_subwindow_with_aroundboxs(
                this_pc,
                tube_box,
                ref_boxs[0],
                scale=1.0,
                offset=0.0,
            )
            num_points_in_search_tube = tube_pc.nbr_points()
            this_frame_pc = union_point_clouds(this_frame_pc, tube_pc)
            target_center_in_tube = point_inside_oriented_crop(
                tube_box, target_center)
        num_points_in_search = this_frame_pc.nbr_points()

        # canonical_box = points_utils.transform_box(ref_boxs[0], ref_boxs[0])
        ref_boxs = [
            points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs
        ]

        prev_points_list = [points_utils.regularize_pc(prev_frame_pc.points.T, self.config.point_sample_size)[0] for prev_frame_pc in prev_frame_pcs] #采样到特定数量,这里的策略是在已有的点里面重复随机选，直到达到特定数量

        if bool(getattr(self.config, "use_time_guided_search", False)):
            this_points, ct_search_sampling = stratified_search_sample(
                baseline_search_points,
                expanded_search_points,
                self.config.point_sample_size,
                baseline_ratio=float(getattr(
                    self.config, "ct_search_baseline_ratio", 0.75)),
                min_expansion_points=int(getattr(
                    self.config, "ct_search_min_expansion_points", 32)),
                seed=1,
            )
            ct_search_active = (
                ct_search_sampling["expansion_sample_count"] > 0)
            num_points_in_search = int(len(baseline_search_points))
            if ct_search_active:
                num_points_in_search += int(
                    ct_search_sampling["expansion_available_count"])
        else:
            this_points, idx_this = points_utils.regularize_pc(
                this_frame_pc.points.T,
                self.config.point_sample_size,
                seed=1)
            ct_search_sampling = {
                "baseline_sample_count": int(self.config.point_sample_size),
                "expansion_sample_count": 0,
                "expansion_available_count": 0,
            }
            ct_search_active = False
        seg_mask_prev_list = [geometry_utils.points_in_box(ref_box, prev_points.T[:3,:], 1.25).astype(float) for ref_box,prev_points in zip(ref_boxs,prev_points_list)]#应当只考虑xyz特征

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            for seg_mask_prev in seg_mask_prev_list:
                # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
                # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
                seg_mask_prev[seg_mask_prev == 0] = 0.2
                seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev_list[0].shape, fill_value=0.5)

        timestamp_prev_list = [
            np.full((self.config.point_sample_size, 1), fill_value=timestamp, dtype=np.float32)
            for timestamp in point_timestamps
        ]
        timestamp_this = np.full(
            (self.config.point_sample_size, 1), fill_value=main_current_value, dtype=np.float32)
        prev_points_list = [
        np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]],
                       axis=-1)
        for prev_points, timestamp_prev, seg_mask_prev in zip(
            prev_points_list, timestamp_prev_list, seg_mask_prev_list)
        ]

        this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

        stack_points_list = prev_points_list + [this_points]
        stack_points = np.concatenate(stack_points_list, axis=0)

        ref_box_thetas = [
            ref_box.orientation.degrees * ref_box.orientation.axis[-1]
            if self.config.degrees else ref_box.orientation.radians *
            ref_box.orientation.axis[-1] for ref_box in ref_boxs
        ]
        ref_box_list = [
            np.append(ref_box.center,
                      theta).astype('float32') for ref_box, theta in zip(
                          ref_boxs, ref_box_thetas)
        ]
        ref_boxs_np = np.stack(ref_box_list, axis=0)

        current_delta_t = delta_t_list[0] if len(delta_t_list) > 0 else default_time_step
        current_delta_t_effective = (
            effective_delta_t_list[0] if len(effective_delta_t_list) > 0
            else float(getattr(self.config, 'dynamics_fixed_delta_t', default_time_step)))

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32), 
                     "ref_boxs":torch.tensor(ref_boxs_np[None, :], device=self.device, dtype=torch.float32), 
                     "valid_mask":torch.tensor(valid_mask, device=self.device, dtype=torch.float32).unsqueeze(0), 
                     "bbox_size":torch.tensor(bbox_size[None, :],device=self.device, dtype=torch.float32),
                     "timestamps": torch.tensor(main_timestamps[None, :], device=self.device, dtype=torch.float32),
                     "delta_t": torch.tensor(np.array(delta_t_list, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "delta_t_real": torch.tensor(np.array(delta_t_list, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "delta_t_effective": torch.tensor(np.array(effective_delta_t_list, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "delta_T": torch.tensor(np.array(corner_timestamps, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "timestamps_real": torch.tensor(local_timestamps[None, :], device=self.device, dtype=torch.float32),
                     "delta_T_real": torch.tensor(np.array(relative_timestamps, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "timestamps_effective": torch.tensor(np.asarray(effective_local_timestamps, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "delta_T_effective": torch.tensor(np.array(effective_relative_timestamps, dtype=np.float32)[None, :], device=self.device, dtype=torch.float32),
                     "current_timestamp": torch.tensor([current_timestamp], device=self.device, dtype=torch.float64),
                     "current_effective_timestamp": torch.tensor([effective_current_timestamp], device=self.device, dtype=torch.float64),
                     "current_delta_t": torch.tensor([current_delta_t], device=self.device, dtype=torch.float32),
                     "current_delta_t_real": torch.tensor([current_delta_t], device=self.device, dtype=torch.float32),
                     "current_delta_t_effective": torch.tensor([current_delta_t_effective], device=self.device, dtype=torch.float32),
                     "dynamics_time_mode_id": torch.tensor([
                         {'true': 0, 'fixed': 1, 'shuffled': 2}[dynamics_time_mode]
                     ], device=self.device, dtype=torch.int64),
                     "num_points_in_search": torch.tensor([num_points_in_search], device=self.device, dtype=torch.float32),
                     "ct_search_used": torch.tensor(
                         [ct_search_active],
                         device=self.device, dtype=torch.float32),
                     "ct_search_expansion_ratio": torch.tensor(
                         [ct_search_sampling["expansion_sample_count"]
                          / float(self.config.point_sample_size)],
                         device=self.device, dtype=torch.float32),
                     "ct_search_baseline_points": torch.tensor(
                         [len(baseline_search_points)],
                         device=self.device, dtype=torch.float32),
                     "ct_search_expansion_points": torch.tensor(
                         [ct_search_sampling["expansion_available_count"]],
                         device=self.device, dtype=torch.float32),
                     "ct_search_query_delta_t": torch.tensor(
                         [ct_search_diagnostics.get(
                             "query_delta_t", effective_delta_t_list[0])],
                         device=self.device, dtype=torch.float32),
                     "ct_search_predicted_displacement": torch.tensor(
                         [ct_search_diagnostics.get("displacement", 0.0)],
                         device=self.device, dtype=torch.float32),
                     }
        if m4_diagnostics_enabled:
            data_dict.update({
                "m4_num_points_search_baseline": torch.tensor(
                    [num_points_in_search_baseline], device=self.device,
                    dtype=torch.float32),
                "m4_num_points_search_tube": torch.tensor(
                    [num_points_in_search_tube], device=self.device,
                    dtype=torch.float32),
                "m4_num_points_search_union": torch.tensor(
                    [num_points_in_search], device=self.device,
                    dtype=torch.float32),
                "m4_prediction_valid": torch.tensor(
                    [bool(
                        m4_prediction
                        and m4_prediction.get("valid", False))],
                    device=self.device, dtype=torch.float32),
            })
            # Evaluation-only oracle diagnostics stay outside model input.
            # They are never visible to ``forward`` or tracker decisions.
            self._m4_last_input_diagnostics = {
                "m4_oracle_target_center_in_baseline": float(
                    target_center_in_baseline),
                "m4_oracle_target_center_in_tube": float(
                    target_center_in_tube),
                "m4_oracle_target_center_in_union": float(
                    target_center_in_baseline or target_center_in_tube),
            }
        else:
            self._m4_last_input_diagnostics = {}
        if tube_box is not None:
            data_dict.update({
                "m4_tube_center": torch.tensor(
                    np.asarray(tube_box.center)[None, :],
                    device=self.device, dtype=torch.float32),
                "m4_tube_size": torch.tensor(
                    np.asarray(tube_box.wlh)[None, :],
                    device=self.device, dtype=torch.float32),
                "m4_tube_width": torch.tensor(
                    [float(tube_box.wlh[0])],
                    device=self.device, dtype=torch.float32),
                "m4_tube_length": torch.tensor(
                    [float(tube_box.wlh[1])],
                    device=self.device, dtype=torch.float32),
            })

        if getattr(self.config, 'box_aware', False):
            stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
            hist_points_list = stack_points_split[:num_hist] 
            candidate_bc_prev_list= [
                points_utils.get_point_to_box_distance(hist_points[:, :3], ref_box)
                for hist_points, ref_box in zip(hist_points_list, ref_boxs)
            ]
            candidate_bc_this = np.zeros_like(candidate_bc_prev_list[0])
            candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
            candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]
