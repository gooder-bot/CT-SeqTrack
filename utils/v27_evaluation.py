"""v27 完整递归评估：每个 endpoint 都记实际输出及同状态动作收益。"""
import json
import time
import numpy as np
import torch

from utils.recursive_state import RecursiveTrackState
from utils.tracking_metrics_v27 import box_metrics, metric_contributions


def evaluate_sequence_v27(host, sequence):
    results, overlaps, distances, rows, diagnostics = [], [], [], [], []
    state = None
    for frame_id, frame in enumerate(sequence):
        target = frame['3d_bbox']
        sidecar, output, diagnostic, batch = {}, {}, {}, {}
        acquisition_ms = forward_ms = elapsed_ms = cuda_peak_mb = 0.
        device = getattr(host, 'device', torch.device('cpu'))
        cuda_profile = torch.device(device).type == 'cuda' and torch.cuda.is_available()
        if frame_id == 0:
            final = observation = bounded = raw = target
            state = RecursiveTrackState(
                tracklet_id=int(sequence[0].get('tracklet_id', 0)),
                tracklet_key=str(sequence[0].get('tracklet_key', sequence[0].get('tracklet_id', 'eval'))),
                first_box=target, timestamps={0: frame.get('timestamp')})
            available, applied, score, presence = False, False, 0., 0.
        else:
            if cuda_profile:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            batch, anchor = host.build_input_dict(
                sequence, frame_id, results, recursive_state=state,
                _ct_diagnostic_sidecar=sidecar)
            if cuda_profile:
                torch.cuda.synchronize(device)
            acquired = time.perf_counter()
            final, _, output = host.evaluate_one_sample(batch, ref_box=anchor)
            if cuda_profile:
                torch.cuda.synchronize(device)
                cuda_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            forwarded = time.perf_counter()
            acquisition_ms = (acquired - started) * 1000
            forward_ms = (forwarded - acquired) * 1000
            elapsed_ms = (forwarded - started) * 1000
            local = output['observation_aux_estimation_boxes'][0].detach().clone()
            observation = host._local_prediction_to_world(local, anchor)
            raw_local = local.clone()
            if output.get('ct_search_unmasked_raw_xy') is not None:
                raw_local[:2] = output['ct_search_unmasked_raw_xy'][0].detach()
            raw = host._local_prediction_to_world(raw_local, anchor)
            def scalar(key, default=0.):
                value = output.get(key)
                value = default if value is None else float(value.detach().reshape(-1)[0].cpu())
                if not np.isfinite(value):
                    raise ValueError(f'nonfinite v27 diagnostic: {key}')
                return value
            available = bool(scalar('ct_policy_candidate_valid'))
            correction = output.get('ct_router_bounded_residual_xy')
            if correction is not None and available and bool(torch.isfinite(correction[0]).all()):
                local[:2] += correction[0].detach()
            else:
                available = False
            bounded = host._local_prediction_to_world(local, anchor)
            applied = bool(scalar('ct_router_applied_gate'))
            if applied and not available:
                raise RuntimeError('v27 action applied without a finite structural candidate')
            score, presence = scalar('ct_b3_action_score'), scalar('ct_b2_extension_presence_probability')
            quality = output.get('ct_observation_quality')
            state.append(frame_id, final, frame.get('timestamp'),
                         quality=None if quality is None else quality[0].detach().cpu().numpy())
            if (bool(getattr(host.config, 'export_proposal_diagnostics', False))
                    or bool(getattr(host.config, 'export_v3_candidate_diagnostics', False))):
                # The shared sampler supplies complete ID-based acquisition diagnostics.
                if 'ct_search_raw_xy' in output or bool(getattr(host.config, 'use_ct_joint_full', False)):
                    diagnostic = host._build_ct_joint_diagnostic_row(
                        output, batch, target, anchor, frame_id,
                        acquisition_diagnostics=sidecar.get('acquisition'))
                    diagnostics.append(diagnostic)
        status = getattr(host, '_ct_action_calibration_status', 'not_installed')
        row = dict(tracklet_id=state.tracklet_key,
                   scene_id=str(sequence[0].get('scene_id', sequence[0].get('scene_name', 'unknown'))),
                   frame_id=frame_id, is_initial=frame_id == 0,
                   structural_available=available, action_score=score,
                   presence_score=presence, action_applied=applied,
                   acquisition_available=bool(sidecar.get('acquisition')),
                   acquisition_ms=acquisition_ms, forward_ms=forward_ms,
                   tracking_elapsed_ms=elapsed_ms, cuda_profile_available=cuda_profile,
                   cuda_peak_allocated_mb=cuda_peak_mb,
                   calibration_status=json.dumps(status, sort_keys=True) if isinstance(status, dict) else str(status))
        for mode, prefix in (('benchmark_compat', ''), ('geometry_exact', 'exact_')):
            for name, box in (('observation', observation), ('raw', raw), ('candidate', bounded), ('final', final)):
                iou, distance = box_metrics(box, target,
                    up_axis=host.config.up_axis, mode=mode, dim=int(host.config.IoU_space))
                metric = metric_contributions(iou, distance, mode=mode)
                row[prefix + name + '_success'] = float(metric[0])
                row[prefix + name + '_precision'] = float(metric[1])
                row[prefix + name + '_iou'] = float(iou)
                row[prefix + name + '_distance'] = float(distance)
                if mode == 'benchmark_compat' and name == 'final':
                    overlaps.append(iou)
                    distances.append(distance)
        row['success_gain'] = row['candidate_success'] - row['observation_success']
        row['precision_gain'] = row['candidate_precision'] - row['observation_precision']
        row['utility_gain'] = (row['success_gain'] + row['precision_gain']) / 2
        row['applied_utility_gain'] = float(applied) * row['utility_gain']
        row['raw_success_gain'] = row['raw_success'] - row['observation_success']
        row['raw_precision_gain'] = row['raw_precision'] - row['observation_precision']
        row['raw_utility_gain'] = (row['raw_success_gain'] + row['raw_precision_gain']) / 2
        if sidecar.get('acquisition'):
            # Scalar funnel counts share exactly the IDs used by acquisition.
            for key, value in sidecar['acquisition'].items():
                if isinstance(value, (int, float, bool, np.number)):
                    row['acquisition_' + key] = float(value)
        # Reuse the sampler's exact labels and actual selected slots, independent
        # of optional heavy proposal diagnostics and without forwarding GT.
        b2_enabled = bool(getattr(host.config, 'ct_enable_b2', 'ct_extension_selected_indices' in output))
        row['b2_enabled'] = b2_enabled
        labels = batch.get('ct_extension_prepool_labels', batch.get('ct_extension_labels'))
        indices = output.get('ct_extension_selected_indices')
        valid = output.get('ct_extension_selected_valid_mask')
        if not b2_enabled:
            row.update(acquisition_selected_point_count=0., acquisition_selected_target_count=0.,
                       acquisition_selected_target_bearing=0., acquisition_consensus_point_count=0.,
                       acquisition_consensus_target_count=0.)
        elif labels is not None and indices is not None and valid is not None:
            indices_np = indices.detach().cpu().numpy().reshape(-1).astype(np.int64)
            valid_np = valid.detach().cpu().numpy().reshape(-1) > 0
            labels_np = labels.detach().cpu().numpy().reshape(-1)
            if np.any(indices_np < 0) or np.any(indices_np >= len(labels_np)):
                raise RuntimeError('v27 selected diagnostic index is outside the prepool')
            selected_labels = labels_np[indices_np] > .5
            row['acquisition_selected_point_count'] = float(valid_np.sum())
            row['acquisition_selected_target_count'] = float((selected_labels & valid_np).sum())
            row['acquisition_selected_target_bearing'] = float((selected_labels & valid_np).any())
            inliers = output.get('ct_vote_mode_inlier_mask')
            if inliers is not None:
                inlier_np = inliers.detach().cpu().numpy().reshape(-1).astype(bool)
                if inlier_np.shape != valid_np.shape or np.any(inlier_np & ~valid_np):
                    raise RuntimeError('v27 top-mode inlier mask differs from selected validity')
                row['acquisition_consensus_point_count'] = float(inlier_np.sum())
                row['acquisition_consensus_target_count'] = float((inlier_np & selected_labels).sum())
        for source_key, row_key in (('motion_input_digest', 'acquisition_input_digest'),
                ('ct_acquisition_fallback_reason', 'acquisition_fallback_reason'),
                ('ct_acquisition_parameter_revision', 'acquisition_parameter_revision'),
                ('search_v3_prior_source_id', 'acquisition_source_id')):
            value = batch.get(source_key)
            if torch.is_tensor(value):
                value = value.detach().reshape(-1)[0].cpu().item()
            elif isinstance(value, (list, tuple)):
                value = value[0] if value else ''
            row[row_key] = value if value is not None else ''
        for key in ('ct_vote_mode_unique_count', 'ct_vote_effective_mass',
                    'ct_search_extension_selected_count', 'ct_router_radius', 'ct_router_clip_rate',
                    'ct_b3_expected_success_gain', 'ct_b3_expected_precision_gain'):
            row[key] = float(output[key].detach().reshape(-1)[0].cpu()) if key in output else 0.
        for key, value in row.items():
            if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                raise ValueError(f'nonfinite v27 endpoint: {key}')
        rows.append(row)
        results.append(final)
    host._ct_v27_sequence_endpoints = rows
    host._proposal_sequence_diagnostics = diagnostics
    host._b3_sequence_rollouts = []
    return overlaps, distances, results
