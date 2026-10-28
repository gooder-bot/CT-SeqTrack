"""v39 七候选设备端评测、恢复审计及严格的同 run epoch 恢复。"""
import hashlib
import json
import numpy as np
import torch

from models.ct_v38.runtime import capture_rng_state, restore_rng_state, loss_episode_summary, motion_slices
from utils.diagnostic_transfer import tensors_to_numpy
from .config import config_identity, normalize_config, inference_identity
from .identity import SCHEMA, CANDIDATE_SCHEMA, runtime_schema
from .tensor_geometry import pair_box_metrics, inside_boxes


CANDIDATE_NAMES = ('q0', 'prior', 'trusted_propagation', 'g_q0', 'g_prior', 'g_trusted', 'g_reacquire')


def resume_payload(config, *, completed_epoch, complete, rows, steps, sampler=None, run_id=None):
    from .entry import source_identity
    from utils.execution_profile import execution_identity
    source = source_identity()
    return dict(schema=runtime_schema(config), model_schema=SCHEMA, config_sha256=config_identity(config),
        completed_epoch=int(completed_epoch), epoch_complete=bool(complete), rows=int(rows),
        optimizer_steps=int(steps), sampler=sampler, rng=capture_rng_state(), source=source,
        run_id=run_id, execution_identity=execution_identity(config, source))


def validate_resume_payload(payload, config, *, training=True):
    from .entry import source_identity
    from utils.execution_profile import validate_execution_identity
    if not isinstance(payload, dict) or payload.get('schema') != runtime_schema(config):
        raise ValueError('v39 checkpoint runtime schema mismatch')
    if payload.get('model_schema') != SCHEMA or payload.get('config_sha256') != config_identity(config):
        raise ValueError('v39 checkpoint model/config/recipe/precision identity mismatch')
    source = source_identity()
    validate_execution_identity(payload.get('execution_identity'), config, source)
    if payload.get('source') != source:
        raise ValueError('v39 checkpoint source identity mismatch; use its original source')
    if not isinstance(payload.get('run_id'), str) or not payload['run_id']:
        raise ValueError('v39 checkpoint requires nonempty run_id')
    if payload.get('epoch_complete') is not True or payload.get('completed_epoch', 0) < 1:
        raise ValueError('v39 resume requires a complete epoch boundary')
    sampler = payload.get('sampler')
    if not isinstance(sampler, dict) or sampler.get('schema') != 'ct_seqtrack.v32.ready_queue.v2':
        raise ValueError('v39 requires the registered ready-queue sampler manifest')
    content = {key: value for key, value in sampler.items() if key != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != sampler.get('manifest_sha256'):
        raise ValueError('v39 sampler checksum mismatch')
    if sampler.get('initial_seed_policy') != config['v35_seed_policy']:
        raise ValueError('v39 initial seed policy mismatch')
    if (payload['completed_epoch'] != sampler['epoch'] + 1 or payload['rows'] != sampler['rows']
            or payload['optimizer_steps'] != sampler['batches']):
        raise ValueError('v39 checkpoint coverage/update counters disagree')
    if training and not isinstance(payload.get('rng'), dict):
        raise ValueError('v39 resume requires RNG state')
    return payload


def _native(value):
    return value.item() if np.ndim(value) == 0 else value.tolist()


class TrackingEvaluation:
    """按完整轨迹评测；GT 仅用于已提交输出的被动统计。"""
    def __init__(self, config):
        self.config = normalize_config(config)
        self.rows = []
        self._initialized = set()
        self._last_frame = {}
        self._lost_since = {}
        self._last_time = {}
        self._recoveries = []
        self._loss_events = 0

    def add_initialization(self, key, *, scene='unknown', timestamp=None):
        key = str(key)
        if key not in self._initialized:
            self._initialized.add(key)
            self._last_frame[key] = 0
            self.rows.append(dict(tracklet=key, frame=0, success=1., precision=1., iou=1.,
                distance=0., initialization=True, scene_id=scene, timestamp=timestamp))

    def add_singleton_tracks(self, dataset):
        from .data import metadata
        for index, length in enumerate(dataset.lengths):
            if length == 1:
                source = dataset.source
                key = source.get_tracklet_key(index) if hasattr(source, 'get_tracklet_key') else str(index)
                first = metadata(source, index, 0)
                self.add_initialization(str(key), scene=str(first.get('scene_id', 'unknown')),
                                        timestamp=float(first['timestamp']))

    @torch.no_grad()
    def add_batch(self, raw_rows, transaction, output, commit_results=None):
        tx = transaction
        boxes = output.candidates.boxes.detach()
        b = len(raw_rows)
        if boxes.shape != (b, 7, 4) or output.candidates.valid.shape != (b, 7):
            raise ValueError('v39 evaluation requires all seven candidate slots')
        selected = output.selected_index.detach().long()
        row = torch.arange(b, device=boxes.device)
        valid = output.candidates.valid.detach().bool()
        if not bool(valid[row, selected].all()):
            raise ValueError('cannot evaluate an invalid selected candidate')
        if not torch.equal(output.accepted_box.detach(), boxes[row, selected]):
            raise ValueError('evaluation accepted box differs from selected candidate')
        metrics = pair_box_metrics(boxes, tx.targets['target_box'], tx.inputs['box_size'],
                                   tx.targets['target_box_size'])
        verification = output.verification
        if verification['logits'].shape != (b, 7, 3):
            raise ValueError('v39 verification logits must preserve seven slots')
        pool = output.pool
        point_probability = verification['targetness_logits'].detach().sigmoid()
        if point_probability.shape != (b, 7, pool['valid'].shape[1]):
            raise ValueError('candidate targetness and current pool must align')
        support = (pool['valid'][:, None].bool()
                   & inside_boxes(pool['xyz'], boxes, tx.inputs['box_size'])
                   & (point_probability >= .5))
        measured = output.candidates.measurement_eligible.detach().bool()
        support = support & measured[..., None]
        target_points = inside_boxes(pool['xyz'], tx.targets['target_box'][:, None],
                                    tx.targets['target_box_size'])[:, 0] & pool['valid'].bool()
        count = support.sum(-1)
        purity = (support & target_points[:, None]).sum(-1).float() / count.clamp_min(1)
        trusted_target = measured & (metrics['exact_iou'] >= .5) & (count >= 3) & (purity >= .5)
        probabilities = verification['logits'].detach().sigmoid()
        utility = probabilities[..., :2].mean(-1)
        truth_utility = (metrics['success'] + metrics['precision']) * .5
        oracle_index = truth_utility.masked_fill(~valid, -torch.inf).argmax(-1)
        packet = dict(boxes=boxes, target=tx.targets['target_box'], size=tx.inputs['box_size'],
            target_size=tx.targets['target_box_size'], anchor=tx.inputs['anchor_box'],
            selected=selected, quality=output.selected_quality.detach(), valid=valid,
            raw_valid=output.candidates.raw_valid.detach(), measured=measured,
            alias=output.candidates.alias_of.detach(), probability=probabilities, utility=utility,
            support_count=count, purity=purity, trusted_target=trusted_target,
            no_match=verification['region_no_match_logits'].detach().sigmoid(),
            q=verification['q_logits'].detach().sigmoid(), oracle_index=oracle_index,
            full=output.full_executed.detach().bool(),
            policy_full=getattr(output, 'policy_full', output.full_executed).detach().bool(),
            search=output.search_triggered.detach().bool(),
            g_pool_points=pool['valid'].sum(-1), g_pool_targets=target_points.sum(-1),
            displacement=tx.targets['physical_displacement'].norm(dim=-1), **metrics)
        if hasattr(output, 'q0_quality'):
            packet['legacy_q0_quality'] = output.q0_quality.detach()
        diagnostics_cpu = getattr(tx, 'diagnostics_cpu', None)
        if diagnostics_cpu is None:
            packet.update({key: value.detach() for key, value in tx.diagnostics.items() if torch.is_tensor(value)})
        copied = tensors_to_numpy(packet, optimized=True)
        if diagnostics_cpu is not None:
            copied.update(diagnostics_cpu)
        for i, raw in enumerate(raw_rows):
            request = raw['request']
            if request.branch != 4:
                raise ValueError('evaluation must use one uninterrupted branch per tracklet')
            key = str(raw['tracklet_key'])
            if key not in self._initialized:
                if request.frame != 1:
                    raise ValueError('evaluation track must begin with first prediction')
                first = raw['first_frame']
                self.add_initialization(key, scene=str(first.get('scene_id', 'unknown')),
                                        timestamp=float(first['timestamp']))
            if request.frame != self._last_frame[key] + 1:
                raise ValueError('evaluation endpoints must be complete and causal')
            self._last_frame[key] = request.frame
            k = int(copied['selected'][i])
            overlap, distance = float(copied['benchmark_iou'][i, k]), float(copied['distance'][i, k])
            now = float(raw['frames'][request.frame]['timestamp'])
            if key in self._last_time and now <= self._last_time[key]:
                raise ValueError('evaluation timestamps must increase')
            self._last_time[key] = now
            if overlap < .1 and key not in self._lost_since:
                self._lost_since[key] = now
                self._loss_events += 1
            recovered = None
            if overlap >= .5 and key in self._lost_since:
                recovered = now - self._lost_since.pop(key)
                self._recoveries.append(recovered)
            candidates = []
            for j, name in enumerate(CANDIDATE_NAMES):
                observation_eligible = bool(copied['measured'][i,j])
                state_quality = 0.
                if observation_eligible:
                    state_quality = (float(copied['legacy_q0_quality'][i])
                        if 'legacy_q0_quality' in copied else None) if j == 0 else float(copied['q'][i,j])
                candidates.append(dict(name=name, index=j, valid=bool(copied['valid'][i,j]),
                    raw_valid=bool(copied['raw_valid'][i,j]), measurement_eligible=bool(copied['measured'][i,j]),
                    alias_of=int(copied['alias'][i,j]), box_local=copied['boxes'][i,j].tolist(),
                    exact_iou=float(copied['exact_iou'][i,j]), iou=float(copied['benchmark_iou'][i,j]),
                    distance=float(copied['distance'][i,j]), success=float(copied['success'][i,j]),
                    precision=float(copied['precision'][i,j]), utility=float(copied['utility'][i,j]),
                    trusted_score=float(copied['probability'][i,j,2]) if observation_eligible else 0.,
                    qG=float(copied['q'][i,j]), state_quality=state_quality,
                    quality_source='legacy_q0' if j == 0 else ('propagation_zero' if j in (1,2) else 'qG'),
                    no_match=float(copied['no_match'][i,j]), support_count=int(copied['support_count'][i,j]),
                    support_purity=float(copied['purity'][i,j]), trusted_target=bool(copied['trusted_target'][i,j])))
            chosen, q0 = candidates[k], candidates[0]
            oracle = candidates[int(copied['oracle_index'][i])]
            replace_q0 = k != 0 and q0['raw_valid']
            selected_utility = (chosen['success']+chosen['precision'])*.5
            q0_utility = (q0['success']+q0['precision'])*.5
            record = dict(schema=CANDIDATE_SCHEMA, candidates=candidates, selected_index=k,
                selected_source=CANDIDATE_NAMES[k], selected_utility=chosen['utility'],
                trusted_score=chosen['trusted_score'], trusted_target=chosen['trusted_target'],
                trusted_valid=chosen['measurement_eligible'],
                oracle=dict(index=oracle['index'], success=oracle['success'], precision=oracle['precision']),
                mechanism=dict(replaced_real_q0=replace_q0,
                    erroneous_takeover=replace_q0 and selected_utility < q0_utility-1e-8,
                    beneficial_takeover=replace_q0 and selected_utility > q0_utility+1e-8,
                    recovered_bad_q0=replace_q0 and q0['exact_iou'] < .1 and chosen['exact_iou'] >= .5,
                    actual_episode_recovery=recovered is not None))
            diagnostic = {name: _native(value[i]) for name, value in copied.items() if name.startswith('diagnostic_')}
            actual_full = bool(copied['full'][i])
            # 没有调用 B0 的帧不存在 B0 crop，不能把占位 0 解释成空 crop。
            for name in ('diagnostic_crop_point_count', 'diagnostic_crop_target_count',
                         'diagnostic_sampled_point_count', 'diagnostic_sampled_target_count',
                         'diagnostic_novel_target_count'):
                if not actual_full:
                    diagnostic[name] = None
            anchor = copied['anchor'][i].astype(np.float64)
            accepted_world = copied['boxes'][i,k].astype(np.float64)
            target_world = copied['target'][i].astype(np.float64)
            accepted_world[:3] += anchor[:3]; target_world[:3] += anchor[:3]
            diagnostic.update(diagnostic_anchor_box_world=anchor.tolist(),
                diagnostic_accepted_box_world=accepted_world.tolist(), diagnostic_target_box_world=target_world.tolist(),
                diagnostic_first_frame_size_lwh=copied['size'][i].tolist(),
                diagnostic_target_size_lwh=copied['target_size'][i].tolist(),
                diagnostic_gt_xy_displacement=float(copied['displacement'][i]),
                diagnostic_moving=bool(copied['displacement'][i] >= .15),
                diagnostic_g_pool_point_count=int(copied['g_pool_points'][i]),
                diagnostic_g_pool_target_count=int(copied['g_pool_targets'][i]),
                recovery_seconds=recovered)
            if commit_results is not None:
                diagnostic.update(commit_results[i])
            self.rows.append(dict(tracklet=key, frame=int(request.frame), initialization=False,
                success=chosen['success'], precision=chosen['precision'], iou=overlap, distance=distance,
                timestamp=now, scene_id=str(raw['frames'][request.frame].get('scene_id', 'unknown')),
                selected_index=k, selected_source=CANDIDATE_NAMES[k], selected_quality=float(copied['quality'][i]),
                diagnostic_candidates=record, trusted_target=chosen['trusted_target'],
                full_executed=actual_full, policy_full=bool(copied['policy_full'][i]),
                search_triggered=bool(copied['search'][i]),
                **{key: value for key, value in diagnostic.items()
                   if key not in ('full_executed', 'policy_full', 'search_triggered')}))

    def summary(self):
        rows = [row for row in self.rows if not row['initialization']]
        records = [row['diagnostic_candidates'] for row in rows]
        n, total = len(rows), len(self.rows)
        episodes = loss_episode_summary(self.rows)
        writes = [row for row in rows if row.get('trusted_write', row.get('supported', False))]
        wrong = sum(not row['trusted_target'] for row in writes)
        diagnostics = dict(loss_events=self._loss_events, recovered_events=len(self._recoveries),
            unrecovered_events=len(self._lost_since),
            mean_recovery_seconds=float(np.mean(self._recoveries)) if self._recoveries else None,
            unrecovered_elapsed_seconds=sum(self._last_time[key]-start for key,start in self._lost_since.items()),
            loss_iou_threshold=.1, recovery_iou_threshold=.5,
            over5m_frames=sum(row['distance']>5 for row in rows),
            over10m_frames=sum(row['distance']>10 for row in rows),
            loss_episode_total_frames=episodes['total_lost_frames'],
            loss_episode_unrecovered_rate=episodes['unrecovered_rate'],
            loss_episode_frames_median=episodes['all']['frames']['median'],
            loss_episode_frames_p90=episodes['all']['frames']['p90'],
            full_executed_frames=sum(row['full_executed'] for row in rows),
            policy_full_frames=sum(row['policy_full'] for row in rows),
            search_triggered_frames=sum(row['search_triggered'] for row in rows),
            motion_reset_frames=sum(row.get('motion_discontinuity',False) for row in rows),
            raw_target_missing_frames=sum(row.get('diagnostic_target_count')==0 for row in rows),
            raw_present_but_b0_crop_miss_frames=sum((row.get('diagnostic_target_count') or 0)>0
                and row.get('diagnostic_crop_target_count')==0 for row in rows),
            raw_present_but_g_pool_miss_frames=sum((row.get('diagnostic_target_count') or 0)>0
                and row.get('diagnostic_g_pool_target_count')==0 for row in rows),
            selected_bad_available_good_frames=sum(record['candidates'][record['selected_index']]['exact_iou']<.1
                and any(c['valid'] and c['exact_iou']>=.5 for c in record['candidates']) for record in records),
            all_candidates_bad_frames=sum(all(not c['valid'] or c['exact_iou']<.1
                for c in record['candidates']) for record in records))
        for name in ('replaced_real_q0','erroneous_takeover','beneficial_takeover','recovered_bad_q0','actual_episode_recovery'):
            diagnostics[name] = sum(record['mechanism'][name] for record in records)
        reliable = [record for record in records if record['trusted_valid']]
        bins = []
        for index in range(10):
            values = [r for r in reliable if min(int(r['trusted_score']*10),9)==index]
            bins.append(dict(lower=index/10,upper=(index+1)/10,count=len(values),
                mean_probability=float(np.mean([r['trusted_score'] for r in values])) if values else None,
                event_frequency=float(np.mean([r['trusted_target'] for r in values])) if values else None))
        return dict(schema=SCHEMA, metric_mode='benchmark_compat', frames=total, prediction_frames=n,
            tracklets=len(self._initialized), success=100*sum(r['success'] for r in self.rows)/max(total,1),
            precision=100*sum(r['precision'] for r in self.rows)/max(total,1), diagnostics=diagnostics,
            inference_policy=inference_identity(self.config), loss_episodes=episodes, motion_slices=motion_slices(self.rows),
            selection={name:dict(count=sum(r['selected_source']==name for r in rows),
                rate=sum(r['selected_source']==name for r in rows)/n if n else None) for name in CANDIDATE_NAMES},
            trusted=dict(writes=len(writes),coverage=len(writes)/n if n else None,wrong_writes=wrong,
                wrong_write_rate=wrong/len(writes) if writes else None,
                velocity_updates=sum(r.get('trusted_velocity_updated',False) for r in rows),
                longest_untrusted_streak=max((r.get('consecutive_untrusted_outputs',0) for r in rows),default=0)),
            trusted_reliability=dict(population='selected_measurement_candidates', count=len(reliable),
                threshold=self.config.v39_g_trust_threshold, bins=bins,
                brier=float(np.mean([(r['trusted_score']-r['trusted_target'])**2 for r in reliable])) if reliable else None),
            computation=dict(actual_full_rate=diagnostics['full_executed_frames']/n if n else None,
                policy_full_rate=diagnostics['policy_full_frames']/n if n else None,
                cheap_rate=1-diagnostics['full_executed_frames']/n if n else None),
            same_state_oracle=dict(interpretation='passive_one_step_not_closed_loop',
                success=100*(len(self._initialized)+sum(r['oracle']['success'] for r in records))/max(total,1),
                precision=100*(len(self._initialized)+sum(r['oracle']['precision'] for r in records))/max(total,1)))
