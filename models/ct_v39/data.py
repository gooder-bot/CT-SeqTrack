"""v39 metadata 端点、GPU 动态准备与唯一递归状态写入者。"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from functools import partial
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.ct_v38.data import (BatchBuilder as LegacyBuilder, RawEndpointDataset as LegacyDataset,
    ReadyQueueBatchSampler, EndpointRequest, stable_seed, metadata, box_array, box_size,
    raw_collate, raw_dataset_factory, KNOWN_GT_BOX, MISSING_BOX, PREDICTED_BOX,
    PERTURBED_BOX, V38AcceptedState)
from .config import normalize_config
from .raw_service import RawFrameService
from .gpu_prepare import inside, canonical_points, sample_indices, voxel_indices, pad_pool, point_targets
from utils.diagnostic_transfer import tensors_to_numpy


class RawEndpointDataset(LegacyDataset):
    """DataLoader 只排 metadata；主进程 raw service 按模型路径解码。"""
    def __init__(self, source, service):
        super().__init__(source)
        self.service = service
        if hasattr(source, 'ct_pointcloud_cache_bytes'):
            source.ct_pointcloud_cache_bytes = 0

    def __getitem__(self, request):
        if not isinstance(request, EndpointRequest):
            raise TypeError('v39 requires EndpointRequest')
        history = tuple(max(0, request.frame - 3 + i) for i in range(3))
        frames = {i: metadata(self.source, request.tracklet, i) for i in set(history + (0, request.frame))}
        key = self.source.get_tracklet_key(request.tracklet) if hasattr(self.source, 'get_tracklet_key') else str(request.tracklet)
        return dict(request=request, frames=frames, first_frame=frames[0], history_ids=history,
                    tracklet_key=str(key), _source=self.source, _service=self.service,
                    track_length=self.lengths[request.tracklet])


def build_loaders(config, roles=('train', 'val'), sources=None, raw_service=None):
    config = normalize_config(config)
    service = raw_service or RawFrameService(config)
    result = {}
    for role in roles:
        data_role = ('dev' if config.dataset.startswith('kitti') else 'test') if role == 'val' else role
        source = sources[role] if sources is not None else raw_dataset_factory(config, data_role)
        raw = RawEndpointDataset(source, service)
        sampler = ReadyQueueBatchSampler(raw.lengths, config.batch_size, seed=config.seed,
            training=role == 'train', short_window=config.v31_short_window,
            long_window=config.v31_long_window, curriculum_epochs=config.v31_curriculum_epochs,
            source_sha256=raw.source_sha256, reserve_windows=config.v32_reserve_windows,
            seed_translation=config.v32_seed_translation, seed_yaw_degrees=config.v32_seed_yaw_degrees,
            initial_seed_policy=config.v35_seed_policy)
        result[role] = DataLoader(raw, batch_sampler=sampler, num_workers=0, collate_fn=raw_collate,
            generator=torch.Generator().manual_seed(stable_seed(config.seed, role)))
    return result


@dataclass
class Transaction:
    inputs: dict
    targets: dict
    diagnostics: dict
    contexts: list
    rows: list
    device: torch.device
    training: bool
    raw: list = field(default_factory=list)
    templates: dict = field(default_factory=dict)
    b0_indices: torch.Tensor | None = None
    b0_inputs: dict | None = None
    b0_targets: dict | None = None
    timings: dict = field(default_factory=dict)
    diagnostics_cpu: dict = field(default_factory=dict)

    def audit_batch(self):
        return dict(self.inputs, **self.targets, **self.diagnostics)


class BatchBuilder(LegacyBuilder):
    def __init__(self, config, raw_service=None):
        self.config = normalize_config(config)
        self.service = raw_service or RawFrameService(self.config)
        self.states = {}
        self._pending = None
        self._raw_memo = None
        self.materialization_count = 0

    def reset(self):
        self.states.clear()
        self._pending = None

    def clear_materialization_cache(self):
        pass

    def abort(self):
        # 只撤销本事务的新初始化，既有 accepted 状态没有修改。
        if self._pending is not None:
            for context in self._pending.contexts:
                if context['new_state']:
                    self.states.pop(context['row']['request'].state_key, None)
        self._pending = None

    def _raw(self, row, frame, device, anchor):
        meta=row['frames'].get(frame)
        if meta is None:meta=metadata(row['_source'],row['request'].tracklet,frame)
        key=self.service.key(row['_source'],row['request'].tracklet,frame,meta)
        cached=self.service.cached_tensors(key,device,anchor)
        if cached is not None:
            return (*cached,key,None)
        key, cloud = self.service.fetch(row['_source'], row['request'].tracklet, frame, meta)
        xyz, ids = self.service.tensors(key, cloud, device, anchor)
        return xyz, ids, key, cloud

    def _initialize(self, row, device):
        r = row['request']
        seed_frames = range(max(0, r.frame - 3), r.frame)
        size = box_size(row['first_frame']['3d_bbox'], row['first_frame'])
        boxes = {i:box_array(row['frames'][i]['3d_bbox']) for i in seed_frames}
        times = {i:float(row['frames'][i]['timestamp']) for i in seed_frames}
        seed_offset=np.zeros(4,np.float64);provenance=KNOWN_GT_BOX
        if r.branch in (1,2,3) and r.frame != 1:
            rng=np.random.default_rng(stable_seed(self.config.seed,row['tracklet_key'],r.state_key,'window_seed'))
            local_xy=rng.uniform(-self.config.v32_seed_translation,self.config.v32_seed_translation,2)
            yaw=boxes[r.frame-1][3];c,s=np.cos(yaw),np.sin(yaw)
            seed_offset[:2]=np.array([[c,-s],[s,c]])@local_xy
            limit=np.deg2rad(self.config.v32_seed_yaw_degrees);seed_offset[3]=rng.uniform(-limit,limit)
            for box in boxes.values():
                box[:3]+=seed_offset[:3]
                box[3]=np.arctan2(np.sin(box[3]+seed_offset[3]),np.cos(box[3]+seed_offset[3]))
            provenance=PERTURBED_BOX
        trusted=[(boxes[i].copy(),times[i]) for i in seed_frames][-2:]
        velocity=np.zeros(3,np.float64)
        if len(trusted)==2:
            gap=trusted[-1][1]-trusted[-2][1]
            if gap<=0:raise ValueError('seed timestamps must strictly increase')
            velocity=(trusted[-1][0][:3]-trusted[-2][0][:3])/gap
        # seed 支持也遵守新 FP32 几何；不复制整云回 CPU 或污染 metadata rows。
        counts=[]
        for frame in seed_frames:
            xyz,_,_,_=self._raw(row,frame,device,boxes[frame])
            local=torch.tensor([0.,0.,0.,boxes[frame][3]],device=device,dtype=torch.float32)
            counts.append(inside(xyz,local,torch.tensor(size,device=device,dtype=torch.float32)).sum())
        counts=torch.stack(counts).cpu().tolist()
        strong_times=[times[frame] for frame,count in zip(seed_frames,counts) if count>=self.config.v39_g_min_support]
        last=times[r.frame-1]
        state=V38AcceptedState(boxes=boxes,times=times,next_frame=r.frame,size=size,trusted=trusted,
            transitions={i:True for i in seed_frames},trusted_velocity=velocity,
            trusted_velocity_valid=len(trusted)==2,previous_quality=1.,
            last_strong_time=max(strong_times) if strong_times else None,last_supported_time=last,
            initialization_time=last,previous_innovation=0.,
            box_provenance={i:provenance for i in seed_frames},seed_offset=seed_offset,
            last_velocity_update_time=last if len(trusted)==2 else None)
        state.motion_epoch = 0
        state.frame_motion_epoch = {frame: 0 for frame in state.boxes}
        state.motion_reset_pending = False
        state.search_anchor = state.trusted[-1][0].copy()
        state.search_failures = 0
        state.last_full_time = state.initialization_time
        state.consecutive_cheap = 0
        state.templates = []
        initial = box_array(row['first_frame']['3d_bbox'])
        state.fixed_template = self._template_from_frame(row, 0, initial, state.size, device, 'initial')
        for frame in list(seed_frames)[-2:]:
            if frame != 0:
                state.templates.append(self._template_from_frame(row, frame, state.boxes[frame], state.size, device, 'seed'))
        return state

    def _template_from_frame(self, row, frame, box, size, device, kind):
        meta = row['frames'].get(frame) or metadata(row['_source'], row['request'].tracklet, frame)
        xyz,ids,_,_=self._raw(row,frame,device,box)
        local_box = torch.tensor(np.r_[np.zeros(3), box[3]], device=device, dtype=torch.float32)
        sz = torch.tensor(size, device=device, dtype=torch.float32)
        selected = sample_indices(ids, inside(xyz, local_box, sz), self.config.v39_template_points,
            stable_seed(self.config.seed, row['request'].epoch, row['tracklet_key'],
                        row['request'].branch, frame, 'template', kind))
        canonical = canonical_points(xyz[selected], local_box)
        gt = box_array(meta['3d_bbox']); gt[:3] -= box[:3]
        gt = torch.tensor(gt, device=device, dtype=torch.float32)
        gt_size = torch.tensor(box_size(meta['3d_bbox'], meta), device=device, dtype=torch.float32)
        labels=tensors_to_numpy(dict(xyz=canonical,ids=ids[selected],
            label_xyz=canonical_points(xyz[selected],gt),label_valid=inside(xyz[selected],gt,gt_size)),optimized=True)
        return dict(**{key:value.copy() for key,value in labels.items()},
            timestamp=float(meta['timestamp']), frame=frame, kind=kind,
        )

    def begin(self, rows, device, *, training):
        if self._pending is not None:
            raise RuntimeError('previous v39 transaction has not been committed')
        before = set(self.states)
        try:
            return self._begin(rows, device, training=training)
        except BaseException:
            # _pending 可能尚未建立：逐行初始化期间失败也必须完整撤销新状态。
            self.abort()
            for key in set(self.states) - before:
                self.states.pop(key, None)
            raise

    def _begin(self, rows, device, *, training):
        if not rows or len({r['request'].state_key for r in rows}) != len(rows):
            raise ValueError('v39 requires nonempty independent window rows')
        device = torch.device(device)
        inputs, targets, contexts = [], [], []
        for row in rows:
            r = row['request']
            fresh = r.starts_window
            if fresh:
                if r.state_key in self.states:
                    raise RuntimeError('duplicate window initialization')
                self.states[r.state_key] = self._initialize(row, device)
            if r.state_key not in self.states:
                raise RuntimeError('missing accepted predecessor; GT reseeding forbidden')
            state = self.states[r.state_key]
            if state.next_frame != r.frame:
                raise RuntimeError('out of order v39 endpoint')
            now = float(row['frames'][r.frame]['timestamp'])
            effective_now=float(row['frames'][r.frame].get('_ct_effective_timestamp',now))
            anchor = state.boxes[r.frame - 1].copy()
            history = np.stack([state.boxes[i] for i in row['history_ids']])
            history[:, :3] -= anchor[:3]
            exists = np.asarray([r.frame - 3 + j >= 0 for j in range(3)])
            times = np.asarray([float(row['frames'][i]['timestamp']) - now for i in row['history_ids']], np.float64)
            effective_times=np.asarray([float(row['frames'][i].get('_ct_effective_timestamp',row['frames'][i]['timestamp']))
                                        -effective_now for i in row['history_ids']],np.float64)
            if not (times < 0).all() or not (effective_times < 0).all():
                raise ValueError('history must be strictly in the past')
            provenance = np.asarray([state.box_provenance[i] if valid else MISSING_BOX
                                     for i, valid in zip(row['history_ids'], exists)])
            pair = exists[:-1] & exists[1:]
            for j in range(2):
                left, right = row['history_ids'][j:j+2]
                pair[j] &= bool(state.transitions.get(right, False))
                pair[j] &= state.frame_motion_epoch.get(left) == state.frame_motion_epoch.get(right) == state.motion_epoch
            fallback = state.trusted[-1][0].copy()
            if state.trusted_velocity_valid:
                fallback[:3] += state.trusted_velocity * (now-state.trusted[-1][1])
            fallback[:3] -= anchor[:3]
            route_window = (training and stable_seed(self.config.seed, r.epoch, row['tracklet_key'],
                r.branch, r.window_start, 'route') / 2**32 < self.config.v39_route_window_fraction)
            item = dict(history_boxes=history.astype(np.float32), history_valid=exists,
                history_pair_valid=pair, history_times=effective_times.astype(np.float32), current_dt=np.float32(-effective_times[-1]),
                frame_times=np.r_[times, 0.].astype(np.float32), box_size=state.size.astype(np.float32),
                anchor_box=anchor.astype(np.float64), fallback_box=fallback.astype(np.float32),
                history_box_provenance=provenance.astype(np.int64),
                history_is_initial=exists & (np.asarray(row['history_ids']) == 0) & (provenance == KNOWN_GT_BOX),
                window_seed_offset=state.seed_offset.astype(np.float32), branch_id=np.int64(r.branch),
                trusted_velocity=state.trusted_velocity.astype(np.float32), trusted_velocity_valid=np.bool_(state.trusted_velocity_valid),
                previous_quality=np.float32(state.previous_quality), weak_age=np.float32(self._strong_age(state, now)),
                supported_age=np.float32(now-state.last_supported_time), previous_innovation=np.float32(state.previous_innovation),
                previous_is_propagation=np.bool_(state.previous_is_propagation),
                consecutive_propagations=np.int64(state.consecutive_propagations),
                consecutive_untrusted_outputs=np.int64(state.consecutive_untrusted_outputs),
                trusted_velocity_age=np.float32(self._velocity_age(state, now)),
                full_age=np.float32(now-state.last_full_time), consecutive_cheap=np.int64(state.consecutive_cheap),
                search_failures=np.int64(state.search_failures), route_window=np.bool_(route_window))
            gt = box_array(row['frames'][r.frame]['3d_bbox']); gt[:3] -= anchor[:3]
            previous = box_array(row['frames'][r.frame-1]['3d_bbox'])
            future, dt, valid = np.zeros((2,2),np.float32), np.zeros(2,np.float32), np.zeros(2,bool)
            if training and hasattr(row['_source'], 'get_frame_metadata'):
                for j,h in enumerate((2,4)):
                    index = r.frame - 1 + h
                    if index < row['track_length']:
                        # 缺 annotation/非法未来标签只屏蔽该 horizon；I/O 故障不能被吞掉。
                        try:
                            future_meta=row['_source'].get_frame_metadata(r.tracklet,index)
                        except KeyError:
                            continue
                        if not isinstance(future_meta,dict) or future_meta.get('3d_bbox') is None:
                            continue
                        try:
                            elapsed=float(future_meta.get('timestamp'))-float(row['frames'][r.frame-1]['timestamp'])
                            displacement=box_array(future_meta['3d_bbox'])[:2]-previous[:2]
                        except (TypeError,ValueError,IndexError):
                            continue
                        same_scene=future_meta.get('scene_id')==row['frames'][r.frame]['scene_id'] if 'scene_id' in row['frames'][r.frame] else True
                        if elapsed>0 and np.isfinite(elapsed) and np.isfinite(displacement).all() and same_scene:
                            future[j],dt[j],valid[j]=displacement,elapsed,True
            target = dict(target_box=gt.astype(np.float32), target_box_size=box_size(row['frames'][r.frame]['3d_bbox'],row['frames'][r.frame]).astype(np.float32),
                physical_displacement=(gt[:2]+anchor[:2]-previous[:2]).astype(np.float32), physical_valid=np.bool_(True),
                future_displacement=future, future_dt=dt, future_valid=valid)
            inputs.append(item); targets.append(target)
            contexts.append(dict(row=row,state=state,anchor=anchor,timestamp=now,new_state=fresh))
        def stack(values):
            return {k: torch.as_tensor(np.stack([v[k] for v in values]),device=device) for k in values[0]}
        tx = Transaction(stack(inputs),stack(targets),{},contexts,rows,device,training)
        self._pending = tx
        # 当前 raw 一批并发解码，同物理帧去重；历史仅在初始化/完整路径需要。
        self.service.prefetch_many([(r['_source'],r['request'].tracklet,r['request'].frame,
                                     r['frames'][r['request'].frame]) for r in rows],device)
        tx.raw = [self._raw(c['row'],c['row']['request'].frame,device,c['anchor'])[:2] for c in contexts]
        self._pack_templates(tx)
        tx.diagnostics['diagnostic_target_count'] = torch.stack([inside(xyz,tx.targets['target_box'][i],tx.targets['target_box_size'][i]).sum()
            for i,(xyz,_) in enumerate(tx.raw)])
        tx.diagnostics['diagnostic_raw_point_count'] = torch.tensor([len(ids) for _,ids in tx.raw],device=device)
        tx.diagnostics['diagnostic_gt_xy_displacement'] = tx.targets['physical_displacement'].norm(dim=-1)
        return tx

    def _pack_templates(self, tx):
        b,p = len(tx.rows),self.config.v39_template_points
        xyz=torch.zeros((b,3,p,3),device=tx.device); valid=torch.zeros((b,3,p),device=tx.device,dtype=torch.bool)
        gt_xyz=torch.zeros_like(xyz); gt_valid=torch.zeros_like(valid); ages=torch.zeros((b,3),device=tx.device)
        for i,c in enumerate(tx.contexts):
            for j,template in enumerate([c['state'].fixed_template]+c['state'].templates[-2:]):
                n=min(p,len(template['xyz']))
                xyz[i,j,:n]=torch.as_tensor(template['xyz'][:n],device=tx.device)
                valid[i,j,:n]=True
                gt_xyz[i,j,:n]=torch.as_tensor(template['label_xyz'][:n],device=tx.device)
                gt_valid[i,j,:n]=torch.as_tensor(template['label_valid'][:n],device=tx.device)
                ages[i,j]=max(0.,c['timestamp']-template['timestamp'])
        tx.templates=dict(xyz=xyz,valid=valid,ages=ages)
        tx.targets.update(template_xyz_gt=gt_xyz,template_target_mask=gt_valid)

    def prepare_current_G(self,tx,boxes,*,allowed=None,maximum=None):
        values=[]
        n=int(self.config.v39_crop_points if maximum is None else maximum)
        pool_kind='g' if maximum is None else 'g_refine'
        if boxes.ndim==2: boxes=boxes[:,None]
        for i,(xyz,ids) in enumerate(tx.raw):
            indexes=[]
            for j in range(boxes.shape[1]):
                if allowed is not None and not bool(allowed[i,j]): continue
                mask=inside(xyz,boxes[i,j].detach(),tx.inputs['box_size'][i],self.config.bb_scale,self.config.bb_offset)
                request=tx.rows[i]['request']
                indexes.append(sample_indices(ids,mask,n,stable_seed(self.config.seed,request.epoch,
                    tx.rows[i]['tracklet_key'],request.branch,request.frame,pool_kind,j)))
            index=torch.unique(torch.cat(indexes),sorted=True) if indexes else ids[:0]
            values.append((xyz[index],ids[index]))
        return pad_pool(values)

    def prepare_search(self,tx,centers,prior,active):
        values=[]
        for i,(xyz,ids) in enumerate(tx.raw):
            if not bool(active[i]):
                values.append((xyz[:0],ids[:0]));continue
            global_search=int(tx.inputs['search_failures'][i])>=self.config.v39_global_search_after_failures
            if global_search:
                mask=torch.ones_like(ids,dtype=torch.bool);maximum=self.config.v39_recovery_global_points
            else:
                sigma=prior.log_sigma[i].detach().exp().amax() if bool(prior.valid[i]) else xyz.new_tensor(2.)
                radius=(4.+3.*sigma+2.*tx.inputs['supported_age'][i]).clamp(8.,24.)
                mask=((xyz[:,None,:2]-centers[i,None,:,:2]).square().sum(-1)<=radius.square()).any(-1)
                maximum=self.config.v39_recovery_local_points
            request=tx.rows[i]['request']
            index=voxel_indices(xyz,ids,mask,maximum,stable_seed(self.config.seed,request.epoch,
                tx.rows[i]['tracklet_key'],request.branch,request.frame,'coarse'),voxel=self.config.v39_coarse_voxel_size)
            values.append((xyz[index],ids[index]))
        return pad_pool(values)

    def materialize_B0(self,tx,indices,crop_boxes):
        indices=indices.detach().long()
        tx.b0_indices=indices
        if not len(indices): return None,None
        host_indices=indices.cpu().tolist()
        requests=[]
        for i in host_indices:
            row=tx.rows[i]
            requests.extend((row['_source'],row['request'].tracklet,f,row['frames'][f]) for f in set(row['history_ids']))
        self.service.prefetch_many(requests,tx.device)
        n=self.config.point_sample_size
        points=torch.zeros((len(indices),4,n,5),device=tx.device)
        ids_out=torch.full((len(indices),4,n),-1,device=tx.device,dtype=torch.long)
        valid=torch.zeros_like(ids_out,dtype=torch.bool)
        labels=torch.full_like(ids_out,-1);bc=torch.zeros((len(indices),4,n,9),device=tx.device)
        target_boxes=torch.zeros((len(indices),4,4),device=tx.device)
        crop_count=torch.zeros(len(tx.rows),device=tx.device,dtype=torch.long)
        crop_target_count=torch.zeros_like(crop_count);sample_count=torch.zeros_like(crop_count);sample_target_count=torch.zeros_like(crop_count)
        for output_i,i in enumerate(host_indices):
            c=tx.contexts[i];row=c['row'];r=row['request']
            for slot,frame in enumerate(row['history_ids']+(r.frame,)):
                xyz,ids=tx.raw[i] if slot==3 else self._raw(row,frame,tx.device,c['anchor'])[:2]
                crop=crop_boxes[i].detach() if slot==3 else tx.inputs['history_boxes'][i,slot]
                mask=inside(xyz,crop,tx.inputs['box_size'][i],self.config.bb_scale,self.config.bb_offset)
                if slot<3: mask=mask & tx.inputs['history_valid'][i,slot]
                selected=sample_indices(ids,mask,n,stable_seed(self.config.seed,r.epoch,row['tracklet_key'],r.branch,r.frame,slot))
                count=len(selected)
                points[output_i,slot,:count,:3]=xyz[selected]
                points[output_i,slot,:count,3]=float(row['frames'][frame]['timestamp'])-c['timestamp']
                hint=inside(xyz[selected],crop,tx.inputs['box_size'][i]).float()
                if slot<3:
                    if int(tx.inputs['history_box_provenance'][i,slot])!=KNOWN_GT_BOX: hint=.2+.6*hint
                else: hint=torch.full_like(hint,.5)
                points[output_i,slot,:count,4]=hint
                ids_out[output_i,slot,:count]=ids[selected];valid[output_i,slot,:count]=True
                meta=row['frames'][frame];target=box_array(meta['3d_bbox']);target[:3]-=c['anchor'][:3]
                target=torch.tensor(target,device=tx.device,dtype=torch.float32)
                size=torch.tensor(box_size(meta['3d_bbox'],meta),device=tx.device,dtype=torch.float32)
                target_boxes[output_i,slot]=target
                label,distance=point_targets(points[output_i,slot,:,:3],valid[output_i,slot],target,size)
                labels[output_i,slot],bc[output_i,slot]=label,distance
                if slot==3:
                    crop_count[i]=mask.sum();crop_target_count[i]=(mask & inside(xyz,target,size)).sum()
                    sample_count[i]=count;sample_target_count[i]=(label==1).sum()
        batch={k:v[indices] for k,v in tx.inputs.items() if torch.is_tensor(v)}
        batch.update(points=points,point_ids=ids_out,point_valid=valid)
        targets={k:v[indices] for k,v in tx.targets.items() if torch.is_tensor(v)}
        targets.update(segmentation_labels=labels,bc_targets=bc,history_target_boxes=target_boxes[:,:3])
        tx.b0_inputs,tx.b0_targets=batch,targets
        tx.diagnostics.update(diagnostic_crop_point_count=crop_count,diagnostic_crop_target_count=crop_target_count,
            diagnostic_sampled_point_count=sample_count,diagnostic_sampled_target_count=sample_target_count,
            diagnostic_novel_target_count=tx.diagnostics['diagnostic_target_count']-crop_target_count)
        self.materialization_count+=len(indices)
        return batch,targets

    def commit(self,output,tx,*,diagnostics=False):
        if tx is not self._pending:
            raise RuntimeError('no matching pending v39 transaction (duplicate commit)')
        decision=output.commit_decision;pool=output.pool
        row=torch.arange(len(tx.rows),device=tx.device)
        selected=output.selected_index.detach().long()
        if selected.shape != (len(tx.rows),) or bool(((selected < 0) | (selected >= 7)).any()):
            raise ValueError('v39 selected candidate must be [B] in [0,7)')
        if not torch.equal(output.accepted_box.detach(),output.candidates.boxes.detach()[row,selected]):
            raise ValueError('accepted must equal selected candidate')
        if not bool(output.candidates.valid[row,selected].all()): raise ValueError('invalid selected candidate')
        probability=output.verification['targetness_logits'][row,selected].detach().sigmoid()
        support=pool['valid'] & inside(pool['xyz'],output.accepted_box.detach(),tx.inputs['box_size']) & (probability>=.5)
        measured=output.candidates.measurement_eligible[row,selected].detach().bool()
        support=support & measured[:,None]
        score=output.verification['logits'][row,selected,2].detach().sigmoid()*measured
        no_match=output.verification['region_no_match_logits'][row,selected].detach().sigmoid()
        eligible=measured & (support.sum(-1)>=self.config.v39_g_min_support) & (score>=self.config.v39_g_trust_threshold) & (no_match<.5)
        if not torch.equal(eligible,decision.eligible) or not torch.equal(support,decision.support_mask):
            raise ValueError('commit authorization differs from current real evidence')
        if not torch.equal(score, decision.trusted_score.detach()):
            raise ValueError('commit trusted score differs from verified current evidence')
        policy_full = output.policy_full.detach().bool()
        actual_full = output.full_executed.detach().bool()
        if policy_full.shape != selected.shape or actual_full.shape != selected.shape:
            raise ValueError('v39 execution and policy masks must be [B]')
        if bool((policy_full & ~actual_full).any()):
            raise ValueError('policy full requires an actual B0 execution')
        # 支持必须来自同帧真实 ID；在 GPU 归约后只搬一次授权证据，不搬完整 raw ID 云。
        membership,coordinates=[],[]
        for i,(raw_xyz,raw_ids) in enumerate(tx.raw):
            ids=pool['point_ids'][i,support[i]]
            if not len(ids):
                membership.append(torch.tensor(True,device=tx.device))
                coordinates.append(torch.tensor(True,device=tx.device))
                continue
            if not len(raw_ids):
                membership.append(torch.tensor(False,device=tx.device))
                coordinates.append(torch.tensor(False,device=tx.device))
                continue
            order=torch.argsort(raw_ids,stable=True);sorted_ids=raw_ids[order]
            position=torch.searchsorted(sorted_ids,ids).clamp_max(len(raw_ids)-1)
            membership.append((sorted_ids[position]==ids).all())
            coordinates.append((raw_xyz[order[position]]==pool['xyz'][i,support[i]]).all())
        raw_membership=torch.stack(membership);raw_coordinates=torch.stack(coordinates)
        fields=dict(box=output.accepted_box.detach(),selected=selected,quality=output.selected_quality.detach(),
            trust=score,eligible=eligible,support=support,ids=pool['point_ids'],
            full=actual_full,policy_full=policy_full,search=output.search_triggered,
            prior=output.prior.box.detach(),no_match=no_match,measured=measured,
            raw_membership=raw_membership,raw_coordinates=raw_coordinates)
        tx.diagnostics['diagnostic_current_G_point_count']=pool['valid'].sum(-1)
        fields.update(self._template_payload(tx, output.accepted_box.detach(), pool, support & eligible[:,None]))
        fields.update({'diagnostic/'+key:value for key,value in tx.diagnostics.items() if torch.is_tensor(value)})
        # 按 device/dtype 打包：raw ID 始终 int64，绝不经 float32 合包损失整数精度。
        transferred=tensors_to_numpy(fields, optimized=True)
        # 先验证、生成模板并计算所有新状态；发生任何错误时 accepted 状态都不写入。
        staged=[]
        for i,c in enumerate(tx.contexts):
            box=transferred['box'][i].astype(np.float64);box[:3]+=c['anchor'][:3]
            if (not np.isfinite(box).all() or not np.isfinite(transferred['quality'][i])
                    or not np.isfinite(transferred['trust'][i]) or not np.isfinite(transferred['prior'][i]).all()
                    or not np.isfinite(transferred['no_match'][i])):
                raise FloatingPointError('nonfinite commit')
            if not 0. <= transferred['quality'][i] <= 1.:
                raise ValueError('selected quality must be a probability')
            mask=transferred['support'][i];ids=transferred['ids'][i,mask]
            if len(np.unique(ids))!=len(ids) or (ids<0).any(): raise ValueError('support IDs must be real and unique')
            if not transferred['raw_membership'][i]: raise ValueError('support ID absent from current raw frame')
            if not transferred['raw_coordinates'][i]:raise ValueError('support coordinates differ from current raw identity')
            if c['state'] is not self.states.get(c['row']['request'].state_key):
                raise RuntimeError('pending transaction state ownership changed')
            template = self._template_from_support(tx,i,transferred) if transferred['eligible'][i] else None
            staged.append((box,mask,template))
        records=[]
        next_states=[]
        for i,(c,(box,mask,template)) in enumerate(zip(tx.contexts,staged)):
            state=deepcopy(c['state']);r=c['row']['request'];now=c['timestamp']
            supported=bool(transferred['eligible'][i]);source=int(transferred['selected'][i]);count=int(mask.sum())
            discontinuity=(source==6) or (supported and state.consecutive_untrusted_outputs>=2)
            innovation=float(np.linalg.norm(transferred['box'][i,:2]-transferred['prior'][i,:2]))
            if discontinuity:
                state.motion_epoch+=1;state.motion_reset_pending=True
                state.trusted_velocity[:]=0.;state.trusted_velocity_valid=False
            pair=not discontinuity and innovation<=max(.5*np.linalg.norm(state.size[:2]),1e-3)
            if state.motion_reset_pending:
                # 恢复等待期只能由同 epoch、相邻的两次可信当前观测重建边。
                # 中间存在不可信输出时，即使当前重新可信，也不能把该输出接成速度。
                trusted_previous_time=state.trusted[-1][1]
                trusted_predecessor=(trusted_previous_time==state.times.get(r.frame-1)
                    and state.frame_motion_epoch.get(r.frame-1)==state.motion_epoch)
                pair=pair and supported and trusted_predecessor
            state.boxes[r.frame]=box;state.times[r.frame]=now;state.box_provenance[r.frame]=PREDICTED_BOX
            state.frame_motion_epoch[r.frame]=state.motion_epoch;state.transitions[r.frame]=pair
            state.next_frame+=1;state.previous_quality=float(transferred['quality'][i]);state.previous_innovation=innovation
            velocity_updated=False
            if supported:
                previous,previous_time=state.trusted[-1]
                adjacent=previous_time==state.times.get(r.frame-1)
                same_epoch=state.frame_motion_epoch.get(r.frame-1)==state.motion_epoch
                if pair and adjacent and same_epoch:
                    if now <= previous_time:
                        raise ValueError('trusted velocity requires strictly increasing physical time')
                    state.trusted_velocity=(box[:3]-previous[:3])/(now-previous_time)
                    state.trusted_velocity_valid=True;state.motion_reset_pending=False
                    state.last_velocity_update_time=now;velocity_updated=True
                else:
                    state.trusted_velocity[:]=0.;state.trusted_velocity_valid=False
                state.trusted.append((box.copy(),now));state.trusted[:]=state.trusted[-2:]
                state.search_anchor=box.copy();state.last_supported_time=now;state.last_supported_quality=float(transferred['trust'][i])
                state.templates.append(template);state.templates[:]=state.templates[-2:]
            if count>=self.config.v39_g_min_support: state.last_strong_time=now
            state.previous_is_propagation=not bool(transferred['measured'][i])
            state.consecutive_propagations=state.consecutive_propagations+1 if state.previous_is_propagation else 0
            state.consecutive_untrusted_outputs=0 if supported else state.consecutive_untrusted_outputs+1
            full=bool(transferred['full'][i]);policy_full=bool(transferred['policy_full'][i])
            state.consecutive_cheap=0 if policy_full else state.consecutive_cheap+1
            if policy_full: state.last_full_time=now
            if supported: state.search_failures=0
            elif bool(transferred['search'][i]): state.search_failures+=1
            names=('q0','prior','trusted_propagation','g_q0','g_prior','g_trusted','g_reacquire')
            records.append(dict(supported=supported,strong=count>=self.config.v39_g_min_support,trusted_write=supported,
                trusted_position_updated=supported,trusted_velocity_updated=velocity_updated,
                trusted_velocity_valid=state.trusted_velocity_valid,commit_source=names[source],
                commit_quality=float(transferred['trust'][i]),geometric_fg_count=count,propagation_only=state.previous_is_propagation,
                full_executed=full,policy_full=policy_full,search_triggered=bool(transferred['search'][i]),motion_epoch=state.motion_epoch,
                motion_discontinuity=discontinuity,consecutive_untrusted_outputs=state.consecutive_untrusted_outputs,
                weak_age=self._strong_age(state,now),supported_age=now-state.last_supported_time))
            for old in list(state.boxes):
                if old<r.frame-2:
                    for mapping in (state.boxes,state.times,state.transitions,state.box_provenance,state.frame_motion_epoch): mapping.pop(old,None)
            next_states.append(state)
        # 全部 staging 完成后一次发布；保留外部持有的 state 对象身份。
        for c, state in zip(tx.contexts, next_states):
            c['state'].__dict__.clear()
            c['state'].__dict__.update(state.__dict__)
            if c['row']['request'].ends_window:
                self.states.pop(c['row']['request'].state_key)
        tx.diagnostics_cpu = {key.split('/',1)[1]:value for key,value in transferred.items()
                              if key.startswith('diagnostic/')}
        self._pending=None
        return records

    def _template_payload(self,tx,boxes,pool,support):
        b,p=len(tx.rows),self.config.v39_template_points
        xyz=pool['xyz'].new_zeros((b,p,3));gt_xyz=torch.zeros_like(xyz)
        ids=torch.full((b,p),-1,device=tx.device,dtype=torch.long)
        valid=torch.zeros((b,p),device=tx.device,dtype=torch.bool);gt_valid=torch.zeros_like(valid)
        for i,c in enumerate(tx.contexts):
            r=c['row']['request']
            index=sample_indices(pool['point_ids'][i],support[i],p,
                stable_seed(self.config.seed,r.epoch,c['row']['tracklet_key'],r.branch,r.frame,'template','trusted'))
            n=len(index);cloud=pool['xyz'][i,index]
            xyz[i,:n]=canonical_points(cloud,boxes[i]);ids[i,:n]=pool['point_ids'][i,index];valid[i,:n]=True
            # GT 数据只产生未来训练标签；在线模板 xyz/valid 的选择由当前支持决定。
            gt_xyz[i,:n]=canonical_points(cloud,tx.targets['target_box'][i])
            gt_valid[i,:n]=inside(cloud,tx.targets['target_box'][i],tx.targets['target_box_size'][i])
        return dict(template_xyz=xyz,template_ids=ids,template_valid=valid,
                    template_gt_xyz=gt_xyz,template_gt_valid=gt_valid)

    def _template_from_support(self,tx,i,transferred):
        c=tx.contexts[i];r=c['row']['request'];mask=transferred['template_valid'][i]
        # 独立小副本避免持有整个 batch 的传输 packet。
        return dict(xyz=transferred['template_xyz'][i,mask].copy(),
            ids=transferred['template_ids'][i,mask].copy(),timestamp=c['timestamp'],frame=r.frame,kind='trusted',
            label_xyz=transferred['template_gt_xyz'][i,mask].copy(),
            label_valid=transferred['template_gt_valid'][i,mask].copy())
