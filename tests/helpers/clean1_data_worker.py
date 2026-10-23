"""冻结 v38/clean1 分进程跑两条固定短轨迹，导出数据与真实模型递推。"""
from pathlib import Path
import argparse
import copy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', choices=('b1', 'full'), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.root).resolve()))
    import numpy as np
    import torch
    from models.ct_v38.config import normalize_config
    from models.ct_v38.data import BatchBuilder, RawEndpointDataset, build_loaders
    from models.ct_v38.model import JointTracker
    from models.ct_v38.runtime import TrackingEvaluation
    from utils.batch_materialization import attach_materialization_cache, clear_materialization_cache

    class ShortTracks:
        """固定真实时间；轨迹0含连续缺测，轨迹1从空序列恢复到稀疏观测。"""
        lengths = (6, 6)

        def get_num_tracklets(self):
            return len(self.lengths)

        def get_num_frames_tracklet(self, index):
            return self.lengths[index]

        def get_tracklet_key(self, index):
            return 'clean1_short/' + str(index)

        def get_frames(self, index, frame_ids):
            rows = []
            for frame in frame_ids:
                center = np.array([frame * .2, index * 20. + frame * .025, 0.])
                if index == 0 and frame == 5:
                    # B0 crop 外、原 band 最大可达域内的目标，覆盖 demand=True 标签。
                    center[0] = 6.
                cloud = np.array([[0., 0., 0.], [.3, .1, .1], [-.3, -.1, -.1],
                                  [.1, -.3, 0.], [5., 0., 0.], [8., 2., 0.]]) + center
                if (index == 0 and frame in (1, 2, 3, 4)) or (index == 1 and frame < 3):
                    cloud = cloud[:0]
                elif index == 1 and frame == 3:
                    cloud = cloud[:2]
                rows.append(dict(pc=cloud, point_ids=np.arange(len(cloud)),
                    **{'3d_bbox': np.r_[center, .1 + frame * .01]},
                    box_size=np.array([4., 2., 2.]),
                    timestamp=100. + (0., .5, 1.3, 1.7, 2.5, 3.)[frame],
                    scene_id='clean1_short_scene'))
            return rows

        def get_frame_metadata(self, index, frame):
            return self.get_frames(index, [frame])[0]

    def snapshot(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, dict):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(snapshot(item) for item in value)
        return copy.deepcopy(value)

    torch.set_num_threads(1)
    config = normalize_config(dict(v38_arm=args.arm, ct_engineering_check=True,
        point_sample_size=8, batch_size=2, workers=0, v32_reserve_windows=4, epoch=11))
    torch.manual_seed(42)
    tracker = JointTracker(config).eval()
    source = ShortTracks()
    loaders = build_loaders(config, roles=('train', 'test'), sources=dict(train=source, test=source))
    manifest = []
    for epoch in (0, 5, 9):
        sampler = loaders['train'].batch_sampler
        sampler.set_epoch(epoch)
        manifest.append(sampler.state_dict())
    builder = BatchBuilder(config)
    evaluator = TrackingEvaluation(config)
    steps = []
    with torch.no_grad():
        for raw_rows in loaders['test']:
            prepared = builder.prepare(raw_rows)
            batch = attach_materialization_cache(prepared, enabled=True)
            prior = tracker.plan_prior(batch)
            if hasattr(builder, 'supervise_prior'):
                builder.supervise_prior(batch, prior, training=True)
            else:
                builder.acquire(batch, prior, training=True)
            output = tracker(batch, prior=prior)
            losses = tracker.compute_losses(batch, output)
            # 只保留真实 Tensor 字段，设备侧瞬时缓存由生命周期释放。
            saved_batch = snapshot({key: value for key, value in batch.items() if torch.is_tensor(value)})
            commit = builder.commit(output, batch, diagnostics=True)
            evaluator.add_batch(raw_rows, batch, output, commit)
            states = {}
            for key, state in builder.states.items():
                states[key] = snapshot({name: value for name, value in vars(state).items() if name != 'memory'})
            steps.append(dict(batch=saved_batch, states=states, commit=commit,
                accepted=snapshot(output.accepted_box), prior=snapshot(prior.box),
                selected=snapshot(output.selected_index), valid=snapshot(output.candidates.valid),
                raw_valid=snapshot(output.candidates.raw_valid), alias=snapshot(output.candidates.alias_of),
                quality=snapshot(output.selected_quality), losses=snapshot(losses)))
            clear_materialization_cache(batch)
    torch.save(dict(steps=steps, sampler=manifest, evaluation=evaluator.summary(),
        rows=evaluator.rows, final_states=builder.states), args.out)


if __name__ == '__main__':
    main()
