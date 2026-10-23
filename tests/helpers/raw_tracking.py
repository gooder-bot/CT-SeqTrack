"""当前数据事务的独立合成输入，不导入历史测试。"""
from types import SimpleNamespace
import numpy as np
import torch
from models.ct_v38.data import EndpointRequest

class TinySource:
    def __init__(self, lengths=(10, 6), dt=.5):
        self.lengths, self.dt = lengths, dt

    def get_num_tracklets(self):
        return len(self.lengths)

    def get_num_frames_tracklet(self, index):
        return self.lengths[index]

    def get_tracklet_key(self, index):
        return 'tiny/' + str(index)

    def get_frames(self, index, frame_ids):
        rows = []
        for i in frame_ids:
            center = np.asarray([i * .2, index * 20., 0.])
            cloud = np.asarray([[0., 0., 0.], [.2, .1, 0.], [-.2, -.1, 0.],
                                [0., -.2, .1], [5., 0., 0.], [8., 0., 0.]]) + center
            rows.append(dict(pc=cloud, point_ids=np.arange(len(cloud)),
                             **{'3d_bbox': np.r_[center, .1]}, box_size=np.asarray([4., 2., 2.]),
                             timestamp=100. + i * self.dt, scene_id='tiny_scene'))
        return rows

    def get_frame_metadata(self, index, frame):
        return self.get_frames(index, [frame])[0]


def request(frame, *, branch=3, start=1, end=9, track=0):
    return EndpointRequest(9, track, branch, start, end, frame)


def fake_prior(batch):
    return SimpleNamespace(box=batch['fallback_box'].clone(),
                           acquisition_fraction=torch.full((len(batch['points']), 2), .3),
                           direction_xy=torch.tensor([[1., 0.]]).expand(len(batch['points']), -1))


def fake_output(batch, *, delta=0., quality=.9):
    accepted = batch['fallback_box'].clone()
    accepted[:, 0] += delta
    return SimpleNamespace(accepted_box=accepted,
        selected_quality=torch.full((len(accepted),), quality),
        selected_index=torch.zeros(len(accepted), dtype=torch.long),
        prior=fake_prior(batch),
        observation=SimpleNamespace(foreground_probability=torch.ones_like(batch['point_valid'], dtype=torch.float)))

