"""与模块臂、窗口边界、worker 无关的具名随机流。"""
from contextlib import contextmanager
import torch
from models.ct_v31.data import stable_seed


@contextmanager
def component_rng(seed, name):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stable_seed(int(seed), 'v36_init', str(name)))
        yield


def sampling_seed(seed, epoch, tracklet, branch, frame, source_frame, stream):
    return stable_seed(int(seed), 'v36_sample', 0 if int(branch) == 4 else int(epoch),
                       tracklet, int(branch), int(frame), int(source_frame), str(stream))
