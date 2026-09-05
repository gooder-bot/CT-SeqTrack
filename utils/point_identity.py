"""原始点身份与采样槽有效性。ID 不作为网络输入。"""
import numpy as np


def raw_point_ids(pc):
    ids = getattr(pc, "point_ids", None)
    if ids is None:
        ids = np.arange(pc.points.shape[1], dtype=np.int64)
        pc.point_ids = ids
    ids = np.asarray(ids, dtype=np.int64)
    if ids.shape != (pc.points.shape[1],):
        raise ValueError("point ID and coordinate lengths disagree")
    return ids


def sampled_identity(pc, indices, sample_size):
    if indices is None:
        ids = np.full(int(sample_size), -1, dtype=np.int64)
    else:
        ids = raw_point_ids(pc)[np.asarray(indices, dtype=np.int64)]
    valid = ids >= 0
    unique = np.zeros(ids.shape, dtype=bool)
    if valid.any():
        positions = np.flatnonzero(valid)
        _, first = np.unique(ids[valid], return_index=True)
        unique[positions[first]] = True
    return ids, valid.astype(np.float32), unique.astype(np.float32)
