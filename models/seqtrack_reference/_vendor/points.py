import copy
import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial.distance import cdist
from datasets.data_classes import PointCloud

def regularize_pc(points, sample_size, seed=None):
    # random sampling from points
    num_points = points.shape[0]
    num_feature = points.shape[1]
    new_pts_idx = None
    rng = np.random if seed is None else np.random.default_rng(seed)
    if num_points > 2:
        if num_points != sample_size:
            new_pts_idx = rng.choice(num_points, size=sample_size, replace=sample_size > num_points)
            # new_pts_idx = random_choice(num_samples=sample_size, size=num_points,
            #                             replacement=sample_size > num_points, seed=seed).numpy()
        else:
            new_pts_idx = np.arange(num_points)
    if new_pts_idx is not None:
        points = points[new_pts_idx, :]
    else:
        points = np.zeros((sample_size, num_feature), dtype='float32') 

    return points, new_pts_idx


def getOffsetBB(box, offset, degrees=True, use_z=False, limit_box=True, inplace=False):
    rot_quat = Quaternion(matrix=box.rotation_matrix) 
    trans = np.array(box.center) 
    if not inplace:
        new_box = copy.deepcopy(box)
    else:
        new_box = box

    new_box.translate(-trans) 
    new_box.rotate(rot_quat.inverse) 
    if len(offset) == 3:
        use_z = False
    # REMOVE TRANSfORM
    if degrees:
        if len(offset) == 3:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], degrees=offset[2]))
        elif len(offset) == 4:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], degrees=offset[3]))
    else:
        if len(offset) == 3:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], radians=offset[2]))
        elif len(offset) == 4:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], radians=offset[3]))
    if limit_box: 
        if offset[0] > new_box.wlh[0]:
            offset[0] = np.random.uniform(-1, 1)
        if offset[1] > min(new_box.wlh[1], 2):
            offset[1] = np.random.uniform(-1, 1)
        if use_z and offset[2] > new_box.wlh[2]:
            offset[2] = 0
    if use_z:
        new_box.translate(np.array([offset[0], offset[1], offset[2]]))
    else:
        new_box.translate(np.array([offset[0], offset[1], 0]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box 


def get_point_to_box_distance(pc, box, wlh_factor=1.0):
    """
    generate the BoxCloud for the given pc and box
    :param pc: Pointcloud object or numpy array
    :param box:
    :return:
    """
    if isinstance(pc, PointCloud):
        points = pc.points.T  # N,3
    else:
        points = pc  # N,3
        assert points.shape[1] == 3
    box_corners = box.corners(wlh_factor=wlh_factor)  # 3,8
    box_centers = box.center.reshape(-1, 1)  # 3,1
    box_points = np.concatenate([box_centers, box_corners], axis=1)  # 3,9

    points2cc_dist = cdist(points, box_points.T)  # N,9
    return points2cc_dist


def crop_pc_axis_aligned(PC, box, offset=0, scale=1.0, return_mask=False):
    """
    crop the pc using the box in the axis-aligned manner
    """
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = copy.deepcopy(PC)
    new_PC.points = PC.points[:, close]
    if return_mask:
        return new_PC, close
    return new_PC


def generate_subwindow_with_aroundboxs(pc, sample_bb, ref_o, scale, offset=2, oriented=True):
    """
    generating the search area using the sample_bb

    :param pc:
    :param sample_bb:
    :param scale:
    :param offset:
    :param oriented: use oriented or axis-aligned cropping
    :return:
    """
    rot_mat = np.transpose(sample_bb.rotation_matrix)
    trans = -sample_bb.center

    rot_mat_o = np.transpose(ref_o.rotation_matrix)
    trans_o =  -ref_o.center

    assert rot_mat.shape == (3, 3), f"rot_matrix shape {rot_mat.shape} should be (3, 3)"
    assert rot_mat_o.shape == (3, 3), f"rot_matrix shape {rot_mat_o.shape}, should be (3, 3)"
    if oriented:
        new_pc = copy.deepcopy(pc) 

        box_tmp = copy.deepcopy(sample_bb)

        # transform to the coordinate system of sample_bb
        new_pc.translate(trans)
        box_tmp.translate(trans)

        new_pc.rotate(rot_mat)
        box_tmp.rotate(Quaternion(matrix=rot_mat))
       
        new_pc = crop_pc_axis_aligned(new_pc, box_tmp, scale=scale, offset=offset)

        new_pc.rotate(np.transpose(rot_mat)) 
        new_pc.translate(-trans) 

        new_pc.translate(trans_o)
        new_pc.rotate(rot_mat_o)


    else:
        new_pc = crop_pc_axis_aligned(pc, sample_bb, scale=scale, offset=offset)

        # transform to the coordinate system of sample_bb
        new_pc.translate(trans)
        new_pc.rotate(rot_mat)

    return new_pc


def transform_box(box, ref_box, inplace=False):
    if not inplace:
        box = copy.deepcopy(box)
    box.translate(-ref_box.center)
    box.rotate(Quaternion(matrix=ref_box.rotation_matrix.T))
    return box


def rotz_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device)
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 1] = -s
    output[..., 1, 0] = s
    output[..., 1, 1] = c
    output[..., 2, 2] = 1
    return output


def get_offset_box_tensor(ref_box_params, offset_box_params):
    """
    transform the ref_box with the give offset
    :param ref_box_params: B,4
    :param offset_box_params: B,4
    :return: B,4
    """
    ref_center = ref_box_params[:, :3]  # B,3
    ref_rot_angles = ref_box_params[:, -1]  # B,
    offset_center = offset_box_params[:, :3]
    offset_rot_angles = offset_box_params[:, -1]
    rot_mat = rotz_batch_tensor(ref_rot_angles)  # B,3,3

    new_center = torch.matmul(rot_mat, offset_center[..., None]).squeeze(dim=-1)  # B,3
    new_center += ref_center
    new_angle = ref_rot_angles + offset_rot_angles
    return torch.cat([new_center, new_angle[:, None]], dim=-1)

