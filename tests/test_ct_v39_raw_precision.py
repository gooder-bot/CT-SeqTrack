"""原始 nuScenes 点先旋转/减原点，不能先量化大世界坐标。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from models.ct_v39.raw_service import RawFrameService


class DirectSource:
    version = 'v1.0-mini'
    ct_coordinate_mode = 'global'

    def __init__(self, path):
        self.path = str(path)
        self.calls = 0

    def get_frames(self, track, frames):
        self.calls += 1
        raise AssertionError('nuScenes direct raw path must not invoke the quantizing world adapter')


def make_cloud(tmp_path, xyz=None):
    xyz = np.asarray(xyz if xyz is not None else [[.01,.02,.03], [.02,.03,.04], [-.03,.01,.02]], np.float32)
    array = np.column_stack((xyz, np.arange(len(xyz),dtype=np.float32)+17.,
                             np.arange(len(xyz),dtype=np.float32)+101.))
    path = tmp_path/'samples'/'sensor.pcd.bin'
    path.parent.mkdir(parents=True)
    array.tofile(path)
    angle = .731
    rotation = np.array([[np.cos(angle),-np.sin(angle),0.],
                         [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]],np.float64)
    transform = np.eye(4,dtype=np.float64)
    transform[:3,:3] = rotation
    transform[:3,3] = [3e8,-4e8,2e8]
    meta = dict(raw_frame_token='same-physical-frame',frame_id='same-physical-frame',
        sensor_to_sequence_world=transform,coordinate_mode='global',
        meta=dict(sample_data_lidar=dict(filename='samples/sensor.pcd.bin'),
                  box_anno=dict(translation=[1.,2.,3.])),
        **{'3d_bbox':np.array([1.,2.,3.,.4])})
    return DirectSource(tmp_path),meta,array


def test_direct_sensor_path_preserves_centimetres_at_huge_world_translation(tmp_path):
    source,meta,array = make_cloud(tmp_path)
    service = RawFrameService(dict(workers=1))
    try:
        key,cloud = service.fetch(source,0,1,meta)
        transform = meta['sensor_to_sequence_world']
        np.testing.assert_array_equal(cloud.ids,np.arange(len(array)))
        np.testing.assert_array_equal(cloud.origin,transform[:3,3])
        assert cloud.xyz.dtype==np.float32 and cloud.origin.dtype==np.float64 and cloud.ids.dtype==np.int64
        assert not cloud.xyz.flags.writeable and not cloud.origin.flags.writeable and not cloud.ids.flags.writeable
        expected_rotated=array[:,:3].astype(np.float64)@transform[:3,:3].T
        np.testing.assert_allclose(cloud.xyz,expected_rotated,rtol=0,atol=2e-9)
        anchor=np.r_[transform[:3,3]+np.array([.011,-.023,.037]),.2]
        relative,ids=service.tensors(key,cloud,'cpu',anchor)
        expected=expected_rotated+(transform[:3,3]-anchor[:3])
        np.testing.assert_allclose(relative.numpy(),expected,rtol=0,atol=5e-9)
        assert torch.linalg.vector_norm(relative[1]-relative[0])>.01
        # 同样的点若先变成 world FP32，所有厘米差已不可逆消失。
        quantized=(expected_rotated+transform[:3,3]).astype(np.float32)
        assert len(np.unique(quantized,axis=0))==1
        assert source.calls==0 and ids.tolist()==list(range(len(array)))
        cached=service.cached_tensors(key,'cpu',anchor)
        torch.testing.assert_close(cached[0],relative,rtol=0,atol=0)
        assert service.gpu.data[('cpu',key)][0][2].dtype==np.float64
    finally:
        service.close()


def test_target_gt_never_changes_raw_identity_or_coordinates(tmp_path):
    source,meta,array=make_cloud(tmp_path)
    service=RawFrameService(dict(workers=1))
    try:
        key,cloud=service.fetch(source,0,1,meta)
        altered=deepcopy(meta)
        altered['3d_bbox'][:]=np.nan
        altered['meta']['box_anno']['translation']=[1e20,-1e20,1e20]
        assert service.key(source,99,77,altered)==key
        again_key,again=service.fetch(source,99,77,altered)
        assert again_key==key and again is cloud and source.calls==0
        # 清空缓存重新解码也不能读取目标标签。
        service.cpu.clear()
        _,decoded=service.fetch(source,99,77,altered)
        np.testing.assert_array_equal(decoded.xyz,cloud.xyz)
        np.testing.assert_array_equal(decoded.ids,cloud.ids)
        np.testing.assert_array_equal(decoded.origin,cloud.origin)
    finally:
        service.close()


@pytest.mark.parametrize('mutation,message', [
    ('shape','finite 4x4'),('nan','finite 4x4'),('last_row','homogeneous'),
    ('scale','orthonormal'),('reflection','proper')])
def test_direct_path_rejects_invalid_transform(tmp_path,mutation,message):
    source,meta,_=make_cloud(tmp_path)
    matrix=meta['sensor_to_sequence_world']
    if mutation=='shape':meta['sensor_to_sequence_world']=np.eye(3)
    elif mutation=='nan':matrix[0,3]=np.nan
    elif mutation=='last_row':matrix[3,0]=.1
    elif mutation=='scale':matrix[:3,:3]*=2.
    else:matrix[:3,0]*=-1.
    with pytest.raises(ValueError,match=message):
        RawFrameService._decode(source,0,0,meta)
    assert source.calls==0


def test_direct_path_rejects_partial_records_and_nonfinite_xyz(tmp_path):
    source,meta,array=make_cloud(tmp_path)
    path=tmp_path/'samples'/'sensor.pcd.bin'
    with path.open('ab') as stream:stream.write(b'X')
    with pytest.raises(ValueError,match='five-float'):
        RawFrameService._decode(source,0,0,meta)
    array[0,0]=np.nan;array.tofile(path)
    with pytest.raises(ValueError,match='XYZ must be finite'):
        RawFrameService._decode(source,0,0,meta)


def test_empty_direct_cloud_keeps_static_origin_and_empty_ids(tmp_path):
    source,meta,_=make_cloud(tmp_path,xyz=np.zeros((0,3)))
    cloud=RawFrameService._decode(source,0,0,meta)
    assert cloud.xyz.shape==(0,3) and cloud.ids.shape==(0,)
    np.testing.assert_array_equal(cloud.origin,meta['sensor_to_sequence_world'][:3,3])


def test_generic_source_retains_custom_ids_order_and_relative_geometry():
    class Generic:
        calls=0
        def get_frames(self,track,frames):
            self.calls+=1
            return [dict(pc=np.array([[1e8+.01,3.,4.],[1e8+.03,5.,6.]],np.float64),
                         point_ids=np.array([2**54+3,2**54+1],np.int64))]
    source=Generic()
    cloud=RawFrameService._decode(source,4,8,dict(frame_id='generic',**{'3d_bbox':'unused'}))
    assert source.calls==1
    np.testing.assert_array_equal(cloud.ids,[2**54+3,2**54+1])
    np.testing.assert_allclose(cloud.xyz,[[0.,0.,0.],[.02,2.,2.]],rtol=0,atol=2e-8)
