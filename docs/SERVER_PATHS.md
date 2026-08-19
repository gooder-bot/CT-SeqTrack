# CT-SeqTrack 服务器路径

本文只记录服务器数据根目录与正式 v25 入口，避免把数据集路径和 Python
源码路径混为一谈。

| 用途 | 服务器路径 | 传给 `--path` |
| --- | --- | --- |
| KITTI Tracking `training` | `/home/lishengjie/data/cxtrack/training/` | 是 |
| nuScenes mini 数据 | `/home/lishengjie/data/nuscenes-mini/` | 是 |
| nuScenes Python 包源码 | `/home/lishengjie/code/SparseFusion-main/nuscenes/` | 否 |

## 推荐环境变量

```bash
export CTSEQ_KITTI_ROOT=/home/lishengjie/data/cxtrack/training
export CTSEQ_NUSCENES_MINI_ROOT=/home/lishengjie/data/nuscenes-mini
export CTSEQ_NUSCENES_PYTHON_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes
export PYTHONPATH="${CTSEQ_NUSCENES_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

KITTI 根目录应直接包含 `label_02/`、`velodyne/` 和 `calib/`。nuScenes mini
根目录应包含 `v1.0-mini/` 以及相应的 `samples/`、`sweeps/`。

## 正式入口

```bash
# nuScenes mini v25 B0；其他 arm 仅替换 cfg 文件并添加协议要求的 artifact。
python main.py \
  --cfg cfgs/ct_seqtrack/25_b0.yaml \
  --path "$CTSEQ_NUSCENES_MINI_ROOT"

# KITTI 基础加载能力仍使用原 SeqTrack3D 配置。
python main.py \
  --cfg cfgs/seqtrack3d_kitti.yaml \
  --path "$CTSEQ_KITTI_ROOT" \
  --seed 42
```

四臂 smoke、20-batch H3、kill/resume 和完整验收统一执行
`tools/run_ct_v25_server_acceptance.sh`，详见
`docs/CTSEQTRACK_V25_SERVER_ACCEPTANCE.md`。本地不反复检查服务器数据、CUDA
或 nuScenes 环境。
