# CT-SeqTrack 当前服务器路径

服务器为 `lishengjie@10.109.253.86`，项目根为 `/home/lishengjie/study/lcyu/CT-SeqTrack`。2026-09-28 18:59只读核验：原环境/mini数据有效、未发现CT-SeqTrack训练进程、服务器尚无v35配置。用户本次选择四组seed42，GPU0/0/1/1；由用户上传、执行一次真实CUDA检查并启动，条件满足后才保留两次seed52复验。代理不上传、不安装、不启动或停止进程。最新命令见 [v35运行说明](CTSEQTRACK_V35_MINI_LAUNCH.md)。

## 数据根

| 数据 | 路径 | 配置 |
|---|---|---|
| nuScenes mini | `/home/lishengjie/data/nuscenes-mini` | `dataset: nuscenes_mf`、`version: v1.0-mini`、`ct_coordinate_mode: global` |
| nuScenes full | `/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/` | `dataset: nuscenes_mf`、`version: v1.0-trainval`、`ct_coordinate_mode: global` |
| KITTI | 由实际部署提供数据根，当前不填未经核验路径 | `dataset: kitti_mf`、`version: kitti_tracking`、`ct_coordinate_mode: sensor_relative` |

通过 `--path` 指定数据根。nuScenes根应有相应版本metadata、`samples/` 与 `sweeps/`；full不能沿用mini数据。KITTI需要对应点云、标签、标定和图像尺寸信息，沿用当前数据接口。接口支持不代表full/KITTI实验已完成。

## Python环境与SDK

2026-09-28再次只读确认 `/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python`：Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2、nuscenes-devkit1.1.9。GPU0/1为A40，各46,068MiB，18:59读取时空闲34,402/38,808MiB；另有其他用户及本用户的非CT任务，可用量随运行变化。本项目分区空闲约2,900GiB。活动网络使用PyTorch算子，不以旧PointNet++ CUDA扩展为前提。

当前nuScenes SDK已安装在 `seqtrack3d/lib/python3.9/site-packages/nuscenes/`，本轮命令直接使用，无需PYTHONPATH fallback。以下只保留历史路径记录：nuScenes包源曾位于 `/home/lishengjie/code/SparseFusion-main/nuscenes`，此次未发现该路径下的 `nuscenes/__init__.py`，不要直接沿用。它也**不是数据根，不能传给 `--path`**。历史fallback写法（只有路径实际可用时才适用）：

```bash
export CTSEQ_NUSCENES_PYTHON_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes
export PYTHONPATH="${CTSEQ_NUSCENES_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

不使用旧 `--preloading`；当前原始云按需读取，每worker缓存256MiB。使用 `CUDA_VISIBLE_DEVICES` 选择物理卡，每进程仍为单卡；设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`、`PYTORCH_CUDA_ALLOC_CONF=backend:native` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

本轮命令见 [v35运行说明](CTSEQTRACK_V35_MINI_LAUNCH.md)，当前协议与工具见 [工具面](FORMAL_TOOLING.md)。旧版本的服务器测试与真实CUDA batch证据不替代v35的真实批次验证。用户上传后执行一次新的v35批次检查；本轮不迁移原环境。旧版本命令仅供 [历史索引](HISTORY_EVIDENCE_INDEX.md) 复现。
