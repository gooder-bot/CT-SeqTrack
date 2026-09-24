# CT-SeqTrack v33 服务器路径

服务器为 `lishengjie@10.109.253.86`，项目根为 `/home/lishengjie/study/lcyu/CT-SeqTrack`。2026-09-24 用户最新要求本轮服务器严格只读；助手只修订本地，由用户上传并启动SeqTrack与三组B0，GPU依次0/0/1/1。此前v33独立快照已通过原环境测试与真实batch检查，但正式训练未启动，服务器主目录仍是旧版。命令见 [v33运行说明](CTSEQTRACK_V33_MINI_LAUNCH.md)。

## 数据根

| 数据 | 路径 | 配置 |
|---|---|---|
| nuScenes mini | `/home/lishengjie/data/nuscenes-mini` | `dataset: nuscenes_mf`、`version: v1.0-mini`、`ct_coordinate_mode: global` |
| nuScenes full | `/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/` | `dataset: nuscenes_mf`、`version: v1.0-trainval`、`ct_coordinate_mode: global` |
| KITTI | 由实际部署提供数据根，当前不填未经核验路径 | `dataset: kitti_mf`、`version: kitti_tracking`、`ct_coordinate_mode: sensor_relative` |

通过 `--path` 指定数据根。nuScenes根应有相应版本metadata、`samples/` 与 `sweeps/`；full不能沿用mini数据。KITTI需要对应点云、标签、标定和图像尺寸信息，沿用当前数据接口。接口支持不代表full/KITTI实验已完成。

## Python环境与SDK

2026-09-22再次只读确认 `/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python`：Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2，GPU0/1为A40，检查时两卡均为0MiB占用。当前v32及独立参考网络使用PyTorch算子，不以旧PointNet++ CUDA扩展为前提。

当前nuScenes SDK已安装在 `seqtrack3d/lib/python3.9/site-packages/nuscenes/`，本轮命令直接使用，无需PYTHONPATH fallback。以下只保留历史路径记录：nuScenes包源曾位于 `/home/lishengjie/code/SparseFusion-main/nuscenes`，此次未发现该路径下的 `nuscenes/__init__.py`，不要直接沿用。它也**不是数据根，不能传给 `--path`**。历史fallback写法（只有路径实际可用时才适用）：

```bash
export CTSEQ_NUSCENES_PYTHON_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes
export PYTHONPATH="${CTSEQ_NUSCENES_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

不使用旧 `--preloading`；当前原始云按需读取，每worker缓存256MiB。使用 `CUDA_VISIBLE_DEVICES` 选择物理卡，每进程仍为单卡；设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`、`PYTORCH_CUDA_ALLOC_CONF=backend:native` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

mini四组后台命令见 [v33运行说明](CTSEQTRACK_V33_MINI_LAUNCH.md)，当前协议与工程验证见 [工具面](FORMAL_TOOLING.md)。当前代码已经通过服务器原环境314项测试（1项不适用跳过）和一次真实CUDA batch，不需重装环境；旧版本命令只通过 [历史索引](HISTORY_EVIDENCE_INDEX.md) 使用。
