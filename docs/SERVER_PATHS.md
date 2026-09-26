# CT-SeqTrack 当前服务器路径

服务器为 `lishengjie@10.109.253.86`，项目根为 `/home/lishengjie/study/lcyu/CT-SeqTrack`。2026-09-26已再次只读确认原Python环境与mini数据有效，旧R/A/B/C/D/E均完整结束；当前无本项目进程，服务器尚无v34配置。用户自行上传并启动八组，GPU0/1各四组；代理不上传、不安装、不启动或停止进程。最新命令见 [v34八组运行说明](CTSEQTRACK_V34_MINI_LAUNCH.md)。

## 数据根

| 数据 | 路径 | 配置 |
|---|---|---|
| nuScenes mini | `/home/lishengjie/data/nuscenes-mini` | `dataset: nuscenes_mf`、`version: v1.0-mini`、`ct_coordinate_mode: global` |
| nuScenes full | `/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/` | `dataset: nuscenes_mf`、`version: v1.0-trainval`、`ct_coordinate_mode: global` |
| KITTI | 由实际部署提供数据根，当前不填未经核验路径 | `dataset: kitti_mf`、`version: kitti_tracking`、`ct_coordinate_mode: sensor_relative` |

通过 `--path` 指定数据根。nuScenes根应有相应版本metadata、`samples/` 与 `sweeps/`；full不能沿用mini数据。KITTI需要对应点云、标签、标定和图像尺寸信息，沿用当前数据接口。接口支持不代表full/KITTI实验已完成。

## Python环境与SDK

2026-09-26只读确认 `/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python`：Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2、nuscenes-devkit1.1.9。GPU0/1为A40，各46,068MiB，读取时各约3.3GB被其他任务占用，约42GiB空闲；可用量随其他任务变化。活动网络使用PyTorch算子，不以旧PointNet++ CUDA扩展为前提。

当前nuScenes SDK已安装在 `seqtrack3d/lib/python3.9/site-packages/nuscenes/`，本轮命令直接使用，无需PYTHONPATH fallback。以下只保留历史路径记录：nuScenes包源曾位于 `/home/lishengjie/code/SparseFusion-main/nuscenes`，此次未发现该路径下的 `nuscenes/__init__.py`，不要直接沿用。它也**不是数据根，不能传给 `--path`**。历史fallback写法（只有路径实际可用时才适用）：

```bash
export CTSEQ_NUSCENES_PYTHON_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes
export PYTHONPATH="${CTSEQ_NUSCENES_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

不使用旧 `--preloading`；当前原始云按需读取，每worker缓存256MiB。使用 `CUDA_VISIBLE_DEVICES` 选择物理卡，每进程仍为单卡；设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`、`PYTORCH_CUDA_ALLOC_CONF=backend:native` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

八组后台命令见 [v34运行说明](CTSEQTRACK_V34_MINI_LAUNCH.md)，当前协议与工具见 [工具面](FORMAL_TOOLING.md)。旧v33快照有服务器314项测试与真实CUDA batch通过证据；这不替代新v34的真实批次验证。上传后只需一次新的v34批次，不需重装环境。旧版本命令仅供 [历史索引](HISTORY_EVIDENCE_INDEX.md) 复现。
