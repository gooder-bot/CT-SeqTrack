# CT-SeqTrack 当前服务器路径

2026-10-09 当前为 [v37 五组](CTSEQTRACK_V37_LAUNCH.md)：B0、B1、B1+B2、Full FP32、Full 局部 BF16，seed42、scratch60，共300epoch/358,500更新。代理仅通过 SSH 只读查看，未上传、安装、运行检查、启动或停止任务。服务器尚无本地新增 v37 与工程提速工具；由用户上传完整最新代码，执行一次 Full CUDA 生命周期及短 BF16 数值检查后启动正式实验。

本次快照：HEAD=`aec7fd6fa41b08fd8d2bdfe7061da176b023f31f`；Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2、nuscenes-devkit1.1.9；mini metadata/samples/sweeps齐全，磁盘约1909GiB可用。两张A40各46,068MiB，CUDA与BF16支持均已确认；GPU0/1分别空闲41,852/42,219MiB，仍有其他计算任务。以实际Full峰值安排并发，不保证五组同时放得下，不停止其他任务。服务器原有工作树内容保持。完整快照及本地复核见 [上传就绪报告](../artifacts/ct_checks/20261009-161150_v37_upload_readiness/REPORT.md)。

以下10月4日及更早的“当前”、空闲量和检查要求均为历史记录。本轮命令以 v37 操作页为准；环境绝对路径与 native allocator 保持，无需升级环境。

2026-10-04 当前为 [v36 五组](CTSEQTRACK_V36_LAUNCH.md)：A/C/D/E/F，seed42、scratch60，共300epoch/358,500更新。代理此次只读核验，未上传、安装、启动或停止任务。服务器尚无36配置；上传后由用户执行一次 `tools/check_v36_full_batch.py` 的真实CUDA检查，再启动五组。

20:04只读快照：HEAD=`188f587585a3c2e23a928f6e147bc9accebb1b1d`；Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2、nuscenes-devkit1.1.9；mini的metadata/samples/sweeps均存在，/home约2.7TiB可用。两张A40各46,068MiB；GPU0有其他用户的两个任务，空闲16,016MiB，GPU1空闲45,413MiB。资源会变化；首次新Full检查使用GPU1，后续并发以新检查峰值和实际空闲量安排，不停止其他任务。

下方“当前六组”“无需CUDA检查”及其显存数均为历史记录，不作为本轮操作命令。当前环境绝对路径和native allocator保持，不升级环境、不使用旧预处理或设备参数。

2026-10-02当前安排为[完整链六组seed42](CTSEQTRACK_V35_FULL_SEED42_LAUNCH.md)，旧六组和重复参考已完成。本次只读确认：服务器HEAD=`89f5ba2b7b89e115385d3a8a07961a53ae880e37`；现有Python3.9.19及mini metadata/samples存在；两张A40总显存各46,068MiB，空闲各45,415MiB，没有计算进程，/home空闲约2.8TiB。快照会变化，不代表新Full显存已实测。

用户选择GPU0并发三组scaled、GPU1并发三组normal。上传后用`check_v35_full_batch.py`做一次真实Full CUDA检查再确认并发；代理不写服务器、不安装、不启动/停止任务。以下9月30日与9月28日说明是历史信息，其“无需CUDA检查”不适用于此次首次正式Full路径。

服务器为 `lishengjie@10.109.253.86`，项目根为 `/home/lishengjie/study/lcyu/CT-SeqTrack`。2026-09-30当前安排是[六组新训练](CTSEQTRACK_V35_W_SIX_RUN_LAUNCH.md)，旧五组seed42已完成；由用户上传和启动，代理不上传、不安装、不启动或停止进程。本次仅追加配置及LR登记，不重复要求真实CUDA batch。下方环境和空闲量来自9月28日历史只读快照，本次未查询服务器当前资源。

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

本轮命令见 [v35 W六组说明](CTSEQTRACK_V35_W_SIX_RUN_LAUNCH.md)，当前协议与工具见 [工具面](FORMAL_TOOLING.md)。此前v35五次完整正式运行是现有计算路径的运行证据；本次不改计算路径、不迁移环境、不重复增加CUDA批次前置。旧版本命令仅供 [历史索引](HISTORY_EVIDENCE_INDEX.md) 复现。
