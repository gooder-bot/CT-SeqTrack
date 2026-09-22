# CT-SeqTrack v32 服务器路径

服务器为 `lishengjie@10.109.253.86`，项目根为 `/home/lishengjie/study/lcyu/CT-SeqTrack`。当前授权仅只读；不得自行同步文件、安装依赖、启动训练或停止进程。

## 数据根

| 数据 | 路径 | 配置 |
|---|---|---|
| nuScenes mini | `/home/lishengjie/data/nuscenes-mini` | `dataset: nuscenes_mf`、`version: v1.0-mini`、`ct_coordinate_mode: global` |
| nuScenes full | `/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/` | `dataset: nuscenes_mf`、`version: v1.0-trainval`、`ct_coordinate_mode: global` |
| KITTI | 由实际部署提供数据根，当前不填未经核验路径 | `dataset: kitti_mf`、`version: kitti_tracking`、`ct_coordinate_mode: sensor_relative` |

通过 `--path` 指定数据根。nuScenes根应有相应版本metadata、`samples/` 与 `sweeps/`；full不能沿用mini数据。KITTI需要对应点云、标签、标定和图像尺寸信息，沿用当前数据接口。接口支持不代表full/KITTI实验已完成。

## Python环境与SDK

已核实正式v31运行使用 `/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python`：Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2，GPU0/1为A40。记录来自既有实验，不宣称本次再次查询了服务器状态。当前v32及独立参考网络使用PyTorch算子，不以旧PointNet++ CUDA扩展为前提。

nuScenes Python包源曾位于 `/home/lishengjie/code/SparseFusion-main/nuscenes`。这是包父目录，**不是数据根，不能传给 `--path`**。环境缺少已安装SDK时，既有fallback为：

```bash
export CTSEQ_NUSCENES_PYTHON_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes
export PYTHONPATH="${CTSEQ_NUSCENES_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

不使用旧 `--preloading`；当前原始云按需读取，每worker缓存256MiB。使用 `CUDA_VISIBLE_DEVICES` 选择物理卡，每进程仍为单卡；设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`、`PYTORCH_CUDA_ALLOC_CONF=backend:native` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

mini四组后台命令见 [v32运行说明](CTSEQTRACK_V32_MINI_LAUNCH.md)，当前协议与工程检查见 [工具面](FORMAL_TOOLING.md)。旧版本预检/启动命令只通过 [历史索引](HISTORY_EVIDENCE_INDEX.md) 使用，不进入v32流程。
