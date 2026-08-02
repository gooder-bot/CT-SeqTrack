# CT-SeqTrack 服务器路径

本文件记录 CT-SeqTrack 在服务器上的固定数据与依赖源码路径。训练、评测、
协议构建和数据检查命令应优先引用这里的路径，避免把数据集根目录与 Python
包源码目录混用。

## 路径清单

| 用途 | 服务器路径 | 传给 `--path` |
| --- | --- | --- |
| KITTI Tracking `training` 根目录 | `/home/lishengjie/data/cxtrack/training/` | 是 |
| nuScenes mini 数据根目录 | `/home/lishengjie/data/nuscenes-mini/` | 是 |
| nuScenes Python 包源码目录 | `/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/` | 否 |

### KITTI Tracking

`datasets/kitti_mf.py` 同时接受包含 `training/` 的父目录和 `training/` 本身。
服务器上的规范写法使用直接数据根目录：

```bash
--path /home/lishengjie/data/cxtrack/training
```

该目录下应直接存在：

```text
label_02/
velodyne/
calib/
```

父目录 `/home/lishengjie/data/cxtrack/` 也能被当前加载器自动解析，但新命令和
实验记录统一保存直接的 `training/` 路径，减少歧义。

### nuScenes mini

`datasets/nuscenes_lidar_mf.py` 会把配置中的 `path` 原样传给
`NuScenes(version=..., dataroot=path)`。服务器命令应使用：

```bash
--path /home/lishengjie/data/nuscenes-mini
```

mini 配置必须同时使用 `version: v1.0-mini`；数据根目录应包含
`v1.0-mini/` 以及对应的 `samples/`、`sweeps/` 等目录。

### nuScenes Python 包源码

项目代码通过 `from nuscenes...` 导入 nuScenes devkit。服务器上已有的包源码是：

```text
/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
```

这个目录不是数据集根目录，不能传给 `--path`。如果当前 Python 环境没有安装
`nuscenes-devkit`，应把包目录的父目录加入 `PYTHONPATH`：

```bash
export CTSEQ_NUSCENES_PYTHON_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes
export PYTHONPATH="${CTSEQ_NUSCENES_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

## 推荐的服务器环境变量

```bash
export CTSEQ_KITTI_ROOT=/home/lishengjie/data/cxtrack/training
export CTSEQ_NUSCENES_MINI_ROOT=/home/lishengjie/data/nuscenes-mini
export CTSEQ_NUSCENES_PYTHON_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes
export PYTHONPATH="${CTSEQ_NUSCENES_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

示例：

```bash
# nuScenes-mini：当前 ct_v2 入口
python tools/ct_v2/run.py train \
  --variant baseline \
  --path "$CTSEQ_NUSCENES_MINI_ROOT"

# KITTI Tracking：直接 main.py 入口
python main.py \
  --cfg cfgs/seqtrack3d_kitti.yaml \
  --path "$CTSEQ_KITTI_ROOT" \
  --seed 42
```

在启动长实验前可检查路径和 Python 导入：

```bash
test -d "$CTSEQ_KITTI_ROOT/label_02"
test -d "$CTSEQ_KITTI_ROOT/velodyne"
test -d "$CTSEQ_KITTI_ROOT/calib"
test -d "$CTSEQ_NUSCENES_MINI_ROOT/v1.0-mini"
test -f "$CTSEQ_NUSCENES_PYTHON_ROOT/nuscenes/__init__.py"
python -c "from nuscenes.nuscenes import NuScenes; print('nuscenes import: OK')"
```

