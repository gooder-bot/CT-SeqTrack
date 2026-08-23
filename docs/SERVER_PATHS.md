# CT-SeqTrack 服务器路径

本文件记录 Safe-SeqTrack v25（以及只读 v24 证据）需要的服务器路径。历史 KITTI/HTV/M2 路径已移出活动文档。

## nuScenes mini

数据根：

```text
/home/lishengjie/data/nuscenes-mini
```

运行时传入：

```bash
--path /home/lishengjie/data/nuscenes-mini
```

配置必须使用 `version: v1.0-mini`，并确认数据根包含 `v1.0-mini/`、
`samples/` 和 `sweeps/`。

## 完整 nuScenes

完整数据根必须在服务器运行前显式确认，并通过 `--path DATA_ROOT` 覆盖配置。
不得把 mini 路径用于 `25_*_nuscenes_full.yaml`；v25 full 配置固定为
`version: v1.0-trainval`。

## Python 环境

服务器现有 nuScenes Python 包源曾位于：

```text
/home/lishengjie/code/SparseFusion-main/nuscenes
```

该路径是 Python 包父目录，不是数据根，不能传给 `--path`。若环境未安装
`nuscenes-devkit`，可使用：

```bash
export CTSEQ_NUSCENES_PYTHON_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes
export PYTHONPATH="${CTSEQ_NUSCENES_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

## 正式启动前检查

```bash
python tools/verify_ct_slimming.py verify
python -m pytest -q
python main.py --cfg cfgs/ct_seqtrack/25_b0.yaml --path DATA_ROOT --help
```

完整环境中的真实 batch 前向/反向、100-step resume 等价和点/框可视化仍是
正式训练前验收项；当前本地环境无法把它们记为已通过。工程验收 checkpoint 必须
在对照完成后丢弃，正式运行仍从 epoch0 开始。
