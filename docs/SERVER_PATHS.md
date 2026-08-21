# CT-SeqTrack 服务器路径

本文件只记录正式 v24 实验需要的服务器路径。历史 KITTI/HTV/M2 路径已移出活动文档。

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
不得把 mini 路径用于 `24_*_nuscenes_full.yaml`。

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
python main.py --cfg cfgs/ct_seqtrack/24_b0.yaml --path DATA_ROOT --help
```

完整环境中的真实 batch 前向/反向、100-step resume 等价和点/框可视化仍是
推荐验收项；用户已明确跳过本轮服务器 smoke，因此不把它们记为已通过，也不再
阻止四臂命令交付。若之后执行，这些工程验收 checkpoint 必须丢弃，正式运行仍从
epoch0 开始。
