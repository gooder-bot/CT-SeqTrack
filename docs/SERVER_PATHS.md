# CT-SeqTrack 服务器路径

本文件记录 CT-SeqTrack v28/v29 使用的服务器路径。v29三臂完整数据使用下述完整nuScenes根路径，
不预加载；新命令见[服务器运行说明](CTSEQTRACK_V29_SERVER_RUNS.md)。历史 KITTI/HTV/M2 路径已移出活动文档。

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

当前用户指定的完整数据根为：

```text
/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
```

它必须包含 `v1.0-trainval/` 与 `samples/LIDAR_TOP/`，并在服务器运行前
通过 `tools/preflight_ct_v28.py` 的真实数据路径验证。通过 `--path` 显式覆盖配置。
不得把 mini 路径用于 `28_*_nuscenes_full.yaml`；v28 full 配置固定为
`version: v1.0-trainval`。

单组Full/Car/seed42、60轮诊断的命令见 [v28完整数据启动](CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。
首次省略`--preloading`，避免按轨迹重复缓存整幅点云造成过大的CPU内存占用。

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
python tools/preflight_ct_v28.py --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml --path /home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes --output artifacts/ct_checks/v28_full_preflight.json
```

不传`--manifest-only`才实际构造数据并检查索引；真实batch短检及正式启动按上述专页执行。
独立100-step与完整epoch resume等价验收仍须单独记录，不能用本地CPU测试替代。
工程checkpoint不得用于正式初始化，正式运行仍从epoch0开始。

本地代码回归以pytest为主；`verify_ct_slimming.py verify`固定要求旧HEAD=`001951a`，
当前后续提交会因历史基点限制失败，不能把它误当成新的full数据运行阻断。
