# V41 KITTI Car 单组联合训练

本页保留逐轮更新困难库的一组 [Full 配方](../cfgs/ct_seqtrack/41_full_kitti_joint_training.yaml)，用于与已经完成的 v3 uniform 比较。当前 1/4/4/8 与四独立训练候选的两组实验见[联合模型配对说明](V41_JOINT_PAIR.md)。

| 部分 | 本轮设置 |
|---|---|
| 主干与优化器 | KITTI Car，uniform XYZ 中心损失，B0、物理时间 B1、MTP 和 B3/R。seed 42，Adam，batch 16，4 个 worker，初始学习率 0.00015，第 20、40 轮各乘 0.2。 |
| q0 横向头 | 第一轮起用当前点监督位移和采用比例。默认第 11 轮开始将横向修正送入递推，B3 对最终 q0 重新取支持。原始中心和航向保留监督，共享特征继续训练。 |
| 搜索与接管 | 六个候选参与选择。局部共 1,024 点，远区每次最多额外精裁一处、256 点。健康时每 1 秒巡查，连续两帧弱观测后每 0.2 秒巡查；待复核位置优先使用下一物理扫描。模板、身份、几何和唯一支持点决定远区接管。 |
| 训练历史 | 普通窗口沿用 1/4/4/8 课程；受扰起点从训练集 E57 历史中按点数档匹配连续 XYZ 残差。每轮整轨辅助流收集本模型提交当前帧前的联合状态，下一轮用一部分 branch 0 槽回放。库中包含运动、模板和待复核状态，随完整轮次 checkpoint 保存。 |
| 默认关闭 | 四候选各自重置、固定 E2 公共前缀库、旧 uniform E10/E60 前缀回放、GT 校正训练。旧 E57 权重仅用于离线生成增强数据。 |

## 启动

在训练服务器的项目根目录、项目 Python 环境中运行。脚本使用已分配的可见 GPU。首次需要可访问的 KITTI 训练集和历史 uniform E57 供体；若已有匹配的增强库，只需提供库和 KITTI 数据。脚本完成数据准备后只启动一次正式训练。

```bash
bash tools/train_v41_joint.sh
```

库在其他位置时可用 `HISTORY_BANK=/path/to/history_e57.json`，历史权重移过目录时可用 `DONOR_RUN=/path/to/uniform`。服务器只有文件快照时，`DONOR_SOURCE=/path/to/frozen-v3` 可指定与 E57 匹配的原始源码。可用 `PYTHON` 和 `DATA_ROOT` 指定解释器与数据根目录。启动脚本接受 `main.py` 的训练参数，例如 `--log_dir`。

准备好增强库后，也可以直接从同一个训练入口启动：

```bash
python main.py --cfg cfgs/ct_seqtrack/41_full_kitti_joint_training.yaml \
  --v41_history_bank /path/to/history_e57.json
```

`--v41_q0_lateral_warmup_epochs 0` 可让横向修正从第一轮就进入递推。`--v41_patrol_seconds` 和 `--v41_weak_patrol_seconds` 可调整两种巡查间隔。运行清单记录实际值；继续同一运行时使用该运行的 resolved 配置与完整轮次 checkpoint。

## 用同一次训练判断结果

每轮评估 test，在 E58、E59、E60 保存完整逐帧结果。将下面的目录替换为本次运行目录。

```bash
python tools/compare_v41_joint_search.py --joint-run /path/to/new-run
```

报告列出实际配方、E60 与末三轮平均 Success/Precision、重现后连续三帧恢复、健康轨新增失跟、错车接管及超过 10 m 的帧数。同时按两边共同可见的物理帧统计 q0 横向、纵向、高度误差及轴向角误差尾部，困难帧由历史 uniform 粗框确定。历史 test 已用于研究失败模式，因此结果描述在这些既有场景中整套修改的表现；单次训练无法分别估计各模块贡献。
