# CT-SeqTrack

CT-SeqTrack 是基于 SeqTrack3D 的连续时间 3D 单目标跟踪项目。当前版本已从大量互相耦合的实验分支收敛为一条论文主线：

```text
SeqTrack3D
  + Continuous-Time Motion Prior
  + Time-Guided Search Expansion
  + Adaptive Proposal Fusion
```

目标不是继续堆叠模块，而是在正常 nuScenes 上稳定涨点，并用同 checkpoint 的 `true / fixed / shuffled` 时间控制确认真实时间没有退化。Random-20% 只作为最终鲁棒性补充，不参与首轮选模。

## 当前模型

1. **Continuous-Time Motion Prior**：从历史框和真实 `delta_t` 学习速度，按当前时间间隔生成候选帧位移先验；训练时以 clean/correlated history 混合替代纯 GT history。
2. **Time-Guided Search Expansion**：保留 SeqTrack3D 原搜索区域，额外构造有界轨迹 tube；总点数仍为 1024，其中默认 75% 来自原搜索区域。
3. **Adaptive Proposal Fusion**：根据观测特征、运动特征、proposal disagreement、点云可靠性、时间间隔和扩展比例，预测小幅有界修正。

TWC、旧 Observability Gate、M3 EMA teacher、M4 Kalman/filter 等路线保留为历史代码，默认配置全部关闭。第一版 v2 不引入 TWC；只有三模块完成正常集晋级后，才考虑第二阶段的一致性损失。

详细数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，论文计划见 [refined_plan.md](refined_plan.md)。

## 唯一推荐入口

配置仅看 `cfgs/ct_v2/`，运行仅用：

```bash
# 1. 同代码 baseline
python tools/ct_v2/run.py train --variant baseline --path /data/nuscenes-mini

# 2. 连续时间运动先验
python tools/ct_v2/run.py train --variant motion --path /data/nuscenes-mini

# 3. 加时间引导搜索扩展
python tools/ct_v2/run.py train --variant motion_search --path /data/nuscenes-mini

# 4. 完整 CT-SeqTrack v2
python tools/ct_v2/run.py train --variant full --path /data/nuscenes-mini
```

正常验证：

```bash
python tools/ct_v2/run.py test \
  --variant full \
  --checkpoint /path/to/last.ckpt \
  --path /data/nuscenes-mini
```

完整模型通过 mini 晋级后，再运行 full nuScenes：

```bash
python tools/ct_v2/run.py train \
  --variant baseline_full \
  --path /data/nuscenes

python tools/ct_v2/run.py train \
  --variant full_dataset \
  --path /data/nuscenes
```

时间负对照使用同一 checkpoint。`fixed` 可直接运行；`shuffled` 先生成冻结 manifest：

> 2026-07-25 以前的 B3 时间控制会通过 observation statistics
> 继续读取真实 `dt`，不能作为 v2 因果证据。修复后的 gate、motion、
> innovation radius 与 search tube 均只消费 `current_delta_t_effective`，
> 因此必须用新代码重新运行三路控制。

```bash
python tools/build_dynamics_time_manifest.py \
  --cfg cfgs/ct_v2/04_ct_seqtrack_v2.yaml \
  --path /data/nuscenes-mini \
  --role test \
  --output protocols/manifests/ct_v2_mini_test_shuffled_seed42.json

python tools/ct_v2/run.py test \
  --variant full \
  --checkpoint /path/to/last.ckpt \
  --path /data/nuscenes-mini \
  --time-mode shuffled \
  --time-manifest protocols/manifests/ct_v2_mini_test_shuffled_seed42.json
```

可选 Random-20% 复测：

```bash
python tools/ct_v2/run.py test \
  --variant full \
  --checkpoint /path/to/last.ckpt \
  --path /data/nuscenes-mini \
  --protocol random20
```

命令可加 `--dry-run` 只打印最终 `main.py` 调用。原有 `main.py --cfg cfgs/<legacy>.yaml ...` 命令保持可用。

## 目录

```text
models/ct_v2/       连续时间运动与自适应融合
utils/ct_search.py  训练/评测共享的时间引导搜索
utils/ct_history.py 轻量的相关历史误差契约
cfgs/ct_v2/         当前唯一活跃的消融配置
tools/ct_v2/run.py  当前唯一推荐运行入口
docs/legacy/        旧阶段计划和运行说明
compare_results/    历史结果与正式分析
```

## 现有证据边界

- 现有 M2 相对历史 A1 在 standard 和 gap1124 有正信号，但包含训练路径混杂。
- 正确时间尚未稳定优于 fixed/shuffled，因此当前不能声称“真实时间已被证明产生因果收益”。
- v2 必须用同代码 baseline、同数据、同 seed、同训练步数重新建立结论。
- B1–B3 使用 candidate0 clean、其余 candidate correlated 的历史混合；
  canonical displacement/velocity 标签不随该输入扰动改变。

历史细节见 [sum_results.md](sum_results.md) 和 [done.md](done.md)。
## 相关工作

- [SeqTrack3D, ICRA 2024](https://arxiv.org/abs/2402.16249)：多帧点云与历史框序列 baseline。
- [StreamTrack, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28196)：流式多帧记忆。
- [HVTrack, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1145_ECCV_2024_paper.php)：历史视角跟踪与扩展搜索的收益/背景噪声边界。
- [TrajTrack, 2025/2026](https://arxiv.org/abs/2509.11453)：显式运动 proposal 与隐式历史轨迹联合细化。
- [ChronoTrack, CVPR 2026 Findings](https://arxiv.org/abs/2604.13789)：对齐后的紧凑时序记忆与一致性目标。
