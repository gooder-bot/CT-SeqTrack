# CT-SeqTrack v27 实施记录

本轮由用户授权实施：公共数据修复、B1 获取先验、B2 新增证据、B3 收益选择，五个 CT 臂加 SeqTrack 参考；全部 scratch、无参数冻结。

完整 nuScenes 使用全部 350 个 train_track 场景训练、官方 150 个 val 场景评测。训练集内部 17/18 个场景只用于阈值拟合/闭环诊断，不从训练中扣除。mini 保留 6/1/1，官方 mini_val 评测。

## 已实施

起始 HEAD 为 `b445ecd`；本轮未提交 Git。历史配置、output 和参考仓库没有改动。

| 范围 | 实现与验收内容 |
|---|---|
| 数据集合 | `utils/v27_protocol.py` 固定 full350/150、训练内部17/18、mini6/1/1；缓存含场景manifest摘要；observation/mechanism使用同一完整训练集合。 |
| 公共通路 | 原始 point ID 贯穿裁剪、坐标变换、采样、memory；区分 padding/重复/唯一证据；保留1/2点，空base返回reference仍可执行extension恢复。 |
| 时间与训练 | effective clock、prepass/重算共享构造器、mechanism临时完整eval/no_grad后恢复flag；四命名Adam组、梯度隔离、公共层命名初始化与独立随机流。 |
| B1 | CfC/GRU并存；145→64→2独立margin分支、81网格novel-target监督；实际AcquisitionRecord与带梯度学习输出分开；CV fallback、长车投影尺寸修复。 |
| B2 | 局部几何、目标/上下文role条件、128+96+32选择、独立relation类别计数、targetness-only投票、长车vote半径和top-mode摘要。 |
| B3 | 共用实际有界动作，S/P gain heads与H1/H3监督；41分位策略筛选、真实calibration闭环、锁定dev诊断；新artifact绑定checkpoint/config/source/scene/metric。 |
| 递归遍历 | v27不丢不等长轨迹尾部；部分槽位batch和必要时有序多tick保证每个有效非首帧每epoch一次，同时保持B0更新预算。多tick插件损失按该事务内endpoint数加权。 |
| 评价报告 | 所有endpoint含首帧/空帧；benchmark_compat与geometry_exact双口径；global→base→support→novel→768→256→top-mode→raw/bounded/accepted漏斗。 |
| 实验入口 | 13份27_* YAML（公共base、五臂mini/full、两个SeqTrack参考），六运行/五类30运行矩阵，逐58/59/60 checkpoint校准与评估命令。 |

方法定义及文献见 [CTSEQTRACK_V27_METHOD.md](CTSEQTRACK_V27_METHOD.md)，协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

## 本地验收结果

- 修改前：176 passed，1 skipped。
- 第一轮实施后：**290 passed，1 skipped**，26.10秒；[第一轮pytest输出](../artifacts/ct_checks/v27_acceptance/pytest.txt)。
- `python -m compileall -q models/ datasets/ utils/ tools/ main.py`：通过。
- `git diff --check`：通过；历史 `cfgs/` 与 `output/` 的 tracked diff 为空。
- 真实sampler/host方法的CPU测试覆盖：GT替换只改监督、B1三种clock一致、NaN learned prior的实际CV获取、selected-presence、精确H1/H3、完整endpoint、政策加载、部分槽位与多tick。
- 五臂B0的参数/BN/梯度/Adam/RNG两步一致性已在真实host事务加CPU小网络替身中通过；GRU/CfC公共层使用真实模块核对。此项不等同于完整CUDA骨干的数值等价验证。
- `verify_ct_slimming.py verify` 仍因要求 HEAD=`001951a` 而拒绝当前起始HEAD=`b445ecd`。这是旧slimming快照限制，不能把该门记为通过，也不据此判断v27失效。

随后完成[训练与checkpoint二次审计](CTSEQTRACK_V27_TRAINING_READINESS.md)：补测实际六臂网络、完整Full训练事务、五臂B0两步更新对齐、普通/双流连续训练与epoch-boundary resume，并修复稀疏重采样、保存顺序、恢复随机流、离线Full加载和旧诊断口径。最新结果以[二次审计摘要](../artifacts/ct_checks/v27_training_reaudit/summary.json)为准。

这些验证使用合成点云与实际CPU模型算子。**真实nuScenes batch、CUDA 100-step和正式60轮仍未执行，暂无v27分数。** 本次按用户要求只检查代码与训练逻辑，不审计服务器环境。

## 服务器下一步

以下命令从服务器的 CT-SeqTrack 根目录执行，工程产物使用全新目录。

```bash
# 实际数据与完整递归遍历预检；full时换27_full_nuscenes_full.yaml及完整数据根
python tools/preflight_ct_v27.py --cfg cfgs/ct_seqtrack/27_full.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --output artifacts/ct_checks/v27_mini_preflight.json

# 使用生产main.py、真实双stream与Adam的100-step工程检查
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/27_full.yaml \
  --path /home/lishengjie/data/nuscenes-mini --steps 100 --workers 0 \
  --artifact-dir artifacts/ct_checks/v27_full_100step

# 在服务器生成六运行清单及逐checkpoint后续命令；加--execute才顺序开始训练
python tools/run_ct_v27_matrix.py --stage mini \
  --path /home/lishengjie/data/nuscenes-mini \
  --output artifacts/ct_checks/v27_mini_matrix
```

100-step检查应覆盖CT五臂并加同臂重复：比较B0输入fingerprint、梯度、BN和Adam hash；CUDA不一致时先与同臂重复区分非确定性。工程checkpoint不用于正式初始化，正式六运行仍从epoch0开始。

full矩阵使用 `--stage full`，数据根见 [SERVER_PATHS.md](SERVER_PATHS.md)。每个Full checkpoint先运行矩阵中的校准命令，锁定策略后才执行官方val评估。报告中的acquisition计时含共享sampler标签/sidecar成本，单独记录网络forward时间；诊断吞吐率不能当部署FPS。

## 尚需实验回答

CfC与GRU谁更好、B2是否保留更多有用目标点、Full闭环净收益和类别涨跌，都需要实际数据支持。单seed最终以官方val的U/S/P及配对区间报告，不声明跨seed稳定性或已取得涨分。

历史配置、输出和参考仓库保持原样；新实验不加载历史 checkpoint。
