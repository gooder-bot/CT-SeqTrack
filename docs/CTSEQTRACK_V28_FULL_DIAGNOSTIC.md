# v28 单组完整 nuScenes 诊断（2026-09-10）

当前结论：代码与本地合同检查支持 **Full 模型、Car、seed42、完整 nuScenes、从头60轮** 的一次诊断实验。用户已明确希望先取得完整数据结果再集中分析；该安排替代此前必须先完成 mini 分数恢复才运行任何 full 数据实验的顺序。完整五类多臂矩阵仍未开始，也没有基线稳定恢复或模块增益结论。

## 为什么可以运行，以及这次能回答什么

完整配置 [28_full_nuscenes_full.yaml](../cfgs/ct_seqtrack/28_full_nuscenes_full.yaml) 已存在。与已完成的 mini Full 相比，仅切换数据版本、数据根、train/val/test split、实验名，并清除 mini 的1262步预期；网络、损失、点预算、优化器及耦合合同不变，无需另造一套网络配置。

| 检查面 | 已核对的合同与证据 |
| --- | --- |
| B0 | 四候选总体 shuffle、整 batch 原目标、BC一次、1024槽/3帧历史、因果首帧尺寸；本轮保持已修复的 CE/池化严格确定性路径。 |
| B1 | 使用预测历史与物理时间，输出获取先验和 margin；实际获取记录包含 CV 回退，物理位移误差与递归 endpoint 误差分开统计。 |
| B2 | 读取 detach 的 Seg 第二层64维真实逐点特征；原始点 ID、frame mask、零槽和去重合同连接一致。新证据是 support ID 减整个 B0 raw crop ID。 |
| B3/host | 实际使用结构合法性与效用策略；Full−B3、选择性动作与校准共享候选；训练由 B0 observation 唯一提交递归状态。 |
| 优化 | 单 Adam 四个命名组，全部启用参数从首个合法事务参与；无冻结、预训练或跨臂 checkpoint。机制前向与 B0 的梯度、BN、RNG 隔离。 |
| 数据 | 350个 train_track 场景训练、150个官方 val 评测；内部17/18场景拟合/诊断，与参数训练重叠。官方 val 不拟合策略。 |
| 更新数 | 按真实 `len(observation_dataset)//16` 计算每轮步数；full 无1262约束。机制流保留全部非首帧端点和轨迹尾部。 |
| 保存/评测 | 每轮完整边界保存 last，另存58/59/60；同 run 完整 epoch 恢复。每个 Full checkpoint 独立校准后评测官方 val。 |

三组 mini 提供了实际 CUDA 训练证据：B0/42 与 Full/42 的75,720步观测 loss、末轮 B0 参数/BN/Adam一致，Full各插件已参与18,000次事务。这支持当前隔离确实执行；不等于任意 full 数据运行已经逐位验收。

基线仍有明确问题：mini final 的 B0/42 为 **45.200/47.243**，B0/52为 **52.876/64.478**。seed42未达到历史健康目标 **49.986/58.962**，不能挑seed52宣布稳定恢复，也不能把 mini 的分数阈值直接套到完整 nuScenes。完整数据这次可观察后期退化是否仍存在、B2是否获得更多真实新增目标点，以及校准后的动作是否有用；不能提前断言增加数据会解决问题。

Full在没有绑定策略时，训练期间验证会回退到B0。**这不代表插件没训练，也不能用该分数判断完整Full的收益。** 训练完成必须做下面的独立校准和闭环评测。

## 启动前这次处理的两处问题

1. `tools/preflight_ct_v28.py` 直接读取 YAML，缺少 `main.py` 的 `preloading=False` 默认值；真实数据工厂会访问该字段并报 `AttributeError`。补齐工具默认值，不修改训练目标。旧的“首轮只能B0”输出文案也不应被当作当前用户安排的限制。
2. 验证/测试导出的场景、轨迹、partition及参数训练重叠标记需要来自真实数据集，不能继续用 `scene_id=unknown`、无条件 `partition=dev`。修复限定在诊断导出，保留原预测输入、递归采样key和历史监控tag，避免为改日志改变分数。`precision/dev` 和 `dev_diagnostics/` 作为旧名称保留，实际划分看导出行与provenance。

本地无完整 nuScenes、CUDA及真实 Lightning 环境，实际 full 数据量、GPU峰值及 epoch恢复仍需服务器证据。独立审计见 [数据与控制面](../artifacts/ct_checks/reports/20260910_v28_full_readiness/data_control_audit.md)、[模块耦合](../artifacts/ct_checks/reports/20260910_v28_full_readiness/coupling_audit.md)；最终本地检查数写入 [验收记录](CTSEQTRACK_V28_LOCAL_VALIDATION.md)。

## 完整数据的资源安排

本次命令**省略 `--preloading`**。当前实现按轨迹逐帧保存整幅点云，未按 frame token 去重；观测训练、机制训练、验证分别持有数据对象，预加载完整集可能消耗大量RAM。按需读取继续使用同一帧读取、采样和训练目标，代价是磁盘读取增加。

即使关闭预加载，完整 epoch 的CPU递归状态与诊断记录仍比mini大；机制尾部若需在一个观测步执行多个tick，反向前的图也可能增加。预检报告会给出实际数据/调度预算，但16步短检不等于已覆盖完整epoch峰值。保留12 workers、batch16，不为缩短时间减少端点或训练步数。总耗时只能用服务器第一轮实际速度估计，不按mini时长直接承诺。

## 服务器启动命令

先同步本次修改的代码和文档，在已激活的 `seqtrack3d` 环境、仓库根目录执行。以下只启动一组 **GPU3 / Full / Car / seed42**；若使用其他空闲卡，只改 `CUDA_VISIBLE_DEVICES`。数据根沿用 [SERVER_PATHS.md](SERVER_PATHS.md) 已登记的完整数据路径。

下面这段命令也已保存为 [launch_full_car_seed42.sh](../artifacts/ct_checks/reports/20260910_v28_full_readiness/launch_full_car_seed42.sh)，可在仓库根目录用`source artifacts/ct_checks/reports/20260910_v28_full_readiness/launch_full_car_seed42.sh`执行；与下面代码块二选一，勿重复启动。

先检查完整索引和16步真实训练，再从头后台启动60轮。短检 checkpoint 保留在工程目录供排错，绝不作为正式初始化；任一步失败，下面不会启动长跑。

```bash
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes
STAMP=$(date +%Y%m%d-%H%M%S)
CHECK_ROOT="artifacts/ct_checks/${STAMP}-v28-full-data-check"
FULL_DIR="output/${STAMP}-28_full-nuscenes_full_car_seed42_60ep_bs16"
mkdir -p "$CHECK_ROOT" "$FULL_DIR"

if python tools/preflight_ct_v28.py \
    --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml \
    --path "$DATA_ROOT" --output "$CHECK_ROOT/preflight.json" \
    > "$CHECK_ROOT/preflight.log" 2>&1 \
  && python tools/check_train_steps.py \
    --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml \
    --path "$DATA_ROOT" --steps 16 --workers 12 --seed 42 \
    --artifact-dir "$CHECK_ROOT/train16" \
    > "$CHECK_ROOT/train16.log" 2>&1; then
  nohup python -u main.py \
    --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml \
    --path "$DATA_ROOT" --category_name Car \
    --batch_size 16 --epoch 60 --workers 12 --seed 42 \
    --check_val_every_n_epoch 5 \
    --tag nuscenes_full_car_seed42_60ep_bs16 --log_dir "$FULL_DIR" \
    > "$FULL_DIR/train.log" 2>&1 < /dev/null &
  echo $! > "$FULL_DIR/train.pid"
  printf '已启动，目录：%s\n' "$FULL_DIR"
else
  printf '真实数据检查失败，未启动长跑。查看：%s\n' "$CHECK_ROOT"
fi
```

查看进度时执行 `tail -f "$FULL_DIR/train.log"`；退出tail用Ctrl+C，不影响后台训练。新终端需重新设置 `FULL_DIR` 为实际带日期目录。正式运行必须从epoch0开始，不传任何checkpoint参数；后续如确需恢复，只使用同目录的完整epoch checkpoint，遵守现有恢复身份校验。

## 跑完后固定评测，不只读训练日志

以下命令在训练真正完成后运行。保留训练时的代码版本与环境，使用该run的 `resolved_config.yaml`；不能只用未包含CLI预算的基础YAML拟合策略，再用不同配置评测。不要复用其他epoch、run或旧代码的policy。

每个58/59/60 checkpoint均运行：同一Full权重的observation闭环对照、内部17场景校准与18场景锁定诊断、官方150场景selective闭环评测。observation结果有助于定位同一模型内B0问题，但不冒充独立训练的正式B0臂或原始SeqTrack对照。

```bash
CFG="$FULL_DIR/resolved_config.yaml"
EVAL_STAMP=$(date +%Y%m%d-%H%M%S)
EVAL_ROOT="$FULL_DIR/evaluation_${EVAL_STAMP}"
mkdir -p "$EVAL_ROOT"

for E in 058 059 060; do
  CKPT="$FULL_DIR/formal_checkpoints/epoch=${E}.ckpt"
  ED="$EVAL_ROOT/epoch_${E}"
  mkdir -p "$ED/observation" "$ED/selective"

  python -u main.py --cfg "$CFG" --checkpoint "$CKPT" --test \
    --proposal_mode observation --log_dir "$ED/observation" \
    > "$ED/observation.log" 2>&1 || break

  python -u tools/calibrate_ct_actions.py --v28 \
    --config "$CFG" --checkpoint "$CKPT" --output "$ED/policy.json" \
    > "$ED/calibration.log" 2>&1 || break

  python -u main.py --cfg "$CFG" --checkpoint "$CKPT" --test \
    --ct_action_calibration_path "$ED/policy.json" \
    --log_dir "$ED/selective" \
    > "$ED/selective.log" 2>&1 || break
done
```

上面的评测循环是前台顺序运行示例。需要后台执行时保存为bash脚本后用nohup；不要让三轮同时拟合并挤占同一GPU。官方val评测不传 `--ct-eval-partition dev`。策略选择never也可能是合法结果，必须查看policy有效性与动作计数，不能强制阈值来制造增益。

结果固定报告final60和58/59/60三个独立评测分数的算术平均，**不是平均权重，也不是挑best**。必要时在同一Full权重下追加 `--proposal_mode bounded_always` 的闭环诊断，用来区分候选动作质量与B3选择问题；不将它标记为独立训练的Full−B3臂。

集中分析至少包括：S/P与后期趋势、逐场景/轨迹差异、首次失跟与空crop恢复、新目标点从获取到pool/selected的流失、结构合法动作及raw/bounded/accepted效用、B0/B1/B2/B3实际更新计数，以及训练/校准耗时与资源。完整数据分数必须最终与同协议、同类别的完整数据对照比较；不能沿用mini健康分数作full验收线。
