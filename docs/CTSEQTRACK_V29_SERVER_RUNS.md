# v29：完整 nuScenes / Car 三臂服务器流程

2026-09-11性能重启：用户已允许昂贵纯诊断抽样；使用独立`*_nuscenes_full_perf.yaml`，
短测速、逐位数值对照和GPU1/2/3分开的新日期后台命令见[性能记录](CTSEQTRACK_V29_PERFORMANCE.md)。
以下原配置命令保留作历史；不把本地CPU检查解释为服务器CUDA/速度已通过。

2026-09-10 登记（按最新重启要求更新）：B0、Full-CfC、Full-GRU，seed42，各自随机初始化训练60轮，batch16、workers4，每5轮验证。全部启用参数从首个合法事务开始优化；不冻结、不使用工程或其他臂的 checkpoint 初始化。v29 改变观测和机制状态合同，是新实验版本，不能把旧 v28 checkpoint 当作续训起点。

这里给出待执行命令。本地 CPU 测试不能替代服务器上的真实数据、CUDA、100步及续训检查；没有在本地启动服务器训练。分数是否改善仍由正式实验决定。

最新用户要求：本次只做轻量配置/历史报错核对，不运行全套哈希检查，也不把工程报告哈希作为启动前置条件。
本次先停止正在运行的v29三臂，再以workers4在新日期output目录scratch重启。旧日志和checkpoint保留，
不传`--checkpoint`，不将workers12运行的状态用于workers4续训。v29原有workers12硬校验已调整为允许4或12，
配置新默认为4；重启前必须同步`utils/online_contract.py`和`cfgs/ct_seqtrack/29_formal_base.yaml`。
三臂已继承`ct_checkpoint_every_n_epochs: 2`，在完整epoch结束后保存
`formal_checkpoints/epoch=002.ckpt、004.ckpt……060.ckpt`，另保存059供58/59/60评测。
`lightning_logs/version_0/checkpoints/last.ckpt`随每两轮保存更新；验证仍每5轮进行。
先同步最新`main.py`、`utils/lightning_runtime.py`、`utils/online_contract.py`和`29_formal_base.yaml`及其余v29实现。

## 1. 可选工程检查（保留命令，本次不作为前置步骤）

在服务器 CT-SeqTrack 仓库和 `seqtrack3d` 环境中执行。完整数据根见 [SERVER_PATHS.md](SERVER_PATHS.md)。全部命令关闭预加载，避免三个进程各自缓存完整 nuScenes。保持严格确定性，不设置 `warn_only`。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
STAMP=$(date +%Y%m%d-%H%M%S)
CHECK_ROOT="artifacts/ct_checks/${STAMP}-v29-engineering"
mkdir -p artifacts/ct_checks

nohup python -u tools/run_ct_v29_checks.py \
  --path "$DATA_ROOT" --gpu 1 --workers 4 --output "$CHECK_ROOT" \
  > "${CHECK_ROOT}.log" 2>&1 < /dev/null &
echo $! > "${CHECK_ROOT}.pid"
printf '工程目录：%s\n' "$CHECK_ROOT"
tail -f "${CHECK_ROOT}.log"
```

检查使用物理 GPU1。每个 GPU 阶段开始前查询已有 compute 进程；该卡不空闲就报错退出，不终止任何其他任务。执行期间也应由调度者保持该卡独占，不并行启动正式训练。`Ctrl+C` 只停止 `tail`，后台检查继续运行。

工具按顺序执行：

1. 构造真实 Full-CfC 数据集，验证350个训练场景、150个官方验证场景及完整机制 endpoint 覆盖；v29 使用生产 `observation_collate → prepare_observation_batch → B0 forward/loss` 检查一个真实 batch，不创建 optimizer。
2. 通过 `main.py` 分别 scratch 运行 B0-A、B0-B、Full-CfC、Full-GRU，各100个 observation/Adam 事务，同卡、同 seed。额外无梯度短 roll-in 不增加有梯度 observation 样本或 Adam 次数。
3. 比较初始化及step1/2/3/4/5/10/100的 B0 输入、前向、loss、BN、梯度、Adam及更新量；B0-A分别对B0-B/CfC/GRU，必须逐位相同。不是全100个step都保存完整张量。
4. 对三臂各执行独立的连续训练与完整epoch边界恢复比较，每个工程epoch限制16步。此项复用 `check_ct_v28_resume.py`，不代替60轮正式训练或正式验证过程。

任一步失败就停止后续步骤，查看 `CHECK_ROOT` 中对应阶段日志。工程文件全部保留在 `artifacts/ct_checks/`，不会成为正式训练初始化。

```bash
python tools/run_ct_v29_checks.py --assert-passed "$CHECK_ROOT/report.json"
```

这条命令仅检查工程报告身份；`--plan-only`不能通过它。本次用户选择直接启动，不运行此报告检查，也不宣称工程检查已通过。

## 2. GPU1/2/3 后台分别训练 B0/CfC/GRU

以下命令无需`CHECK_ROOT`或工程哈希报告。三个输出目录带同一日期时间戳，日志为各目录的 `train.log`，PID为 `train.pid`。

```bash
(
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
STAMP=$(date +%Y%m%d-%H%M%S)
B0_DIR="output/${STAMP}-29_b0-nuscenes_car_seed42_60ep_bs16"
CFC_DIR="output/${STAMP}-29_full_cfc-nuscenes_car_seed42_60ep_bs16"
GRU_DIR="output/${STAMP}-29_full_gru-nuscenes_car_seed42_60ep_bs16"
mkdir -p "$B0_DIR" "$CFC_DIR" "$GRU_DIR"

nohup env CUDA_VISIBLE_DEVICES=1 python -u main.py \
  --cfg cfgs/ct_seqtrack/29_b0_nuscenes_full.yaml --path "$DATA_ROOT" \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag nuscenes_car_seed42_60ep_bs16 \
  --log_dir "$B0_DIR" > "$B0_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$B0_DIR/train.pid"

nohup env CUDA_VISIBLE_DEVICES=2 python -u main.py \
  --cfg cfgs/ct_seqtrack/29_full_cfc_nuscenes_full.yaml --path "$DATA_ROOT" \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag nuscenes_car_seed42_60ep_bs16 \
  --log_dir "$CFC_DIR" > "$CFC_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$CFC_DIR/train.pid"

nohup env CUDA_VISIBLE_DEVICES=3 python -u main.py \
  --cfg cfgs/ct_seqtrack/29_full_gru_nuscenes_full.yaml --path "$DATA_ROOT" \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag nuscenes_car_seed42_60ep_bs16 \
  --log_dir "$GRU_DIR" > "$GRU_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$GRU_DIR/train.pid"
printf '正式输出目录：\n%s\n%s\n%s\n' "$B0_DIR" "$CFC_DIR" "$GRU_DIR"
)
```

这些命令没有 `--preloading`、`--checkpoint` 或 `--init_checkpoint`。训练期间 Full 尚无离线策略文件时，正式验证的选择动作仍按缺失策略回退；训练机制流自己的三种行为策略从epoch0参与，不依赖先做离线拟合。因此训练日志的 Full 验证分数不能代替后续安装有效策略后的 Full 正式结果。

## 3. 训练结束后的58/59/60逐checkpoint评测

确认每组都出现 `max_epochs=60` 且存在 `formal_checkpoints/epoch=058.ckpt`、`059.ckpt`、`060.ckpt`。`last.ckpt` 存在不等于训练完成。评测和拟合均读取该run的 `resolved_config.yaml`，保留训练时的代码与环境，不复用其他run配置。以下操作均在对应训练结束后执行；`RUN_STAMP` 替换为第2步实际目录日期。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
RUN_STAMP=替换为实际日期时间戳
B0_DIR="output/${RUN_STAMP}-29_b0-nuscenes_car_seed42_60ep_bs16"
CFC_DIR="output/${RUN_STAMP}-29_full_cfc-nuscenes_car_seed42_60ep_bs16"
GRU_DIR="output/${RUN_STAMP}-29_full_gru-nuscenes_car_seed42_60ep_bs16"

(
set -e
for E in 058 059 060; do
  DEST="$B0_DIR/evaluation/epoch=$E"
  mkdir -p "$DEST"
  CUDA_VISIBLE_DEVICES=1 python -u main.py \
    --cfg "$B0_DIR/resolved_config.yaml" --path "$DATA_ROOT" \
    --checkpoint "$B0_DIR/formal_checkpoints/epoch=$E.ckpt" --test \
    --workers 4 --seed 42 --log_dir "$DEST" \
    > "$DEST/eval.log" 2>&1
done
)
```

每个 Full checkpoint 必须单独拟合策略并绑定其 checkpoint SHA、配置、源代码及场景身份；不能把60轮的策略文件用于58/59轮。使用训练内部17场景拟合：始终评测never与always，再加入最多3个预筛阈值，至多5次真实完整闭环；以U、S、P、较少动作依次排序。随后在内部18场景对锁定策略和never各做一次闭环诊断。两部分都与参数训练场景重叠，不能宣称独立验证或概率校准。官方150个val场景只评测，不选阈值。

```bash
(
set -e
for BACKEND in cfc gru; do
  if [ "$BACKEND" = cfc ]; then RUN_DIR="$CFC_DIR"; GPU=2; else RUN_DIR="$GRU_DIR"; GPU=3; fi
  CFG="$RUN_DIR/resolved_config.yaml"
  for E in 058 059 060; do
    CKPT="$RUN_DIR/formal_checkpoints/epoch=$E.ckpt"
    DEST="$RUN_DIR/evaluation/epoch=$E"
    mkdir -p "$DEST"
    CUDA_VISIBLE_DEVICES="$GPU" python -u tools/calibrate_ct_actions.py --v29 \
      --config "$CFG" --checkpoint "$CKPT" --path "$DATA_ROOT" \
      --device cuda --output "$DEST/policy.json" > "$DEST/policy_fit.log" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU" python -u main.py --cfg "$CFG" --path "$DATA_ROOT" \
      --checkpoint "$CKPT" --ct_action_calibration_path "$DEST/policy.json" --test \
      --workers 4 --seed 42 --log_dir "$DEST/official_val" \
      > "$DEST/eval.log" 2>&1
  done
done
)
```

策略拟合复用完全相同的role/policy闭环缓存，不通过重新筛选旧轨迹假装闭环。JSONL和逐轨迹诊断会占用磁盘；每个checkpoint至多5个calibration闭环和2个dev闭环，另加1个完整官方val闭环。final报告060；late-3是058/059/060各自S/P的算术平均，不平均权重，不挑最佳轮。

## 4. 已知 CUDA 故障提醒

v28启动时出现过两类严格确定性报错，修复在v29继续保留，详见 [故障记录](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)：

- `nll_loss2d_forward_out_cuda_template`：空间CE归约的CUDA实现不支持所用严格确定性路径；当前保留类别轴log-softmax，将点整理成二维NLL，保留全batch加权分母。
- `cumsum_cuda_kernel`：Full的二值AP诊断做了浮点累计；当前先用int64累计再计算precision。该诊断不参与训练loss。
- 随后的 `tqdm.__del__` 异常是主异常退出后的次生错误。不能通过关闭deterministic或改成warn-only把工程检查算作通过。

v29原始观测样本包含teacher张量包和roll-in原始帧，普通默认DataLoader collate无法正确拼接；不要直接用尚未适配v29的旧 `check_forward_batch.py` 代替上述预检。新预检和100步检查走生产v29数据入口。服务器若再出现首个异常，以该异常为准修复后从新工程目录重跑，保留失败日志。
