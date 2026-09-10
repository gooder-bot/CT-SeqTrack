# v29 完整 nuScenes 运行性能（2026-09-11）

用户报告 B0 在第19619/50687步约1.64秒/步。本轮优化保持观测输入、短窗口roll-in、采样和几何顺序、损失归约、Adam更新数、参数所有权及递归状态提交方式。昂贵的纯诊断允许抽样；不能减少实际训练样本或用旧预测缓存代替当前模型的roll-in。

## 配置与结果解释

新增三份独立 `29_b0_nuscenes_full_perf.yaml`、`29_full_cfc_nuscenes_full_perf.yaml`、`29_full_gru_nuscenes_full_perf.yaml`。旧配置保留原执行路径。性能配置显式启用 `ct_runtime_optimization=equivalent_v1` 和 `ct_diagnostic_policy=sampled_v1`；具体参数以三份配置的继承解析结果为准。训练内部诊断抽样不是训练数据抽样，未采集项不得写成测量值0。

准确的约束表述是：**训练目标与参数更新轨迹不变，诊断覆盖率改变**。不能写“所有训练计算完全不变”：H3影子前向和排名指标确实减少，额外buffer快照及隔离检查也有成本。参数轨迹的实际逐位验收仍区分本地CPU与服务器CUDA。

需要完整同步本次代码及配置；不能只复制新YAML。已启动的旧进程不会读取更新后的Python实现。新实验为完整nuScenes Car、seed42、batch16、workers4、scratch60，不预加载，不冻结。checkpoint仍每两轮保存并额外保留59轮，每5轮验证。

本地没有服务器nuScenes数据和训练CUDA环境。目前不能承诺加速比例、CUDA通过或分数恢复。短工具运行报告只是工程证据，不代替60轮正式结果。已有输出目录和checkpoint保留，不用工程产物作为正式初始化。

## 原因与不能更改的边界

- B0每个观测batch都要执行最多三步当前模型roll-in，这是v29已确认的训练方法，保留。不能通过减少历史或点槽、复用前一epoch预测、取消roll-in来报告等价加速。
- 原roll-in对每个CUDA buffer逐一`torch.equal`，使主线程反复等待GPU。优化聚合设备上的比较结果，只恢复变化buffer，保留未变buffer的版本计数。完整snapshot clone保留，不能声称消除了其带宽成本。NaN仍按`torch.equal`判为不相等；CPU、特殊layout、shape/dtype差异走原比较，存在共享storage的buffer保留原逐项比较和恢复顺序。
- 原日志对全部标量逐项`.item()`，每事务直接写TensorBoard。优化批量传输标量、降低写入频率；class balance、优化器更新计数、BN和实际训练目标仍按原事务更新。
- 内存、分类排名、H3等纯诊断与训练目标分开。采样后的计数/覆盖率必须随结果披露，不把未做H3视为零收益，也不把前100步采样统计当成完整轨迹结论。
- 原输入指纹只在前100步；B0/Adam哈希在initial、step1/100以及epoch边界执行，不是第19619步持续慢的已知原因。本轮未通过关闭正式审计夸大速度。

保留故障修复：分割保持原类别轴log-softmax后二维NLL及全batch加权分母；两处AP保持int64累计；固定分组`max`保持并列最大值首索引梯度。仍为严格deterministic、FP32、TF32关闭、cuDNN benchmark关闭、Adam foreach/fused关闭、启动前`CUBLAS_WORKSPACE_CONFIG=:4096:8`。不得用warn-only或关闭确定性提速。`tqdm.__del__`错误仍先追溯前面的主RuntimeError，见[CUDA故障记录](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)。

## 9月11日补充：全量统计与抽样写出分层

- `self.log(..., on_epoch=True)`仍每步调用；loss、类别统计、梯度归属、更新计数、资源采集不抽样。额外的observation/mechanism loss累加器在每个事务更新，按实际行数加权，epoch末一次写出完整统计。
- 直接TensorBoard loss标量在前5步及每50步写；`loss_observation`、`loss_mechanism`明确区分两类事务。排名、二值校准诊断、直方图在前5步及每100步抽样。只保留每类最后一个事务的detached载荷，**由`on_train_epoch_end`统一flush尚未写出的尾批**；不根据`num_training_batches`猜最后一步，周期步已经写出的尾批不重复写。
- relation排名输出在`ct_relation_sampled`组内使用`sampled_ct_relation_*`及`sampled_relation_points/rows`。epoch二值指标使用`ct_epoch_calibration_sampled`组与`sampled_*`字段、实际样本数。它们都是抽样统计；不能冒充完整epoch AP/AUROC/ECE。checkpoint、scheduler、模型选择不得配置为消费这些指标；正式StepLR和固定epoch保存未改。
- H3仅允许v29新性能合同抽样，并要求`instantaneous_sp_gain_v1`。稳定SHA-256摘要含seed、epoch、tracklet、frame、事件名和`ct_v29_h3_sampling_v1`版本，不使用Python `hash()`或训练RNG。
- H3日志分别记录`scheduled_events / sampled_events / executed_events / valid_events / not_sampled_events`及逐行failure reason；valid收益均值仅除以实际valid事件数。未抽样仍为missing，不能当成零收益。B0没有H3，不应期待这部分加速。
- B3 loss通过H1专用标签视图，运行时拒绝读取H3字段；实际抽中事件前后断言policy输出、accepted框和主递归状态没有变化，并隔离BN/buffer、训练标志和RNG。H3内的确定性异常继续上抛，不能吞成普通缺失诊断后宣称CUDA检查通过。
- metadata是新增零点云IO的窄接口；通用`get_frames()`返回及预加载别名语义保留。裁剪保持点/ID顺序、dtype、shape、stride和可写性；未知子类或附加属性继续原deepcopy路径，不强行套快速复制。

验证/最终评测/训练内部策略拟合维持完整覆盖。workers固定4，不打开persistent_workers；它会保留worker内Dataset实例，本项目epoch状态须另行设计同步。性能建议参考[PyTorch调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)与[PyTorch2.0.1 DataLoader实现](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/utils/data/dataloader.py)，本轮不引入新的网络方法。

## 短测速和逐位对照

新工具复用`main.py`的真实DataLoader、roll-in、机制事务、Lightning反传和Adam。`benchmark`默认每臂ABBA：原1→优化1→优化2→原2，三臂共12次独立scratch短跑；`profile/equivalence`保留每臂AB两次。普通工程限制为100步，不运行全套源码哈希门禁；启动前检查目标卡空闲，不终止其他用户进程。

三种模式必须分开解释：

| 模式 | 用途 | 产物与限制 |
|---|---|---|
| `benchmark` | 20步预热+80步平均、P50/P90；只在batch边界同步 | `cycle_ms`包含batch和批间等待，`interbatch_ms`包含loader/Lightning/传输等。仍在前100步输入指纹期，不能直接外推第19619步稳态；首次数据加载不计入80步统计。 |
| `profile` | 同样20+80，记录loader next等待、roll-in CPU准备/传输/前向/恢复、观测和机制前向、loss、H3、backward、optimizer及日志 | 阶段边界同步会增加开销；包含时间可以嵌套，不能相加，也不能用这个模式的加速比声称正式吞吐。只记录实际调用的阶段，无调用时不伪造观测。 |
| `equivalence` | 100步中选0/1/2/3/4/5/10/100保存紧凑状态快照，直接逐字节比较 | 所有模型参数、buffer/BN、梯度、Adam、scheduler、RNG及递归状态都比较；选择步的输入、训练loss及观测/动作输出也比较。只排除明确的`ct_h3_*`和`ct_shadow_*`纯诊断输入字段，不广泛忽略`ct_*`。另外比较优化后三臂共享B0。没有逐层激活/源码哈希清单，但快照仍需磁盘空间；此模式耗时不用于测速。 |

工具会移除子进程继承的`CT_V28_AUDIT_DIR/CT_V28_AUDIT_ACTIVATIONS`，避免把原全层审计开销混进短测速。正式运行本就不应设置这些变量。工具没有关闭训练确定性、改变workers或修改正式配置。内存报告为测量期间观察到的峰值；已有分阶段峰值重置会限制它对瞬时峰值的覆盖。

停止旧任务并同步代码后，在服务器项目根目录执行。下面使用GPU1，必须先空闲：

```bash
DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
CHECK_ROOT="artifacts/ct_checks/$(date +%Y%m%d-%H%M%S)-v29_performance"

# 同卡ABBA；前20步预热，后80步统计，三臂依次运行。
python tools/profile_ct_v29_training.py --path "$DATA_ROOT" --gpu 1 \
  --mode benchmark --workers 4 --output "$CHECK_ROOT/benchmark"

# 训练数值及共享B0隔离对照；该耗时不能当训练速度。
python tools/profile_ct_v29_training.py --path "$DATA_ROOT" --gpu 1 \
  --mode equivalence --workers 4 --output "$CHECK_ROOT/equivalence"

# 如需进一步定位B0开销，再单独运行分段插桩。
python tools/profile_ct_v29_training.py --path "$DATA_ROOT" --gpu 1 \
  --arms b0 --mode profile --workers 4 --output "$CHECK_ROOT/profile_b0"

# H3单独微基准：每个Full从真实通路寻找首个合法事件，固定输入/权重做ABBA。
# 每阶段2次预热+5次实际执行，不依赖100步内偶然抽中几个10%事件。
python tools/benchmark_ct_v29_h3.py --path "$DATA_ROOT" --gpu 1 \
  --arm full_cfc --output "$CHECK_ROOT/h3_cfc"
python tools/benchmark_ct_v29_h3.py --path "$DATA_ROOT" --gpu 1 \
  --arm full_gru --output "$CHECK_ROOT/h3_gru"

# 三卡均空闲后，再测四波ABBA，每波三个任务并行；固定B0/CfC/GRU→1/2/3。
python tools/profile_ct_v29_training.py --path "$DATA_ROOT" --mode benchmark \
  --parallel --gpus 1 2 3 --workers 4 --output "$CHECK_ROOT/parallel_abba"
```

普通模式写`report.json`；每次运行写`train.log`和`runtime.json`，数值模式另写`snapshots/`。报告均值、P50/P90、吞吐与两次重复差异；改善小于重复波动时证据不足。第一处错误或数值差异使工具非零退出并停止后续阶段。并行工具失败仅清理自己启动的进程会话，不按GPU杀外部进程。`--plan-only`只打印命令，不宣称检查通过。

H3单独写`h3_microbenchmark.json`，强制执行同一合法事件以比较实际前向开销，并逐位对照H3标签及主状态不变；这不是10%覆盖收益或所有H3事件分布的估计。若前100个生产batch没有合法事件，工具明确失败/缺少覆盖，不能算通过。更广的事件分布留待服务器长一些的诊断观测，不据此修改正式采样或seed。

代码是否等价由数值对照判定；有无加速由同卡ABBA、三卡并行和随后正式首轮吞吐确认。未证明改善前，新perf配置只是显式候选配置，不替换旧默认配置。

## 三个正式后台命令

### 本地实施验收（2026-09-11）

最终专项 **220 passed / 2 skipped**；两项跳过均为本地无CUDA（CE重复、CUDA buffer恢复）。
这不是全仓pytest计数，也不是服务器速度或分数验收。运行范围为新数据/roll-in/日志/性能合同/测速/H3微基准，
以及相邻v27 host、v28 CE和控制面、v29合同测试。实际host覆盖两个Full在主计算图存活时开/关H3后反传与Adam，
并比较loss、梯度、BN/buffer、参数、优化器和RNG；三臂旧/新执行对照及混合teacher/roll-in输入也通过。

`compileall models/ datasets/ utils/ tools/`、所有23个变更/新增Python文件的Python3.9语法、
`git diff --check`、三份配置合同检查、两个新工具的`--plan-only`及文档四组Bash命令的`bash -n`通过。
保留旧配置及历史output，不执行固定HEAD的slimming哈希门禁。

仍需服务器执行：真实CUDA前后向、独立逐位对照、同卡ABBA、H3合法事件微基准和三卡并行测速。
本地实现允许开始短验证；只有真实速度改善且数值合同通过后，才按下面命令采用perf配置正式重启。

工程结果满足本轮要求后，从新日期目录重新scratch。GPU物理编号由环境变量指定，程序内仍为单卡cuda:0；不用DDP。

**GPU1：B0**

```bash
cd ~/study/lcyu/CT-SeqTrack
B0_DIR="output/$(date +%Y%m%d-%H%M%S)-29_b0_perf-nuscenes_car_seed42_60ep_bs16"
mkdir -p "$B0_DIR"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  python -u main.py --cfg cfgs/ct_seqtrack/29_b0_nuscenes_full_perf.yaml \
  --path /home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/ \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 --check_val_every_n_epoch 5 \
  --tag nuscenes_car_seed42_60ep_bs16 --log_dir "$B0_DIR" \
  > "$B0_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$B0_DIR/train.pid"
echo "$B0_DIR"
```

**GPU2：Full-CfC**

```bash
cd ~/study/lcyu/CT-SeqTrack
CFC_DIR="output/$(date +%Y%m%d-%H%M%S)-29_full_cfc_perf-nuscenes_car_seed42_60ep_bs16"
mkdir -p "$CFC_DIR"
nohup env CUDA_VISIBLE_DEVICES=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  python -u main.py --cfg cfgs/ct_seqtrack/29_full_cfc_nuscenes_full_perf.yaml \
  --path /home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/ \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 --check_val_every_n_epoch 5 \
  --tag nuscenes_car_seed42_60ep_bs16 --log_dir "$CFC_DIR" \
  > "$CFC_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$CFC_DIR/train.pid"
echo "$CFC_DIR"
```

**GPU3：Full-GRU**

```bash
cd ~/study/lcyu/CT-SeqTrack
GRU_DIR="output/$(date +%Y%m%d-%H%M%S)-29_full_gru_perf-nuscenes_car_seed42_60ep_bs16"
mkdir -p "$GRU_DIR"
nohup env CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  python -u main.py --cfg cfgs/ct_seqtrack/29_full_gru_nuscenes_full_perf.yaml \
  --path /home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/ \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 --check_val_every_n_epoch 5 \
  --tag nuscenes_car_seed42_60ep_bs16 --log_dir "$GRU_DIR" \
  > "$GRU_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$GRU_DIR/train.pid"
echo "$GRU_DIR"
```

新终端按`20*-29_b0_perf-nuscenes_car_seed42_60ep_bs16`、`29_full_cfc_perf`、`29_full_gru_perf`匹配相应目录，再`tail -n 20 -F "$RUN_DIR/train.log"`。旧不带`_perf`的目录只用于历史记录。
