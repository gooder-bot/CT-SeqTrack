# v29 实施合同（2026-09-10）

本版是 **SeqTrack 适应化 B0**，不再声称与原 SeqTrack 或 v28 前向逐位等价。
当前登记三臂为 B0、Full-CfC、Full-GRU，完整 nuScenes Car、seed42、60 epoch。
代码和本地测试完成不等于 CUDA 工程验收、基线恢复或涨分；实际检查记录见
[本地验收](CTSEQTRACK_V29_LOCAL_VALIDATION.md)，服务器操作见[运行说明](CTSEQTRACK_V29_SERVER_RUNS.md)。

## 输入与监督

- `utils/v29_contracts.py` 统一定义配置、checkpoint/resume、策略身份。v29 继承 v28
  的确定性数值路径和场景协议，替换观测语义身份；旧 `28_*` 配置和历史输出保留。
- FeaturePointNet 以 `[B,C,L*N]→[B,C,L,N]→[B,L,C,N]→[B*L,C,N]` 排列。
  3 个历史帧、每帧 1024 槽、128 个 source token、网络宽度和层数保持。
- 注意力 source 可见性为帧存在与真实采样测量的交集，local/global encoder 和
  decoder cross-attention 共用；全无效 key 产生零注意力，source 输出继续归零。
  **当前 Decoder 实际没有独立 self-attention**，因此历史存在性控制 corner query
  的 cross-attention 和输出，没有凭空增加 self-attention 参数；当前 query 始终有效。
- 0 点为空，1/2 点保留真实点并重复补槽；真实重复点只计一次物理证据。
  不存在的历史 ID=-1、valid/unique=false，不生成 memory 证据。
  普通 BN、Seg/Mini 全序列池化和原槽位损失仍保留；不能声称全网络 padding 不变性。
- 三臂共用整批一次 B0 loss，BC 一次，四候选总体 shuffle/drop_last，不要求 batch 内均衡。

`utils/v29_rollin.py` 使数据 worker 只提供 raw 窗口。原请求 candidate0 使用现有
teacher 路径（保留至多64次全索引空历史重抽，记录 original/actual/retry）；模式不会
因重抽得到不同 candidate 而切换。candidate1—3 从 `s=max(0,t-4)` 的 GT 初始化框开始，
仅此处施加候选扰动，尺寸取整条轨迹首帧。host 在当前 Adam 更新前最多三波
`eval + no_grad + observation-only` 预测历史，当前帧只执行一次梯度前向。
空输入或漂移不触发 GT 重置，局部窗口前历史无效，绝对帧号/时间和初始化 query 显式区分。
历史框可靠性另行标记：扰动窗口起点采用0.2/0.8软prior，真实部署首帧GT初始化采用0/1 prior。
frame0 端点没有中间 roll-in；仍作为原始候选总体中的监督样本，不丢弃或重复补步。

roll-in 不使用插件状态、缓存或旧 checkpoint，恢复 BN/缓冲区、训练标志和全局 RNG。
`ct_b0_rollin_*` 单独记录样本等价前向次数、耗时和 teacher/roll-in 行数。
crop/prior/ref 输入使用预测历史；seg、BC、ref 标签来自 GT 历史并转换至输入 anchor。
最终输出监督当前 GT；物理 motion 与 coarse 目标是输入 anchor 轴下的真实位移及相对转角，
通过 `b0_coarse_target` 显式传递，避免用同一 motion 输出同时拟合绝对漂移纠正。

## 获取、动作与递归

- B1 保持相对物理运动与 margin 获取先验，不能根据共同偏移的历史唯一识别绝对漂移。
- `utils/acquisition_v29.py` 将 endpoint/tube/corridor 原 Z 区间与实际 B0 crop Z 区间取包络。
  XY、margin `[2,1]→[6,3]`、9×9 网格、90%覆盖目标、768→256、128/96/32 配额不变。
  最大可达范围和 margin 监督复用实际 support 的中心、方向、Z 与合法 margin 上限。
  AcquisitionRecord 记录三类 support 的 Z 区间/存在性，但不把这些诊断拼进网络。
- 新证据仍是 support raw ID 减 **整个 B0 raw crop ID**。Seg 第二层64维特征 detach
  后供当前 base 和短 memory 使用。结构合法不代表目标存在；负例训练 presence/效用，
  只有 target-bearing 的新增点参与中心/vote 回归，没有 presence 硬门。
- `ct_acquisition.v5` 增加 novel-only XY/XYZ/Z排除、最大可达/不可达及各级保留统计；
  旧 v4 字段保留。长距离漂移超过合法搜索范围时，不能将失败归因于选择器。
- 机制轨迹以稳定 hash `(seed,epoch,true_tracklet_key,mechanism_behavior_v1)%4`
  分配 never/always/threshold0/threshold0，从 epoch0 生效，轨迹内固定。
  这是哈希桶的期望比例，不保证每个 epoch 恰好25%/25%/50%的轨迹或端点。
- `utils/v29_policy.py::policy_transition` 为训练与部署共用纯转移，host 是递归状态唯一写者。
  Full 的 accepted 状态仅用于机制流，不回灌共享 B0 训练流。训练行为通过事务参数处理，
  不伪造 `calibrated=True`；部署缺失或身份不匹配的策略仍回退 observation。
- B3 仅拟合即时 bounded-action Success/Precision 增益，两者均值为 q；保留即时
  gain MSE 和 help/harm BCE，`loss_ct_b3_instantaneous` 明确命名。H3 仅为单次干预后
  B0 回退续推两步的诊断，不进入本版 q 的损失。旧输出中的 H3 别名为兼容字段，不能
  用作“预测长期效用”的证据。动作半径仍为 `min(0.5+0.5*dt,2)`，只更新 XY。

## 实验与结论边界

三臂均350个 train_track 场景训练、150个官方 val 场景验证/评测，Car、seed42、
batch16、workers12、60 epoch、每5轮验证。Adam lr=1e-4、betas=(0.5,0.999)、
eps=1e-6、wd=0、foreach/fused=false；StepLR20×0.1、FP32、无新增梯度裁剪。
每轮更新次数取实际完整候选总体长度/drop_last；增加的 roll-in 计算量单独披露。
完整数据不预加载，产物存入带日期的新 output 目录；scratch 或同run完整 epoch 边界恢复，
所有启用参数从首个合法事务学习，无冻结和跨臂 checkpoint。

两个 Full 的58/59/60 checkpoint **分别**在内部17场景进行策略拟合，18场景只做锁定诊断，
披露与参数训练重叠；never、always及最多3个筛选阈值都经过真正闭环运行。
按整体效用、S、P、较少动作打破并列，官方150场景不选阈值/后端/epoch。
报告 final60 与三个独立分数的算术平均，补同权重 observation 和 bounded-always 对照。
不把阈值拟合写成概率校准、统计安全保证或全局重定位；三臂不能单独证明每个模块贡献。

首批不要求额外 mini60。工程检查通过后运行三臂完整60轮，首个真实完整 epoch 核对资源、
吞吐和端点覆盖。历史 v28/seed42 S/P=45.200/47.243、seed52=52.876/64.478，
尚不能据此声称稳定恢复；完整同协议原 SeqTrack 对照及独立模块消融另行登记。

确定性保留已修复的 fixed max pooling、二维 NLL 与 AP int64 cumsum；禁止 warn_only 降级。
`tqdm.__del__` 异常先追溯前面的主 RuntimeError，详见[CUDA历史](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)。
