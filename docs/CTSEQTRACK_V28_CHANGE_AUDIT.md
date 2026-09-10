# v28 修改必要性与证据边界（2026-09-10）

本轮实现对应已批准的“先恢复 SeqTrack 观测底座”路线。9月8日实施审计的历史基点为 `8b8b8d9`；该次审计没有改写旧配置、冻结参考仓库和 `output/`。新版本配置及 checkpoint 身份独立，历史 checkpoint 仅用于诊断。9月9日用户已授权服务器正式训练写入带日期的新`output/`目录，历史目录仍不覆盖。

| 修改 | 原问题与必要性 | 本轮边界 |
| --- | --- | --- |
| 独立 `seqtrack_reference_loss` | 旧 CT 在四个 view 内分别归约，与全 batch 加权 CE、moving 分母不等价；共享 Tensor 的原地累加还可能使 BC 重复进入 B0。 | 直接构造原算式，BC 一次；保留原槽位监督。不能将这两个偏差单独归因为历史二十多分的跳变。 |
| 全候选总体 shuffle 与原重抽 | 原参考从四候选总体训练；强制每 batch 候选平衡、固定 candidate 的重抽改变训练分布。 | 独立 RNG 内按完整索引重抽；仅三帧 GT 都无目标点时拒绝，最多64次替换；机制/评测不按 GT 可见性筛端点。 |
| B0 专用采样及原 crop 运算 | v27 的真实1/2点槽、公共 canonical crop 改变原观测输入和浮点路径。 | 只在 v28 B0 恢复原操作；B1/B2 继续保留真实稀疏测量及 ID。未新建补采样支路。 |
| 训练/在线 prior 与空输入 | candidate0 不能表示在线预测历史的可靠性；仅当前空 crop 强制静止会抹掉网络从历史得到的运动预测。 | 首个 query 用初始化 prior，后续用软 prior；只有全部采样无真实测量才 hold，使用有效位。首帧尺寸继续作为因果输入。 |
| 固定最大池化与 Adam 数值设置 | strict CUDA 下原 adaptive max backward 存在确定性限制；默认优化器实现路径也需固定。 | 固定/整除池化使用 `max` 保留首个最大值梯度，非法形状报错；foreach/fused/TF32关闭。CPU等价不证明CUDA历史根因。 |
| 分割CE二维NLL确定性兼容 | 三组服务器训练均在`nll_loss2d_forward_out_cuda_template`退出；空间NLL的加权均值归约不支持该CUDA路径的严格确定性。 | 保留原三维class-axis log_softmax，仅将其结果展平后计算普通NLL；全batch类别权重分母不变。`ct_b0_ce_contract=class_axis_logsoftmax_flat_nll_v1`绑定配置与resume，不关闭确定性，不宣称与旧CUDA归约逐位相同。 |
| AP整数累计统计 | Full在AP指标中的浮点`cumsum_cuda_kernel`触发严格确定性限制，B0未进入该指标路径。 | 二值标签先做整数cumsum，再转回浮点；两处同类统计一并处理，不改变训练损失。CE/AP是本次执行阻断的根因，不能据此解释历史得分崩坏；tqdm异常是退出清理的次生报错。 |
| B2 SegPointNet second64 | FeaturePointNet 原帧/通道排列不能解释为真实逐点特征。 | 一次分割前向额外返回第二层；当前证据和 memory 都消费其 detach，原 FeaturePointNet 主干行为继续保留。 |
| 统一结构候选 | learned presence 的额外硬门会让训练、bounded 动作和校准看见不同候选集合。 | 有限、合法、含真实唯一 extension 的同一候选贯穿全链；不扩大动作半径，不增加动作幅度。 |
| B1 物理/endpoint 诊断分离 | `GT_t−GT_(t−1)` 与 `GT_t−pred_(t−1)` 混用会把递归漂移误当物理预测误差。 | 在 forward 后使用 GT 计算诊断，按 source/reference 坐标旋转；分别报告物理、endpoint、anchor误差。 |
| 新场景/训练/身份合同 | mini6/1/1 的内部dev分数不能与全8场景训练后的官方mini_val直接比较。 | mini8/2，full350/150；内部cal/dev明确与参数训练重叠。新loss/采样/数值合同进入resume与校准身份。 |
| 工程逐步及续训审计 | 旧记录只有初始化和有限参数指纹，不能定位前向、反向还是 Adam 首先分叉。 | 保存选定步的输入、逐层激活/梯度指纹、池化索引、loss、参数梯度、BN、RNG、Adam和实际更新；完整激活按需保存。严格逐位比较失败返回非零。 |

本轮明确继承原 FeaturePointNet 排列、Transformer 历史 mask 行为、BN 及硬 argmax。它们可能限制性能，后续应独立消融；本轮没有同时修正后宣称仍与原网络兼容。没有新增时序网络、运动头、冻结阶段、跨臂初始化或训练预算扩张。

## 本地证据

9月10日完整数据就绪审计新增两处最小修复：preflight补YAML缺省的`preloading=False`，
避免真实数据工厂的AttributeError；验证/测试导出层补真实场景、轨迹、划分和参数训练重叠
标记，修复unknown/dev遗留元信息。只改导出行，不改变前向输入、递归采样key、网络或损失。
用户已明确进行一组完整数据Full诊断，该安排不以mini分数已恢复为前提；
范围、资源和命令见[完整数据单组诊断](CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。

`tests/fixtures/seqtrack_v28_reference.py` 保存原 SeqTrack 前向/损失算式及来源 SHA256，只有 CE class weight 的 CUDA 固定位置改为随输入设备。真实网络测试覆盖不均衡候选和 moving 数量0/1/3/4，逐位比较完整前向、各项loss、BC梯度、全部参数梯度和 Adam 更新。

另以真实 B0/GRU/CfC/Full−B3/Full/reference 六配置比较两步 B0梯度、参数、BN、Adam和RNG；检查插件loss不回流B0、Seg特征真实顺序、稀疏数据身份、全空回退以及低presence的结构合法动作。完整回归结果记录在本轮交付的本地验收报告；CUDA及真实数据测试单独标记未执行。

9月9日CE与AP修复分别取得专项47 passed/1 skipped、15 passed/1 skipped；两次覆盖不同，不能相加作为全量或CUDA验收。本地无CUDA，修复后的服务器运行仍须确认。错误链、修复文件及后续提醒集中维护在 [CUDA排错记录](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)。

恢复工具按项目锁定的 Lightning 2.0.2 处理 `Checkpoint` 回调：保存发生在模型 epoch-end hook 之后；普通 StepLR 在最后训练 batch 后更新。这一核对依据 [回调源码](https://raw.githubusercontent.com/Lightning-AI/lightning/2.0.2/src/lightning/pytorch/trainer/call.py) 和 [训练循环源码](https://raw.githubusercontent.com/Lightning-AI/lightning/2.0.2/src/lightning/pytorch/loops/training_epoch_loop.py)。实际恢复一致性仍须运行服务器工具验证。

## 得分验收

历史健康参照 S/P 为 **50.986/59.962**，首轮目标分别为 **S≥49.986、P≥58.962**。这只是健康恢复目标，严格“下降≤1个百分点”还需同协议 reference 的 final60 和58/59/60均值分别比较；不能用两次同样低分宣布恢复。

9月10日已读取三组完整60轮结果：B0/42为45.200/47.243、B0/52为52.876/64.478、Full/42未校准为45.200/47.243。Full/B0 seed42观测训练和末轮状态一致，说明插件没有改变这次B0训练；固定seed42仍未达到健康恢复目标。完整证据见[三组分析](../artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md)。三组均scratch60epoch、batch16、workers12、每5轮官方mini_val验证，正式输出位于`output/20260909-003318-28_模块-mini_car_seedXX_60ep_bs16/`。固定报告两个已登记seed，不挑seed、best epoch或延长训练遮盖失败；其余模块及完整nuScenes论文实验仍待B0恢复证据。历史命令见 [服务器运行说明](CTSEQTRACK_V28_SERVER_RUNS.md)。
