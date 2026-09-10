# v29 修改必要性、独立复核与继承限制（2026-09-10）

v29 登记为 **SeqTrack 适应化 B0**。三臂是 `29_b0_nuscenes_full.yaml`、
`29_full_cfc_nuscenes_full.yaml`、`29_full_gru_nuscenes_full.yaml`，首批完整 nuScenes
Car、seed42、60 epoch。它们共享新的 B0 观测训练协议；不再将 v29 称为原 SeqTrack
或 v28 的逐位等价复现。旧 `28_*` 配置与历史输出继续保留。

历史 mini 的 B0/42 为 S/P=45.200/47.243，B0/52 为52.876/64.478；这些记录不足以
说明固定 seed42 已稳定恢复。用户已明确授权完整数据三臂实验，因此本版不重复要求
mini60作为启动条件；启动前的实际数据、严格 CUDA 确定性和续训工程检查仍需执行。
历史分数依据见[三组实验报告](../artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md)。

## 从 v28 到 v29 的改动及必要性

| 改动 | 原问题与本版处理 | 边界 |
| --- | --- | --- |
| 先修帧布局，再接 mask | 原 `[B,C,L*N]` 直接 reshape 为 `[B*L,C,N]` 混合帧与通道，不能直接按历史帧解释。现在显式拆帧、交换轴、展平，再从帧存在性及采样有效性构造 local/global/cross mask。 | 不增加 Transformer 层数、宽度、参数或新的时序网络。历史版本默认调用保持原行为。 |
| 无效 source 与 corner 输出处理 | 原历史 valid mask 没有有效进入注意力。只 mask logits 会使全空行产生均匀伪注意力或 NaN；只 mask 一层又会被 residual/FFN bias 重新填入值。 | v29 全无效 key 的注意力和加权输出为零；每层 source 与不存在的历史 corner query 输出继续归零。当前 corner query 始终存在。 |
| 1/2点真实槽 | v28 原兼容规则在2→3点处由全零突变为完整点输入。v29 对1/2个真实点重复补槽，沿用公共采样算法。 | 0点及至少3点的采样规则不变。真实重复点保留身份，但只能计一次物理证据；B0仍按槽训练。 |
| 当前权重短 roll-in | 仅用 GT/扰动 GT 历史训练与长时间预测历史部署存在分布差异。原请求 candidate0 保留 teacher，1—3在窗口起点扰动一次，再由当前 B0 递归产生历史。 | `s=max(0,t-4)`，最多预测3步，当前帧一次有梯度监督；没有上一 epoch 缓存、GT纠偏重置、插件状态或跨臂 checkpoint。 |
| 输入框与 GT 标签分离 | online predicted boxes 若覆盖 `prev_boxs` 后继续生成 seg/BC/ref GT，会把预测误差当真值监督。v29 在覆盖前保留 GT 历史，标签与输入均转换到同一预测 anchor。 | GT只用于标签和诊断；原请求teacher的重抽规则仍有GT可见性条件，roll-in/机制/评测不因漂移或空观测筛除端点。 |
| 物理 motion/coarse 与 endpoint 分工 | `GT_t−GT_(t−1)` 和 `GT_t−pred_(t−1)` 在历史漂移时不同。若共享粗运动输出同时被两个目标拉动，会形成监督冲突。 | `b0_coarse_target` 与 motion 使用输入 anchor 轴下的物理位移及GT相对转角；最终 Transformer 输出仍监督当前 GT 在该 anchor 下的位置。 |
| 初始化可靠性显式区分 | 局部第一个 query 不代表其初始化框准确；roll-in起点已经候选扰动。 | `is_initial_query` 与 `history_reference_reliable` 分别表达时序和可靠性；扰动起点及预测历史使用软prior，真实部署首帧GT初始化保留0/1prior。 |
| 获取 support 的 Z 包络统一 | XY投影可覆盖目标，而过窄 support Z 会在真实获取时丢掉测量；最大可达诊断若使用另一套几何，会错误归因于选择器。 | endpoint/tube/corridor 与实际 B0 crop 的 Z 取包络；margin监督和最大可达范围复用同中心、方向、Z及合法margin。XY、动作半径、768→256预算不变。 |
| Full 已接受状态参与后续机制输入 | 只沿 observation 状态训练，部署接受动作后访问的 crop/history 状态没有得到同样训练。 | 每轨迹按稳定hash使用 never/always/threshold0 行为，epoch0起生效；host提交最终框，下一机制输入读该状态。不会回灌共享B0观测流。 |
| B3 即时效用与闭环策略选择 | 即时动作回报与H3诊断混在一个q目标中，含义不清；只按固定轨迹one-step数据选择策略又忽略后续状态改变。 | q只拟合即时bounded动作S/P增益；H3只作诊断。内部17场景实际闭环比较never、always及最多3个阈值，18场景仅锁定诊断，官方val不拟合策略。 |
| 版本与续训身份 | 改观测、目标、状态转移或校准合同后复用旧checkpoint，会使实验解释及恢复路径不明确。 | `utils/v29_contracts.py`字段进入配置、resume及策略身份；正式训练只能scratch或同run完整epoch边界恢复。 |

## 保留的限制，不能被本版结论覆盖

1. **mask 的范围是 Transformer。** 当前 DecoderLayer 实际只有 cross-attention 与 FFN，
   没有独立的 corner self-attention；本版没有添加一个原本不存在的子层。普通 BN、
   SegPointNet/MiniPointNet 的全序列计算、固定槽位分割/BC损失和硬 argmax 保留。
   因此不能声称整个 B0 对 padding 数量、重复点或缺失帧完全不变；BN受训练批次统计影响。
   不存在的历史 corner 输出被屏蔽，但 ref loss仍保留原整体分母，不另做有效帧重归约。

2. **短 roll-in 不是整条轨迹 on-policy 训练。** 每个样本在局部GT起点初始化，含25%的
   原请求teacher候选，局部最多3次模型递推。它让当前模型接触自身短期误差，仍不能覆盖
   长距离失跟、长空crop或全程累计漂移。四候选总体shuffle/drop_last保留；teacher重抽
   可改变实际candidate，因此teacher比例指原请求模式，不能从actual candidate反推。

3. **计算量增加必须披露。** 有梯度观测样本数、batch16、60epoch及Adam更新数不变，
   但成熟端点的roll-in候选额外执行3次样本等价前向；全候选总体约四分之三走此路径，
   早帧更少。Full另有完整机制事务。`ct_b0_rollin_*`和实际epoch吞吐用于记录成本，
   不能将“相同更新数”写成“相同FLOPs/运行时间”。无梯度前向暂时eval以隔离BN/RNG，
   不是参数冻结；所有启用模块从首个合法事务学习。

4. **全空回退有明确位置。** 全段没有真实采样测量时，eval/roll-in持有参考框；只当前
   crop为空但历史有测量时仍允许网络预测。注意力为零不意味着所有FFN偏置或训练输出
   自然为零；整段全空的最终hold由host处理。真实1/2点保留不保证低信息输入可可靠定位。

5. **B1和B2受物理可辨识性限制。** 历史框整体共同偏移无法单凭相对运动唯一识别；
   超过最大合法support范围的目标不属于B2选择器能够恢复的测量。新增证据始终是
   `support raw ID − 整个 B0 raw crop ID`，不会把crop内未采样点重新标为新增。
   当前base和短memory继续读已有seg监督的second64特征并detach；重复真实点仅计一次，
   不存在历史ID=-1且不产生memory token。扩大Z可能增加背景，须报告实际获取质量。

6. **状态行为与策略效用不是安全保证。** 25%/25%/50%是hash桶的期望比例，不是每轮
   精确端点配额；训练always行为可能接受有害动作，以覆盖部署可访问状态。B3按结构合法
   候选学习正负效用，不能用presence硬门删除负例。缺失/错误策略仍按已有合同回退B0。
   内部17/18场景与参数训练重叠，必须披露；这里是效用策略拟合，不是概率校准或统计
   安全保证。三臂也不足以分别证明B1、B2、B3贡献，需要后续独立消融。

## 本轮独立复核范围与实测证据

复核了 `utils/v29_rollin.py`、`models/seqtrack3d.py`、`datasets/sampler.py`、注意力层、
采样身份、物理/endpoint损失、行为转移及epoch/resume接口。没有改冻结参考仓库。
在已覆盖的CPU实际host路径内，没有发现新的必现接口阻断。边界如下：

- 布局哨兵覆盖batch/frame/channel/point，不只检查shape；部分历史无效、全空source、
  decoder导出、全有效时旧新attention前向/梯度逐位等价均有测试。
- 真实sampler验证预测历史不是seg/BC/ref GT，改变当前或未来GT只改变监督，不能改变
  因果输入；0/1/2点、空输入hold、真实点身份和support Z经过集成检查。
- `tests/test_ct_v29_b0_host.py`以完整batch16比较三臂B0前向/loss/梯度/BN/Adam逐位
  一致，插件梯度不回流B0。与注意力采样专项合并执行取得 **23 passed、1 CUDA skipped**。
- 独立补充 `tests/test_ct_v29_review.py`，执行真实 `training_step → backward → Adam`：
  同一teacher+roll-in批次，Full额外顺序处理frame1和frame2两个机制tick。三臂B0参数、
  梯度、Adam、BN、全局RNG与观测fingerprint逐位相同；Full状态确实提交至frame2，
  safe/transaction标志恢复。该补充取得 **1 passed**，不是伪造网络输出的流程测试。
- accepted后下一query读取最终框、获取训练/评测输入一致性、策略闭环及合同身份由
  另组本地集成回归覆盖。完整pytest结果以[本地验收记录](CTSEQTRACK_V29_LOCAL_VALIDATION.md)
  为准，不将不同专项数量相加当作全量或CUDA验收。

本地没有运行真实nuScenes、Lightning/CUDA服务器训练、100步同卡审计或实际epoch续训。
人工检查epoch接口只能确认：下一epoch重置机制轨迹，数据/采样epoch与随机流恢复；
roll-in没有跨batch隐藏缓存。实际连续/恢复逐位一致仍须执行
[服务器检查与运行流程](CTSEQTRACK_V29_SERVER_RUNS.md)。

## 历史CUDA错误必须保留为回归条件

| 主错误 | 根因及已有处理 | v29必须保持 |
| --- | --- | --- |
| `nll_loss2d_forward_out_cuda_template` | 原分割空间NLL加权mean不支持该CUDA环境的严格确定性。保留原class-axis log_softmax，仅展平所得log概率后用二维NLL。 | 整批加权分母不变，不关闭deterministic。 |
| `cumsum_cuda_kernel`，仅Full触发 | AP统计对二值标签使用浮点CUDA累计。改为先int64 cumsum，再转换浮点统计。 | 两处相关AP路径都覆盖；这不是优化器或模型得分的因果解释。 |
| AdaptiveMaxPool CUDA反向限制 | 全局池化及1024→128固定槽池化改为固定分组`max`，保留并列最大值首索引梯度。 | 不使用改变梯度分配的`amax`；非法形状报错，不静默回退旧算子。 |
| `tqdm.__del__`清理异常 | 前面训练异常退出引起的次生错误。 | 先查日志最早主RuntimeError，不以改tqdm掩盖CUDA问题。 |

strict deterministic、CUBLAS workspace、TF32关闭、cuDNN benchmark关闭及
Adam foreach/fused关闭继续继承。当前attention保持显式matmul/softmax实现；
CPU反向通过不证明所有服务器CUDA版本都支持。不得改成`warn_only`后称工程检查通过。
详细历史见[CUDA排错记录](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)。

## 参考来源与得分结论

[SeqTrack3D论文](https://arxiv.org/abs/2402.16249)及本地冻结 `../seqtrack/`用于核查观测
结构和历史目标；[M²-Track：Beyond 3D Siamese Tracking](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Beyond_3D_Siamese_Tracking_A_Motion-Centric_Paradigm_for_3D_Single_CVPR_2022_paper.pdf)
与其[官方代码](https://github.com/ghostish/open3dsot)作为运动建模的相关参考。
此前计划里的M²-Track arXiv链接笔误不应继续复制；没有参考TrajTrack。

这些来源不能证明本次改动必然涨分。正式报告固定final60和58/59/60独立评测分数的
算术平均，Full各checkpoint单独拟合策略，并报告同权重observation与bounded-always。
与SeqTrack“下降不超过1个百分点”的严格结论仍需要同数据/预算/评价器的reference，
S和P分别判断；代码修复、CPU测试通过或两个同样低分的实验都不能代替恢复证据。
