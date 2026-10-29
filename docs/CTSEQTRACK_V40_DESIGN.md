# V40：dense B0、B1 多时域辅助与独立搜索保护

状态：代码实现与静态审阅；**按用户要求，本机没有执行环境检查、pytest、compileall、训练或 CUDA smoke**。没有上传服务器，没有运行正式实验。操作入口见 [服务器验收与启动](CTSEQTRACK_V40_LAUNCH.md)。

## 基底与四组

源码基底为提交 `6dc7c4e` 中的 `v38_clean1_concat_b1_selective_update_v1`。`40_registration.json` 记录基底文件 SHA256、新生产源码清单、四份 YAML 字节及验收工具/测试 SHA256。V40 独立命名空间，不导入 v39 运行时模块；旧源码、配置、登记、output 和既有 artifacts 保留。

| label | 可学习 B1 | MTP | B3 | 当前搜索 |
|---|---|---|---|---|
| b0 | 无 | 无 | 无 | accepted 单 ROI |
| b1 | 有 | 无 | 无 | accepted 单 ROI |
| b1_mtp | 有 | 有 | 无 | accepted 单 ROI |
| full | 有 | 有 | 有 | measurement eligibility 触发双 ROI |

四组共用 FA2 与数学等价 CPU acquisition 优化。B1−B0 检验可学习 B1；B1M−B1 检验额外 MTP；Full−B1M 检验 **B3＋搜索保护联合机制**，不是纯 B3 消融。没有自动新增第五组。B0 仍包含历史观测、coarse/q0/history/context/local reader 和非学习 trusted-CV fallback，不能称“完全无历史或运动”。

## 网络与 MTP

保留 dense PointNet、masked BN、first-max、四帧×1024 点、每帧128 source token、32 query、1024 K/V 上限，当前每帧执行一次 B0。没有 G、跳骨干、多路 GPU raw cache 或未来递归图。

FA2只替换三层local、三层global、三层cross的attention core。4头×64维，scale=1/8，非因果，attention dropout训练0.1/评估0，外层dropout0.2不变。投影和其他网络/几何/loss/Adam为FP32，只有kernel入口Q/K/V为BF16，输出返回FP32。Trainer precision32，无全局AMP；FA2固定2.5.8、deterministic backward，正式路径缺依赖或不支持设备明确失败。varlen有效长度仍动态但有上界；padding不能进入softmax分母。math只用于工程参考。

M接在B1原128维 `prior.feature`。128维context与两个查询时间特征拼为130维，`Linear(130,64) → LayerNorm → ReLU → Linear(64,2)`，8642参数，末层零初始化。输入context不detach；B1原mean、sigma、acquisition和decoder耦合合同不改。

当前帧t，以s=t−1为起点，h=2/4对应t+1/t+3；标签是GT_xy(s+h)−GT_xy(s)，不以预测框作GT原点。真实时间FP64相减再转FP32。时间特征为log1p(Δh/0.5)、log1p(Δh/current_dt)。位移尺度max(sqrt(LW),1m)，SmoothL1 beta1，horizon权重1/0.5，每端点按有效horizon归一后对有效端点取均值，λM=0.05，受prior.valid约束。

worker只在训练且enable_m时读取未来annotation元数据，返回future_displacement[2,2]、future_dt[2]、future_valid[2]；不读未来点云、不跨scene、尾部缺标签不删当前样本。在线输入白名单排除当前/未来GT标签与审计字段。M只在训练loss调用，不产生当前候选、搜索中心或状态写入。零初始化意味着首步对context的M梯度为零，验收须看数次更新。

B1/B1M/Full保留原acquisition；B0不注册运行时prior/adapter/M/verifier，不执行其损失或生成acquisition标签。B0在初始化阶段临时消费原decoder/prior随机流，再移除对应模块；公共B0/decoder（排除adapter）初值一致，B1三组prior/adapter初值一致，M和B3独立RNG。候选槽保持q0/prior/trusted_propagation，B0的prior槽无效，全空明确选slot2。

## accepted、trusted 与 SearchPlan

accepted仍写入每帧输出。trusted只允许最终真实q0、当前唯一有效支持≥3且T≥0.5的测量更新，选择margin0、同分q0→prior→fallback。位置更新不自动恢复可信速度；原相邻时间与innovation规则保留。

Full记录上次实际提交的measurement eligibility；初始化None与上次True均用原单ROI，上次False触发双ROI。主ROI仍为accepted；备用ROI使用最后trusted的尺寸/朝向与 `trusted_position + valid_velocity * min(now−trusted_position_time,1.5s)`，速度无效时保持trusted位置。SearchPlan保存两框、原因、配额、真实年龄及截断外推时长。1.5秒只限制新搜索框，**不改B1时间、原recovery/fallback、trusted年龄或速度更新分母**。

取两个定向ROI的真实并集，不取包围大框。主域768、备用独有域256，总1024；原始ID去重、重叠归主域、余量互相回填，不复制点补数。无新增备用点保持旧seed/旧采样；有新增点使用独立命名采样流。所有点仍减同一个accepted anchor，禁止两个ROI分别归零。三个历史帧裁剪不改，不扩为2B。

acquisition保持原L/R几何，R仍用未截断recovery，排除实际裁剪并集全部raw IDs；B3支持只读实际采样点。裁剪前覆盖和采样后支持分别记录，备用点不自动可信。GT仅用于标签与被动统计，不用“目标在裁剪外”的GT条件触发搜索。

## 执行与显存合同

CPU acquisition按同一FP64几何预计算旋转/唯一ID，81网格以4096点分块广播membership，保留q90、前背景计数、面积及u的并列顺序。测试需与冻结v38逐例核对标签，不因提速容忍标签改变。

仅保留原有有界CPU raw cache；递推状态CPU化、不跨批保留grad_fn、临时cache正常/异常释放。Adam前拒绝非有限梯度，不添加梯度裁剪。训练日志detach，逐批内存CPU标量落盘；验证逐帧JSONL流式导出，汇总容器仅含CPU数据。每批不empty_cache。

正式allocator为 `backend:native,roundup_power2_divisions:2`；它是需服务器验证的默认，不是V40显存效果证据。验收要求跨批活跃量不积累、同工作负载缓存进入平台，不要求reserved=allocated。内存工具保留allocator、CPU快照原址恢复模型/BN/Adam/RNG及递推状态，核对输入/mask/长度/分支指纹；三次重放、至少256成熟批及train→val→train。allocated/reserved配对差上限64/256MiB，持续增长独立判失败；预热后retries/OOM增量为0。两波实际双进程NVML与各进程峰值校核，预留max(2GiB,总显存10%)，不得静默缩batch或换精度。

## 配方、指标与文献边界

mini Car、global、真实时间、seed42、bs16、workers4、scratch60；LR1e-4，完成20/40轮后×0.2，Adam betas(0.5,0.999)/eps1e-6/wd0；原1/4/4/8及10轮窗口课程保留，所有启用模块epoch1参与。保留训练合法GT窗口种子及扰动规则；评估仅首帧初始化后完整递推，不允许失跟后GT重置。每轮验证，周期checkpoint与最后58/59/60，正式final60/late3，禁止各组挑best。

记录overall、moving endpoints、ever-moving tracks、低运动/启停边界、>5m/>10m、失跟段长度与删失、原始有目标而main/union无目标、备用实际采样目标及正确trusted写入关联。最后一项是关联证据，不自动证明单点因果贡献。M loss降低不等于跟踪涨分；mini单seed和late3不能证明稳健增益。历史v38 LR/精度不同，不作本次严格对照。

[SeqTrack3D](https://arxiv.org/html/2402.16249v1)支持点云和历史框序列联合建模，但不是本方案未来MTP的直接验证。[Fast and Furious](https://arxiv.org/html/2012.12395v1)提供共享表示与未来监督动机，但任务为检测/跟踪/预测联合。FA使用[官方2.5.8实现](https://github.com/Dao-AILab/flash-attention/blob/v2.5.8/README.md)；缓存解释遵循[PyTorch 2.0.1](https://github.com/pytorch/pytorch/blob/v2.0.1/docs/source/notes/cuda.rst#memory-management)。双ROI、固定配额及1.5秒策略均为本项目待验证设计，不宣称文献已证明涨分。
