# B0 的 Git 历史回查：哪些变化需要保留，哪些需要修正

审计日期：2026-09-08。只读比较 `2bdacd8`、`62e1f90`、实际高分运行提交 `049de82`、`b8222bb`、`b445ecd`、`a43a8ee`、`8b8b8d9`。未修改历史仓库、模型、配置或 output，未读取 TrajTrack。历史分数及实际配置由本次 [historical_b0_recheck.json](historical_b0_recheck.json) 提供；源码结论使用 `git show REV:PATH` / `git diff`，不以提交标题证明性能。

**不建议整体退回“恢复 B0”提交。** 历史中确实有能达到较高分数的 safe-auto B0，但它已经使用 canonical 0.5、首帧尺寸和本次发现的 BC 双计。v27 相对这条高分路径新增了真实的输入语义变化；同时，早在 v25 两次运行之间已经存在首次 Adam 更新分叉。因此“修一个旧 bug 就解释所有下降”和“撤回全部 v27 就恢复高分”均没有证据。

## 1. 历史实际分数对推断的限制

| 运行提交与路径 | 最终 S / P | 实际评价与训练集合 | 能说明什么 |
|---|---:|---|---|
| `2bdacd8`，20260822 的 24_b0 restore | 31.4147 / 31.1028 | mini_val 106 轨迹、2285 帧；训练 274 轨迹、5051 帧 | 提交标题含“恢复高分语义”，实际该运行并未恢复至历史高分 |
| `049de82`，20260824 的 25_b0 retryfix | 50.6904 / 59.2801 | 同上述 mini_train / mini_val | safe-auto、canonical 0.5、首帧尺寸可以与较高分数同时存在 |
| `b8222bb`，后续 v25 B0 | 29.870 / 30.398 | 由主报告汇总对应运行 | 首批输入/初始化相同而首个更新已分叉；不能用相同的损失设置单独解释下降 |
| `8b8b8d9`，本次 27_b0 | 17.9517 / 15.5530 | dev 12 轨迹、321 帧；训练 228 轨迹、4231 帧 | 与历史 mini_val 不是同一评价分母，不能直接作性能差值 |

高分 `049de82` 的 B0 **确实执行 BC 双计**，不是源码中存在但运行没走到：

1. `models/seqtrack3d.py:5274` 将 `b0_transaction_loss` 与 `loss_total` 绑定到同一 Tensor。
2. `6853–6855` 先 `loss_total += BC`，再 `b0_transaction_loss = b0_transaction_loss + BC`。
3. ordinary observation 的插件路由关闭，payload 无 `ct_b0_auxiliary_only`，因此没有提前返回或 B2 loss 打断 alias。
4. `8211` 调用 candidate-weighted helper；`7961` 对每个 view 取 `loss_b0_transaction`，`7966` 以加权结果覆盖实际返回的 `loss_total`。

因此高分 v25 与这次 CT 五臂同为四 view 各含 2×BC。当前实模 CPU 的数值/梯度复现见 [b0_training_distribution.md](b0_training_distribution.md)。**BC 双计是必须修正的目标合同错误，但不是历史 v25 高低分差的充分解释。** 同理，canonical 0.5、safe first-frame size、继承的 frame/channel reshape 错误均已存在于高分版本，不能单独作为后续跳水的新增原因。

## 2. Git 变化、实际影响与处理建议

| 提交或区间 | 可核对的真实改变 | 对 B0/比较的影响 | 建议 |
|---|---|---|---|
| `2bdacd8` | observation 与 mechanism 双流；B0 观测路由关闭插件；mechanism 独立模块更新、BN 统计隔离及 RNG 恢复 | 将 B0 更新与新增模块解耦，避免机制遍历改变 B0 的输入预算和 BN 统计 | **保留所有权及隔离合同**，不能为追分重新混合 GT reseed 或隐式插件梯度 |
| `2bdacd8` | B0 候选从 shared-SE2 退回 independent；配置权重设为四个 0.25 | 恢复旧独立小扰动分布。该版 observation 使用普通 batch 归约，不等同后来每 view 单独归约；手动 B0 optimizer 读取 transaction | **保留独立 observation 对照**。不能据 .25 配置和提交名宣称严格恢复原 loss |
| `2bdacd8` | 删除 recursive GT reseed；保留 `observation_safe_bbox_size=true` | 递归只用预测历史，尺寸从首帧获取，减少当前 GT 信息进入输入 | **保留**。高分 v25 已有安全尺寸，没必要恢复当前 GT 尺寸 |
| `2bdacd8` | `ct_b0_steps_per_epoch=1262`；训练期直接评估官方 mini_val | 对应旧 5051 帧训练集的固定预算与旧评价分母 | **撤回硬编码**，保留 v27 实际 dataset 长度预算。历史成绩需在同集合重评后比较 |
| `62e1f90`，由 `049de82` 实际运行 | unified-auto：一个 Adam、命名参数组；观测 stateless candidate-balanced batch；四 view 单独 loss 后按 0.5/1⁄6/1⁄6/1⁄6 归约 | GT canonical 在 loss 中占 50%；分支内归约与原 batch mean 不完全等价；B0 有梯度训练仍是 GT/小独立扰动历史 | **保留可审计单 Adam 和独立随机流**。canonical 权重作为训练设计做受控对照，不应静默改动注册协议；高分 v25 说明它并非必然低分 |
| `25586b27` 引入，贯穿上述各版本 | transaction Tensor alias 造成 BC 重复 | CT 目标与宣称公式/reference 不一致。`2bdacd8` 手动 optimizer 同样读 transaction，故也受影响 | **修正**独立构造模块 loss，禁止依赖原地累加的 alias；不能靠 `bc_weight=0.5` 掩盖 |
| `62e1f90 → b445ecd` | v26 主要增加获取/证据设计、诊断和最后三轮保存；B0 仍沿用 v25 observation 主路径 | 首帧尺寸、canonical 权重、safe-auto 均未在 v27 首次出现 | **保留公共 B0 路径的稳定部分**。插件自身新增不等于 B0 必然被训练梯度污染 |
| `b445ecd → a43a8ee` | mechanism B0 从 `no_grad()` + 外层 BN 隔离改为临时整模 `eval()+no_grad()`，逐模块恢复 training flag | 修正 Dropout 仍处于训练态的问题，使 mechanism observation 与推理/H3 状态生成一致；不冻结有梯度 observation 训练 | **保留**。退回仅 BN 隔离会重新引入状态生成差异 |
| `b445ecd → a43a8ee` | `build_input_dict` 改用 `build_v27_eval_input` → 通用 sampler，eval 永远 candidate0 | 绕过旧 eval 的 `frame_id!=1` 软 prior：预测历史从 0.2/0.8 变成 GT 风格 0/1。统一函数但历史来源语义不统一 | **修正**：prior 根据真实初始化/预测/无效历史来源生成；保留公共构造器，撤回“candidate0 等同 GT 历史”的推断 |
| `b445ecd → a43a8ee` | point ID、N=1/2 真实点保留、padding 有效性、seg/BC point loss mask、全 endpoint 计数 | 保留真实稀疏证据与恢复机会，纠正 XYZ 判空和浮点去重问题；同时改变稀疏训练样本 | **保留基础合同**。继续把 padding 对 feature/BN/pooling 的影响处理一致，不能重回 N≤2 一律清零 |
| `b445ecd → a43a8ee` | sampler 对 v27 跳过 all-GT-history-empty 的断言；当前为空时 eval 返回 reference，训练仍回归网络 box | observation 训练可见性分布改变；空 current 的可部署动作与 box 监督条件不同 | **修正/对照验证**：计数后明确不可观测行的训练损失，不应一边部署 hold、一边把永不采用的空 current box 动作作为同等监督。无需删掉评测 endpoint 或阻断合法 extension |
| `b445ecd → a43a8ee` | mini 参数训练落实 6 场景；训练期 validation role 改 dev。full 参数训练 350 与内部17/18、官方150分开 | 训练帧、1057 steps/epoch 与评价分母均变化；YAML 的 `val_split: mini_val` 不再代表训练日志来自 mini_val | **保留已注册场景合同和显式 role**；主报告区分 dev/mini_val。比较历史需补同集合 reference，不能使用旧 1262 预算冒充同协议 |
| `b445ecd → a43a8ee` | Checkpoint 回调顺序及 DataLoader generator resume 恢复；新 manifest/schema 身份 | 修复 epoch-boundary 保存与恢复的真实时序，防止恢复后 worker/validation RNG 改变 | **保留**。属于可恢复性正确性，不是得分开关 |
| `a43a8ee → 8b8b8d9` | sampler 9行修改，B1 所有 slot 预置四个 margin 诊断计数字段；增加混合 slot 合同测试 | 修复早期/成熟 slot 合批缺键报错；未改 B0 forward/loss/eval/优化器行为 | **保留**，没有证据把这次“解决报错”提交作为 B0 分数下降原因 |

## 3. 不应继续沿用的数值归因

旧 `20260827_b822_baseline_path_divergence` 报告把 PointNet2 自定义算子的 `atomicAdd` 列为优先归因。代码所在文件导入了 `PointnetSAModule`，但 B0 实际构造的是 `SegPointNet`、`MiniPointNet`、`FeaturePointNet` 和 Transformer；这三个网络使用 Conv1d、BatchNorm、ReLU、池化等，`FeaturePointNet.forward` 不调用 PointNet2 SA/grouping。**“仓库里有该 CUDA 扩展”不等于“本次 B0 backward 经过它”。** 不能未经 profiler/调用链证明便继续引用具体 `atomicAdd` 算子。

目前能证明的是：本次五臂和历史 v25 对照都可在初始化/输入指纹一致时出现首个参数更新 hash 分叉；本次首个 mechanism 到 global step 4 才出现，而 step1 已分叉。CPU 真实生产 observation 路由下五臂 loss/梯度目标一致。因而仍需在相同 GPU 顺序执行同臂短跑重复，再对照跨臂，保存首个 forward、各层 gradient、Adam 输入/输出及 BN 状态。需要定位实际数值分叉位置，而不能把“存在非确定性”当作已经解释二十多个点的分差。

## 4. 从历史中可以形成的最小行动顺序

1. **先保存评估证据**：当前和旧高分 checkpoint 优先在共同的官方 mini_val 轨迹各跑原输入与恢复软 prior 的敏感性评价，结果只用于定位输入变化；不把它与修正后 scratch 实验混作一组。旧8场景训练包含本次dev，旧权重在dev的结果不能当作独立泛化比较。
2. **修正明确语义错误**：loss alias、预测历史 prior、frame/channel 及 B2 点特征对齐。BC/布局改动会改变学习问题，正式结果必须从头训练；旧 checkpoint 只作诊断。
3. **保持必要公共修复**：ID、稀疏点、全 endpoint、首帧尺寸、无 GT reseed、eval 模式隔离、有效时间、epoch-boundary resume、350/150。不要把工程正确性修复整批撤回。
4. **只给尚无证据的设计做消融**：canonical 0.5 对 .25、空观测监督方式、B0 预测历史覆盖/相关扰动。先在共享 B0 和匹配 reference 上确定影响，再复跑插件臂；不要将可选设计和已确认 bug 合并成一个不可解释的“大回滚”。

本表判断的是代码语义及必要性，未承诺修复后的涨分幅度。历史高分是路径可行的证据；最终是否恢复、插件是否带来收益，仍须以同数据、同预算、同评价器的实际闭环输出验证。
