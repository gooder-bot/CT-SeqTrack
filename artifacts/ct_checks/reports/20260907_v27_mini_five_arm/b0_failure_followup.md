# B0 失跟专项复审：首次失效、输入语义和时序主干

日期：2026-09-07。范围：只读 v27 B0 第60轮全部321个 dev endpoint及12次验证、原 SeqTrack 只读源码；不使用 TrajTrack。未修改模型或 output。下述复现使用真实 B0、FeaturePointNet、Transformer 和 v27 sampler，点云为已有 CPU 合同测试的合成序列。

## 1. 判断需要更新

低分不仅是不同评测集合不能直接比较。当前 dev 中存在严重且真实的递归失跟：309个预测帧有260帧误差>2m，137帧>10m；11/12条轨迹曾越过2m，10条在首次越界后再未恢复到2m以内。这里的“崩溃”是跟踪轨迹失效，不是 loss/权重 NaN 或训练进程崩溃。

本轮新增确认三个代码/合同问题，其中前两个来自原主干，而第三个由 v27 合并训练/推理输入构造后引入。继承的问题不能独立证明“相对原版下降了多少”，但确实损坏既定数据语义，且会妨碍新增时空融合模块。

## 2. 新确认：按帧拆分实际混合了帧轴与通道轴

位置：`models/seqtrack3d.py:3752`（搜索 `solo_x = x.reshape`）。

`x` 的逻辑布局是 `[B,C,L*N]`，其中点按过去3帧和当前帧依次拼接。当前直接：

```python
solo_x = x.reshape(B * L, -1, chunk_size)
```

这不等于把每一帧独立交给 FeaturePointNet。正确语义需要先显式分出帧轴，再把帧轴移到通道轴前：

```python
solo_x = x.reshape(B, C, L, chunk_size)
solo_x = solo_x.permute(0, 2, 1, 3).reshape(B * L, C, chunk_size)
```

实际 `box_aware=True` 时 C=14，L=4。当前被当成“当前帧”的伪帧，14个输入通道全部取自原输入的第10–13通道，即历史/当前 candidate box-distance 中的一部分，完全没有本应属于当前帧的 XYZ 和时间通道。真正的当前 XYZ 被分散到其他伪帧中。**这不是整个 B0 完全看不到当前点云**：SegPointNet/MiniPointNet 和全局 attention 仍可能通过其他路径使用它；错误在于帧级特征语义与其标注身份不一致。

后续 `point_aligned_feature.reshape(B,L,N,64)` 又把伪帧特征标记为原帧/原点，对 B2 的 `b0_point_aligned_features` 和 memory 点身份构成新增污染。ID传播本身正确不等于特征与ID真正对齐。

原冻结 SeqTrack 的 `models/seqtrack3d.py:141` 也有同样写法，因此属于继承错误，不能把该发现伪装为v27唯一退化来源。

证据：

- `analyze_b0_failure_followup.py` 用互不相同的 batch/channel/frame/point 编码最小张量复现，336个元素中324个不符合正确帧拆分。
- `reproduce_b0_input_semantics.py` 对真实 model.feature_pointnet 注册 pre-hook，收到的 `[8,14,1024]` 输入与正确拆分不同，最大绝对差12.2103。
- 只纠正这行虽然不改变参数shape，却改变模型学习的输入语义。新正式实验必须scratch；不能将旧checkpoint换上新拆分后的分数当作恢复/涨分证据。保留原冻结 SeqTrack reference，并将纠正后的公共B0显式标记。

## 3. 新确认：历史有效mask在Transformer中没有被消费

`models/attn/Models.py:125` 接收 `valid_mask`，但函数内部不使用它。局部encoder、全局encoder、decoder均没有传入历史有效性mask，decoder调用仍是 `self.decoder(trg_seq_, None, enc_output, None)`。

真实Transformer在eval模式、相同token输入下，将 `[1,1,1]` 换为冷启动 `[1,0,0]` 后输出逐位相同。这证明当前mask没有实现其接口语义。

同时，`b0_valid_mask`/`b0_unique_mask`目前进入点级损失和质量统计，但没有成为 SegPointNet/FeaturePointNet 池化或 attention 的mask。空padding虽然不承担点级监督，仍可以通过时间、prior、box-distance、卷积/BN偏置参与特征，不能将“loss排除padding”等同于“前向不受padding影响”。这是现存架构限制；大幅修池化/归一化需要作为新B0版本测试，不能静默改变原reference。

## 4. v27 特有：预测历史仍使用GT式0/1 prior

训练 observation 的 canonical candidate0 使用真实历史框并设置prior为0/1；扰动candidate1/2/3将prior设置为0.2/0.8，见 `datasets/sampler.py:1615`。原 SeqTrack 和旧CT推理在 `frame_id != 1` 后将预测历史prior软为0.2/0.8，见 `models/base_model.py:3083`。

但 v27推理走 `utils/v27_input.py`，强制 `candidate_id=0`，再调用训练sampler。sampler仍只用candidate_id判断prior，因此第8帧的历史全部来自预测时，prior依然只有0/1。真实sampler复现已确认这一点。

这意味着训练里与“GT式精确历史”绑定的置信语义，在推理被用于含递归误差的历史；错误框可被当成高置信身份线索。不能根据代码注释“boxcloud存在时prior值不重要”忽略它：该数值实际进入SegPointNet/FeaturePointNet输入，网络有机会依赖。

建议按历史框来源/预测质量构造prior，而不是按candidate_id替代来源。首帧已知框与其余预测框区分；与回放/训练扰动同步修改。影响量仍需在同checkpoint、同endpoint输入和完整闭环上验证，不能直接声称解释全部降分。

## 5. 首次失效往往发生在目标仍位于base时

下面是11条曾超过2m的轨迹**第一次**越界；目标数为实际点ID诊断，非重复采样槽。

| 轨迹 | 首次>2m帧 | 误差m | base点数 | base目标 / 全局目标 | 之后恢复≤2m帧数 |
|---|---:|---:|---:|---:|---:|
| 0 | 8 | 2.60 | 9 | 8 / 8 | 0 |
| 1 | 2 | 5.51 | 7 | 2 / 2 | 0 |
| 2 | 3 | 2.15 | 114 | 13 / 13 | 2 |
| 3 | 1 | 3.19 | 1 | 0 / 0 | 0 |
| 4 | 2 | 6.28 | 37 | 4 / 4 | 0 |
| 5 | 2 | 2.27 | 1 | 1 / 1 | 0 |
| 6 | 5 | 3.25 | 8 | 8 / 8 | 0 |
| 7 | 1 | 2.64 | 53 | 8 / 8 | 0 |
| 8 | 1 | 3.98 | 0 | 0 / 0 | 0 |
| 10 | 1 | 4.23 | 1 | 0 / 0 | 0 |
| 11 | 14 | 2.62 | 67 | 13 / 13 | 0 |

**8/11条轨迹的首次越界帧，所有全局可见目标点都已经在base中。** 另外3条全局没有目标点。因此，不能把最初的失效统一归因于B1范围太小或B2没有新增证据。

轨迹7是最清楚的链式例子：第1帧base有全部8个目标点，B0仍偏2.64m；第2帧base有全部25个目标点，偏差增至9.24m；第3帧才变为base无目标。顺序是早期观测/身份回归失误→偏移锚点→目标离开裁剪→错误状态继续递归，而不是开始就裁不到目标。

轨迹8则属于不可观测/空输入链：开始3帧全云无目标，base为空，v27输出reference，误差约3.98→7.96→11.94m。之后目标点重新可见时，裁剪仍停在错误位置。时空融合可以在有可靠历史运动信息时缓解，不能从只有首帧框、当前全局无目标的序列中保证恢复未知速度。

## 6. 训练loss下降但递归仍坏的机制原因

这些是源代码支持的学习/部署差异，尚需实验划分贡献：

- B0训练来自GT历史附近的独立扰动，平移只在±0.3m、yaw约±1.5°；50%权重落在零扰动canonical。推理可能面对2–10m以上误差，两者远非同分布。
- mechanism stream虽使用预测递归历史，但 `models/seqtrack3d.py:8657` 在临时eval+no_grad内读取B0，仅训练插件；B0不会因为见过mechanism难例就学到恢复。这个设计保证梯度隔离，同时保留了B0的部署误差缺口，不应被描述为B0已接受预测历史训练。
- v27保留以前会被全空历史assert跳过的轨迹，但回归目标仍无条件要求输出GT位置。无有效当前/历史证据的输入也承担位置损失，目标可能不可辨识。不要重新删掉难例；应拆分当前可观测、历史可支持、不可观测状态和各自监督。
- v27推理在当前raw crop为空时强制局部输出为0（reference），训练模式没有同样覆盖，且仍算回归损失。这是原计划明确选择的部署策略，而不是随机报错；117/309帧走空base，其恢复能力有限。历史视觉/运动可用时应考虑显式的无当前观测预测动作，和“观察到目标后纠正”分开。
- `argmax`分割硬筛选和motion-class硬门继承原主干，在少量点时可能让轻微logit改变变成不连续动作；已存在的Transformer后续可修正，但目前主干帧语义也错了。没有逐帧logit，不能断言每次跳变都由argmax造成。

第5轮与第60轮对比：>10m帧数64→137，base空帧73→117，base无目标201→231；数据全局可见帧恒为230。随着学习变化，递归落入难恢复状态更频繁，与训练loss持续下降并不矛盾。但单场景mini不能分离过拟合、目标函数/置信语义、GPU数值分叉各自占比。

## 7. 对时空融合位置的直接含义

应先修帧拆分和mask/来源语义，然后考虑两层用途：

1. 针对B0首次失效，应在当前点特征进入最终观测回归前，用可靠历史目标token与当前base点做带相对位姿/时间的融合，或在独立插件中同时读取base和extension做身份条件候选。只在B2 extension上继续加层，无法覆盖当前base已有全部目标、没有novel点的首次失败。
2. B2继续extension-only时，历史融合适合用于局部几何后、relation选点前，以及候选vote前的身份关联，避免把近期错误框的上下文当目标。输入点坐标和特征先恢复真实对齐；否则更加复杂的时空模块只会融合错配数据。

若修改B0或允许B2使用base，需登记新模型/新消融；原B0及SeqTrack reference保留。所有启用模块从epoch0共同学习，不冻结。已有checkpoint可做输入语义/动作敏感性诊断；不能替代改结构后scratch正式训练。

## 8. 复核产物

- `b0_failure_followup.json`：309帧切片、逐轨迹首次失败、12轮分层与最小布局映射。
- `analyze_b0_failure_followup.py`：只读数据复算脚本。
- `reproduce_b0_input_semantics.py`、`b0_input_semantics_reproduction.json`：真实模块/真实sampler复现。
- 验证：`python -m pytest -q artifacts/ct_checks/reports/20260907_v27_mini_five_arm/reproduce_b0_input_semantics.py` → **1 passed in 2.81s**。该测试断言当前缺陷存在，不是修复后的验收测试；源码未修改。
