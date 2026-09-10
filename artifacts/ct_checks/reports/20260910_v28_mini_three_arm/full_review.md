# v28 Full seed42 完整训练结果核查（2026-09-10）

Full 已完成60轮、75,720次观测更新，CUDA 严格确定性没有再阻断此运行。**当前验证结果仍是未校准的 observation 回退，不能作为 Full 模块涨分或无增益的结论。** 同 seed 的 Full/B0 观测参数、BN、Adam、全程观测 loss 完全一致，插件未污染这两组的 B0 训练路径；但是最终 B0 的45.200/47.243还没有达到注册恢复门槛49.986/58.962，暂不满足进入后续大规模正式实验的条件。

运行根目录：`output/20260909-003318-28_full-mini_car_seed42_60ep_bs16/`。下文简称“Full根目录”；B0对照为同日期 `28_b0-mini_car_seed42_60ep_bs16/`。机器可读详情见同目录 `full_review.json`。

## 完成状态与模块优化

- `train.log` 有 `` `Trainer.fit` stopped: `max_epochs=60` reached. ``；`formal_checkpoints/epoch=060.ckpt` 的 `epoch=59`、`global_step=75720`、`ct_epoch_boundary_complete=true`。
- 已保存正式 epoch58/59/60；StepLR `last_epoch=60`，所有组初始学习率1e-4，20/40/60轮结束后衰减，最终已切至下一轮学习率1e-7。
- `ct_module_audit`：B0更新75,720次，B1/B2/B3各18,000个机制事务，参数组为单个Adam的b0/b1/b2/b3，`active_frozen_parameters=[]`。每轮1,262次观测更新、300次机制事务，机制流4,777预测端点、268条非单帧tracklet。

| 模块 | 首个非零loss日志step（从0计） | 全运行非零loss条数 | Adam末端逐参数step | 最大梯度范数 |
| --- | ---: | ---: | --- | ---: |
| B1 | 8 | 17,940 | 20个张量均18,000 | 15.7063 |
| B2 | 4 | 18,000 | 45个张量18,000；21个张量17,177 | 71.1955 |
| B3 | 8 | 17,177 | 12个张量均18,000 | 2.9837 |

首个非零目标都在epoch0；证据支持没有先冻结阶段。18,000是机制事务/更新计数，并不代表每个模块每次都有非零梯度。不可用历史或无合法支持的事务可以是零loss；B2部分分支只有在17,177个可用事务中建立梯度。来源：checkpoint `ct_module_audit` / `optimizer_states`，以及 `lightning_logs/version_0/loss_loss_b{1,2,3}_transaction/` TensorBoard事件。

环境记录：Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2、CUDA11.8、cuDNN8700、A40、驱动535.247.01。deterministic=true、warn_only=false、TF32/benchmark=false、CUBLAS=`:4096:8`、Adam foreach/fused=false。来源：checkpoint `ct_v28_runtime_environment`、`run_provenance.json`。

## 官方 mini_val 曲线

| 完成轮数 | Success | Precision | 接受动作数 |
| ---: | ---: | ---: | ---: |
| 5 | 29.658 | 31.852 | 0 |
| 10 | 39.270 | 45.909 | 0 |
| 15 | 26.005 | 24.056 | 0 |
| 20 | 49.533 | 65.557 | 0 |
| 25 | 52.043 | 60.449 | 0 |
| 30 | 44.170 | 45.606 | 0 |
| 35 | 41.863 | 43.724 | 0 |
| 40 | 41.925 | 41.009 | 0 |
| 45 | 44.428 | 46.133 | 0 |
| 50 | 45.209 | 47.007 | 0 |
| 55 | 45.496 | 47.248 | 0 |
| 60 | 45.200 | 47.243 | 0 |

来源：`lightning_logs/version_0/dev_diagnostics/epoch_XX_summary.json`；TensorBoard S/P与上述数值仅存在记录精度差异。每5轮验证，没有epoch58/59评价文件，所以不能把50/55/60称为late-3；58/59/60的算术平均目前缺失。第25轮分数高不替代固定final/late-3验收。

`run_provenance.json` 记录训练为官方8个mini_train场景、274tracklet、5,051帧，验证为官方mini_val的scene-0103/scene-0916、106tracklet、2,285帧。诊断路径和CSV虽然写dev，但不是历史v27内部dev划分。CSV中scene_id全部unknown，导致summary的scenes=1是元数据缺失；不可据此声称只评价一个场景，亦不能直接拿此CSV做按场景分析。

## Full 与 B0 seed42 的观测路径一致性

读两组epoch60 checkpoint及TensorBoard事件，获得以下逐位比较：

- 全部320个B0模型state张量（包括BN）相同；236个B0参数的最终Adam状态全部相同。
- 63个B0状态hash（初始化、step1、step100、epoch1—60）全部相同；前100个输入fingerprint与初始化/step1/100的Adam hash相同。
- 全部75,720步观测loss值和step一致。

注意 Full 的 `loss_loss_b0_transaction` 有93,720条：额外18,000条是机制shadow记录。比较时按每个global_step第一条选出观测事件；不能直接把整个Full目录该tag的平均当成B0观测均值。

这些证据支持插件在这次运行中未改变观测训练，且不是仅“最终分数接近”。但没有完整同卡B0-A/B重复的逐层激活/梯度工程审计，不扩展为任意GPU/环境下都已验收确定性。

## B3未校准，以及B2新增测量不足

`run_provenance.json` 的 `ct_action_calibration_path=null`、`ct_require_action_calibration=true`。epoch60的2,179个预测端点 `b3_calibrated=0`、`router_applied_gate=0`；12次验证的动作数全为0。因此Full当前沿B0回退输出，不能据此判断完成校准后的Full有没有收益。

epoch60的证据供给存在明确瓶颈（数量按point-frame累计）：

| 阶段/机会 | 目标点数量 | 补充 |
| --- | ---: | --- |
| 当前整帧GT内点 | 140,546 | 1,755预测帧目标可见 |
| B0整个raw crop内 | 117,831 | 新证据必须减去这部分原始ID |
| B0 raw crop外可获取目标点 | 22,715 | 192帧存在机会 |
| support原始区域内 | 83,891 | 大部分与B0 raw crop重合 |
| support减B0 raw crop后的新增点 | **4** | 仅2帧；新增区域共15,790点 |
| pool/prepool/selected/共识主模态保留 | **4** | 本次没有在这些后续阶段丢失目标点 |

新增目标点获取率为4/22,715=**0.0176%**，机会帧覆盖2/192=**1.0417%**。这将当前瓶颈定位在新增测量的获取阶段，不能仅靠B3阈值拟合或B2排序宣称解决；仍需区分几何范围、B1预测端点/递归误差和训练验证分布差异的作用。B1物理位移平均误差0.212m，CV为0.254m；递归endpoint平均误差分别2.770m/2.816m，不能将前者当作后者的恢复证据。

训练epoch60 acquisition supply有210个目标正例候选行、pool目标7,325点、sampled目标5,095点；其行召回1.0、点保留率69.556%。因此“训练目标正例存在/模块在更新”与“官方mini_val新增目标证据极少”可以同时成立。来源：`acquisition_supply/epoch_60.json`。

## 动作质量的当前边界

epoch60有567个结构合法动作候选，占2,179预测帧的26.02%（含初始化的全2,285帧口径为24.814%）。同一当前状态上的动作诊断：

| 动作阶段 | helpful帧 | harmful帧 | 候选平均utility gain |
| --- | ---: | ---: | ---: |
| raw | 21 | 491 | -0.36944 |
| bounded | 157 | 321 | -0.07434 |
| accepted | 0 | 0 | 不适用 |

bounded降低伤害，但剩余有害候选仍多于有益候选。以上是固定当前状态的一步反事实，不是Full−B3闭环总分；也不能把0动作解释成B3筛选效果好，因为其校准artifact缺失。

建议继续完成当前结果的final/late-3评价、B0同协议reference与观测恢复定位，并开展B1/B2获取几何的有界诊断。B0 seed42 final未达到恢复门槛，且Full还未校准，不应据此放行完整nuScenes多臂大规模论文长跑。
