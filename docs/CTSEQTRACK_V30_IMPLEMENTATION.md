# CT-SeqTrack v30 实现说明

本版实现“时间条件获取—多模式新增证据—实际动作收益选择”。所有三臂共同升级B0，
插件增益须相对这个共同B0测量。本地实现/测试不代表已在CUDA训练通过或取得分数提升。
当前排程以[正式协议](EXPERIMENT_PROTOCOL.md)顶部为准。

## 数据通路与所有权

```text
原始帧/唯一ID/真实时间 + 已接受历史框
  ├─ 实际B0 raw crop（一次，缓存点和ID）→ 4×1024有效槽 → B0观测框
  └─ 历史框/时间 + crop的4项统计 → B1 21维上下文
       → 物理运动终点 + 实际crop投影包络外获取带
       → rawIDs(support) − rawIDs(B0 crop) → 768 → 256
       → B2三模式（中心/协方差/成员/64维特征/质量/身份）
       → detach → B3六个模式×幅度动作 + observation
       → 共享host提交唯一最终框 → 后续机制流历史
```

GT只进入监督、合法窗口初始化和独立诊断；不改变crop上下文、模式排序或动作网络输入。
共享B0训练流和机制accepted状态流独立。B0只按自己的损失更新；机制前向隔离B0 BN/RNG；
B3损失不能回传B0/B1/B2。启用的网络从首个合法事务训练，无冻结或warm start。

## B0 共同基线

`utils/masked_observation.py`复用原BN参数和running-stat键。有效点少于2时读取已有统计而不更新；
全有效路径沿用普通BN。无效槽先屏蔽，池化以负无穷屏蔽后将全空结果置零；
固定1024→128分桶max同时输出token mask，进入局部、全局及交叉注意力。
Seg/Mini/FeaturePointNet均接受mask，真实1/2点重复补槽保留。

9月15日晚显存修复去除Conv→BN和ReLU周围重复mask，保留每段入口及BN/最终输出的无效隔离。
用户接受约+5GB显存后，`ct_b0_masked_bn_recompute:false`作为默认，直接计算有效BN。
显式true才只保存输入、mask/affine并在反向重算原公式；两者running均仅前向更新一次。
可选重算不支持二阶导数，当前正式Adam仅用一阶导。三臂共享此执行设置并绑定恢复身份。
详见[预算与文献依据](CTSEQTRACK_V30_MEMORY_TRADEOFF.md)及[故障证据](CTSEQTRACK_V30_CUDA_MEMORY_FIX.md)。

Seg保持三维class-axis log_softmax→二维NLL，以有效标签对应类别权重和为分母；BC按有效点归约；
reference只按存在历史帧监督。空测量项返回可反传零值，当前框和物理运动监督继续训练。
全空点云且历史框合法时保留历史query预测，v30不执行旧部署专有的强制零位移覆盖。

动静目标为历史到当前真实时间跨度上的速度>0.3m/s；coarse motion使用moving概率。
四候选总体不变，candidate0为teacher；两个候选始终短窗口，另一个以稳定哈希各半选择短/长窗口，
形成25/62.5/12.5请求比例；短轨迹按可用历史截短。当前B0无梯度roll-in最多3或8步，端点一次梯度前向。

## B1 获取带

`utils/acquisition_v30.py`构建实际crop在运动轴上的投影与预测终点初始尺寸框的共同包络；
在其外扩band，而不是仅在物体框外加margin。Z保留v29垂直包络。
默认min[.25,.25]、max[4,3]、init[.75,.5]m；紧范围消融max[2,1.5]m。
旧17维获取上下文追加归一化raw点数、empty及两个log crop halfsize，main/aux复用同一当前crop事实。
首个query没有合法运动转换时也允许最小获取带，不伪造学习有效性。

实际、最大可达及9×9枚举复用固定core/轴和同一band族；满足90%可达新增目标的候选中优先最少背景。
q=.9 pinball、外层权重.05；需求与无需求组各.5，缺组按存在组归一化，既有年龄归约保留。
有全局目标但最大范围内无新增目标的样本target_valid=0，不混入收缩负例。
sigma独立保留，不控制band，也不作为B2唯一空间尺度。

## B2 三个证据模式

`models/ct_v2/evidence_v30.py`用固定形状张量计算1m targetness加权局部支持量，
按证据选seed，.75m空间去重，K=3。启发分数使用加权质量与紧致性，移除额外未加权数量比例。
三模式以`EvidenceModeSet`传递：centers[B,3,2]、covariance[B,3,2,2]、valid[B,3]、
member_mask[B,3,256]、features[B,3,64]、quality[B,3]，以及unique数、mass、identity。
成员raw IDs去重；模式编号由证据决定，不用GT或B3收益重排。固定归约无浮点scatter累加。

空间特征按support半尺寸归一化，vote/距离用初始物体尺度，log sigma保留独立质量特征。
保留128/96/32点配额、768→256预算、36 memory tokens，不扩展长期记忆。
删除整行raw center回归；只保留真正前景vote。质量目标为unique前景纯度×中心距离高斯因子，
尺度是初始平面半对角线；中心/成员/标签detach，soft BCE训练B2特征和质量头。

总损失为`.25 relation + .2 targetness + vote_fg + .1 basepresence + .1 extpresence + .1 modequality`。

## B3 模式与幅度联合选择

`models/ct_v2/action_v30.py`生成3模式×{.5,1}动作，clip半径`min(.5+.5Δt,2)`；
Z/yaw/尺寸继承B0。共享网络每动作输出ΔS、ΔP、help/harm、归一化距离改善和IoU改善。
执行q=(ΔS+ΔP)/2；最大q达不到阈值则observation，同分优先小位移，再按固定模式号。
模式64维特征、质量与实际动作特征输入全部detach；最终只提交一个框。

`utils/v30_training.py`在同一状态/同一B0/同一GT下监督全部合法动作，按行先平均动作，再平均合法行。
总B3损失为`.2 × [LS+LP+.1help+.1harm+.1(distance+IoU)]`，不因6动作放大权重，
几何辅助目标与S/P目标保持独立。训练机制使用轨迹级稳定哈希2/1/1/4个桶分配never/首模式全幅/探索/q最大；
探索动作使用独立帧hash，shared B0流不使用该accepted历史。

`utils/action_calibration_v30.py`从never轨迹每行max_action_q获取数值阈值，最多10个真实拟合闭环；
锁定dev只做诊断。策略与checkpoint、resolved config、数据manifest、动作集合及源码绑定；
缺失/错误策略回退observation。`utils/v30_action_output.py`统一训练/部署所选动作与全部标量诊断。

## 数据、效率与报告

nuScenes真实LiDAR时间及sensor→ego→global；mini/full使用同一适配器。KITTI使用完整R_rect @ Tr_velo_cam，
同步转换点与框，并在水平化传感器相对坐标中统一yaw；原始frame×.1s保留跳帧跨度，不假设OXTS存在。
所有角色关闭当前GT预裁剪，缺测表示零真实点。sequence manifest、坐标模式、frame stride进入身份。

辅助B1历史读取metadata；H3抽样先于未来云读取；定位轨迹使用累计边界二分；
同进程复用不可变nuScenes元数据，整云LRU每进程/worker256MiB，保持点序/ID/dtype及独立副本。
workers4、不启用persistent workers。训练漏斗每事务累计，额外诊断写出可抽样；计数最终批量传CPU。
没有以减少IO调用次数冒充实测加速比。

训练`ct_v30_full_funnel`与评测`v27_endpoint_summary.json`内部v30 schema记录逐层点/事件/条件分母，
区分已测零值与未知，并按稀疏、物理速度、递归年龄、连续失跟分组。
评测最大可达计数直接使用raw-ID sidecar，不需要运行81种监督枚举。
mini晋级工具检查同一完整官方测试人口与协议，固定final60及late-3，拒绝用训练验证或校准集替代。

## 文件与验证

新增9个主臂配置（mini/full/KITTI各3个）及6项mini消融、1项紧范围消融。
局部历史兼容路径继续接受v24–v29配置；新策略/模型不得冒用旧身份。
本地测试覆盖真实host前向/反向/Adam、跨臂B0更新/BN/RNG一致、GT不进入模型输入、
空/1/2点梯度、实际获取/标签一致、两个共识反例、正确次模式选择、六动作世界坐标标签、
合法H3、接受状态与epoch边界恢复、KITTI标定及完整尾批。
结果与未执行的CUDA/正式实验见[验证记录](../artifacts/ct_checks/20260915_v30_implementation/REPORT.md)。
