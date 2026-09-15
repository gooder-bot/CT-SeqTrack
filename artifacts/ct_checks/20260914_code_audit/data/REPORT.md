# v29 数据通路与 mini/full 适配审计（2026-09-14）

生产代码只读；本目录仅有诊断脚本、输出和报告。不读取 nuScenes 实际点云、冻结参考或大 checkpoint。

## 已验证的性能问题

### 1. B1 辅助历史加载未消费的完整点云

`datasets/sampler.py:2816–2825` 在每个 Full 机制样本中调用 `get_frames` 加载三个辅助历史帧。
实际消费端 `datasets/sampler.py:649–832` 只读取这些帧的 `3d_bbox`、`timestamp` 和
`_ct_effective_timestamp`，没有读取其 `pc`。全仓对 `motion_aux_prev_frames` 的引用检索也未发现另一处点云消费者。
`datasets/nuscenes_lidar_mf.py:995–999` 已有同 annotation 的零点云 IO metadata 接口，
但现有 perf 仅在首帧路径使用它（`datasets/sampler.py:2754–2764`）。

本次使用真实 `motion_processing_mf`、v29 Full-CfC perf 配置和项目现有合成点云夹具，
删除辅助历史全部 `pc`，所有输出字段逐项完全相等：

```text
python -m pytest -q artifacts/ct_checks/20260914_code_audit/data/test_aux_metadata_equivalence.py
1 passed in 2.98s
```

用 AST 原样抽取 `_online_raw_view` 和 `_first_and_current_frames`，替换为计数 dataset，
hist_num=3、合法中间帧、辅助 offset=[2,4,6]，得到：

| 路径 | 普通机制行整云读取数 | scheduled H3 行整云读取数 |
|---|---:|---:|
| 原 v29 | 8 | 24 |
| 当前 perf | 7 | 21 |
| 其中仅 aux 使用、可由 metadata 替代 | 3 | 9 |

因此直接优化点是将 aux 帧取为 metadata，保留全部框、时间、身份、顺序与辅助监督。
普通 perf 机制行由7次整云读取变为4次；这是调用次数下降，不是已经测得的训练加速比。
此优化不会减少B1辅助时间间隔训练。

### 2. H3 抽样发生在点云 IO/worker 传输之后

`utils/recursive_state.py:198–224` 在每隔两个机制 batch 的一个合法slot安排shadow；
`datasets/sampler.py:2831–2839` 对所有安排事件递归构建未来两帧完整 raw。
10% keep 判定却在 host 的 `utils/v27_training.py:138–140` 才执行。
因此未采集的90%安排事件仍完成未来帧的读取、坐标转换和 DataLoader IPC。

上述计数表中当前 perf scheduled 行比普通行多14次整云读取。
可将与 `utils/v29_diagnostics.py:104–115` 完全相同的纯函数采样判断提前到 worker，
对未抽中事件保留 scheduled、future_exists、事件键及 not_sampled 状态，只延迟/跳过 future_raw 的点云材料化。
必须保留同一稳定种子、epoch、tracklet、frame和event版本；不能将未采样改成无未来或零收益。
H3微基准强制寻找/执行合法事件的入口应能显式覆盖延迟读取。

注意10%是安排事件中的10%，不是所有预测端点的10%；满16槽且每2batch一事件时，安排上限约为预测端点的1/32。

### 3. 全局样本到轨迹的映射是线性扫描

`datasets/sampler.py:2650–2654` 每次 `_locate_tracklet` 遍历全部轨迹边界。
v29 teacher与roll-in都调用它（`sampler.py:2866`、`utils/v29_rollin.py:29`）。
已构造好的 `tracklet_start_ids` 是累计边界（`sampler.py:2396–2401`），适合
`bisect_right(start_ids, anno_id)-1` 加原有范围检查，且不改变任何RNG或样本顺序。

实际原方法AST抽取后的合成CPU微基准，4000次查询、3次测量中位数：

| 合成轨迹数量 | 原线性查询 | bisect查询 | 索引/帧号一致 |
|---|---:|---:|---|
| 274 | 35.926 ms | 1.395 ms | 是 |
| 10000 | 2023.371 ms | 1.883 ms | 是 |

10000是刻意检验复杂度的合成规模，不冒充真实full轨迹数；这个局部倍数不代表训练提速。
若轨迹长度为零产生重复边界，bisect_right仍跳过空区间，需覆盖边界和越界情况。

更长期可按 sample_data token 对只读世界坐标整云做有界缓存，避免同帧多物体和重叠窗口重复解码变换。
`datasets/nuscenes_lidar_mf.py:971–979` 当前每个frame_id独立读取，
`:1014–1028` 每次执行文件读取和传感器→ego→global变换。
不能直接开启目前按轨迹复制整云的全量preloading；有界缓存需保留原始点顺序、ID、dtype、可写性和拷贝语义。

## mini/full适配：正确接线与仍需观测的分布

场景接线正确：`datasets/__init__.py:22–33` 对v29继承的v28合同选取场景；
`utils/v28_protocol.py:14–17` 将参数训练恢复为完整训练集合。
`utils/v27_protocol.py:17–36` 定义 mini为8/2，full为train_track350/val150，内部17/18与训练重叠。
`main.py:837–840` 对v28/v29验证传入val，最终映射官方test角色，不是旧v27内部dev。

观测样本量为总帧数F的4倍（`datasets/sampler.py:2407–2417`），
`utils/sampling_utils.py:46–65` 全总体shuffle并自然drop_last，
`main.py:980–999` 采用实际loader长度，full没有mini1262步硬限制。
机制 sampler 在显式scene manifest下遍历全部长度>1轨迹并覆盖尾部，
不再额外丢弃hash分区（`utils/recursive_state.py:135–171`、`:192–240`）。
`utils/dual_stream.py:82–111` 把整整一遍机制流均匀嵌入观测epoch，没有固定mini机制步数。

若F为全轨迹帧数、K为轨迹数，机制端点为F-K（忽略长度0特殊情况）；
观测更新约4F/16，机制batch约(F-K)/16，两者比例约(1-K/F)/4。
因此扩大到full不会使机制比例简单缩小40倍。长度分布和有效监督频率会改变实际比例，
应从provenance记录真实F/K、机制数以及正例/有效候选率，而不是按场景数直接缩放loss权重。
学习率和总更新预算由主审计另行分析；同样60轮不代表mini/full相同Adam更新数。

## 时间、类别和递归窗口

官方schema说明sample是2Hz带标注关键帧、sample_annotation.next是同一实例的后续标注；
sample_data关键帧时间戳靠近对应sample。项目按annotation.next遍历并取sample的LIDAR_TOP
（`datasets/nuscenes_lidar_mf.py:165–181`），并将LiDAR微秒时间转秒（`:1010`）。
这不是非关键帧sweep序列；改`key_frame_only=False`也不会自动取得20Hz历史。
mini/full均来自同一传感器/标注协议，不能按数据集规模把0.5秒时间尺度放大。
官方依据：[nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#sample)。

本地解析v29 perf配置：hist_num=3、time_scale=default_time_step=0.5、
use_real_time=true、main_time_source=order、pseudo_time_step=0.1。
所以B1消费真实物理时间；B0主干仍消费序号时间编码，这是既定模块设计，不应说全部主干都使用物理时间。
B0通常看到最近约1.5秒；短roll-in从t-4局部初始化，至多3次无梯度预测后训练端点，
覆盖约2秒局部窗口（`utils/v29_rollin.py:30`、`:69–91`、`:138–159`）。
从mini扩到full增加多样性，不自动覆盖长时间失跟状态；加长roll-in或困难状态采样属于新方法对照，非等价加速。

官方devkit提供train/val/test700/150/150和train_track/train_detect两个350场景半集，
mini8/2为开发子集。项目350并不是误丢一半已登记数据，而是采用了跟踪训练协议；
论文需写清train_track350而非笼统全部700训练场景。
依据：[官方splits.py](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py)。

项目Car→vehicle.car，Bus合并bendy/rigid、Pedestrian合并四类，
`datasets/nuscenes_lidar_mf.py:54–66` 与官方映射一致。
依据：[官方类别映射](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/tracking/utils.py#L20)。
当前任务仅Car，不能把同组几何范围在五类上有效当作已验证。

首帧尺寸在roll-in初始框、后续预测框和bbox_size均固定
（`utils/v29_rollin.py:52–57`、`:155`；`datasets/sampler.py:1809–1813`）。
GT历史保存为ground_truth_history，输入历史由online状态替换，Seg/BC/ref监督仍用GT
（`sampler.py:305–330`、`:1259`、`:1635`、`:1739–1743`）。
没有发现当前/未来GT重新灌入预测历史或full特有尺寸泄漏。

仍应重点诊断teacher与roll-in训练分布：训练实例首帧允许0点，评测/内部拟合要求首帧至少1点
（`datasets/__init__.py:44–46`；`nuscenes_lidar_mf.py:160`）。
teacher对历史GT框全空的样本重抽且可跨candidate，roll-in保持所有原端点
（`sampler.py:421–432`、`:2851–2885`；`v29_rollin.py:26–30`）。
所以25%是原请求teacher比例，不是25%干净GT实际样本；3个roll-in分支各占25%的总体请求。
应分teacher/rollin、当前真实点0/1/2/3+、合法历史数和原/实际重抽类别报告有效监督与loss贡献。
主审计已单独复现padding误监督问题，本报告不重复其结论。

低优先健壮性事项：真实时间模式启动时不调用`_endpoint_records`的严格单调校验
（`nuscenes_lidar_mf.py:696–718`、`:916–931`）；通用`compute_history_timestamps`
会对非正delta静默回退（`datasets/misc_utils.py:116–126`）。
目前没有实际nuScenes时间坏例，不能当成现有低分根因；可在metadata级preflight检测有序有限时间，不改变合法padding处理。

## 参数适配矩阵

当前仅登记v29 full YAML。不能只把数据根换成mini根：版本和scene manifest仍指向trainval。
本次内存修改配置对象后直接调用真实`validate_scratch_training_contract`：
仅将version/train/val/test split改成mini，报错
`ct_v28_expected_mini_car_updates_per_epoch=None (expected 1262)`；再明确设置1262后，
B0/Full-CfC/Full-GRU三份配置均通过合同检查，未执行训练。
依据：`utils/online_contract.py:714–716`、`:746–748`；
当前`ct_v26_full_formal_arm=False`，所以`:653–694`的旧full-only限制不适用。
这说明数据/网络可适配mini，但需要另行登记清晰的mini配置和data root，不能将通过配置检查写成真实训练成功。

| 参数 | 当前值 | mini→full是否应自动改 | 应依据什么再调整 |
|---|---|---|---|
| 关键帧时间/归一化 | keyframe；time_scale/default_dt=0.5s | 不应；两者同为2Hz标注 | 实测时间间隔异常/控制实验，禁止按数据规模乘系数 |
| 历史长度 | hist_num=3 | 无自动扩容依据；保持约1.5s历史是合理起点 | 失跟持续时间、变速/遮挡分层，延长属于新对照 |
| 当前模型roll-in | 最多3步，t-4局部起点 | 增大训练集不扩大递归状态支持 | 历史误差与部署误差分布，不能为等价提速删减或缓存过时预测 |
| B0点槽 | 每帧1024 | 不应按场景数放大 | raw点数/unique比例、空槽误监督、稀疏目标覆盖；先修mask再讨论增点 |
| B2证据 | pool768→selected256，relation/spatial/random=128/96/32 | 不应按场景数放大 | reachable→support→pool→selected漏斗；support本身缺目标时加selected无济于事 |
| 隐状态/特征宽度 | B1 hidden128/step64，融合/路由hidden64 | 保持作容量基线合理，尚无full最优证明 | 学习曲线、正负效用可分性、模块训练/诊断差距；独立消融容量 |
| 训练预算 | 60epoch，StepLR每20epoch×0.1 | 更多更新本身不是bug；保持每样本60次曝光有明确含义 | 同时报告自然更新数、样本曝光、有效机制更新和后期闭环泛化；变更须固定新协议 |
| loss权重/类别平衡 | 当前统一合同 | 不按full/mini场景比直接缩放 | 每项有效标签分母、正例率、梯度量级，尤其无效点槽与B3少量结构候选 |
| 数据读取并发/缓存 | workers4，不preload | 依据资源瓶颈，不能由mini内存外推full | loader等待/IPC/磁盘读量；优先aux metadata与有限token缓存，保留epoch同步 |

上表区分物理单位、样本内表示预算和优化预算；没有把“mini参数沿用full”本身判为错误。
