# v39 最新配置说明（2026-10-28）

正式文件为`cfgs/ct_seqtrack/39_full_a_mini.yaml`、`39_full_b_mini.yaml`；完整默认值由`models/ct_v39/config.py`解析，源码与两份解析配置登记在`39_registration.json`。两份YAML不包含全部内部固定常数，不应自行补入未开放字段。

## 共同训练与执行

| 项目 | 固定值 |
|---|---|
| 数据/类别 | nuScenes v1.0-mini，Car，global坐标 |
| 随机种子/初始化 | seed42，scratch，无init checkpoint；A/B初始权重hash相同 |
| 训练 | 60 epoch，batch16，workers4，每epoch验证 |
| 窗口 | 1/4/4/8，10轮窗口课程；不是模块启用课程 |
| 模块启用 | epoch1全部启用，不冻结、不warmup、不分阶段 |
| Adam | betas=(.5,.999)，eps=1e-6，weight_decay=0 |
| 主参数/几何/损失 | FP32；时间差及大世界原点差先FP64 |
| 特征/attention | BF16，FA2 2.5.8 varlen、deterministic；4×64，非因果，attention dropout训练.1/评测0 |
| 当前运行安排 | 两独立进程同时devices=0；线程OMP/MKL/OPENBLAS各2，原云解码workers仍4 |
| 环境名 | `ctseqtrack3d`，Python3.9.19/torch2.0.1+cu118/Lightning2.0.2/flash-attn2.5.8 |

A：1–20轮1.5e-4，21–50轮1.5e-5，51–60轮1.5e-6。B：1–20轮1e-4，21–40轮2e-5，41–60轮4e-6。衰减在相应轮完成后生效；B不是全程更低学习率。

## E、G、M和自适应

| 部分 | 首版默认值 |
|---|---|
| raw缓存 | 共享CPU512MiB/GPU256MiB，物理帧去重，缓存无目标标签 |
| 原云精度 | nuScenes sensor原云直接读取，FP64静态旋转、局部FP32、world原点FP64；避免先写入大世界FP32再相减 |
| PointNet | 压紧真实点、宽度不变，masked BN总体/first-max池化规则保持 |
| G模板 | 永久首模板＋最近2个可信模板，各最多64点 |
| G当前证据 | prior/trusted各最多256点，正常去重后≤512；恢复增加两个≤256点精化区 |
| G匹配 | k8，4→32→64，LN/GELU，5轮FP32 dustbin Sinkhorn |
| 匹配GT半径 | clip(.1×物体对角线,.15m,.5m)，仅loss使用 |
| 几何纠偏 | XY≤min(2m,.5平面对角线)，Z≤min(.5m,.25高度)，yaw≤30° |
| 有效对应 | 至少3个唯一点；yaw至少6点且XY协方差比≥.05，否则只平移 |
| 七槽 | q0/prior/trusted-CV/G(q0)/G(prior)/G(trusted-CV)/G(reacquire) |
| 恢复 | R=clip(4+3σmax+2×可信年龄,8,24)m；局部2048代表点；连续2帧失败后下一帧全云4096 |
| 粗搜索 | 4个yaw假设、.5m网格、raw-ID去重投票、NMS后2峰；每帧最多1次粗搜索/2次精化/1次B0 |
| 恢复送B0 | 原区域缺目标支持时，reacquire T≥.5且唯一支持≥3 |
| 可信写入 | 当前观测来源，T≥.8，唯一支持≥3，区域no-match<.5 |
| 速度档跳B0 | T≥.9，唯一支持≥5，no-match≤.1，不同几何候选效用差≥.05，连续轻帧≤3，距完整B0≤1.5s |
| 训练路由 | 25%整窗稳定hash分配按预测路由真实提交；所有样本仍计算完整B0监督 |
| M | h=2/4，s=t−1，真实Δt，live128维context＋2时间特征→共享130→64→2，末层零初始化 |
| M目标 | GTxy(s+h)−GTxy(s)，尺寸归一化，权重1/.5，整体.05；无12m限幅；推理不执行未来头 |

accepted/trusted/search anchor分离。恢复或连续≥2帧不可信后重新可信会开启新motion epoch；历史点云/模板保留，旧运动边禁用。等待期只有两个相邻、同epoch可信观测才能重建速度和运动边。GT不参与候选、路由、搜索或状态提交。

逐候选诊断分别记录`qG`和实际写入的`state_quality`：真实q0沿用原quality，G观测使用qG，原始传播候选quality和可信T均为零；不把T或占位q0当作当前观测质量。

## 损失与评价

B0原损失保留；B1 physical/sigma=.1/.05；删除acquisition .05，加入M .05。G的S/P/T/ranking=.125/.125/.5/.1；匹配/纠偏/targetness/区域no-match/qG=.5/.5/.2/.2/.5。按有效点/候选再endpoint归一化；空恢复分支不能稀释其他端点纠偏监督。25%端点增加近/远扰动辅助候选，不进入正式七槽提交。T在exact IoU的.5±.001灰区屏蔽二元监督。

每轮验证精度档；正式比较final60与late3(58–60)。58–60同时评测默认速度档；60轮扫.8/.9/.95，完整保存两组和逐候选/恢复记录。无当前B0的轻帧不统计“修复坏q0”机制。共享GPU时间标明共享状态，不当独占FPS。

## 已修复的兼容与契约问题

- G混合精度验收按实际算子区分：隐藏Linear输出BF16，LayerNorm遵循autocast使用FP32，分类/质量出口保持FP32。不能因LayerNorm后的特征为FP32就判定BF16未启用；完整batch工具记录逐层dtype并核验真正的Linear执行精度，不改变模型或强制量化出口。[PyTorch2.0.1 autocast源码](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/autocast_mode.cpp#L323-L379)
- PyTorch2.0 SDPA无`scale`关键字：采用等价Q预缩放，注册head64/temperature8时缩放为1。
- PyTorch2.0.1的CUDA `scatter_reduce(amax)`会在严格确定性模式报错（[官方源码](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cuda/ScatterGatherKernel.cu#L470-L485)）：恢复投票改为稳定分组后的`segment_reduce(max)`，CPU与原实现逐值相同；真实batch工具强制执行非空双峰恢复审计，避免随机batch未触发恢复而漏验。
- PyTorch2.0 CPU oneDNN GELU反向对特定非连续梯度异常：CPU梯度连续化，CUDA前向/参数和checkpoint不变。
- FP32 packing不宣称逐元素bitwise等价；FP64输出/输入和参数梯度严格对照，FP32相对FP64的L2/L∞参数梯度误差门为1e-4，输出/输入梯度与BN检查保持。
- 整云世界坐标提前FP32量化、未来缺标删除当前样本、恢复期间使用不可信边、teacher Full刷新轻路径时间、空恢复稀释loss、占位q0误算恢复均有对应实现修正/回归检查。
- 原二维NLL、整数排序/计数、唯一raw ID、全空attention、退化拟合、原子提交及epoch-boundary身份规则保留。15:08正式默认值的真实CUDA单批生命周期已通过；单批验收不等于全60轮或全部边界场景验证。

环境创建、一次CUDA检查、两条独立后台训练与新终端tail命令见[启动页](CTSEQTRACK_V39_LAUNCH.md)。
