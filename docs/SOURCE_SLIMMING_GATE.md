# 源码瘦身门禁与待迁移清单

本轮已经完成运行树外围、配置、文档、工具和测试的收敛，但没有提前物理删除兼容宿主源码。原因不是这些源码都仍属正式方法，而是当前机器缺少 nuScenes 数据及完整 Lightning 依赖，且用户明确跳过本轮服务器 smoke/等价检查。真实 batch、100-step/resume 和逐帧点框验收均记为“未执行”，不能写成“已通过”。

## 已确认的源码集中区

| 文件 | 当前规模 | 瘦身后职责 | 当前处理 |
|---|---:|---|---|
| `models/seqtrack3d.py` | 7,725 行 | SeqTrack/B0 与隔离 B4 | 保留兼容实现；真实等价后抽取 B0 helper、迁走 B1--B3 编排，再删不可达分支 |
| `models/ctseqtrack.py` | 46 行 | B1--B3 正式 composition root | 先保持公共类与继承关系，等价后承接编排 |
| `models/base_model.py` | 2,551 行 | 公共训练/evaluator 宿主 | CRPA 等旧入口暂不物理删除 |
| `models/ct_v2/motion.py` | 2,372 行 | 物理时间函数、`OrderedPhysicalMotionEncoder`、`physical_motion_uncertainty_loss` | 参数名和构造顺序未动；真实等价后移除其余多代 encoder/fusion/router |
| `utils/ct_search.py` | 1,262 行 | 当前 B1 support 与 B2 extension-only sampling | 保留当前调用链及兼容函数，待真实 call graph 验证 |
| `utils/replay_cache.py` | 306 行 | 仅剩 checkpoint/hash/resume 所需部分 | 先保留；真实等价后把所需函数迁入 `online_contract` 并删除 replay cache 实现 |

这些保留项不代表旧 CRPA、旧 Search/Gate、旧 replay 或多代 fusion 仍属于论文方法。活动配置与正式工具均不以它们作为实验入口。

## 允许物理删除的必要条件

同一源码阶段必须一次性通过：

1. 22 个活动配置逐键等价；
2. seed42 参数名、shape、数量、初始化 hash 和 optimizer/scheduler 分组等价；
3. 固定真实 batch 的 B0/B1/B2/B3 键、shape、dtype、loss、梯度归属与递归提交等价；
4. scratch 100-step 与 epoch-boundary resume/连续运行等价；
5. 真实序列含非空点云、有限预测框、逐帧输出与正常 Success/Precision；
6. `git diff --check`、全量 `py_compile` 和全量 pytest 通过。

门禁未满足时，只允许保留兼容实现，不得根据静态“看似未使用”结果删除宿主代码。本轮可以声明“本地运行树、配置、工具与文档瘦身完成”，但不得声明“服务器行为等价验收完成”或“宿主源码物理瘦身完成”。
