-- Portable materialization of the reviewed CT-SeqTrack paper-readiness snapshot.
-- Quantitative values were transcribed from the reviewed reports listed in
-- artifact.json. Qualitative rows are the audit judgments derived from those
-- reports, the code at commit 473738f, the user-provided 2026-07-22 server log,
-- and the cited literature/venue sources.

DROP VIEW IF EXISTS m03_proposal_errors;
CREATE TEMP VIEW m03_proposal_errors AS
SELECT 'Observation' AS proposal, 1.3492 AS mean_error_m,
       0.363567 AS median_error_m, 6.02805 AS p95_error_m, 1311 AS endpoints
UNION ALL SELECT 'Dynamics', 0.309334, 0.117888, 1.27717, 1311
UNION ALL SELECT 'Segment oracle', 0.231557, 0.09248, 0.986982, 1311;

DROP VIEW IF EXISTS innovations;
CREATE TEMP VIEW innovations AS
SELECT 1 AS "order", 'Matched irregular-cadence protocol' AS innovation,
       '已实现并冻结' AS maturity,
       'split-aware manifests；same endpoint/geometry/checkpoint；true/fixed/shuffled 只改 effective time' AS evidence,
       '可作为 protocol contribution；尚未扩成跨 tracker benchmark' AS claim_boundary
UNION ALL SELECT 2, 'Order/physical dual clocks', 'M1 已实现，E0–E5 通过',
       '主干保持 order time；real/effective time 只进入 dynamics、adapter、R(dt)',
       '可称结构设计；不能称通用 continuous-time theory'
UNION ALL SELECT 3, 'Bounded proposal innovation', 'M2 已实现，正式训练中',
       'stop-gradient residual、dt-dependent radius、frozen alpha、exact fallbacks',
       '正式 online tracking gate 未返回，暂不能声称有效'
UNION ALL SELECT 4, 'Shared-SE(2) canonical supervision', '已实现并验证几何不变量',
       '样本级世界 SE(2)；labels 来自 unperturbed GT + real dt',
       '是物理一致性修复，不应夸成单独重大算法'
UNION ALL SELECT 5, 'M3 history distillation', '仅计划',
       '无完成代码与正式实验', '不能写进当前贡献'
UNION ALL SELECT 6, 'M4 continuous-discrete filter/tube', '仅计划',
       '无 persistent state/covariance/filter update',
       '当前不应使用 state-filtering/continuous-time headline';

DROP VIEW IF EXISTS research_history;
CREATE TEMP VIEW research_history AS
SELECT 1 AS "order", '2026-05' AS period,
       '真实 timestamp、TimeEncoding、DynamicsEncoder、TWC、observability gate' AS work,
       '建立物理时间管线，但主干直接吃真实时间风险很高' AS result
UNION ALL SELECT 2, '2026-05 至 06', 'A1/A2/P5、多 seed、180 epoch 和动态残差消融',
       'raw real-time 主干崩溃；order-time 恢复；A2 seed 不稳定'
UNION ALL SELECT 3, '2026-07-08 起', '转向 variable-rate/HTV，设计 gap、burst、random 协议',
       '研究问题从常规涨点收敛到 irregular cadence 鲁棒性'
UNION ALL SELECT 4, '2026-07-11 至 20',
       '修 TWC 坐标/采样，做 crop reachability、recursive CV、reliability 和 frozen time controls',
       '排除 TWC/reliability 作为主方法；识别 crop 与监督污染'
UNION ALL SELECT 5, '2026-07-21', 'TWC A/B/C、M0 proposal oracle、candidate jitter audit',
       'oracle 解锁 M2；candidate audit 冻结 shared-SE(2)'
UNION ALL SELECT 6, '2026-07-22', '实现 M1/M2，完成 E0–E5，冻结 E6，启动 GPU2 正式训练',
       '工程门关闭；正式性能门仍在运行';

DROP VIEW IF EXISTS evidence;
CREATE TEMP VIEW evidence AS
SELECT 1 AS "order", 'Main baseline / raw real-time A1' AS experiment,
       '50.99/59.96 vs 28.28/27.43 (Success/Precision)' AS result,
       '真实秒数直接替换主干时间语义会严重破坏模型' AS interpretation
UNION ALL SELECT 2, 'A1 order / A2 order-dyn seeds',
       'A1 51.23/57.86；A2 seed42 50.96/63.31，seed43 23.64/23.77，seed44 46.90/52.62',
       '单 seed Precision 正信号不稳定，不能作为论文主结论'
UNION ALL SELECT 3, 'TWC A/B/C',
       'A 50.01/58.20；B 34.71/34.02；C 43.01/45.76',
       'TWC 能修复 paired-view 损伤的一部分，但仍低于 matched single-view'
UNION ALL SELECT 4, 'P0-C old feature concat',
       'true-fixed +0.438/+0.523；true-shuffled -0.123/+0.056；CI 跨 0',
       '旧结构没有可靠地利用正确物理时间'
UNION ALL SELECT 5, 'P0-B4 reliability',
       'gap AUROC 0.680/recall 0.568；burst 0.712/0.609，低于 0.75/0.70',
       'observation reliability gate No-Go'
UNION ALL SELECT 6, 'M0-3 proposal oracle',
       '1311 endpoints；d_obs 1.349m，d_dyn 0.309m，oracle 0.232m；dynamics gain 0.803m CI [0.633,0.988]',
       '有足够离线机制信号实现 bounded innovation，但还不是 tracking 涨点'
UNION ALL SELECT 7, 'Frozen E6 rule',
       'alpha 0.75；R=min(0.5+0.5dt,2)；gain 0.288m；clamp 34.48%',
       '冻结规则在离线 cohort 为正，可进入唯一正式实验'
UNION ALL SELECT 8, 'Candidate jitter audit',
       'velocity P50 0.611m/s；acceleration P50 2.128m/s²；matched penalty +0.0104m',
       '逐帧独立 jitter 是结构性伪运动，shared-SE(2) 必要';

DROP VIEW IF EXISTS literature_boundaries;
CREATE TEMP VIEW literature_boundaries AS
SELECT 1 AS "order", 'SC3D / P2B' AS work,
       '3D SOT 与 point-to-box proposal 基础范式' AS already_claimed,
       '不能声称首次 3D SOT 或首次 point-to-box' AS ct_boundary
UNION ALL SELECT 2, 'CXTrack', 'contextual information 提升点云跟踪',
       '你的核心不是一般上下文，而是 matched physical-time intervention'
UNION ALL SELECT 3, 'SeqTrack3D', '历史点云与 box sequence 的 seq2seq 跟踪',
       '保持其 order-time 观测语义，只在物理运动支路处理 irregular cadence'
UNION ALL SELECT 4, 'P2P', 'part-to-part motion cues',
       '不能泛称首次运动建模；强调真实 dt 和错误时间控制'
UNION ALL SELECT 5, 'PillarTrack / MVCTrack', 'pillar backbone 与 multimodal guidance',
       '你的工作不是 backbone 或 multimodal novelty'
UNION ALL SELECT 6, 'TrajTrack', '历史 bbox trajectory proposal/predict/refine',
       '不能泛称首次 trajectory guidance；强调 dt-conditioned bounded innovation 与公平 GT-free 评测'
UNION ALL SELECT 7, 'HVTrack / ChronoTrack', 'fixed-interval memory、temporal consistency',
       'within-track variable cadence 与 matched true/fixed/shuffled intervention 是更窄差异'
UNION ALL SELECT 8, 'NCDSSM / ContiFormer', '一般 irregular time-series 的连续/离散建模',
       '你的贡献是应用于 3D SOT 的轻量双时钟方法，不是基础连续时间理论';

DROP VIEW IF EXISTS publication_gaps;
CREATE TEMP VIEW publication_gaps AS
SELECT 1 AS "order", 'P0' AS priority, '当前 C2 formal result' AS gap,
       '没有 online Success/Precision 就没有方法结论' AS why,
       'true 同时胜 fixed/shuffled，standard guardrail 通过，日志/provenance/hash 完整' AS completion_rule
UNION ALL SELECT 2, 'P0', 'matched training-budget controls',
       'A1 init 后再训练 60ep 会把模块收益与额外优化步数混在一起',
       'C0 continuation、C1 adapter-only、C2 full 同 init/步数/数据路径'
UNION ALL SELECT 3, 'P0', '模块因果消融',
       '当前 full config 同时改变 shared-SE(2)、adapter、innovation',
       '至少 data-only、adapter-only、innovation-only/full 的配对比较'
UNION ALL SELECT 4, 'P1', '多 seed', '历史 A2 已出现严重 seed collapse',
       '42/43/44，报告均值、方差和 paired CI'
UNION ALL SELECT 5, 'P1', 'full dataset 与独立测试',
       'mini_val 已反复用于开发，不能充当最终盲测',
       'full nuScenes + held-out schedule/test；不得再据 test 调参'
UNION ALL SELECT 6, 'P1', '第二数据集',
       '单数据集难以证明 irregular-cadence 机制可泛化',
       'Waymo 优先；或 KITTI-HV 用于 interval robustness 对齐'
UNION ALL SELECT 7, 'P1', '公平现代基线',
       '只比原 SeqTrack3D 不能定位 2026 年竞争力',
       'SeqTrack3D、HVTrack、P2P、TrajTrack GT-free，条件允许加 ChronoTrack/其他最新方法'
UNION ALL SELECT 8, 'P1', '机制与效率', '审稿人会问模块为何有效以及代价',
       'dt/sparsity/failure bins；clamp/applied/fallback；params/FLOPs/FPS/memory'
UNION ALL SELECT 9, 'P2', 'benchmark 规模或真正 state filter',
       '决定能否冲更高的 PR/T-RO/CV venue',
       '≥2 datasets×≥5 trackers，或完成 persistent state/covariance/filter/tube';

DROP VIEW IF EXISTS next_experiments;
CREATE TEMP VIEW next_experiments AS
SELECT 1 AS "order", '现在' AS stage, '完成 GPU2 C2；不重复启动' AS action,
       '保留 last.ckpt、run.log、provenance、manifest/hash' AS decision
UNION ALL SELECT 2, 'C2 完成后',
       'GPU3 跑同 checkpoint 的 true/fixed/shuffled + A1 standard/gap/burst',
       '关闭 seed42 causal gate'
UNION ALL SELECT 3, '并行冻结', '定义 C0 continuation 与 C1 adapter-only',
       '消除额外训练预算和模块归因混杂'
UNION ALL SELECT 4, '结果后处理',
       '扩展 summary：innovation、adapter、clamp、alpha、fallback + tracklet bootstrap',
       '形成机制表和 failure recovery 分析'
UNION ALL SELECT 5, 'seed42 GO 后', '跑 seeds43/44，完全相同冻结规则',
       '排除历史 seed instability'
UNION ALL SELECT 6, '多 seed GO 后', 'full nuScenes、Waymo/KITTI-HV、held-out cadence',
       '完成泛化和独立测试证据'
UNION ALL SELECT 7, '最终包', '公平基线、效率、消融、代码/manifests release',
       '开始撰写并按 venue 压缩篇幅'
UNION ALL SELECT 8, '任一主 gate No-Go', '停止堆 M3/M4，转跨 tracker robustness benchmark',
       '重新定义文章，不把负结果包装成方法成功';

DROP VIEW IF EXISTS venue_fit;
CREATE TEMP VIEW venue_fit AS
SELECT 1 AS "order", 'IEEE RA-L' AS venue, '首选' AS fit,
       'M2 正 gate；matched-budget ablations；full nuScenes + 第二数据集 + 多 seed + 效率' AS required_package,
       '最适合简洁、可复现的机器人感知/跟踪方法；篇幅短，贡献必须聚焦' AS assessment
UNION ALL SELECT 2, 'IEEE T-IV', '很合适',
       '完整自动驾驶场景验证、多类别/多数据集、鲁棒性和实时性',
       'irregular LiDAR cadence 与智能车感知主题自然匹配'
UNION ALL SELECT 3, 'IEEE T-ITS', '合适但需扩展',
       '突出交通/自动驾驶系统价值，实验规模和应用意义更完整',
       '适合更长的系统性版本，不宜只交 mini 单类实验'
UNION ALL SELECT 4, 'IEEE IV 2027', '合适的会议路线',
       '方法正结果、完整 automotive validation；关注官方后续 deadline',
       '应用导向强；若想先发会议，这是现实目标'
UNION ALL SELECT 5, 'Pattern Recognition', '扩展版/高门槛',
       '更普适的方法论、强基线、≥2 数据集，最好同时形成 benchmark 贡献',
       '当前 mini 阶段不够；只有 evidence package 显著扩大后才建议'
UNION ALL SELECT 6, 'TCSVT', '次选',
       '强调时序视觉建模并提供充分视频/点云实验',
       'LiDAR-only 的题目不如 RA-L/T-IV 贴合'
UNION ALL SELECT 7, 'T-RO', '当前不建议',
       '真正 state filtering/uncertainty/tube，重大机器人学推进和广泛验证',
       '当前 bounded correction 新颖度与证据规模不足'
UNION ALL SELECT 8, 'CVPR/ICCV/ECCV', '当前不建议',
       '更强算法新颖性、最新 SOTA、大规模数据与清楚的视觉社区影响',
       'M1/M2 现阶段偏增量；完成 M3/M4 或大 benchmark 后再评估'
UNION ALL SELECT 9, 'Workshop / technical report', '仅在资源受限时',
       'mini + 单 seed 可作为阶段性公开材料',
       '不应在主项目仍有上升空间时过早终止为低证据版本';
