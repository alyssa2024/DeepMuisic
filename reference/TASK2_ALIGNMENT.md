# 任务二(BTT)对齐表 —— plan_final.md ↔ 已探索工作

**目的**:把 `reference/plan_final.md` 任务二(§9–§16, Batch 3/4)的每一项,对照本仓库**已经做过**的
工作,标清「已做 / 部分 / 未做」,避免执行 plan_final 时重复已验证的内容。

**当前约束(用户 2026-06-27 确认)**:
- **K 已知** —— 合成数据集 K 已知,定阶问题(plan_final §16.B4 / 我们 plan.md §3.5 两层定阶)
  **暂不碰**。两份 plan 关于定阶的冲突(occupancy head vs BIC/pruning)悬置,不影响本表。
- **任务一(V0/V1/V2 通用 DeepMUSIC 验证)假设已在别的系统完成** —— 本表只覆盖任务二。

**证据来源**:`comparison/FR_FRONT_ELBO_FINDINGS.md`、`comparison/results/plan.md`、相关 artifacts。

---

## 0. 一句话结论

任务二的**前端(FR-front)+ 离散选择(q(z))+ 捕获域核对 + 大量基线对比**我们都做过了;
**核心实证结论「不训连续 δ 头、改 per-instance 精修」与 plan_final §9 完全一致**。
**还没做的是把验证过的东西正式工程化**:GN solver、conditional alternating 多分量、
local covariance 输出、丰富候选特征。**B0–B3 的"验证性"内容覆盖约 60-70%**。

---

## 1. 逐项对齐表

| plan_final 项 | 要求 | 已探索工作 | 状态 | 复用 / 缺口 |
|---|---|---|---|---|
| **§9 不训 δ 头** | BTT 不再训连续频率 δ 摊销头 | 三连实验(无监督/监督/自由参数)钉死 δ 摊销失败;自由参数 0.42→0.11 | ✅**已证** | 直接作为路线的实证基础,见 FINDINGS §7-8 |
| **§10 BTT 信号模型** | 线性 Campbell f_k(t)=a0+a1·Ω(t),相位含 Θ(t) | 现有合成数据是**定常频率**(per-sequence 固定 f),非 Campbell 时变 | ⚠️**部分** | 现数据可做常频 BTT;Campbell 时变需扩 `BTTSequenceDataset` |
| **§11.1 候选生成 FR-front** | nudft FR-front 提候选 | nudft FR-front 已建,几何鲁棒(±10% recall 1.0,MAE 0.50-0.65) | ✅**已做** | ckpt: `comparison/artifacts/geomrobust_nudft_speedrand/geomrobust_fr.pt` |
| **§11.1 候选特征(9项)** | freq, fr_score, profile_rss, local_fr(re/im/mag), rotor_speed, probe_geom + 局部FR邻域±4bin | 现 FR pool 只给 `[freqs, radius, scores]` 3项 | ⚠**部分** | `run_frfront_elbo.py:_fr_pool_factory` 只输出3项,需扩到9项+9bin邻域 |
| **§11.2 候选标签** | 含真值的 basin 索引;多覆盖取 profile RSS 最小 | 现 label = 真频最近候选(`__getitem__` 已算) | ✅**已做** | `LegacyFourComponentCandidateDataset` 已有 label |
| **§12 离散 encoder q(z|y,C)** | observation enc → h_y;cand MLP → h_m;score=MLP([h_y,h_m]);softmax | `FourComponentCandidateEncoder` 已实现(transformer + cand_mlp + z_head) | ✅**已做** | 比 plan_final 要求更强(已含 component embedding) |
| **§12.3 候选损失** | 仅 -log q(z*|y,C),梯度不穿 refinement/FR/GN | z_loss(soft_basin/soft_freq)已实现,Stage1 z-only 训练 | ✅**已做** | 梯度隔离已满足(Stage1 backbone 冻结/独立) |
| **§5.1/§13 marginal solver** | 复幅值边缘化 NLL: yᴴC⁻¹y+logdetC | `JointConstantFrequencyDecoder.marginal_nll_per_sample` 已实现 | ✅**已做** | `run_legacy_...py:1235`;tau² 经验贝叶斯估计 |
| **§5.2/§13 profile-MAP** | profile-MAP(含幅值高斯先验,ridge) | `solve_reconstruction`(point_ls + ridge)已实现 | ✅**已做** | `:1185`;已验证 marginal≈point_ls(well-sep) |
| **B0:oracle basin + 复现 0.11** | marginal/profile-MAP **+ GN** 在绝对频率上复现 ~0.11 | **正式 B0 已跑**:阻尼牛顿+绝对频率,marginal MAE **0.081**/profile_map 0.081(init 0.48→精修)。优于 LSF 0.127,逼近 CRB 0.099 | ✅**已做** | `experiments/btt_oracle_continuous_refinement.py`(分支 deepmusic_validation_and_btt)。marginal≈profile_map |
| **B1/§13.3 GN 捕获域** | 扫初值误差 ±2Hz,记录收敛/MAE/发散率 | **GN 捕获域扫描已跑**:±1.5Hz 内 success 98.8%/diverge 0%,±2.0 仍 96%;收敛后 MAE 0.084。FR 初值误差(max 1.84)**全在捕获域内** | ✅**已做** | `experiments/btt_gn_capture_region.py`。**捕获域比预期宽 → sub-bin 插值基本非必需**(仅 >1.8Hz 极尾部保险) |
| **§13.2 conditional alternating GN** | Π⊥ 投影残差,逐分量 GN,不用 CLEAN | **已实现并实测,但推翻了 plan 假设**:`solvers/conditional_alternating_gn.py`。well-sep MAE 0.173(劣于联合牛顿 0.082);**近邻直接崩溃**(spacing 5Hz→1.79, 3Hz→2.13,联合牛顿仅 0.16) | ⚠️**已做但否决** | **联合牛顿(marginal NLL)才是主力 solver**(见下方 §5)。Π⊥ 在近邻把真分量能量一起投影掉 → 反而有害 |
| **B2:candidate encoder + K已知** | top1 / Recall@3 / basin_hit / downstream MAE | q(z) selection: **top1=1.0, basin_hit=1.0**(各配置稳) | ✅**已做** | F2 冻结区;Recall@3 / downstream MAE 口径需补测 |
| **B3:多源对比** | FR raw / greedy / physics-only / encoder-only / hybrid / oracle | FR raw 0.50、greedy raw 0.16、DeepFreq-grid 0.495/+LS 0.273、LSF 0.127、amortized_vi 0.136 | ✅**大部已有** | 见 FINDINGS;缺正式 hybrid score(§14 s_m) |
| **§14 完整 hybrid 推理** | top-L + per-cand GN + s_m=-RSS/σ²+γ·log q + 报 physics/encoder/hybrid 三路 | **已实现并跑通**:`experiments/btt_full_pipeline.py`。真实候选(非oracle)端到端 FR→q(z)→牛顿→打分,**encoder/hybrid MAE 0.084**(=B0 oracle 水平),raw_sel 0.528→精修 0.084 | ✅**已做** | well-sep 三路同为 0.084(易场景区分不出);差异需到近邻/弱分量/低SNR 才显现 |
| **§15 local covariance 不确定性** | 每 basin H⁻¹(Fisher CRB);mixture q(f|y)=Σπ_m N(f̂_m,H⁻¹) | **已实现并验证校准**:`experiments/btt_local_covariance.py`。收敛点 Hessian⁻¹ 给的 σ_pred=0.102≈RMSE 0.105,**z-std 0.994、95%覆盖 94.2%、校准比 0.801** | ✅**已做** | H⁻¹ 就是校准的频率后验协方差(=CRB),解析免费,无需变分。连续 uncert=H⁻¹,basin uncert=π_m(§14 s_m) |
| **B4:未知 K** | BIC/pruning(plan_final) vs occupancy head(我们 plan) | **悬置**(用户:K 已知,先不碰) | ⏸**暂缓** | 两 plan 冲突待裁决 |
| **B5:真实数据** | 重构残差/轨迹连续性/段一致性 | 历史有 synthetic→real 探索(见 memory) | ❌**未做(本路线)** | 见 [[synthetic-real-bridge-findings]] |

---

## 2. 按 plan_final Batch 划分的"还需做什么"

### Batch 3(BTT oracle 连续精修)—— 覆盖 ~50%
- ✅ marginal / profile-MAP solver 已存在(直接复用)。
- ⚠️ **B0**:把 Adam+δ 的 0.11 验证,改写成 **GN + 绝对频率**版,正式复现 0.11。
- ❌ **B1/§13.3**:实现 **GN 自身**的捕获域扫描(给定初值误差 → GN 收敛率),区别于我们已做的"FR 初值误差分布"。
- ❌ **§13.2**:实现 conditional alternating GN(Π⊥ 投影,多分量)。

### Batch 4(候选选择 + 完整 pipeline)—— 覆盖 ~60%
- ✅ FR-front + q(z) selection 已就绪(top1=1.0)。
- ⚠️ **§11.1**:候选特征从 3 项扩到 9 项 + 局部 FR 邻域(±4 bin)。
- ⚠️ **B2**:补 Recall@3 / downstream refined MAE 口径。
- ❌ **§14**:完整 hybrid 推理串联(top-L → per-cand GN → s_m 打分 → physics/encoder/hybrid 三路报告)。
- ❌ **§15**:local covariance(Hessian/CRB)+ mixture 后验输出。

---

## 3. 直接可复用的资产清单

| 资产 | 路径 | 用途 |
|---|---|---|
| nudft FR-front ckpt | `comparison/artifacts/geomrobust_nudft_speedrand/geomrobust_fr.pt` | §11 候选生成 |
| q(z) selection encoder | `FourComponentCandidateEncoder` + Stage1 ckpt | §12 离散选择 |
| marginal/profile solver | `JointConstantFrequencyDecoder`(`:1152`) | §5/§13 目标函数 |
| 合成数据 + 候选 + label | `LegacyFourComponentCandidateDataset`(`:638`) | §11.2 候选标签 |
| 0.11 可达性侧证 | `comparison/probe_delta_optim.py` | B0 基准 |
| FR 初值误差分布 | `comparison/results/plan.md §3` | B1 初值侧 |
| 基线对比数(LSF/DeepFreq/amortized_vi) | `FR_FRONT_ELBO_FINDINGS.md §8` | B3 对照 |

---

## 4. 关键提醒(执行 plan_final 任务二时)

1. **不要重做 B2 selection**——已 top1=1.0,直接复用,只补 Recall@3 / downstream MAE 口径。
2. **不要重新验证"δ 头不行"**——已三连钉死(FINDINGS §7-8),plan_final §9/§20.8 已采纳。
3. **B0 的 0.11 是 Adam+δ 侧证,不是 GN+绝对频率**——不能直接宣称 B0 完成,需正式跑 GN 版。
4. **真正的新代码集中在**:conditional alternating GN(§13.2)、完整 hybrid 推理(§14)、
   local covariance(§15)、候选特征扩充(§11.1)。前端和选择层是现成的。
5. **现合成数据是定常频率,非 Campbell 时变(§10)**——若要时变需扩数据集;常频 BTT 可先用现有。
6. **定阶(B4)悬置**:K 已知前提下不实现;两 plan 的 occupancy-vs-BIC 冲突待后续裁决。

---

## 5. 执行中推翻的 plan 假设(重要,需回流到 plan_final)

### §13.2 conditional alternating GN(Π⊥ 投影)—— 否决,改用联合牛顿
plan_final §13.2 假设「条件交替 + Π⊥ 投影」比联合更新鲁棒(尤其近邻),并明令不用 CLEAN。
**实测推翻了这个假设:**

| 场景 | 联合牛顿(marginal NLL) | 条件交替(Π⊥+profile-RSS) |
|---|---|---|
| well-sep c16 (4分量) | **0.082** | 0.173 |
| 近邻 spacing 20Hz | 0.060 | 0.091 |
| 近邻 spacing 10Hz | 0.068 | 0.181 |
| 近邻 spacing 5Hz | **0.159** | **1.79**(崩) |
| 近邻 spacing 3Hz | **0.164** | **2.13**(崩) |

**机制:** 两分量近邻时其原子高度共线,Π⊥ 把「其他分量」投影掉会**连真分量能量一起抹掉**,
残差 r_k 里真信号没了 → 单频拟合跑飞。联合 marginal 同时处理所有分量 + logdet 正则,反而稳。

**结论:主力 solver = 在【绝对频率】上对【联合 marginal NLL】做【阻尼牛顿】**
(`experiments/btt_oracle_continuous_refinement.py:newton_refine`),
而非 plan_final §13.2 的条件交替。条件交替脚本保留作记录,不进主 pipeline。

### §13.3 / plan.md §3 sub-bin 插值 —— 基本非必需
GN 捕获域(B1)实测 ±1.5Hz 全稳、±2.0 仍 96%,FR 初值误差(max 1.84)全在域内。
原计划的「sub-bin 插值兜尾部」**基本不必要**,仅对极少数 >1.8Hz 样本有保险价值。

### §14 三路 SNR 扫描(K=4 同中心,encoder 不 OOD)—— `btt_full_pipeline.py --snr_db`
低 SNR 才让 physics/encoder/hybrid 三路分化(精修后 freq MAE):

| SNR | physics | encoder | hybrid | raw_sel(未精修) |
|---|---|---|---|---|
| 20 dB | 0.087 | 0.087 | 0.087 | 0.526 |
| 10 dB | 0.278 | 0.278 | 0.278 | 0.589 |
| 5 dB | 0.500 | 0.500 | 0.500 | 0.768 |
| **0 dB** | **12.79** | **2.51** | **2.51** | 2.25 |

**结论:**
1. **20–5dB 三路一致**:SNR 够时 FR 峰高选 = q(z) 选,physics-only 够用。
2. **0dB:physics 崩(12.8),encoder 稳(2.51,好 5×)**——**encoder/q(z) 价值的硬证据**:
   低 SNR 下 FR 峰被噪声污染,q(z) 聚合全局观测证据选 basin 更抗噪(= DeepMUSIC "encoder
   聚合全局证据" 思想)。
3. **0dB encoder≈raw_sel**:选错 basin 时精修救不回 → **瓶颈从精修转移到 selection**。
4. **价值互补**:physics 高SNR够用;encoder 价值在低SNR(抗噪选洞);精修价值在中高SNR(洞选对后)。

### oracle vs encoder SNR 对比(分离「SNR 固有难度」vs「encoder OOD」)—— `btt_oracle_continuous_refinement.py --snr_db`
⚠️ **关键修正**:前置 q(z) encoder 是【生产条件 20dB 单工况】训的。用 oracle selection
(绕开 encoder)重测,分离两种难度:

| SNR | oracle+精修(固有下界) | encoder+精修(§14) | 差距来源 |
|---|---|---|---|
| 20 dB | 0.084 | 0.087 | ≈0,encoder 够用 |
| 10 dB | 0.268 | 0.278 | ≈0 |
| 5 dB | 0.484 | 0.500 | ≈0 |
| **0 dB** | **0.941** | **2.51** | **encoder OOD 拖累 2.7×** |

**结论:**
1. **5dB 以上:oracle≈encoder**——basin 选择是粗粒度任务,对 SNR 不敏感,生产 encoder
   够用,**此区间精修是瓶颈**(随 SNR 降变差 = CRB,非 encoder 问题)。
2. **0dB:oracle 0.94 vs encoder 2.51**——**坐实"encoder 工况不泛化"**:0dB 固有下界 0.94,
   encoder 只到 2.51,多出的 1.6 是 20dB-训-encoder 在 0dB **选错 basin** 造成。
3. → **极低 SNR 下 encoder 泛化是真问题**(需多工况重训,plan_final §3 的"多参数覆盖"要求);
   但精修层(任务二核心增量)是 **selection-free** 的,oracle 下结论不受污染,稳。
4. **下一阶段**:多工况重训 q(z) encoder(SNR/freq-band/K/speed 覆盖),才能诚实测全工况。

### 弱分量(强+弱,oracle K=2,spacing 40Hz)—— `experiments/btt_weak_component.py`
此前所有测试都是等幅;补测强+弱后:

| amp_ratio | f_strong MAE | f_weak MAE | a_strong relerr | a_weak relerr |
|---|---|---|---|---|
| 0 dB | 0.063 | 0.048 | 2.5% | 2.4% |
| -10 dB | 0.046 | 0.113 | 1.8% | 5.5% |
| -20 dB | 0.043 | 0.353 | 1.7% | 16.6% |
| -30 dB | 0.043 | 1.28 | 1.7% | 52% |

**结论:**
1. **强分量不被弱分量拖垮**(联合牛顿无"强梯度压弱"问题),弱分量随 ratio 单调退化。
2. **-20dB 是坎**:弱分量有效 SNR ≈ 标称+ratio,-20dB 时弱分量有效 SNR≈0dB,落入 CRB 阈值区
   → 这是**信息论下界决定的,非方法缺陷**(任何方法到此都难)。-10dB 内可用(频率0.11/幅值5.5%)。
3. **marginal ≈ profile_map,弱分量上也没拉开差距**(修正此前"marginal logdet 在弱分量更优"的猜测;
   在已选对 basin + oracle 初值下两者实质等价)。
4. **方法直接给出幅值估计**(0/-10dB 误差个位数%)——兑现"克服 DeepFreq 不估幅值"的立项目标。
- ⚠️ 待测:弱分量 + **selection**(非 oracle)——弱分量 FR 峰可能被淹没,q(z) 能否选到其 basin 未验证。
