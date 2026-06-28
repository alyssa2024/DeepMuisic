# FR-front + ELBO 路线：实验结论汇总

> 目标回顾：用几何鲁棒的 **nudft-FR basin proposal** 替换原 amortized-VI 里昂贵、需已知频率数的 **greedy CLEAN 候选生成**，下游（q(z) 选择 + ELBO 局部高斯后验，Stage1 z-only → Stage2 无监督 ELBO）**保持不变**。期望：精度对齐原 greedy amortized-VI（c16@20dB，freq_mae ~0.16–0.17 Hz），同时几何鲁棒、不需已知频率数、更便宜、更少超参。

日期：2026-06-27。本文档汇总到目前为止**验证过**的路线与**实证确认**的瓶颈。

---

## 0. 一句话结论

**问题不是「ELBO 做不了训练损失」**——ELBO 的重构项是有效梯度信号（自由参数实验直接证明）。
**真正的瓶颈是 amortized 的 δ 局部精修头（delta_head）从头到尾就没真正工作过**：在
greedy / FR / 大 lr / 软钳位 所有配置下，δ 都不动甚至帮倒忙。原 production 的 0.169
**几乎全部来自 greedy 候选本身的质量（raw 0.16）**，ELBO 的「局部精修」是空壳。

因此 FR 路线卡在 **0.50 = FR 候选的网格量化精度**：FR 峰锁在 1.95 Hz 网格上（半 bin 0.98 Hz），
而 δ 既然不工作，最终精度就停在候选精度上。

---

## 1. 几何鲁棒前端（已验证，可用）

- **nudft-FR 前端**几何鲁棒：±10% 转速 recall 1.0，MAE 平稳 0.50–0.65 Hz。
  结构性把采样时刻 t_n 注入相位（s_m = Σ_n y_n·exp(-j2πf_m·t_n)），优于朴素通道拼接，
  远优于官方 DeepFreq（后者转速 ramp 去混叠崩溃，recall 0.20–0.25）。
- checkpoint：`comparison/artifacts/geomrobust_nudft_speedrand/geomrobust_fr.pt`。
- **selection（q(z) 选 basin）在 FR 候选上完美**：component_top1 = 1.0，basin_hit = 1.0
  （所有 Stage2 实验全程稳定，大 lr 也没崩）。

> 即：proposal 与 selection 两层都 OK。问题完全在**局部频率精修**这一层。

---

## 2. 决定性实验：δ 精修到底工作没有

### 2.1 自由参数实验（`comparison/probe_delta_optim.py`）
绕过 encoder，把 δ 设成每样本/每分量的**自由标量**，只用 cartesian_topk 的 recon 梯度优化：

| step | recon | freq_mae |
|---|---|---|
| 0 | 112.6 | 0.4167 (=raw) |
| 399 | 52.7 | **0.1112** |

→ **纯 recon 梯度能把频率从 0.42 推到 0.11**。ELBO 重构项是有效训练信号，landscape 有正确最小值。

### 2.2 recon 曲率探针（`comparison/probe_delta_curvature.py`）
固定其它分量，手动扫单分量 δ，画 recon NLL：

- NLL 跨度 44–111（δ 偏 1 Hz，recon 涨几十），**完全不平**。
- argmin_delta 与 true_delta 同号同量级 → **最小值确实落在真频附近**。
- **point_ls 与 marginal 两条曲线几乎完全重合**（差 <0.05）→ 幅值边缘化与否不是瓶颈。

### 2.3 encoder 却学不动（多组训练实证）

| 实验 | 候选来源 | 候选 raw | δ 精修后 | δ 有效？ |
|---|---|---|---|---|
| production（归档 metrics）| greedy | 0.162 | 0.169 | ❌ 略变差 |
| greedy 对照（本次 ep1）| greedy | 0.142 | 0.216 | ❌ 变差 |
| FR + lr 扫描 lr3e3/b256/40ep | FR | 0.501 | 0.496 | ❌ 不动 |
| FR + KL=0（A/B）| FR | 0.501 | 0.500 | ❌ 不动 |
| FR + marginal（A/B）| FR | 0.501 | 0.498 | ❌ 不动 |
| **自由参数（非 amortized）** | FR-near | 0.417 | **0.111** | ✅ |

**唯一差别**：δ 是「直接自由参数」还是「从特征 h 经 delta_head 回归」。recon 梯度强、最小值正确，
但 encoder 学不出来 → **delta_head 的输入里没有定位 sub-bin 偏移所需的信息**（FR 候选锁网格，
`candidate_mode=profile_ls` 又清零了 alias/bse 特征，delta_head 实际只看到「候选频率 + FR 峰高」），
其最优策略就是恒输出 δ=0。

---

## 3. 反驳的若干假说（已逐一证伪）

| 假说 | 验证方式 | 结论 |
|---|---|---|
| KL 把 δ 拉回 0 | `--local_kl_weight 0` | ❌ δ 仍坍缩、仍不动 |
| point_ls 用幅值当捷径吸收频偏 | `--recon_mode marginal` | ❌ marginal 曲线与 point_ls 几乎重合，仍不动 |
| recon 对 δ 平（无梯度）| 曲率探针 2.2 | ❌ 曲率很强 |
| logvar 撞硬钳位 -8 | sigmoid 软钳位（min -6）| ⚠️ 软钳位更健康，但 δ_mu 仍不动（非根因）|
| 优化不充分（lr/epoch 太小）| lr 3e-3、batch 256、40ep | ❌ 仍卡 0.50（后续 1e-2/3e-2 加码确认中）|
| δ 监督预热缺失 | 查 production args | production `local_nll_weight=0`、`z_loss_weight=0`，**纯无监督也没用监督** |

---

## 4. 根因定位

原 production 的「局部 δ 精修」**从未真正生效**：
- `raw 0.162 → 精修 0.169`（归档），`0.142 → 0.216`（对照复跑），δ 均无改进或帮倒忙。
- 0.169 完全是 **greedy 连续搜索 + 局部 LS** 把候选直接做到 0.16 的功劳。

把「连续频率精修」交给一个 **amortized 的 δ 回归头**这条路径本身是脆弱的：
- 它需要输入里含 sub-bin 残差信息；FR 网格候选 + profile_ls 特征清零后没有。
- 即便信息够（greedy 的 alias_hybrid 特征更全），amortized 回归这个任务也没学好（对照 ep1 反而变差）。

而 **非 amortized 的直接优化**（自由参数 / 每样本 recon 梯度 / 局部 LS）能精修到 0.11。

---

## 5. 候选下游修正方向（待决策）

- **方向 A — 把精度做进 proposal（放弃 δ 回归）**：FR 峰位 **sub-bin 插值**（抛物线/重心）
  + 峰邻域**局部 LS/相位细化**，让候选 raw 从 0.50 逼近 greedy 的 0.14。δ 退化或只做极小 polish。
  （greedy 的 0.14 本质就是这么来的。）
- **方向 B — 保留 ELBO，δ 改 test-time 优化**：推理时对每样本直接用 recon 梯度优化 δ
  （如自由参数实验），而非 encoder 一次前向回归。已证明能到 0.11，代价是推理变慢。
- **方向 C（诊断 b 的对症修法）**：给 delta_head 喂能反映 sub-bin 残差的输入特征
  （FR 峰邻域插值得到的连续偏移估计作为特征），再看 amortized δ 能否学动。

---

## 6. 相关文件

- 前端/路线：`comparison/deepfreq_btt_geomrobust.py`、`comparison/run_frfront_elbo.py`
- 判定实验：`comparison/probe_delta_curvature.py`、`comparison/probe_delta_optim.py`
- 对照/扫描：`comparison/run_greedy_control.py`、`comparison/run_stage2_only_ab.py`
- 原 pipeline：`btt_amortized_vi_project/run_legacy_four_component_candidate_vi.py`
  （loss 组装 L1505；cartesian_topk L1295；δ 重参数化 L1464-1468；refined_all=cand+radius·δ L1468）
- production 归档：`.../remote_gpu_archive/c16_production_repulsion/`（freq_mae 0.169，raw 0.162）

## 7. 后续实证更新（2026-06-27 续）

### 7.1 lr 扫描三组（已完成）— 调 lr 救不动 δ
所有组 KL=0、point_ls、sigmoid 软钳位(min -6)、FR 候选：

| lr | batch | ep | TEST mae | raw | top1 |
|---|---|---|---|---|---|
| 3e-3 | 256 | 40 | 0.496 | 0.502 | 1.0 |
| 1e-2 | 256 | 40 | 0.501 | 0.502 | 1.0 |
| 3e-2 | 512 | 60 | **175.9** | 176.6 | **0.25** |

→ lr 偏小：δ 不动（mae=raw）。lr 过大（3e-2）：**选择崩溃**（top1 0.25），δ 仍没动起来。
**瓶颈不是优化**。与「输入缺 t_n」一致。

### 7.2 根因再定位：encoder 看不到真实采样时刻 t_n
查 `legacy_reuse/current_project/dataset.py` L43-53：encoder 的 6 维输入是
`[Re, Im, sinθ, cosθ, rev_norm, speed_norm]`——**有原始信号 y_n，但「时间」只有
rev_norm（整数圈号归一化），真实非均匀采样时刻 t_n 没进特征**（t 单独返回，仅
`--include_local_time` 时才追加 t_local/dt_norm）。我们之前所有 FR/greedy 实验**都没开**
这个开关。

亚 bin 频偏的信息编码在相位 2π·f·t_n 随**真实 t_n** 的演化里。delta_head 手里只有
rev_norm（抹平了非均匀间隔）→ **确实没有原料做亚 bin 定位** → 恒输出 δ=0。这精确解释了
「选 basin 能做对（粗粒度，rev_norm 够）/ δ 精修做不了（需精确 t_n）」。

→ 这是瓶颈定位为 **(b1) 输入缺 t_n** 而非 (b2) amortized 难学 的直接证据。

### 7.3 决定性实验（进行中）：开 --include_local_time 重跑
input_dim 6→8（追加 t_local + dt_norm）。因 input_proj 维度变化，production/旧 Stage1
ckpt 该层不匹配 → 随机初始化，Stage1 重训。判据：
- δ 开始动、freq_mae 跌破 0.50 → 坐实 (b1)，补时间通道即解；
- 仍卡 0.50 → (b2)，转 test-time 优化 / 旋转因子可微模型（神经元=旋转因子、权重=复幅值）。

### 7.4 判决实验结果（已完成）：include_local_time 也救不动 δ —— 推翻 (b1)
开 `--include_local_time`（input_dim 6→8，喂真实采样时刻 t_local + dt_norm），Stage1 重训：

- **Stage1（8 维）选择依然完美**：top1=1.0、basin_hit=1.0 全程 → 加时间通道不破坏 selection。
- **Stage2 δ 依然完全不动**：mae 全程 0.50=raw，dlogv 又坍缩到 -8。TEST mae=0.5006 raw=0.5015。

→ **不是输入缺 t_n（b1 被推翻）。喂了真实采样时刻，amortized q(δ) 仍学不出亚 bin 精修。**
坐实 **(b2)：连续后验 q(δ) 的 amortized 回归这条路本身走不通**。

### 7.5 最终结论（修订 §0）：分层看「分布建模」
完整证据链：

| 实验 | δ 的信息来源 | δ 动了吗 |
|---|---|---|
| 自由参数（非 amortized） | 直接标量 + recon 梯度 | ✅ 0.42→0.11 |
| FR + 默认 6 维特征 | 缺 t_n | ❌ 0.50 |
| FR + 大 lr (3e-3/1e-2) | 缺 t_n | ❌ 0.50 |
| FR + lr 3e-2 | 缺 t_n | ❌ 选择崩(top1 0.25) |
| **FR + include_local_time** | **有 t_n** | ❌ **0.50** |
| greedy / production | alias_hybrid 全特征 | ❌ 0.16→0.17 帮倒忙 |

**分层结论：**
1. **离散分布 q(z) 选 basin：成功**（top1/basin_hit=1.0，各配置稳）。混叠多峰用离散分布
   表达是对的——原始设计的这一半没问题。
2. **连续高斯后验 q(δ|z) 做 amortized 亚 bin 精修：失败，且与输入无关**（喂 t_n 仍不动）。
   amortized「一次前向回归出连续后验」在此任务学不出来。
3. **同样 recon 损失，换 per-sample 点估计 + 连续 BP（非 amortized）→ 立即精修到 0.11**。

→ 不是「ELBO 不能当损失」，而是 **连续频率精修不该用 amortized 变分后验，应改用点估计 +
per-sample 连续优化（MNN 式）**。与文献一致：MIMO(#4) 角度用点估计不用变分；MNN(#1) 用点估计
BP（FFT 找 basin → BP 精修），文献列为 BTT 优先复现第一；PROSAIL(#11) 靠参数 MSE 主监督而非
靠 KL/重构定位参数。

### 7.6 下一步：转 MNN 式可微信号模型（= 用户「旋转因子」想法）
保留已验证 OK 的 **FR-front + q(z) selection** 做 basin proposal / 频率初始化，
**basin 内换成 per-sample 连续优化**：神经元=旋转因子 e^{j2πf t}、权重=复幅值 a_k，
forward = Σ_k a_k e^{j2πf_k t}，loss=‖y-ŷ‖²/σ²(+稀疏先验定 K)。可同时拿到亚 bin 精度
(已侧证 0.11) + 幅值估计 + 未知 K。文献接口见 MNN 8 步流程。

### 7.7 监督 δ 实验（已完成）—— 直接给标准答案也学不动 μ
走【全监督分支】（不开 unsup_elbo，local_nll_weight=1.0），用 target_delta=(true-cand)/radius
直接监督 δ_mu。先验证 target_delta 不是 0：abs_mean=**0.499**、88% 样本 |δ_target|>0.1、
均值≈0（每样本方向随机但量级显著）。结果：

- local_nll 确实在降（0.029→-0.067），dlogv 稳步动（-1.06→-3.79，未坍缩）→ 监督信号进去了、δ 头在学。
- **但 freq_mae 全程 0.497 ≈ raw 0.50，δ 对精度的贡献仅 0.004 Hz。**
- 由 freq_mae 算式 `pred_freq=cand+radius·δ_mu` 反推：**δ_mu 实际输出 ≈ 0**。local_nll 的下降
  几乎全来自 gaussian_nll 的 logvar 项，μ 没学到 target。

→ **即便直接监督，amortized δ_mu 也回归不出那个标量偏移。** 这是 (b2) 的最强证据：
不是无监督信号弱，是 amortized 回归这个亚 bin 偏移的任务本身学不出来。

### 7.8 greedy 对照完整轨迹（已完成）
production 配置 unsup ELBO，原生 greedy 候选：mae 0.2156→0.1460，**始终 ≥ raw 0.1424**。
δ 从「帮倒忙」被拉回到「约等于不动」，从未把候选精度往下压。与 production 归档（0.162→0.169）一致。

## 8. 最终定论
**离散选择 q(z) 成功；连续亚 bin 精修的 amortized 回归（无论无监督 recon、还是直接监督
target_delta）全部失败，δ_mu 恒≈0。** 同样损失换 per-sample 点估计 BP 立即到 0.11。
→ **放弃 amortized δ 头，转 MNN 式 per-sample 连续优化（神经元=旋转因子、权重=复幅值）。**

DeepFreq 基准对照（c16@20dB，同协议）：DeepFreq-grid 0.495、DeepFreq+local-LS 0.273；
均靠「峰位 + LS 细化」而非网络回归亚 bin，且都输给 LSF 0.127 / amortized_vi 0.136；DeepFreq 不估幅值。
我们 FR 路线现卡在 0.50 = DeepFreq-grid 水平。MNN 目标：越过 local-LS 0.273，逼近 LSF 0.13 / CRB 0.099。
（DeepFreq 官方前端的转速 ramp 失效待补测；nudft FR 已确认转速鲁棒：recall 全 1.0、MAE 0.50-0.65 平稳。）

- [x] greedy 对照完整轨迹（0.216→0.146≥raw，δ 不压精度）
- [x] include_local_time 判决：救不动 δ
- [x] 监督 δ 判决：直接监督 δ_mu 仍≈0
- [ ] MNN 原型：FR-basin 初始化 → LS 初始幅值 → Adam 联合精修 → 剪枝定阶
- [ ] DeepFreq 官方前端转速 ramp 扫描（补测，对照 nudft/naive4ch）
