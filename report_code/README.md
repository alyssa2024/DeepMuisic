# DeepMUSIC

BTT（叶尖定时）非均匀采样下的多分量频率估计。两阶段：**编码器四阶段训练**（网络）
+ **逐实例变分精修**（推断期优化，不训练网络）。含推理、评测与绘图入口。

## 依赖

Python 3 + PyTorch + NumPy。TensorBoard 可选（缺失时自动跳过，不影响训练）。

## 目录结构

```
report_code/
├── config.py               全部超参
├── synthesis_dataset.py    信号合成（探针几何/转速/相位积分/加噪）
├── dataset.py              单窗数据集（阶段 1-2）
├── crosswin_data.py        跨窗序列数据集（阶段 3-4、推理）
├── encoder.py              FTF 单窗编码器（NUDFT + 频率轴 attention + 三头）
├── axial_vit.py            轴向 ViT（频率轴 / 逐 bin 时间轴交替）
├── window_fusion.py        窗融合（沿 M 平均，无参）
├── loss.py                 anchor 三头损失 + 跨窗包装
├── elbo_marginal.py        解析边际似然 ELBO
├── peaks.py                取峰（NMS + top-L）
├── scheduler.py            lr 调度
├── train_report.py         四阶段共用报告器
│
├── train_freq_axis.py      阶段 1-2  ┐
├── train_time_axis.py      阶段 3    │ 训练入口（窗融合无参，无训练脚本）
├── calib_time_axis.py      阶段 4    ┘
├── infer.py                第二阶段端到端推理（不读真值）
│
├── eval/                   评测（均需真值）
│   ├── matching.py             ⭐统一口径：覆盖率 + 匈牙利一对一
│   ├── crb.py                  多正弦 CRB
│   ├── sweep_snr_elbo.py       本文方法六档 SNR
│   ├── sweep_snr_recall.py     编码器 R@L / allK@L（六档 SNR × 多个 L，不精修）
│   ├── run_elbo_marginal.py    本文方法单档（init vs 精修）
│   ├── scan_classical_snr.py   ┐
│   ├── scan_classical_nrev.py  │ 四组扫描：五种传统法 + 本文方法
│   ├── scan_classical_fluct.py │ 共用 matching.py
│   └── scan_classical_sep.py   ┘
│
└── plot/
    └── plot_spectrum.py    幅值谱图：变分前 vs 变分后
```

子目录脚本从项目根目录调用即可，它们自行把上级加入 `sys.path`：

```bash
python eval/scan_classical_snr.py ...
python plot/plot_spectrum.py ...
```

**第一阶段：编码器训练**

| 脚本 | 阶段 | 训练部件 |
|---|---|---|
| `train_freq_axis.py` | 1 & 2 | 1: 主干 + obj/δ 头；2: 仅 σ 头（129 参数） |
| `train_time_axis.py` | 3 | 仅时间轴层（频率轴冻结） |
| `calib_time_axis.py` | 4 | 仅共享 σ 头 |

窗融合（沿 M 平均）**无参数**，不占训练阶段：三头直接复用逐窗三头。

**第二阶段：推理**

| 脚本 | 作用 |
|---|---|
| `infer.py` | **端到端推理**：序列 → K 个频率/幅值及各自不确定度。全程不读真值 |

**评测 `eval/`**（均需真值）

| 脚本 | 作用 |
|---|---|
| `matching.py` | **统一评估口径**：覆盖率（不配对）+ 匈牙利一对一。四个扫描脚本共用 |
| `sweep_snr_elbo.py` | 本文方法六档 SNR 扫描：分位数 + 幅值分层 + 假峰比 |
| `sweep_snr_recall.py` | **编码器**六档 SNR × 多个 L 的 R@L / allK@L（**不精修**，只测网络候选质量）；三路 agg / sum / mean 口径分别对齐 `metrics.json` 与 `metrics_mean_fusion.json` |
| `run_elbo_marginal.py` | 本文方法单档快评（init vs 精修对照） |
| `scan_classical_snr.py` | 传统法 + 本文方法 vs **SNR**（恒速档） |
| `scan_classical_nrev.py` | 传统法 + 本文方法 vs **序列总转数**（孔径，恒速档） |
| `scan_classical_fluct.py` | 传统法 + 本文方法 vs **转速波动**（重点组） |
| `scan_classical_sep.py` | 传统法 + 本文方法 vs **最小频率间隔**（K=2，恒速档） |
| `recompute_recall.py` | 已训 ckpt 的 recall/allK 重算：**bin 中心 vs 亚-bin** 口径对照（不重训） |
| `RESULTS.md` | **评测结果留档**：六档 SNR × 1000 序列，精修 vs 纯初值完整对照 |
| `crb.py` | 多正弦 CRB |

传统法五种：Block-OMP / ADMM / SPICE / NUFT / MUSIC。四个扫描脚本**都**可加本文方法
（`NN_AXIAL=<axial.pt>`），与传统法进同一张表、同一套口径。

**绘图 `plot/`**

| 脚本 | 作用 |
|---|---|
| `plot_spectrum.py` | 幅值谱图：变分**前** vs **后**，真值黑 ✕、估计蓝 stem、标题带 hit K/K |

**模块**

| 文件 | 内容 |
|---|---|
| `config.py` | 全部超参；`grid_size` / `signal_dim` / `resolve_probe_angles` |
| `synthesis_dataset.py` | 信号合成：探针几何、逐圈转速、相位积分、加噪 |
| `dataset.py` | 单窗数据集（阶段 1/2）+ 转速摘要 |
| `crosswin_data.py` | 跨窗序列数据集（阶段 3-4、推理） |
| `encoder.py` | FTF 单窗编码器：显式 NUDFT + 频率轴 self-attention + 三头 |
| `axial_vit.py` | 轴向 ViT：频率轴层与逐 bin 时间轴层交替堆叠 |
| `window_fusion.py` | 窗融合 `WindowFusionMean`（沿 M 平均，无参，复用逐窗三头） |
| `loss.py` | anchor 三头损失 + 跨窗聚合器包装 |
| `elbo_marginal.py` | 解析边际似然 ELBO（幅值闭式积分掉，含 `logdet(S)`） |
| `peaks.py` | 取峰：局部极大（NMS）+ top-L |
| `scheduler.py` | 前 N 步恒定 lr、之后 cosine 退火 |
| `train_report.py` | 四阶段共用：log / train_log.json / TB / 选模型 / metrics.json |

## 运行

### 第一阶段：四阶段串行

每一步的输入是上一步的 `best.pt`：

```bash
# 阶段 1：频率轴 phase 1（从零训）
python train_freq_axis.py --stage stage1

# 阶段 2：频率轴 phase 2（σ 校准）
python train_freq_axis.py --stage stage2 --stage1_ckpt artifacts/ftf_stage1/best.pt

# 阶段 3：时间轴层（频率轴冻结）
python train_time_axis.py artifacts/ftf_stage2/best.pt artifacts/ftf_stage2/config.json

# 阶段 4：时间轴 σ 校准
python calib_time_axis.py artifacts/axial_vit_frozenfreq/best.pt \
                          artifacts/axial_vit_frozenfreq/config.json

```

阶段 4 的 `artifacts/axial_sigcalib/best.pt` 即最终编码器权重 —— 窗融合是沿 M 的代数
平均（无参、无需训练），推理时直接在其上做。

输出目录由 `config.CONFIG["checkpoint"]["dir"]` 决定（默认 `artifacts/`）。

**环境变量**

| 变量 | 默认 | 作用 |
|---|---|---|
| `FREEZE_FREQ` | `1` | 阶段三冻结频率轴；`0` 退回端到端联合微调（消融用） |

> ⚠ Windows/CPU 上跑需把 `config.CONFIG["training"]["num_workers"]` 设为 `0`，
> 否则 DataLoader worker 会崩。

### 第二阶段：推理

```bash
python infer.py <axial_ckpt> <config.json> [n_seq] [snr_db] [steps]

# 例：用阶段四的权重，跑 8 条序列
python infer.py artifacts/axial_sigcalib/best.pt \
                artifacts/axial_sigcalib/config.json 8 20 600
```

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `N_SAMPLE` | `16` | MC 重参数样本数 R |
| `N_EXTRA` | `2` | 候选数 $K_c = K + M_{\rm extra}$ |
| `SEED` | `999901` | 数据 seed（须 out-of-training） |

输出每条序列的 K 个估计并落盘 `inference.json`：

```
   k     f (Hz)   σ_f (Hz)       |c|       σ_c
   0    123.456     0.0210    0.8134    0.0142
```

四个量对应频率 $\hat f_k$、频率不确定度 $\sigma_k$、幅值 $|\hat c_k|$、
幅值不确定度 $\sqrt{[S_0^{-1}]_{kk}}$。

`infer.py` **全程不读真值**（`f_win` / `camp_re` / `camp_im`），且 $\sigma_n^2$ 是自由变量，
所以同一套代码也能跑真实 BTT 数据——只需提供 $(y, t_n)$。

### 评测

`eval/sweep_snr_elbo.py` 复用 `infer.py` 的同一套 pipeline 函数（`encode_candidates` /
`refine` / `select_topk`），只是额外读真值算指标，因此推理与评测口径不会漂移。

```bash
python eval/sweep_snr_elbo.py <axial_ckpt> <config.json> \
                         [n_inst] [snrs] [steps] [chunk]

# 例：每档 500 实例，六档 SNR，分块 128
python eval/sweep_snr_elbo.py artifacts/axial_sigcalib/best.pt \
                         artifacts/axial_sigcalib/config.json \
                         500 -5,0,5,10,20,30 600 128
```

每档报告的指标：

| 指标 | 含义 |
|---|---|
| 候选召回 | $K_c$ 个候选覆盖了多少真频（编码器给的候选够不够） |
| 筛出命中 | 按 $\|\hat c\|$ 筛到 $K$ 个之后还剩多少 |
| μ_med / p90 / p99 / RMSE_f | 频率误差的分位数与均方根 |
| CRB、/CRB | 逐实例 CRB 及比值（恰达 CRB = 0.674×） |
| 强/CRB、弱/CRB | 按真值幅度上/下 1/3 分位分层，看精度是否被弱分量拖累 |
| σ覆盖、σ-err、corr | σ 校准三件套（理想覆盖 68%，σ-err → 1） |
| 幅值、RMSE_a、σ_c | 去全局尺度后的相对幅值误差与幅值不确定度 |
| 假峰比 | median(真峰 $\|\hat c\|$) / median(假峰 $\|\hat c\|$)，越大越好 |

> ⚠ **必须同时看分位数与 RMSE**：误差分布重尾（选错混叠 band 是量级跳变），
> 只看 median 会掩盖尾部，只看 RMSE 会被尾部主导。
>
> ⚠ 配对用**匈牙利一对一**（全局最小总距离），再按 τ 筛；覆盖率则不配对。
> 两者的差值 = 多个真频挤向同一估计的量。
>
> ⚠ 边际似然没有 L1 稀疏化，假峰全靠 $\|\hat c_k\|$ 判别 —— **"筛出命中"这一列
> 专门检验这一点**，它与"候选召回"的差距就是这一步丢掉的量。

`eval/run_elbo_marginal.py` 是单档版，额外给出 init（未精修）与精修后的对照。

#### 只测编码器：R@L / allK@L

`eval/sweep_snr_recall.py` **不做第二阶段精修**，只跑网络前向 + 取峰，量的是编码器
给出的候选够不够好——即"$L$ 个候选覆盖了多少真频"。用来定 $K_c$（候选数）该取几。

```bash
python eval/sweep_snr_recall.py <axial_ckpt> <config.json> \
                                [n_eval] [snrs] [Ls] [batch]

# 例：每档 256 序列，六档 SNR，L 扫 6/8/10
python eval/sweep_snr_recall.py artifacts/axial_sigcalib/best.pt \
                                artifacts/axial_sigcalib/config.json \
                                256 -5,0,5,10,20,30 6,8,10 32
```

三路，口径**逐行对齐**既有的两个 metric 脚本：

| 路 | 谱 | 对齐对象 |
|---|---|---|
| `agg` | 逐窗 $\sigma(\text{obj\_logit})$ | `calib_time_axis.py` 的 model 路 → `metrics.json` 的 `recall`/`allk` |
| `sum` | `obj0` 沿 $M$ 求和后逐窗取峰 | `calib_time_axis.py` 的 sum 路 → `metrics.json` 的 `sum_recall`/`sum_allk` |
| `mean` | `obj_logit`/`delta`/`log_sigma` 三者沿 $M$ 平均成单一全局后验 | `eval_fusion_mean.py` 的 `perwin` 模式 → `metrics_mean_fusion.json` |

脚本末尾自带**核验表**，把 $L=K$ 时三路与两个 json 的差值逐档打出来。实测 `mean` 路
全六档差 `0.00000`（逐比特一致）；`agg`/`sum` 残留 $\le 0.004$，相当于 3072 个窗里约
13 个窗在 $\tau=2$ Hz 边界上判定翻转，容差设为 `2e-3`。

> ⚠ 峰频取**亚-bin** $\mu=\text{grid}[peak]+\delta[peak]\cdot\Delta$，与两个 metric 脚本一致。
>
> ⚠ recall/allK 是**逐 batch 算完再对 batch 取平均**（不是全局池化）——照抄原脚本，
> 否则对不上（batch=32 时二者差 ~0.1 点）。
>
> ⚠ `agg`/`sum` 用逐窗真值 `f_win` 全部展平 $[B\!\cdot\!M, K]$；`mean` 融合成单一后验，
> 用 `f_win[:,0,:]`（恒频下 $M$ 行相同），每序列判一次 —— 分母不同，不可混谈。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `SEED` | `999000` | 数据 seed（对齐两个 metric 脚本，非 `infer.py` 的 999901） |

### 与传统方法对比

四组扫描共用 `eval/matching.py` 的口径，传统法（Block-OMP / ADMM / SPICE / NUFT /
MUSIC）与本文方法进同一张表：

```bash
# vs SNR（恒速档 —— MUSIC 唯一可用的设定）
python eval/scan_classical_snr.py

# vs 孔径（序列总转数）
python eval/scan_classical_nrev.py

# vs 转速波动（重点组：变速下协方差类方法失效）
python eval/scan_classical_fluct.py

# vs 最小频率间隔（K=2）
python eval/scan_classical_sep.py
```

加上本文方法这一列：

```bash
NN_AXIAL=artifacts/axial_sigcalib/best.pt \
  python eval/scan_classical_fluct.py
```

| 环境变量 | 作用 |
|---|---|
| `N_SEQ` | 每档序列数（默认 32） |
| `SNRS` / `NREVS` / `FLUCTS` / `SEPS` | 覆盖对应脚本的扫描档位 |
| `NN_AXIAL` | 给了才把本文方法纳入（`NN_STEPS` 控精修步数） |

> ⚠ 传统法逐序列吃 $(y,t_n)$，本文方法要整个 batch（切窗 + FiLM 转速摘要）。
> 脚本按 batch 前置算好 NN 结果、循环里查表，**评估口径与传统法逐字节相同**。
>
> ⚠ `scan_classical_sep.py` 如何精确控制间隔：K=2 时把采样带压到宽度 $w=1.15\,\Delta f$，
> 带内采两个频率并要求间隔 $\ge \Delta f$，实际间隔必落在 $[\Delta f,\ w]$。带位置随 seed
> 变，故 $f_0$ 仍覆盖全频带，只有**间隔**被钉住 —— 数据仍走 `CrossWinDataset`，本文方法可接入。

### 绘图

```bash
python plot/plot_spectrum.py <axial_ckpt> <config.json> \
                             [seq_idx] [snr_db] [steps]
```

两行幅值谱：变分**前**（编码器读出）与变分**后**（边际 ELBO 精修）。真值黑 ✕、
估计蓝 stem、误差棒为 $\mu\pm\sigma$（传统法给不了的量）、灰色为被 $M_{\rm extra}$
丢弃的候选，标题给 hit $K$/$K$。幅值统一用 LS 反解，与传统法定幅口径一致。

| 环境变量 | 作用 |
|---|---|
| `OUT` | 输出路径（默认与 axial_ckpt 同目录） |
| `XLIM` | 局部放大，如 `XLIM=150,250` |

## 训练配置

| 阶段 | 步数 | batch | lr | 退火起点 | 验证批数 |
|---|---|---|---|---|---|
| 1 频率轴 phase1 | 60 000 | 256 | 3e-4 | 40 000 | 8 |
| 2 频率轴 phase2 | 20 000 | 256 | 3e-4 | 6 000 | 8 |
| 3 时间轴 | 8 000 | 32 | 1e-3 | 4 800 | 8 |
| 4 时间轴校准 | 20 000 | 32 | 3e-4 | — | 16 |

优化器一律 AdamW，**不做梯度裁剪**。batch 分两档：频率轴单窗 256，跨窗 32
（一条序列含 M=12 个窗）。

**各阶段参数量与训练耗时**（单卡 RTX 4080 Super；耗时**只含训练**，不含每阶段收尾
的分档评估，后者各约 7 s）：

| 阶段 | 解冻部件 | 可训参数 | 训练耗时 |
|---|---|---|---|
| 1 频率轴 phase1 | 主干 + obj/δ 头 | 579 906 | 76.5 min |
| 2 频率轴 phase2 | σ 头 | 129 | 8.6 min |
| 3 时间轴 | 时间轴层 | 529 920 | 33.9 min |
| 4 时间轴校准 | σ 头 | 129 | 51.2 min |
| **合计** | | **1 110 084** | **2.83 h** |

窗融合无参数、不训练，故不占阶段。

> 两个校准阶段各只训 **129** 个参数（σ 头 = `Linear(128,1)`），却占了总耗时的
> 相当部分——因为它们跑 20 000 步且是跨窗（每步前向 M=12 个窗），耗时由**步数×前向代价**
> 决定，与可训参数量无关。

第二阶段：每条序列 Adam **固定学习率**跑 600 步（无退火），R=16 个 MC 样本。
三组学习率——$\mu_k$ 用 3e-2，$\log\sigma_k$ 用 3e-1（收敛慢，取 10 倍），
$\tau_k^2$ 与 $\sigma_n^2$ 用 3e-2（均在对数域优化）。

## 数据

数据**全程在线现采**，每条样本一个独立 seed 且只用一次，无固定训练集可过拟合；
验证 seed 与训练 seed 不相交。

| 项 | 值 |
|---|---|
| 探头 | 8 个，周向角 `[0, 9.47, 29.00, 43.12, 70.08, 80.66, 94.06, 112.54]°` |
| 转速 | `f_rot = f_0(1 + ρ·r + Σηᵢ)`，`f_0 ~ U[25,45]` Hz，`ρ=1.5e-3`，`ς=3e-3` |
| 频漂 | **关闭**（`freq_drift=False`，全流程恒频）；转速波动仍在，两者是不同的量 |
| 分量 | K=6，最小间隔 6 Hz |
| 频率先验 | `U[5, 495]` Hz |
| 幅值 | `(|w| + 0.1)·e^{jφ}`，`w ~ N(0,1)`，`φ ~ U[0,2π)` |
| 网格 | `[0, 500]` Hz，Δ=2 Hz ⇒ G=251 |
| 序列 | 104 转，窗长 16 转，hop 8（50% 重叠）⇒ M=12 |
| SNR | `U[0,50]` dB，按**最弱分量**定义 `SNR = 10log₁₀(min_k α_k² / σ_n²)` |

## 输出

每个训练阶段在结果目录写：

| 文件 | 内容 |
|---|---|
| `config.json` | 开训时落盘的完整 config（产物目录自解释） |
| `train_log.json` | 逐评测点的 loss 分项与 metric，**训练中实时更新** |
| `last.pt` / `best.pt` | 最新 / 最优权重 |
| `metrics.json` | 收尾汇总，含 `eval_snr_db` 各档的**分档评估** |

评测脚本另在 **ckpt 所在目录**写各自的结果 json（都可直接被绘图脚本读）：

| 文件 | 由谁写 | 内容 |
|---|---|---|
| `sweep_snr_recall.json` | `eval/sweep_snr_recall.py` | 编码器 R@L / allK@L：`rows[i]` 含 `rec_{agg,sum,mean}_L{6,8,10}` 与对应 `allk_*`；`cfg.aligned_with` 记录三路各自对齐的 metric 文件 |
| `sweep_snr_elbo.json` | `eval/sweep_snr_elbo.py` | 精修后全指标：召回 / 分位数 / CRB 比 / σ 校准 / 幅值 / 假峰比 |
| `elbo_marginal_result.json` | `eval/run_elbo_marginal.py` | 单档 init vs 精修对照 |
| `metrics_mean_fusion.json` | `eval_fusion_mean.py` | mean 融合（无参）的分档 recall/allK 与 σ 校准 |
| `inference.json` | `infer.py` | 每条序列的 K 个 $(\hat f,\sigma_f,|\hat c|,\sigma_c)$（**不含真值**） |

**训练期打印**：train/val 的总 loss 与每一分项成对成列（loss 只有一项时不重复列出），
model 路与 sum 路两个 recall/allK；训方差头的阶段（2/4/6）另加 σcorr 与 σ/err。
loss 均为 200 步滑动均值。单窗阶段无 sum 路，该列显示 `nan`。

**recall 的取峰口径**：峰频率取**亚-bin** `μ = grid[peak] + δ[peak]·Δ`，**不是 bin 中心**
`grid[peak]`——与 `infer.py` / `eval/` 同口径，量的是第二阶段拿到的那个初值。
> 容差 `basin_hz = 2.0` 恰好等于网格 Δ=2 Hz，bin 中心到它所含真频最远只有 1 Hz、本来就在容差内，
> 所以两种口径的 R/allK 实测只差 `<0.001`（`eval/recompute_recall.py` 可复现该对照）。
> δ 影响的是频率误差，不是 recall。

**选模型判据**：阶段 1/3/5 按 `val_recall@K@20dB` 最大；阶段 2/4/6 按 `val_loss` 最小。
校准阶段点估计冻结、recall 按构造是常量，用 recall 选会让 `best.pt` 永远停在第 1 步。
sum 路只进日志，不参与选模型。

**两个 SNR 的分工**：`monitor_snr_db`（20 dB）用于训练期每个评测点的 recall，与历史数字
同口径；`eval_snr_db`（六档）用于训练结束的分档评估；val_loss 走 `train_snr_choices`
宽 SNR，与训练同分布才可比。
