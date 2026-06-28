# DeepMUSIC 可行性验证与 BTT 分层重构实施方案

## 0. 总目标

本项目必须回答两个问题。

### 任务一：验证 DeepMUSIC 核心频率后验思想

在标准、均匀采样、满足 Nyquist 条件、模型阶数 (K) 已知的多正弦估计问题中，验证：

[
\boxed{
\text{验证对象不是原始全参数实现，而是其最小可行频率后验版本}
}
]

不再验证原始 DeepMUSIC 的全参数神经后验是否成立。验证对象改为：

```text
encoder 基于观测输出频率后验，
并通过固定物理生成模型和变分目标训练。
```

其中幅值和相位不再作为神经后验，而是解析消元。必须回答：

1. 单分量情况下，摊销频率后验 + 幅值边缘化 ELBO 是否可以训练；
2. 多分量情况下，失败是否来自 factorized posterior 无法表达联合频率相关性；
3. 有序全协方差后验是否能改善近邻多分量频率估计；
4. 若频率后验版本仍失败，明确失败来自：

   * 置换对称；
   * posterior parameterization；
   * 幅值边缘化目标或矩阵求解；
   * posterior variance 坍缩；
   * amortization gap；
   * 或 encoder 无法提取频率证据。

最终结论只分为两类：

* 通用场景下成立：说明摊销频率后验思想可行；
* 通用场景下失败：分析频率后验、训练目标、置换对称和 amortization 的问题。

通用估计器只用于方法可行性验证，不用于模拟 BTT，也不考虑 BTT 的探针布局、亚采样、转速和 Campbell 结构。

### 任务二：建立可用的 BTT 方案

针对 BTT 的非均匀、严重亚采样和 alias 多 basin 特性，建立：

[
\text{物理候选生成}
\rightarrow
\text{摊销离散候选选择}
\rightarrow
\text{逐实例 marginal/profile-MAP + GN 连续精修}
\rightarrow
\text{局部协方差}
]

该方案尽量保留 DeepMUSIC 的：

* 固定物理生成模型；
* encoder 聚合全局观测证据；
* 数据依赖的近似后验输出。

但明确不再使用已经失败的“一次前向连续频率精修头”。

---

# 1. 分支与代码修改原则

## 1.1 基础分支

从当前 `bayesianLS` 分支创建新分支，例如：

```text
deepmusic_validation_and_btt
```

不得直接破坏 `bayesianLS` 原有可运行路径。

## 1.2 目录结构

新增以下相互隔离的实验入口：

```text
experiments/
    general_v0_single_component.py
    general_v1_diag_ordered_gap.py
    general_v2_fullcov_ordered_gap.py
    general_semamortized_eval.py
    btt_oracle_continuous_refinement.py
    btt_candidate_encoder.py
    btt_full_pipeline.py
```

配置拆分为：

```text
configs/
    general_v0.py
    general_v1.py
    general_v2.py
    btt_oracle_refinement.py
    btt_candidate_encoder.py
    btt_full_pipeline.py
```

共用代码放入：

```text
models/
    general_encoder.py
    ordered_gap_posterior_head.py
    btt_candidate_encoder.py

data/
    general_sinusoid_dataset.py
    btt_synthetic_dataset.py

losses/
    marginal_frequency_elbo_loss.py
    candidate_ce_loss.py

solvers/
    complex_marginal_likelihood.py
    complex_profile_map.py
    frequency_gn.py
    conditional_alternating_gn.py

metrics/
    frequency_metrics.py
    calibration_metrics.py
    basin_metrics.py
```

不得把 V0、V1、V2、BTT 四类目标写入同一个复杂训练脚本。

## 1.3 禁止 Codex 擅自增加的内容

除非本方案明确要求，否则不得增加：

* normalizing flow；
* mixture density network；
* diffusion；
* 未知 (K) 的 existence head；
* Hungarian 训练损失；
* teacher forcing；
* 额外监督频率回归损失；
* 额外频谱图标签；
* 额外预训练阶段；
* 自适应动态 loss weighting；
* 多阶段 curriculum；
* 新的外部依赖库；
* 修改任务定义以提高指标。

如某项无法实现，应记录原因，不得自行替换为其他方法。

---

# 2. 共用信号模型与输入格式

## 2.1 复多正弦模型

所有通用实验采用：

[
y_n
===

\sum_{k=1}^{K}
c_k e^{j2\pi f_k t_n}
+
w_n,
\qquad
w_n\sim\mathcal{CN}(0,\sigma_w^2).
]

其中：

[
c_k=A_ke^{j\phi_k}.
]

第一轮数字频率定义在圆周区间：

[
f_k\in[0,1).
]

等价地也可使用 ([-1/2,1/2))，但第一轮实现优先使用 ([0,1))，以简化周期边界处理。

## 2.2 输入必须包含观测与采样时间

即使通用数据为均匀采样，encoder 输入也必须显式包含：

[
(y_n,t_n).
]

每个 token 使用：

[
x_n=
[
\operatorname{Re}(y_n),
\operatorname{Im}(y_n),
\tilde t_n,
\Delta\tilde t_n
].
]

其中：

[
\tilde t_n
==========

\frac{t_n-t_0}{t_{N-1}-t_0+\epsilon},
]

[
\Delta\tilde t_n
================

\begin{cases}
0,&n=0,\
\tilde t_n-\tilde t_{n-1},&n>0.
\end{cases}
]

当前通用实验中 (t_n=n)，但数据接口必须允许未来传入非均匀时间。

不得只将数组索引作为隐式位置并删除时间特征。

## 2.3 输出语义与 ordered gap 后验

多频率第一版不得使用无序 factorized frequency slots。通用估计器的频率后验定义在 ordered gap latent 上：

[
q_\theta(g\mid y)
=================

\mathcal N
\left(
g;\mu_g(y),\Sigma_g(y)
\right),
\qquad
g\in\mathbb R^{K+1}.
]

其中 (g) 是无约束 gap latent。采样必须使用重参数化：

[
g^{(s)}
=======

\mu_g+L_g\epsilon^{(s)},
\qquad
\epsilon^{(s)}\sim\mathcal N(0,I).
]

通过 ordered transform 得到频率：

[
f_1<f_2<\cdots<f_K.
]

定义：

[
d_i=\operatorname{softplus}(g_i)+\epsilon,
\qquad i=1,\ldots,K+1.
]

累积映射：

[
f_k
===

f_{\min}
+
(f_{\max}-f_{\min})
\frac{\sum_{i=1}^{k}d_i}
{\sum_{i=1}^{K+1}d_i}.
]

这样同时保证：

* slot 语义来自有序 gap，而不是标签排序；
* 所有频率严格落在 ([f_{\min},f_{\max}]) 内；
* 左右边界保留尾部间隔；
* 多分量后验可以通过 (\Sigma_g) 表达相关性；
* 训练可重参数化采样。

第一版只允许两种 posterior covariance：

```python
posterior_covariance = "diagonal"   # V1 baseline
posterior_covariance = "full"       # V2 main comparison
```

不得在第一版引入 flow、mixture posterior 或其他更强分布。只有当全协方差 ordered gap 高斯在近邻扫描中明显不足时，才考虑后续更强分布。

KL 在 gap latent 空间定义。第一版使用宽高斯 gap prior：

[
p(g)=\mathcal N(0,\sigma_g^2I).
]

配置：

```python
posterior_parameterization = "ordered_gap_gaussian"
prior_space = "gap_latent"
gap_prior_std = 2.0
```

这意味着频率先验不是严格 full-band uniform，而是由 gap transform 诱导出的弱有序先验。
不得再声称第一版 KL 精确复刻数据生成先验。

训练必须从后验采样：

```python
use_posterior_mean_only = False
num_posterior_samples = 8
```

推理时可以同时报告 `posterior_mean_estimate`、`posterior_sample_mean_estimate` 与
`posterior_map_or_mode_estimate`，但训练不得退化成只把 (\mu_\theta(y)) 送进 likelihood。

若采样训练数值不稳定，应修复重参数化、边界参数化、矩阵求解或梯度尺度；
不得自行改成均值路径。

---

# 3. 通用数据集

## 3.1 基本设定

通用数据集合成必须参考 DeepFreq 官方实现：

```text
comparison/deepfreq_official/data/data.py
comparison/deepfreq_official/data/noise.py
comparison/deepfreq_official/generate_dataset.py
```

实现目标是 DeepFreq-compatible，而不是重新发明一套不同的数据先验。除本方案明确要求的 fixed-K 与
K 递进外，频率、幅值、相位、归一化和加噪语义应尽量与 DeepFreq 的 `gen_signal` / `noise_torch`
保持一致。

* 均匀采样；
* Nyquist 区间内；
* (K) 固定且已知；
* 每条样本独立抽取参数；
* 训练、验证、测试按参数实例划分；
* 不允许将同一固定参数长序列切成窗口后分到 train/test。

第一阶段也是在多参数数据集上训练，不是单个固定参数实例训练，也不是围绕某个频率中心的小范围训练。
训练集、验证集、测试集均由独立参数实例组成。

每个样本独立生成：

[
f_1,\ldots,f_K,
\quad
A_1,\ldots,A_K,
\quad
\phi_1,\ldots,\phi_K,
\quad
\sigma_w^2.
]

因此数据先验是多参数覆盖；条件后验 (q(g\mid y)) 在 ordered gap latent 空间中采用高斯近似；
模型先验 (p(g)) 只作为 ELBO 正则，不等同于完整 data prior。

## 3.2 频率生成

频率不需要为了幅值边缘化而选择特殊共轭分布。幅值能否解析积分掉只取决于
(c\sim\mathcal{CN}(0,\Sigma_c))，与频率先验无关。频率分布应根据通用可行性验证任务设计。

第一轮使用归一化圆周频率：

[
f\in[0,1).
]

单分量时：

[
p(f)=\mathcal U(0,1).
]

多分量时，不使用“独立均匀后排序”作为概念定义，而是在满足有序和最小循环间距约束的频率集合上均匀分布：

[
p_{\mathrm{data}}(f_{1:K})
\propto
\mathbf 1
\left[
0\leq f_1<\cdots<f_K<1
\right]
\prod_{i<j}
\mathbf 1
\left[
d_{\mathrm{circ}}(f_i,f_j)\geq\Delta_{\min}
\right].
]

即：

[
\boxed{
\text{在满足有序和最小间距约束的频率集合上均匀分布}
}
]

第一阶段按 (K) 递进执行：

```python
N = 64
frequency_domain = [0.0, 1.0]
K_schedule = [1, 2, 3, 4, 5, 6]
frequency_distribution = "uniform_ordered_with_min_separation"
circular_distance = True
default_min_separation = 2.0 / N
variable_num_freq = False
```

数据生成使用拒绝采样：

1. 在 ([0,1)) 均匀采样 (K) 个频率；
2. 按升序排列；
3. 检查循环最小间距；
4. 不满足则重新采样。

执行顺序固定为：

1. 先做 (K=1)，验证数据、encoder、幅值边缘化 ELBO 与评估链路；
2. 再做 (K=2)，重点扫描最小间隔；
3. 最后做 (K=3,4,5,6)，先使用 `default_min_separation=2.0/N`，确认多分量扩展是否稳定。

(K=2) 最小间隔扫描：

```python
min_separation_over_rayleigh = [0.5, 1.0, 1.5, 2.0, 3.0]
```

其中：

[
\Delta_{\min}
=============

\text{min_separation_over_rayleigh}/N.
]

(K=3,4,5,6) 的间隔扫描放到 Batch 2，不混入 Batch 1 的第一轮验收。

DeepFreq 式近邻间距分布不进入第一轮。它更强调近分辨极限样本，适合作为第二阶段泛化测试：

[
\Delta f
========

\frac1N+
\left|
\mathcal N
\left(
0,\left(\frac{2.5}{N}\right)^2
\right)
\right|.
]

第一轮必须保持固定且相对宽松的 (\Delta_{\min}=2/N)，以便失败时能区分 posterior 表示问题、
训练问题和分辨极限问题。

必须明确区分：

* 数据生成先验：全频带、有序、最小间距约束下的均匀集合分布；
* 变分 posterior：ordered gap latent 高斯 (q_\theta(g\mid y))；
* KL 中的模型先验：宽高斯 gap prior (p(g)=\mathcal N(0,\sigma_g^2I))。

gap prior 只是弱正则，不声称等同于严格均匀频率先验。

## 3.3 复幅值、幅值与相位

幅值和相位的第一轮数据分布必须与幅值边缘化模型先验一致。

直接采样复幅值：

[
c_k\sim\mathcal{CN}(0,\tau_k^2).
]

等价地：

[
A_k\sim\operatorname{Rayleigh}\left(\frac{\tau_k}{\sqrt2}\right),
\qquad
\phi_k\sim\mathcal U(-\pi,\pi),
\qquad
A_k\perp\phi_k.
]

第一轮使用各向同性同方差复幅值先验：

[
\Sigma_c=\tau_c^2 I.
]

配置：

```python
complex_amplitude_distribution = "complex_gaussian"
amplitude_prior_covariance = "isotropic"
tau_c = 1.0
phase_min = -pi
phase_max = pi
normalize_clean_signal_rms = True
```

这既保证复幅值能被闭式边缘化，也使数据分布与模型幅值先验完全一致。

DeepFreq 只作为频率生成、RMS 归一化、加噪和数据接口风格的参考；其 `uniform`、`normal_floor`
或 `alternating` 幅值分布不进入 Batch 1 主线。若后续作为 robustness 对照，必须单独报告：

```python
amplitude_distribution = "deepfreq_normal_floor"  # optional robustness only
```

后续弱分量实验可单独配置，但不得混入第一轮验收。

> **✅ 弱分量实测(2026-06-27,`experiments/btt_weak_component.py`,oracle K=2,spacing 40Hz,20dB):**
> 强+弱分量扫幅值比的连续精修 + 幅值 LS 结果:
>
> | amp_ratio | f_weak MAE | a_weak relerr |
> |---|---|---|
> | 0 dB | 0.048 | 2.4% |
> | -10 dB | 0.113 | 5.5% |
> | -20 dB | 0.353 | 16.6% |
> | -30 dB | 1.28 | 52% |
>
> 结论:(1) **强分量不被弱分量拖垮**(f 0.04/a 2%,联合牛顿无"强梯度压弱");(2) 弱分量
> 单调退化,**-20dB 是坎**——弱分量有效 SNR≈标称+ratio,落入 CRB 阈值区,是**信息论下界**
> 而非方法缺陷,-10dB 内可用;(3) **marginal≈profile_map**,弱分量上也未拉开(修正"logdet
> 在弱分量更优"的预期);(4) **方法直接给幅值**(0/-10dB 个位数%),兑现"克服 DeepFreq 不估幅值"。
> ⚠️ 待测:弱分量 + 真实 selection(非 oracle)——弱分量 FR 峰可能被淹没,q(z) 选中率未验证。

## 3.4 噪声

加噪参考 DeepFreq `noise_torch`。Batch 1 使用固定 SNR 的 Gaussian 噪声：

```python
snr_db = 20
snr_linear = 10 ** (snr_db / 10)
noise_kind = "gaussian"
```

DeepFreq 训练默认的 `gaussian_blind` 可作为后续鲁棒性对照，但不得混入 Batch 1 默认配置。

后续扫描：

```python
snr_db_list = [-5, 0, 5, 10, 15, 20, 30]
```

## 3.5 数据规模

初始配置：

```python
train_size = 100000
val_size = 10000
test_size = 10000
seed = 42
```

如果资源不足，可降低到：

```python
train_size = 20000
val_size = 2000
test_size = 5000
```

但必须在结果中记录使用的规模，不得静默修改。

## 3.6 数据返回字段

每条样本至少返回：

```python
{
    "y_complex": complex_tensor[N],
    "clean_complex": complex_tensor[N],
    "time": float_tensor[N],
    "token_features": float_tensor[N, 4],
    "freq_true_sorted": float_tensor[K],
    "complex_amp_true_sorted_raw": complex_tensor[K],
    "complex_amp_true_sorted_norm": complex_tensor[K],
    "normalization_scale": scalar,
    "noise_variance": scalar,
    "snr_db": scalar,
    "frequency_distribution": "uniform_ordered_with_min_separation",
    "frequency_domain": [0.0, 1.0],
    "min_separation": scalar,
    "circular_distance": bool,
    "complex_amplitude_distribution": "complex_gaussian",
    "tau_c_raw": scalar,
    "tau_c_norm": scalar,
}
```

若 `normalize_clean_signal_rms=True`，`complex_amp_true_sorted_norm` 必须与 `y_complex` 的归一化尺度一致；
幅值边缘化 likelihood 使用的 (\Sigma_c) 也必须处在同一尺度，即使用 `tau_c_norm`。
不得一边用归一化观测，一边使用未归一化幅值先验。

---

# 4. 通用 encoder

## 4.1 共用 backbone

V0、V1、V2 必须使用同一个 encoder backbone，以保证比较公平。

建议直接复用 `bayesianLS` 当前可运行 encoder，并只修改输入维度与输出 head。

输入：

```python
input_dim = 4
```

即：

```text
Re(y), Im(y), normalized_time, normalized_delta_time
```

第一轮使用：

```python
hidden_dim = 128
num_layers = 4
num_heads = 4
dropout = 0.0
```

全局聚合方式沿用 `bayesianLS` 当前实现，不得在 V0、V1、V2 之间改变。

如果 `bayesianLS` 使用 mean pooling，则两者都用 mean pooling。

不得在比较中一边使用 Transformer，另一边使用 SSM。

## 4.2 输出约束

Batch 1 默认使用第 2.3 节的 ordered gap Gaussian 后验。encoder head 输出：

```python
mu_g: [B, K + 1]
covariance_parameters:
    diagonal: log_std_g [B, K + 1]
    full:     lower_cholesky_g [B, K + 1, K + 1]
```

V0、V1、V2 必须共用同一 backbone，只改变 posterior covariance 结构。
不得为了兼容性退回无序 frequency slots。

后验方差使用：

[
\sigma_k
========

\sigma_{\min}
+
\operatorname{softplus}(v_k).
]

配置：

```python
posterior_std_min = 1e-4
posterior_std_max = 5.0
```

必要时 clip 到该范围。

---

# 5. 通用频率后验方案

## 5.1 主方法：幅值边缘化 ELBO

通用估计器不再训练幅值和相位神经后验。设复幅值先验为：

[
c\sim\mathcal{CN}(0,\Sigma_c).
]

第一轮使用：

[
\Sigma_c=\tau_c^2I.
]

若观测经过 DeepFreq 风格 RMS normalization，则这里的 (\tau_c) 必须使用归一化后的 `tau_c_norm`。

观测模型为：

[
y\mid f,c
\sim
\mathcal{CN}(\Phi(f)c,\sigma^2I).
]

积分掉 (c) 后：

[
p(y\mid f)
==========

\mathcal{CN}
\left(
y;0,C(f)
\right),
]

其中：

[
C(f)
====

\sigma^2I+\Phi(f)\Sigma_c\Phi(f)^H.
]

对应负对数边缘似然：

[
\mathcal L_{\mathrm{marg}}(f)
=============================

y^HC(f)^{-1}y
+
\log\det C(f)
+
N\log\pi.
]

变分训练目标：

[
\mathcal J
==========

\frac1S
\sum_{s=1}^{S}
\mathcal L_{\mathrm{marg}}
\left(
f^{(s)}
\right)
+
\beta
\operatorname{KL}
\left(
q_\theta(g\mid y)\|p(g)
\right).
]

其中：

[
g^{(s)}\sim q_\theta(g\mid y),
\qquad
f^{(s)}=\operatorname{OrderedTransform}(g^{(s)}).
]

配置固定为：

```python
amplitude_elimination = "marginal_likelihood"  # 主方法
posterior_parameterization = "ordered_gap_gaussian"
prior_space = "gap_latent"
gap_prior_std = 2.0
use_posterior_mean_only = False
num_posterior_samples = 8
beta_gap_kl = 1.0
```

训练必须使用后验采样。不得因为数值不稳定而改成 posterior mean path。

## 5.2 profile-MAP 对照

profile-MAP 只作为计算简化对照，不是主方法。

给定频率后：

[
\hat c(f)
=========

\arg\min_c
\frac{|y-\Phi(f)c|^2}{\sigma^2}
+
c^H\Sigma_c^{-1}c.
]

[
\mathcal L_{\mathrm{prof}}(f)
=============================

\frac{|y-\Phi(f)\hat c(f)|^2}{\sigma^2}
+
\hat c(f)^H\Sigma_c^{-1}\hat c(f).
]

配置：

```python
amplitude_elimination = "profile_map"  # 对照实验
```

这里不能称为纯 LS；含幅值高斯先验时是 MAP/ridge。必须通过线性求解器实现，不得显式求逆。

## 5.3 半摊销精修

半摊销精修仅用于测试，不参与 encoder 训练。

输入为 encoder 后验的代表点，例如 `posterior_map_or_mode_estimate` 或 `posterior_sample_mean_estimate`。
对单条测试序列执行固定步数的 marginal/profile 目标优化：

```python
refinement_method = "gradient"  # first implementation
refinement_steps = 10
refinement_lr = 1e-2
```

随后可实现 GN：

```python
refinement_method = "gauss_newton"
refinement_steps = 5
```

必须分别报告：

* encoder-only；
* encoder + gradient refinement；
* encoder + GN refinement。

不得用 refinement 结果反向更新 encoder。

---

# 6. 通用验证实验

## 6.1 V0：单分量 sanity check

[
K=1.
]

验证：

* posterior sampling 是否稳定；
* marginal likelihood 是否正确；
* encoder 能否定位频率；
* 后验方差是否随 SNR/N 收紧；
* 半摊销是否提高精度。

V0 是排除数据、likelihood、重参数化和矩阵求解问题的基础。

## 6.2 V1：多分量 diagonal ordered-gap baseline

[
q(g\mid y)=\mathcal N(\mu,\operatorname{diag}(\sigma^2)).
]

目的：作为 factorized gap posterior baseline，检查多分量失败是否来自后验相关性缺失。

## 6.3 V2：多分量 full-covariance ordered-gap posterior

[
q(g\mid y)=\mathcal N(\mu,LL^T).
]

目的：检验全协方差是否解决近邻频率之间的相关后验问题。

V1 与 V2 使用相同 encoder backbone、相同幅值边缘化 likelihood、相同 gap latent prior，只改变协方差结构。

## 6.4 第一版禁止项

第一版不得引入：

* mixture posterior；
* normalizing flow；
* supervised frequency loss；
* posterior mean-only training；
* 无序 factorized frequency slots；
* 幅值/相位神经后验。

---

# 7. 通用实验指标

## 7.1 频率匹配

评价时使用 Hungarian matching 或排序后对应，仅用于指标，不进入无监督训练损失。

记录：

[
\mathrm{RMSE}_f
===============

\sqrt{
\frac1K
\sum_k
(\hat f_k-f_k^\star)^2
}.
]

同时记录 circular frequency error，避免 (-1/2) 与 (1/2) 边界问题。

## 7.2 间距扫描

主要指标：

* matched frequency RMSE；
* resolution probability；
* estimated gap error；
* all-components success rate。

定义 resolution success：

```text
所有 K 个估计频率在容差内与 K 个真值一一匹配。
```

容差建议：

[
\tau_f=\frac{0.25}{N}.
]

slot exchange rate 仅作为辅助指标，不作为主验收标准。

## 7.3 SNR 与 N 扫描

扫描：

```python
N_list = [32, 64, 128, 256]
snr_db_list = [-5, 0, 5, 10, 20, 30]
```

比较：

* V0/V1/V2 encoder-only；
* V0/V1/V2 semi-amortized；
* diagonal vs full-covariance posterior；
* marginal likelihood vs profile-MAP；
* DeepFreq；
* periodogram；
* MUSIC；
* oracle initialized profile optimizer；
* CRB。

## 7.4 校准

只对 V0/V1/V2 的采样后验执行：

* empirical coverage；
* posterior standard deviation vs empirical error；
* SBC rank histogram；
* P–P plot。

profile-MAP 对照不做 posterior calibration 结论。

---

# 8. 通用实验决策树

## 情形 A：V0 成功，V1/V2 成功

结论：

```text
DeepMUSIC 的核心摊销频率后验思想在通用多正弦场景成立；
幅值和相位解析消元后，encoder 可以学习数据依赖频率后验。
```

## 情形 B：V0 成功，V1 失败，V2 成功

结论：

```text
多分量失败主要来自 factorized posterior 无法表达频率相关性；
ordered full-covariance gap posterior 是必要修改。
```

必须进一步比较：

* diagonal vs full covariance；
* 近邻分量相关性；
* posterior variance；
* reconstruction/marginal NLL；
* semi-amortized refinement 收益。

## 情形 C：V0 成功，V1/V2 都失败

必须执行诊断，不得直接进入复杂模型：

1. 检查 ordered transform 与边界梯度；
2. 检查 marginal likelihood 曲线；
3. 检查 profile-MAP 对照；
4. 检查单样本非摊销优化；
5. 检查 encoder 梯度；
6. 检查 posterior covariance 是否坍缩；
7. 检查 amortization gap。

输出明确失败归因。

不得在此阶段引入 flow、mixture 或 supervised frequency loss。

## 情形 D：V0 失败

必须先修复基础链路：

1. 数据生成；
2. 幅值边缘化 NLL；
3. gap 重参数化采样；
4. KL；
5. 矩阵求解与梯度尺度。

V0 失败前不得推进多分量结论。

---

# 9. BTT 方案总结构

BTT 系统固定为：

[
\text{candidate generator}
\rightarrow
q_\eta(z\mid y,\mathcal C)
\rightarrow
\text{marginal/profile objective refinement}
\rightarrow
\text{conditional alternating GN}
\rightarrow
\text{local covariance}.
]

不再训练连续频率 (\delta) 头。

BTT 与通用估计器共享同一原则：幅值解析消元后的频率目标是核心目标。
首选 marginal likelihood；profile-MAP 作为计算简化对照。

---

# 10. BTT 信号模型

采样时刻：

[
t_{r,p}
]

由转速轨迹和探针角位置决定。

频率轨迹：

[
f_k(t)
======

g_k(\Omega(t);\theta_k^{\mathrm{freq}}).
]

第一阶段只实现线性 Campbell：

[
f_k(t)
======

a_{0k}+a_{1k}\Omega(t).
]

相位：

[
\Phi_k(t)
=========

2\pi a_{0k}t
+
2\pi a_{1k}\Theta(t)
+
\phi_k,
\qquad
\Theta(t)=\int_0^t\Omega(\tau)d\tau.
]

观测：

[
y_n
===

\sum_{k=1}^{K}
c_k e^{j\Phi_k(t_n)}
+
w_n.
]

---

# 11. BTT 候选集合接口

## 11.1 候选生成

沿用当前 FR-front 或 production candidate generator。

每个实例生成：

[
\mathcal C(y)
=============

{(z_m,\xi_m)}_{m=1}^{M}.
]

每个候选特征必须至少包含：

```python
candidate_features = [
    candidate_frequency_or_parameters,
    fr_score,
    profile_rss,
    local_fr_real,
    local_fr_imag,
    local_fr_magnitude,
    rotor_speed_summary,
    probe_geometry_summary,
]
```

局部 FR 邻域使用固定长度：

```python
local_bins_each_side = 4
```

即总长度 9。

不得只把候选绝对频率和单个峰高交给 encoder。

## 11.2 候选标签

真值 basin 标签定义为：

```text
包含真值连续参数的候选 basin 索引。
```

若多个候选都覆盖真值，选择 profile RSS 最小者作为主标签，同时记录所有有效候选用于 Recall@L。

候选类别是实例内索引，不是全数据集固定绝对频率类别。

---

# 12. BTT 离散 encoder

## 12.1 输入

两部分输入：

1. 原始观测 token：
   [
   [\operatorname{Re}(y_n),\operatorname{Im}(y_n),\tilde t_n,
   \Delta\tilde t_n,\Omega_n,\sin\alpha_p,\cos\alpha_p]
   ]

2. candidate feature tokens：
   [
   \xi_m.
   ]

## 12.2 结构

先实现最小结构：

```text
observation encoder
→ global context h_y

candidate MLP
→ candidate embedding h_m

score_m = MLP([h_y, h_m])
softmax over candidates
```

不得第一版直接实现复杂 cross-attention、set transformer 或 mixture architecture。

如果 `bayesianLS` encoder 可复用，则用其作为 observation encoder。

## 12.3 损失

仅使用：

[
\mathcal L_{\mathrm{candidate}}
===============================

-\log q_\eta(z^\star\mid y,\mathcal C).
]

不得让梯度穿过 marginal/profile-MAP refinement、FR-front 或 GN。

不得增加连续 offset loss。

---

# 13. BTT 逐实例连续精修

## 13.1 第一阶段：oracle basin

脚本：

```text
btt_oracle_continuous_refinement.py
```

输入真值 basin 初值，不使用 encoder。

流程：

1. 由 basin 中心初始化 (\theta^{\mathrm{freq}})；
2. 使用 marginal likelihood 主目标，或 profile-MAP 对照解全部复幅值；
3. 条件交替更新每个频率参数；
4. 每次更新后重新计算联合 marginal/profile-MAP 目标；
5. 固定迭代次数或基于相对 RSS 停止。

## 13.2 连续精修求解器

> **⚠️ 实测修正(2026-06-27,见 `reference/TASK2_ALIGNMENT.md §5`):原计划的「条件交替 + Π⊥
> 投影」假设被推翻。改用【联合阻尼牛顿】作为主力 solver。**

**主力(实测最优):在【绝对频率 f_{1:K}】上对【联合 marginal NLL】做【阻尼牛顿/GN】。**
- B0 实测:oracle 初值 0.48 → 精修后 **MAE 0.082**(优于 LSF 0.127,逼近 CRB 0.099),
  marginal ≈ profile-MAP。
- 实现:`experiments/btt_oracle_continuous_refinement.py:newton_refine`
  (autograd 求一阶梯度 + 全 Hessian,(H+λI)d=−g,Levenberg 阻尼)。

**已否决:条件交替 + Π⊥ 投影**(`solvers/conditional_alternating_gn.py`,保留作记录)。
更新第 k 个分量时投影掉其他分量字典 Φ_{\\k} 的方案,实测:

| 场景 | 联合牛顿 | 条件交替(Π⊥) |
|---|---|---|
| well-sep c16 | **0.082** | 0.173 |
| 近邻 5Hz | **0.159** | 1.79(崩) |
| 近邻 3Hz | **0.164** | 2.13(崩) |

**机制**:近邻分量原子高度共线,Π⊥ 把「其他分量」投影掉会**连真分量能量一起抹掉**,残差里
真信号没了 → 单频拟合跑飞。联合 marginal 同时处理所有分量 + logdet 正则,反而稳。
**故近邻场景尤其不能用 Π⊥ 交替;CLEAN 永久剥离同样禁用(剥离误差累积)。**

## 13.3 捕获域实验

扫描初始误差：

```python
initial_offset_hz = [
    -2.0, -1.5, -1.0, -0.5, -0.25,
     0.0,
     0.25, 0.5, 1.0, 1.5, 2.0
]
```

记录：

* convergence success；
* final frequency MAE；
* iteration count；
* RSS decrease；
* divergence rate。

据此确定 FR 候选必须达到的最大初值误差。

> **✅ 实测结果(2026-06-27,`btt_gn_capture_region.py`):捕获域比预期宽。**
> ±1.5 Hz 内 success 98.8% / diverge 0%,±2.0 仍 96%;收敛后 MAE 0.084。
> FR 候选实测初值误差(mean 0.52 / p95 1.11 / **max 1.84** Hz)**全部落在捕获域内**。
> **推论:原计划「FR 峰 sub-bin 插值兜尾部」基本非必需**,仅对极少数 >1.8 Hz 样本有保险价值。

---

# 14. BTT 完整推理

流程固定为：

```text
raw observations
→ build observation tokens
→ FR candidate generation
→ build candidate feature set
→ candidate encoder q(z | y, C)
→ keep top-L candidates
→ per-candidate marginal/profile-MAP + GN
→ score refined candidates
→ select or retain posterior mixture
→ output frequencies, amplitudes, phases, uncertainty
```

默认：

```python
top_l = 3
```

最终候选分数：

[
s_m
===

-\frac{\operatorname{RSS}*m}{\hat\sigma^2}
+
\gamma \log q*\eta(z_m\mid y,\mathcal C).
]

第一轮：

```python
gamma = 1.0
```

同时报告：

* physics-only；
* encoder-only；
* hybrid。

不得只报告 hybrid。

---

# 15. BTT 不确定性输出

对于每个保留 basin：

[
q(f\mid z_m,y)
\approx
\mathcal N
(\hat f_m,H_m^{-1}).
]

完整近似后验：

[
q(f\mid y)
\approx
\sum_{m\in\mathrm{topL}}
\pi_m
\mathcal N(\hat f_m,H_m^{-1}).
]

其中：

[
\pi_m
=====

\operatorname{softmax}(s_m).
]

输出必须区分：

* basin uncertainty；
* basin 内 continuous uncertainty。

不得只输出单个标准差而忽略 basin 多峰。

---

# 16. BTT 实验顺序

严格按以下顺序执行。

## B0：oracle basin + (K) 已知

验证 marginal/profile-MAP + GN 能否复现约 (0.11) Hz。

## B1：FR candidate + oracle candidate selection

验证 FR 候选是否进入 GN 捕获域。

## B2：candidate encoder + (K) 已知

训练离散候选选择。

指标：

* top1；
* Recall@3；
* basin hit；
* downstream refined frequency MAE。

## B3：完整候选选择 + GN

比较：

* FR raw；
* production greedy；
* physics-only candidate score；
* encoder-only；
* hybrid；
* oracle basin。

## B4：未知 (K)

只有 B0–B3 成功后才实现。

先使用：

* residual noise-floor；
* BIC；
* pruning。

不得第一版加入 learned existence head。

## B5：真实数据

只在合成流程稳定后执行。

真实数据评价：

* reconstruction residual；
* frequency trajectory continuity；
* reversal/steady segment consistency；
* physics plausibility；
* 不同窗口重叠区域的一致性。

---

# 17. 日志与产物要求

每个实验必须保存：

```text
config.json
git_commit.txt
train_log.csv
eval_metrics.json
best_checkpoint.pt
last_checkpoint.pt
plots/
predictions/
```

通用实验额外保存：

```text
frequency_scatter.png
rmse_vs_snr.png
rmse_vs_N.png
resolution_vs_spacing.png
coverage_plot.png
sbc_rank_histogram.png
```

BTT 实验额外保存：

```text
candidate_scores.png
capture_region.png
rss_before_after_gn.png
basin_confusion.png
trajectory_overlay.png
```

不得仅输出终端日志。

---

# 18. 最终报告模板

Codex 完成后必须生成：

```text
artifacts/final_summary.md
```

内容固定包括：

## A. V0 单分量结果

* 是否收敛；
* 频率指标；
* posterior sampling 是否稳定；
* 后验方差是否随 SNR/N 收紧；
* 半摊销是否提高精度。

## B. V1/V2 多分量结果

* diagonal ordered-gap posterior；
* full-covariance ordered-gap posterior；
* encoder-only；
* semi-amortized；
* posterior calibration；
* 与 DeepFreq/MUSIC/periodogram/CRB 对比。

## C. DeepMUSIC 判断

只能选择以下之一：

```text
1. DeepMUSIC 核心摊销频率后验思想在通用场景成立。
2. 单分量成立，但多分量需要 full-covariance ordered posterior。
3. 即使使用幅值边缘化和 ordered posterior，通用摊销频率后验仍失败。
```

必须给出实验依据。

## D. BTT 结果

* oracle basin 连续精修；
* FR 捕获率；
* candidate selection；
* downstream MAE；
* physics-only vs encoder-only vs hybrid；
* 当前是否达到可用于真实数据的标准。

## E. 最终方法定位

根据 V0/V1/V2 结果自动采用对应叙述：

### V0/V1/V2 成功

```text
BTT 方法是摊销频率后验思想在 alias 多峰条件下的离散—连续分层扩展。
```

### V0 成功、V1 失败、V2 成功

```text
BTT 方法保留固定物理 decoder 与摊销后验思想；
多分量相关性通过 full-covariance ordered posterior 或离散 basin 后验处理。
```

### V0/V1/V2 均失败

```text
BTT 方法不再宣称是通用连续摊销频率后验的直接扩展；
仅保留固定物理 decoder、解析幅值消元和全局 encoder 的 model-based learning 思想。
```

---

# 19. 第一批必须执行的最小实验

先只实现以下内容，不要一次实现全部后续扩展。

## Batch 1

1. 通用数据集；
2. 时间输入接口；
3. 当前主目录幅值边缘化 ELBO loss 接口；
4. ordered gap posterior + 后验采样训练；
5. 多参数训练/验证/测试数据集，每条样本独立采样频率、幅值、相位和噪声；
6. (K) 递进：先 (K=1)，再 (K=2) 并控制最小间隔，随后 (K=3,4,5,6)；
7. V0 单分量 sanity check；
8. V1 diagonal ordered-gap posterior；
9. V2 full-covariance ordered-gap posterior；
10. 三个 seed；
11. 20 dB、(N=64)；
12. 保存完整指标。

Batch 1 的主线训练损失只使用：

```text
幅值边缘化 NLL + beta * KL(q(g | y) || p(g))
```

其中 (q(g | y)) 是 ordered gap latent 后验；
(p(g)) 是宽高斯 gap prior；
data prior 是多参数采样分布，不要求被 KL 精确复刻。

训练必须使用 posterior samples，不得使用 posterior mean-only path。
不再实现 G0 全参数神经后验。

## Batch 2

1. V0/V1/V2 半摊销 refinement；
2. 间距扫描；
3. SNR 扫描；
4. 与 periodogram/MUSIC 对比；
5. DeepFreq 对比接口。

## Batch 3

1. BTT oracle basin；
2. marginal likelihood 与 profile-MAP；
3. GN 捕获域；
4. 复现约 (0.11) Hz。

## Batch 4

1. FR candidate interface；
2. candidate feature set；
3. candidate encoder；
4. top1 与 Recall@3；
5. 完整 downstream MAE。

在 Batch 1 完成并生成报告前，不得开始更强 posterior（flow/mixture）或未知 (K)。

---

# 20. 验收纪律

Codex 必须遵守以下原则：

1. 不修改研究问题；
2. 不增加未规定模块；
3. 不因某实验失败而自动切换成监督频率回归；
4. 不因训练不稳定而删除时间输入；
5. 不用标签排序替代有序参数化；
6. 不把 Hungarian 加入 profile 重构损失；
7. 不把无 KL 或 mean-only 结果解释为校准后验不确定性；
8. 不在 BTT 中恢复连续 (\delta) 摊销头；
9. 不在已知 (K) 验证完成前实现未知 (K)；
10. 任何配置变更必须写入 config 和 final summary。
