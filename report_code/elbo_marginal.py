"""解析边际似然 ELBO —— 幅值 c 用**闭式积分**边际化掉 (不是变分近似)。

═══ 为什么不是变分 q(c) (已证伪, diag_amp_posterior) ═══

高斯先验 p(c)=CN(0,τ²I) 下, ∫p(y|f,c)p(c)dc **有闭式解**, 根本不需要变分。
变分版实测: σ_c/|c| 坍缩到 0.017 (log_lik≈1965 vs kl_c≈82, 似然大24×, KL拦不住)
⇒ c 退化成点估计 ⇒ σ_f **更小** (0.00169→0.00078, 覆盖率 69.9%→37.2%) —— 方向反了。

═══ 解析边际似然 (照搬主目录 VAE.amplitude_marginal_nll) ═══

    S = diag(1/τ²) + Φ(f)ᴴΦ(f)/σ_n²                    Bayesian LS 系统矩阵

    −log p(y|f) = quad + logdet
      quad   = ‖r‖²/σ_n² − rᴴΦ S⁻¹ Φᴴr / σ_n⁴          (r = y − Φμ_prior, μ_prior=0 ⇒ r=y)
      logdet = N·log σ_n² + Σ log τ² + logdet(S)        ⭐

⭐ **logdet(S) 就是 c 的不确定性向 f 传播的载体**:
   S 依赖 f (通过 ΦᴴΦ), logdet(S) 对 f 的曲率 = 幅值不确定性的贡献。
   profile 版 −N·log(RSS/N) **完全没有这一项** ⇒ σ_f 系统性偏小 (用户的原始假说)。

对比:
| | 公式 | c 的不确定性 |
|---|---|---|
| profile | −N·log(RSS/N) | ❌ 忽略 (c 钉在 MLE) |
| **边际** | quad + logdet(S) | ✅ **logdet(S)** |

═══ ⚠ 两个超参必须让数据定, 别拍脑袋 ═══
  τ² (幅值先验方差): 真实 |c|∈[0.3,1.5], E|c|²≈0.8。**变分版用 τ²=1.0 是坍缩原因之一。**
  σ_n² (噪声方差):   做成**自由变量** ⇒ 保住"不需要 SNR" (可上真实数据)。
  两个都是逐实例自由变量, 和 (μ_f, σ_f) 一起 Adam 优化。

⚠ 边际似然**没有 L1**: 假峰不会被压到 0 (τ² 先验是 L2 不是 L1)。
   假峰判别改看**后验幅值 |c_post|** (Bayesian LS 的后验均值, 弱证据会被先验拉向 0)。
"""
import math
import torch

# TEX_NOCLIP=1 -> 去掉本文件全部范围限制。逐项说明:
#   * tau2/sn2 的 clamp_min(eps): 作用在 **Adam 优化的自由变量** 上 —— 一旦
#     exp(log_tau2) 落进饱和区, clamp 的梯度为 0, 该参数**永久卡死**, 比闭式
#     更新那边(下轮还能弹回)严重得多。
#   * quad 的 clamp_min(0.0): quad 数学上必非负, 截到 0 说明数值上算出了负值
#     (S 求解病态 / σ_n² 被低估)。截断把这个**告警信号**吞掉, 似然变常数、梯度消失。
#   * tau2_0 的 clamp_min(1e-6): 幅值平方下限, 比 tex 那边的 1e-12 高六个量级,
#     幅值 <1e-3 的弱分量会被直接抬高。
#   * sn2_0 的 clamp_min(1e-10): 噪声方差初值下限。
# 保留 amp_var 的 clamp_min(0.0): 协方差对角负值纯属浮点误差, 不是建模量。
import os as _os
NOCLIP = _os.environ.get("TEX_NOCLIP", "1").lower() in ("1", "true", "yes")

# mu 的箱约束半宽 = BAND_FRAC * (stage-1 band spacing), 与 elbo_stage2_tex 同口径。
# 调用方通过 refine_marginal_batch(grid_step_hz=...) 传实际网格步长; 传 0 或不传
# 则不做约束 (历史行为)。约束是**投影**式的: 每个 Adam step 之后把 mu 夹回
# [init_mu - hw, init_mu + hw], 因此候选永远不会离开自己的 band ⇒ 召回不低于 stage-1。
BAND_FRAC = float(_os.environ.get("TEX_BAND_FRAC", 0.5))


TWO_PI = 2.0 * math.pi


def marginal_terms_batch(mu, logs, log_tau2, log_sn2, y_full, t_full,
                         n_sample=16, generator=None, eps=1e-10):
    """解析边际似然 ELBO 的各项 (批量, 实例维并行)。

    mu/logs      [B,Kc]  q(f) = N(μ_f, σ_f²)
    log_tau2     [B,Kc]  幅值先验方差 (自由变量, 逐分量)
    log_sn2      [B]     噪声方差 (自由变量 ⇒ 不需 SNR)
    y_full [B,N] 复; t_full [B,N]

    返回 (log_lik [B], kl_ent [B], c_post [B,Kc] 幅值后验均值, sigma_n2 [B],
          logdet [B], amp_var [B,Kc] 幅值后验方差 = [S₀⁻¹]_kk)。
    """
    B, Kc = mu.shape
    N = t_full.shape[-1]
    sig = logs.exp()
    tau2 = log_tau2.exp()                                       # [B,Kc]
    sn2 = log_sn2.exp()                                         # [B]
    if not NOCLIP:      # ⚠ clamp 饱和区梯度=0 ⇒ 参数永久卡死, 见文件头 NOCLIP
        tau2 = tau2.clamp_min(eps)
        sn2 = sn2.clamp_min(eps)

    # ---- 重参数化: 只采样 f (c 已解析边际化掉) ----
    e = torch.randn((B, n_sample, Kc), device=mu.device, generator=generator)
    f_s = mu[:, None, :] + sig[:, None, :] * e                  # [B,S,Kc] 对 μ/σ 可导
    ph = TWO_PI * f_s[:, :, None, :] * t_full[:, None, :, None]  # [B,S,N,Kc]
    Phi = torch.exp(1j * ph.to(torch.complex64))                # [B,S,N,Kc]
    PhiH = Phi.conj().transpose(-1, -2)                         # [B,S,Kc,N]
    gram = PhiH @ Phi                                           # [B,S,Kc,Kc] = ΦᴴΦ

    # ---- S = diag(1/τ²) + ΦᴴΦ/σ_n²  (Bayesian LS 系统矩阵) ----
    prec = torch.diag_embed((1.0 / tau2).to(gram.dtype))        # [B,Kc,Kc]
    S = prec[:, None] + gram / sn2.view(B, 1, 1, 1).to(gram.dtype)   # [B,S,Kc,Kc]

    # ---- quad = ‖y‖²/σ_n² − yᴴΦ S⁻¹ Φᴴy / σ_n⁴   (μ_prior=0 ⇒ residual = y) ----
    yS = y_full[:, None, :]                                     # [B,1,N]
    rhs = torch.einsum("bskn,bin->bsk", PhiH, yS).unsqueeze(-1)  # [B,S,Kc,1] = Φᴴy
    sol = torch.linalg.solve(S, rhs)                            # [B,S,Kc,1] = S⁻¹Φᴴy
    corr = (rhs.conj().transpose(-2, -1) @ sol).real.squeeze(-1).squeeze(-1)  # [B,S]
    y_norm = (y_full.abs() ** 2).sum(-1)                        # [B]
    quad = (y_norm[:, None] / sn2[:, None]
            - corr / sn2[:, None].pow(2))                       # [B,S]
    if not NOCLIP:      # ⚠ 负值=数值病态的告警, 截断会把它静默吞掉
        quad = quad.clamp_min(0.0)

    # ---- logdet = N·log σ_n² + Σlog τ² + logdet(S)  ⭐ 幅值不确定性的载体 ----
    _, ld_S = torch.linalg.slogdet(S)                           # [B,S] (复矩阵 → 实 logabsdet)
    logdet = (N * torch.log(sn2)[:, None]
              + log_tau2.sum(-1)[:, None]
              + ld_S.real)                                      # [B,S]

    log_lik = (-(quad + logdet)).mean(1)                        # [B] MC 平均, 最大化
    kl_ent = -logs.sum(-1)                                      # [B] 熵项 (窄带均匀先验)

    # ---- 幅值后验 (Bayesian LS, 在 μ_f 处): 均值 c_post = S₀⁻¹Φᴴy/σ_n², 方差 [S₀⁻¹]_kk ----
    with torch.no_grad():
        ph0 = TWO_PI * mu[:, None, :] * t_full[:, :, None]      # [B,N,Kc]
        P0 = torch.exp(1j * ph0.to(torch.complex64))
        P0H = P0.conj().transpose(-1, -2)
        S0 = prec + (P0H @ P0) / sn2.view(B, 1, 1).to(gram.dtype)
        S0_inv = torch.linalg.inv(S0)                           # [B,Kc,Kc]
        c_post = (S0_inv @ (P0H @ y_full.unsqueeze(-1))).squeeze(-1) \
            / sn2.view(B, 1).to(gram.dtype)                     # [B,Kc] 复
        # ⭐ 幅值不确定度 = 后验协方差的对角 (算法 L1020 的第四个输出)。
        #   S₀ 是 Hermitian 正定 ⇒ 对角元为实且 >0; 取 .real 去掉数值虚部。
        amp_var = S0_inv.diagonal(dim1=-2, dim2=-1).real.clamp_min(0.0)   # [B,Kc]
    return (log_lik, kl_ent, c_post, sn2.detach(), logdet.mean(1).detach(),
            amp_var)


def refine_marginal_batch(init_mu, init_logs, y_full, t_full, beta_ent=1.0,
                          n_steps=800, lr=3e-2, lr_sigma=3e-1, lr_hyp=3e-2,
                          n_sample=16, generator=None, verbose=False,
                          trace=None, grid_step_hz=None):
    """逐实例优化解析边际 ELBO。μ_f/σ_f/τ²/σ_n² 全是自由变量 (⇒ 不需 SNR)。

    初值:
      τ² ← 当前 f 的 LS 解的 |c|² (幅值尺度, 别拍 1.0 —— 变分版就栽在这)
      σ_n² ← 残差 MLE

    trace: 给一个 callable(step, mu, elbo_per_inst, c_post) 就在每步被调用
           (画收敛曲线用)。c_post 是当步的幅值后验均值 (复), 闭式解, 无额外开销。
           ⚠ 只读, 且在 no_grad 外 —— 回调里要 .detach() 自己拿数, 别改动张量。
    """
    B, Kc = init_mu.shape
    N = t_full.shape[-1]
    mu = init_mu.clone().detach().requires_grad_(True)
    logs = init_logs.clone().detach().requires_grad_(True)

    with torch.no_grad():                                       # τ²/σ_n² 的物理初值
        ph = TWO_PI * init_mu[:, None, :] * t_full[:, :, None]
        P = torch.exp(1j * ph.to(torch.complex64))
        PH = P.conj().transpose(-1, -2)
        g = PH @ P + 1e-3 * torch.eye(Kc, device=init_mu.device, dtype=P.dtype)
        c0 = torch.linalg.solve(g, PH @ y_full.unsqueeze(-1)).squeeze(-1)   # [B,Kc]
        r0 = y_full - torch.einsum("bnk,bk->bn", P, c0)
        sn2_0 = (r0.abs() ** 2).mean(-1)                        # [B]
        tau2_0 = (c0.abs() ** 2)                                # [B,Kc] ⭐ 幅值真实尺度
        if not NOCLIP:  # ⚠ tau2_0 的 1e-6 会抬高幅值 <1e-3 的弱分量
            sn2_0 = sn2_0.clamp_min(1e-10)
            tau2_0 = tau2_0.clamp_min(1e-6)

    log_tau2 = tau2_0.log().clone().detach().requires_grad_(True)
    log_sn2 = sn2_0.log().clone().detach().requires_grad_(True)

    # μ 的箱约束: 半宽 = BAND_FRAC * band spacing (调用方给 grid_step_hz)
    _hw = None
    if grid_step_hz:
        _hw = BAND_FRAC * float(grid_step_hz)
        if _hw > 0:
            _mu_lo = (init_mu - _hw).detach()
            _mu_hi = (init_mu + _hw).detach()
        else:
            _hw = None

    # 三组固定学习率 (无退火): μ / logσ (收敛慢, 10× lr) / 超参 (τ², σ_n²)
    opt = torch.optim.Adam([
        {"params": [mu], "lr": lr},
        {"params": [logs], "lr": lr_sigma},
        {"params": [log_tau2, log_sn2], "lr": lr_hyp},
    ])

    for step in range(n_steps):
        ll, kl, c_post, sn2, ld, _ = marginal_terms_batch(
            mu, logs, log_tau2, log_sn2, y_full, t_full,
            n_sample=n_sample, generator=generator)
        loss = -(ll - beta_ent * kl).sum()
        if trace is not None:                  # 收敛曲线: 记**更新前**的状态 (step=0 是初值)
            trace(step, mu.detach(), (ll - beta_ent * kl).detach(), c_post.detach())
        opt.zero_grad(); loss.backward()
        opt.step()
        if _hw is not None:               # 投影回 [init_mu +- hw]
            with torch.no_grad():
                mu.clamp_(min=_mu_lo, max=_mu_hi)
        if verbose and (step % max(n_steps // 8, 1) == 0 or step == n_steps - 1):
            print(f"  step{step:>4} elbo={float((ll-kl).mean()):>10.1f} "
                  f"loglik={float(ll.mean()):>10.1f} logdet={float(ld.mean()):>8.1f} "
                  f"σf={float(logs.detach().exp().median()):.5f} "
                  f"τ²={float(log_tau2.detach().exp().median()):.4f} "
                  f"σn²={float(sn2.mean()):.2e}", flush=True)

    with torch.no_grad():
        _, _, c_post, sn2, _, amp_var = marginal_terms_batch(
            mu, logs, log_tau2, log_sn2, y_full, t_full,
            n_sample=n_sample, generator=generator)
    return {"mu": mu.detach(), "sigma": logs.detach().exp(),
            "amp": c_post.abs(), "amp_c": c_post,
            "amp_var": amp_var, "amp_sigma": amp_var.sqrt(),
            "tau2": log_tau2.detach().exp(), "sigma_n2": sn2}
