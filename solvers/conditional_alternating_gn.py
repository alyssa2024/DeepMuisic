"""plan_final §13.2: conditional alternating GN(条件交替高斯牛顿)。

多分量频率精修。更新第 k 个分量时,把其他分量的字典 Phi_{\\k} 投影掉,在正交补
Pi^perp_{Phi_\\k} 上对单个 f_k 做 GN。逐分量轮转至收敛。

vs B0 的【联合牛顿】(所有 f 一起一个 Hessian):
  - 条件交替每步是【单变量】问题 → 良条件,近邻分量也不病态;
  - 不用 CLEAN 永久剥离 —— 每轮都基于完整联合模型重算 Phi_{\\k}(plan_final §13.2 红线)。

目标函数复用 profile/marginal 的"投影残差 RSS":对单个 f_k,固定其他频率,残差
r_k = Pi^perp_{Phi_\\k} y,其能量随 f_k 变化 → 在 r_k 上拟合单原子 exp(j2pi f_k t)。
单频 profile RSS(对幅值闭式消元)= -|a(f_k)^H r_k|^2 / (a^H a)。对 f_k 用 autograd
求一阶/二阶导做阻尼牛顿步。
"""
import torch


def _atom(t, f):
    # exp(j 2pi f t), t:[N], f scalar tensor -> [N] complex
    return torch.polar(torch.ones_like(t), 2.0 * torch.pi * f * t)


def _proj_perp_residual(y, t, freqs_other, mask):
    """r = Pi^perp_{Phi_other} y。freqs_other:[K-1]。返回 [N] complex(已乘 mask)。"""
    w = mask.to(y.real.dtype)
    yw = y * w
    if freqs_other.numel() == 0:
        return yw
    phase = 2.0 * torch.pi * t.unsqueeze(-1) * freqs_other.unsqueeze(0)  # [N, K-1]
    Phi = torch.polar(torch.ones_like(phase), phase) * w.unsqueeze(-1)
    # 最小二乘投影: yhat = Phi (Phi^H Phi)^-1 Phi^H y
    PhiH = Phi.conj().transpose(-2, -1)
    gram = PhiH @ Phi
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    coef = torch.linalg.solve(gram + 1e-6 * eye, PhiH @ yw.unsqueeze(-1))
    yhat = (Phi @ coef).squeeze(-1)
    return yw - yhat


def _single_freq_rss(f_k, r_k, t, mask):
    """单原子 profile RSS(幅值闭式消元): RSS = ||r||^2 - |a^H r|^2/(a^H a)。
    对 f_k 可微。返回标量(我们最小化它)。"""
    w = mask.to(r_k.real.dtype)
    a = _atom(t, f_k) * w
    aH_r = torch.sum(a.conj() * r_k)
    aH_a = torch.sum((a.conj() * a).real).clamp_min(1e-12)
    rss = torch.sum((r_k.conj() * r_k).real) - (aH_r.conj() * aH_r).real / aH_a
    return rss


def conditional_alternating_gn(y, t, f_init, mask, outer_rounds=5, inner_steps=3,
                               damping=1e-3, max_step_hz=2.0):
    """y:[N] complex, t:[N], f_init:[K]. 返回精修后 [K]。"""
    f = f_init.clone().detach()
    K = f.numel()
    for _ in range(outer_rounds):
        for k in range(K):
            others = torch.cat([f[:k], f[k + 1:]])
            r_k = _proj_perp_residual(y, t, others, mask).detach()
            fk = f[k].clone().detach().requires_grad_(True)
            for _ in range(inner_steps):
                rss = _single_freq_rss(fk, r_k, t, mask)
                g, = torch.autograd.grad(rss, fk, create_graph=True)
                h, = torch.autograd.grad(g, fk, retain_graph=False)
                lam = damping * (h.abs() + 1e-9)
                step = (-g / (h + lam)).clamp(-max_step_hz, max_step_hz)
                fk = (fk + step).clamp_min(1e-6).detach().requires_grad_(True)
            f = f.clone()
            f[k] = fk.detach()
    return f.detach()
