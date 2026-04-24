import torch
import numpy as np

def compute_harmonic_elbo(x_target, x_hat, dist_params, prior_a_w=1.0):
    (mu_A, logvar_A), a_w, (mu_phi, kappa_phi) = dist_params
    
    # ---------------------------------------------------------
    # 1. 复数重构损失 (Complex Reconstruction Loss)
    # ---------------------------------------------------------
    # 计算复数误差张量
    error = x_hat - x_target.squeeze()
    
    # 复数误差的平方等于实部平方加虚部平方: |z|^2 = Re(z)^2 + Im(z)^2
    # 使用 torch.abs(error) 获取模长，然后平方并求和
    recon_loss = torch.sum(torch.abs(error) ** 2)

    # ---------------------------------------------------------
    # 2. KL 散度 - 保持不变
    # ---------------------------------------------------------
    # 幅度 A (Gaussian)
    kl_A = -0.5 * torch.sum(1 + logvar_A - mu_A.pow(2) - logvar_A.exp())

    # 频率 w (Maxwell-Boltzmann 解析解)
    ratio_sq = (a_w / prior_a_w).pow(2)
    kl_w = 1.5 * torch.sum(ratio_sq - 1.0 - torch.log(ratio_sq))

    # 相位 phi (von Mises 解析解)
    i0e_k = torch.special.i0e(kappa_phi)
    i1e_k = torch.special.i1e(kappa_phi)
    bessel_ratio = i1e_k / (i0e_k + 1e-8)
    log_i0 = kappa_phi + torch.log(i0e_k + 1e-8)
    kl_phi = torch.sum(kappa_phi * bessel_ratio - log_i0)

    # 汇总
    total_kl = kl_A + kl_w + kl_phi
    elbo_loss = recon_loss + total_kl
    
    return elbo_loss, recon_loss, total_kl