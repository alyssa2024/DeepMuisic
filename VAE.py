import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicalHarmonicVAE(nn.Module):
    def __init__(self, encoder: nn.Module):
        """
        传入你写好的 VariationalIndependentTimeSeriesTransformer 实例
        """
        super().__init__()
        self.encoder = encoder
        self.num_harmonics = encoder.output_dim

    def reparameterize(self, mu_A, logvar_A, a_w, mu_phi, kappa_phi):
        """
        对三种物理量进行不同的重参数化采样
        """
        # 1. 幅度 A (Gaussian): z = mu + sigma * eps
        std_A = torch.exp(0.5 * logvar_A)
        eps_A = torch.randn_like(std_A)
        A = mu_A + eps_A * std_A

        # 2. 频率 w (Maxwell-Boltzmann): 长度为 a 的三维高斯向量的模长
        # x = a * sqrt(X1^2 + X2^2 + X3^2), where Xi ~ N(0,1)
        eps_w = torch.randn(*a_w.shape, 3, device=a_w.device)
        w = a_w * torch.sqrt(torch.sum(eps_w**2, dim=-1))

        # 3. 相位 phi (von Mises): 使用包裹正态分布(Wrapped Normal)近似实现可导采样
        # sigma^2 ≈ 1/kappa
        std_phi = torch.sqrt(1.0 / kappa_phi)
        eps_phi = torch.randn_like(mu_phi)
        phi = (mu_phi + eps_phi * std_phi) % (2 * np.pi) - np.pi

        return A, w, phi

    def decode(self, A, w, phi, t):
            """
            复数物理方程解码器
            A, w, phi shape: (batch_size, num_harmonics)
            t shape: (batch_size, seq_len)
            """
            # 扩展维度以进行广播运算 -> (batch_size, num_harmonics, 1)
            A = A.unsqueeze(-1)
            w = w.unsqueeze(-1)
            phi = phi.unsqueeze(-1)
            
            # t 扩展为 (batch_size, 1, seq_len)
            if t.dim() == 2:
                t = t.unsqueeze(1) 

            # 计算复指数的相位角: theta = w * t + phi
            # shape: (batch_size, num_harmonics, seq_len)
            theta = w * t + phi

            # 使用 torch.polar 构造复数张量 A * e^{j(theta)}
            # 这会自动将 A 作为模长，theta 作为幅角，生成复数张量
            harmonics = torch.polar(A, theta)
            
            # 沿谐波维度 (dim=1) 求和，得到预测的复数时域信号 x_hat
            # 结果 shape: (batch_size, seq_len)
            x_hat = torch.sum(harmonics, dim=1)
            
            return x_hat

    def forward(self, x, t):
        """
        x: 观测信号 (batch, seq, dim)
        t: 对应的时间戳 (batch, seq)
        """
        # 1. Encoder 提取分布参数
        dist_params = self.encoder(x)
        (mu_A, logvar_A), a_w, (mu_phi, kappa_phi) = dist_params

        # 2. 采样潜变量
        A, w, phi = self.reparameterize(mu_A, logvar_A, a_w, mu_phi, kappa_phi)

        # 3. 解码重构信号
        x_hat = self.decode(A, w, phi, t)

        return x_hat, dist_params