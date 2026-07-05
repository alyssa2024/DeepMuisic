"""损失: 按 encoder.type 分派。

回归/rope : 高斯核伪谱 MSE  (输出 [B,fr_size] vs 目标 [B,fr_size])
分类/resfreq: 多热 BCEWithLogits (输出 [B,G] vs 多热 [B,G])

注意伪谱输出维 fr_size != 网格 G, 所以回归/rope 的高斯目标必须在 fr_size 网格上
现算 (fr_grid), 不能复用 dataset 的 G 网格目标。fr_grid 与 G 网格覆盖同一物理
频段 [f_min,f_max], 只是点数不同。
"""
import numpy as np
import torch
import torch.nn.functional as F

from win_encoder import is_regression


def make_fr_grid(cfg, fr_size):
    """伪谱输出对应的物理频率网格 [fr_size] (Hz)。"""
    fcfg = cfg["frequency"]
    return torch.linspace(fcfg["f_min"], fcfg["f_max"], fr_size)


def gaussian_fr_target(freqs_hz, n_freq, fr_grid, sigma_hz):
    """批量高斯核目标 [B,fr_size]。freqs_hz [B,K] 含 -1e4 填充。"""
    device = fr_grid.device
    freqs = freqs_hz.to(device)                         # [B,K]
    B, K = freqs.shape
    valid = (torch.arange(K, device=device)[None, :] < n_freq.to(device)[:, None])  # [B,K]
    sigma = max(float(sigma_hz), 1e-6)
    diff = fr_grid[None, None, :] - freqs[:, :, None]   # [B,K,fr]
    g = torch.exp(-(diff ** 2) / (sigma ** 2))          # [B,K,fr]
    g = g * valid[:, :, None].float()
    return g.sum(dim=1)                                 # [B,fr]


class UnifiedLoss:
    def __init__(self, cfg, fr_size=None, device="cpu"):
        self.cfg = cfg
        self.regression = is_regression(cfg)
        self.device = device
        if self.regression:
            assert fr_size is not None, "regression/rope need fr_size for target grid"
            self.fr_grid = make_fr_grid(cfg, fr_size).to(device)
            self.sigma_hz = cfg["frequency"]["target_sigma_hz"]
        else:
            pw = cfg["loss"]["bce_pos_weight"]
            self.pos_weight = None if pw is None else torch.tensor(float(pw), device=device)

    def __call__(self, output, batch):
        if self.regression:
            target = gaussian_fr_target(
                batch["freqs_hz"], batch["n_freq"], self.fr_grid, self.sigma_hz)
            return F.mse_loss(output, target)
        target = batch["multihot"].to(self.device)
        return F.binary_cross_entropy_with_logits(output, target, pos_weight=self.pos_weight)


def build_loss(cfg, fr_size=None, device="cpu"):
    return UnifiedLoss(cfg, fr_size=fr_size, device=device)
