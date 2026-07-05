"""四种局部编码器 + 统一工厂 build_encoder(cfg)。

统一 forward 约定 (main/eval 只调 encoder_forward):
    时序类 (regression_cnn/classification_cnn/rope): forward(signal[B,2,N], t_n[B,N])
    协方差类 (resfreq):                              forward(cov_channels[B,3,D,D])
输出:
    regression_cnn / rope : 伪谱 [B, fr_size]   (高斯核 MSE)
    classification_cnn    : 多热 logits [B, G]  (BCE)
    resfreq               : 多热 logits [B, G]  (BCE)

CNN/RoPE 主干移植自 comparison/deepfreq_official/{modules,rope_module}.py,
ResFreqNet 移植自 resfreq_reproduction/model/resfreq.py。此处 vendored 保持
unified_project 自包含。
"""
import math
from typing import Tuple

import torch
import torch.nn as nn

from config import grid_size, signal_dim as _signal_dim


# =================================================================== #
# DeepFreq CNN 主干 — 回归 (原生伪谱)                                  #
# =================================================================== #
class RegressionCNN(nn.Module):
    def __init__(self, signal_dim, n_filters=64, n_layers=20, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        pad = "same" if torch.__version__ >= "1.7.0" else kernel_size // 2
        mod = []
        for _ in range(n_layers):
            mod += [nn.Conv1d(n_filters, n_filters, kernel_size, padding=pad, bias=False,
                              padding_mode="circular"),
                    nn.BatchNorm1d(n_filters), nn.ReLU()]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2,
                                            output_padding=1, bias=False)

    def forward(self, signal, t_n=None):
        b = signal.size(0)
        x = self.in_layer(signal.reshape(b, -1)).view(b, self.n_filters, -1)
        x = self.mod(x)
        return self.out_layer(x).view(b, -1)


# =================================================================== #
# DeepFreq CNN 主干 — 分类 (逐位置多热头, 避开 GAP 丢位置)              #
# =================================================================== #
class ClassificationCNN(nn.Module):
    def __init__(self, signal_dim, grid_size, n_filters=64, n_layers=20, inner_dim=125,
                 kernel_size=3):
        super().__init__()
        self.grid_size = grid_size
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        pad = "same" if torch.__version__ >= "1.7.0" else kernel_size // 2
        mod = []
        for _ in range(n_layers):
            mod += [nn.Conv1d(n_filters, n_filters, kernel_size, padding=pad, bias=False,
                              padding_mode="circular"),
                    nn.BatchNorm1d(n_filters), nn.ReLU()]
        self.mod = nn.Sequential(*mod)
        # 逐位置头: 1x1 conv 压通道保留频率位置, Linear 重采样到 grid_size
        self.to_logit = nn.Conv1d(n_filters, 1, kernel_size=1)
        self.resample = nn.Linear(inner_dim, grid_size)

    def forward(self, signal, t_n=None):
        b = signal.size(0)
        x = self.in_layer(signal.reshape(b, -1)).view(b, self.n_filters, -1)
        x = self.mod(x)
        x = self.to_logit(x).view(b, -1)
        return self.resample(x)


# =================================================================== #
# RoPE-Transformer (big) — 原生伪谱回归, 真实 t_n 驱动相位              #
# =================================================================== #
def _rope_cache_from_t(t_n, head_dim, base):
    inv_freq = base ** (-torch.arange(0, head_dim, 2, device=t_n.device).float() / head_dim)
    ang = t_n[..., None] * inv_freq
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def _apply_rope(x, cos, sin):
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return x * cos + rot * sin


class _RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, t, d))


class _RoPELayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.attn = _RoPEAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, d_model))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class RoPETransformer(nn.Module):
    def __init__(self, signal_dim, d_model=256, n_heads=8, n_layers=4, ffn_dim=512,
                 base=10000.0, t_scale=1.0, n_filters=64, inner_dim=125,
                 upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.signal_dim = signal_dim
        self.n_filters = n_filters
        self.head_dim = d_model // n_heads
        self.base = base
        self.t_scale = t_scale
        self.token_embed = nn.Linear(2, d_model)
        self.layers = nn.ModuleList([_RoPELayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.chan_proj = nn.Linear(d_model, n_filters)
        self.time_resample = nn.Linear(signal_dim, inner_dim, bias=False)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2,
                                            output_padding=1, bias=False)

    def forward(self, signal, t_n=None):
        b = signal.size(0)
        x = signal.transpose(1, 2)                    # [B,T,2]
        if t_n is None:
            t_n = torch.arange(self.signal_dim, device=signal.device,
                               dtype=signal.dtype)[None, :].expand(b, -1)
        cos, sin = _rope_cache_from_t(t_n * self.t_scale, self.head_dim, self.base)
        x = self.token_embed(x)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        x = self.chan_proj(x).transpose(1, 2)         # [B,n_filters,T]
        x = self.time_resample(x)                     # [B,n_filters,inner_dim]
        return self.out_layer(x).view(b, -1)


# =================================================================== #
# ResFreq — 协方差图分类                                               #
# =================================================================== #
class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class _ResStage(nn.Module):
    def __init__(self, in_c, out_c, num_blocks, stride):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                  nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        layers += [_ResBlock(out_c) for _ in range(num_blocks)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResFreqNet(nn.Module):
    def __init__(self, grid_size, upsample_channels=8,
                 stage_channels=(64, 128, 256, 512), residual_blocks_per_stage=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, upsample_channels, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(upsample_channels), nn.ReLU(inplace=True))
        first = stage_channels[0]
        layers = [nn.Conv2d(upsample_channels, first, 3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(first), nn.ReLU(inplace=True),
                  nn.MaxPool2d(3, stride=2, padding=1)]
        in_c = first
        for i, out_c in enumerate(stage_channels):
            layers.append(_ResStage(in_c, out_c, residual_blocks_per_stage,
                                    stride=1 if i == 0 else 2))
            in_c = out_c
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(stage_channels[-1], grid_size))

    def forward(self, cov_channels, t_n=None):
        x = self.upsample(cov_channels)
        x = self.backbone(x)
        return self.head(x)


# =================================================================== #
# 工厂                                                                #
# =================================================================== #
TIME_DOMAIN_TYPES = {"regression_cnn", "classification_cnn", "rope"}
COV_DOMAIN_TYPES = {"resfreq"}


def build_encoder(cfg):
    etype = cfg["encoder"]["type"]
    signal_dim = _signal_dim(cfg)
    G = grid_size(cfg)

    if etype == "regression_cnn":
        c = cfg["encoder"]["cnn"]
        return RegressionCNN(signal_dim, c["n_filters"], c["n_layers"], c["inner_dim"],
                             c["kernel_size"], c["upsampling"], c["kernel_out"])
    if etype == "classification_cnn":
        c = cfg["encoder"]["cnn"]
        return ClassificationCNN(signal_dim, G, c["n_filters"], c["n_layers"],
                                 c["inner_dim"], c["kernel_size"])
    if etype == "rope":
        c = cfg["encoder"]["rope"]
        t_scale = c["t_scale"] if c["t_scale"] is not None else cfg["data"]["f_rot_hz"]
        return RoPETransformer(signal_dim, c["d_model"], c["n_heads"], c["n_layers"],
                               c["ffn_dim"], c["base"], t_scale, c["n_filters"],
                               c["inner_dim"], c["upsampling"], c["kernel_out"])
    if etype == "resfreq":
        c = cfg["encoder"]["resfreq"]
        return ResFreqNet(G, c["upsample_channels"], tuple(c["stage_channels"]),
                          c["residual_blocks_per_stage"])
    raise ValueError(f"unknown encoder.type={etype!r}")


def encoder_forward(model, batch, cfg, device):
    """按 encoder 域取对应输入并前向。返回模型原始输出。"""
    etype = cfg["encoder"]["type"]
    if etype in COV_DOMAIN_TYPES:
        return model(batch["cov_channels"].to(device))
    # 时序类: rope 用真实 t_n, cnn 忽略 t_n
    t_n = batch["t_n"].to(device) if etype == "rope" else None
    return model(batch["signal"].to(device), t_n)


def is_regression(cfg):
    """回归/rope 输出伪谱 (高斯 MSE); 分类/resfreq 输出多热 (BCE)。"""
    return cfg["encoder"]["type"] in {"regression_cnn", "rope"}
