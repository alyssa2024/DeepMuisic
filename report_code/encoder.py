"""单窗编码器 FTF: NUDFT 频率-token Transformer + anchor 三头。

    y, t_n  --显式 NUDFT (每个 bin 一个频率基, 无参, 物理精确)-->  S_g ∈ C, g=1..G
            [Re,Im] --Linear(2→d)--> + freq_embed[g] --> token_g [B,G,d]
            --FiLM(rate_summary 4维)--> L 层频率轴 self-attention --> [B,G,d]
            --> 三个 per-bin 头 Linear(d→1): obj_logit / δ / logσ

为什么是这个架构 (对比已证死的路线):
  - 1D conv backbone + MLP 回归 K 个标量: **死** (Linear 对输入位置轴置换不变, 位置被
    打散; frozen 主干上四种回归头 MAE 30-47Hz)。
  - 每个频率 bin 天生是一个 token, 位置身份由 token 自身携带 (freq_embed), 三个头是
    per-token 的 Linear(d→1) —— 全程零压缩、零重采样, 绕开上面那个死因。
  - 显式 NUDFT 直接给出每 bin 的频域证据 (不让 attention 去学隐式 NUDFT);
    self-attention 在**频率轴**上压制旁瓣/混叠 —— BTT 的 ±k·f_rot 混叠是长程的,
    attention 一层即全局。

统一 forward 约定: 只调 encoder_forward(model, batch, cfg, device), **不要手拼输入** ——
FiLM 吃 rate_summary(4维), 漏了会 OOD 归零 (实测 recall 0.96→0.50)。
"""
import math

import torch
import torch.nn as nn

from config import grid_size, signal_dim as _signal_dim
from dataset import film_input, film_input_dim, rate_ref_hz


class FiLM(nn.Module):
    """转速摘要 [B,S] -> per-channel γ/β 调制特征图 x[B,C,L]。恒速档摘要≈0 -> γ≈1,β≈0。

    必要性: NUDFT 的 t_n 已含转速, 但窗长 T=n_rev/f_rot 随样本变 -> 峰宽 (Rayleigh=1/T)
    随样本变 1.56-2.81Hz, 而 freq_embed 是全样本共享的静态参数, 学不到"这个样本的峰该多宽"。
    **相位对齐的是频率基, 不是峰宽** —— 所以显式 NUDFT 仍然需要 FiLM 注入绝对转速。
    代价 1.7 万参数, 换回 fluct 缺口的 54-63% (+0.054 R@8)。
    """

    def __init__(self, summary_dim, n_channels, hidden=64):
        super().__init__()
        self.C = n_channels
        self.mlp = nn.Sequential(
            nn.Linear(summary_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * n_channels))
        # 末层零初始化 -> 起始 γ=1,β=0 (不扰动), 让网络逐步学转速调制
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, summary):                 # x:[B,C,L], summary:[B,S]
        gb = self.mlp(summary)                      # [B,2C]
        gamma, beta = gb[:, :self.C], gb[:, self.C:]
        return (1.0 + gamma[..., None]) * x + beta[..., None]


class _FreqSelfAttnLayer(nn.Module):
    """频率轴上的 self-attention (无 RoPE, 位置由可学 freq_embed 提供)。"""

    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, d_model))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class NUDFTFreqTransformerAnchor(nn.Module):
    """FTF: 频率 token + 频率轴 self-attention + 三个 per-bin 头 (obj / δ / logσ)。

    三个头的分工:
      obj_logit [B,G]  per-bin objectness (sigmoid 独立判断) -> 读峰/召回
      delta     [B,G]  亚-bin 偏移 ∈[-0.5,0.5] -> μ = grid[g] + δ_g·Δ, 突破网格量化
      log_sigma [B,G]  该 bin 的频率不确定度 -> 逆方差融合 / ELBO 初值 σ0

    σ 头**无界** (log_sigma = 原始输出, 不夹范围)。物件对照: 无界 corr 0.60 / σ-err 1.62
    稳定, sigmoid 夹范围仅 0.52 且会回落到 0.45 / 1.98 (梯度饱和 + 丢动态范围)。
    """

    def __init__(self, signal_dim, grid_size, f_lo=0.0, f_hi=500.0,
                 d_model=128, n_heads=8, n_layers=4, ffn_dim=256,
                 dropout=0.0, use_film=True, summary_dim=4):
        super().__init__()
        self.grid_size = grid_size
        self.signal_dim = signal_dim
        # 频率基 = 输出网格本身 (每 bin 一个 token), 固定不可学 (物理定义)
        self.register_buffer("freqs", torch.linspace(f_lo, f_hi, grid_size))
        self.in_proj = nn.Linear(2, d_model)                 # [Re,Im] -> d
        self.freq_embed = nn.Parameter(torch.randn(grid_size, d_model) * 0.02)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_film = bool(use_film)
        self.film = FiLM(summary_dim, d_model) if self.use_film else None
        self.layers = nn.ModuleList([_FreqSelfAttnLayer(d_model, n_heads, ffn_dim)
                                     for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        # per-bin 头: Linear(d,1) 只看本 bin。频率轴 self-attention 已让每个 token 看到
        # 全局所有 bin, 头上再加宽卷积是多余的 (CNN-anchor 那条线因主干是局部卷积才需要)。
        self.obj_head = nn.Linear(d_model, 1)
        self.delta_head = nn.Linear(d_model, 1)
        self.sigma_head = nn.Linear(d_model, 1)

    def forward(self, signal, t_n=None, rate_summary=None):
        b = signal.size(0)
        yr, yi = signal[:, 0], signal[:, 1]                  # [B,N]
        if t_n is None:
            t_n = torch.arange(self.signal_dim, device=signal.device,
                               dtype=signal.dtype)[None, :].expand(b, -1)
        # 显式 NUDFT: 每个网格频率一个复系数 (物理精确, 无参)
        phase = 2 * math.pi * self.freqs[None, :, None] * t_n[:, None, :]    # [B,G,N]
        cos, sin = torch.cos(phase), torch.sin(phase)
        sr = (yr[:, None, :] * cos + yi[:, None, :] * sin).sum(-1)           # [B,G]
        si = (yi[:, None, :] * cos - yr[:, None, :] * sin).sum(-1)           # [B,G]
        s = torch.stack([sr, si], dim=-1) / math.sqrt(self.signal_dim)       # [B,G,2] 归一
        x = self.in_proj(s) + self.freq_embed[None]                          # [B,G,d]
        if self.use_film and rate_summary is not None:
            # FiLM 在 d 维调制 (期望 [B,C,L]); 逐 token 同一 γ/β, 不破坏频率位置身份
            x = self.film(x.transpose(1, 2), rate_summary).transpose(1, 2)   # [B,G,d]
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)                                     # 频率轴 self-attention
        x = self.final_norm(x)                               # [B,G,d]
        obj_logit = self.obj_head(x).squeeze(-1)             # [B,G]
        delta = 0.5 * torch.tanh(self.delta_head(x).squeeze(-1))
        # σ 头无界: log_sigma 直接取原始输出, 不夹范围
        log_sigma = self.sigma_head(x).squeeze(-1)
        return {"obj_logit": obj_logit, "delta": delta, "log_sigma": log_sigma}


def build_encoder(cfg):
    """cfg -> FTF 编码器。工厂形式让 ckpt 里的 config.json 能原样回放。"""
    fcfg = cfg["frequency"]
    acfg = cfg["encoder"]["anchor"]
    t = cfg["encoder"]["ftf"]
    # ⚠ rate_ref_hz 必须 ≈ f_rot_range 的中值。它把绝对转速压到 O(1) (abs_mean 除以它)。
    #   参考值错了 -> FiLM 输入分布偏移 -> 转速这一路实质失效, 而且**静默**。
    frr = cfg["data"]["f_rot_range"]
    if frr is not None:
        ref, mid = rate_ref_hz(cfg), 0.5 * (frr[0] + frr[1])
        if abs(ref - mid) > 0.15 * max(mid, 1e-6):
            raise ValueError(
                f"data.rate_ref_hz={ref} 与 f_rot_range={frr} 的中值 {mid} 相差过大。"
                f"它是 abs_mean 的归一化参考, 必须≈中值, 否则 FiLM 输入尺度错乱。")
    return NUDFTFreqTransformerAnchor(
        signal_dim=_signal_dim(cfg), grid_size=grid_size(cfg),
        f_lo=fcfg["f_min"], f_hi=fcfg["f_max"],
        d_model=int(t["d_model"]), n_heads=int(t["n_heads"]),
        n_layers=int(t["n_layers"]), ffn_dim=int(t["ffn_dim"]),
        dropout=float(t["dropout"]),
        use_film=bool(cfg["encoder"]["rate_inject"]),
        summary_dim=film_input_dim(cfg))


# FiLM 输入模式 -> batch 里对应的键
_FILM_KEY = {"summary": "rate_summary",        # [B,4]      窗内统计摘要
             "per_rev": "rev_speeds_norm"}     # [B,n_rev]  逐圈归一化转速


def encoder_forward(model, batch, cfg, device):
    """唯一的前向入口。

    FTF 吃三样: signal[B,2,N] / t_n[B,N] (进 NUDFT 相位) / 转速 (进 FiLM)。

    ⚠ **不要手拼输入。** 漏掉转速那一路会让 FiLM 收到 None 而**静默跳过** —— 不报错,
      只是 OOD 归零 (实测 recall 0.96 → 0.50)。跨窗那边尤其容易踩。
    """
    signal = batch["signal"].to(device)
    t_n = batch["t_n"].to(device)
    rate = None
    if cfg["encoder"]["rate_inject"]:
        rate = batch[_FILM_KEY[film_input(cfg)]].to(device)
    return model(signal, t_n, rate_summary=rate)
