"""轴向二维 ViT (十字/TimeSformer 式): 频率轴层 + 时间轴层**交替堆叠**, 全程逐 bin。

与 factorized_vit.py 的**本质区别** —— token 粒度:
    factorized_vit: 时间轴在**窗级** token (G→1 池化后 M 个), O(M²d)。跨窗粒度=窗被压成一向量。
    axial_vit     : 时间轴在**逐 bin** token (M×G 个, 同频跨窗), O(M²·G·d)。每个频率 bin
                    独立跨窗 —— 保留了 factorized 丢掉的"逐 bin 跨窗交互"。

复杂度定位 (三者之间):
    轴分解(factorized) O(MG²d + M²d)  <  轴向(本文) O(MG²d + M²Gd)  <  全联合 O(M²G²d)
    轴向比轴分解贵 G≈201 倍 (时间轴从窗级变逐 bin), 但仍比全联合省 G 倍 (时间轴不跨频)。
    ⇒ 这是论文表 I 里缺的**中间态 baseline**: 回答"为什么不保留逐 bin 跨窗交互"。
       立论(III-B-0 补充点3)是这类交互在 BTT 基本可省 ⇒ 预期轴向 ≈ 轴分解, 贵而不更好。
       但这要 **ablation 裁定**, 不能假设 (memory: 物理上对≠有净增益)。

结构 (L 组, 每组 = 频率轴层 + 时间轴层):
    NUDFT 前端 (无参) → token [B,M,G,d] = in_proj(S) + freq_embed[g]
    for 每组:
        频率轴层: 对每个窗 m, 在 G 个 bin 内 self-attention (= FTF 的 _FreqSelfAttnLayer)
        时间轴层: 对每个 bin g, 在 M 个窗内 self-attention (窗序列位置由 tau/RoPE 提供)
    → anchor 逐 bin 三头 (obj/δ/logσ), 逐窗输出 [B,M,G]

复用 (由 AskUserQuestion 决策):
  - 频率轴层 = FTF 的 _FreqSelfAttnLayer, NUDFT 前端 / in_proj / freq_embed / FiLM / 三头
    全部照搬 FTF ⇒ 可从 R8=0.957 的 FTF ckpt **strict 加载**这些部件。
  - 时间轴层 = **新增**, 零初始化输出投影 ⇒ 残差=0 ⇒ 起点与 FTF 逐窗**逐比特一致**。
    (时间轴层是全新参数, 不在 FTF ckpt 里 ⇒ 加载用 strict=False, 只有时间轴层是 missing。)

⚠ 时间轴 attention 是逐 bin 独立的 batched attention: reshape [B,M,G,d]→[B·G,M,d],
   attention over M, 再 reshape 回来。位置=窗序列 (tau), 用 RoPE (同聚合器 _GlobalRoPEAttn)。
"""
import math

import torch
from torch import nn

from encoder import NUDFTFreqTransformerAnchor, _FreqSelfAttnLayer, FiLM

TWO_PI = 2.0 * math.pi


class _TimeAxisRoPEAttn(nn.Module):
    """时间轴 self-attention: 对每个频率 bin, 在 M 个窗内注意力。位置=累计转数 tau (RoPE)。

    照搬 aggregator._GlobalRoPEAttn 的 RoPE, 但作用在**逐 bin**的窗序列上:
      输入 x[B,G,M,d] (已把 bin 挪到 batch 维旁), attention over M (窗), 每个 bin 独立。
    ⭐ 输出投影 o **零初始化** ⇒ 时间轴层残差=0 ⇒ 起点=频率轴透传 (等价性)。
    """
    def __init__(self, d, nhead, fmin=0.1, fmax=10.0):
        super().__init__()
        self.d, self.h, self.dh = d, nhead, d // nhead
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        nn.init.zeros_(self.o.weight)          # 零初始化 → 起始残差=0 (透传)
        nn.init.zeros_(self.o.bias)
        nf = self.dh // 2
        freqs = torch.logspace(math.log10(fmin), math.log10(fmax), nf)
        self.register_buffer("rot_w", TWO_PI * freqs)

    def _rope(self, x, tau):
        # x[Bg,h,M,dh], tau[Bg,M] (每个 bin 共享同一 tau, 已 expand)
        Bg, h, M, dh = x.shape
        nf = dh // 2
        ang = self.rot_w.view(1, 1, 1, nf) * tau.view(Bg, 1, M, 1)
        cos, sin = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    def forward(self, x, tau):
        # x[Bg,M,d], tau[Bg,M] → attention over M (窗)
        Bg, M, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(Bg, M, self.h, self.dh).transpose(1, 2)
        k = k.view(Bg, M, self.h, self.dh).transpose(1, 2)
        v = v.view(Bg, M, self.h, self.dh).transpose(1, 2)
        q, k = self._rope(q, tau), self._rope(k, tau)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        o = (a.softmax(-1) @ v).transpose(1, 2).reshape(Bg, M, self.d)
        return self.o(o)


class _TimeAxisLayer(nn.Module):
    """时间轴层: pre-LN + 逐 bin 窗序列 RoPE attention + FFN。零初始化 ⇒ 起始恒等透传。"""
    def __init__(self, d, nhead, ff):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = _TimeAxisRoPEAttn(d, nhead)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        nn.init.zeros_(self.ff[-1].weight)     # FFN 也零初始化 → 整层起始残差=0
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x, tau):
        # x[B,M,G,d] → 时间轴 attention (每 bin 独立, over M)
        B, M, G, d = x.shape
        xr = x.permute(0, 2, 1, 3).reshape(B * G, M, d)     # [B·G, M, d] bin 挪到 batch
        taur = tau[:, None, :].expand(B, G, M).reshape(B * G, M)   # 每 bin 共享窗序列 tau
        xr = xr + self.attn(self.n1(xr), taur)
        xr = xr + self.ff(self.n2(xr))
        return xr.reshape(B, G, M, d).permute(0, 2, 1, 3)   # 回 [B,M,G,d]


class AxialViT(nn.Module):
    """轴向二维 ViT: 频率轴层 (复用 FTF) + 时间轴层 (新增, 逐 bin 跨窗) 交替堆叠。

    forward 契约与 factorized_vit / 训练评测口径对齐:
      输入 batch (CrossWinDataset): signal[B,M,2,N] / t_n[B,M,N] / tau[B,M] / f_rot[B]
                                    / rate_summary or rev_speeds_norm
      输出 dict: obj_logit/delta/log_sigma [B,M,G] (逐窗三头)
                 + obj0/delta0/log_sig0 [B,M,G] (纯频率轴, 时间轴层未生效时的透传对照)
    """
    def __init__(self, ftf: NUDFTFreqTransformerAnchor, cfg, n_time_layers=None):
        super().__init__()
        self.cfg = cfg
        self.G = int(ftf.grid_size)
        self.signal_dim = int(ftf.signal_dim)
        # ── 频率轴部件: 全部借用 FTF 的**同一批** module (共享参数 ⇒ ckpt strict 对齐) ──
        self.register_buffer("freqs", ftf.freqs.clone())
        self.in_proj = ftf.in_proj
        self.freq_embed = ftf.freq_embed
        self.drop = ftf.drop
        self.use_film = ftf.use_film
        self.film = ftf.film
        self.freq_layers = ftf.layers                    # ModuleList[_FreqSelfAttnLayer] × L
        self.final_norm = ftf.final_norm
        self.obj_head, self.delta_head, self.sigma_head = ftf.obj_head, ftf.delta_head, ftf.sigma_head
        # ── 时间轴层: 新增, 数量 = 频率轴层数 (交替一对一) ──
        L = len(self.freq_layers)
        n_time = L if n_time_layers is None else int(n_time_layers)
        acfg = cfg["crosswin"]["time_axis"]
        d = self.freq_embed.shape[1]
        self.time_layers = nn.ModuleList([
            _TimeAxisLayer(d, int(acfg["n_heads"]), int(acfg["ffn_dim"]))
            for _ in range(n_time)])
        # 交替调度: 前 min(L,n_time) 组交替, 多出的频率轴层单独跑 (无配对时间轴层)
        self.n_pair = min(L, n_time)

    @staticmethod
    def _apply_head(h, layer):
        return layer(h).squeeze(-1)                      # h[B,M,G,d] → [B,M,G]

    def _nudft(self, yr, yi, t_n):
        # [B·M,N] → S[B·M,G,2] → token[B·M,G,d]. 与 FTF forward 里的 NUDFT 逐比特一致。
        phase = 2 * math.pi * self.freqs[None, :, None] * t_n[:, None, :]
        cos, sin = torch.cos(phase), torch.sin(phase)
        sr = (yr[:, None, :] * cos + yi[:, None, :] * sin).sum(-1)
        si = (yi[:, None, :] * cos - yr[:, None, :] * sin).sum(-1)
        s = torch.stack([sr, si], dim=-1) / math.sqrt(self.signal_dim)
        return self.in_proj(s) + self.freq_embed[None]

    def _freq_tokens(self, batch, device):
        """逐窗展平走 NUDFT + FiLM, 得频率 token [B,M,G,d]。"""
        from dataset import film_input           # film_input 在 dataset.py
        from encoder import _FILM_KEY
        B, M = batch["signal"].shape[:2]
        sig = batch["signal"].to(device).reshape(B * M, 2, -1)
        t_n = batch["t_n"].to(device).reshape(B * M, -1)
        yr, yi = sig[:, 0], sig[:, 1]
        x = self._nudft(yr, yi, t_n)                     # [B·M,G,d]
        if self.use_film and self.cfg["encoder"]["rate_inject"]:
            rk = _FILM_KEY[film_input(self.cfg)]
            rate = batch[rk].to(device)
            rate = rate.reshape(B * M, *rate.shape[2:])
            x = self.film(x.transpose(1, 2), rate).transpose(1, 2)
        x = self.drop(x)
        return x.reshape(B, M, self.G, -1)

    def _heads(self, x_freq):
        # x_freq[B,M,G,d] → 三头 [B,M,G]
        obj = self._apply_head(x_freq, self.obj_head)
        delta = 0.5 * torch.tanh(self._apply_head(x_freq, self.delta_head))
        log_sigma = self._apply_head(x_freq, self.sigma_head)     # σ 头无界
        return obj, delta, log_sigma

    def forward(self, batch, device):
        B, M = batch["signal"].shape[:2]
        tau = batch["tau"].to(device)                    # [B,M] 窗序列位置
        x = self._freq_tokens(batch, device)             # [B,M,G,d]

        # ── 纯频率轴透传输出 (对照: 时间轴层零初始化时应与此逐比特一致) ──
        x0 = x
        for fl in self.freq_layers:
            xf = x0.reshape(B * M, self.G, -1)
            x0 = fl(xf).reshape(B, M, self.G, -1)
        x0 = self.final_norm(x0)
        obj0, delta0, log_sig0 = self._heads(x0)

        # ── 交替堆叠: 频率轴层 → 时间轴层 ──
        for i in range(len(self.freq_layers)):
            xf = x.reshape(B * M, self.G, -1)
            x = self.freq_layers[i](xf).reshape(B, M, self.G, -1)   # 频率轴 (同窗 G bin)
            if i < self.n_pair:
                x = self.time_layers[i](x, tau)                     # 时间轴 (同频 M 窗)
        x = self.final_norm(x)
        obj_t, delta_t, log_sig_t = self._heads(x)
        return {"obj_logit": obj_t, "delta": delta_t, "log_sigma": log_sig_t,
                "obj0": obj0, "delta0": delta0, "log_sig0": log_sig0,
                # ⭐ 三头前的逐窗特征 [B,M,G,d] (final_norm 后) —— 供可学融合层
                #   (RNN over M) 把 M 个逐窗后验融合成单一全局后验。
                "feat": x}


def load_axial_vit(cfg, grid, grid_step, enc_ckpt, device="cpu", n_time_layers=None):
    """建轴向 ViT 并加载 FTF 权重 (频率轴部件 strict, 时间轴层 missing 是预期的)。

    时间轴层零初始化 ⇒ 起点 = FTF 逐窗透传 (obj_t=obj0)。冒烟脚本验这一点。
    """
    from encoder import build_encoder
    ftf = build_encoder(cfg).to(device)
    sd = torch.load(enc_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    ftf.load_state_dict(sd, strict=True)                 # FTF 本体 strict
    model = AxialViT(ftf, cfg, n_time_layers=n_time_layers).to(device)
    return model
