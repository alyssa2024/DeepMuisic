"""损失: anchor 三头 (单窗/逐窗展平) + 跨窗聚合器包装。

单窗三头:
    stage1  L_obj(加权 BCE) + L_delta(δ 的 L2)   训检测器 + μ 精度 (σ 头无梯度)
    stage2  纯 GNLL (只在 argmax bin)            冻主干+obj+δ, 只校准 σ

跨窗 (阶段三):
    L = anchor_loss(逐窗展平, stage1) + β·GNLL
    ⚠ 主损失**必须**是 anchor 损失, 不能换成纯似然 —— 似然只给"被采中的峰"梯度,
      obj 头无监督 ⇒ 弱峰的 objectness 会塌 (实测召回 0.771 → 0.513, 40 步陡降)。
    ⚠ 当前配置 β=0 ⇒ 实际只有 anchor 项。
"""
import torch
import torch.nn.functional as F

from dataset import make_freq_grid


# =================================================================== #
# 标签构造                                                            #
# =================================================================== #
def build_anchor_targets(freqs_hz, n_freq, amps, grid_hz, obj_sigma_bins=1.0):
    """batch 标签。freqs_hz/amps [B,K] (填充 freq=-1e4/amp=0), grid_hz [G] tensor。

    Returns dict:
      obj_soft   [B,G]  邻域高斯软标签 (多真频 max 聚合) -> objectness BCE 目标
      argmax_bin [B,K]  第 k 真频最近的 bin (填充位 = -1)
      delta_true [B,K]  (f_k - grid[argmax_bin])/Δ ∈ [-0.5,0.5] (填充位 = 0)
      valid      [B,K]  真频有效位
    """
    device = grid_hz.device
    G = grid_hz.shape[0]
    gs = float(grid_hz[1] - grid_hz[0])
    f = freqs_hz.to(device)                                    # [B,K]
    a = amps.to(device)
    B, K = f.shape
    valid = (torch.arange(K, device=device)[None, :] < n_freq.to(device)[:, None])  # [B,K]

    # 每真频最近 bin + bin 内偏移 (min_sep 6Hz / grid 2Hz -> 一个 bin 至多一个真频)
    d = (grid_hz[None, None, :] - f[:, :, None]).abs()         # [B,K,G]
    argmax_bin = d.argmin(dim=-1)                              # [B,K]
    bin_center = grid_hz[argmax_bin]                           # [B,K]
    delta_true = ((f - bin_center) / gs).clamp(-0.5, 0.5)      # [B,K] 单模态, 好学
    argmax_bin = torch.where(valid, argmax_bin, torch.full_like(argmax_bin, -1))
    delta_true = torch.where(valid, delta_true, torch.zeros_like(delta_true))

    # objectness 标签: obj_sigma_bins>0 -> 邻域高斯软标签; <=0 -> 硬 one-hot
    bins = torch.arange(G, device=device).float()             # [G]
    ab = argmax_bin.clamp(min=0).float()                      # [B,K]
    if obj_sigma_bins > 0:
        g = torch.exp(-((bins[None, None, :] - ab[:, :, None]) ** 2) / (2 * obj_sigma_bins ** 2))
    else:
        g = (bins[None, None, :] == ab[:, :, None]).float()
    g = g * valid[:, :, None].float()                         # [B,K,G]
    obj_soft = g.max(dim=1).values                            # [B,G]

    return {"obj_soft": obj_soft, "argmax_bin": argmax_bin,
            "delta_true": delta_true, "valid": valid}


# =================================================================== #
# 单窗: anchor 三头损失                                                #
# =================================================================== #
def anchor_loss(out, tgt, stage="stage1"):
    """out=dict{obj_logit, delta, log_sigma} [B,G]; tgt=build_anchor_targets 的输出。

    stage1  obj BCE + δ 的 L2 (无 σ/GNLL)   -> 训好检测器 + μ 精度
    stage2  纯 GNLL (δ/σ 头, 只在 argmax bin -> σ 不被 G-K 个空 bin 稀释)

    返回 (total, parts)。parts 的键随 stage 变: stage1 -> {l_obj,l_delta};
    stage2 -> {l_reg}。训练脚本据此逐项报 loss。
    """
    obj_logit, delta, log_sigma = out["obj_logit"], out["delta"], out["log_sigma"]
    vb = tgt["valid"]
    ab = tgt["argmax_bin"].clamp(min=0)                        # [B,K]
    nvalid = vb.float().sum().clamp_min(1.0)

    delta_g = torch.gather(delta, 1, ab)                       # [B,K] argmax bin 的 δ
    err2 = (delta_g - tgt["delta_true"]) ** 2                  # [B,K] (bin 单位, 单模态)

    if stage == "stage1":
        l_obj = F.binary_cross_entropy_with_logits(obj_logit, tgt["obj_soft"])
        l_delta = (err2 * vb.float()).sum() / nvalid
        return l_obj + l_delta, {"l_obj": float(l_obj.detach()),
                                 "l_delta": float(l_delta.detach())}

    s_g = torch.gather(log_sigma, 1, ab)                       # [B,K] argmax bin 的 logσ
    gnll = s_g + err2 / (2 * torch.exp(2 * s_g))               # [B,K]
    l_reg = (gnll * vb.float()).sum() / nvalid

    # stage2: 纯 GNLL。**GNLL 是有监督的**: err2 = (δ − δ_true)², 对 σ 求导令零
    # ⇒ σ* = |δ − δ_true|, 即 σ 的"真值"就是 δ 的绝对误差。
    # 冻结主干+obj+δ ⇒ σ 只能从**已有特征**被动读误差, 点估计按构造不变。
    return l_reg, {"l_reg": float(l_reg.detach())}


# =================================================================== #
# 跨窗: 聚合器损失 (阶段三)                                            #
# =================================================================== #
class AggregatorLoss:
    """跨窗三头 [B,M,G] -> anchor(逐窗展平) + β·GNLL。"""

    def __init__(self, cfg, grid, device="cpu"):
        acfg = cfg["encoder"]["anchor"]
        tcfg = cfg["crosswin"]["train"]
        self.grid = grid
        self.device = device
        _gs = float(cfg["frequency"]["grid_step_hz"])
        _sig_hz = acfg["obj_sigma_hz"]
        self.obj_sigma_bins = (float(_sig_hz) / _gs if _sig_hz is not None
                               else float(acfg["obj_sigma_bins"]))
        self.beta_gnll = float(tcfg["beta_gnll"])

    def __call__(self, obj_t, delta_t, log_sig_t, batch):
        """三头 [B,M,G] + 跨窗 batch -> (total, dict 分量)。"""
        B, M, G = obj_t.shape
        dev = self.device
        # 逐窗展平成 [B*M] 个"伪单窗", 复用单窗 anchor 损失 -> 保住读峰性质
        fw = batch["f_win"].to(dev).reshape(B * M, -1)                    # [B*M,K] 逐窗真频
        amp = batch["amp_true"].to(dev)[:, None, :].expand(B, M, -1).reshape(B * M, -1)
        nf = batch["n_freq"].to(dev).repeat_interleave(M)
        tgt = build_anchor_targets(fw, nf, amp, self.grid,
                                   obj_sigma_bins=self.obj_sigma_bins)
        out = {"obj_logit": obj_t.reshape(B * M, G),
               "delta": delta_t.reshape(B * M, G),
               "log_sigma": log_sig_t.reshape(B * M, G)}

        l_anchor, sub = anchor_loss(out, tgt, stage="stage1")
        l_gnll, _ = anchor_loss(out, tgt, stage="stage2")

        total = l_anchor + self.beta_gnll * l_gnll
        # ⭐ 直接给出 l_obj / l_delta 两个分项 (不合并成 anchor), 让训练脚本逐项报。
        parts = {**sub, "gnll": float(l_gnll.detach())}
        return total, parts


def build_aggregator_loss(cfg, grid, device="cpu"):
    return AggregatorLoss(cfg, grid, device=device)
