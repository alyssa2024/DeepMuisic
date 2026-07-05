"""统一评测: 四种编码器同一套指标, 按 SNR 分档 (mixed-train/separated-eval)。

输出统一映射到物理频率网格上取峰:
    回归/rope : 伪谱 [B,fr_size] -> fr_grid 取峰
    分类/resfreq: logits [B,G]    -> G 网格取峰 (sigmoid 不改峰序)
指标: recall@L (±basin_hz 命中) / peak_density / freq MAE / 成功率。
"""
import numpy as np
import torch
import torch.nn.functional as F

from config import grid_size
from win_encoder import encoder_forward, is_regression
from loss import make_fr_grid
from dataset import make_freq_grid


def _local_maxima_mask(values, min_distance):
    win = 2 * int(min_distance) + 1
    pooled = F.max_pool1d(values[:, None, :], kernel_size=win, stride=1, padding=win // 2)[:, 0, :]
    is_max = values >= pooled
    left = F.pad(values[:, :-1], (1, 0), value=-float("inf"))
    return is_max & (values > left)


def _topl_peak_indices(values, L, min_distance):
    B, G = values.shape
    mask = _local_maxima_mask(values, min_distance)
    masked = torch.where(mask, values, torch.full_like(values, -float("inf")))
    L_eff = min(L, G)
    idx = torch.topk(masked, k=L_eff, dim=-1).indices
    fallback = values.argmax(dim=-1, keepdim=True).expand(B, L_eff)
    valid = torch.gather(masked, 1, idx) > -float("inf")
    idx = torch.where(valid, idx, fallback)
    if L_eff < L:
        idx = torch.cat([idx, fallback[:, :1].expand(B, L - L_eff)], dim=1)
    return idx


@torch.no_grad()
def _metrics(peak_f, freqs_hz, n_freq, basin_hz, Ls, density):
    """peak_f [B,maxL] (物理Hz, 降序), freqs_hz [B,K] 含 -1e4 填充。"""
    device = peak_f.device
    K = freqs_hz.shape[1]
    valid = (torch.arange(K, device=device)[None, :] < n_freq[:, None])   # [B,K]
    out = {}
    for L in Ls:
        pf = peak_f[:, :L]                                                # [B,L]
        d = (pf[:, None, :] - freqs_hz[:, :, None]).abs()                # [B,K,L]
        covered = (d.min(dim=-1).values <= basin_hz) & valid             # [B,K]
        out[f"recall@{L}"] = float(covered.float().sum() / valid.float().sum().clamp_min(1))
    out["peak_density"] = float(density)
    return out


@torch.no_grad()
def evaluate(model, cfg, device, snr_db, eval_batches, seed_base=99991, loss_fn=None):
    """在固定 SNR 上评测 eval_batches 个在线批, 返回聚合指标。

    loss_fn 给定时额外算 val_loss (与训练同损失, 在该 SNR 的验证数据上)。
    """
    from dataset import UnifiedBTTDataset
    from torch.utils.data import DataLoader

    model.eval()
    ecfg = cfg["eval"]
    Ls = ecfg["recall_L"]
    basin_hz = ecfg["basin_hz"]
    bs = cfg["training"]["batch_size"]

    # 输出所在的物理频率网格
    if is_regression(cfg):
        fr_size = model.fr_size
        grid = make_fr_grid(cfg, fr_size).to(device)
    else:
        grid = torch.as_tensor(make_freq_grid(cfg), device=device)
    grid_step = float(grid[1] - grid[0])
    min_dist = max(int(round(ecfg["nms_distance_hz"] / grid_step)), 1)

    agg = {f"recall@{L}": [] for L in Ls}
    agg["peak_density"] = []
    abs_errs = []
    val_losses = []
    maxL = max(Ls)

    ds = UnifiedBTTDataset(cfg, size=bs * eval_batches, seed=seed_base + int(snr_db),
                           snr_db=snr_db)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    for batch in loader:
        output = encoder_forward(model, batch, cfg, device)      # [B, fr_size or G]
        if loss_fn is not None:
            val_losses.append(float(loss_fn(output, batch)))
        if not is_regression(cfg):
            output = torch.sigmoid(output)
        peaks = _topl_peak_indices(output, maxL, min_dist)       # [B,maxL]
        peak_f = grid[peaks]                                     # [B,maxL]
        freqs_hz = batch["freqs_hz"].to(device)
        n_freq = batch["n_freq"].to(device)

        density = float(_local_maxima_mask(output, min_dist).sum(-1).float().mean())
        m = _metrics(peak_f, freqs_hz, n_freq, basin_hz, Ls, density)
        for k in agg:
            agg[k].append(m[k])

        # freq MAE: 每样本 top-n_freq 峰匹配最近真频
        for b in range(peak_f.size(0)):
            nf = int(n_freq[b].item())
            pf = peak_f[b, :nf].cpu().numpy()
            tf = freqs_hz[b, :nf].cpu().numpy()
            if nf == 0:
                continue
            d = np.abs(pf[:, None] - tf[None, :])
            abs_errs.append(d.min(axis=0))

    result = {k: float(np.mean(v)) for k, v in agg.items()}
    if abs_errs:
        ae = np.concatenate(abs_errs)
        result["freq_mae_hz"] = float(np.mean(ae))
        result["success_rate"] = float(np.mean(ae <= ecfg["success_tol_hz"]))
    if val_losses:
        result["val_loss"] = float(np.mean(val_losses))
    return result


@torch.no_grad()
def evaluate_at_snr(model, cfg, device, snr_db, loss_fn=None):
    """单档 SNR 评测 (训练期监控用, 默认 20dB)。返回该档指标 (含 val_loss)。"""
    return evaluate(model, cfg, device, snr_db, cfg["eval"]["eval_batches"], loss_fn=loss_fn)


@torch.no_grad()
def evaluate_all_snr(model, cfg, device, loss_fn=None):
    """按每个 SNR 档分别评测, 返回 {snr: metrics} (训练结束后的完整评估)。"""
    return {snr: evaluate(model, cfg, device, snr, cfg["eval"]["eval_batches"], loss_fn=loss_fn)
            for snr in cfg["eval"]["eval_snr_db"]}
