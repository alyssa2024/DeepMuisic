"""取峰: 局部极大 (NMS) + top-L 选择。

三头输出的 obj 谱 [B,G] -> L 个峰的 bin 索引。配合 delta 头得到亚-bin 频率:
    mu = grid[bin] + delta[bin] * Δ
"""
import torch
import torch.nn.functional as F


def local_maxima_mask(values, min_distance):
    """[B,G] -> 局部极大掩码 [B,G] (窗宽 2*min_distance+1)。

    第二个条件 values > left 用来打破平台 (相等的连续 bin 只留最左一个),
    否则一段常数谱会被整段判为极大。
    """
    win = 2 * int(min_distance) + 1
    pooled = F.max_pool1d(values[:, None, :], kernel_size=win, stride=1,
                          padding=win // 2)[:, 0, :]
    is_max = values >= pooled
    left = F.pad(values[:, :-1], (1, 0), value=-float("inf"))
    return is_max & (values > left)


def topl_peak_indices(values, L, min_distance):
    """[B,G] -> top-L 峰的 bin 索引 [B,L] (先 NMS 再取 top-L)。

    极大点不足 L 个时用全局 argmax 补位 (保证输出形状固定, 下游不必处理变长)。
    """
    B, G = values.shape
    mask = local_maxima_mask(values, min_distance)
    masked = torch.where(mask, values, torch.full_like(values, -float("inf")))
    L_eff = min(L, G)
    idx = torch.topk(masked, k=L_eff, dim=-1).indices
    fallback = values.argmax(dim=-1, keepdim=True).expand(B, L_eff)
    valid = torch.gather(masked, 1, idx) > -float("inf")
    idx = torch.where(valid, idx, fallback)
    if L_eff < L:
        idx = torch.cat([idx, fallback[:, :1].expand(B, L - L_eff)], dim=1)
    return idx
