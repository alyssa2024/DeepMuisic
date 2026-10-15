"""窗融合: M 个逐窗特征 -> 单一全局后验 (沿 M 求平均, **无参数**)。

一个 band 的 M 个逐窗后验是同一个频率的 M 次估计, 而任务要每 band 一个后验, 故沿
窗轴归约:

    axial 三头前特征 feat [B,M,G,d] --mean over M--> [B,G,d] --逐窗三头--> [B,G]

  - **无参数**: 纯代数平均, 不引入任何可训权重, 也不需要单独的训练阶段。
  - **复用逐窗三头**: obj/σ 头是 Linear, 与求平均可交换 ⇒ "先平均特征再读出" 与
    "先逐窗读出再平均 logit" **逐比特相同** (已实测 allclose)。δ 头因外套 tanh 严格
    说不可交换, 但差异极小。
  - **可变 M**: 平均对窗数无要求。

为什么不用 GRU (原实现, 99 072 参数):
    同 axial ckpt / 同 config / 同 eval seed=999000 / n=256 六档对照 (eval_fusion_mean.py):
      R@K   : mean 全档 ≥ GRU (GRU 赢 0 档, 平 2 档, 输 4 档)
      allK  : 同上, 最大差 0.008
      σcorr : 3:3 互有胜负, 无系统差异
      σ/err : 六档差异 ≤0.05, 实质打平
    ⇒ 99 072 参数换来零召回增益, 故改为无参平均, 训练阶段由 6 降为 4。

⚠ 不能把平均特征喂给旧 GRU ckpt 的全局三头: 那套头是在 GRU 末隐状态的特征分布上训的,
   喂平均特征会把 logit 幅度从 ±8.7 压到 ±1.8, 取峰全错 (实测 recall=0)。
"""
import torch
import torch.nn as nn


class WindowFusionMean(nn.Module):
    """M 个逐窗特征 [B,M,G,d] -> 沿 M 平均 (无参) -> 逐窗三头 -> 全局三头输出 [B,G]。

    heads: 提供 obj_head / delta_head / sigma_head 的模块 (即 AxialViT 本身)。
           三头权重共享自逐窗读出, 本层不新增任何参数。
    """

    def __init__(self, heads):
        super().__init__()
        self.obj_head = heads.obj_head
        self.delta_head = heads.delta_head
        self.sigma_head = heads.sigma_head

    def forward(self, feat):
        z = feat.mean(1)                                    # [B,M,G,d] -> [B,G,d] 代数平均
        obj = self.obj_head(z).squeeze(-1)                  # [B,G]
        delta = 0.5 * torch.tanh(self.delta_head(z).squeeze(-1))
        log_sigma = self.sigma_head(z).squeeze(-1)          # σ 头无界
        return obj, delta, log_sigma
