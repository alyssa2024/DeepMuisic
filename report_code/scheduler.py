"""学习率调度: 前 anneal_start 步恒定, 之后 cosine 退火到 lr × anneal_min_factor。"""
import math

import torch


def make_scheduler(opt, scfg):
    """前 anneal_start 步恒定 lr, 之后 cosine 退火到 lr × anneal_min_factor。

    用 LambdaLR 而非 CosineAnnealingLR: 后者从第 0 步就开始退, 这里要的是"先恒定再退"。
    lambda 返回**倍率** (作用在每个 param group 各自的 lr 上), 所以分组 lr 天然兼容。

    anneal_start >= iters ⇒ 全程恒定 (不退火)。
    """
    iters = int(scfg["iters"])
    start = int(scfg["anneal_start"])
    floor = float(scfg["anneal_min_factor"])

    def factor(it):
        if it < start or iters <= start:
            return 1.0
        prog = (it - start) / max(iters - start, 1)          # 0 -> 1
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)
