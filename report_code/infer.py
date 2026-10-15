"""端到端推理: 一条序列 -> K 个频率及其不确定度 + 复幅值及其不确定度。

**纯推理, 不做评测** —— 全程不读真值 (f_win / camp_re / camp_im), 所以同一套代码
既能跑合成数据, 也能跑真实 BTT 数据 (只需提供 y 与 t_n)。

pipeline (对应论文两阶段):
  第一阶段 (网络, 已训好):
    ① 序列切窗 -> axial 编码器 (频率轴 + 逐 bin 时间轴) -> 三头前特征 [B,M,G,d]
    ② 窗融合: 沿 M 求平均 (**无参**) -> 逐窗三头 -> 每 band 一个全局后验 [B,G]
    ③ 取 ρ_g 最大的 Kc = K + M_extra 个 band 作候选 -> (μ_0, σ_0)
  第二阶段 (逐实例变分, 不训练网络):
    ④ 以 (μ_0, σ_0) 为初值, Adam 最大化解析边际 ELBO
       自由变量 λ = {μ_k, σ_k, τ_k², σ_n²}; σ_n² 自由 ⇒ **不需要已知 SNR**
    ⑤ 幅值后验闭式: ĉ = S₀⁻¹Φᴴy/σ_n², 不确定度 [S₀⁻¹]_kk
    ⑥ 按 |ĉ_k| 取最大的 K 个 (边际似然无 L1, 假峰靠后验幅值判别)

用法:
  python infer.py <axial_ckpt> <config.json> [n_seq] [snr_db] [steps]
  (融合无参 ⇒ 不再需要 fusion_ckpt; 兼容旧式调用, 第二个参数给 .pt 会被忽略)

输出: 打印每条序列的 K 个估计, 并落盘 inference.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_ckpt_config
from crosswin_data import CrossWinDataset, n_windows
from dataset import make_freq_grid
from axial_vit import AxialViT
from encoder import build_encoder
from elbo_marginal import refine_marginal_batch
from peaks import topl_peak_indices
from window_fusion import WindowFusionMean

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLE = int(os.environ.get("N_SAMPLE", 16))     # MC 重参数样本数 R
N_EXTRA = int(os.environ.get("N_EXTRA", 2))        # 候选数 Kc = K + M_extra
SEED = int(os.environ.get("SEED", 999901))         # out-of-training

# 全局推理上下文 —— 由 setup(cfg) 填充。
# ⚠ 模块级不解析 sys.argv: 本模块会被 sweep_snr_elbo.py import 复用 pipeline 函数。
cfg = K = Kc = grid = G = D = M = min_dist = d_model = None


def setup(cfg_in):
    """设定推理上下文 (网格 / K / Kc / NMS 间距等)。import 复用时先调它。"""
    global cfg, K, Kc, grid, G, D, M, min_dist, d_model
    cfg = cfg_in
    cfg["data"]["freq_drift"] = False              # 融合层前提: M 窗估计同一个量
    K = int(cfg["signal"]["num_components"])
    Kc = K + N_EXTRA
    grid = torch.as_tensor(make_freq_grid(cfg), dtype=torch.float32, device=DEV)
    G = grid.numel(); D = float(grid[1] - grid[0]); M = n_windows(cfg)
    min_dist = max(int(round(cfg["eval"]["nms_distance_hz"] / D)), 1)
    d_model = int(cfg["encoder"]["ftf"]["d_model"])
    return cfg


# =================================================================== #
# 第一阶段: 网络前向 -> 候选 (μ_0, σ_0)                                 #
# =================================================================== #
def load_stage1(axial_ck, fusion_ck=None):
    """载入阶段四 axial (只做前向) + 无参窗融合。

    fusion_ck 保留为可选形参仅为兼容旧式调用, 已不再使用 —— 融合是沿 M 的代数平均,
    三头直接复用 axial 的逐窗三头, 无独立权重。
    """
    ftf = build_encoder(cfg).to(DEV)
    axial = AxialViT(ftf, cfg).to(DEV)
    sd = torch.load(axial_ck, map_location=DEV, weights_only=False)
    axial.load_state_dict(sd.get("model_state", sd), strict=True)
    axial.eval()
    fusion = WindowFusionMean(axial).to(DEV)      # 无参, 三头共享自 axial
    fusion.eval()
    return axial, fusion


# SUM_BASELINE=1: 无参 sum 基线 —— **绕开时间轴 attention**, 直接平均逐窗后验。
#   ⚠ 与默认路径的区别只有一处: 默认吃 out["feat"] (已被时间轴层跨窗混过的特征),
#     基线吃 out["obj0"/"delta0"/"log_sig0"] (纯频率轴逐窗三头, 窗间零交互)。
#     两者共用**同一个 ckpt、同一批窗、同一套三头权重**, 故差值 = 时间轴建模的净贡献。
#   ⚠ 平均在**逐窗 logit** 上做 (obj/log_sigma 是 Linear 输出, 与 mean 可交换 ⇒ 与
#     "先平均特征再读出"等价; δ 外套 tanh 严格说不可交换, 但 window_fusion.py 已实测
#     差异极小)。这正是 window_fusion 文档里"先逐窗读出再平均 logit"的那条路径。
SUM_BASELINE = os.environ.get("SUM_BASELINE", "0").lower() in ("1", "true", "yes")


# ⭐ NUDFT_INIT=1: 后验初值**完全不用 transformer**, 改用无参 NUDFT 谱取峰。
#   动机: 隔离"编码器学到的先验"与"ELBO 精修本身"的贡献 —— 网络只要不参与初值,
#   剩下的就是纯物理前端 (NUDFT) + 变分精修。
#   ⚠ 三样输出各自的来源与网络版不同, 必须逐项说明:
#     mu0  = NUDFT 幅度谱 |S_g| 取峰 (NMS + top-Kc) 的 bin 中心。**无 delta 头**
#            ⇒ 没有亚-bin 修正, 初值量化到网格 (D Hz), 比网络版粗一个量级;
#            ELBO 精修会把它拉到亚-bin, 所以这不是致命的, 但低 SNR 下初值更差。
#     sig0 = 恒定 D/2 (半个 bin)。**无 log_sigma 头** ⇒ 只能给一个与"网格量化
#            不确定度"相称的宽度, 而不是逐分量自适应的方差。
#     rho  = |S_g| 归一化到 (0,1) 的相对强度。**不是概率** —— 网络的 rho 是
#            sigmoid(obj) 有校准语义, 这里只是把谱高压到 (0,1) 当软权重。
#            ⇒ 用 NUDFT_INIT 时通常应同时设 TEX_RHO_UNIFORM=1, 否则送进
#            _subset_score 的 logit 是个没有概率含义的量。
NUDFT_INIT = os.environ.get("NUDFT_INIT", "0").lower() in ("1", "true", "yes")


@torch.no_grad()
def nudft_spectrum(batch):
    """无参 NUDFT 幅度谱 [B,G] —— 逐窗算再沿 M 平均 (与网络前端同一套基)。

    与 axial_vit._nudft 的相位/归一化逐位一致 (2*pi*f*t, 除 sqrt(N)), 只是
    不过 in_proj/freq_embed, 直接取 |S|。
    """
    import math
    # batch 键与 axial_vit._freq_tokens 一致: signal [B,M,2,N] / t_n [B,M,N]
    sig = batch["signal"].to(DEV)                     # [B,M,2,N]
    B, Mw = sig.shape[:2]
    N = sig.shape[-1]
    yr = sig[:, :, 0].reshape(B * Mw, N)
    yi = sig[:, :, 1].reshape(B * Mw, N)
    tt = batch["t_n"].to(DEV).reshape(B * Mw, N)
    phase = 2 * math.pi * grid[None, :, None] * tt[:, None, :]
    cos, sin = torch.cos(phase), torch.sin(phase)
    sr = (yr[:, None, :] * cos + yi[:, None, :] * sin).sum(-1)
    si = (yi[:, None, :] * cos - yr[:, None, :] * sin).sum(-1)
    mag = torch.sqrt(sr ** 2 + si ** 2) / math.sqrt(N)       # [B*M,G]
    return mag.reshape(B, Mw, G).mean(1)                     # [B,G] 沿窗平均


@torch.no_grad()
def encode_candidates_nudft(batch):
    """NUDFT 取峰给 (mu0, log sig0, rho) —— 不碰 transformer。见 NUDFT_INIT。"""
    mag = nudft_spectrum(batch)                       # [B,G]
    pk = topl_peak_indices(mag, Kc, min_dist)         # [B,Kc]
    mu0 = grid[pk]                                    # 无 delta 头 ⇒ 落在 bin 中心
    sig0 = torch.full_like(mu0, 0.5 * D)              # 半个 bin 的量化不确定度
    # rho: 谱高归一到 (0,1) 的**相对强度**, 非概率 (见上面的 ⚠)
    v = torch.gather(mag, 1, pk)
    vmax = v.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    rho = (v / vmax).clamp(1e-6, 1.0 - 1e-6)
    return mu0, sig0.log(), rho


@torch.no_grad()
def encode_candidates(axial, fusion, batch):
    """网络前向 -> Kc 个候选的 (μ_0, σ_0) 与 band 概率 ρ。

    窗融合 (沿 M 平均) 每 band 直接给出单一全局后验 (论文式 globalpost)。
    SUM_BASELINE=1 时改走无参 sum 基线 (逐窗后验直接平均, 无时间轴交互)。
    """
    out = axial(batch, DEV)
    if SUM_BASELINE:
        # 纯频率轴逐窗三头 [B,M,G] -> 沿 M 平均 -> [B,G]; 时间轴层完全不参与。
        obj = out["obj0"].mean(1)
        delta = out["delta0"].mean(1)
        log_sigma = out["log_sig0"].mean(1)
    else:
        obj, delta, log_sigma = fusion(out["feat"])          # [B,G] 全局三头
    rho = torch.sigmoid(obj)                                 # [B,G] band 概率
    pk = topl_peak_indices(rho, Kc, min_dist)                # [B,Kc] 候选 band
    mu0 = grid[pk] + torch.gather(delta, 1, pk) * D          # [B,Kc] 亚-bin 频率
    sig0 = torch.gather(log_sigma, 1, pk).exp() * D                     # [B,Kc] Hz
    if not NOCLIP:                      # 历史行为: σ 下限 1e-4 Hz
        sig0 = sig0.clamp_min(1e-4)
    return mu0, sig0.log(), torch.gather(rho, 1, pk)


# =================================================================== #
# 第二阶段: 逐实例边际 ELBO 精修                                        #
# =================================================================== #
# TEX_NOCLIP=1 -> 去掉推理路径上的全部范围限制 (σ 下限 / ρ clip / τ²,σ_w² 下限)。
# 用于检验这些限制是在做数值保护, 还是在静默掩盖参数塌陷。
NOCLIP = os.environ.get("TEX_NOCLIP", "1").lower() in ("1", "true", "yes")

LR_MU = float(os.environ.get("LR_MU", 3e-2))
LR_SIGMA = float(os.environ.get("LR_SIGMA", 3e-1))   # logσ 收敛慢, 默认 10× lr_mu


# 精修的 MC 重参数采样固定 seed ⇒ 同输入必得同输出。
# ⚠ 低 SNR 下弱分量常卡在 keep/drop 边界, 不固定 seed 时同一条序列多次跑会给出
#   不同的 hit 数 (实测 -15dB 同配置出过 4/6 和 5/6)。REFINE_SEED= 置空则退回随机。
REFINE_SEED = os.environ.get("REFINE_SEED", "1234")


def refine(mu0, ls0, y_full, t_full, steps, verbose=True):
    gen = None
    if REFINE_SEED != "":
        gen = torch.Generator(device=mu0.device)
        gen.manual_seed(int(REFINE_SEED))
    return refine_marginal_batch(mu0, ls0, y_full, t_full, beta_ent=1.0,
                                 n_steps=steps, lr=LR_MU, lr_sigma=LR_SIGMA,
                                 lr_hyp=3e-2, n_sample=N_SAMPLE, generator=gen,
                                 verbose=verbose, grid_step_hz=D)


def select_topk(r, k):
    """按后验幅值 |ĉ_k| 取最大的 k 个 (边际似然无 L1, 假峰不会被压到 0)。

    返回按频率升序排列的 dict of [B,k] 张量。
    """
    idx = r["amp"].argsort(dim=-1, descending=True)[:, :k]           # [B,k]
    sel = {n: torch.gather(r[n], 1, idx)
           for n in ("mu", "sigma", "amp", "amp_sigma")}
    order = sel["mu"].argsort(dim=-1)                                 # 频率升序, 便于阅读
    return {n: torch.gather(v, 1, order) for n, v in sel.items()}


# =================================================================== #
def main():
    # 新式: <axial> <config> [...]; 旧式: <axial> <fusion.pt> <config> [...] (fusion 忽略)
    args = sys.argv[1:]
    AXIAL_CK = args[0]
    if len(args) > 1 and args[1].endswith(".pt"):
        args = [args[0]] + args[2:]                  # 丢掉旧式的 fusion_ckpt
    CFG = args[1]
    N_SEQ = int(args[2]) if len(args) > 2 else 8
    SNR = float(args[3]) if len(args) > 3 else 20.0
    STEPS = int(args[4]) if len(args) > 4 else 600
    setup(load_ckpt_config(CFG))
    axial, fusion = load_stage1(AXIAL_CK)
    print(f"[ckpt] axial  = {AXIAL_CK}")
    print(f"[融合] 沿 M 平均 (无参, 三头复用 axial 逐窗三头)")
    print(f"[配置] K={K} Kc={Kc}(+{N_EXTRA}) G={G} M={M} | steps={STEPS} R={N_SAMPLE}")
    print(f"[数据] 合成 {N_SEQ} 条序列 | SNR={SNR}dB | seed={SEED}\n")

    ds = CrossWinDataset(cfg, N_SEQ, seed=SEED, snr_db=SNR)
    batch = next(iter(DataLoader(ds, batch_size=N_SEQ, num_workers=0)))
    yf = batch["y_full"].to(DEV)
    y_full = torch.complex(yf[:, 0], yf[:, 1])               # [B,N_seq] 复序列
    t_full = batch["t_full"].to(DEV)                         # [B,N_seq] 采样时刻

    t0 = time.time()
    mu0, ls0, rho = encode_candidates(axial, fusion, batch)
    t_enc = time.time() - t0
    print(f"[第一阶段] 网络前向 {t_enc*1000:.0f}ms  "
          f"候选 σ_0 median={float(ls0.exp().median()):.4f}Hz")

    t0 = time.time()
    r = refine(mu0, ls0, y_full, t_full, STEPS)
    t_ref = time.time() - t0
    print(f"[第二阶段] ELBO 精修 {t_ref:.1f}s ({t_ref/N_SEQ*1000:.0f}ms/序列)  "
          f"σ_n² median={float(r['sigma_n2'].median()):.3e} (自由变量, 未用真 SNR)\n")

    sel = select_topk(r, K)
    MU = sel["mu"].cpu().numpy(); SG = sel["sigma"].cpu().numpy()
    AM = sel["amp"].cpu().numpy(); AS = sel["amp_sigma"].cpu().numpy()

    for i in range(min(N_SEQ, 5)):                           # 只打印前 5 条
        print(f"序列 {i}:")
        print(f"  {'k':>2} {'f (Hz)':>10} {'σ_f (Hz)':>10} {'|c|':>9} {'σ_c':>9}")
        for k in range(K):
            print(f"  {k:>2} {MU[i,k]:>10.3f} {SG[i,k]:>10.4f} "
                  f"{AM[i,k]:>9.4f} {AS[i,k]:>9.4f}")
    if N_SEQ > 5:
        print(f"... 其余 {N_SEQ-5} 条见 json")

    from ckpt_root import no_clobber
    out_path = no_clobber(os.path.join(os.path.dirname(os.path.abspath(AXIAL_CK)),
                                       "inference.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"cfg": {"n_seq": N_SEQ, "snr_db": SNR, "steps": STEPS,
                           "n_sample": N_SAMPLE, "K": K, "Kc": Kc, "seed": SEED,
                           "axial_ckpt": AXIAL_CK, "fusion": "mean (parameter-free)"},
                   "timing": {"encode_ms": t_enc * 1000,
                              "refine_ms_per_seq": t_ref / N_SEQ * 1000},
                   "estimates": [{"freq_hz": MU[i].tolist(),
                                  "freq_sigma_hz": SG[i].tolist(),
                                  "amp": AM[i].tolist(),
                                  "amp_sigma": AS[i].tolist(),
                                  "sigma_n2": float(r["sigma_n2"][i])}
                                 for i in range(N_SEQ)]}, fh,
                  indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
