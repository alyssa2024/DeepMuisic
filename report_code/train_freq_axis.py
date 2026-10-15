"""[草稿 v2] 阶段一/二 频率轴单窗训练 — 统一报告口径版 (由 main.py 的 train() 改造)。

阶段一 (phase 1): 从零训主干 + obj/δ 头, loss = 加权BCE(obj) + L2(δ)。σ 头无梯度。
阶段二 (phase 2): 载阶段一权重, 冻结除 σ 头外全部, 纯 GNLL。点估计按构造不变。

统一口径差异 (2026-07-28):
  一、log      train/val 总 loss + 各分项 (obj/delta/gnll) 滑动均值 (不打瞬时值);
               metric: 阶段一不训 σ ⇒ recall@K@20dB + allK@20dB;
                       阶段二训 σ  ⇒ 再加 σcorr + σ/err;
               train_log.json 每评测点更新; loss/metric 全部写 TB。
  二、metrics  结束写 metrics.json + eval_snr_db 分档评估 (阶段二含 σ 分档指标)。
  三、config   开训即落 config 到结果目录。
  四、选模型   阶段一按 val_recall@K@20dB 最大; 阶段二按 val_loss (GNLL) 最小。
  五、双 recall 单窗阶段**无 sum 路** (sum 是跨窗概念) ⇒ 该列恒为 NaN, 仅为表头统一。

⚠ 本脚本自带 allK / σ 校准 / loss 分项的计算, 不改动 eval.py 与 loss.py
  (它们被在跑的旧脚本共用, 改了会波及)。

用法:
  python train_freq_axis_v2.py --stage stage1 [--iters N]
  python train_freq_axis_v2.py --stage stage2 --stage1_ckpt <path/best.pt>
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from dataset import UnifiedBTTDataset, make_freq_grid
from peaks import topl_peak_indices as _topl_peak_indices
from loss import anchor_loss, build_anchor_targets
from encoder import build_encoder, encoder_forward
import scheduler as M
from train_report import RunningMean, TrainReporter, sigma_calibration


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    ap.add_argument("--stage1_ckpt", type=str, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    return ap.parse_args()


args = parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = copy.deepcopy(CONFIG)
# 频带消融 (对照实验用): BAND=lo,hi 同时改真频先验与网格范围。
#   动机: f_rot ∈ [25,45] 时, 带下界低于 f_rot 上限会让低阶混叠 f−f_rot 在网格里
#   有 bin 可占; 带 50-450 则落在带外。默认 5-495 不受影响。
_band = os.environ.get("BAND")
if _band:
    _lo, _hi = (float(v) for v in _band.split(","))
    cfg["signal"] = {**cfg["signal"], "f_lo": _lo, "f_hi": _hi}
    cfg["frequency"] = {**cfg["frequency"], "f_min": _lo, "f_max": _hi}
    print(f"[BAND] 真频与网格均设为 {_lo:g}-{_hi:g} Hz")
STAGE = args.stage
TRAIN_SIGMA = STAGE == "stage2"            # 阶段二训方差头 ⇒ 报 σ 指标
SELECT = "loss" if TRAIN_SIGMA else "recall"

scfg = cfg["training"][STAGE]
ITERS = int(args.iters or scfg["iters"])
BS = int(cfg["training"]["batch_size"])
EVAL_EVERY = int(cfg["eval"]["eval_every"])
EVAL_BATCHES = int(cfg["eval"]["eval_batches"])
MON_SNR = float(cfg["eval"]["monitor_snr_db"])
SNR_BANDS = cfg["eval"]["eval_snr_db"]
FINAL_DL = list(cfg["eval"].get("final_recall_dL", [0]))   # 仅结束时分档评估用 (见 config)
SNR = cfg["training"]["train_snr_choices"]

K = int(cfg["signal"]["num_components"])
grid = torch.as_tensor(make_freq_grid(cfg), dtype=torch.float32, device=DEV)
G = grid.numel(); Δ = float(grid[1] - grid[0])
ecfg = cfg["eval"]; acfg = cfg["encoder"]["anchor"]
basin = ecfg["basin_hz"]
min_dist = max(int(round(ecfg["nms_distance_hz"] / Δ)), 1)
_sig_hz = acfg["obj_sigma_hz"]
obj_sigma_bins = (float(_sig_hz) / float(cfg["frequency"]["grid_step_hz"])
                  if _sig_hz is not None else float(acfg["obj_sigma_bins"]))

# ---- 模型 + 冻结策略 ----
model = build_encoder(cfg).to(DEV)
if STAGE == "stage2":
    assert args.stage1_ckpt, "stage2 需 --stage1_ckpt"
    sd = torch.load(args.stage1_ckpt, map_location=DEV, weights_only=False)
    model.load_state_dict(sd.get("model_state_dict", sd.get("model_state", sd)), strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = list(model.sigma_head.parameters())
    for p in train_params:
        p.requires_grad_(True)
    print(f"[stage2] 冻结除 σ 头外全部 | 可训 {sum(p.numel() for p in train_params)} 参数")
else:
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[stage1] 从零训 | 可训 {sum(p.numel() for p in train_params)/1e6:.3f}M 参数")

opt = torch.optim.AdamW(train_params, lr=float(scfg["lr"]), weight_decay=1e-2)
sched = M.make_scheduler(opt, {"iters": ITERS,
                               "anneal_start": int(scfg.get("anneal_start", ITERS)),
                               "anneal_min_factor": float(scfg.get("anneal_min_factor", 0.01))})
LOSS_STAGE = "stage1" if STAGE == "stage1" else "stage2"


def targets(batch):
    return build_anchor_targets(batch["freqs_hz"], batch["n_freq"], batch["amps"],
                                grid, obj_sigma_bins)


def compute_loss(output, batch):
    """返回 (total, parts dict) —— parts 逐项进 log。"""
    return anchor_loss(output, targets(batch), stage=LOSS_STAGE)


@torch.no_grad()
def _run(snr, want_loss, dls=(0,)):
    """单窗评测: recall@K + allK + (阶段二) σ 校准 + 可选 val loss。

    dls: 额外报 recall@(K+dL) 的偏移 (逐样本取前 n_freq+dL 个峰)。dL=0 写 recall/allk,
      其余写 recall@K+dL / allk@K+dL。按上界 K+max(dL) 取峰时 topk 已排序, 前 K 个峰
      与只取 K 个完全相同 ⇒ dL=0 的数值不受额外档影响。
    """
    dls = sorted(set(int(x) for x in dls) | {0})
    L_MAX = K + max(dls)
    ds = UnifiedBTTDataset(cfg, size=BS * EVAL_BATCHES,
                           seed=99991 + int(float(snr)), snr_db=snr)
    loader = DataLoader(ds, batch_size=BS, shuffle=False)
    rs = {d: [] for d in dls}; aks = {d: [] for d in dls}
    vl, vp, sig_all, err_all = [], [], [], []
    for batch in loader:
        out = encoder_forward(model, batch, cfg, DEV)
        if want_loss:
            l, parts = compute_loss(out, batch)
            vl.append(float(l))
            vp.append({k: float(v) for k, v in (parts or {}).items()})
        obj = torch.sigmoid(out["obj_logit"])
        # ⚠ 可变 K: 取峰数必须是**逐样本的 n_freq**, 不能一律取上界 K。
        #   一律取 K 会让 K 小的样本白得 (K - n_freq) 个额外猜测机会
        #   (K=1 的样本拿 15 个峰去覆盖 1 个真频, recall 必然虚高) ⇒ 指标失真。
        #   做法: 仍按上界取 K 个峰 (张量定长), 但**逐样本只保留前 n_freq 个**,
        #   多出的峰置 +inf 使其不可能落进 basin。
        peaks = _topl_peak_indices(obj, L_MAX, min_dist)
        mu_all = grid[None, :] + out["delta"] * Δ
        pf_all = torch.gather(mu_all, 1, peaks)                            # [B,L_MAX]
        ft = batch["freqs_hz"].to(DEV); nf = batch["n_freq"].to(DEV)
        valid = torch.arange(ft.shape[1], device=DEV)[None] < nf[:, None]
        for dL in dls:
            keep = torch.arange(L_MAX, device=DEV)[None] < (nf + dL)[:, None]   # [B,L_MAX]
            pf = torch.where(keep, pf_all, torch.full_like(pf_all, float("inf")))
            d = (pf[:, None, :] - ft[:, :, None]).abs()
            cov = (d.min(-1).values <= basin) & valid
            rs[dL].append(float(cov.float().sum() / valid.float().sum().clamp_min(1)))
            aks[dL].append(float((cov | ~valid).all(1).float().mean()))
        if TRAIN_SIGMA:
            tgt = targets(batch)
            ai = tgt["argmax_bin"].clamp(min=0).cpu().numpy()
            av = tgt["valid"].cpu().numpy(); ds_ = tgt["delta_true"].cpu().numpy()
            dl = out["delta"].cpu().numpy(); ls = out["log_sigma"].cpu().numpy()
            for i in range(dl.shape[0]):
                for j in range(ai.shape[1]):
                    if not bool(av[i, j]):
                        continue
                    g = int(ai[i, j])
                    err_all.append(abs(float(dl[i, g]) - float(ds_[i, j])) * Δ)
                    sig_all.append(float(np.exp(ls[i, g])) * Δ)
    o = {"recall": float(np.mean(rs[0])), "allk": float(np.mean(aks[0])),
         "sum_recall": float("nan"), "sum_allk": float("nan")}   # 单窗无 sum 路
    for dL in dls:
        if dL:
            o[f"recall@K+{dL}"] = float(np.mean(rs[dL]))
            o[f"allk@K+{dL}"] = float(np.mean(aks[dL]))
    if TRAIN_SIGMA:
        c, r = sigma_calibration(sig_all, err_all)
        o["sigma_corr"], o["sigma_ratio"] = c, r
    if want_loss:
        o["val_parts"] = {"total": float(np.mean(vl))}
        for k in (vp[0] if vp else {}):                 # val 侧同样逐分项
            o["val_parts"][k] = float(np.mean([p[k] for p in vp]))
    return o


@torch.no_grad()
def evaluate():
    model.eval()
    m = _run(MON_SNR, want_loss=True)
    model.train()
    return m


@torch.no_grad()
def evaluate_per_snr():
    """[要求二] 分档评估 (阶段二含 σ 指标)。"""
    model.eval()
    keys = ["recall", "allk"] + (["sigma_corr", "sigma_ratio"] if TRAIN_SIGMA else [])
    keys += [f"{m}@K+{d}" for d in FINAL_DL if d for m in ("recall", "allk")]
    out = {}
    for snr in SNR_BANDS:
        m = _run(float(snr), want_loss=False, dls=FINAL_DL)
        out[str(snr)] = {k: m[k] for k in keys}
    return out


OUT_DIR = args.out_dir or os.path.join(cfg["checkpoint"]["dir"], f"ftf_{STAGE}")
# TB 目录跟随 out_dir: 自定义 out_dir (对照实验) 单独成目录, 否则两个 run 的
# events 落进同一个 stage_<stage>/ 会被 TB 叠成一条乱曲线。
TB_NAME = f"stage_{STAGE}" if args.out_dir is None else f"stage_{STAGE}_{os.path.basename(OUT_DIR.rstrip('/'))}"
rep = TrainReporter(OUT_DIR, cfg, stage=f"frequency_axis_{STAGE}", select=SELECT,
                    train_sigma=TRAIN_SIGMA, tb_name=TB_NAME,
                    recall_L=K, monitor_snr_db=MON_SNR)

ds = UnifiedBTTDataset(cfg, size=BS * ITERS, seed=cfg["seed"], snr_db=SNR)
nw = int(cfg["training"]["num_workers"])
dl_kw = dict(num_workers=nw)
if nw > 0:
    dl_kw.update(persistent_workers=True, prefetch_factor=4)
loader = DataLoader(ds, batch_size=BS, shuffle=False, **dl_kw)
print(f"选模型: {'val_loss(GNLL) 最小' if SELECT == 'loss' else f'val_recall@{K}@{MON_SNR:g}dB 最大'}")
rep.header()

runner = RunningMean(window=200)
it = 0
data_iter = iter(loader)
while it < ITERS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader); batch = next(data_iter)
    it += 1
    model.train()
    out = encoder_forward(model, batch, cfg, DEV)
    loss, parts = compute_loss(out, batch)
    opt.zero_grad(set_to_none=True); loss.backward()
    opt.step(); sched.step()
    _p = {k: float(v) for k, v in (parts or {}).items()}
    tr_parts = runner.update({"total": float(loss.detach()), **_p})

    if it == 1 or it == ITERS or it % EVAL_EVERY == 0:
        m = evaluate()
        val_parts = m.pop("val_parts")
        lrs = {g.get("name", f"g{i}"): g["lr"] for i, g in enumerate(opt.param_groups)}
        ck = {"model_state_dict": model.state_dict(),
              "optimizer_state_dict": opt.state_dict(), "iter": it, "cfg": cfg,
              "recall": m["recall"], "val_loss": val_parts["total"]}
        rep.log_step(it, tr_parts, val_parts, m, lrs, ckpt=ck)
        rep.print_row(it, tr_parts, val_parts, m, list(lrs.values())[0])

print("\n跑分档评估 (eval_snr_db 各档)...")
per_snr = evaluate_per_snr()
rep.finish(per_snr)
