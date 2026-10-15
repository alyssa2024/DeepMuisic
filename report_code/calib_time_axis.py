"""[草稿 v2] 阶段四 时间轴 σ 校准 — 统一报告口径版 (由 calib_sigma_axial.py 改造)。

训练本身不变: 冻 axial 全部点估计 (freq_layers/time_layers/obj/δ), 只训共享 sigma_head
(Linear(128,1), 129 参数), 纯 GNLL。点估计冻结 ⇒ recall/μ 按构造不变。

统一口径差异 (2026-07-28):
  一、log      train/val 总 loss (纯 GNLL ⇒ 只有一项) 滑动均值; metric 报 model@K/sum@K
               两路 recall+allK, **本阶段训方差头 ⇒ additionally σcorr + σ/err**;
               train_log.json 每评测点更新; 全部写 TB。
  二、metrics  结束写 metrics.json, 分档评估含 σ 指标 (训了方差头)。
  三、config   开训即落 config 到结果目录。
  四、选模型   阶段四属 2/4/6 组 ⇒ 按 **val_loss (GNLL) 最小** 选。
               ⚠ 改自旧版的 "σcorr 最大" —— 点估计冻结时 recall 是常量, GNLL 才是本阶段
                 真正在优化的目标; corr 高不等于 σ 数值吻合 (σ/err 才是)。
  五、双 recall sum@K 只进 log, 不参与选模型 (本阶段 recall 按构造不变, 仅作 sanity check)。

用法: python calib_sigma_axial_v2.py <axial_ckpt> <config.json> [iters] [lr]
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_ckpt_config
from dataset import make_freq_grid
from peaks import topl_peak_indices as _topl_peak_indices
from crosswin_data import CrossWinDataset, n_windows
from axial_vit import AxialViT
from encoder import build_encoder
from loss import anchor_loss, build_anchor_targets
from train_report import RunningMean, TrainReporter, sigma_calibration

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AXIAL_CK, CFG = sys.argv[1], sys.argv[2]
# 6000 步时 val_loss/σcorr/σ-err 仍在改善 (σcorr 走 U 形, 谷底 0.520@5.5k 后回升到
# 0.605@20k); 20000 步才收敛: σ/err 1.563 (理想 1.48), σcorr 0.605。
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-4

cfg = load_ckpt_config(CFG)
K = cfg["signal"]["num_components"]
ecfg = cfg["eval"]; acfg = cfg["encoder"]["anchor"]
grid = torch.as_tensor(make_freq_grid(cfg), dtype=torch.float32, device=DEV)
G = grid.numel(); Δ = float(grid[1] - grid[0]); M = n_windows(cfg)
basin = ecfg["basin_hz"]
min_dist = max(int(round(ecfg["nms_distance_hz"] / Δ)), 1)
MONITOR_SNR = float(cfg["crosswin"]["eval"]["monitor_snr_db"])
EVAL_SIZE = int(cfg["crosswin"]["eval"]["eval_size"])
SNR_BANDS = ecfg["eval_snr_db"]            # 分档评估: 六档, 与阶段1/2同口径
FINAL_DL = list(ecfg.get("final_recall_dL", [0]))   # 仅结束时分档评估用 (见 config)
_gs = float(cfg["frequency"]["grid_step_hz"])
_sig_hz = acfg["obj_sigma_hz"]
obj_sigma_bins = float(_sig_hz) / _gs if _sig_hz is not None else float(acfg["obj_sigma_bins"])

ftf = build_encoder(cfg).to(DEV)
model = AxialViT(ftf, cfg).to(DEV)
sd = torch.load(AXIAL_CK, map_location=DEV, weights_only=False)
model.load_state_dict(sd["model_state"] if "model_state" in sd else sd, strict=True)
for p in model.parameters():
    p.requires_grad_(False)
sig_params = list(model.sigma_head.parameters())
for p in sig_params:
    p.requires_grad_(True)
print(f"[σ-calib axial] 只训 sigma_head ({sum(p.numel() for p in sig_params)} 参数) | "
      f"lr={LR:g} | G={G} M={M} K={K}")

opt = torch.optim.AdamW(sig_params, lr=LR)


def anchor_targets(batch):
    fw = batch["f_win"].to(DEV).reshape(-1, batch["f_win"].shape[-1])
    B = batch["f_win"].shape[0]
    amp = batch["amp_true"].to(DEV)[:, None, :].expand(B, M, -1).reshape(fw.shape[0], -1)
    nf = batch["n_freq"].to(DEV).repeat_interleave(M)
    return build_anchor_targets(fw, nf, amp, grid, obj_sigma_bins=obj_sigma_bins)


def flat_out(out):
    return {"obj_logit": out["obj_logit"].reshape(-1, G),
            "delta": out["delta"].reshape(-1, G),
            "log_sigma": out["log_sigma"].reshape(-1, G)}


@torch.no_grad()
def _run(snr, n, want_loss, dls=(0,)):
    """model/sum 两路 recall+allK + σ 校准 (+ 可选 val loss)。
    dls 非 0 档额外写 recall@K+dL / allk@K+dL / sum_recall@K+dL / sum_allk@K+dL
    (逐样本取前 n_freq+dL 个峰; topk 已排序 ⇒ dL=0 数值不受额外档影响)。"""
    dls = sorted(set(int(x) for x in dls) | {0})
    L_MAX = K + max(dls)
    ev = CrossWinDataset(cfg, n, seed=999000, snr_db=snr)
    res = {(k, d): {"r": [], "a": []} for k in ["model", "sum"] for d in dls}
    sig_all, err_all, vl, vp = [], [], [], []
    for b in DataLoader(ev, batch_size=32, num_workers=0):
        out = model(b, DEV)
        obj0, obj_t = out["obj0"], out["obj_logit"]
        if want_loss:
            fo = flat_out(out)
            l, parts = anchor_loss(fo, anchor_targets(b), stage="stage2")   # 纯 GNLL
            vl.append(float(l))
            vp.append({k: float(v) for k, v in (parts or {}).items()})
        # δ 与谱同源, 峰频率取亚-bin μ = grid + δ·Δ (与 infer/eval 同口径)
        specs = {"model": (torch.sigmoid(obj_t), out["delta"]),
                 "sum": (torch.sigmoid(obj0.sum(1, keepdim=True).expand_as(obj0)),
                         out["delta0"])}
        for k, (s, dl) in specs.items():
            flat = s.reshape(-1, G)
            peaks = _topl_peak_indices(flat, L_MAX, min_dist)
            mu_all = grid[None, :] + dl.reshape(-1, G) * Δ
            pf_all = torch.gather(mu_all, 1, peaks)
            ft = b["f_win"].reshape(-1, b["f_win"].shape[-1]).to(DEV)
            nf = b["n_freq"].to(DEV).repeat_interleave(M)
            valid = torch.arange(ft.shape[1], device=DEV)[None] < nf[:, None]
            for dL in dls:
                # ⚠ 可变 K: 逐样本只保留前 n_freq+dL 个峰 (否则 K 小的样本 recall 虚高)。
                keep = torch.arange(L_MAX, device=DEV)[None] < (nf + dL)[:, None]
                pf = torch.where(keep, pf_all, torch.full_like(pf_all, float("inf")))
                d = (pf[:, None, :] - ft[:, :, None]).abs()
                cov = (d.min(-1).values <= basin) & valid
                res[(k, dL)]["r"].append(float(cov.float().sum() / valid.float().sum().clamp_min(1)))
                res[(k, dL)]["a"].append(float((cov | ~valid).all(1).float().mean()))
        # σ 校准样本 (锚点 bin)
        tgt = anchor_targets(b); fo = flat_out(out)
        aidx = tgt["argmax_bin"].clamp(min=0); aval = tgt["valid"]; dstar = tgt["delta_true"]
        dl = fo["delta"].cpu().numpy(); ls = fo["log_sigma"].cpu().numpy()
        av = aval.cpu().numpy(); ai = aidx.cpu().numpy(); ds_ = dstar.cpu().numpy()
        for i in range(dl.shape[0]):
            for j in range(ai.shape[1]):
                if not bool(av[i, j]):
                    continue
                g = int(ai[i, j])
                err_all.append(abs(float(dl[i, g]) - float(ds_[i, j])) * Δ)
                sig_all.append(float(np.exp(ls[i, g])) * Δ)
    corr, ratio = sigma_calibration(sig_all, err_all)
    o = {"recall": float(np.mean(res[("model", 0)]["r"])),
         "allk": float(np.mean(res[("model", 0)]["a"])),
         "sum_recall": float(np.mean(res[("sum", 0)]["r"])),
         "sum_allk": float(np.mean(res[("sum", 0)]["a"])),
         "sigma_corr": corr, "sigma_ratio": ratio}
    for dL in dls:
        if dL:
            o[f"recall@K+{dL}"] = float(np.mean(res[("model", dL)]["r"]))
            o[f"allk@K+{dL}"] = float(np.mean(res[("model", dL)]["a"]))
            o[f"sum_recall@K+{dL}"] = float(np.mean(res[("sum", dL)]["r"]))
            o[f"sum_allk@K+{dL}"] = float(np.mean(res[("sum", dL)]["a"]))
    if want_loss:
        o["val_parts"] = {"total": float(np.mean(vl))}
        for k in (vp[0] if vp else {}):
            o["val_parts"][k] = float(np.mean([p[k] for p in vp]))
    return o


@torch.no_grad()
def evaluate(n=None):
    n = n or EVAL_SIZE
    model.eval()
    m = _run(MONITOR_SNR, n, want_loss=True)
    model.train()
    return m


@torch.no_grad()
def evaluate_per_snr(n=None):
    """[要求二] 分档评估 —— 训了方差头 ⇒ 每档含 σcorr / σ/err。"""
    n = n or EVAL_SIZE
    model.eval()
    out = {}
    keys = ["recall", "allk", "sum_recall", "sum_allk", "sigma_corr", "sigma_ratio"]
    keys += [f"{m}@K+{d}" for d in FINAL_DL if d
             for m in ("recall", "allk", "sum_recall", "sum_allk")]
    for snr in SNR_BANDS:
        m = _run(float(snr), n, want_loss=False, dls=FINAL_DL)
        out[str(snr)] = {k: m[k] for k in keys}
    return out


OUT_DIR = os.path.join(cfg["checkpoint"]["dir"], "axial_sigcalib")
# [要求三/四] 落 config; 阶段四按 val_loss 最小选
rep = TrainReporter(OUT_DIR, cfg, stage="time_axis_calibration", select="loss",
                    train_sigma=True, tb_name="stage4_sigma_calib",
                    recall_L=K, monitor_snr_db=MONITOR_SNR)

SNR = cfg["training"]["train_snr_choices"]
BS = cfg["crosswin"]["train"]["batch_size"]
ds = CrossWinDataset(cfg, size=BS * ITERS, seed=7, snr_db=SNR)
loader = DataLoader(ds, batch_size=BS, num_workers=0)
print(f"选模型: val_loss(GNLL) 最小 | recall 按构造不变 (sanity check)")
rep.header()

runner = RunningMean(window=200)
for it, batch in enumerate(loader, 1):
    out = model(batch, DEV)
    fo = flat_out(out)
    loss, _parts = anchor_loss(fo, anchor_targets(batch), stage="stage2")   # 纯 GNLL
    opt.zero_grad(); loss.backward()
    opt.step()
    tr_parts = runner.update({"total": float(loss.detach()),
                              **{k: float(v) for k, v in (_parts or {}).items()}})

    if it == 1 or it % 500 == 0:
        m = evaluate()
        val_parts = m.pop("val_parts")
        ck = {"model_state": model.state_dict(), "iter": it, "cfg": cfg,
              "recall": m["recall"], "sigma_corr": m["sigma_corr"],
              "sigma_ratio": m["sigma_ratio"], "val_loss": val_parts["total"]}
        rep.log_step(it, tr_parts, val_parts, m, {"sigma_head": LR}, ckpt=ck)
        rep.print_row(it, tr_parts, val_parts, m, LR)
    if it >= ITERS:
        break

print("\n跑分档评估 (eval_snr_db 各档, 含 σ 指标)...")
per_snr = evaluate_per_snr()
rep.finish(per_snr)
