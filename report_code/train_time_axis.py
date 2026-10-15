"""[草稿 v2] 阶段三 时间轴层训练 — 统一报告口径版 (由 train_axial_vit.py 改造)。

与 train_axial_vit.py 的差异 (全部来自 2026-07-28 统一口径要求):
  一、log      train/val 总 loss + 各分项 (anchor/gnll), 均为 200 步滑动均值 (不打瞬时);
               metric 报 model@K 与 sum@K 两路 recall+allK; 本阶段不训方差头 ⇒ 不报 σ;
               train_log.json 每个评测点更新; loss/metric 全部写 TB。
  二、metrics  结束写 metrics.json, 含 eval_snr_db 各档分档评估。
  三、config   开训即复制 config 到结果目录。
  四、选模型   阶段三属 1/3/5 组 ⇒ 按 **val_recall@K@20dB (model 路)** 最大选。
  五、双 recall sum@K 只进 log/TB, **不参与选模型**。

训练本身不变: 频率轴默认全程冻结, 只训时间轴层; 方差头冻结 (β=0)。

用法: python train_axial_vit_v2.py <ftf_ckpt> <ftf_config.json> [iters] [eval_every]
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
from axial_vit import load_axial_vit
from loss import build_aggregator_loss
from scheduler import make_scheduler
from train_report import RunningMean, TrainReporter

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FTF_CK, CFG = sys.argv[1], sys.argv[2]

cfg = load_ckpt_config(CFG)
K = cfg["signal"]["num_components"]
etype = cfg["encoder"]["type"]

grid = torch.as_tensor(make_freq_grid(cfg), dtype=torch.float32, device=DEV)
G = grid.numel(); Δ = float(grid[1] - grid[0]); M = n_windows(cfg)
basin = cfg["eval"]["basin_hz"]
min_dist = max(int(round(cfg["eval"]["nms_distance_hz"] / Δ)), 1)

_tc = cfg["crosswin"]["train"]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else int(_tc["iters"])
EVAL_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else int(cfg["crosswin"]["eval"]["eval_every"])

model = load_axial_vit(cfg, grid, Δ, FTF_CK, device=DEV)
n_freq = sum(p.numel() for p in model.freq_layers.parameters())
n_time = sum(p.numel() for p in (pp for tl in model.time_layers for pp in tl.parameters()))
print(f"[axial] 频率轴(FTF layers)={n_freq/1e6:.3f}M "
      f"时间轴({len(model.time_layers)}层逐bin)={n_time/1e6:.3f}M")
print(f"enc={etype} | G={G} n_win={M} K={K} | 时间轴 token=M×G={M*G}")


def recall_allk(spec, f_win, n_freq_, delta=None, dls=(0,)):
    """delta [B,Mw,G]: 给了则峰频率取亚-bin μ = grid + δ·Δ (与 infer/eval 同口径)。

    ⚠ 不给 δ 时退回 bin 中心, 会系统性低估 allK 约 0.03 —— 真频偏向 bin 边缘时
      邻近 bin 的假峰可能比正确 bin 更近, 被 d.min() 抢走。
    返回 {dL: (recall, allK)}: 逐样本取前 n_freq+dL 个峰。按上界 K+max(dL) 取峰时
      topk 已排序, 前 K 个峰与只取 K 个相同 ⇒ dL=0 数值不受额外档影响。
    """
    dls = sorted(set(int(x) for x in dls) | {0})
    L_MAX = K + max(dls)
    B, Mw, _ = spec.shape
    flat = spec.reshape(B * Mw, G)
    peaks = _topl_peak_indices(flat, L_MAX, min_dist)
    if delta is None:
        pf_all = grid[peaks]
    else:
        mu_all = grid[None, :] + delta.reshape(B * Mw, G) * Δ
        pf_all = torch.gather(mu_all, 1, peaks)
    ft = f_win.reshape(B * Mw, -1).to(DEV)
    nf = n_freq_.to(DEV).repeat_interleave(Mw)
    valid = torch.arange(ft.shape[1], device=DEV)[None] < nf[:, None]
    res = {}
    for dL in dls:
        # ⚠ 可变 K: 逐样本只保留前 n_freq+dL 个峰, 否则 K 小的样本白得额外猜测机会
        #   (K=1 拿 15 个峰覆盖 1 个真频 ⇒ recall 虚高)。多出的峰置 +inf。
        keep = torch.arange(L_MAX, device=DEV)[None] < (nf + dL)[:, None]
        pf = torch.where(keep, pf_all, torch.full_like(pf_all, float("inf")))
        d = (pf[:, None, :] - ft[:, :, None]).abs()
        covered = (d.min(-1).values <= basin) & valid
        res[dL] = (float(covered.float().sum() / valid.float().sum().clamp_min(1)),
                   float((covered | ~valid).all(1).float().mean()))
    return res


_ecfg = cfg["crosswin"]["eval"]
MONITOR_SNR = float(_ecfg["monitor_snr_db"]); EVAL_SIZE = int(_ecfg["eval_size"])
SNR = cfg["training"]["train_snr_choices"]
SNR_BANDS = cfg["eval"]["eval_snr_db"]     # 分档评估: 六档, 与阶段1/2同口径
FINAL_DL = list(cfg["eval"].get("final_recall_dL", [0]))   # 仅结束时分档评估用 (见 config)


@torch.no_grad()
def _run(snr, n, want_loss, dls=(0,)):
    """跑一批评测。返回 model/sum 两路 recall+allK, 以及 (可选) val loss 分项。
    dls 非 0 档额外写 recall@K+dL / allk@K+dL / sum_recall@K+dL / sum_allk@K+dL。"""
    dls = sorted(set(int(x) for x in dls) | {0})
    ev = CrossWinDataset(cfg, n, seed=999000, snr_db=snr)
    res = {(k, d): {"r": [], "a": []} for k in ["model", "sum"] for d in dls}
    vl, vp = [], []
    for b in DataLoader(ev, batch_size=32, num_workers=0):
        out = model(b, DEV)
        obj0, obj_t = out["obj0"], out["obj_logit"]
        if want_loss:
            loss, parts = loss_fn(out["obj_logit"], out["delta"], out["log_sigma"], b)
            vl.append(float(loss))
            vp.append(parts)
        # δ 与谱同源: model 用时间轴层的 δ, sum 用融合前的逐窗 δ0
        specs = {"model": (torch.sigmoid(obj_t), out["delta"]),                    # 模型输出
                 "sum": (torch.sigmoid(obj0.sum(1, keepdim=True).expand_as(obj0)),
                         out["delta0"])}                                           # 仅 log
        for k, (s, dl) in specs.items():
            for d, (r, a) in recall_allk(s, b["f_win"], b["n_freq"], dl, dls=dls).items():
                res[(k, d)]["r"].append(r); res[(k, d)]["a"].append(a)
    o = {"recall": float(np.mean(res[("model", 0)]["r"])),
         "allk": float(np.mean(res[("model", 0)]["a"])),
         "sum_recall": float(np.mean(res[("sum", 0)]["r"])),
         "sum_allk": float(np.mean(res[("sum", 0)]["a"]))}
    for d in dls:
        if d:
            o[f"recall@K+{d}"] = float(np.mean(res[("model", d)]["r"]))
            o[f"allk@K+{d}"] = float(np.mean(res[("model", d)]["a"]))
            o[f"sum_recall@K+{d}"] = float(np.mean(res[("sum", d)]["r"]))
            o[f"sum_allk@K+{d}"] = float(np.mean(res[("sum", d)]["a"]))
    if want_loss:
        o["val_parts"] = {"total": float(np.mean(vl))}
        for k in vp[0]:
            if k not in ACTIVE_PARTS:            # 权重为 0 的项不进总和, 不报
                continue
            o["val_parts"][k] = float(np.mean([p[k] for p in vp]))
    return o


@torch.no_grad()
def evaluate(n=None):
    """recall 用固定 20dB (与历史同口径); val_loss 用宽 SNR (与训练同分布)。"""
    n = n or EVAL_SIZE
    m = _run(MONITOR_SNR, n, want_loss=False)
    lo = _run(SNR, n, want_loss=True)
    m["val_parts"] = lo["val_parts"]
    return m


@torch.no_grad()
def evaluate_per_snr(n=None):
    """[要求二] 分档评估: eval_snr_db 每档单独跑 (本阶段不训 σ ⇒ 不含 σ 指标)。"""
    n = n or EVAL_SIZE
    out = {}
    keys = ["recall", "allk", "sum_recall", "sum_allk"]
    keys += [f"{m}@K+{d}" for d in FINAL_DL if d
             for m in ("recall", "allk", "sum_recall", "sum_allk")]
    for snr in SNR_BANDS:
        m = _run(float(snr), n, want_loss=False, dls=FINAL_DL)
        out[str(snr)] = {k: m[k] for k in keys}
    return out


loss_fn = build_aggregator_loss(cfg, grid, device=DEV)

# 方差头冻结 (β=0) ⇒ 本阶段 train_sigma=False, 不报 σ 指标
_beta = float(_tc.get("beta_gnll", 0.0))
FREEZE_SIGMA = _beta <= 0

# ⭐ 报哪些 loss 分项:
#   AggregatorLoss 直接给出 l_obj / l_delta (与单窗阶段同口径)。权重为 0 的项**不进
#   total**, 报出来只会误导 (β=0 时 gnll 不参与优化) ⇒ 只报在优化的项。
ACTIVE_PARTS = {"l_obj", "l_delta"}
if _beta > 0:
    ACTIVE_PARTS.add("gnll")




TRAIN_SIGMA = not FREEZE_SIGMA
lr_time = float(_tc["lr"])
lr_freq = float(_tc.get("lr_freq", lr_time * 0.1))

freq_params, frozen = [], []
freq_modules = [model.in_proj, model.freq_embed, model.film, model.freq_layers,
                model.final_norm, model.obj_head, model.delta_head]
seen = set()
for mod in freq_modules:
    if mod is None:
        continue
    it_ = mod.parameters() if hasattr(mod, "parameters") else [mod]
    for p in it_:
        if id(p) not in seen:
            seen.add(id(p)); freq_params.append(p)
for p in model.sigma_head.parameters():
    if FREEZE_SIGMA:
        p.requires_grad_(False); frozen.append(p)
    else:
        freq_params.append(p)
time_params = [p for tl in model.time_layers for p in tl.parameters()]

# 频率轴默认全程冻结 (论文阶段三: 只训新增时间轴层)。FREEZE_FREQ=0 退回端到端 (消融)。
FREEZE_FREQ = os.environ.get("FREEZE_FREQ", "1") == "1"
if FREEZE_FREQ:
    for p in freq_params:
        p.requires_grad_(False)
    opt = torch.optim.AdamW([{"params": time_params, "lr": lr_time}])
    print(f"[FREEZE_FREQ] 冻结频率轴 ({sum(p.numel() for p in freq_params)/1e6:.3f}M), "
          f"只训时间轴 ({sum(p.numel() for p in time_params)/1e6:.3f}M) | lr={lr_time:g}")
else:
    opt = torch.optim.AdamW([
        {"params": time_params, "lr": lr_time},
        {"params": freq_params, "lr": lr_freq},
    ])
    print(f"[端到端] 时间轴={lr_time:g} 频率轴={lr_freq:g} (消融配置, 非论文主配置)")

_sch_cfg = {"iters": ITERS, "anneal_start": int(_tc.get("anneal_start", ITERS)),
            "anneal_min_factor": float(_tc.get("anneal_min_factor", 0.01))}
if _sch_cfg["anneal_start"] > ITERS:
    _sch_cfg["anneal_start"] = int(round(ITERS * 0.6))
sched = make_scheduler(opt, _sch_cfg)

# 续训: RESUME=<阶段三 ckpt (last.pt/best.pt)> 载入整模权重再训 ITERS 步。
#   ckpt 不含 optimizer 状态 ⇒ AdamW 重新初始化, lr 从头热重启再退火。
#   数据 seed 换成 RESUME_SEED (默认 2000000): 样本 rng = seed+idx, 首轮 seed=1 占
#   [1, 1+BS*ITERS), 验证集 seed=999000 ⇒ 2e6 起步与两者都不重叠 (seed=2 只是平移 1 位)。
RESUME = os.environ.get("RESUME")
DATA_SEED = 1
if RESUME:
    _rs = torch.load(RESUME, map_location=DEV, weights_only=False)
    model.load_state_dict(_rs["model_state"], strict=True)
    DATA_SEED = int(os.environ.get("RESUME_SEED", "2000000"))
    print(f"[RESUME] {RESUME} (iter={_rs.get('iter')} recall={_rs.get('recall')}) "
          f"| 再训 {ITERS} 步, 数据 seed={DATA_SEED}")

OUT_DIR = os.path.join(cfg["checkpoint"]["dir"],
                       ("axial_vit_frozenfreq" if FREEZE_FREQ else "axial_vit")
                       + ("_resume" if RESUME else ""))

# TB 目录跟随 checkpoint.dir (同 train_freq_axis.py): 硬写 "stage3_time_axis" 会让
# 不同 run 的 events 全落进同一个目录, 被 TB 叠成一条来回跳的乱曲线。
_ck_root = os.path.basename(os.path.normpath(cfg["checkpoint"]["dir"]))
TB_NAME = ("stage3_time_axis" if _ck_root == "artifacts"
           else f"stage3_time_axis_{_ck_root}") + ("_resume" if RESUME else "")

# [要求三/四] 报告器: 开训即落 config; 阶段三按 val_recall 选模型
rep = TrainReporter(OUT_DIR, cfg, stage="time_axis", select="recall",
                    train_sigma=TRAIN_SIGMA, tb_name=TB_NAME,
                    recall_L=K, monitor_snr_db=MONITOR_SNR)

BS = _tc["batch_size"]
ds = CrossWinDataset(cfg, size=BS * ITERS, seed=DATA_SEED, snr_db=SNR)
loader = DataLoader(ds, batch_size=BS, num_workers=0)
d = cfg["data"]
_cpl = f" coupling={d.get('drift_coupling_range')}" if d["freq_drift"] else ""
print(f"数据: regime={d['regime']} freq_drift={d['freq_drift']}{_cpl}"
      f" | SNR~U[{SNR['min']},{SNR['max']}]")
print(f"选模型: val_recall@{K}@{MONITOR_SNR:g}dB (model 路; sum 仅 log)")
rep.header()

runner = RunningMean(window=200)
all_train = time_params if FREEZE_FREQ else (time_params + freq_params)
for it, batch in enumerate(loader, 1):
    out = model(batch, DEV)
    loss, parts = loss_fn(out["obj_logit"], out["delta"], out["log_sigma"], batch)
    opt.zero_grad(); loss.backward()
    opt.step(); sched.step()

    # [要求一] 总 loss + 各分项 (anchor 已拆成 l_obj/l_delta; 只报权重非 0 的项)
    tr_parts = runner.update({"total": float(loss.detach()),
                              **{k: v for k, v in parts.items() if k in ACTIVE_PARTS}})

    if it == 1 or it % EVAL_EVERY == 0:
        model.eval(); m = evaluate(); model.train()
        cur_lr = opt.param_groups[0]["lr"]
        val_parts = m.pop("val_parts")
        ck = {"model_state": model.state_dict(), "iter": it, "cfg": cfg,
              "recall": m["recall"], "allk": m["allk"], "val_loss": val_parts["total"]}
        rep.log_step(it, tr_parts, val_parts, m, {"time_axis": cur_lr}, ckpt=ck)
        rep.print_row(it, tr_parts, val_parts, m, cur_lr)
    if it >= ITERS:
        break

# [要求二] 结束: 分档评估 + metrics.json
print("\n跑分档评估 (eval_snr_db 各档)...")
model.eval()
per_snr = evaluate_per_snr()
rep.finish(per_snr)
