"""B 组序列级评测 —— 与 A 组 (单窗 104 转) 逐比特同口径。

动机: train_time_axis.py 里的 recall_allk 把 [B,M,G] reshape 成 [B*M,G], allk 是
**窗级**全或无 (分母 B×M); 而 A 组 (train_freq_axis.py) 的 allk 是**序列级** (分母 B)。
两者分母口径不同, 不能直接比。本脚本把 M 个窗融合成整条序列的一张谱 [B,G], 再用与
train_freq_axis.py._run() 完全相同的方式取峰/判命中, 使 recall/allk 都在序列级可比。

融合方式 (--fuse):
  sum   : Σ_m sigmoid(obj_t[m])          窗间等权求和 (贝叶斯最优 @ 恒频)
  mean  : 同 sum (取峰只看相对大小, 结果与 sum 逐比特一致; 保留为可读别名)
  max   : max_m sigmoid(obj_t[m])        任一窗点亮即可
  logsum: Σ_m log(1-p) 的补 (概率或)     独立性假设下的"至少一窗命中"

δ 取融合权重最大那一窗的 δ (与谱同源, 不混窗)。

用法:
  python eval_seq_level.py <axial_ckpt> <config.json> [--n 1024] [--fuse sum]
                           [--snr 20] [--seed 999000] [--out result.json]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_ckpt_config
from dataset import make_freq_grid
from peaks import topl_peak_indices
from crosswin_data import CrossWinDataset, n_windows
from axial_vit import AxialViT, load_axial_vit
from encoder import build_encoder


def load_trained_axial(cfg, ckpt, device):
    """加载**已训好的完整 axial 模型** (含 time_layers)。

    ⚠ 不能用 axial_vit.load_axial_vit —— 那个吃的是 FTF 编码器 ckpt (键
      'model_state_dict', 只有频率轴), 用来把时间轴层零初始化建起来。
      阶段三产物的键是 'model_state' 且已含 time_layers, 必须整体 strict 加载。
    """
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    st = sd["model_state"] if "model_state" in sd else sd
    model = AxialViT(build_encoder(cfg), cfg).to(device)
    model.load_state_dict(st, strict=True)
    it = sd.get("iter")
    print(f"[ckpt] {os.path.basename(ckpt)} iter={it} "
          f"recall={sd.get('recall')} (训练期窗级口径)")
    return model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("cfg_json")
    ap.add_argument("--n", type=int, default=1024, help="评测序列数")
    ap.add_argument("--fuse", default="sum", choices=["sum", "mean", "max", "logsum"])
    ap.add_argument("--snr", type=float, default=None,
                    help="单档 SNR; 不给则跑 eval.eval_snr_db 全档")
    ap.add_argument("--seed", type=int, default=999000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def fuse_windows(p, mode):
    """[B,M,G] per-window 概率 -> [B,G] 序列级谱 + 每 bin 的主窗索引 [B,G]。"""
    if mode in ("sum", "mean"):
        s = p.sum(1)
    elif mode == "max":
        s = p.max(1).values
    elif mode == "logsum":
        # 1 - Π(1-p): 独立性下"至少一窗点亮"
        s = 1.0 - torch.exp(torch.log1p(-p.clamp(max=1 - 1e-6)).sum(1))
    else:
        raise ValueError(mode)
    return s, p.argmax(1)               # 主窗 = 该 bin 概率最大的窗


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_ckpt_config(args.cfg_json)

    K = int(cfg["signal"]["num_components"])
    grid = torch.as_tensor(make_freq_grid(cfg), dtype=torch.float32, device=dev)
    G = grid.numel()
    D = float(grid[1] - grid[0])
    basin = float(cfg["eval"]["basin_hz"])
    # ⚠ 与 train_freq_axis.py 同式: min_dist = round(nms_hz / Δ), 下限 1
    min_dist = max(int(round(cfg["eval"]["nms_distance_hz"] / D)), 1)
    M = n_windows(cfg)

    model = load_trained_axial(cfg, args.ckpt, dev)
    model.eval()

    print(f"[seq-eval] G={G} Δ={D:g}Hz M={M} K={K} basin={basin:g} "
          f"min_dist={min_dist} ({min_dist*D:g}Hz) fuse={args.fuse}")
    print(f"[seq-eval] n_seq={args.n} seed={args.seed} dev={dev}")

    snrs = [args.snr] if args.snr is not None else [float(s) for s in cfg["eval"]["eval_snr_db"]]
    results = {}

    for snr in snrs:
        ds = CrossWinDataset(cfg, args.n, seed=args.seed, snr_db=snr)
        rs_seq, ak_seq, rs_win, ak_win = [], [], [], []

        with torch.no_grad():
            for b in DataLoader(ds, batch_size=args.batch, num_workers=0):
                out = model(b, dev)
                p = torch.sigmoid(out["obj_logit"])          # [B,M,G]
                dl = out["delta"]                            # [B,M,G]
                B = p.shape[0]

                # 真值: 恒频下 f_win 各窗相同 -> 序列级真频取第 0 窗
                f_win = b["f_win"].to(dev)                   # [B,M,K]
                nf = b["n_freq"].to(dev)                     # [B]
                ft = f_win[:, 0, :]                          # [B,K]
                valid = torch.arange(ft.shape[1], device=dev)[None] < nf[:, None]

                # ---------- 序列级 (与 A 组同口径) ----------
                spec, main_w = fuse_windows(p, args.fuse)    # [B,G], [B,G]
                peaks = topl_peak_indices(spec, K, min_dist)  # [B,K]
                # δ 与谱同源: 取该 bin 主窗的 δ
                w = torch.gather(main_w, 1, peaks)           # [B,K] 主窗索引
                dsel = torch.gather(dl, 1,
                                    w[:, None, :].expand(B, 1, K)).squeeze(1)
                pf = grid[peaks] + dsel * D                  # [B,K] 亚-bin μ

                d = (pf[:, None, :] - ft[:, :, None]).abs()  # [B,K_true,K_pred]
                cov = (d.min(-1).values <= basin) & valid
                rs_seq.append(float(cov.float().sum() / valid.float().sum().clamp_min(1)))
                ak_seq.append(float((cov | ~valid).all(1).float().mean()))

                # ---------- 窗级 (复现 train_time_axis 口径, 供对照) ----------
                flat = p.reshape(B * M, G)
                pk_w = topl_peak_indices(flat, K, min_dist)
                mu_w = grid[None, :] + dl.reshape(B * M, G) * D
                pf_w = torch.gather(mu_w, 1, pk_w)
                ft_w = f_win.reshape(B * M, -1)
                nf_w = nf.repeat_interleave(M)
                vd_w = torch.arange(ft_w.shape[1], device=dev)[None] < nf_w[:, None]
                dw = (pf_w[:, None, :] - ft_w[:, :, None]).abs()
                cw = (dw.min(-1).values <= basin) & vd_w
                rs_win.append(float(cw.float().sum() / vd_w.float().sum().clamp_min(1)))
                ak_win.append(float((cw | ~vd_w).all(1).float().mean()))

        r = {"seq_recall": float(np.mean(rs_seq)), "seq_allk": float(np.mean(ak_seq)),
             "win_recall": float(np.mean(rs_win)), "win_allk": float(np.mean(ak_win))}
        results[str(snr)] = r
        print(f"  SNR {snr:+6.1f}dB | seq R={r['seq_recall']:.4f} allK={r['seq_allk']:.4f} "
              f"| win R={r['win_recall']:.4f} allK={r['win_allk']:.4f}")

    meta = {"ckpt": args.ckpt, "cfg": args.cfg_json, "n_seq": args.n,
            "fuse": args.fuse, "seed": args.seed, "M": M, "G": G, "grid_step_hz": D,
            "basin_hz": basin, "min_dist_bins": min_dist, "K": K,
            "n_rev_win": cfg["data"]["n_rev"], "n_rev_seq": cfg["crosswin"]["n_rev_seq"],
            "hop_rev": cfg["crosswin"]["hop_rev"]}
    payload = {"meta": meta, "results": results}
    if args.out:
        from ckpt_root import no_clobber
        args.out = no_clobber(args.out)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
