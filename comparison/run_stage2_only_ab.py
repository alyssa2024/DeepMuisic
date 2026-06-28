"""Stage2-only A/B control: reuse the EXISTING FR Stage1 checkpoint and run ONLY
the unsupervised ELBO Stage2 under two configs, to isolate WHY delta collapses
(train_delta_logvar_mean -> -7.9, freq_mae frozen at raw 0.50):

  A) kl0      : --local_kl_weight 0  -> test the "KL pins delta to 0" hypothesis.
                If delta now moves to the true freq, the N(0,1) prior was the cause.
  B) marginal : --recon_mode marginal -> test the "amplitude is a free shortcut"
                hypothesis. point_ls solves amplitude analytically, so a freq offset
                is absorbed by re-fitting amplitude -> recon flat in delta. marginal
                integrates the amp prior out (quad+logdet Occam), so the offset
                CANNOT be absorbed losslessly -> recon regains curvature in delta.

Everything else matches the production-aligned Stage2 (sample_mode, recon_weight,
local_radius, FR pool monkeypatch). Stage1 is NOT re-run.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_BTT = _HERE.parent / "btt_amortized_vi_project"
for p in (str(_HERE), str(_HERE / "deepfreq_official"),
          str(_BTT), str(_BTT / "legacy_reuse" / "current_project")):
    if p not in sys.path:
        sys.path.insert(0, str(p))

from deepfreq_btt_geomrobust import build_front  # noqa: E402
from run_frfront_elbo import _fr_pool_factory, _load_runmod  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr_checkpoint", default="comparison/artifacts/geomrobust_nudft_speedrand/geomrobust_fr.pt")
    ap.add_argument("--stage1_ckpt", default="comparison/artifacts/frfront_elbo2_stage1/four_component_candidate_encoder.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train_samples", type=int, default=2000)
    ap.add_argument("--num_cycles", type=int, default=16)
    ap.add_argument("--local_radius_hz", type=float, default=1.0)
    ap.add_argument("--recon_weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_root", default="comparison/artifacts/frfront_elbo2_s2ab")
    cli = ap.parse_args()

    runmod = _load_runmod()
    ck = torch.load(cli.fr_checkpoint, map_location="cpu", weights_only=False)
    from argparse import Namespace
    fa = Namespace(**ck["args"])
    fr_model = build_front(fa, int(fa.num_cycles) * 4)
    fr_model.load_state_dict(ck["model"]); fr_model.eval()
    grid_hz = np.linspace(float(fa.global_fmin_hz), float(fa.global_fmax_hz),
                          int(fa.fr_size), endpoint=False)
    runmod.make_greedy_basin_pool_candidates = _fr_pool_factory(
        fr_model, grid_hz, cli.local_radius_hz)

    # 判定实验已证明: recon 对 delta 曲率强、最小值在真频附近, 自由参数 Adam lr=0.05
    # 400 步就把 freq_mae 从 0.42 -> 0.11。所以 delta 不动是 ENCODER 优化不充分, 不是
    # loss/landscape。这里只扫 lr / epoch (其余固定: KL=0 排除 prior 干扰, point_ls,
    # sigmoid 软 logvar 让方差不撞硬底), 看「调大 lr + 多训」能否让 delta_head 动起来。
    common = [
        "--candidate_support_mode", "basin",
        "--candidate_mode", "profile_ls",
        "--basin_min_half_width_hz", "8.0",
        "--global_bins", "512", "--top_j", "24",
        "--num_cycles", str(cli.num_cycles),
        "--train_samples", str(cli.train_samples),
        "--unsupervised_elbo_train",
        "--elbo_frequency_source", "cartesian_topk", "--cartesian_topk", "3",
        "--z_repulsion_weight", "1.0",
        "--recon_weight", "1.0",
        "--sample_mode", "sample",
        "--local_nll_weight", "0.0",
        "--recon_mode", "point_ls", "--profile_ridge", "1e-3",
        "--local_kl_weight", "0.0",
        "--delta_logvar_param", "sigmoid",   # 软钳位, logvar 不撞 -8 硬底
        "--delta_logvar_min", "-6.0",
        "--init_encoder_checkpoint", cli.stage1_ckpt,
    ]

    # (名称, lr, batch_size, epochs)
    configs = {
        "lr3e3_b256_e40": ("3e-3", "256", "40"),
        "lr1e2_b256_e40": ("1e-2", "256", "40"),
        "lr3e2_b512_e60": ("3e-2", "512", "60"),
    }

    for name, (lr, bs, ep) in configs.items():
        out = f"{cli.out_root}_{name}"
        argv = common + ["--lr", lr, "--batch_size", bs, "--epochs", ep,
                         "--out_dir", out]
        print(f"\n===== Stage2-only [{name}] lr={lr} bs={bs} ep={ep} -> {out} =====")
        runmod.train(runmod.parse_args(argv))
        print(f"[done] {name} -> {out}")


if __name__ == "__main__":
    main()
