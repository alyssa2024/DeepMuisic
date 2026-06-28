"""判决实验最后一块: 给 amortized δ 头加【直接监督】, 看 δ 能否学动。

此前所有 δ 失败都在【纯无监督】(--unsupervised_elbo_train -> local_nll 被硬置零,
δ 只靠弱 recon 信号)。这里走【全监督分支】(既不开 z_only 也不开 unsup_elbo):
  local_nll = gaussian_nll(δ_mu, δ_logvar, target_delta),  target_delta=(true_freq-cand)/radius
即用真频算出的标准答案直接监督 δ_mu。

判据:
  - 监督下 δ 学动 (freq_mae < 0.50) -> amortized 头【有能力】, 之前是无监督 recon 信号太弱。
    -> PROSAIL「参数 MSE 主监督」思路成立, ELBO/amortized 路线可救 (监督预训 δ + 无监督微调)。
  - 监督下 δ 仍学不动 -> amortized 回归这个任务【本身难】(给了标准答案也回归不出),
    -> 彻底转 MNN per-sample 点估计。

复用 8 维 (include_local_time) Stage1 ckpt 做 init (input_proj 维度匹配)。
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
    ap.add_argument("--stage1_ckpt", default="comparison/artifacts/frfront_elbo_localtime_stage1/four_component_candidate_encoder.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train_samples", type=int, default=2000)
    ap.add_argument("--num_cycles", type=int, default=16)
    ap.add_argument("--local_radius_hz", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", default="comparison/artifacts/frfront_supervised_delta")
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

    argv = [
        "--candidate_support_mode", "basin",
        "--candidate_mode", "profile_ls",
        "--basin_min_half_width_hz", "8.0",
        "--global_bins", "512", "--top_j", "24",
        "--num_cycles", str(cli.num_cycles),
        "--train_samples", str(cli.train_samples),
        "--include_local_time",          # 8 维输入, 与 stage1_ckpt 匹配
        # 全监督分支: 不开 z_only / unsup_elbo
        "--z_loss_type", "soft_basin", "--z_loss_weight", "1.0",
        "--local_nll_weight", "1.0",      # ← δ 直接监督 (target_delta), 关键开关
        "--local_kl_weight", "0.001",
        "--recon_weight", "0.01",
        "--sample_mode", "mean",
        "--lr", str(cli.lr), "--batch_size", str(cli.batch_size),
        "--init_encoder_checkpoint", cli.stage1_ckpt,
        "--epochs", str(cli.epochs), "--out_dir", cli.out_dir,
    ]
    print("=== 监督 δ 实验: FR 候选 + 全监督 (z_loss + local_nll) ===")
    runmod.train(runmod.parse_args(argv))
    print(f"[done] -> {cli.out_dir}")


if __name__ == "__main__":
    main()
