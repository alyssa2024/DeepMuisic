"""The CORRECT pipeline: nudft-FR basin proposal (replacing greedy) feeding the
user's original two-stage amortized-VI — Stage1 z-only selection, Stage2
unsupervised ELBO local-posterior optimization. Gaussian local prior, multi-modal
posterior q(f)=sum_k q(z_k) N(delta_k). The ONLY change vs the original is the
candidate source: nudft FR peaks instead of greedy CLEAN.

Self-consistent data flow (no profile-LS):
  y,t -> nudft FR (one forward) -> peaks => basin {center, half-width} + FR-derived
         features. --candidate_mode profile_ls zeroes alias/bse features, leaving
         [freq_norm, FR_score_norm, 0, 0, rank_norm, comp_norm] where FR_score is
         the FR peak height (monkeypatched in) -> ~pure FR-derived features.
  -> q(z|y) select basin + q(delta|z,y) local Gaussian posterior
  -> Stage1 z-only (FR pool) then Stage2 unsupervised ELBO (FR pool)

Goal: posterior quality (freq MAE) ON PAR with the original greedy-based amortized
VI, while being geometry-robust, unknown-#freq-capable, cheaper, fewer hyperparams.
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

from deepfreq_btt_geomrobust import build_front, _time_channels  # noqa: E402


def _load_runmod():
    spec = importlib.util.spec_from_file_location(
        "run_legacy_four_vi_elbo", _BTT / "run_legacy_four_component_candidate_vi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fr_pool_factory(fr_model, grid_hz, half_width, device="cpu"):
    from deepfreq_btt_baseline import local_peak_indices
    def fr_pool(y_complex, t, args):
        y = np.asarray(y_complex, dtype=np.complex128); y = y / (np.std(y) + 1e-12)
        t = np.asarray(t, dtype=np.float64)
        tl, dn = _time_channels(t)
        chans = torch.tensor(np.stack([y.real, y.imag, tl, dn]).astype(np.float32)[None], device=device)
        with torch.no_grad():
            fr = fr_model(chans, torch.tensor(t.astype(np.float32)[None], device=device)).cpu().numpy()[0]
        top_j = int(args.top_j)
        pk = local_peak_indices(fr)
        order = pk[np.argsort(fr[pk])[::-1]] if pk.size else np.argsort(fr)[::-1]
        # Greedy min-gap dedupe (mirrors greedy's basin_candidate_min_gap_bins): drop
        # peaks within min_gap_hz of an already-kept higher peak. Without this, two
        # near-coincident FR peaks make the cartesian_topk LS Gram singular.
        spacing = float(grid_hz[1] - grid_hz[0])
        min_gap_bins = int(getattr(args, "basin_candidate_min_gap_bins", 2))
        min_gap = max(min_gap_bins * spacing, 1e-6)
        kept = []
        for idx in order:
            f = float(grid_hz[idx])
            if all(abs(f - float(grid_hz[j])) >= min_gap for j in kept):
                kept.append(int(idx))
        order = np.asarray(kept, dtype=int)
        if order.size < top_j:
            used = set(int(i) for i in order)
            for i in np.argsort(fr)[::-1]:
                i = int(i)
                if i in used:
                    continue
                if all(abs(float(grid_hz[i]) - float(grid_hz[j])) >= min_gap for j in order):
                    order = np.append(order, i); used.add(i)
                if order.size >= top_j:
                    break
        order = order[:top_j]
        freqs = grid_hz[order].astype(np.float64); scores = fr[order].astype(np.float64)
        radius = np.full(freqs.shape, half_width, dtype=np.float64)
        srt = np.argsort(freqs)
        return freqs[srt], radius[srt], scores[srt]
    return fr_pool


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr_checkpoint", default="comparison/artifacts/geomrobust_nudft_speedrand/geomrobust_fr.pt")
    ap.add_argument("--prod_checkpoint", default=str(
        _BTT / "artifacts/legacy_four_component_candidate_vi/remote_gpu_archive/"
        "c16_production_repulsion/four_component_candidate_encoder.pt"))
    ap.add_argument("--out_root", default="comparison/artifacts/frfront_elbo")
    ap.add_argument("--train_samples", type=int, default=2000)
    ap.add_argument("--stage1_epochs", type=int, default=40)
    ap.add_argument("--stage2_epochs", type=int, default=60)
    ap.add_argument("--num_cycles", type=int, default=16)
    ap.add_argument("--basin_half_width_hz", type=float, default=8.0,
                    help="Basin half-width for z-selection grouping (FR features). NOT the "
                         "local refinement radius — see --local_radius_hz.")
    ap.add_argument("--local_radius_hz", type=float, default=1.0,
                    help="Local-posterior refinement radius: f = center + radius*delta, "
                         "delta in [-1,1]. Production used 0.25 for greedy candidates "
                         "(already sub-bin, raw_mae 0.375). FR peaks sit on a 1.95-Hz grid "
                         "(half-bin 0.98 Hz, raw_mae 0.50), so radius must cover ~1 Hz; "
                         "0.25 cannot reach a peak that far off, and 8.0 (the basin "
                         "half-width I previously mis-passed) is far too coarse to learn "
                         "the fine offset -> delta stayed unrefined (freq_mae stuck at 0.50).")
    # Stage2 ELBO knobs
    ap.add_argument("--recon_mode", choices=["point_ls", "marginal"], default="point_ls",
                    help="point_ls = production default (reproduce 0.169 first). marginal "
                         "integrates out a zero-mean complex-Gaussian amplitude prior "
                         "(quad+logdet); ~equivalent to point_ls when well-separated, better "
                         "for close tones. Try marginal as a separate run after point_ls.")
    ap.add_argument("--amp_prior_var_scale", type=float, default=1.0,
                    help="Scale on the empirical-Bayes amplitude prior variance tau^2 "
                         "(tunable; sweep before making tau^2 learnable). marginal only.")
    ap.add_argument("--local_kl_weight", type=float, default=0.001,  # = production
                    help="delta-KL coefficient. Production used 0.001; larger values pin "
                         "delta to 0 and prevent local refinement.")
    ap.add_argument("--sample_mode", choices=["sample", "mean"], default="sample",
                    help="MUST be 'sample' so the reparameterized draw delta = mu + "
                         "exp(0.5*logvar)*eps flows recon gradient into BOTH mu AND logvar "
                         "-> the local posterior learns its WIDTH (genuine uncertainty), "
                         "which is essential at low SNR. 'mean' freezes delta_logvar (no "
                         "gradient) -> a fake MAP point estimate; production used mean only "
                         "because its greedy candidates were already sub-bin.")
    ap.add_argument("--recon_weight", type=float, default=1.0,
                    help="Recon-NLL coefficient -- the ONLY gradient driving delta in the "
                         "unsup branch (local_nll is zeroed). delta optimizes against the "
                         "CONTINUOUS recon likelihood, so even an 8-Hz basin can be refined "
                         "to the true freq if selection is correct -- the 0.50 plateau is an "
                         "OPTIMIZATION failure (weak gradient), NOT a grid floor. Production "
                         "used 0.01 only because greedy candidates were already sub-bin. "
                         "Raised to 1.0 so the recon gradient can drive delta the full bin.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Stage2 learning rate (runmod default 3e-4). Raised to push delta "
                         "further per step across the full grid bin.")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Stage2 batch size (runmod default 32). Larger -> less noisy recon "
                         "gradient, important under sample_mode stochasticity.")
    ap.add_argument("--kl_anneal_floor", type=float, default=1.0,
                    help="If <1.0, anneal local_kl from 1.0 down to this floor over training "
                         "(--kl_anneal_mode down) so the N(0,1) delta prior stops pulling "
                         "delta back to 0 while recon pushes it out. 1.0 = no anneal.")
    ap.add_argument("--profile_ridge", type=float, default=1e-3,
                    help="Decoder LS ridge (point_ls). Restored to 1e-3: the min-gap dedupe "
                         "is NOT sufficient because cartesian_topk evaluates 3^4 frequency "
                         "COMBINATIONS, and a delta-shifted top-3 from two basins can still "
                         "coincide -> singular Gram (crashed at ep11 with 1e-5).")
    ap.add_argument("--include_local_time", action="store_true",
                    help="把真实非均匀采样时刻喂进 encoder: 追加 t_local=(t-t0)/(t_max-t0) "
                         "与 dt_norm=(t_n-t_{n-1})/median(dt) 两通道 (input_dim 6->8)。亚 bin "
                         "频偏的信息编码在相位 2*pi*f*t_n 随真实 t_n 的演化里; 默认 6 维特征只有 "
                         "rev_norm(整数圈号)+角度, 抹平了非均匀间隔 -> delta_head 没有原料做亚 bin "
                         "定位 -> delta 恒为 0、freq_mae 卡在候选网格精度 0.50。开此项验证 (b1)。 "
                         "注意: input_proj 维度 6->8, production/旧 Stage1 ckpt 的该层不匹配, "
                         "会随机初始化并在 Stage1 重新适配。")
    cli = ap.parse_args()

    runmod = _load_runmod()
    ckpt = torch.load(cli.fr_checkpoint, map_location="cpu", weights_only=False)
    from argparse import Namespace
    fa = Namespace(**ckpt["args"])
    fr_model = build_front(fa, int(fa.num_cycles) * 4)
    fr_model.load_state_dict(ckpt["model"]); fr_model.eval()
    grid_hz = np.linspace(float(fa.global_fmin_hz), float(fa.global_fmax_hz), int(fa.fr_size), endpoint=False)
    # Replace greedy candidate generation with FR peaks for BOTH stages (shared dataset).
    # The factory's returned radius becomes local_radius_hz (the refinement radius
    # f = center + radius*delta, delta in [-1,1]) -- NOT the basin half-width. Pass
    # local_radius_hz here so delta can reach the FR half-bin (~0.98 Hz).
    runmod.make_greedy_basin_pool_candidates = _fr_pool_factory(
        fr_model, grid_hz, cli.local_radius_hz)

    common = [
        "--candidate_support_mode", "basin",
        "--candidate_mode", "profile_ls",   # zeroes alias/bse -> ~pure FR-derived features
        "--basin_min_half_width_hz", str(cli.basin_half_width_hz),
        "--global_bins", "512", "--top_j", "24",
        "--num_cycles", str(cli.num_cycles),
        "--train_samples", str(cli.train_samples),
    ]
    if cli.include_local_time:
        common += ["--include_local_time"]

    # ---- Stage 1: z-only selection on the FR pool ----
    s1_out = f"{cli.out_root}_stage1"
    argv1 = common + [
        "--z_only_train", "--z_loss_type", "soft_freq", "--z_target_temperature_hz", "1.0",
        "--init_encoder_checkpoint", cli.prod_checkpoint,
        "--epochs", str(cli.stage1_epochs), "--out_dir", s1_out,
    ]
    print("=== Stage 1: z-only on FR pool ===")
    runmod.train(runmod.parse_args(argv1))

    # ---- Stage 2: unsupervised ELBO local-posterior refinement on the FR pool ----
    s1_ckpt = str(Path(s1_out) / "four_component_candidate_encoder.pt")
    s2_out = f"{cli.out_root}_stage2"
    argv2 = common + [
        "--unsupervised_elbo_train",
        "--elbo_frequency_source", "cartesian_topk", "--cartesian_topk", "3",
        "--z_repulsion_weight", "1.0",
        # Unlike production (greedy candidates were sub-bin so recon_weight 0.01 sufficed
        # for delta to merely polish), FR candidates are grid-locked at 1.95 Hz: raw_mae
        # is frozen at 0.50 and delta must traverse the full ~0.5 Hz on the recon gradient
        # ALONE (local_nll zeroed). recon_weight 0.01 was too weak (0.505->0.495 in 10 ep).
        # Strengthen recon and optionally anneal the delta-KL down so N(0,1) stops pulling
        # delta back to 0.
        "--recon_weight", str(cli.recon_weight),
        "--sample_mode", cli.sample_mode,  # 'sample' -> delta_logvar trainable (low-SNR uncertainty)
        "--local_nll_weight", "0.0",      # (already zeroed in the unsup branch; explicit)
        "--local_kl_weight", str(cli.local_kl_weight),  # default 0.001 = production
        "--recon_mode", cli.recon_mode,   # point_ls first (reproduce 0.169), marginal later
        "--amp_prior_var_scale", str(cli.amp_prior_var_scale),
        # point_ls decoder ridge: cartesian_topk combines a delta-shifted top-3 from each
        # of the 4 components into 3^4 LS solves; two can coincide -> singular Gram. 1e-5
        # crashed at ep11. marginal wouldn't need this (prior precision regularizes).
        "--profile_ridge", str(cli.profile_ridge),
        "--lr", str(cli.lr),
        "--batch_size", str(cli.batch_size),
        "--init_encoder_checkpoint", s1_ckpt,
        "--epochs", str(cli.stage2_epochs), "--out_dir", s2_out,
    ]
    if cli.kl_anneal_floor < 1.0:
        # Anneal delta-KL down: strong N(0,1) prior early (stabilize), relax it later so
        # delta can refine the full grid offset without being pulled back to 0.
        steps_per_epoch = max(1, cli.train_samples // 64)
        argv2 += [
            "--kl_anneal_mode", "down",
            "--kl_anneal_floor", str(cli.kl_anneal_floor),
            "--kl_anneal_steps", str(int(0.5 * cli.stage2_epochs * steps_per_epoch)),
        ]
    print("\n=== Stage 2: unsupervised ELBO on FR pool ===")
    runmod.train(runmod.parse_args(argv2))
    print(f"\n[done] stage1 -> {s1_out}\n[done] stage2 -> {s2_out}")


if __name__ == "__main__":
    main()
