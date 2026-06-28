"""Geometry-robust DeepFreq front-end for BTT.

The official DeepFreq FR module flattens [Re;Im] through a Linear(2N,...) first
layer, so the sampling times t_n are only IMPLICIT in the weights -> it memorizes
one sampling geometry and its de-aliasing collapses under speed ramps (verified:
+5% ramp -> MAE 0.39 -> 39.8 Hz). This script keeps everything downstream of the
first layer identical to the official FR module and only changes how t_n enters,
comparing two injections:

  --front naive4ch : in_layer = Linear(4N,...), input = [Re, Im, t_local, dt_norm].
      t_n is given as extra channels; the FC must learn to use it. Cheap.

  --front nudft    : in_layer = a learnable non-uniform DFT. For M learnable
      frequencies f_m, compute  s_m = sum_n y_n exp(-j 2 pi f_m t_n)  (real+imag),
      i.e. t_n multiplies f_m INSIDE the phase -- a learnable profile-LS. Geometry
      robustness is STRUCTURAL (true t_n used in the phase), not memorized.

Both are trained with speed-randomized data (constant / linear_up / linear_down
with random magnitude) so the front-end must cope with varying geometry, matching
the deployment-covers-all-types assumption.

This is a PROPOSAL front-end: we report top-M basin recall (does the FR cover the
true tones?) -- the metric that matters for replacing the OMP/greedy candidate
generator. Final sub-Hz accuracy and amplitude are the downstream VI's job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DEEPFREQ_DIR = ROOT / "comparison" / "deepfreq_official"
BTT_LEGACY_DIR = ROOT / "btt_amortized_vi_project" / "legacy_reuse" / "current_project"
for p in (str(DEEPFREQ_DIR), str(BTT_LEGACY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, str(p))

from config import CONFIG  # noqa: E402
from synthesis_dataset import generate_one_btt_sequence  # noqa: E402
from synthesis_dataset import compute_frequency_support  # noqa: E402
# Reuse the official baseline's eval/peak/target helpers for 1:1 comparability.
from deepfreq_btt_baseline import (  # noqa: E402
    gaussian_frequency_target, top_m_peaks, match_errors, summarize_errors,
)


# --------------------------------------------------------------------------- #
# Data: variable speed geometry + t_n channels                                 #
# --------------------------------------------------------------------------- #
def _time_channels(t):
    n = t.shape[0]
    span = float(t[-1] - t[0]) if n > 1 else 1.0
    t_local = (t - t[0]) / (span + 1e-12)
    dt = np.empty_like(t)
    if n > 1:
        dt[1:] = np.diff(t)
        med = float(np.median(dt[1:])) or 1.0
        dt[0] = med
    else:
        med = 1.0; dt[0] = med
    return t_local.astype(np.float32), (dt / (med + 1e-12)).astype(np.float32)


class BTTGeomDataset(Dataset):
    """Speed-randomized BTT proposal-training data. Returns [Re,Im,t_local,dt_norm],
    the raw t (for the nudft front), the Gaussian FR target, and true freqs."""

    def __init__(self, n_samples, seed, args, grid_hz):
        self.n = n_samples
        self.seed = seed
        self.args = args
        self.grid_hz = np.asarray(grid_hz, dtype=np.float64)
        fc, sc, dc = CONFIG["frequency"], CONFIG["signal"], CONFIG["data"]
        self.lower, self.upper, _, _ = compute_frequency_support(
            freq_center_hz=fc["center_hz"], relative_half_band=fc["relative_half_band"])
        self.amp_real = np.asarray(sc["amp_real_center_m"], dtype=np.float64)
        self.amp_imag = np.asarray(sc["amp_imag_center_m"], dtype=np.float64)
        self.dc = dc

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        # frequency: jitter inside the production band (same as BTTSequenceDataset).
        freq = rng.uniform(self.lower, self.upper)
        # speed geometry randomization
        if self.args.train_speed_random:
            prof = rng.choice(["constant", "linear_up", "linear_down"])
            change = 0.0 if prof == "constant" else float(rng.uniform(0.0, self.args.train_speed_max))
        else:
            prof, change = "constant", 0.0
        s = generate_one_btt_sequence(
            num_cycles=self.args.num_cycles, base_freq=self.dc["base_freq"],
            fluctuation_delta=self.args.fluctuation_delta, probe_angles=self.dc["probes"],
            freq_hz=freq.tolist(), amp_real=self.amp_real.tolist(), amp_imag=self.amp_imag.tolist(),
            snr_db=self.args.snr_db, rng=rng, speed_profile=prof, speed_change=change)
        y = s["x_observed"].astype(np.complex128)
        y = y / (np.std(y) + 1e-12)
        t = s["t_samples"].astype(np.float64)
        t_local, dt_norm = _time_channels(t)
        chans = np.stack([y.real.astype(np.float32), y.imag.astype(np.float32), t_local, dt_norm])
        target = gaussian_frequency_target(freq, self.grid_hz, self.args.target_sigma_hz)
        return {
            "chans": torch.tensor(chans, dtype=torch.float32),   # [4, N]
            "t": torch.tensor(t, dtype=torch.float32),           # [N]
            "target_fr": torch.tensor(target, dtype=torch.float32),
            "true_freq_hz": torch.tensor(np.sort(freq), dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Two geometry-robust first layers; shared conv+upsample tail (official FR).    #
# --------------------------------------------------------------------------- #
class _FRTail(nn.Module):
    """Official FR conv stack + transpose-conv upsampler (verbatim structure)."""
    def __init__(self, n_filters, n_layers, inner_dim, kernel_size, upsampling, kernel_out):
        super().__init__()
        self.n_filters = n_filters
        mod = []
        for _ in range(n_layers):
            mod += [nn.Conv1d(n_filters, n_filters, kernel_size, padding="same",
                              bias=False, padding_mode="circular"),
                    nn.BatchNorm1d(n_filters), nn.ReLU()]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(
            n_filters, 1, kernel_out, stride=upsampling,
            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)

    def forward(self, x):  # x: [B, n_filters, inner_dim]
        x = self.mod(x)
        return self.out_layer(x).view(x.size(0), -1)


class Naive4ChFR(nn.Module):
    def __init__(self, signal_dim, n_filters, n_layers, inner_dim, kernel_size, upsampling, kernel_out):
        super().__init__()
        self.in_layer = nn.Linear(4 * signal_dim, inner_dim * n_filters, bias=False)
        self.n_filters = n_filters
        self.tail = _FRTail(n_filters, n_layers, inner_dim, kernel_size, upsampling, kernel_out)

    def forward(self, chans, t=None):  # chans: [B,4,N]
        b = chans.size(0)
        x = self.in_layer(chans.reshape(b, -1)).view(b, self.n_filters, -1)
        return self.tail(x)


class NUDFTFR(nn.Module):
    """Learnable non-uniform DFT front: s_m = sum_n y_n exp(-j2pi f_m t_n).
    f_m are learnable (init = uniform over the band). t_n enters the phase, so the
    transform tracks the actual sampling geometry -> structural robustness."""
    def __init__(self, signal_dim, n_filters, n_layers, inner_dim, kernel_size,
                 upsampling, kernel_out, n_basis, fmin_hz, fmax_hz):
        super().__init__()
        self.n_filters = n_filters
        self.freqs = nn.Parameter(torch.linspace(fmin_hz, fmax_hz, n_basis))  # [M]
        self.proj = nn.Linear(2 * n_basis, inner_dim * n_filters, bias=False)
        self.tail = _FRTail(n_filters, n_layers, inner_dim, kernel_size, upsampling, kernel_out)

    def forward(self, chans, t):  # chans: [B,4,N]; t: [B,N]
        b = chans.size(0)
        yr, yi = chans[:, 0], chans[:, 1]           # [B,N]
        phase = 2 * np.pi * self.freqs[None, :, None] * t[:, None, :]  # [B,M,N]
        cos, sin = torch.cos(phase), torch.sin(phase)
        # s_m = sum_n (yr+i yi)(cos - i sin) ; real = yr cos + yi sin, imag = yi cos - yr sin
        sr = (yr[:, None, :] * cos + yi[:, None, :] * sin).sum(-1)  # [B,M]
        si = (yi[:, None, :] * cos - yr[:, None, :] * sin).sum(-1)  # [B,M]
        feat = torch.cat([sr, si], dim=-1)                         # [B,2M]
        x = self.proj(feat).view(b, self.n_filters, -1)
        return self.tail(x)


def build_front(args, signal_dim):
    common = dict(n_filters=args.fr_n_filters, n_layers=args.fr_n_layers,
                  inner_dim=args.fr_inner_dim, kernel_size=args.fr_kernel_size,
                  upsampling=args.fr_upsampling, kernel_out=args.fr_kernel_out)
    if args.front == "naive4ch":
        return Naive4ChFR(signal_dim, **common)
    if args.front == "nudft":
        return NUDFTFR(signal_dim, n_basis=args.nudft_basis,
                       fmin_hz=args.global_fmin_hz, fmax_hz=args.global_fmax_hz, **common)
    raise ValueError(args.front)


# --------------------------------------------------------------------------- #
def make_loader(n, seed, args, grid, shuffle):
    return DataLoader(BTTGeomDataset(n, seed, args, grid), batch_size=args.batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_speed(model, args, grid, device):
    """Top-M basin recall + grid MAE across a speed ramp -- the geometry-robustness test."""
    out = {}
    for signed in args.eval_speed_changes:
        prof = "constant" if signed == 0 else ("linear_up" if signed > 0 else "linear_down")
        change = abs(signed)
        rng = np.random.default_rng(args.seed + 777)
        abs_errs, recalls = [], []
        fc = CONFIG["frequency"]
        lower, upper, _, _ = compute_frequency_support(
            freq_center_hz=fc["center_hz"], relative_half_band=fc["relative_half_band"])
        dc, sc = CONFIG["data"], CONFIG["signal"]
        for _ in range(args.eval_samples):
            freq = rng.uniform(lower, upper)
            s = generate_one_btt_sequence(
                num_cycles=args.num_cycles, base_freq=dc["base_freq"],
                fluctuation_delta=args.fluctuation_delta, probe_angles=dc["probes"],
                freq_hz=freq.tolist(), amp_real=sc["amp_real_center_m"], amp_imag=sc["amp_imag_center_m"],
                snr_db=args.snr_db, rng=rng, speed_profile=prof, speed_change=change)
            y = s["x_observed"].astype(np.complex128); y = y / (np.std(y) + 1e-12)
            t = s["t_samples"].astype(np.float64)
            tl, dn = _time_channels(t)
            chans = torch.tensor(np.stack([y.real, y.imag, tl, dn]).astype(np.float32)[None], device=device)
            tt = torch.tensor(t.astype(np.float32)[None], device=device)
            fr = model(chans, tt).cpu().numpy()[0]
            true = np.sort(freq)
            pred = top_m_peaks(fr, grid, true.size)
            ae, _ = match_errors(pred, true); abs_errs.append(ae)
            cand = top_m_peaks(fr, grid, 8)
            dist = np.min(np.abs(cand[:, None] - true[None, :]), axis=0)
            recalls.append(dist <= args.basin_recall_tol_hz)
        ae = np.concatenate([a.reshape(1, -1) for a in abs_errs])
        rec = np.concatenate([r for r in recalls])
        out[f"{signed:+.2f}"] = {"grid_mae_hz": float(np.mean(np.abs(ae))),
                                 "topm8_basin_recall": float(np.mean(rec))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="comparison/artifacts/deepfreq_geomrobust_smoke")
    p.add_argument("--front", choices=["naive4ch", "nudft"], default="nudft")
    p.add_argument("--seed", type=int, default=20260626)
    p.add_argument("--train_samples", type=int, default=4000)
    p.add_argument("--eval_samples", type=int, default=200)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_cycles", type=int, default=16)
    p.add_argument("--snr_db", type=float, default=20.0)
    p.add_argument("--fluctuation_delta", type=float, default=0.001)
    p.add_argument("--train_speed_random", action="store_true")
    p.add_argument("--train_speed_max", type=float, default=0.10)
    p.add_argument("--eval_speed_changes", type=float, nargs="+", default=[-0.10, -0.05, 0.0, 0.05, 0.10])
    p.add_argument("--global_fmin_hz", type=float, default=1.0)
    p.add_argument("--global_fmax_hz", type=float, default=1000.0)
    p.add_argument("--fr_size", type=int, default=512)
    p.add_argument("--target_sigma_hz", type=float, default=3.0)
    p.add_argument("--fr_n_layers", type=int, default=6)
    p.add_argument("--fr_n_filters", type=int, default=32)
    p.add_argument("--fr_kernel_size", type=int, default=3)
    p.add_argument("--fr_kernel_out", type=int, default=17)
    p.add_argument("--fr_inner_dim", type=int, default=64)
    p.add_argument("--fr_upsampling", type=int, default=8)
    p.add_argument("--nudft_basis", type=int, default=256)
    p.add_argument("--basin_recall_tol_hz", type=float, default=8.0)
    args = p.parse_args()
    assert args.fr_size == args.fr_inner_dim * args.fr_upsampling

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = np.linspace(args.global_fmin_hz, args.global_fmax_hz, args.fr_size, endpoint=False)
    signal_dim = args.num_cycles * CONFIG["data"]["num_probes"]
    model = build_front(args, signal_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr = make_loader(args.train_samples, args.seed + 1, args, grid, True)

    print(f"front={args.front} speed_random={args.train_speed_random} signal_dim={signal_dim} "
          f"device={device}")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); tot = 0.0; cnt = 0
        for b in tr:
            chans = b["chans"].to(device); t = b["t"].to(device); tgt = b["target_fr"].to(device)
            fr = model(chans, t)
            loss = F.mse_loss(fr, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * chans.size(0); cnt += chans.size(0)
        if epoch % 20 == 0 or epoch == 1:
            print(f"[epoch {epoch:3d}] train_mse={tot/cnt:.6f}", flush=True)

    model.eval()
    speed_metrics = evaluate_speed(model, args, grid, device)
    print(f"\n=== Geometry-robustness (front={args.front}, "
          f"speed_random={args.train_speed_random}) ===")
    print(f"{'speed':>7}{'grid_mae_hz':>13}{'topm8_recall':>14}")
    for k, v in speed_metrics.items():
        print(f"{k:>7}{v['grid_mae_hz']:>13.3f}{v['topm8_basin_recall']:>14.3f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "speed_metrics": speed_metrics,
               "runtime_seconds": time.time() - t0},
              open(out / "metrics.json", "w"), indent=2)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "geomrobust_fr.pt")
    print(f"\n[saved] {out/'metrics.json'}")


if __name__ == "__main__":
    main()
