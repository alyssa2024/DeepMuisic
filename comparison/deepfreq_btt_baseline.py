"""Minimal DeepFreq baseline on the synthetic BTT task.

This script intentionally keeps the official DeepFreq frequency-representation
network intact and adapts only the data/target/evaluation layer:

    BTT complex sequence -> [real, imag] -> DeepFreq spectrum -> top-K peaks.

It is meant as a controlled proposal-front-end baseline, not a full BTT model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEEPFREQ_DIR = ROOT / "comparison" / "deepfreq_official"
BTT_LEGACY_DIR = ROOT / "btt_amortized_vi_project" / "legacy_reuse" / "current_project"

for path in (DEEPFREQ_DIR, BTT_LEGACY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import modules as deepfreq_modules  # noqa: E402
from config import CONFIG  # noqa: E402
from dataset import BTTSequenceDataset  # noqa: E402
from synthesis_dataset import compute_frequency_support  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_frequency_support(args):
    freq_cfg = CONFIG["frequency"]
    signal_cfg = CONFIG["signal"]
    if args.controlled_spacing_hz > 0:
        n_comp = int(args.num_components)
        spacing = float(args.controlled_spacing_hz)
        center = float(args.center_band_hz)
        centers = center + spacing * (np.arange(n_comp) - (n_comp - 1) / 2.0)
        jitter = min(float(args.spacing_jitter_hz), 0.45 * spacing)
        lower = np.maximum(centers - jitter, 1e-6)
        upper = centers + jitter
        amp_real = np.asarray(signal_cfg["amp_real_center_m"], dtype=np.float64)
        amp_imag = np.asarray(signal_cfg["amp_imag_center_m"], dtype=np.float64)
        recycle = np.arange(n_comp) % len(amp_real)
        return lower, upper, amp_real[recycle], amp_imag[recycle]
    lower, upper, _centers, _half = compute_frequency_support(
        freq_center_hz=freq_cfg["center_hz"],
        relative_half_band=freq_cfg["relative_half_band"],
    )
    return (
        lower,
        upper,
        np.asarray(signal_cfg["amp_real_center_m"], dtype=np.float64),
        np.asarray(signal_cfg["amp_imag_center_m"], dtype=np.float64),
    )


def gaussian_frequency_target(freq_hz: np.ndarray, grid_hz: np.ndarray, sigma_hz: float) -> np.ndarray:
    target = np.zeros((grid_hz.shape[0],), dtype=np.float32)
    sigma = max(float(sigma_hz), 1e-6)
    for freq in np.asarray(freq_hz, dtype=np.float64):
        target += np.exp(-((grid_hz - float(freq)) ** 2) / (sigma**2)).astype(np.float32)
    return target


class BTTDeepFreqDataset(Dataset):
    def __init__(self, n_samples: int, seed: int, args, grid_hz: np.ndarray):
        data_cfg = CONFIG["data"]
        signal_cfg = CONFIG["signal"]
        freq_lower, freq_upper, amp_real, amp_imag = build_frequency_support(args)
        self.base = BTTSequenceDataset(
            num_sequences=n_samples,
            num_cycles=args.num_cycles,
            num_probes=data_cfg["num_probes"],
            base_freq=data_cfg["base_freq"],
            fluctuation_delta=args.fluctuation_delta,
            probe_angles=data_cfg["probes"],
            freq_lower=freq_lower,
            freq_upper=freq_upper,
            amp_real_center=amp_real,
            amp_imag_center=amp_imag,
            amp_relative_half_band=signal_cfg["amp_data_prior"]["relative_half_band"],
            amp_min_half_band=signal_cfg["amp_data_prior"]["min_half_band_m"],
            snr_db=args.snr_db,
            seed=seed,
            normalization=data_cfg.get("normalization", "per_sequence_std"),
            speed_profile=args.speed_profile,
            speed_change=args.speed_change,
        )
        self.grid_hz = np.asarray(grid_hz, dtype=np.float64)
        self.target_sigma_hz = float(args.target_sigma_hz)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        y = item["target"].float().transpose(0, 1).contiguous()  # [2, N]
        freq = item["true_freq_hz"].float()
        target = gaussian_frequency_target(freq.numpy(), self.grid_hz, self.target_sigma_hz)
        return {
            "signal": y,
            "target_fr": torch.as_tensor(target, dtype=torch.float32),
            "true_freq_hz": freq,
            "t": item["t"].float(),
            "target_complex": item["target"].float(),
        }


def make_loaders(args, grid_hz):
    specs = [
        ("train", args.train_samples, args.seed + 1),
        ("val", args.val_samples, args.seed + 2),
        ("test", args.test_samples, args.seed + 3),
    ]
    datasets = {
        split: BTTDeepFreqDataset(n_samples, seed, args, grid_hz)
        for split, n_samples, seed in specs
    }
    return {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size),
    }


def local_peak_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        return np.asarray([], dtype=np.int64)
    if values.size < 3:
        return np.arange(values.size, dtype=np.int64)
    peaks = np.where((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
    if values[0] >= values[1]:
        peaks = np.r_[0, peaks]
    if values[-1] >= values[-2]:
        peaks = np.r_[peaks, values.size - 1]
    return peaks.astype(np.int64)


def top_m_peaks(fr_values: np.ndarray, grid_hz: np.ndarray, m: int) -> np.ndarray:
    peak_idx = local_peak_indices(fr_values)
    if peak_idx.size:
        order = peak_idx[np.argsort(fr_values[peak_idx])[::-1]]
    else:
        order = np.asarray([], dtype=np.int64)
    if order.size < m:
        used = set(int(i) for i in order)
        fallback = [int(i) for i in np.argsort(fr_values)[::-1] if int(i) not in used]
        order = np.asarray(list(order) + fallback, dtype=np.int64)
    return np.sort(grid_hz[order[:m]].astype(np.float64))


def match_errors(pred: np.ndarray, true: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    k = min(pred.size, true.size)
    best = None
    for perm in itertools.permutations(range(pred.size), k):
        err = pred[list(perm)] - true[:k]
        sse = float(np.sum(err**2))
        if best is None or sse < best[0]:
            best = (sse, np.abs(err), err)
    return best[1], best[2]


def residual_after_joint_tones(y_complex, t, freqs_hz, ridge=1e-5):
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    if freqs.size == 0:
        return y_complex
    phase = 2.0 * np.pi * t[:, None] * freqs[None, :]
    phi = np.exp(1j * phase)
    gram = phi.conj().T @ phi
    rhs = phi.conj().T @ y_complex
    amp = np.linalg.solve(gram + ridge * np.eye(freqs.size), rhs)
    return y_complex - phi @ amp


def joint_sse(y_complex, t, freqs_hz, ridge=1e-5):
    residual = residual_after_joint_tones(y_complex, t, freqs_hz, ridge=ridge)
    return float(np.sum(np.abs(residual) ** 2))


def refine_local_ls(y_complex, t, init_freqs, args):
    freqs = [float(f) for f in init_freqs]
    if not freqs:
        return np.asarray(freqs, dtype=np.float64)
    for _ in range(int(args.local_ls_passes)):
        for idx in range(len(freqs)):
            lo = max(float(args.global_fmin_hz), freqs[idx] - float(args.local_ls_half_width_hz))
            hi = min(float(args.global_fmax_hz), freqs[idx] + float(args.local_ls_half_width_hz))
            grid = np.linspace(lo, hi, int(args.local_ls_bins))
            best_f = freqs[idx]
            best_sse = float("inf")
            for candidate in grid:
                trial = list(freqs)
                trial[idx] = float(candidate)
                sse = joint_sse(y_complex, t, trial, ridge=args.profile_ridge)
                if sse < best_sse:
                    best_sse = sse
                    best_f = float(candidate)
            freqs[idx] = best_f
    return np.sort(np.asarray(freqs, dtype=np.float64))


def summarize_errors(abs_err_rows, signed_err_rows, success_tol_hz):
    abs_err = np.concatenate(abs_err_rows, axis=0)
    signed_err = np.concatenate(signed_err_rows, axis=0)
    return {
        "freq_mae_hz": float(np.mean(abs_err)),
        "freq_rmse_hz": float(np.sqrt(np.mean(signed_err**2))),
        "success_rate": float(np.mean(abs_err <= float(success_tol_hz))),
    }


@torch.no_grad()
def evaluate(model, loader, grid_hz, args, device):
    model.eval()
    grid_abs_err, grid_signed_err = [], []
    ls_abs_err, ls_signed_err = [], []
    recall_hits = {int(m): [] for m in args.recall_m}
    inference_times = []
    for batch in loader:
        signal = batch["signal"].to(device)
        start = time.perf_counter()
        output = model(signal)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_times.append((time.perf_counter() - start) / signal.shape[0])
        out_np = output.detach().cpu().numpy()
        true_np = batch["true_freq_hz"].numpy()
        t_np = batch["t"].numpy()
        target_np = batch["target_complex"].numpy()
        for idx in range(out_np.shape[0]):
            pred_topk = top_m_peaks(out_np[idx], grid_hz, args.known_k)
            abs_err, signed_err = match_errors(pred_topk, true_np[idx])
            grid_abs_err.append(abs_err.reshape(1, -1))
            grid_signed_err.append(signed_err.reshape(1, -1))
            for m in args.recall_m:
                candidates = top_m_peaks(out_np[idx], grid_hz, int(m))
                dist = np.min(np.abs(candidates[:, None] - true_np[idx][None, :]), axis=0)
                recall_hits[int(m)].append(dist <= args.basin_recall_tol_hz)
            if args.eval_local_ls:
                y_complex = target_np[idx, :, 0].astype(np.float64) + 1j * target_np[idx, :, 1].astype(np.float64)
                refined = refine_local_ls(y_complex, t_np[idx].astype(np.float64), pred_topk, args)
                abs_ls, signed_ls = match_errors(refined, true_np[idx])
                ls_abs_err.append(abs_ls.reshape(1, -1))
                ls_signed_err.append(signed_ls.reshape(1, -1))
    metrics = {
        "grid": summarize_errors(grid_abs_err, grid_signed_err, args.success_tol_hz),
        "topm_basin_recall": {
            str(m): float(np.mean(np.concatenate(recall_hits[int(m)], axis=0)))
            for m in args.recall_m
        },
        "mean_inference_time_ms_per_sample": float(1000.0 * np.mean(inference_times)),
    }
    if args.eval_local_ls:
        metrics["local_ls"] = summarize_errors(ls_abs_err, ls_signed_err, args.success_tol_hz)
    return metrics


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        signal = batch["signal"].to(device)
        target = batch["target_fr"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(signal)
        loss = F.mse_loss(output, target, reduction="mean")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * signal.shape[0]
        total_count += int(signal.shape[0])
    return total_loss / max(total_count, 1)


@torch.no_grad()
def representation_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        signal = batch["signal"].to(device)
        target = batch["target_fr"].to(device)
        output = model(signal)
        loss = F.mse_loss(output, target, reduction="mean")
        total_loss += float(loss.detach().cpu()) * signal.shape[0]
        total_count += int(signal.shape[0])
    return total_loss / max(total_count, 1)


def build_model_args(args, signal_dim):
    ns = argparse.Namespace()
    ns.fr_module_type = args.fr_module_type
    ns.signal_dim = int(signal_dim)
    ns.fr_size = int(args.fr_size)
    ns.fr_n_filters = int(args.fr_n_filters)
    ns.fr_inner_dim = int(args.fr_inner_dim)
    ns.fr_n_layers = int(args.fr_n_layers)
    ns.fr_kernel_size = int(args.fr_kernel_size)
    ns.fr_upsampling = int(args.fr_upsampling)
    ns.fr_kernel_out = int(args.fr_kernel_out)
    ns.use_cuda = False
    return ns


def main(argv=None):
    parser = argparse.ArgumentParser(description="DeepFreq baseline on synthetic BTT data")
    parser.add_argument("--out_dir", default="comparison/artifacts/deepfreq_btt_smoke")
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--device", default="")
    parser.add_argument("--train_samples", type=int, default=256)
    parser.add_argument("--val_samples", type=int, default=64)
    parser.add_argument("--test_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_cycles", type=int, default=4)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--speed_profile", choices=["random", "constant", "linear_up", "linear_down"], default="constant")
    parser.add_argument("--speed_change", type=float, default=0.0)
    parser.add_argument("--fluctuation_delta", type=float, default=0.001)
    parser.add_argument("--controlled_spacing_hz", type=float, default=0.0)
    parser.add_argument("--num_components", type=int, default=2)
    parser.add_argument("--center_band_hz", type=float, default=500.0)
    parser.add_argument("--spacing_jitter_hz", type=float, default=0.0)
    parser.add_argument("--known_k", type=int, default=4)
    parser.add_argument("--global_fmin_hz", type=float, default=1.0)
    parser.add_argument("--global_fmax_hz", type=float, default=1000.0)
    parser.add_argument("--fr_size", type=int, default=512)
    parser.add_argument("--target_sigma_hz", type=float, default=3.0)
    parser.add_argument("--fr_module_type", choices=["fr", "psnet"], default="fr")
    parser.add_argument("--fr_n_layers", type=int, default=6)
    parser.add_argument("--fr_n_filters", type=int, default=32)
    parser.add_argument("--fr_kernel_size", type=int, default=3)
    parser.add_argument("--fr_kernel_out", type=int, default=17)
    parser.add_argument("--fr_inner_dim", type=int, default=64)
    parser.add_argument("--fr_upsampling", type=int, default=8)
    parser.add_argument("--success_tol_hz", type=float, default=1.0)
    parser.add_argument("--basin_recall_tol_hz", type=float, default=8.0)
    parser.add_argument("--recall_m", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--eval_local_ls", action="store_true")
    parser.add_argument("--local_ls_half_width_hz", type=float, default=8.0)
    parser.add_argument("--local_ls_bins", type=int, default=17)
    parser.add_argument("--local_ls_passes", type=int, default=1)
    parser.add_argument("--profile_ridge", type=float, default=1e-5)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    grid_hz = np.linspace(args.global_fmin_hz, args.global_fmax_hz, args.fr_size, endpoint=False)
    loaders = make_loaders(args, grid_hz)
    signal_dim = int(args.num_cycles) * int(CONFIG["data"]["num_probes"])
    model_args = build_model_args(args, signal_dim)
    model = deepfreq_modules.set_fr_module(model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    best_state = None
    start_all = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss = representation_loss(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = evaluate(model, loaders["val"], grid_hz, args, device)
    test_metrics = evaluate(model, loaders["test"], grid_hz, args, device)
    payload = {
        "args": vars(args),
        "history": history,
        "best_val_loss": best_val,
        "val": val_metrics,
        "test": test_metrics,
        "runtime_seconds": time.time() - start_all,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.save_model:
        torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "deepfreq_btt_fr.pt")
    lines = [
        "# DeepFreq BTT Baseline",
        "",
        f"- Runtime seconds: `{payload['runtime_seconds']:.1f}`",
        f"- Best val representation loss: `{best_val:.6f}`",
        "",
        "## Test Metrics",
        "",
        "| Variant | MAE Hz | RMSE Hz | Success |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| DeepFreq-grid | {test_metrics['grid']['freq_mae_hz']:.4f} | "
            f"{test_metrics['grid']['freq_rmse_hz']:.4f} | "
            f"{test_metrics['grid']['success_rate']:.4f} |"
        ),
    ]
    if args.eval_local_ls:
        lines.append(
            f"| DeepFreq+local-LS | {test_metrics['local_ls']['freq_mae_hz']:.4f} | "
            f"{test_metrics['local_ls']['freq_rmse_hz']:.4f} | "
            f"{test_metrics['local_ls']['success_rate']:.4f} |"
        )
    lines.extend(["", "## Top-M Basin Recall", "", "| M | Recall |", "| ---: | ---: |"])
    for key, value in test_metrics["topm_basin_recall"].items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend([
        "",
        f"- Mean inference time: `{test_metrics['mean_inference_time_ms_per_sample']:.4f}` ms/sample",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir / 'metrics.json'}")
    print(f"[done] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
