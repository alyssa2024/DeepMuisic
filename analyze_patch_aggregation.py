import json
import os
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import (
    BTTPatchDataset,
    build_btt_point_features,
    chronological_train_val_split,
)
from Encoder import VariationalIndependentTimeSeriesTransformer
from eval import collect_patch_predictions
from main import set_global_seed
from synthesis_dataset import (
    generate_complex_harmonic_displacement,
    simulate_fluctuating_speed_btt,
)
from VAE import PhysicalHarmonicVAE


def build_model(device):
    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    model_cfg = CONFIG["model"]
    prior_cfg = CONFIG.get("prior", {})

    encoder = VariationalIndependentTimeSeriesTransformer(
        input_dim=data_cfg["input_dim"],
        output_dim=data_cfg["num_harmonics"],
        hidden_dim=model_cfg["hidden_dim"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        hidden_dim_dense=model_cfg["hidden_dim_dense"],
        num_probes=data_cfg["num_probes"],
        use_standard_pe=model_cfg["use_standard_pe"],
        device=device,
        f_center_hz=prior_cfg.get("f_center_hz", signal_cfg["freqs_hz"]),
        f_band_hz=prior_cfg.get("f_band_hz", 15.0),
    )

    model = PhysicalHarmonicVAE(
        encoder,
        ls_ridge=model_cfg.get("ls_ridge", 1e-6),
        use_amp_residual=model_cfg.get("use_amp_residual", True),
        amp_residual_hidden=model_cfg.get("amp_residual_hidden", 128),
        amp_residual_gamma=model_cfg.get("amp_residual_gamma", 0.0),
        use_freq_mean_for_ls=model_cfg.get("use_freq_mean_for_ls", True),
    ).to(device)
    return model


def group_records_by_revolution(records, group_span_revs=64):
    groups = defaultdict(list)
    for record in records:
        group_id = record["rev_start"] // group_span_revs
        groups[group_id].append(record)
    return groups


def _normalize_weights(w):
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    return w / np.sum(w)


def aggregate_group(records, method="mean"):
    freqs = np.stack([r["mu_f"] for r in records], axis=0)  # [N, K]
    coeffs = np.stack([r["c_hat_ls_m"] for r in records], axis=0)  # [N, K]

    if method == "mean":
        w = np.ones(len(records), dtype=np.float64)
    elif method == "residual_weighted":
        residuals = np.array([r["recon_residual"] for r in records], dtype=np.float64)
        w = 1.0 / (residuals + 1e-12)
    elif method == "boundary_weighted":
        margins = np.array([r["boundary_margin"] for r in records], dtype=np.float64)
        w = np.maximum(margins, 1e-6)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    w = _normalize_weights(w)
    f_agg = np.sum(freqs * w[:, None], axis=0)
    c_agg = np.sum(coeffs * w[:, None], axis=0)
    return f_agg, c_agg


def smooth_then_mean(records, alpha=0.6):
    """
    Simple EMA smoothing on patch-level frequency tracks, then average.
    """
    freqs = np.stack([r["mu_f"] for r in records], axis=0)  # [N, K]
    smoothed = np.zeros_like(freqs)
    smoothed[0] = freqs[0]
    for i in range(1, len(freqs)):
        smoothed[i] = alpha * smoothed[i - 1] + (1.0 - alpha) * freqs[i]

    f_agg = smoothed.mean(axis=0)

    coeffs = np.stack([r["c_hat_ls_m"] for r in records], axis=0)
    c_agg = coeffs.mean(axis=0)
    return f_agg, c_agg


def evaluate_aggregated_frequency(groups, true_freqs_hz, tol_hz=1.0):
    true_freqs = np.asarray(true_freqs_hz, dtype=np.float64)
    results = {}

    for method in ["mean", "residual_weighted", "boundary_weighted", "smooth_then_mean"]:
        freq_errs = []
        success = []

        for records in groups.values():
            if len(records) == 0:
                continue

            if method == "smooth_then_mean":
                f_hat, _ = smooth_then_mean(records)
            else:
                f_hat, _ = aggregate_group(records, method=method)

            err = f_hat - true_freqs
            freq_errs.append(err)
            success.append(float(np.all(np.abs(err) <= tol_hz)))

        if len(freq_errs) == 0:
            results[method] = {
                "freq_rmse_hz": float("nan"),
                "freq_mae_hz": float("nan"),
                "detection_success_rate": float("nan"),
                "num_groups": 0,
            }
            continue

        freq_errs = np.stack(freq_errs, axis=0)
        rmse = np.sqrt(np.mean(freq_errs ** 2))
        mae = np.mean(np.abs(freq_errs))
        success_rate = float(np.mean(success))

        results[method] = {
            "freq_rmse_hz": float(rmse),
            "freq_mae_hz": float(mae),
            "detection_success_rate": success_rate,
            "num_groups": int(len(freq_errs)),
        }

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Patch aggregation ablation analysis.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional full checkpoint path. If omitted, use CONFIG['checkpoint']['dir']/best_freq_rmse.pt",
    )
    parser.add_argument(
        "--group-span-revs",
        type=int,
        default=64,
        help="Number of revolutions per aggregation group.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(CONFIG.get("seed", 42))

    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    eval_cfg = CONFIG.get("eval", {})

    run_dir = CONFIG.get("run_dir", ".")
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    ckpt_path = args.ckpt or os.path.join(CONFIG["checkpoint"]["dir"], "best_freq_rmse.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path}")

    t_samples, freqs_per_rev, rev_ids, probe_ids, theta_samples, freqs_at_samples = (
        simulate_fluctuating_speed_btt(
            n_revs=data_cfg["n_revs"],
            base_freq_x=data_cfg["base_freq"],
            delta=data_cfg["fluctuation_delta"],
            probe_angles=data_cfg["probes"],
        )
    )
    _ = freqs_per_rev

    x_observed, _ = generate_complex_harmonic_displacement(
        t=t_samples,
        freqs=signal_cfg["freqs_hz"],
        amp_real=signal_cfg["amp_real_m"],
        amp_imag=signal_cfg["amp_imag_m"],
        snr_db=signal_cfg["snr_db"],
    )

    amp_scale = np.std(x_observed)
    x_observed_norm = x_observed / (amp_scale + 1e-12)

    features, t_samples, rev_ids, probe_ids = build_btt_point_features(
        x_observed=x_observed_norm,
        t_samples=t_samples,
        rev_ids=rev_ids,
        probe_ids=probe_ids,
        theta_samples=theta_samples,
        freqs_at_samples=freqs_at_samples,
        base_freq=data_cfg["base_freq"],
        n_revs=data_cfg["n_revs"],
    )

    dataset = BTTPatchDataset(
        features=features,
        t_samples=t_samples,
        rev_ids=rev_ids,
        probe_ids=probe_ids,
        window_revs=data_cfg["window_revs"],
        hop_revs=data_cfg["hop_revs"],
        num_probes=data_cfg["num_probes"],
    )

    _, val_set, split_info = chronological_train_val_split(
        dataset,
        val_ratio=eval_cfg.get("val_ratio", 0.2),
    )
    print("Split info:", split_info)

    val_loader = DataLoader(
        val_set,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    records = collect_patch_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        amp_scale=amp_scale,
    )

    groups = group_records_by_revolution(records, group_span_revs=args.group_span_revs)

    results = evaluate_aggregated_frequency(
        groups=groups,
        true_freqs_hz=signal_cfg["freqs_hz"],
        tol_hz=CONFIG["loss"].get("freq_success_tol_hz", 1.0),
    )

    out_path = os.path.join(run_dir, "aggregation_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
