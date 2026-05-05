import argparse
import json
import os

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
from eval import compare_ls_with_predfreq_vs_truefreq
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


def parse_args():
    parser = argparse.ArgumentParser(description="LS attribution: predfreq vs truefreq.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional full checkpoint path. Default: CONFIG['checkpoint']['dir']/best_freq_rmse.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(CONFIG.get("seed", 42))

    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    eval_cfg = CONFIG.get("eval", {})
    loss_cfg = CONFIG.get("loss", {})

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

    # Rebuild the same synthetic dataset as training/eval.
    t_samples, _freqs_per_rev, rev_ids, probe_ids, theta_samples, freqs_at_samples = (
        simulate_fluctuating_speed_btt(
            n_revs=data_cfg["n_revs"],
            base_freq_x=data_cfg["base_freq"],
            delta=data_cfg["fluctuation_delta"],
            probe_angles=data_cfg["probes"],
        )
    )

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

    results = compare_ls_with_predfreq_vs_truefreq(
        model=model,
        dataloader=val_loader,
        device=device,
        true_freqs_hz=signal_cfg["freqs_hz"],
        true_amp_real=signal_cfg["amp_real_m"],
        true_amp_imag=signal_cfg["amp_imag_m"],
        amp_scale=amp_scale,
        amp_success_tol_m=loss_cfg.get("amp_success_tol_m", 1e-4),
    )

    out_path = os.path.join(run_dir, "ls_attribution.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
