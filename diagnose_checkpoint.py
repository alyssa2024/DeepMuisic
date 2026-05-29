import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import BTTSequenceDataset
from Encoder import VariationalIndependentTimeSeriesTransformer
from eval import evaluate_model
from synthesis_dataset import compute_frequency_support
from VAE import PhysicalHarmonicVAE


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_val_loader(data_cfg, signal_cfg, freq_lower, freq_upper, seed):
    amp_prior_cfg = signal_cfg["amp_data_prior"]
    val_set = BTTSequenceDataset(
        num_sequences=data_cfg["num_val_sequences"],
        num_cycles=data_cfg["num_cycles"],
        num_probes=data_cfg["num_probes"],
        base_freq=data_cfg["base_freq"],
        fluctuation_delta=data_cfg["fluctuation_delta"],
        probe_angles=data_cfg["probes"],
        freq_lower=freq_lower,
        freq_upper=freq_upper,
        amp_real_center=signal_cfg["amp_real_center_m"],
        amp_imag_center=signal_cfg["amp_imag_center_m"],
        amp_relative_half_band=amp_prior_cfg["relative_half_band"],
        amp_min_half_band=amp_prior_cfg["min_half_band_m"],
        snr_db=signal_cfg["snr_db"],
        seed=seed + 100000,
        normalization=data_cfg.get("normalization", "per_sequence_std"),
    )
    return DataLoader(
        val_set,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose sequence-level posterior checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. Default uses config.checkpoint.resume_from.",
    )
    args = parser.parse_args()

    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    freq_cfg = CONFIG["frequency"]
    model_cfg = CONFIG["model"]
    loss_cfg = CONFIG["loss"]
    ckpt_cfg = CONFIG.get("checkpoint", {})
    seed = CONFIG.get("seed", 42)
    set_global_seed(seed)

    ckpt_path = args.checkpoint or ckpt_cfg.get("resume_from")
    if not ckpt_path:
        raise ValueError("No checkpoint provided. Use --checkpoint or set checkpoint.resume_from.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    freq_lower, freq_upper, _, _ = compute_frequency_support(
        freq_center_hz=freq_cfg["center_hz"],
        relative_half_band=freq_cfg["relative_half_band"],
    )
    posterior_cfg = freq_cfg.get("posterior", {})
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
        freq_lower_hz=freq_lower,
        freq_upper_hz=freq_upper,
        min_log_rho2=posterior_cfg.get("min_log_rho2", -12.0),
        max_log_rho2=posterior_cfg.get("max_log_rho2", -4.0),
    )
    model = PhysicalHarmonicVAE(
        encoder=encoder,
        ls_ridge=model_cfg.get("ls_ridge", 1e-6),
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_loader = _build_val_loader(data_cfg, signal_cfg, freq_lower, freq_upper, seed)
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        loss_cfg=loss_cfg,
        dense_factor=CONFIG.get("eval", {}).get("dense_factor", 4),
    )

    mu_f_list = []
    std_f_list = []
    with torch.no_grad():
        for batch in val_loader:
            x_batch = batch["x"].to(device)
            t_batch = batch["t"].to(device)
            probe_ids = batch["probe_ids"].to(device)
            t_local = t_batch - t_batch[:, :1]
            outputs = model(x_batch, t_local, probe_ids=probe_ids)
            mu_f_list.append(outputs["mu_f"])
            std_f_list.append(outputs["std_f"])

    mu_f_all = torch.cat(mu_f_list, dim=0)
    std_f_all = torch.cat(std_f_list, dim=0)

    print("==== Frequency Posterior Diagnostics ====")
    print("freq_lower_hz:", model.encoder.freq_lower.detach().cpu().numpy())
    print("freq_upper_hz:", model.encoder.freq_upper.detach().cpu().numpy())
    print("mu_f_mean_hz:", mu_f_all.mean(dim=0).detach().cpu().numpy())
    print("mu_f_std_across_sequences_hz:", mu_f_all.std(dim=0, unbiased=False).detach().cpu().numpy())
    print("posterior_std_mean_hz:", std_f_all.mean(dim=0).detach().cpu().numpy())

    print("==== Eval Metrics ====")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6g}")


if __name__ == "__main__":
    main()
