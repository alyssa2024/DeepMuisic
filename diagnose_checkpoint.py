import argparse
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
from VAE import PhysicalHarmonicVAE
from eval import evaluate_model
from synthesis_dataset import (
    generate_complex_harmonic_displacement,
    simulate_fluctuating_speed_btt,
)


def build_prior_a_w(freqs_hz):
    true_w = 2 * np.pi * np.array(freqs_hz)
    return true_w / (2.0 * np.sqrt(2.0 / np.pi))


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Diagnose harmonic saturation and deterministic vs sampled eval from checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. Default uses config.checkpoint.resume_from.")
    args = parser.parse_args()

    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    model_cfg = CONFIG["model"]
    loss_cfg = CONFIG["loss"]
    eval_cfg = CONFIG.get("eval", {})
    ckpt_cfg = CONFIG.get("checkpoint", {})
    seed = CONFIG.get("seed", 42)
    set_global_seed(seed)

    ckpt_path = args.checkpoint or ckpt_cfg.get("resume_from")
    if not ckpt_path:
        raise ValueError("No checkpoint provided. Use --checkpoint or set checkpoint.resume_from in config.py")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")

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
    )
    model = PhysicalHarmonicVAE(encoder).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    t_samples, _, rev_ids, probe_ids, theta_samples, freqs_at_samples = simulate_fluctuating_speed_btt(
        n_revs=data_cfg["n_revs"],
        base_freq_x=data_cfg["base_freq"],
        delta=data_cfg["fluctuation_delta"],
        probe_angles=data_cfg["probes"],
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
        amp_agg_patches=data_cfg.get("amp_agg_patches", 1),
        amp_agg_mode=data_cfg.get("amp_agg_mode", "center"),
    )

    val_ratio = eval_cfg.get("val_ratio", 0.2)
    batch_size = data_cfg["batch_size"]
    dense_factor = eval_cfg.get("dense_factor", 4)

    _, val_set, split_info = chronological_train_val_split(
        dataset,
        val_ratio=val_ratio,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    print(
        "Eval split: "
        f"total={split_info['n_total']}, "
        f"val={split_info['n_val']}, "
        f"gap_windows={split_info['gap_windows']}, "
        f"val_start_idx={split_info['val_start_idx']}"
    )

    prior_a_w = build_prior_a_w(signal_cfg["freqs_hz"])
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        true_freqs_hz=signal_cfg["freqs_hz"],
        true_amp_real=signal_cfg["amp_real_m"],
        true_amp_imag=signal_cfg["amp_imag_m"],
        amp_scale=amp_scale,
        prior_a_w=prior_a_w,
        loss_cfg=loss_cfg,
        dense_factor=dense_factor,
    )

    mu_f_list = []
    amp_real_list = []
    amp_imag_list = []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 5:
                x_batch, t_batch, probe_ids_batch, _rev_ids_batch, target_batch = batch
                amp_t_batch = None
                amp_target_batch = None
            else:
                (
                    x_batch,
                    t_batch,
                    probe_ids_batch,
                    _rev_ids_batch,
                    target_batch,
                    amp_t_batch,
                    amp_target_batch,
                ) = batch
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            probe_ids_batch = probe_ids_batch.to(device)
            target_batch = target_batch.to(device)
            t_local = t_batch - t_batch[:, :1]
            if amp_t_batch is not None:
                amp_t_batch = amp_t_batch.to(device)
                amp_target_batch = amp_target_batch.to(device)
                amp_t_local = amp_t_batch - t_batch[:, :1]
                y_complex_ls = torch.complex(amp_target_batch[..., 0], amp_target_batch[..., 1])
                t_for_ls = amp_t_local
            else:
                y_complex_ls = torch.complex(target_batch[..., 0], target_batch[..., 1])
                t_for_ls = t_local
            mu_f, _ = model.encoder(x_batch, probe_ids=probe_ids_batch)
            mu_amp_real, mu_amp_imag, _ = model.solve_amplitudes_ls(y_complex_ls, mu_f, t_for_ls)
            mu_f_list.append(mu_f)
            amp_real_list.append(mu_amp_real)
            amp_imag_list.append(mu_amp_imag)

    mu_f_all = torch.cat(mu_f_list, dim=0)
    amp_real_all = torch.cat(amp_real_list, dim=0)
    amp_imag_all = torch.cat(amp_imag_list, dim=0)
    f_offset = mu_f_all - model.encoder.f_center[None, :]
    f_ratio = f_offset / model.encoder.f_band

    print("==== Harmonic Diagnostics ====")
    print("f_center(Hz):", model.encoder.f_center.detach().cpu().numpy())
    print("f_band(Hz):", float(model.encoder.f_band))
    print("mu_f_mean(Hz):", mu_f_all.mean(dim=0).detach().cpu().numpy())
    print("f_offset_mean(Hz):", f_offset.mean(dim=0).detach().cpu().numpy())
    print("f_ratio_mean:", f_ratio.mean(dim=0).detach().cpu().numpy())
    print("f_ratio_p95:", torch.quantile(f_ratio, 0.95, dim=0).detach().cpu().numpy())
    print("amp_real_mean:", amp_real_all.mean(dim=0).detach().cpu().numpy())
    print("amp_imag_mean:", amp_imag_all.mean(dim=0).detach().cpu().numpy())

    print("==== Eval Metrics ====")
    print(f"recon_btt_mse(sampled): {metrics['recon_btt_mse']:.6f}")
    print(f"recon_btt_mse(deterministic): {metrics['recon_btt_mse_det']:.6f}")
    print(f"recon_dense_mse: {metrics['recon_dense_mse']:.6f}")
    print(f"freq_mae_hz: {metrics['freq_mae_hz']:.6f}")
    print(f"complex_coeff_rel_err: {metrics['complex_coeff_rel_err']:.6f}")
    print(f"amp_mape: {metrics['amp_mape']:.6f}")
    print(f"phase_circ_mae_rad: {metrics['phase_circ_mae_rad']:.6f}")


if __name__ == "__main__":
    main()
