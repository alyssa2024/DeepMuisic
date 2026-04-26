from torch.utils.data import DataLoader
import torch
import numpy as np

from dataset import BTTPatchDataset, build_btt_point_features
from Encoder import VariationalIndependentTimeSeriesTransformer
from VAE import PhysicalHarmonicVAE
from loss import compute_harmonic_elbo
from synthesis_dataset import (
    simulate_fluctuating_speed_btt,
    generate_complex_harmonic_displacement,
)
from config import CONFIG


def build_prior_a_w(freqs_hz):
    """Build Maxwell scale prior from target frequencies."""
    true_w = 2 * np.pi * np.array(freqs_hz)
    return true_w / (2.0 * np.sqrt(2.0 / np.pi))


def main():
    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    model_cfg = CONFIG["model"]
    train_cfg = CONFIG["training"]
    loss_cfg = CONFIG["loss"]

    prior_a_w = build_prior_a_w(signal_cfg["freqs_hz"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    t_samples, freqs_per_rev, rev_ids, probe_ids, theta_samples, freqs_at_samples = (
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
        alphas=signal_cfg["amplitudes_m"],
        phis=signal_cfg["phases_rad"],
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

    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    model.train()
    for epoch in range(train_cfg["epochs"]):
        for x_batch, t_batch, probe_ids, rev_ids, target_batch in dataloader:
            x_batch = x_batch.to(device)  # [B, L, 6]
            t_batch = t_batch.to(device)  # [B, L]
            probe_ids = probe_ids.to(device)  # [B, L]
            target_batch = target_batch.to(device)  # [B, L, 2]
            t_local = t_batch - t_batch[:, :1]

            optimizer.zero_grad()

            x_hat, dist_params = model(
                x_batch,
                t_local,
                probe_ids=probe_ids,
            )

            loss, recon, kl = compute_harmonic_elbo(
                x_target=target_batch,
                x_hat=x_hat,
                dist_params=dist_params,
                beta=loss_cfg["beta"],
                prior_A_mu=loss_cfg["prior_A_mu"],
                prior_A_var=loss_cfg["prior_A_var"],
                prior_a_w=prior_a_w,
                prior_phi_mu=loss_cfg["prior_phi_mu"],
                prior_phi_kappa=loss_cfg["prior_phi_kappa"],
                use_kl_A=loss_cfg["use_kl_A"],
                use_kl_w=loss_cfg["use_kl_w"],
                use_kl_phi=loss_cfg["use_kl_phi"],
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=train_cfg["max_grad_norm"],
            )
            optimizer.step()

        with torch.no_grad():
            (mu_A, logvar_A), (mu_w, logvar_w), (mu_phi, kappa_phi) = dist_params
            print(
                f"epoch={epoch:04d} "
                f"loss={loss.item():.6f} "
                f"recon={recon.item():.6f} "
                f"kl={kl.item():.6f} | "
                f"A_mu_mean={mu_A.mean().item():.4e} "
                f"A_mu_std={mu_A.std().item():.4e} | "
                f"phi_mean={mu_phi.mean().item():.4f} "
                f"kappa_mean={kappa_phi.mean().item():.4f}"
            )
            print("w_mean per harmonic:", mu_w.mean(dim=0).detach().cpu().numpy())


if __name__ == "__main__":
    main()
