import math
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from loss import (
    compute_harmonic_elbo,
    frequency_kl_gaussian_to_maxwell,
    gaussian_kl,
    von_mises_kl,
)


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _complex_ri_mse(x_hat_complex: torch.Tensor, target_ri: torch.Tensor) -> torch.Tensor:
    x_hat_ri = torch.stack([x_hat_complex.real, x_hat_complex.imag], dim=-1)
    return torch.mean((x_hat_ri - target_ri) ** 2)


def _synthesize_complex_batch(
    t_abs: torch.Tensor,
    freqs_hz: torch.Tensor,
    amps: torch.Tensor,
    phases: torch.Tensor,
) -> torch.Tensor:
    omega = 2.0 * torch.pi * freqs_hz
    phase = omega[None, None, :] * t_abs[:, :, None] + phases[None, None, :]
    real = amps[None, None, :] * torch.cos(phase)
    imag = amps[None, None, :] * torch.sin(phase)
    return torch.complex(real.sum(dim=-1), imag.sum(dim=-1))


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    true_freqs_hz: Sequence[float],
    true_amps: Sequence[float],
    true_phases_rad: Sequence[float],
    amp_scale: float,
    prior_a_w: np.ndarray,
    loss_cfg: Dict,
    dense_factor: int = 4,
) -> Dict[str, float]:
    model.eval()

    true_freqs = torch.tensor(true_freqs_hz, dtype=torch.float32, device=device)
    true_amps_norm = torch.tensor(true_amps, dtype=torch.float32, device=device) / (amp_scale + 1e-12)
    true_phases = torch.tensor(true_phases_rad, dtype=torch.float32, device=device)
    true_omega = 2.0 * torch.pi * true_freqs

    stats = {
        "loss": 0.0,
        "recon_btt_mse": 0.0,
        "recon_btt_mse_det": 0.0,
        "total_kl": 0.0,
        "kl_A": 0.0,
        "kl_w": 0.0,
        "kl_phi": 0.0,
        "freq_mae_hz": 0.0,
        "amp_mape": 0.0,
        "phase_circ_mae_rad": 0.0,
        "recon_dense_mse": 0.0,
        "patch_freq_std_hz": 0.0,
        "harmonic_order_consistency": 0.0,
        "nan_or_inf_rate": 0.0,
    }

    total_samples = 0
    bad_samples = 0

    with torch.no_grad():
        for x_batch, t_batch, probe_ids, _rev_ids, target_batch in dataloader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            probe_ids = probe_ids.to(device)
            target_batch = target_batch.to(device)
            t_local = t_batch - t_batch[:, :1]

            x_hat, dist_params = model(x_batch, t_local, probe_ids=probe_ids)
            (mu_A, logvar_A), (mu_w, logvar_w), (mu_phi, kappa_phi) = dist_params

            loss, recon, total_kl = compute_harmonic_elbo(
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

            kl_A = gaussian_kl(
                mu_A,
                logvar_A,
                prior_mu=loss_cfg["prior_A_mu"],
                prior_var=loss_cfg["prior_A_var"],
            ) if loss_cfg["use_kl_A"] else torch.zeros((), device=device)

            kl_w = frequency_kl_gaussian_to_maxwell(
                mu_w=mu_w,
                logvar_w=logvar_w,
                prior_a_w=prior_a_w,
                n_samples=1,
            ) if loss_cfg["use_kl_w"] else torch.zeros((), device=device)

            kl_phi = von_mises_kl(
                mu_q=mu_phi,
                kappa_q=kappa_phi,
                mu_p=loss_cfg["prior_phi_mu"],
                kappa_p=loss_cfg["prior_phi_kappa"],
            ) if loss_cfg["use_kl_phi"] else torch.zeros((), device=device)

            pred_freq_hz = mu_w / (2.0 * torch.pi)
            freq_mae = (pred_freq_hz - true_freqs[None, :]).abs().mean()

            # Keep evaluation amplitude definition consistent with decoder-side non-negative mapping.
            amp_eval = F.softplus(mu_A) + 1e-8
            amp_rel_err = (amp_eval - true_amps_norm[None, :]).abs() / (true_amps_norm[None, :].abs() + 1e-8)
            amp_mape = amp_rel_err.mean()
            x_hat_det = model.decode(amp_eval, mu_w, mu_phi, t_local)
            recon_det = _complex_ri_mse(x_hat_det, target_batch)

            t0 = t_batch[:, :1]
            phi_true_local = _wrap_to_pi(true_phases[None, :] + true_omega[None, :] * t0)
            phase_err = _wrap_to_pi(mu_phi - phi_true_local)
            phase_circ_mae = phase_err.abs().mean()

            batch_size, seq_len = t_batch.shape
            dense_len = max(seq_len * dense_factor, seq_len)
            dense_grid = torch.linspace(0.0, 1.0, dense_len, device=device)[None, :].repeat(batch_size, 1)
            t_start = t_batch[:, :1]
            t_end = t_batch[:, -1:]
            t_dense_abs = t_start + (t_end - t_start) * dense_grid
            t_dense_local = t_dense_abs - t_start

            x_dense_true = _synthesize_complex_batch(
                t_abs=t_dense_abs,
                freqs_hz=true_freqs,
                amps=true_amps_norm,
                phases=true_phases,
            )
            x_dense_hat = model.decode(amp_eval, mu_w, mu_phi, t_dense_local)
            x_dense_true_ri = torch.stack([x_dense_true.real, x_dense_true.imag], dim=-1)
            dense_mse = _complex_ri_mse(x_dense_hat, x_dense_true_ri)

            patch_freq_std_hz = pred_freq_hz.std(dim=0, unbiased=False).mean()
            harmonic_order_ok = (pred_freq_hz[:, 1:] > pred_freq_hz[:, :-1]).float().mean()

            finite_mask = torch.isfinite(mu_A).all() and torch.isfinite(mu_w).all() and torch.isfinite(mu_phi).all() and torch.isfinite(loss)
            bad_samples += int(not finite_mask)

            n = batch_size
            total_samples += n
            stats["loss"] += loss.item() * n
            stats["recon_btt_mse"] += recon.item() * n
            stats["recon_btt_mse_det"] += recon_det.item() * n
            stats["total_kl"] += total_kl.item() * n
            stats["kl_A"] += kl_A.item() * n
            stats["kl_w"] += kl_w.item() * n
            stats["kl_phi"] += kl_phi.item() * n
            stats["freq_mae_hz"] += freq_mae.item() * n
            stats["amp_mape"] += amp_mape.item() * n
            stats["phase_circ_mae_rad"] += phase_circ_mae.item() * n
            stats["recon_dense_mse"] += dense_mse.item() * n
            stats["patch_freq_std_hz"] += patch_freq_std_hz.item() * n
            stats["harmonic_order_consistency"] += harmonic_order_ok.item() * n

    total_samples = max(total_samples, 1)
    for key in stats:
        if key == "nan_or_inf_rate":
            continue
        stats[key] /= total_samples

    stats["nan_or_inf_rate"] = bad_samples / total_samples
    return stats
