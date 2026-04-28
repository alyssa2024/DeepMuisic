import math
from typing import Dict, Sequence

import numpy as np
import torch

from loss import (
    compute_harmonic_elbo,
    frequency_kl_gaussian_to_maxwell,
    gaussian_kl,
)


def _complex_ri_mse(x_hat_complex: torch.Tensor, target_ri: torch.Tensor) -> torch.Tensor:
    x_hat_ri = torch.stack([x_hat_complex.real, x_hat_complex.imag], dim=-1)
    return torch.mean((x_hat_ri - target_ri) ** 2)


def _synthesize_complex_batch(
    t_abs: torch.Tensor,
    freqs_hz: torch.Tensor,
    amp_real: torch.Tensor,
    amp_imag: torch.Tensor,
) -> torch.Tensor:
    complex_amp = torch.complex(amp_real, amp_imag)
    phase = 2.0 * torch.pi * freqs_hz[None, None, :] * t_abs[:, :, None]
    unit_complex = torch.polar(torch.ones_like(phase), phase)
    return (complex_amp[None, None, :] * unit_complex).sum(dim=-1)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    true_freqs_hz: Sequence[float],
    true_amp_real: Sequence[float],
    true_amp_imag: Sequence[float],
    amp_scale: float,
    prior_a_w: np.ndarray,
    loss_cfg: Dict,
    dense_factor: int = 4,
) -> Dict[str, float]:
    model.eval()

    true_freqs = torch.tensor(true_freqs_hz, dtype=torch.float32, device=device)
    true_amp_real_norm = torch.tensor(true_amp_real, dtype=torch.float32, device=device) / (amp_scale + 1e-12)
    true_amp_imag_norm = torch.tensor(true_amp_imag, dtype=torch.float32, device=device) / (amp_scale + 1e-12)

    stats = {
        "loss": 0.0,
        "recon_btt_mse": 0.0,
        "recon_btt_mse_det": 0.0,
        "total_kl": 0.0,
        "kl_amp_real": 0.0,
        "kl_amp_imag": 0.0,
        "kl_w": 0.0,
        "freq_mae_hz": 0.0,
        "amp_real_mape": 0.0,
        "amp_imag_mape": 0.0,
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
            (mu_amp_real, logvar_amp_real), (mu_amp_imag, logvar_amp_imag), (mu_f, logvar_f) = dist_params

            loss, recon, total_kl = compute_harmonic_elbo(
                x_target=target_batch,
                x_hat=x_hat,
                dist_params=dist_params,
                beta=loss_cfg["beta"],
                prior_amp_real_mu=loss_cfg["prior_amp_real_mu"],
                prior_amp_real_var=loss_cfg["prior_amp_real_var"],
                prior_amp_imag_mu=loss_cfg["prior_amp_imag_mu"],
                prior_amp_imag_var=loss_cfg["prior_amp_imag_var"],
                prior_a_w=prior_a_w,
                use_kl_amp_real=loss_cfg["use_kl_amp_real"],
                use_kl_amp_imag=loss_cfg["use_kl_amp_imag"],
                use_kl_w=loss_cfg["use_kl_w"],
            )

            kl_amp_real = gaussian_kl(
                mu_amp_real,
                logvar_amp_real,
                prior_mu=loss_cfg["prior_amp_real_mu"],
                prior_var=loss_cfg["prior_amp_real_var"],
            ) if loss_cfg["use_kl_amp_real"] else torch.zeros((), device=device)

            kl_amp_imag = gaussian_kl(
                mu_amp_imag,
                logvar_amp_imag,
                prior_mu=loss_cfg["prior_amp_imag_mu"],
                prior_var=loss_cfg["prior_amp_imag_var"],
            ) if loss_cfg["use_kl_amp_imag"] else torch.zeros((), device=device)

            kl_w = frequency_kl_gaussian_to_maxwell(
                mu_w=2.0 * torch.pi * mu_f,
                logvar_w=math.log((2.0 * math.pi) ** 2) + logvar_f,
                prior_a_w=2.0 * math.pi * torch.as_tensor(prior_a_w, dtype=mu_f.dtype, device=mu_f.device),
                n_samples=1,
            ) if loss_cfg["use_kl_w"] else torch.zeros((), device=device)

            pred_freq_hz = mu_f
            freq_mae = (pred_freq_hz - true_freqs[None, :]).abs().mean()

            amp_real_rel_err = (mu_amp_real - true_amp_real_norm[None, :]).abs() / (true_amp_real_norm[None, :].abs() + 1e-8)
            amp_imag_rel_err = (mu_amp_imag - true_amp_imag_norm[None, :]).abs() / (true_amp_imag_norm[None, :].abs() + 1e-8)
            amp_real_mape = amp_real_rel_err.mean()
            amp_imag_mape = amp_imag_rel_err.mean()
            x_hat_det = model.decode(mu_amp_real, mu_amp_imag, mu_f, t_local)
            recon_det = _complex_ri_mse(x_hat_det, target_batch)

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
                amp_real=true_amp_real_norm,
                amp_imag=true_amp_imag_norm,
            )
            x_dense_hat = model.decode(mu_amp_real, mu_amp_imag, mu_f, t_dense_local)
            x_dense_true_ri = torch.stack([x_dense_true.real, x_dense_true.imag], dim=-1)
            dense_mse = _complex_ri_mse(x_dense_hat, x_dense_true_ri)

            patch_freq_std_hz = pred_freq_hz.std(dim=0, unbiased=False).mean()
            harmonic_order_ok = (pred_freq_hz[:, 1:] > pred_freq_hz[:, :-1]).float().mean()

            finite_mask = (
                torch.isfinite(mu_amp_real).all()
                and torch.isfinite(mu_amp_imag).all()
                and torch.isfinite(mu_f).all()
                and torch.isfinite(loss)
            )
            bad_samples += int(not finite_mask)

            n = batch_size
            total_samples += n
            stats["loss"] += loss.item() * n
            stats["recon_btt_mse"] += recon.item() * n
            stats["recon_btt_mse_det"] += recon_det.item() * n
            stats["total_kl"] += total_kl.item() * n
            stats["kl_amp_real"] += kl_amp_real.item() * n
            stats["kl_amp_imag"] += kl_amp_imag.item() * n
            stats["kl_w"] += kl_w.item() * n
            stats["freq_mae_hz"] += freq_mae.item() * n
            stats["amp_real_mape"] += amp_real_mape.item() * n
            stats["amp_imag_mape"] += amp_imag_mape.item() * n
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
