from typing import Dict

import torch

from loss import compute_sequence_posterior_recon_loss


def _complex_ri_mse(x_hat_complex: torch.Tensor, target_ri: torch.Tensor) -> torch.Tensor:
    x_hat_ri = torch.stack([x_hat_complex.real, x_hat_complex.imag], dim=-1)
    return torch.mean((x_hat_ri - target_ri) ** 2)


def _align_true_complex_coeff_to_local_time(
    true_complex: torch.Tensor,
    true_freq_hz: torch.Tensor,
    t0: torch.Tensor,
) -> torch.Tensor:
    phase_shift = 2.0 * torch.pi * true_freq_hz * t0[:, None]
    return true_complex * torch.exp(1j * phase_shift)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_cfg: Dict,
    dense_factor: int = 4,
) -> Dict[str, float]:
    del dense_factor  # Kept in the signature for runner compatibility.
    model.eval()

    stats = {
        "loss": 0.0,
        "recon_mse_mean": 0.0,
        "recon_mse_sampled": 0.0,
        "freq_prior_reg": 0.0,
        "freq_rmse_hz_mean": 0.0,
        "freq_success_rate_mean": 0.0,
        "amp_success_rate_mean": 0.0,
        "joint_success_rate_mean": 0.0,
        "freq_sample_std_mean": 0.0,
        "freq_sample_outside_rate": 0.0,
        "ls_cond_mean": 0.0,
        "ls_cond_p95": 0.0,
        "ls_amp_norm_mean": 0.0,
        "ls_amp_norm_p95": 0.0,
        "nan_or_inf_rate": 0.0,
    }

    success_cfg = loss_cfg.get("success", {})
    freq_relative_tol = float(success_cfg.get("freq_relative_tol", 0.02))
    amp_relative_tol = float(success_cfg.get("amp_relative_tol", 0.05))
    rec_cfg = loss_cfg.get("reconstruction", {})
    s_seq = int(rec_cfg.get("sequence_posterior_samples", 1))

    total_sequences = 0
    bad_batches = 0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            total_batches += 1
            x_batch = batch["x"].to(device)
            t_batch = batch["t"].to(device)
            probe_ids = batch["probe_ids"].to(device)
            target_batch = batch["target"].to(device)
            true_freq = batch["true_freq_hz"].to(device)
            true_amp = torch.complex(
                batch["true_amp_real"].to(device),
                batch["true_amp_imag"].to(device),
            )
            amp_scale = batch["amp_scale"].to(device)

            t0 = t_batch[:, :1]
            t_local = t_batch - t0

            outputs = model(x_batch, t_local, probe_ids=probe_ids)
            mu_f = outputs["mu_f"]
            std_f = outputs["std_f"]

            y_complex = torch.complex(target_batch[..., 0], target_batch[..., 1])

            amp_real_mean, amp_imag_mean, c_mean, cond_mean = model.solve_amplitudes_ls(
                y_complex=y_complex,
                f=mu_f,
                t=t_local,
                ridge_lambda=model.ls_ridge,
                return_condition=True,
            )
            x_hat_mean = model.decode(
                amp_real=amp_real_mean,
                amp_imag=amp_imag_mean,
                f=mu_f,
                t=t_local,
            )
            recon_mse_mean = _complex_ri_mse(x_hat_mean, target_batch)

            recon_mse_sampled, sampled_diag = compute_sequence_posterior_recon_loss(
                y_complex=y_complex,
                t=t_local,
                mu_f=mu_f,
                std_f=std_f,
                model=model,
                sequence_posterior_samples=s_seq,
                ridge_lambda=model.ls_ridge,
            )

            lower = model.encoder.freq_lower.to(device=mu_f.device, dtype=mu_f.dtype)
            upper = model.encoder.freq_upper.to(device=mu_f.device, dtype=mu_f.dtype)
            f_samples = sampled_diag["f_samples"]
            outside = (f_samples < lower.view(1, 1, -1)) | (
                f_samples > upper.view(1, 1, -1)
            )
            freq_sample_outside_rate = outside.float().mean()

            freq_err = mu_f - true_freq
            freq_rmse = torch.sqrt(torch.mean(freq_err.pow(2)))
            freq_rel_err = torch.abs(freq_err) / (torch.abs(true_freq) + 1e-12)

            c_pred_m = torch.complex(amp_real_mean, amp_imag_mean) * amp_scale[:, None]
            c_true_local = _align_true_complex_coeff_to_local_time(
                true_complex=true_amp,
                true_freq_hz=true_freq,
                t0=t0.squeeze(1),
            )
            amp_rel_err = torch.abs(c_pred_m - c_true_local) / (
                torch.abs(c_true_local) + 1e-12
            )

            freq_ok = freq_rel_err <= freq_relative_tol
            amp_ok = amp_rel_err <= amp_relative_tol
            joint_ok = freq_ok & amp_ok

            freq_success_rate = torch.all(freq_ok, dim=1).float().mean()
            amp_success_rate = torch.all(amp_ok, dim=1).float().mean()
            joint_success_rate = torch.all(joint_ok, dim=1).float().mean()

            beta_freq = float(loss_cfg.get("beta_freq", 1e-5))
            freq_prior_reg = torch.zeros((), dtype=mu_f.dtype, device=mu_f.device)
            below = torch.relu(lower.view(1, -1) - mu_f)
            above = torch.relu(mu_f - upper.view(1, -1))
            width = upper.view(1, -1) - lower.view(1, -1)
            freq_prior_reg = ((below + above) / (width + 1e-12)).pow(2).mean()
            loss = recon_mse_sampled + beta_freq * freq_prior_reg

            finite = (
                torch.isfinite(loss)
                and torch.isfinite(mu_f).all()
                and torch.isfinite(recon_mse_mean)
            )
            if not finite:
                bad_batches += 1

            n = x_batch.shape[0]
            total_sequences += n
            stats["loss"] += loss.item() * n
            stats["recon_mse_mean"] += recon_mse_mean.item() * n
            stats["recon_mse_sampled"] += recon_mse_sampled.item() * n
            stats["freq_prior_reg"] += freq_prior_reg.item() * n
            stats["freq_rmse_hz_mean"] += freq_rmse.item() * n
            stats["freq_success_rate_mean"] += freq_success_rate.item() * n
            stats["amp_success_rate_mean"] += amp_success_rate.item() * n
            stats["joint_success_rate_mean"] += joint_success_rate.item() * n
            stats["freq_sample_std_mean"] += sampled_diag["freq_sample_std_mean"].item() * n
            stats["freq_sample_outside_rate"] += freq_sample_outside_rate.item() * n
            stats["ls_cond_mean"] += sampled_diag["ls_cond_mean"].item() * n
            stats["ls_cond_p95"] += sampled_diag["ls_cond_p95"].item() * n
            stats["ls_amp_norm_mean"] += sampled_diag["ls_amp_norm_mean"].item() * n
            stats["ls_amp_norm_p95"] += sampled_diag["ls_amp_norm_p95"].item() * n

    total_sequences = max(total_sequences, 1)
    for key in stats:
        if key == "nan_or_inf_rate":
            continue
        stats[key] /= total_sequences
    stats["nan_or_inf_rate"] = bad_batches / max(total_batches, 1)
    return stats
