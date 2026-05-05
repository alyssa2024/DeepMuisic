import math
from typing import Dict, Sequence

import numpy as np
import torch

from loss import (
    compute_harmonic_elbo,
    frequency_kl_gaussian_to_maxwell,
)


def _complex_ri_mse(x_hat_complex: torch.Tensor, target_ri: torch.Tensor) -> torch.Tensor:
    x_hat_ri = torch.stack([x_hat_complex.real, x_hat_complex.imag], dim=-1)
    return torch.mean((x_hat_ri - target_ri) ** 2)


def _complex_to_complex_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(a - b) ** 2)


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


def _align_true_complex_coeff_to_local_time(
    true_complex: torch.Tensor,
    true_freq_hz: torch.Tensor,
    t0: torch.Tensor,
) -> torch.Tensor:
    if true_complex.dim() == 1:
        true_complex = true_complex.unsqueeze(0).expand(t0.shape[0], -1)
    if true_freq_hz.dim() == 1:
        true_freq_hz = true_freq_hz.unsqueeze(0).expand(t0.shape[0], -1)

    phase_shift = 2.0 * torch.pi * true_freq_hz * t0[:, None]
    phase_factor = torch.exp(1j * phase_shift)
    return true_complex * phase_factor


def _complex_coeff_rel_err(c_hat: torch.Tensor, c_true: torch.Tensor) -> torch.Tensor:
    return (
        torch.linalg.norm(c_hat - c_true, dim=-1)
        / (torch.linalg.norm(c_true, dim=-1) + 1e-8)
    ).mean()


def _circular_phase_mae(c_hat: torch.Tensor, c_true: torch.Tensor) -> torch.Tensor:
    phi_hat = torch.angle(c_hat)
    phi_true = torch.angle(c_true)
    phase_err = torch.atan2(
        torch.sin(phi_hat - phi_true),
        torch.cos(phi_hat - phi_true),
    ).abs()
    return phase_err.mean()


def _amp_mape(c_hat: torch.Tensor, c_true: torch.Tensor) -> torch.Tensor:
    amp_hat = torch.abs(c_hat)
    amp_true = torch.abs(c_true)
    return (torch.abs(amp_hat - amp_true) / (amp_true + 1e-8)).mean()


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
        "recon_btt_mse_model": 0.0,
        "recon_btt_mse_ls": 0.0,
        "total_kl": 0.0,
        "kl_w": 0.0,
        "freq_rmse_hz": 0.0,
        "freq_success_rate": 0.0,
        "amp_success_rate": 0.0,
        "joint_success_rate": 0.0,
        "detection_success_rate": 0.0,
        "complex_coeff_rel_err": 0.0,
        "amp_mape": 0.0,
        "phase_circ_mae_rad": 0.0,
        "complex_coeff_rel_err_ls_local": 0.0,
        "amp_mape_ls": 0.0,
        "phase_circ_mae_rad_ls_local": 0.0,
        "complex_coeff_rel_err_model_local": 0.0,
        "amp_mape_model": 0.0,
        "phase_circ_mae_rad_model_local": 0.0,
        "complex_coeff_rel_err_global": 0.0,
        "complex_coeff_rel_err_local": 0.0,
        "phase_circ_mae_rad_global": 0.0,
        "phase_circ_mae_rad_local": 0.0,
        "recon_dense_mse": 0.0,
        "recon_dense_mse_ls": 0.0,
        "recon_dense_mse_model": 0.0,
        "eval_amp_residual_norm": 0.0,
        "eval_amp_residual_scaled_norm": 0.0,
        "eval_amp_residual_rel": 0.0,
        "model_vs_ls_coeff_rel": 0.0,
        "model_vs_ls_recon_btt_mse": 0.0,
        "patch_freq_std_hz": 0.0,
        "harmonic_order_consistency": 0.0,
        "nan_or_inf_rate": 0.0,
    }
    num_harmonics = len(true_freqs_hz)
    for h in range(num_harmonics):
        stats[f"freq_success_h{h + 1}"] = 0.0
        stats[f"amp_success_h{h + 1}"] = 0.0

    total_samples = 0
    bad_samples = 0

    with torch.no_grad():
        for x_batch, t_batch, probe_ids, _rev_ids, target_batch in dataloader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            probe_ids = probe_ids.to(device)
            target_batch = target_batch.to(device)
            t0 = t_batch[:, 0]
            t_local = t_batch - t0[:, None]

            model_out = model(x_batch, t_local, probe_ids=probe_ids)
            if len(model_out) == 3:
                x_hat, dist_params, aux = model_out
            else:
                x_hat, dist_params = model_out
                aux = None
            mu_f, logvar_f = dist_params

            loss, recon, total_kl, _residual_loss = compute_harmonic_elbo(
                x_target=target_batch,
                x_hat=x_hat,
                dist_params=dist_params,
                beta=loss_cfg["beta"],
                prior_a_w=prior_a_w,
                use_kl_w=loss_cfg["use_kl_w"],
                aux=aux,
                residual_weight=loss_cfg.get("residual_weight", 0.0),
            )

            kl_w = frequency_kl_gaussian_to_maxwell(
                mu_w=2.0 * torch.pi * mu_f,
                logvar_w=math.log((2.0 * math.pi) ** 2) + logvar_f,
                prior_a_w=2.0 * math.pi * torch.as_tensor(prior_a_w, dtype=mu_f.dtype, device=mu_f.device),
                n_samples=1,
            ) if loss_cfg["use_kl_w"] else torch.zeros((), device=device)

            pred_freq_hz = mu_f
            freq_err = pred_freq_hz - true_freqs[None, :]
            freq_abs_err = freq_err.abs()
            freq_rmse = torch.sqrt(torch.mean(freq_err ** 2))

            y_complex = torch.complex(target_batch[..., 0], target_batch[..., 1])

            amp_real_ls, amp_imag_ls, _ = model.solve_amplitudes_ls(y_complex, mu_f, t_local)
            c_hat_ls = torch.complex(amp_real_ls, amp_imag_ls)

            if aux is not None and "amp_real" in aux and "amp_imag" in aux:
                amp_real_model = aux["amp_real"]
                amp_imag_model = aux["amp_imag"]
                f_model = aux.get("f_used", mu_f)
            else:
                amp_real_model = amp_real_ls
                amp_imag_model = amp_imag_ls
                f_model = mu_f

            c_hat_model = torch.complex(amp_real_model, amp_imag_model)
            c_hat_model_m = c_hat_model * (amp_scale + 1e-12)

            c_true_global = torch.complex(true_amp_real_norm, true_amp_imag_norm)
            c_true_global_batch = c_true_global.unsqueeze(0).expand_as(c_hat_ls)
            c_true_local_batch = _align_true_complex_coeff_to_local_time(
                true_complex=c_true_global,
                true_freq_hz=true_freqs,
                t0=t0,
            )
            c_true_local_batch_m = c_true_local_batch * (amp_scale + 1e-12)

            # Detection success criterion:
            # each harmonic must satisfy both frequency and amplitude thresholds.
            freq_tol_hz = float(loss_cfg.get("freq_success_tol_hz", 1.0))
            amp_tol_m = float(loss_cfg.get("amp_success_tol_m", 1e-4))  # 0.1 mm
            amp_abs_err_m = torch.abs(torch.abs(c_hat_model_m) - torch.abs(c_true_local_batch_m))
            freq_ok_per_harmonic = freq_abs_err <= freq_tol_hz
            amp_ok_per_harmonic = amp_abs_err_m <= amp_tol_m
            joint_ok_per_harmonic = freq_ok_per_harmonic & amp_ok_per_harmonic

            freq_success_per_patch = torch.all(freq_ok_per_harmonic, dim=1).float()
            amp_success_per_patch = torch.all(amp_ok_per_harmonic, dim=1).float()
            joint_success_per_patch = torch.all(joint_ok_per_harmonic, dim=1).float()

            freq_success_rate = freq_success_per_patch.mean()
            amp_success_rate = amp_success_per_patch.mean()
            joint_success_rate = joint_success_per_patch.mean()
            detection_success_rate = joint_success_rate

            freq_success_per_harmonic = freq_ok_per_harmonic.float().mean(dim=0)
            amp_success_per_harmonic = amp_ok_per_harmonic.float().mean(dim=0)

            complex_coeff_rel_err_global = _complex_coeff_rel_err(c_hat_ls, c_true_global_batch)
            phase_circ_mae_rad_global = _circular_phase_mae(c_hat_ls, c_true_global_batch)

            complex_coeff_rel_err_ls_local = _complex_coeff_rel_err(c_hat_ls, c_true_local_batch)
            amp_mape_ls = _amp_mape(c_hat_ls, c_true_local_batch)
            phase_circ_mae_rad_ls_local = _circular_phase_mae(c_hat_ls, c_true_local_batch)

            complex_coeff_rel_err_model_local = _complex_coeff_rel_err(c_hat_model, c_true_local_batch)
            amp_mape_model = _amp_mape(c_hat_model, c_true_local_batch)
            phase_circ_mae_rad_model_local = _circular_phase_mae(c_hat_model, c_true_local_batch)

            complex_coeff_rel_err = complex_coeff_rel_err_model_local
            amp_mape = amp_mape_model
            phase_circ_mae_rad = phase_circ_mae_rad_model_local
            complex_coeff_rel_err_local = complex_coeff_rel_err_ls_local
            phase_circ_mae_rad_local = phase_circ_mae_rad_ls_local

            if aux is not None:
                eval_amp_residual_norm = aux["amp_residual_norm"]
                eval_amp_residual_scaled_norm = aux["amp_residual_scaled_norm"]
                eval_amp_residual_rel = aux["amp_residual_rel"]
            else:
                eval_amp_residual_norm = torch.zeros((), device=device)
                eval_amp_residual_scaled_norm = torch.zeros((), device=device)
                eval_amp_residual_rel = torch.zeros((), device=device)

            model_vs_ls_coeff_rel = (
                torch.linalg.norm(c_hat_model - c_hat_ls, dim=-1)
                / (torch.linalg.norm(c_hat_ls, dim=-1) + 1e-8)
            ).mean()

            x_hat_ls = model.decode(amp_real_ls, amp_imag_ls, mu_f, t_local)
            recon_btt_mse_ls = _complex_ri_mse(x_hat_ls, target_batch)
            x_hat_model = x_hat
            recon_btt_mse_model = _complex_ri_mse(x_hat_model, target_batch)
            model_vs_ls_recon_btt_mse = _complex_to_complex_mse(x_hat_model, x_hat_ls)

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
            x_dense_true_ri = torch.stack([x_dense_true.real, x_dense_true.imag], dim=-1)
            x_dense_hat_ls = model.decode(amp_real_ls, amp_imag_ls, mu_f, t_dense_local)
            dense_mse_ls = _complex_ri_mse(x_dense_hat_ls, x_dense_true_ri)
            x_dense_hat_model = model.decode(amp_real_model, amp_imag_model, f_model, t_dense_local)
            dense_mse_model = _complex_ri_mse(x_dense_hat_model, x_dense_true_ri)
            dense_mse = dense_mse_model

            patch_freq_std_hz = pred_freq_hz.std(dim=0, unbiased=False).mean()
            harmonic_order_ok = (pred_freq_hz[:, 1:] > pred_freq_hz[:, :-1]).float().mean()

            finite_mask = torch.isfinite(mu_f).all() and torch.isfinite(loss)
            bad_samples += int(not finite_mask)

            n = batch_size
            total_samples += n
            stats["loss"] += loss.item() * n
            stats["recon_btt_mse"] += recon_btt_mse_model.item() * n
            stats["recon_btt_mse_model"] += recon_btt_mse_model.item() * n
            stats["recon_btt_mse_ls"] += recon_btt_mse_ls.item() * n
            stats["total_kl"] += total_kl.item() * n
            stats["kl_w"] += kl_w.item() * n
            stats["freq_rmse_hz"] += freq_rmse.item() * n
            stats["freq_success_rate"] += freq_success_rate.item() * n
            stats["amp_success_rate"] += amp_success_rate.item() * n
            stats["joint_success_rate"] += joint_success_rate.item() * n
            stats["detection_success_rate"] += detection_success_rate.item() * n
            for h in range(num_harmonics):
                stats[f"freq_success_h{h + 1}"] += freq_success_per_harmonic[h].item() * n
                stats[f"amp_success_h{h + 1}"] += amp_success_per_harmonic[h].item() * n
            stats["complex_coeff_rel_err"] += complex_coeff_rel_err.item() * n
            stats["complex_coeff_rel_err_ls_local"] += complex_coeff_rel_err_ls_local.item() * n
            stats["complex_coeff_rel_err_model_local"] += complex_coeff_rel_err_model_local.item() * n
            stats["complex_coeff_rel_err_global"] += complex_coeff_rel_err_global.item() * n
            stats["complex_coeff_rel_err_local"] += complex_coeff_rel_err_local.item() * n
            stats["amp_mape"] += amp_mape.item() * n
            stats["amp_mape_ls"] += amp_mape_ls.item() * n
            stats["amp_mape_model"] += amp_mape_model.item() * n
            stats["phase_circ_mae_rad"] += phase_circ_mae_rad.item() * n
            stats["phase_circ_mae_rad_ls_local"] += phase_circ_mae_rad_ls_local.item() * n
            stats["phase_circ_mae_rad_model_local"] += phase_circ_mae_rad_model_local.item() * n
            stats["phase_circ_mae_rad_global"] += phase_circ_mae_rad_global.item() * n
            stats["phase_circ_mae_rad_local"] += phase_circ_mae_rad_local.item() * n
            stats["recon_dense_mse"] += dense_mse.item() * n
            stats["recon_dense_mse_ls"] += dense_mse_ls.item() * n
            stats["recon_dense_mse_model"] += dense_mse_model.item() * n
            stats["eval_amp_residual_norm"] += eval_amp_residual_norm.item() * n
            stats["eval_amp_residual_scaled_norm"] += eval_amp_residual_scaled_norm.item() * n
            stats["eval_amp_residual_rel"] += eval_amp_residual_rel.item() * n
            stats["model_vs_ls_coeff_rel"] += model_vs_ls_coeff_rel.item() * n
            stats["model_vs_ls_recon_btt_mse"] += model_vs_ls_recon_btt_mse.item() * n
            stats["patch_freq_std_hz"] += patch_freq_std_hz.item() * n
            stats["harmonic_order_consistency"] += harmonic_order_ok.item() * n

    total_samples = max(total_samples, 1)
    for key in stats:
        if key == "nan_or_inf_rate":
            continue
        stats[key] /= total_samples

    stats["nan_or_inf_rate"] = bad_samples / total_samples
    return stats
