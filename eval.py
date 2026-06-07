from typing import Dict

import torch

from loss import (
    compute_group_profile_recon_loss,
    compute_harmonic_loss,
    compute_sequence_posterior_recon_loss,
    kl_trunc_normal_trunc_normal,
    kl_trunc_normal_uniform,
    resolve_profile_ls_ridge,
)


def _complex_ri_mse(x_hat_complex: torch.Tensor, target_ri: torch.Tensor) -> torch.Tensor:
    target_complex = torch.complex(target_ri[..., 0], target_ri[..., 1])
    return torch.mean(torch.abs(x_hat_complex - target_complex) ** 2)


def _align_true_complex_coeff_to_local_time(
    true_complex: torch.Tensor,
    true_freq_hz: torch.Tensor,
    t0: torch.Tensor,
) -> torch.Tensor:
    phase_shift = 2.0 * torch.pi * true_freq_hz * t0[:, None]
    return true_complex * torch.exp(1j * phase_shift)


def _circular_abs_phase_error(
    pred_complex: torch.Tensor,
    true_complex: torch.Tensor,
) -> torch.Tensor:
    phase_delta = torch.angle(pred_complex) - torch.angle(true_complex)
    return torch.abs(torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta)))


def _add_per_harmonic_metrics(
    stats: Dict[str, float],
    prefix: str,
    values: torch.Tensor,
):
    for k, value in enumerate(values.detach().cpu().tolist(), start=1):
        stats[f"{prefix}_h{k}"] = float(value)


def _compute_freq_kl(mu_f, std_f, model, loss_cfg):
    kl_cfg = loss_cfg.get("kl", {})
    kl_type = kl_cfg.get("type", "trunc_normal_to_trunc_normal")
    if kl_type == "trunc_normal_to_uniform":
        return kl_trunc_normal_uniform(
            mu_f=mu_f,
            std_f=std_f,
            freq_lower=model.encoder.freq_lower,
            freq_upper=model.encoder.freq_upper,
        )
    if kl_type == "trunc_normal_to_trunc_normal":
        prior_cfg = loss_cfg.get("prior", loss_cfg.get("loss_prior", {}))
        prior_std_ratio = float(prior_cfg.get("std_ratio_to_half_band", 0.5))
        return kl_trunc_normal_trunc_normal(
            mu_q=mu_f,
            std_q=std_f,
            mu_p=model.encoder.freq_mid,
            std_p=prior_std_ratio * model.encoder.freq_half,
            lower=model.encoder.freq_lower,
            upper=model.encoder.freq_upper,
        )
    raise ValueError(f"Unsupported loss.kl.type={kl_type!r}")


def _to_single_instance_batch(split, device):
    return {
        key: value.to(device)
        for key, value in split.items()
        if torch.is_tensor(value)
    }


def evaluate_single_instance_splits(
    model: torch.nn.Module,
    sample: Dict,
    device: torch.device,
    loss_cfg: Dict,
) -> Dict[str, float]:
    model.eval()
    train = _to_single_instance_batch(sample["train"], device)
    true_freq = sample["true_freq_hz"].to(device)
    true_amp = torch.complex(
        sample["true_amp_real"].to(device),
        sample["true_amp_imag"].to(device),
    )
    total_train_patches = int(sample["total_train_patches"].item())
    x_train = train["x"].unsqueeze(0)
    t_train = train["t_local"].unsqueeze(0)
    probe_train = train["probe_ids"].unsqueeze(0)
    target_train = train["target"].unsqueeze(0)
    noise_train = train["noise_var_norm"].unsqueeze(0)
    amp_scale_train = train["amp_scale"].unsqueeze(0)

    with torch.no_grad():
        outputs = model(x_train, t_train, probe_ids=probe_train)
        train_loss, _train_recon, _train_kl, train_diag = compute_harmonic_loss(
            x_target=target_train,
            model_outputs=outputs,
            model=model,
            t=t_train,
            loss_cfg=loss_cfg,
            noise_var_norm=noise_train,
            amp_scale=amp_scale_train,
            total_num_patches=total_train_patches,
        )

        metrics = {
            "loss": float(train_loss.item()),
            "train_loss": float(train_loss.item()),
        }
        for prefix, split_name in (
            ("train", "train"),
            ("val", "val"),
            ("test", "test"),
        ):
            split = train if split_name == "train" else _to_single_instance_batch(
                sample[split_name],
                device,
            )
            target = split["target"].unsqueeze(0)
            t_local = split["t_local"].unsqueeze(0)
            noise_var = split["noise_var_norm"].unsqueeze(0)
            amp_scale = split["amp_scale"].unsqueeze(0)
            split_loss, _split_recon, _split_kl, split_diag = compute_harmonic_loss(
                x_target=target,
                model_outputs=outputs,
                model=model,
                t=t_local,
                loss_cfg=loss_cfg,
                noise_var_norm=noise_var,
                amp_scale=amp_scale,
                total_num_patches=target.shape[1],
            )
            metrics[f"{prefix}_loss"] = float(split_loss.item())
            metrics[f"{prefix}_data_nll_core"] = float(
                split_diag["data_nll_core"].item()
            )
            metrics[f"{prefix}_amp_map_prior_core"] = float(
                split_diag["amp_map_prior_core"].item()
            )
            metrics[f"{prefix}_map_profile_core"] = float(
                split_diag["map_profile_core"].item()
            )
            metrics[f"{prefix}_recon_mse_sampled"] = float(
                split_diag["recon_mse_sampled"].item()
            )
            metrics[f"{prefix}_recon_btt_mse"] = metrics[
                f"{prefix}_recon_mse_sampled"
            ]
            metrics[f"{prefix}_map_lambda_mean"] = float(
                split_diag.get("map_lambda_mean", torch.tensor(0.0)).item()
            )

        lower = model.encoder.freq_lower.to(device=device, dtype=outputs["mu_f"].dtype)
        upper = model.encoder.freq_upper.to(device=device, dtype=outputs["mu_f"].dtype)
        freq_half = (upper - lower) / 2.0
        mu_f = outputs["mu_f"].squeeze(0)
        std_f = outputs["std_f"].squeeze(0)
        freq_err = mu_f - true_freq
        freq_norm_err = freq_err / (freq_half + 1e-12)
        metrics.update(
            {
                "freq_rmse_hz_mean": float(torch.sqrt(freq_err.pow(2).mean()).item()),
                "freq_nrmse_band_mean": float(
                    torch.sqrt(freq_norm_err.pow(2).mean()).item()
                ),
                "posterior_std_hz_mean": float(std_f.mean().item()),
                "posterior_std_rel_mean": float((std_f / (freq_half + 1e-12)).mean().item()),
                "freq_kl": float(train_diag["freq_kl"].item()),
                "freq_kl_raw": float(train_diag["freq_kl_raw"].item()),
                "recon_mse_mean": metrics["val_recon_btt_mse"],
                "recon_mse_sampled": metrics["val_recon_btt_mse"],
                "recon_btt_mse": metrics["val_recon_btt_mse"],
                "recon_nll_sampled": metrics["val_data_nll_core"],
                "recon_nll_full": float(train_diag["recon_nll_full"].item()),
            }
        )
        for k, value in enumerate(mu_f.detach().cpu().tolist(), start=1):
            metrics[f"global_freq_mu_h{k}_hz"] = float(value)
        for k, value in enumerate(std_f.detach().cpu().tolist(), start=1):
            metrics[f"global_freq_std_h{k}_hz"] = float(value)
        coverage68 = (torch.abs(freq_err) <= std_f).float().mean()
        coverage95 = (torch.abs(freq_err) <= 1.96 * std_f).float().mean()
        metrics["global_freq_coverage_68"] = float(coverage68.item())
        metrics["global_freq_coverage_95"] = float(coverage95.item())
        if "fusion_max_weight" in outputs:
            max_weight = outputs["fusion_max_weight"].squeeze(0)
            eff = outputs["fusion_effective_num_patches"].squeeze(0)
            metrics["fusion_max_weight_mean"] = float(max_weight.mean().item())
            metrics["fusion_effective_num_patches_mean"] = float(eff.mean().item())
            for k, value in enumerate(max_weight.detach().cpu().tolist(), start=1):
                metrics[f"fusion_max_weight_h{k}"] = float(value)
            for k, value in enumerate(eff.detach().cpu().tolist(), start=1):
                metrics[f"fusion_effective_num_patches_h{k}"] = float(value)
        metrics["true_amp_norm_mean"] = float(torch.abs(true_amp).mean().item())
    return metrics


def _evaluate_group_profile_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_cfg: Dict,
) -> Dict[str, float]:
    stats = {
        "loss": 0.0,
        "recon_mse_mean": 0.0,
        "recon_mse_sampled": 0.0,
        "recon_nll_sampled": 0.0,
        "recon_nll_full": 0.0,
    }
    success_cfg = loss_cfg.get("success", {})
    freq_relative_tol = float(success_cfg.get("freq_relative_tol", 0.02))
    amp_relative_tol = float(success_cfg.get("amp_relative_tol", 0.05))
    complex_coeff_relative_tol = float(
        success_cfg.get("complex_coeff_relative_tol", amp_relative_tol)
    )
    rec_cfg = loss_cfg.get("reconstruction", {})
    s_seq = int(rec_cfg.get("sequence_posterior_samples", 1))
    include_log_const = bool(rec_cfg.get("include_log_const", False))

    total_groups = 0
    total_patches = 0
    total_freq_elements = 0
    total_amp_elements = 0
    total_order_pairs = 0
    num_harmonics = None

    freq_sqerr_sum = None
    freq_abs_err_sum = None
    freq_nsqerr_sum = None
    freq_success_sum = None
    center_sqerr_sum = None
    amp_mape_sum = None
    amp_abs_err_sum = None
    amp_success_sum = None
    complex_rel_err_sum = None
    complex_success_sum = None
    phase_circ_err_sum = None

    freq_sequence_success_sum = 0.0
    amp_sequence_success_sum = 0.0
    joint_amp_freq_sequence_success_sum = 0.0
    complex_sequence_success_sum = 0.0
    complex_vector_rel_err_sum = 0.0
    harmonic_order_success_sum = 0.0
    posterior_std_sum = 0.0
    posterior_std_rel_sum = 0.0
    posterior_std_local_sum = 0.0
    freq_sample_outside_sum = 0.0
    freq_prior_reg_sum = 0.0
    freq_kl_sum = 0.0
    freq_kl_raw_sum = 0.0
    ls_cond_sum = 0.0
    ls_amp_norm_sum = 0.0

    posterior_std_values = []
    ls_cond_values = []
    ls_amp_norm_values = []

    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch["x"].to(device)
            t_batch = batch["t"].to(device)
            probe_ids = batch["probe_ids"].to(device)
            target_batch = batch["target"].to(device)
            noise_var_norm = batch["noise_var_norm"].to(device)
            true_freq = batch["true_freq_hz"].to(device)
            true_amp = torch.complex(
                batch["true_amp_real"].to(device),
                batch["true_amp_imag"].to(device),
            )
            amp_scale = batch["amp_scale"].to(device)

            group_size, patches_per_group, seq_len, _ = target_batch.shape
            t0 = t_batch[:, :, :1]
            t_local = t_batch - t0

            outputs = model(x_batch, t_local, probe_ids=probe_ids)
            mu_f = outputs["mu_f"]
            std_f = outputs["std_f"]
            y_complex = torch.complex(target_batch[..., 0], target_batch[..., 1])
            k_count = mu_f.shape[1]
            ridge_lambda = resolve_profile_ls_ridge(
                loss_cfg=loss_cfg,
                fallback_ridge=model.ls_ridge,
                noise_var_norm=noise_var_norm,
            )
            if num_harmonics is None:
                num_harmonics = k_count
                freq_sqerr_sum = torch.zeros(k_count, dtype=torch.float64)
                freq_abs_err_sum = torch.zeros(k_count, dtype=torch.float64)
                freq_nsqerr_sum = torch.zeros(k_count, dtype=torch.float64)
                freq_success_sum = torch.zeros(k_count, dtype=torch.float64)
                center_sqerr_sum = torch.zeros(k_count, dtype=torch.float64)
                amp_mape_sum = torch.zeros(k_count, dtype=torch.float64)
                amp_abs_err_sum = torch.zeros(k_count, dtype=torch.float64)
                amp_success_sum = torch.zeros(k_count, dtype=torch.float64)
                complex_rel_err_sum = torch.zeros(k_count, dtype=torch.float64)
                complex_success_sum = torch.zeros(k_count, dtype=torch.float64)
                phase_circ_err_sum = torch.zeros(k_count, dtype=torch.float64)

            f_flat = (
                mu_f[:, None, :]
                .expand(group_size, patches_per_group, k_count)
                .reshape(group_size * patches_per_group, k_count)
            )
            y_flat = y_complex.reshape(group_size * patches_per_group, seq_len)
            t_flat = t_local.reshape(group_size * patches_per_group, seq_len)
            amp_real_mean, amp_imag_mean, c_mean, cond_mean = model.solve_amplitudes_ls(
                y_complex=y_flat,
                f=f_flat,
                t=t_flat,
                ridge_lambda=(
                    ridge_lambda.reshape(group_size * patches_per_group)
                    if torch.is_tensor(ridge_lambda)
                    else ridge_lambda
                ),
                return_condition=True,
            )
            x_hat_mean = model.decode(
                amp_real=amp_real_mean,
                amp_imag=amp_imag_mean,
                f=f_flat,
                t=t_flat,
            ).reshape(group_size, patches_per_group, seq_len)
            recon_mse_mean = _complex_ri_mse(x_hat_mean, target_batch)

            recon_nll_sampled, sampled_diag = compute_group_profile_recon_loss(
                y_complex=y_complex,
                t=t_local,
                mu_f=mu_f,
                std_f=std_f,
                model=model,
                sequence_posterior_samples=s_seq,
                ridge_lambda=ridge_lambda,
                noise_var_norm=noise_var_norm,
                include_log_const=include_log_const,
            )
            recon_mse_sampled = sampled_diag["recon_mse_sampled"]
            recon_nll_full = sampled_diag["recon_nll_full"]

            lower = model.encoder.freq_lower.to(device=mu_f.device, dtype=mu_f.dtype)
            upper = model.encoder.freq_upper.to(device=mu_f.device, dtype=mu_f.dtype)
            freq_half = (upper - lower) / 2.0
            freq_center = (upper + lower) / 2.0
            freq_kl_per_item = _compute_freq_kl(mu_f, std_f, model, loss_cfg)
            freq_kl_raw = freq_kl_per_item.sum(dim=-1).mean()
            freq_kl = freq_kl_raw
            f_samples = sampled_diag["f_samples"]
            outside = (f_samples < lower.view(1, 1, -1)) | (
                f_samples > upper.view(1, 1, -1)
            )
            freq_sample_outside_rate = outside.float().mean()

            freq_err = mu_f - true_freq
            freq_abs_err = torch.abs(freq_err)
            freq_norm_err = freq_err / (freq_half.view(1, -1) + 1e-12)
            freq_rel_err = torch.abs(freq_err) / (torch.abs(true_freq) + 1e-12)

            c_pred_m = c_mean.reshape(group_size, patches_per_group, k_count)
            c_pred_m = c_pred_m * amp_scale[:, :, None]
            true_freq_patch = (
                true_freq[:, None, :]
                .expand(group_size, patches_per_group, k_count)
                .reshape(group_size * patches_per_group, k_count)
            )
            c_true_local = _align_true_complex_coeff_to_local_time(
                true_complex=true_amp.reshape(group_size * patches_per_group, k_count),
                true_freq_hz=true_freq_patch,
                t0=t0.squeeze(-1).reshape(group_size * patches_per_group),
            ).reshape(group_size, patches_per_group, k_count)

            amp_hat = torch.abs(c_pred_m)
            amp_true = torch.abs(c_true_local)
            amp_abs_err = torch.abs(amp_hat - amp_true)
            amp_mape = amp_abs_err / (amp_true + 1e-12)
            complex_rel_err = torch.abs(c_pred_m - c_true_local) / (
                amp_true + 1e-12
            )
            complex_vector_rel_err = torch.linalg.norm(
                c_pred_m - c_true_local,
                dim=-1,
            ) / (torch.linalg.norm(c_true_local, dim=-1) + 1e-12)
            phase_circ_err = _circular_abs_phase_error(c_pred_m, c_true_local)
            c_norm_mean = torch.linalg.norm(c_pred_m, dim=-1)

            freq_ok = freq_rel_err <= freq_relative_tol
            amp_ok = amp_mape <= amp_relative_tol
            complex_ok = complex_rel_err <= complex_coeff_relative_tol
            joint_amp_freq_ok = freq_ok[:, None, :] & amp_ok

            freq_success_rate = torch.all(freq_ok, dim=1).float().mean()
            amp_success_rate = torch.all(amp_ok.reshape(group_size, -1), dim=1).float().mean()
            joint_amp_freq_success_rate = torch.all(
                joint_amp_freq_ok.reshape(group_size, -1),
                dim=1,
            ).float().mean()
            complex_success_rate = torch.all(
                complex_ok.reshape(group_size, -1),
                dim=1,
            ).float().mean()

            if k_count > 1:
                order_ok = mu_f[:, 1:] > mu_f[:, :-1]
                harmonic_order_success_sum += order_ok.float().sum().item()
                total_order_pairs += order_ok.numel()

            beta_freq = float(loss_cfg.get("beta_freq", 1.0))
            loss = recon_nll_sampled + beta_freq * freq_kl

            total_groups += group_size
            total_patches += group_size * patches_per_group
            total_freq_elements += group_size * k_count
            total_amp_elements += group_size * patches_per_group * k_count
            stats["loss"] += loss.item() * group_size
            stats["recon_mse_mean"] += recon_mse_mean.item() * group_size
            stats["recon_mse_sampled"] += recon_mse_sampled.item() * group_size
            stats["recon_nll_sampled"] += recon_nll_sampled.item() * group_size
            stats["recon_nll_full"] += recon_nll_full.item() * group_size

            freq_sqerr_sum += freq_err.pow(2).sum(dim=0).double().cpu()
            freq_abs_err_sum += freq_abs_err.sum(dim=0).double().cpu()
            freq_nsqerr_sum += freq_norm_err.pow(2).sum(dim=0).double().cpu()
            freq_success_sum += freq_ok.float().sum(dim=0).double().cpu()
            center_sqerr_sum += (
                (freq_center.view(1, -1) - true_freq).pow(2).sum(dim=0).double().cpu()
            )

            amp_mape_sum += amp_mape.sum(dim=(0, 1)).double().cpu()
            amp_abs_err_sum += amp_abs_err.sum(dim=(0, 1)).double().cpu()
            amp_success_sum += amp_ok.float().sum(dim=(0, 1)).double().cpu()
            complex_rel_err_sum += complex_rel_err.sum(dim=(0, 1)).double().cpu()
            complex_success_sum += complex_ok.float().sum(dim=(0, 1)).double().cpu()
            phase_circ_err_sum += phase_circ_err.sum(dim=(0, 1)).double().cpu()

            freq_sequence_success_sum += freq_success_rate.item() * group_size
            amp_sequence_success_sum += amp_success_rate.item() * group_size
            joint_amp_freq_sequence_success_sum += (
                joint_amp_freq_success_rate.item() * group_size
            )
            complex_sequence_success_sum += complex_success_rate.item() * group_size
            complex_vector_rel_err_sum += complex_vector_rel_err.sum().item()

            posterior_std_sum += std_f.sum().item()
            posterior_std_rel_sum += (std_f / (freq_half.view(1, -1) + 1e-12)).sum().item()
            if "std_f_local" in outputs:
                posterior_std_local_sum += outputs["std_f_local"].sum().item()
            freq_sample_outside_sum += freq_sample_outside_rate.item() * group_size
            freq_prior_reg_sum += freq_kl.item() * group_size
            freq_kl_sum += freq_kl.item() * group_size
            freq_kl_raw_sum += freq_kl_raw.item() * group_size
            ls_cond_sum += cond_mean.sum().item()
            ls_amp_norm_sum += c_norm_mean.sum().item()

            posterior_std_values.append(std_f.detach().reshape(-1).cpu())
            ls_cond_values.append(cond_mean.detach().reshape(-1).cpu())
            ls_amp_norm_values.append(c_norm_mean.detach().reshape(-1).cpu())

    total_groups = max(total_groups, 1)
    total_patches = max(total_patches, 1)
    total_freq_elements = max(total_freq_elements, 1)
    total_amp_elements = max(total_amp_elements, 1)
    for key in stats:
        stats[key] /= total_groups
    if num_harmonics is None:
        return stats

    freq_rmse_h = torch.sqrt(freq_sqerr_sum / total_groups)
    freq_mae_h = freq_abs_err_sum / total_groups
    freq_nrmse_h = torch.sqrt(freq_nsqerr_sum / total_groups)
    freq_success_h = freq_success_sum / total_groups
    amp_mape_h = amp_mape_sum / total_patches
    amp_abs_err_h = amp_abs_err_sum / total_patches
    amp_success_h = amp_success_sum / total_patches
    complex_rel_err_h = complex_rel_err_sum / total_patches
    complex_success_h = complex_success_sum / total_patches
    phase_circ_mae_h = phase_circ_err_sum / total_patches

    stats.update(
        {
            "freq_rmse_hz_mean": float(torch.sqrt(freq_sqerr_sum.sum() / total_freq_elements)),
            "freq_mae_hz_mean": float(freq_abs_err_sum.sum() / total_freq_elements),
            "freq_nrmse_band_mean": float(
                torch.sqrt(freq_nsqerr_sum.sum() / total_freq_elements)
            ),
            "freq_success_rate_mean": freq_sequence_success_sum / total_groups,
            "center_baseline_freq_rmse_hz": float(
                torch.sqrt(center_sqerr_sum.sum() / total_freq_elements)
            ),
            "harmonic_order_consistency": (
                harmonic_order_success_sum / total_order_pairs
                if total_order_pairs > 0
                else 1.0
            ),
            "posterior_std_hz_mean": posterior_std_sum / total_freq_elements,
            "posterior_std_hz_p95": float(
                torch.quantile(torch.cat(posterior_std_values), 0.95)
            ),
            "posterior_std_rel_mean": posterior_std_rel_sum / total_freq_elements,
            "posterior_std_local_hz_mean": (
                posterior_std_local_sum / max(total_amp_elements, 1)
            ),
            "freq_sample_outside_rate": freq_sample_outside_sum / total_groups,
            "freq_prior_reg": freq_prior_reg_sum / total_groups,
            "freq_kl": freq_kl_sum / total_groups,
            "freq_kl_raw": freq_kl_raw_sum / total_groups,
            "amp_mape_mean": float(amp_mape_sum.sum() / total_amp_elements),
            "amp_success_rate_mean": amp_sequence_success_sum / total_groups,
            "joint_amp_freq_success_rate_mean": (
                joint_amp_freq_sequence_success_sum / total_groups
            ),
            "complex_coeff_rel_err_mean": float(
                complex_rel_err_sum.sum() / total_amp_elements
            ),
            "complex_coeff_rel_err_vector": (
                complex_vector_rel_err_sum / total_patches
            ),
            "complex_coeff_success_rate": (
                complex_sequence_success_sum / total_groups
            ),
            "phase_circ_mae_rad": float(phase_circ_err_sum.sum() / total_amp_elements),
            "ls_cond_mean": ls_cond_sum / total_patches,
            "ls_cond_p95": float(torch.quantile(torch.cat(ls_cond_values), 0.95)),
            "ls_amp_norm_mean": ls_amp_norm_sum / total_patches,
            "ls_amp_norm_p95": float(torch.quantile(torch.cat(ls_amp_norm_values), 0.95)),
        }
    )

    _add_per_harmonic_metrics(stats, "freq_rmse", freq_rmse_h)
    _add_per_harmonic_metrics(stats, "freq_mae", freq_mae_h)
    _add_per_harmonic_metrics(stats, "freq_nrmse", freq_nrmse_h)
    for k in range(1, num_harmonics + 1):
        stats[f"freq_rmse_h{k}_hz"] = stats.pop(f"freq_rmse_h{k}")
        stats[f"freq_mae_h{k}_hz"] = stats.pop(f"freq_mae_h{k}")
        stats[f"freq_nrmse_h{k}_band"] = stats.pop(f"freq_nrmse_h{k}")
    _add_per_harmonic_metrics(stats, "freq_success", freq_success_h)
    _add_per_harmonic_metrics(stats, "amp_mape", amp_mape_h)
    _add_per_harmonic_metrics(stats, "amp_abs_err", amp_abs_err_h)
    for k in range(1, num_harmonics + 1):
        stats[f"amp_abs_err_h{k}_m"] = stats.pop(f"amp_abs_err_h{k}")
    _add_per_harmonic_metrics(stats, "amp_success", amp_success_h)
    for k in range(1, num_harmonics + 1):
        stats[f"amp_success_h{k}_magnitude"] = stats.pop(f"amp_success_h{k}")
    _add_per_harmonic_metrics(stats, "complex_coeff_rel_err", complex_rel_err_h)
    _add_per_harmonic_metrics(stats, "complex_coeff_success", complex_success_h)
    _add_per_harmonic_metrics(stats, "phase_circ_mae", phase_circ_mae_h)
    for k in range(1, num_harmonics + 1):
        stats[f"phase_circ_mae_h{k}_rad"] = stats.pop(f"phase_circ_mae_h{k}")

    return stats


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_cfg: Dict,
    dense_factor: int = 4,
) -> Dict[str, float]:
    del dense_factor  # Kept in the signature for runner compatibility.
    model.eval()
    if getattr(dataloader.dataset, "patches_per_group", 1) > 1:
        return _evaluate_group_profile_model(
            model=model,
            dataloader=dataloader,
            device=device,
            loss_cfg=loss_cfg,
        )

    stats = {
        "loss": 0.0,
        "recon_mse_mean": 0.0,
        "recon_mse_sampled": 0.0,
        "recon_nll_sampled": 0.0,
        "recon_nll_full": 0.0,
    }

    success_cfg = loss_cfg.get("success", {})
    freq_relative_tol = float(success_cfg.get("freq_relative_tol", 0.02))
    amp_relative_tol = float(success_cfg.get("amp_relative_tol", 0.05))
    complex_coeff_relative_tol = float(
        success_cfg.get("complex_coeff_relative_tol", amp_relative_tol)
    )
    rec_cfg = loss_cfg.get("reconstruction", {})
    s_seq = int(rec_cfg.get("sequence_posterior_samples", 1))

    total_sequences = 0
    total_freq_elements = 0
    total_order_pairs = 0
    num_harmonics = None

    freq_sqerr_sum = None
    freq_abs_err_sum = None
    freq_nsqerr_sum = None
    freq_success_sum = None
    center_sqerr_sum = None

    amp_mape_sum = None
    amp_abs_err_sum = None
    amp_success_sum = None

    complex_rel_err_sum = None
    complex_success_sum = None
    phase_circ_err_sum = None

    freq_sequence_success_sum = 0.0
    amp_sequence_success_sum = 0.0
    joint_amp_freq_sequence_success_sum = 0.0
    complex_sequence_success_sum = 0.0
    complex_vector_rel_err_sum = 0.0
    harmonic_order_success_sum = 0.0

    posterior_std_sum = 0.0
    posterior_std_rel_sum = 0.0
    freq_sample_outside_sum = 0.0
    freq_prior_reg_sum = 0.0
    freq_kl_sum = 0.0
    freq_kl_raw_sum = 0.0
    ls_cond_sum = 0.0
    ls_amp_norm_sum = 0.0

    posterior_std_values = []
    ls_cond_values = []
    ls_amp_norm_values = []

    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch["x"].to(device)
            t_batch = batch["t"].to(device)
            probe_ids = batch["probe_ids"].to(device)
            target_batch = batch["target"].to(device)
            noise_var_norm = batch["noise_var_norm"].to(device)
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
            n = x_batch.shape[0]
            k_count = mu_f.shape[1]
            if num_harmonics is None:
                num_harmonics = k_count
                freq_sqerr_sum = torch.zeros(k_count, dtype=torch.float64)
                freq_abs_err_sum = torch.zeros(k_count, dtype=torch.float64)
                freq_nsqerr_sum = torch.zeros(k_count, dtype=torch.float64)
                freq_success_sum = torch.zeros(k_count, dtype=torch.float64)
                center_sqerr_sum = torch.zeros(k_count, dtype=torch.float64)
                amp_mape_sum = torch.zeros(k_count, dtype=torch.float64)
                amp_abs_err_sum = torch.zeros(k_count, dtype=torch.float64)
                amp_success_sum = torch.zeros(k_count, dtype=torch.float64)
                complex_rel_err_sum = torch.zeros(k_count, dtype=torch.float64)
                complex_success_sum = torch.zeros(k_count, dtype=torch.float64)
                phase_circ_err_sum = torch.zeros(k_count, dtype=torch.float64)

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

            recon_nll_sampled, sampled_diag = compute_sequence_posterior_recon_loss(
                y_complex=y_complex,
                t=t_local,
                mu_f=mu_f,
                std_f=std_f,
                model=model,
                sequence_posterior_samples=s_seq,
                ridge_lambda=model.ls_ridge,
                noise_var_norm=noise_var_norm,
                include_log_const=bool(rec_cfg.get("include_log_const", False)),
            )
            recon_mse_sampled = sampled_diag["recon_mse_sampled"]
            recon_nll_full = sampled_diag["recon_nll_full"]

            lower = model.encoder.freq_lower.to(device=mu_f.device, dtype=mu_f.dtype)
            upper = model.encoder.freq_upper.to(device=mu_f.device, dtype=mu_f.dtype)
            freq_half = (upper - lower) / 2.0
            freq_center = (upper + lower) / 2.0
            freq_kl_per_item = _compute_freq_kl(mu_f, std_f, model, loss_cfg)
            freq_kl_raw = freq_kl_per_item.sum(dim=-1).mean()
            freq_kl = freq_kl_raw
            f_samples = sampled_diag["f_samples"]
            outside = (f_samples < lower.view(1, 1, -1)) | (
                f_samples > upper.view(1, 1, -1)
            )
            freq_sample_outside_rate = outside.float().mean()

            freq_err = mu_f - true_freq
            freq_abs_err = torch.abs(freq_err)
            freq_norm_err = freq_err / (freq_half.view(1, -1) + 1e-12)
            freq_rel_err = torch.abs(freq_err) / (torch.abs(true_freq) + 1e-12)

            c_pred_m = torch.complex(amp_real_mean, amp_imag_mean) * amp_scale[:, None]
            c_true_local = _align_true_complex_coeff_to_local_time(
                true_complex=true_amp,
                true_freq_hz=true_freq,
                t0=t0.squeeze(1),
            )
            amp_hat = torch.abs(c_pred_m)
            amp_true = torch.abs(c_true_local)
            amp_abs_err = torch.abs(amp_hat - amp_true)
            amp_mape = amp_abs_err / (amp_true + 1e-12)
            complex_rel_err = torch.abs(c_pred_m - c_true_local) / (
                amp_true + 1e-12
            )
            complex_vector_rel_err = torch.linalg.norm(
                c_pred_m - c_true_local,
                dim=-1,
            ) / (torch.linalg.norm(c_true_local, dim=-1) + 1e-12)
            phase_circ_err = _circular_abs_phase_error(c_pred_m, c_true_local)
            c_norm_mean = torch.linalg.norm(c_pred_m, dim=-1)

            freq_ok = freq_rel_err <= freq_relative_tol
            amp_ok = amp_mape <= amp_relative_tol
            complex_ok = complex_rel_err <= complex_coeff_relative_tol
            joint_amp_freq_ok = freq_ok & amp_ok

            freq_success_rate = torch.all(freq_ok, dim=1).float().mean()
            amp_success_rate = torch.all(amp_ok, dim=1).float().mean()
            joint_amp_freq_success_rate = torch.all(
                joint_amp_freq_ok,
                dim=1,
            ).float().mean()
            complex_success_rate = torch.all(complex_ok, dim=1).float().mean()

            if k_count > 1:
                order_ok = mu_f[:, 1:] > mu_f[:, :-1]
                harmonic_order_success_sum += order_ok.float().sum().item()
                total_order_pairs += order_ok.numel()

            beta_freq = float(loss_cfg.get("beta_freq", 1.0))
            loss = recon_nll_sampled + beta_freq * freq_kl

            total_sequences += n
            total_freq_elements += n * k_count
            stats["loss"] += loss.item() * n
            stats["recon_mse_mean"] += recon_mse_mean.item() * n
            stats["recon_mse_sampled"] += recon_mse_sampled.item() * n
            stats["recon_nll_sampled"] += recon_nll_sampled.item() * n
            stats["recon_nll_full"] += recon_nll_full.item() * n

            freq_sqerr_sum += freq_err.pow(2).sum(dim=0).double().cpu()
            freq_abs_err_sum += freq_abs_err.sum(dim=0).double().cpu()
            freq_nsqerr_sum += freq_norm_err.pow(2).sum(dim=0).double().cpu()
            freq_success_sum += freq_ok.float().sum(dim=0).double().cpu()
            center_sqerr_sum += (
                (freq_center.view(1, -1) - true_freq).pow(2).sum(dim=0).double().cpu()
            )

            amp_mape_sum += amp_mape.sum(dim=0).double().cpu()
            amp_abs_err_sum += amp_abs_err.sum(dim=0).double().cpu()
            amp_success_sum += amp_ok.float().sum(dim=0).double().cpu()

            complex_rel_err_sum += complex_rel_err.sum(dim=0).double().cpu()
            complex_success_sum += complex_ok.float().sum(dim=0).double().cpu()
            phase_circ_err_sum += phase_circ_err.sum(dim=0).double().cpu()

            freq_sequence_success_sum += freq_success_rate.item() * n
            amp_sequence_success_sum += amp_success_rate.item() * n
            joint_amp_freq_sequence_success_sum += joint_amp_freq_success_rate.item() * n
            complex_sequence_success_sum += complex_success_rate.item() * n
            complex_vector_rel_err_sum += complex_vector_rel_err.sum().item()

            posterior_std_sum += std_f.sum().item()
            posterior_std_rel_sum += (std_f / (freq_half.view(1, -1) + 1e-12)).sum().item()
            freq_sample_outside_sum += freq_sample_outside_rate.item() * n
            freq_prior_reg_sum += freq_kl.item() * n
            freq_kl_sum += freq_kl.item() * n
            freq_kl_raw_sum += freq_kl_raw.item() * n
            ls_cond_sum += cond_mean.sum().item()
            ls_amp_norm_sum += c_norm_mean.sum().item()

            posterior_std_values.append(std_f.detach().reshape(-1).cpu())
            ls_cond_values.append(cond_mean.detach().reshape(-1).cpu())
            ls_amp_norm_values.append(c_norm_mean.detach().reshape(-1).cpu())

    total_sequences = max(total_sequences, 1)
    for key in stats:
        stats[key] /= total_sequences

    if num_harmonics is None:
        return stats

    total_freq_elements = max(total_freq_elements, 1)
    freq_rmse_h = torch.sqrt(freq_sqerr_sum / total_sequences)
    freq_mae_h = freq_abs_err_sum / total_sequences
    freq_nrmse_h = torch.sqrt(freq_nsqerr_sum / total_sequences)
    freq_success_h = freq_success_sum / total_sequences

    amp_mape_h = amp_mape_sum / total_sequences
    amp_abs_err_h = amp_abs_err_sum / total_sequences
    amp_success_h = amp_success_sum / total_sequences

    complex_rel_err_h = complex_rel_err_sum / total_sequences
    complex_success_h = complex_success_sum / total_sequences
    phase_circ_mae_h = phase_circ_err_sum / total_sequences

    stats.update(
        {
            "freq_rmse_hz_mean": float(torch.sqrt(freq_sqerr_sum.sum() / total_freq_elements)),
            "freq_mae_hz_mean": float(freq_abs_err_sum.sum() / total_freq_elements),
            "freq_nrmse_band_mean": float(
                torch.sqrt(freq_nsqerr_sum.sum() / total_freq_elements)
            ),
            "freq_success_rate_mean": freq_sequence_success_sum / total_sequences,
            "center_baseline_freq_rmse_hz": float(
                torch.sqrt(center_sqerr_sum.sum() / total_freq_elements)
            ),
            "harmonic_order_consistency": (
                harmonic_order_success_sum / total_order_pairs
                if total_order_pairs > 0
                else 1.0
            ),
            "posterior_std_hz_mean": posterior_std_sum / total_freq_elements,
            "posterior_std_hz_p95": float(
                torch.quantile(torch.cat(posterior_std_values), 0.95)
            ),
            "posterior_std_rel_mean": posterior_std_rel_sum / total_freq_elements,
            "freq_sample_outside_rate": freq_sample_outside_sum / total_sequences,
            "freq_prior_reg": freq_prior_reg_sum / total_sequences,
            "freq_kl": freq_kl_sum / total_sequences,
            "freq_kl_raw": freq_kl_raw_sum / total_sequences,
            "amp_mape_mean": float(amp_mape_sum.sum() / total_freq_elements),
            "amp_success_rate_mean": amp_sequence_success_sum / total_sequences,
            "joint_amp_freq_success_rate_mean": (
                joint_amp_freq_sequence_success_sum / total_sequences
            ),
            "complex_coeff_rel_err_mean": float(
                complex_rel_err_sum.sum() / total_freq_elements
            ),
            "complex_coeff_rel_err_vector": (
                complex_vector_rel_err_sum / total_sequences
            ),
            "complex_coeff_success_rate": (
                complex_sequence_success_sum / total_sequences
            ),
            "phase_circ_mae_rad": float(phase_circ_err_sum.sum() / total_freq_elements),
            "ls_cond_mean": ls_cond_sum / total_sequences,
            "ls_cond_p95": float(torch.quantile(torch.cat(ls_cond_values), 0.95)),
            "ls_amp_norm_mean": ls_amp_norm_sum / total_sequences,
            "ls_amp_norm_p95": float(torch.quantile(torch.cat(ls_amp_norm_values), 0.95)),
        }
    )

    _add_per_harmonic_metrics(stats, "freq_rmse", freq_rmse_h)
    _add_per_harmonic_metrics(stats, "freq_mae", freq_mae_h)
    _add_per_harmonic_metrics(stats, "freq_nrmse", freq_nrmse_h)
    for k in range(1, num_harmonics + 1):
        stats[f"freq_rmse_h{k}_hz"] = stats.pop(f"freq_rmse_h{k}")
        stats[f"freq_mae_h{k}_hz"] = stats.pop(f"freq_mae_h{k}")
        stats[f"freq_nrmse_h{k}_band"] = stats.pop(f"freq_nrmse_h{k}")
    _add_per_harmonic_metrics(stats, "freq_success", freq_success_h)

    _add_per_harmonic_metrics(stats, "amp_mape", amp_mape_h)
    _add_per_harmonic_metrics(stats, "amp_abs_err", amp_abs_err_h)
    for k in range(1, num_harmonics + 1):
        stats[f"amp_abs_err_h{k}_m"] = stats.pop(f"amp_abs_err_h{k}")
    _add_per_harmonic_metrics(stats, "amp_success", amp_success_h)
    for k in range(1, num_harmonics + 1):
        stats[f"amp_success_h{k}_magnitude"] = stats.pop(f"amp_success_h{k}")

    _add_per_harmonic_metrics(stats, "complex_coeff_rel_err", complex_rel_err_h)
    _add_per_harmonic_metrics(stats, "complex_coeff_success", complex_success_h)

    _add_per_harmonic_metrics(stats, "phase_circ_mae", phase_circ_mae_h)
    for k in range(1, num_harmonics + 1):
        stats[f"phase_circ_mae_h{k}_rad"] = stats.pop(f"phase_circ_mae_h{k}")

    return stats
