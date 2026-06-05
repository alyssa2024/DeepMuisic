from typing import Dict

import torch

from batch_utils import extract_dataset_state
from loss import (
    _complex_global_to_local,
    _static_global_amp_prior_cfg,
    _select_amp_warmup_frequency,
    build_normalized_amp_prior,
    complex_diag_gaussian_kl,
    compute_frequency_kl,
    compute_sequence_bayesian_nnamp_recon_loss,
    compute_sequence_nnamp_recon_loss,
    compute_sequence_posterior_recon_loss,
    compute_sequence_strict_global_elbo_recon_loss,
    sample_sequence_frequencies,
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


def _align_local_complex_coeff_to_global_time(
    local_complex: torch.Tensor,
    freq_hz: torch.Tensor,
    t0: torch.Tensor,
) -> torch.Tensor:
    phase_shift = 2.0 * torch.pi * freq_hz * t0[:, None]
    return local_complex * torch.exp(-1j * phase_shift)


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


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_cfg: Dict,
    signal_cfg: Dict = None,
    dense_factor: int = 4,
) -> Dict[str, float]:
    del dense_factor  # Kept in the signature for runner compatibility.
    model.eval()

    stats = {
        "loss": 0.0,
        "optimization_loss": 0.0,
        "loss_scale": 0.0,
        "recon_mse_mean": 0.0,
        "recon_mse_sampled": 0.0,
        "recon_loss_per_point": 0.0,
        "recon_nll_sampled": 0.0,
        "recon_nll_full": 0.0,
        "amp_prior_quad": 0.0,
        "marginal_nll": 0.0,
        "marginal_quad": 0.0,
        "marginal_logdet": 0.0,
        "amp_supervision_loss": 0.0,
        "amp_supervision_loss_raw": 0.0,
        "amp_supervision_weight": 0.0,
        "amp_supervision_target_norm_mean": 0.0,
        "amp_supervision_error_norm_mean": 0.0,
    }

    success_cfg = loss_cfg.get("success", {})
    freq_relative_tol = float(success_cfg.get("freq_relative_tol", 0.02))
    amp_relative_tol = float(success_cfg.get("amp_relative_tol", 0.05))
    complex_coeff_relative_tol = float(
        success_cfg.get("complex_coeff_relative_tol", amp_relative_tol)
    )
    rec_cfg = loss_cfg.get("reconstruction", {})
    s_seq = int(rec_cfg.get("sequence_posterior_samples", 1))
    use_posterior_sampling = bool(rec_cfg.get("use_posterior_sampling", True))
    normalize_by_num_points = bool(rec_cfg.get("normalize_by_num_points", False))
    include_log_const = bool(rec_cfg.get("include_log_const", False))
    recon_weight = float(loss_cfg.get("reconstruction_weight", 1.0))
    amp_warmup_cfg = loss_cfg.get("amplitude_warmup", {})
    amp_warmup_enabled = bool(amp_warmup_cfg.get("enabled", False))
    amp_freq_source = amp_warmup_cfg.get("frequency_source", "mu_f")
    detach_amp_frequency = bool(amp_warmup_cfg.get("detach_frequency", True))
    amp_sup_cfg = loss_cfg.get("amp_supervision", {})
    amp_sup_enabled = bool(amp_sup_cfg.get("enabled", False))
    amp_sup_weight = float(amp_sup_cfg.get("weight", 1.0))

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
    local_complex_rel_err_sum = None
    local_phase_circ_err_sum = None

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
    amp_lambda_mean_sum = 0.0
    amp_prior_var_norm_mean_sum = 0.0
    amp_post_std_norm_sum = 0.0
    amp_post_std_m_sum = 0.0
    amp_post_var_trace_sum = 0.0
    amp_uncertainty_to_prior_ratio_sum = 0.0

    posterior_std_values = []
    ls_cond_values = []
    ls_amp_norm_values = []
    data_state_sums = {}

    mode, amp_prior_cfg = _static_global_amp_prior_cfg(loss_cfg)
    if signal_cfg is None:
        amp_prior_cfg["enabled"] = False
    use_amp_prior = bool(amp_prior_cfg.get("enabled", False))
    amp_mode = amp_prior_cfg.get("mode", "map")
    use_strict_elbo = mode == "static_global_strict_elbo"
    use_nn_amp = mode in (
        "static_global_nnamp",
        "static_global_bayesian_nnamp",
        "static_global_strict_elbo",
    )
    use_bayesian_nnamp = mode == "static_global_bayesian_nnamp"
    amp_representation = getattr(model, "amplitude_nn_representation", "segment_local")
    ls_at_pred_recon_mse_sum = 0.0
    ls_at_pred_amp_mape_sum = 0.0
    ls_at_pred_complex_rel_err_sum = 0.0
    ls_at_pred_phase_err_sum = 0.0
    amp_kl_sum = 0.0
    amp_kl_raw_sum = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch["x_windows"].to(device)
            t_batch = batch["t_windows"].to(device)
            probe_ids = batch["probe_ids_windows"].to(device)
            target_batch = batch["target_windows"].to(device)
            window_start_cycle = batch["window_start_cycle"].to(device)
            noise_var_norm = batch["noise_var_norm"].to(device)
            dataset_state = extract_dataset_state(batch, device)
            true_freq = batch["true_freq_hz"].to(device)
            true_amp = torch.complex(
                batch["true_amp_real"].to(device),
                batch["true_amp_imag"].to(device),
            )
            amp_scale = batch["amp_scale"].to(device)

            batch_size, num_windows, seq_len, _ = target_batch.shape
            t0 = t_batch[:, 0, :1]
            t_global = (t_batch - t0.view(batch_size, 1, 1)).reshape(
                batch_size,
                num_windows * seq_len,
            )
            target_global = target_batch.reshape(batch_size, num_windows * seq_len, 2)

            outputs = model.forward_global(
                x_batch,
                probe_ids_windows=probe_ids,
                window_start_cycle=window_start_cycle,
            )
            mu_f = outputs["mu_f"]
            std_f = outputs["std_f"]

            y_complex = torch.complex(target_global[..., 0], target_global[..., 1])
            n = x_batch.shape[0]
            k_count = mu_f.shape[1]
            if amp_warmup_enabled:
                f_amp_eval = _select_amp_warmup_frequency(
                    source=amp_freq_source,
                    mu_f=mu_f,
                    model=model,
                    true_freq_hz=true_freq,
                    detach=detach_amp_frequency,
                )
            else:
                f_amp_eval = mu_f
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
                local_complex_rel_err_sum = torch.zeros(k_count, dtype=torch.float64)
                local_phase_circ_err_sum = torch.zeros(k_count, dtype=torch.float64)

            amp_post_var_diag = None
            amp_prior_mean_nn = None
            amp_prior_var_nn = None
            if use_bayesian_nnamp:
                amp_prior_mean_nn, amp_prior_var_nn = build_normalized_amp_prior(
                    f=mu_f,
                    t0=t0.squeeze(1),
                    amp_scale=amp_scale,
                    signal_cfg=signal_cfg,
                    amp_prior_cfg=loss_cfg.get("amplitude_prior", {}),
                )

            if use_nn_amp:
                c_model = outputs["c_nn"]
                if mode == "static_global_nnamp" and amp_representation == "parent_global":
                    c_mean = _complex_global_to_local(
                        c_global=c_model,
                        f_hz=f_amp_eval,
                        t0=t0.squeeze(1),
                    )
                else:
                    c_mean = c_model
                amp_real_mean = c_mean.real
                amp_imag_mean = c_mean.imag
                cond_mean = torch.zeros(
                    batch_size,
                    device=device,
                    dtype=target_batch.dtype,
                )
            elif use_amp_prior:
                amp_prior_mean, amp_prior_var = build_normalized_amp_prior(
                    f=mu_f,
                    t0=t0.squeeze(1),
                    amp_scale=amp_scale,
                    signal_cfg=signal_cfg,
                    amp_prior_cfg=amp_prior_cfg,
                )
                if amp_mode == "marginal_likelihood":
                    c_mean, amp_post_var_diag, bayes_diag = model.solve_amplitudes_bayes(
                        y_complex=y_complex,
                        f=mu_f,
                        t=t_global,
                        amp_prior_mean=amp_prior_mean,
                        amp_prior_var=amp_prior_var,
                        noise_var_norm=noise_var_norm,
                        return_cov_diag=True,
                        return_condition=True,
                    )
                    amp_real_mean = c_mean.real
                    amp_imag_mean = c_mean.imag
                    cond_mean = bayes_diag["bayes_cond"]
                else:
                    amp_real_mean, amp_imag_mean, c_mean, cond_mean = (
                        model.solve_amplitudes_map(
                            y_complex=y_complex,
                            f=mu_f,
                            t=t_global,
                            amp_prior_mean=amp_prior_mean,
                            amp_prior_var=amp_prior_var,
                            noise_var_norm=noise_var_norm,
                            return_condition=True,
                        )
                    )
            else:
                amp_real_mean, amp_imag_mean, c_mean, cond_mean = (
                    model.solve_amplitudes_ls(
                        y_complex=y_complex,
                        f=mu_f,
                        t=t_global,
                        ridge_lambda=model.ls_ridge,
                        return_condition=True,
                    )
                )
            x_hat_mean = model.decode(
                amp_real=amp_real_mean,
                amp_imag=amp_imag_mean,
                f=f_amp_eval if mode == "static_global_nnamp" else mu_f,
                t=t_global,
            )
            recon_mse_mean = _complex_ri_mse(x_hat_mean, target_global)

            if use_strict_elbo:
                sampled_recon_loss, amp_kl_raw, sampled_diag = (
                    compute_sequence_strict_global_elbo_recon_loss(
                        y_complex=y_complex,
                        t=t_global,
                        mu_f=mu_f,
                        std_f=std_f,
                        amp_mu=outputs["c_nn"],
                        amp_var=outputs["amp_var_nn"],
                        model=model,
                        noise_var_norm=noise_var_norm,
                        amp_scale=amp_scale,
                        t0=t0.squeeze(1),
                        signal_cfg=signal_cfg,
                        amp_prior_cfg=amp_prior_cfg,
                        num_samples=s_seq,
                        include_log_const=include_log_const,
                        normalize_by_num_points=normalize_by_num_points,
                    )
                )
            elif use_bayesian_nnamp:
                if use_posterior_sampling:
                    f_eval_samples = sample_sequence_frequencies(
                        mu_f=mu_f,
                        std_f=std_f,
                        num_samples=s_seq,
                        freq_lower=model.encoder.freq_lower,
                        freq_upper=model.encoder.freq_upper,
                    )
                else:
                    f_eval_samples = mu_f.unsqueeze(0)
                sampled_recon_loss, sampled_diag = compute_sequence_bayesian_nnamp_recon_loss(
                    y_complex=y_complex,
                    t=t_global,
                    mu_f=mu_f,
                    amp_mu=outputs["c_nn"],
                    amp_var=outputs["amp_var_nn"],
                    model=model,
                    noise_var_norm=noise_var_norm,
                    include_log_const=include_log_const,
                    normalize_by_num_points=normalize_by_num_points,
                    f_samples=f_eval_samples,
                )
                amp_kl_per_item = complex_diag_gaussian_kl(
                    mu_q=outputs["c_nn"],
                    var_q=outputs["amp_var_nn"],
                    mu_p=amp_prior_mean_nn,
                    var_p=amp_prior_var_nn,
                )
                amp_kl_raw = amp_kl_per_item.mean()
            elif use_nn_amp:
                sampled_recon_loss, sampled_diag = compute_sequence_nnamp_recon_loss(
                    y_complex=y_complex,
                    t=t_global,
                    mu_f=f_amp_eval if mode == "static_global_nnamp" else mu_f,
                    c_nn=c_mean,
                    model=model,
                    noise_var_norm=noise_var_norm,
                    include_log_const=include_log_const,
                    normalize_by_num_points=normalize_by_num_points,
                )
                amp_kl_raw = torch.zeros((), device=device, dtype=target_batch.dtype)
            else:
                sampled_recon_loss, sampled_diag = compute_sequence_posterior_recon_loss(
                    y_complex=y_complex,
                    t=t_global,
                    mu_f=mu_f,
                    std_f=std_f,
                    model=model,
                    sequence_posterior_samples=s_seq,
                    ridge_lambda=model.ls_ridge,
                    noise_var_norm=noise_var_norm,
                    include_log_const=include_log_const,
                    amp_scale=amp_scale,
                    t0=t0.squeeze(1),
                    signal_cfg=signal_cfg,
                    amp_prior_cfg=amp_prior_cfg,
                    use_posterior_sampling=use_posterior_sampling,
                    normalize_by_num_points=normalize_by_num_points,
                )
                amp_kl_raw = torch.zeros((), device=device, dtype=target_batch.dtype)
            recon_mse_sampled = sampled_diag["recon_mse_sampled"]
            recon_nll_sampled = sampled_diag["recon_nll"]
            recon_nll_full = sampled_diag["recon_nll_full"]

            lower = model.encoder.freq_lower.to(device=mu_f.device, dtype=mu_f.dtype)
            upper = model.encoder.freq_upper.to(device=mu_f.device, dtype=mu_f.dtype)
            freq_half = (upper - lower) / 2.0
            freq_center = (upper + lower) / 2.0
            freq_kl_per_item = compute_frequency_kl(
                mu_f=mu_f,
                std_f=std_f,
                model=model,
                loss_cfg=loss_cfg,
            )
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

            f_coeff_eval = f_amp_eval if mode == "static_global_nnamp" else mu_f
            c_pred_local_m = (
                torch.complex(amp_real_mean, amp_imag_mean) * amp_scale[:, None]
            )
            if mode == "static_global_nnamp" and amp_representation == "parent_global":
                c_pred_global_m = outputs["c_nn"] * amp_scale[:, None]
            else:
                c_pred_global_m = _align_local_complex_coeff_to_global_time(
                    local_complex=c_pred_local_m,
                    freq_hz=f_coeff_eval,
                    t0=t0.squeeze(1),
                )
            c_true_local = _align_true_complex_coeff_to_local_time(
                true_complex=true_amp,
                true_freq_hz=true_freq,
                t0=t0.squeeze(1),
            )
            amp_hat = torch.abs(c_pred_global_m)
            amp_true = torch.abs(true_amp)
            amp_abs_err = torch.abs(amp_hat - amp_true)
            amp_mape = amp_abs_err / (amp_true + 1e-12)
            complex_rel_err = torch.abs(c_pred_global_m - true_amp) / (
                amp_true + 1e-12
            )
            complex_vector_rel_err = torch.linalg.norm(
                c_pred_global_m - true_amp,
                dim=-1,
            ) / (torch.linalg.norm(true_amp, dim=-1) + 1e-12)
            phase_circ_err = _circular_abs_phase_error(c_pred_global_m, true_amp)
            local_complex_rel_err = torch.abs(c_pred_local_m - c_true_local) / (
                torch.abs(c_true_local) + 1e-12
            )
            local_phase_circ_err = _circular_abs_phase_error(
                c_pred_local_m,
                c_true_local,
            )
            c_norm_mean = torch.linalg.norm(c_pred_local_m, dim=-1)

            ls_amp_real_pred, ls_amp_imag_pred, c_ls_pred, _ = model.solve_amplitudes_ls(
                y_complex=y_complex,
                f=f_amp_eval if mode == "static_global_nnamp" else mu_f,
                t=t_global,
                ridge_lambda=model.ls_ridge,
                return_condition=True,
            )
            x_hat_ls_pred = model.decode(
                amp_real=ls_amp_real_pred,
                amp_imag=ls_amp_imag_pred,
                f=f_amp_eval if mode == "static_global_nnamp" else mu_f,
                t=t_global,
            )
            ls_at_pred_recon_mse = _complex_ri_mse(x_hat_ls_pred, target_global)
            c_ls_local_m = c_ls_pred * amp_scale[:, None]
            c_ls_global_m = _align_local_complex_coeff_to_global_time(
                local_complex=c_ls_local_m,
                freq_hz=f_coeff_eval,
                t0=t0.squeeze(1),
            )
            ls_amp_hat = torch.abs(c_ls_global_m)
            ls_amp_mape = torch.abs(ls_amp_hat - amp_true) / (amp_true + 1e-12)
            ls_complex_rel_err = torch.abs(c_ls_global_m - true_amp) / (
                amp_true + 1e-12
            )
            ls_phase_circ_err = _circular_abs_phase_error(c_ls_global_m, true_amp)

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
            amp_kl_cfg = loss_cfg.get("amplitude_kl", {})
            beta_amp = float(amp_kl_cfg.get("beta_amp", amp_kl_cfg.get("beta", 1.0)))
            amp_kl_enabled = bool(
                amp_kl_cfg.get("enabled", use_bayesian_nnamp or use_strict_elbo)
            )
            amp_kl = amp_kl_raw if amp_kl_enabled else torch.zeros_like(amp_kl_raw)
            amp_sup_loss_raw = torch.zeros((), device=device, dtype=target_batch.dtype)
            amp_sup_target_norm = torch.zeros((), device=device, dtype=target_batch.dtype)
            amp_sup_error_norm = torch.zeros((), device=device, dtype=target_batch.dtype)
            if amp_sup_enabled:
                if amp_sup_cfg.get("target", "ls_at_mu_f") != "ls_at_mu_f":
                    raise ValueError(
                        "loss.amp_supervision.target currently only supports "
                        "'ls_at_mu_f'"
                    )
                if "c_nn" not in outputs:
                    raise KeyError("amp_supervision requires model outputs['c_nn']")
                amp_sup_error = c_mean - c_ls_pred
                err2 = torch.sum(torch.abs(amp_sup_error) ** 2, dim=-1)
                ref2 = torch.sum(torch.abs(c_ls_pred.detach()) ** 2, dim=-1).clamp_min(
                    1e-8
                )
                amp_sup_loss_raw = (err2 / ref2).mean()
                amp_sup_target_norm = torch.linalg.norm(c_ls_pred, dim=-1).mean()
                amp_sup_error_norm = torch.linalg.norm(amp_sup_error, dim=-1).mean()
            amp_sup_loss = amp_sup_weight * amp_sup_loss_raw
            loss = (
                recon_weight * sampled_recon_loss
                + amp_sup_loss
                + beta_freq * freq_kl
                + beta_amp * amp_kl
            )
            objective_num_points = max(int(num_windows * seq_len), 1)
            scale_modes = {
                "static_global_strict_elbo",
                "static_global_nnamp",
                "static_global_bayesian_nnamp",
            }
            loss_scale = (
                1.0 / float(objective_num_points)
                if mode in scale_modes and not normalize_by_num_points
                else 1.0
            )
            optimization_loss = loss * loss_scale

            total_sequences += n
            total_freq_elements += n * k_count
            stats["loss"] += loss.item() * n
            stats["optimization_loss"] += optimization_loss.item() * n
            stats["loss_scale"] += loss_scale * n
            stats["recon_mse_mean"] += recon_mse_mean.item() * n
            stats["recon_mse_sampled"] += recon_mse_sampled.item() * n
            stats["recon_loss_per_point"] += sampled_recon_loss.item() * loss_scale * n
            stats["recon_nll_sampled"] += recon_nll_sampled.item() * n
            stats["recon_nll_full"] += recon_nll_full.item() * n
            stats["amp_prior_quad"] += sampled_diag["amp_prior_quad"].item() * n
            stats["marginal_nll"] += sampled_diag["marginal_nll"].item() * n
            stats["marginal_quad"] += sampled_diag["marginal_quad"].item() * n
            stats["marginal_logdet"] += sampled_diag["marginal_logdet"].item() * n
            stats["amp_supervision_loss"] += amp_sup_loss.item() * n
            stats["amp_supervision_loss_raw"] += amp_sup_loss_raw.item() * n
            stats["amp_supervision_weight"] += amp_sup_weight * n
            stats["amp_supervision_target_norm_mean"] += amp_sup_target_norm.item() * n
            stats["amp_supervision_error_norm_mean"] += amp_sup_error_norm.item() * n
            amp_kl_sum += amp_kl.item() * n
            amp_kl_raw_sum += amp_kl_raw.item() * n
            ls_at_pred_recon_mse_sum += ls_at_pred_recon_mse.item() * n
            ls_at_pred_amp_mape_sum += ls_amp_mape.sum().item()
            ls_at_pred_complex_rel_err_sum += ls_complex_rel_err.sum().item()
            ls_at_pred_phase_err_sum += ls_phase_circ_err.sum().item()
            for key, value in dataset_state.items():
                if not torch.is_tensor(value) or value.numel() == 0:
                    continue
                v_float = value if torch.is_floating_point(value) else value.float()
                data_state_sums.setdefault(f"data_state/{key}_mean", 0.0)
                data_state_sums[f"data_state/{key}_mean"] += v_float.mean().item() * n

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
            local_complex_rel_err_sum += (
                local_complex_rel_err.sum(dim=0).double().cpu()
            )
            local_phase_circ_err_sum += local_phase_circ_err.sum(dim=0).double().cpu()

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
            amp_lambda_mean_sum += sampled_diag["amp_lambda_mean"].item() * n
            amp_prior_var_norm_mean_sum += (
                sampled_diag["amp_prior_var_norm_mean"].item() * n
            )
            amp_post_var_trace_sum += sampled_diag["amp_post_var_trace"].item() * n
            amp_post_std_norm_sum += sampled_diag["amp_post_std_mean"].item() * n
            amp_uncertainty_to_prior_ratio_sum += (
                sampled_diag["amp_uncertainty_to_prior_ratio_mean"].item() * n
            )
            if amp_post_var_diag is not None:
                amp_post_std_m_sum += (
                    torch.sqrt(amp_post_var_diag.clamp_min(1e-12))
                    * amp_scale[:, None]
                ).mean().item() * n

            posterior_std_values.append(std_f.detach().reshape(-1).cpu())
            ls_cond_values.append(cond_mean.detach().reshape(-1).cpu())
            ls_amp_norm_values.append(c_norm_mean.detach().reshape(-1).cpu())

    total_sequences = max(total_sequences, 1)
    for key in stats:
        stats[key] /= total_sequences
    for key, value in data_state_sums.items():
        stats[key] = value / total_sequences

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
    local_complex_rel_err_h = local_complex_rel_err_sum / total_sequences
    local_phase_circ_mae_h = local_phase_circ_err_sum / total_sequences

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
            "amp_kl": amp_kl_sum / total_sequences,
            "amp_kl_raw": amp_kl_raw_sum / total_sequences,
            "amp_mape_mean": float(amp_mape_sum.sum() / total_freq_elements),
            "nn_amp_mape_mean": (
                float(amp_mape_sum.sum() / total_freq_elements) if use_nn_amp else 0.0
            ),
            "amp_success_rate_mean": amp_sequence_success_sum / total_sequences,
            "joint_amp_freq_success_rate_mean": (
                joint_amp_freq_sequence_success_sum / total_sequences
            ),
            "complex_coeff_rel_err_mean": float(
                complex_rel_err_sum.sum() / total_freq_elements
            ),
            "nn_complex_coeff_rel_err_mean": (
                float(complex_rel_err_sum.sum() / total_freq_elements)
                if use_nn_amp
                else 0.0
            ),
            "complex_coeff_rel_err_vector": (
                complex_vector_rel_err_sum / total_sequences
            ),
            "complex_coeff_success_rate": (
                complex_sequence_success_sum / total_sequences
            ),
            "phase_circ_mae_rad": float(phase_circ_err_sum.sum() / total_freq_elements),
            "nn_phase_circ_mae_rad": (
                float(phase_circ_err_sum.sum() / total_freq_elements)
                if use_nn_amp
                else 0.0
            ),
            "local_complex_coeff_rel_err_mean": float(
                local_complex_rel_err_sum.sum() / total_freq_elements
            ),
            "local_phase_circ_mae_rad": float(
                local_phase_circ_err_sum.sum() / total_freq_elements
            ),
            "nn_recon_mse": stats["recon_mse_mean"] if use_nn_amp else 0.0,
            "ls_at_pred_recon_mse": ls_at_pred_recon_mse_sum / total_sequences,
            "ls_at_pred_amp_mape_mean": (
                ls_at_pred_amp_mape_sum / total_freq_elements
            ),
            "ls_at_pred_complex_coeff_rel_err_mean": (
                ls_at_pred_complex_rel_err_sum / total_freq_elements
            ),
            "ls_at_pred_phase_circ_mae_rad": (
                ls_at_pred_phase_err_sum / total_freq_elements
            ),
            "ls_cond_mean": ls_cond_sum / total_sequences,
            "ls_cond_p95": float(torch.quantile(torch.cat(ls_cond_values), 0.95)),
            "ls_amp_norm_mean": ls_amp_norm_sum / total_sequences,
            "ls_amp_norm_p95": float(torch.quantile(torch.cat(ls_amp_norm_values), 0.95)),
            "map_amp_norm_mean": ls_amp_norm_sum / total_sequences,
            "map_amp_norm_p95": float(torch.quantile(torch.cat(ls_amp_norm_values), 0.95)),
            "amp_lambda_mean": amp_lambda_mean_sum / total_sequences,
            "amp_prior_var_norm_mean": (
                amp_prior_var_norm_mean_sum / total_sequences
            ),
            "amp_post_std_norm_mean": amp_post_std_norm_sum / total_sequences,
            "amp_post_std_m_mean": amp_post_std_m_sum / total_sequences,
            "amp_post_var_trace_mean": amp_post_var_trace_sum / total_sequences,
            "amp_uncertainty_to_prior_ratio_mean": (
                amp_uncertainty_to_prior_ratio_sum / total_sequences
            ),
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

    _add_per_harmonic_metrics(
        stats,
        "local_complex_coeff_rel_err",
        local_complex_rel_err_h,
    )
    _add_per_harmonic_metrics(stats, "local_phase_circ_mae", local_phase_circ_mae_h)
    for k in range(1, num_harmonics + 1):
        stats[f"local_phase_circ_mae_h{k}_rad"] = stats.pop(
            f"local_phase_circ_mae_h{k}"
        )

    return stats
