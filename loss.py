import math

import torch
import torch.nn.functional as F


def complex_mse_loss(x_hat_complex, target_ri):
    if not torch.is_complex(x_hat_complex):
        raise TypeError(f"x_hat_complex must be complex, got {x_hat_complex.dtype}")
    if target_ri.ndim != 3 or target_ri.shape[-1] != 2:
        raise ValueError(f"target_ri must have shape [B, L, 2], got {target_ri.shape}")

    x_hat_ri = torch.stack([x_hat_complex.real, x_hat_complex.imag], dim=-1)
    return F.mse_loss(x_hat_ri, target_ri, reduction="mean")


def standard_normal_pdf(x):
    return torch.exp(-0.5 * x.pow(2)) / math.sqrt(2.0 * math.pi)


def standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _as_frequency_bounds(freq_lower, freq_upper, ref_tensor):
    lower = freq_lower.to(device=ref_tensor.device, dtype=ref_tensor.dtype).view(1, -1)
    upper = freq_upper.to(device=ref_tensor.device, dtype=ref_tensor.dtype).view(1, -1)
    if torch.any(upper <= lower):
        raise ValueError("freq_upper must be greater than freq_lower")
    return lower, upper


def sample_truncated_normal_frequencies(
    mu_f,
    std_f,
    num_samples,
    freq_lower,
    freq_upper,
    eps=1e-6,
):
    """
    Reparameterized inverse-CDF sampling from
    TN_[freq_lower, freq_upper](mu_f, std_f^2).

    Args:
        mu_f:       [B, K]
        std_f:      [B, K]
        num_samples: int
        freq_lower: [K]
        freq_upper: [K]

    Returns:
        f_samples: [S, B, K]
    """
    if mu_f.ndim != 2 or std_f.ndim != 2:
        raise ValueError(f"mu_f/std_f must be [B, K], got {mu_f.shape}/{std_f.shape}")
    if mu_f.shape != std_f.shape:
        raise ValueError(f"mu_f and std_f shape mismatch: {mu_f.shape} vs {std_f.shape}")

    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    std_f = std_f.clamp_min(eps)
    lower, upper = _as_frequency_bounds(freq_lower, freq_upper, mu_f)

    alpha = (lower - mu_f) / std_f
    beta = (upper - mu_f) / std_f

    cdf_alpha = standard_normal_cdf(alpha)
    cdf_beta = standard_normal_cdf(beta)
    z_mass = (cdf_beta - cdf_alpha).clamp_min(eps)

    u = torch.rand(
        num_samples,
        *mu_f.shape,
        device=mu_f.device,
        dtype=mu_f.dtype,
    )
    target_cdf = cdf_alpha.unsqueeze(0) + u * z_mass.unsqueeze(0)
    target_cdf = target_cdf.clamp(eps, 1.0 - eps)

    normal = torch.distributions.Normal(
        torch.zeros_like(target_cdf),
        torch.ones_like(target_cdf),
    )
    z = normal.icdf(target_cdf)
    f_samples = mu_f.unsqueeze(0) + std_f.unsqueeze(0) * z

    # Numerical safety only; inverse-CDF sampling is the actual sampler.
    f_samples = torch.minimum(
        torch.maximum(f_samples, lower.unsqueeze(0)),
        upper.unsqueeze(0),
    )
    assert f_samples.ndim == 3
    assert f_samples.shape[1:] == mu_f.shape
    return f_samples


def kl_trunc_normal_uniform(mu_f, std_f, freq_lower, freq_upper, eps=1e-8):
    """
    KL( TN_[a,b](mu_f, std_f^2) || Uniform(a,b) ).

    Args:
        mu_f:       [B, K]
        std_f:      [B, K]
        freq_lower: [K]
        freq_upper: [K]

    Returns:
        kl_per_item: [B, K]
    """
    if mu_f.ndim != 2 or std_f.ndim != 2:
        raise ValueError(f"mu_f/std_f must be [B, K], got {mu_f.shape}/{std_f.shape}")
    if mu_f.shape != std_f.shape:
        raise ValueError(f"mu_f and std_f shape mismatch: {mu_f.shape} vs {std_f.shape}")

    std_f = std_f.clamp_min(eps)
    lower, upper = _as_frequency_bounds(freq_lower, freq_upper, mu_f)

    alpha = (lower - mu_f) / std_f
    beta = (upper - mu_f) / std_f

    cdf_alpha = standard_normal_cdf(alpha)
    cdf_beta = standard_normal_cdf(beta)
    z = (cdf_beta - cdf_alpha).clamp_min(eps)

    pdf_alpha = standard_normal_pdf(alpha)
    pdf_beta = standard_normal_pdf(beta)

    interval = (upper - lower).clamp_min(eps)

    entropy = (
        torch.log(std_f)
        + torch.log(z)
        + 0.5 * math.log(2.0 * math.pi * math.e)
        + (alpha * pdf_alpha - beta * pdf_beta) / (2.0 * z)
    )

    kl = torch.log(interval) - entropy
    return torch.clamp(kl, min=0.0)


def kl_trunc_normal_trunc_normal(
    mu_q,
    std_q,
    mu_p,
    std_p,
    lower,
    upper,
    eps=1e-8,
):
    """
    KL( TN_[lower, upper](mu_q, std_q^2)
        ||
        TN_[lower, upper](mu_p, std_p^2) ).
    """
    if mu_q.ndim != 2 or std_q.ndim != 2:
        raise ValueError(f"mu_q/std_q must be [B, K], got {mu_q.shape}/{std_q.shape}")
    if mu_q.shape != std_q.shape:
        raise ValueError(f"mu_q and std_q shape mismatch: {mu_q.shape} vs {std_q.shape}")

    std_q = std_q.clamp_min(eps)
    lower, upper = _as_frequency_bounds(lower, upper, mu_q)

    mu_p = mu_p.to(device=mu_q.device, dtype=mu_q.dtype)
    std_p = std_p.to(device=mu_q.device, dtype=mu_q.dtype).clamp_min(eps)
    if mu_p.ndim == 1:
        mu_p = mu_p.view(1, -1)
    if std_p.ndim == 1:
        std_p = std_p.view(1, -1)

    alpha_q = (lower - mu_q) / std_q
    beta_q = (upper - mu_q) / std_q
    alpha_p = (lower - mu_p) / std_p
    beta_p = (upper - mu_p) / std_p

    z_q = (standard_normal_cdf(beta_q) - standard_normal_cdf(alpha_q)).clamp_min(eps)
    z_p = (standard_normal_cdf(beta_p) - standard_normal_cdf(alpha_p)).clamp_min(eps)

    pdf_alpha_q = standard_normal_pdf(alpha_q)
    pdf_beta_q = standard_normal_pdf(beta_q)
    m_q = (pdf_alpha_q - pdf_beta_q) / z_q
    s_q = 1.0 + (alpha_q * pdf_alpha_q - beta_q * pdf_beta_q) / z_q

    delta = mu_q - mu_p
    prior_quad = (
        delta.pow(2)
        + 2.0 * delta * std_q * m_q
        + std_q.pow(2) * s_q
    ) / std_p.pow(2)

    kl = (
        torch.log(std_p)
        + torch.log(z_p)
        - torch.log(std_q)
        - torch.log(z_q)
        + 0.5 * (prior_quad - s_q)
    )
    return torch.clamp(kl, min=0.0)


def compute_beta_anneal(loss_cfg, step=None):
    kl_cfg = loss_cfg.get("kl", {})
    if not bool(kl_cfg.get("enabled", True)):
        return 0.0

    warmup_steps = int(kl_cfg.get("warmup_steps", 0))
    if step is None or warmup_steps <= 0:
        return 1.0

    return min(1.0, float(step) / float(warmup_steps))


def _validate_dataset_state_for_current_loss(dataset_state):
    if dataset_state is None:
        return

    mode = dataset_state.get("dataset_mode", None)
    if mode is None:
        return

    if "use_long_sequence" in dataset_state and "is_windowed" in dataset_state:
        use_long_sequence = dataset_state["use_long_sequence"]
        is_windowed = dataset_state["is_windowed"]
        if torch.any(use_long_sequence != is_windowed):
            raise ValueError("Inconsistent dataset_state: use_long_sequence != is_windowed")


def _summarize_dataset_state(dataset_state, ref_tensor):
    """
    Convert per-sample dataset state tensors into scalar diagnostics.

    This records state only; it does not change the objective.
    """
    if dataset_state is None:
        return {}

    out = {}
    for key, value in dataset_state.items():
        if not torch.is_tensor(value):
            continue

        v = value.to(device=ref_tensor.device)
        if v.numel() == 0:
            continue

        v_float = v if torch.is_floating_point(v) else v.float()
        out[f"data_state/{key}_first"] = v.reshape(-1)[0].detach()
        out[f"data_state/{key}_mean"] = v_float.mean().detach()

        if v.numel() > 1:
            out[f"data_state/{key}_min"] = v_float.min().detach()
            out[f"data_state/{key}_max"] = v_float.max().detach()

    return out


def build_normalized_amp_prior(
    f,
    t0,
    amp_scale,
    signal_cfg,
    amp_prior_cfg=None,
    eps=1e-12,
):
    """
    Build the centered complex Gaussian amplitude prior in normalized space.

    The generator defines raw global-time coefficients. Because the decoder
    uses local time, the prior mean is phase-aligned to each sequence t0.
    """
    if f.ndim != 2:
        raise ValueError(f"f must have shape [B, K], got {f.shape}")
    if t0.ndim == 2 and t0.shape[1] == 1:
        t0 = t0.squeeze(1)
    if t0.ndim != 1 or t0.shape[0] != f.shape[0]:
        raise ValueError(f"t0 must have shape [B], got {t0.shape}")
    if amp_scale.ndim != 1 or amp_scale.shape[0] != f.shape[0]:
        raise ValueError(f"amp_scale must have shape [B], got {amp_scale.shape}")

    amp_prior_cfg = amp_prior_cfg or {}
    data_prior_cfg = signal_cfg.get("amp_data_prior", {})
    if data_prior_cfg.get("type", "independent_uniform") not in (
        "independent_uniform",
        "fixed_rho",
    ):
        raise ValueError(
            "Amplitude prior expects signal.amp_data_prior.type to be "
            "'independent_uniform' or 'fixed_rho'"
        )

    device = f.device
    real_dtype = f.dtype
    center_real = torch.as_tensor(
        signal_cfg["amp_real_center_m"],
        device=device,
        dtype=real_dtype,
    )
    center_imag = torch.as_tensor(
        signal_cfg["amp_imag_center_m"],
        device=device,
        dtype=real_dtype,
    )
    if center_real.shape != (f.shape[1],) or center_imag.shape != (f.shape[1],):
        raise ValueError(
            "Amplitude prior center length must match harmonic count: "
            f"{center_real.shape}/{center_imag.shape} vs K={f.shape[1]}"
        )

    relative_half_band = float(
        amp_prior_cfg.get(
            "relative_half_band",
            amp_prior_cfg.get(
                "relative_band",
                data_prior_cfg.get("relative_half_band", 0.2),
            ),
        )
    )
    min_half_band = float(
        amp_prior_cfg.get(
            "min_half_band_m",
            data_prior_cfg.get("min_half_band_m", 1e-5),
        )
    )
    min_half_band_t = torch.as_tensor(min_half_band, device=device, dtype=real_dtype)
    real_half = relative_half_band * torch.maximum(center_real.abs(), min_half_band_t)
    imag_half = relative_half_band * torch.maximum(center_imag.abs(), min_half_band_t)
    amp_prior_var_raw = (real_half.pow(2) + imag_half.pow(2)) / 3.0

    amp_prior_mean_raw = torch.complex(center_real, center_imag).view(1, -1)
    if bool(amp_prior_cfg.get("local_time_align", True)):
        phase_shift = 2.0 * torch.pi * f * t0.view(-1, 1)
        amp_prior_mean_raw = amp_prior_mean_raw * torch.polar(
            torch.ones_like(phase_shift),
            phase_shift,
        )
    else:
        amp_prior_mean_raw = amp_prior_mean_raw.expand(f.shape[0], -1)

    amp_scale = amp_scale.to(device=device, dtype=real_dtype).clamp_min(eps)
    amp_prior_mean = amp_prior_mean_raw / amp_scale.view(-1, 1)
    amp_prior_var = amp_prior_var_raw.view(1, -1) / amp_scale.view(-1, 1).pow(2)

    min_tau2_norm = float(amp_prior_cfg.get("min_tau2_norm", eps))
    amp_prior_var = amp_prior_var.clamp_min(min_tau2_norm)
    return amp_prior_mean, amp_prior_var


def sample_sequence_frequencies(mu_f, std_f, num_samples, freq_lower, freq_upper):
    """
    Sample sequence-level frequency vectors from truncated Gaussian posterior.

    Args:
        mu_f:       [B, K]
        std_f:      [B, K]
        num_samples: S_seq
        freq_lower: [K]
        freq_upper: [K]

    Returns:
        f_samples: [S_seq, B, K]
    """
    return sample_truncated_normal_frequencies(
        mu_f=mu_f,
        std_f=std_f,
        num_samples=num_samples,
        freq_lower=freq_lower,
        freq_upper=freq_upper,
    )


def sample_complex_normal_amplitudes(amp_mu, amp_var, num_samples, eps=1e-8):
    """
    Reparameterized sampling from diagonal circular complex Gaussian:

        q(c|Y) = CN(amp_mu, diag(amp_var))

    Convention:
        amp_var_k = E[ |c_k - mu_k|^2 ]
    """
    if not torch.is_complex(amp_mu):
        raise TypeError(f"amp_mu must be complex, got {amp_mu.dtype}")
    if amp_var.shape != amp_mu.shape:
        raise ValueError(
            f"amp_var shape must match amp_mu: {amp_var.shape} vs {amp_mu.shape}"
        )

    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    real_dtype = amp_mu.real.dtype
    amp_var = amp_var.to(device=amp_mu.device, dtype=real_dtype) + eps

    eps_re = torch.randn(
        num_samples,
        *amp_mu.shape,
        device=amp_mu.device,
        dtype=real_dtype,
    )
    eps_im = torch.randn(
        num_samples,
        *amp_mu.shape,
        device=amp_mu.device,
        dtype=real_dtype,
    )

    scale = torch.sqrt(0.5 * amp_var).unsqueeze(0)
    noise = torch.complex(scale * eps_re, scale * eps_im)

    return amp_mu.unsqueeze(0) + noise


def compute_sequence_posterior_recon_loss(
    y_complex,
    t,
    mu_f,
    std_f,
    model,
    sequence_posterior_samples,
    ridge_lambda,
    f_samples=None,
    noise_var_norm=None,
    include_log_const=False,
    amp_scale=None,
    t0=None,
    signal_cfg=None,
    amp_prior_cfg=None,
    use_posterior_sampling=True,
    normalize_by_num_points=False,
    eps=1e-8,
):
    """
    Reconstruct each sequence from sequence-level posterior frequency samples.
    """
    if f_samples is None:
        if use_posterior_sampling:
            f_samples = sample_sequence_frequencies(
                mu_f=mu_f,
                std_f=std_f,
                num_samples=sequence_posterior_samples,
                freq_lower=model.encoder.freq_lower,
                freq_upper=model.encoder.freq_upper,
            )
        else:
            f_samples = mu_f.unsqueeze(0)
    else:
        if f_samples.ndim != 3:
            raise ValueError(f"f_samples must be [S, B, K], got {f_samples.shape}")
        if f_samples.shape[1:] != mu_f.shape:
            raise ValueError(
                f"f_samples shape {f_samples.shape} does not match mu_f {mu_f.shape}"
            )

    if noise_var_norm is None:
        noise_var = torch.ones(
            y_complex.shape[0],
            device=y_complex.device,
            dtype=y_complex.real.dtype,
        )
    else:
        noise_var = noise_var_norm.to(device=y_complex.device, dtype=y_complex.real.dtype)
        if noise_var.ndim != 1 or noise_var.shape[0] != y_complex.shape[0]:
            raise ValueError(
                f"noise_var_norm must have shape [B], got {noise_var_norm.shape}"
            )
    noise_var = noise_var.clamp_min(eps)

    amp_prior_cfg = amp_prior_cfg or {}
    use_amp_prior = bool(amp_prior_cfg.get("enabled", False))
    amp_mode = amp_prior_cfg.get("mode", "map")
    include_prior_penalty = bool(amp_prior_cfg.get("include_prior_penalty", True))
    if use_amp_prior:
        prior_type = amp_prior_cfg.get(
            "type",
            "centered_complex_gaussian_from_data_uniform",
        )
        if prior_type != "centered_complex_gaussian_from_data_uniform":
            raise ValueError(f"Unsupported amplitude_prior.type={prior_type!r}")
        if signal_cfg is None or amp_scale is None or t0 is None:
            raise ValueError(
                "Stage 3A amplitude prior requires signal_cfg, amp_scale, and t0"
            )
        if amp_mode not in ("map", "marginal_likelihood"):
            raise ValueError(f"Unsupported amplitude_prior.mode={amp_mode!r}")

    y_hat_samples = []
    c_hat_samples = []
    ls_cond_samples = []
    amp_prior_mean_samples = []
    amp_prior_var_samples = []
    amp_lambda_samples = []
    marginal_nll_samples = []
    marginal_quad_samples = []
    marginal_logdet_samples = []
    amp_post_var_trace_samples = []
    amp_post_std_samples = []
    amp_uncertainty_ratio_samples = []

    for s in range(f_samples.shape[0]):
        f_s = f_samples[s]
        if use_amp_prior and amp_mode == "map":
            amp_prior_mean_s, amp_prior_var_s = build_normalized_amp_prior(
                f=f_s,
                t0=t0,
                amp_scale=amp_scale,
                signal_cfg=signal_cfg,
                amp_prior_cfg=amp_prior_cfg,
                eps=eps,
            )
            amp_real_s, amp_imag_s, c_s, cond_s = model.solve_amplitudes_map(
                y_complex=y_complex,
                f=f_s,
                t=t,
                amp_prior_mean=amp_prior_mean_s,
                amp_prior_var=amp_prior_var_s,
                noise_var_norm=noise_var,
                return_condition=True,
                eps=eps,
            )
            amp_prior_mean_samples.append(amp_prior_mean_s)
            amp_prior_var_samples.append(amp_prior_var_s)
            amp_lambda_samples.append(noise_var[:, None] / amp_prior_var_s.clamp_min(eps))
        elif use_amp_prior and amp_mode == "marginal_likelihood":
            amp_prior_mean_s, amp_prior_var_s = build_normalized_amp_prior(
                f=f_s,
                t0=t0,
                amp_scale=amp_scale,
                signal_cfg=signal_cfg,
                amp_prior_cfg=amp_prior_cfg,
                eps=eps,
            )
            nll_s, c_s, post_var_diag_s, bayes_diag_s = model.amplitude_marginal_nll(
                y_complex=y_complex,
                f=f_s,
                t=t,
                amp_prior_mean=amp_prior_mean_s,
                amp_prior_var=amp_prior_var_s,
                noise_var_norm=noise_var,
                include_log_const=include_log_const,
                eps=eps,
            )
            amp_real_s = c_s.real
            amp_imag_s = c_s.imag
            cond_s = bayes_diag_s["bayes_cond"]
            amp_prior_mean_samples.append(amp_prior_mean_s)
            amp_prior_var_samples.append(amp_prior_var_s)
            amp_lambda_samples.append(noise_var[:, None] / amp_prior_var_s.clamp_min(eps))
            marginal_nll_samples.append(nll_s)
            marginal_quad_samples.append(bayes_diag_s["marginal_quad"])
            marginal_logdet_samples.append(bayes_diag_s["marginal_logdet"])
            amp_post_var_trace_samples.append(bayes_diag_s["amp_post_var_trace"])
            amp_post_std_samples.append(bayes_diag_s["amp_post_std_mean"])
            amp_uncertainty_ratio_samples.append(
                bayes_diag_s["amp_uncertainty_to_prior_ratio_mean"]
            )
        else:
            amp_real_s, amp_imag_s, c_s, cond_s = model.solve_amplitudes_ls(
                y_complex=y_complex,
                f=f_s,
                t=t,
                ridge_lambda=ridge_lambda,
                return_condition=True,
            )
        y_hat_s = model.decode(
            amp_real=amp_real_s,
            amp_imag=amp_imag_s,
            f=f_s,
            t=t,
        )

        y_hat_samples.append(y_hat_s)
        c_hat_samples.append(c_s)
        ls_cond_samples.append(cond_s)

    y_hat_samples = torch.stack(y_hat_samples, dim=0)
    c_hat_samples = torch.stack(c_hat_samples, dim=0)
    ls_cond_samples = torch.stack(ls_cond_samples, dim=0)

    assert y_hat_samples.ndim == 3
    assert y_hat_samples.shape[0] == f_samples.shape[0]
    assert c_hat_samples.shape[:2] == f_samples.shape[:2]

    sqerr = torch.abs(y_hat_samples - y_complex.unsqueeze(0)) ** 2
    recon_mse = sqerr.mean()

    recon_nll_core_per_sequence = (
        sqerr / noise_var.view(1, -1, 1)
    ).sum(dim=-1)
    if normalize_by_num_points:
        recon_nll_core_per_sequence = recon_nll_core_per_sequence / y_complex.shape[1]
    recon_nll_core = recon_nll_core_per_sequence.mean()
    log_const = y_complex.shape[1] * torch.log(math.pi * noise_var).mean()
    if normalize_by_num_points:
        log_const = log_const / y_complex.shape[1]
    recon_nll_full = recon_nll_core + log_const

    zero = torch.zeros((), device=y_complex.device, dtype=sqerr.dtype)
    amp_prior_quad = zero
    amp_lambda_mean = zero
    amp_lambda_min = zero
    amp_lambda_max = zero
    amp_prior_var_norm_mean = zero
    marginal_nll = zero
    marginal_quad = zero
    marginal_logdet = zero
    amp_post_var_trace = zero
    amp_post_std_mean = zero
    amp_uncertainty_to_prior_ratio_mean = zero
    if use_amp_prior:
        amp_prior_mean_samples = torch.stack(amp_prior_mean_samples, dim=0)
        amp_prior_var_samples = torch.stack(amp_prior_var_samples, dim=0)
        amp_lambda_samples = torch.stack(amp_lambda_samples, dim=0)
        amp_lambda_mean = amp_lambda_samples.mean()
        amp_lambda_min = amp_lambda_samples.min()
        amp_lambda_max = amp_lambda_samples.max()
        amp_prior_var_norm_mean = amp_prior_var_samples.mean()
        if amp_mode == "map":
            amp_prior_quad_per_sequence = (
                torch.abs(c_hat_samples - amp_prior_mean_samples) ** 2
                / amp_prior_var_samples.clamp_min(eps)
            ).sum(dim=-1)
            if include_prior_penalty:
                amp_prior_quad = amp_prior_quad_per_sequence.mean()
        elif amp_mode == "marginal_likelihood":
            marginal_nll = torch.stack(marginal_nll_samples).mean()
            marginal_quad = torch.stack(marginal_quad_samples).mean()
            marginal_logdet = torch.stack(marginal_logdet_samples).mean()
            amp_post_var_trace = torch.stack(amp_post_var_trace_samples).mean()
            amp_post_std_mean = torch.stack(amp_post_std_samples).mean()
            amp_uncertainty_to_prior_ratio_mean = torch.stack(
                amp_uncertainty_ratio_samples
            ).mean()

    recon_loss_base = recon_nll_full if include_log_const else recon_nll_core
    if use_amp_prior and amp_mode == "marginal_likelihood":
        recon_loss = marginal_nll
        recon_nll_core = marginal_nll
        recon_nll_full = (
            marginal_nll
            if include_log_const
            else marginal_nll + y_complex.shape[1] * math.log(math.pi)
        )
    else:
        recon_loss = recon_loss_base + amp_prior_quad

    amp_norm = torch.linalg.norm(c_hat_samples, dim=-1)
    diagnostics = {
        "f_samples": f_samples,
        "y_hat_samples": y_hat_samples,
        "c_hat_samples": c_hat_samples,
        "recon_mse_sampled": recon_mse,
        "recon_nll": recon_nll_core,
        "recon_nll_full": recon_nll_full,
        "amp_prior_quad": amp_prior_quad,
        "amp_lambda_mean": amp_lambda_mean,
        "amp_lambda_min": amp_lambda_min,
        "amp_lambda_max": amp_lambda_max,
        "amp_prior_var_norm_mean": amp_prior_var_norm_mean,
        "marginal_nll": marginal_nll,
        "marginal_quad": marginal_quad,
        "marginal_logdet": marginal_logdet,
        "amp_post_var_trace": amp_post_var_trace,
        "amp_post_std_mean": amp_post_std_mean,
        "amp_uncertainty_to_prior_ratio_mean": amp_uncertainty_to_prior_ratio_mean,
        "noise_var_norm_mean": noise_var.mean(),
        "noise_var_norm_min": noise_var.min(),
        "noise_var_norm_max": noise_var.max(),
        "freq_sample_std_mean": f_samples.std(dim=0, unbiased=False).mean(),
        "ls_cond_mean": ls_cond_samples.mean(),
        "ls_cond_p95": torch.quantile(ls_cond_samples.reshape(-1), 0.95),
        "ls_amp_norm_mean": amp_norm.mean(),
        "ls_amp_norm_p95": torch.quantile(amp_norm.reshape(-1), 0.95),
        "map_amp_norm_mean": amp_norm.mean(),
        "map_amp_norm_p95": torch.quantile(amp_norm.reshape(-1), 0.95),
    }

    return recon_loss, diagnostics


def compute_sequence_nnamp_recon_loss(
    y_complex,
    t,
    mu_f,
    c_nn,
    model,
    noise_var_norm=None,
    include_log_const=False,
    normalize_by_num_points=True,
    eps=1e-8,
):
    """
    Reconstruct a global sequence using deterministic NN-predicted amplitudes.

    c_nn is in the same normalized domain as y_complex.
    """
    if not torch.is_complex(y_complex):
        raise TypeError(f"y_complex must be complex, got {y_complex.dtype}")
    if not torch.is_complex(c_nn):
        raise TypeError(f"c_nn must be complex, got {c_nn.dtype}")
    if c_nn.shape != mu_f.shape:
        raise ValueError(f"c_nn shape must match mu_f: {c_nn.shape} vs {mu_f.shape}")

    if noise_var_norm is None:
        noise_var = torch.ones(
            y_complex.shape[0],
            device=y_complex.device,
            dtype=y_complex.real.dtype,
        )
    else:
        noise_var = noise_var_norm.to(device=y_complex.device, dtype=y_complex.real.dtype)
        if noise_var.ndim != 1 or noise_var.shape[0] != y_complex.shape[0]:
            raise ValueError(
                f"noise_var_norm must have shape [B], got {noise_var_norm.shape}"
            )
    noise_var = noise_var.clamp_min(eps)

    phi = model.build_dictionary(mu_f, t)
    y_hat = (phi * c_nn.unsqueeze(1)).sum(dim=-1)
    sqerr = torch.abs(y_hat - y_complex) ** 2
    recon_mse = sqerr.mean()

    recon_nll_per_sequence = (
        sqerr / noise_var.view(-1, 1)
    ).sum(dim=-1)
    if normalize_by_num_points:
        recon_nll_per_sequence = recon_nll_per_sequence / y_complex.shape[1]
    recon_nll = recon_nll_per_sequence.mean()

    log_const = y_complex.shape[1] * torch.log(math.pi * noise_var).mean()
    if normalize_by_num_points:
        log_const = log_const / y_complex.shape[1]
    recon_nll_full = recon_nll + log_const
    recon_loss = recon_nll_full if include_log_const else recon_nll

    zero = torch.zeros((), device=y_complex.device, dtype=y_complex.real.dtype)
    amp_norm = torch.linalg.norm(c_nn, dim=-1)
    diagnostics = {
        "f_samples": mu_f.unsqueeze(0),
        "y_hat_samples": y_hat.unsqueeze(0),
        "c_hat_samples": c_nn.unsqueeze(0),
        "recon_mse_sampled": recon_mse,
        "recon_nll": recon_nll,
        "recon_nll_full": recon_nll_full,
        "amp_prior_quad": zero,
        "amp_lambda_mean": zero,
        "amp_lambda_min": zero,
        "amp_lambda_max": zero,
        "amp_prior_var_norm_mean": zero,
        "marginal_nll": zero,
        "marginal_quad": zero,
        "marginal_logdet": zero,
        "amp_post_var_trace": zero,
        "amp_post_std_mean": zero,
        "amp_uncertainty_to_prior_ratio_mean": zero,
        "noise_var_norm_mean": noise_var.mean(),
        "noise_var_norm_min": noise_var.min(),
        "noise_var_norm_max": noise_var.max(),
        "freq_sample_std_mean": zero,
        "ls_cond_mean": zero,
        "ls_cond_p95": zero,
        "ls_amp_norm_mean": zero,
        "ls_amp_norm_p95": zero,
        "map_amp_norm_mean": zero,
        "map_amp_norm_p95": zero,
        "nn_amp_norm_mean": amp_norm.mean(),
        "nn_amp_norm_p95": torch.quantile(amp_norm.reshape(-1), 0.95),
    }

    return recon_loss, diagnostics


def complex_diag_gaussian_kl(mu_q, var_q, mu_p, var_p, eps=1e-8):
    """
    KL between diagonal circular complex Gaussians CN(mu_q, var_q)
    and CN(mu_p, var_p), summed over harmonics per batch item.
    """
    if not torch.is_complex(mu_q) or not torch.is_complex(mu_p):
        raise TypeError("mu_q and mu_p must be complex tensors")
    if var_q.shape != mu_q.shape or var_p.shape != mu_q.shape:
        raise ValueError(
            "Complex Gaussian KL expects var_q/var_p to match mu_q shape: "
            f"{var_q.shape}/{var_p.shape} vs {mu_q.shape}"
        )
    var_q = var_q.to(device=mu_q.device, dtype=mu_q.real.dtype) + eps
    var_p = var_p.to(device=mu_q.device, dtype=mu_q.real.dtype).clamp_min(eps)
    diff2 = torch.abs(mu_q - mu_p.to(device=mu_q.device, dtype=mu_q.dtype)) ** 2
    kl_per_harmonic = torch.log(var_p / var_q) + (var_q + diff2) / var_p - 1.0
    return kl_per_harmonic.sum(dim=-1)


def compute_sequence_bayesian_nnamp_recon_loss(
    y_complex,
    t,
    mu_f,
    amp_mu,
    amp_var,
    model,
    noise_var_norm=None,
    include_log_const=False,
    normalize_by_num_points=True,
    f_samples=None,
    eps=1e-8,
):
    """
    Expected reconstruction NLL under q(c|Y)=CN(amp_mu, diag(amp_var)).

    Frequency is evaluated deterministically at mu_f in this first strict
    NN-amplitude ELBO path; frequency uncertainty is regularized by KL.
    """
    if not torch.is_complex(y_complex):
        raise TypeError(f"y_complex must be complex, got {y_complex.dtype}")
    if not torch.is_complex(amp_mu):
        raise TypeError(f"amp_mu must be complex, got {amp_mu.dtype}")
    if amp_mu.shape != mu_f.shape or amp_var.shape != mu_f.shape:
        raise ValueError(
            "amp_mu/amp_var shapes must match mu_f: "
            f"{amp_mu.shape}/{amp_var.shape} vs {mu_f.shape}"
        )

    if noise_var_norm is None:
        noise_var = torch.ones(
            y_complex.shape[0],
            device=y_complex.device,
            dtype=y_complex.real.dtype,
        )
    else:
        noise_var = noise_var_norm.to(device=y_complex.device, dtype=y_complex.real.dtype)
        if noise_var.ndim != 1 or noise_var.shape[0] != y_complex.shape[0]:
            raise ValueError(
                f"noise_var_norm must have shape [B], got {noise_var_norm.shape}"
            )
    noise_var = noise_var.clamp_min(eps)
    amp_var = amp_var.to(device=y_complex.device, dtype=y_complex.real.dtype) + eps

    if f_samples is None:
        f_samples = mu_f.unsqueeze(0)
    elif f_samples.ndim != 3 or f_samples.shape[1:] != mu_f.shape:
        raise ValueError(f"f_samples must be [S, B, K], got {f_samples.shape}")

    y_hat_samples = []
    expected_sqerr_samples = []
    for s in range(f_samples.shape[0]):
        phi = model.build_dictionary(f_samples[s], t)
        y_hat_s = (phi * amp_mu.unsqueeze(1)).sum(dim=-1)
        sqerr_mean_s = torch.abs(y_hat_s - y_complex) ** 2
        # For unit-magnitude complex exponentials, diag variance contributes
        # sum_k Var[c_k] at every time sample.
        amp_var_contrib = amp_var.sum(dim=-1).view(-1, 1)
        expected_sqerr_samples.append(sqerr_mean_s + amp_var_contrib)
        y_hat_samples.append(y_hat_s)

    y_hat_samples = torch.stack(y_hat_samples, dim=0)
    expected_sqerr = torch.stack(expected_sqerr_samples, dim=0)
    sqerr_mean = torch.abs(y_hat_samples - y_complex.unsqueeze(0)) ** 2
    recon_mse = sqerr_mean.mean()
    expected_recon_mse = expected_sqerr.mean()

    recon_nll_per_sequence = (
        expected_sqerr / noise_var.view(1, -1, 1)
    ).sum(dim=-1)
    if normalize_by_num_points:
        recon_nll_per_sequence = recon_nll_per_sequence / y_complex.shape[1]
    recon_nll = recon_nll_per_sequence.mean()

    log_const = y_complex.shape[1] * torch.log(math.pi * noise_var).mean()
    if normalize_by_num_points:
        log_const = log_const / y_complex.shape[1]
    recon_nll_full = recon_nll + log_const
    recon_loss = recon_nll_full if include_log_const else recon_nll

    zero = torch.zeros((), device=y_complex.device, dtype=y_complex.real.dtype)
    amp_norm = torch.linalg.norm(amp_mu, dim=-1)
    diagnostics = {
        "f_samples": f_samples,
        "y_hat_samples": y_hat_samples,
        "c_hat_samples": amp_mu.unsqueeze(0).expand(f_samples.shape[0], -1, -1),
        "recon_mse_sampled": recon_mse,
        "expected_recon_mse": expected_recon_mse,
        "recon_nll": recon_nll,
        "recon_nll_full": recon_nll_full,
        "amp_prior_quad": zero,
        "amp_lambda_mean": zero,
        "amp_lambda_min": zero,
        "amp_lambda_max": zero,
        "amp_prior_var_norm_mean": zero,
        "marginal_nll": zero,
        "marginal_quad": zero,
        "marginal_logdet": zero,
        "amp_post_var_trace": amp_var.sum(dim=-1).mean(),
        "amp_post_std_mean": torch.sqrt(amp_var).mean(),
        "amp_uncertainty_to_prior_ratio_mean": zero,
        "noise_var_norm_mean": noise_var.mean(),
        "noise_var_norm_min": noise_var.min(),
        "noise_var_norm_max": noise_var.max(),
        "freq_sample_std_mean": zero,
        "ls_cond_mean": zero,
        "ls_cond_p95": zero,
        "ls_amp_norm_mean": zero,
        "ls_amp_norm_p95": zero,
        "map_amp_norm_mean": zero,
        "map_amp_norm_p95": zero,
        "nn_amp_norm_mean": amp_norm.mean(),
        "nn_amp_norm_p95": torch.quantile(amp_norm.reshape(-1), 0.95),
        "nn_amp_var_mean": amp_var.mean(),
        "nn_amp_var_min": amp_var.min(),
        "nn_amp_var_max": amp_var.max(),
    }

    return recon_loss, diagnostics


def compute_sequence_strict_global_elbo_recon_loss(
    y_complex,
    t,
    mu_f,
    std_f,
    amp_mu,
    amp_var,
    model,
    noise_var_norm,
    amp_scale,
    t0,
    signal_cfg,
    amp_prior_cfg,
    num_samples,
    include_log_const=True,
    normalize_by_num_points=False,
    eps=1e-8,
):
    """
    Strict global ELBO reconstruction term for z=(f, c).

    Estimates E_q(f|Y)q(c|Y)[-log p(Y|f,c)] by Monte Carlo sampling both
    frequency and amplitude latents. The likelihood never substitutes a
    posterior mean for f or c.
    """
    if not torch.is_complex(y_complex):
        raise TypeError(f"y_complex must be complex, got {y_complex.dtype}")
    if not torch.is_complex(amp_mu):
        raise TypeError(f"amp_mu must be complex, got {amp_mu.dtype}")
    if amp_mu.shape != mu_f.shape:
        raise ValueError(f"amp_mu shape must match mu_f: {amp_mu.shape} vs {mu_f.shape}")
    if amp_var.shape != mu_f.shape:
        raise ValueError(f"amp_var shape must match mu_f: {amp_var.shape} vs {mu_f.shape}")
    if noise_var_norm is None:
        raise ValueError(
            "Strict complex Gaussian ELBO requires noise_var_norm. It must be "
            "sigma^2 = E[|w|^2] in the normalized signal domain."
        )

    noise_var = noise_var_norm.to(
        device=y_complex.device,
        dtype=y_complex.real.dtype,
    )
    if noise_var.ndim != 1 or noise_var.shape[0] != y_complex.shape[0]:
        raise ValueError(f"noise_var_norm must have shape [B], got {noise_var.shape}")
    noise_var = noise_var.clamp_min(eps)
    amp_var = amp_var.to(device=y_complex.device, dtype=y_complex.real.dtype) + eps

    f_samples = sample_sequence_frequencies(
        mu_f=mu_f,
        std_f=std_f,
        num_samples=num_samples,
        freq_lower=model.encoder.freq_lower,
        freq_upper=model.encoder.freq_upper,
    )
    c_samples = sample_complex_normal_amplitudes(
        amp_mu=amp_mu,
        amp_var=amp_var,
        num_samples=num_samples,
        eps=eps,
    )

    y_hat_samples = []
    sqerr_per_sample = []
    nll_per_sample = []
    seq_len = y_complex.shape[1]

    for s in range(int(num_samples)):
        phi_s = model.build_dictionary(f_samples[s], t)
        y_hat_s = (phi_s * c_samples[s].unsqueeze(1)).sum(dim=-1)
        sqerr_s = torch.abs(y_hat_s - y_complex) ** 2
        nll_s = (sqerr_s / noise_var.view(-1, 1)).sum(dim=-1)
        if include_log_const:
            nll_s = nll_s + seq_len * torch.log(math.pi * noise_var)
        if normalize_by_num_points:
            nll_s = nll_s / seq_len

        y_hat_samples.append(y_hat_s)
        sqerr_per_sample.append(sqerr_s)
        nll_per_sample.append(nll_s)

    y_hat_samples = torch.stack(y_hat_samples, dim=0)
    sqerr_per_sample = torch.stack(sqerr_per_sample, dim=0)
    nll_per_sample = torch.stack(nll_per_sample, dim=0)
    recon_loss = nll_per_sample.mean()

    amp_kl_samples = []
    amp_prior_mean_samples = []
    amp_prior_var_samples = []
    for s in range(int(num_samples)):
        amp_prior_mean_s, amp_prior_var_s = build_normalized_amp_prior(
            f=f_samples[s],
            t0=t0,
            amp_scale=amp_scale,
            signal_cfg=signal_cfg,
            amp_prior_cfg=amp_prior_cfg,
            eps=eps,
        )
        amp_kl_samples.append(
            complex_diag_gaussian_kl(
                mu_q=amp_mu,
                var_q=amp_var,
                mu_p=amp_prior_mean_s,
                var_p=amp_prior_var_s,
                eps=eps,
            )
        )
        amp_prior_mean_samples.append(amp_prior_mean_s)
        amp_prior_var_samples.append(amp_prior_var_s)

    amp_kl_samples = torch.stack(amp_kl_samples, dim=0)
    amp_kl_raw = amp_kl_samples.mean()
    amp_prior_mean_samples = torch.stack(amp_prior_mean_samples, dim=0)
    amp_prior_var_samples = torch.stack(amp_prior_var_samples, dim=0)

    recon_mse_sampled = sqerr_per_sample.mean()
    zero = torch.zeros((), device=y_complex.device, dtype=y_complex.real.dtype)
    amp_norm = torch.linalg.norm(c_samples, dim=-1)

    diagnostics = {
        "f_samples": f_samples,
        "y_hat_samples": y_hat_samples,
        "c_hat_samples": c_samples,
        "recon_mse_sampled": recon_mse_sampled,
        "expected_recon_mse": recon_mse_sampled,
        "recon_nll": recon_loss,
        "recon_nll_full": recon_loss,
        "amp_prior_quad": zero,
        "amp_lambda_mean": zero,
        "amp_lambda_min": zero,
        "amp_lambda_max": zero,
        "amp_prior_var_norm_mean": amp_prior_var_samples.mean(),
        "marginal_nll": zero,
        "marginal_quad": zero,
        "marginal_logdet": zero,
        "amp_post_var_trace": amp_var.sum(dim=-1).mean(),
        "amp_post_std_mean": torch.sqrt(amp_var).mean(),
        "amp_uncertainty_to_prior_ratio_mean": (
            amp_var.unsqueeze(0) / amp_prior_var_samples.clamp_min(eps)
        ).mean(),
        "noise_var_norm_mean": noise_var.mean(),
        "noise_var_norm_min": noise_var.min(),
        "noise_var_norm_max": noise_var.max(),
        "freq_sample_std_mean": f_samples.std(dim=0, unbiased=False).mean(),
        "ls_cond_mean": zero,
        "ls_cond_p95": zero,
        "ls_amp_norm_mean": zero,
        "ls_amp_norm_p95": zero,
        "map_amp_norm_mean": zero,
        "map_amp_norm_p95": zero,
        "nn_amp_norm_mean": amp_norm.mean(),
        "nn_amp_norm_p95": torch.quantile(amp_norm.reshape(-1), 0.95),
        "nn_amp_var_mean": amp_var.mean(),
        "nn_amp_var_min": amp_var.min(),
        "nn_amp_var_max": amp_var.max(),
        "strict_mc_nll_mean": recon_loss,
        "strict_mc_num_samples": torch.as_tensor(
            num_samples,
            device=y_complex.device,
            dtype=y_complex.real.dtype,
        ),
    }

    return recon_loss, amp_kl_raw, diagnostics


def uniform_support_penalty(f_samples, freq_lower, freq_upper):
    if f_samples.ndim != 3:
        raise ValueError(f"f_samples must be [S, B, K], got {f_samples.shape}")

    lower = freq_lower.to(device=f_samples.device, dtype=f_samples.dtype).view(1, 1, -1)
    upper = freq_upper.to(device=f_samples.device, dtype=f_samples.dtype).view(1, 1, -1)

    below = torch.relu(lower - f_samples)
    above = torch.relu(f_samples - upper)
    width = upper - lower

    penalty = ((below + above) / (width + 1e-12)).pow(2).mean()
    outside = (f_samples < lower) | (f_samples > upper)
    outside_rate = outside.float().mean()

    return penalty, outside_rate


def _static_global_amp_prior_cfg(loss_cfg):
    elbo_cfg = loss_cfg.get("elbo", {})
    mode = elbo_cfg.get("mode", "static_global_bayesianls")
    amp_prior_cfg = dict(loss_cfg.get("amplitude_prior", {}))

    if mode in ("static_global_ls", "static_global_nnamp"):
        amp_prior_cfg["enabled"] = False
    elif mode == "static_global_bayesian_nnamp":
        amp_prior_cfg["enabled"] = True
        amp_prior_cfg["mode"] = "nnamp_kl"
        amp_prior_cfg["include_prior_penalty"] = False
    elif mode == "static_global_strict_elbo":
        amp_prior_cfg["enabled"] = True
        amp_prior_cfg["mode"] = "strict_nnamp_kl"
        amp_prior_cfg["include_prior_penalty"] = False
    elif mode == "static_global_mapls":
        amp_prior_cfg["enabled"] = True
        amp_prior_cfg["mode"] = "map"
        amp_prior_cfg["include_prior_penalty"] = True
    elif mode == "static_global_bayesianls":
        amp_prior_cfg["enabled"] = True
        amp_prior_cfg["mode"] = "marginal_likelihood"
        amp_prior_cfg["include_prior_penalty"] = False
    else:
        raise ValueError(
            "loss.elbo.mode must be one of "
            "'static_global_ls', 'static_global_nnamp', "
            "'static_global_bayesian_nnamp', 'static_global_strict_elbo', "
            "'static_global_mapls', 'static_global_bayesianls'; "
            f"got {mode!r}"
        )

    return mode, amp_prior_cfg


def _slice_windows_for_objective(
    target_windows,
    t_windows_abs,
    objective_cycles=None,
    short_num_cycles=None,
    segment_mode="prefix",
):
    """
    Slice parent windows according to the current objective length.
    """
    if objective_cycles is None:
        return target_windows, t_windows_abs, target_windows.shape[1]

    if short_num_cycles is None or int(short_num_cycles) <= 0:
        raise ValueError(
            "short_num_cycles must be provided and positive when objective_cycles is used"
        )
    if segment_mode != "prefix":
        raise ValueError(
            "Only segment_mode='prefix' is supported in the first implementation, "
            f"got {segment_mode!r}"
        )

    num_windows = target_windows.shape[1]
    num_objective_windows = max(
        1,
        int(math.ceil(int(objective_cycles) / int(short_num_cycles))),
    )
    num_objective_windows = min(num_objective_windows, num_windows)

    return (
        target_windows[:, :num_objective_windows],
        t_windows_abs[:, :num_objective_windows],
        num_objective_windows,
    )


def compute_frequency_kl(mu_f, std_f, model, loss_cfg):
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
        prior_mean = prior_cfg.get("mean", "center")
        if prior_mean != "center":
            raise ValueError(f"Unsupported loss prior mean={prior_mean!r}")
        prior_std_ratio = float(prior_cfg.get("std_ratio_to_half_band", 0.5))
        prior_mu_f = model.encoder.freq_mid
        prior_std_f = prior_std_ratio * model.encoder.freq_half
        return kl_trunc_normal_trunc_normal(
            mu_q=mu_f,
            std_q=std_f,
            mu_p=prior_mu_f,
            std_p=prior_std_f,
            lower=model.encoder.freq_lower,
            upper=model.encoder.freq_upper,
        )
    raise ValueError(f"Unsupported loss.kl.type={kl_type!r}")


def _select_amp_warmup_frequency(
    source: str,
    mu_f: torch.Tensor,
    model,
    true_freq_hz: torch.Tensor = None,
    detach: bool = True,
) -> torch.Tensor:
    if source == "mu_f":
        f_amp = mu_f
    elif source == "center":
        f_amp = model.encoder.freq_mid.to(device=mu_f.device, dtype=mu_f.dtype).view(
            1,
            -1,
        ).expand_as(mu_f)
    elif source == "oracle":
        if true_freq_hz is None:
            raise ValueError(
                "amplitude_warmup.frequency_source='oracle' requires true_freq_hz"
            )
        f_amp = true_freq_hz.to(
            device=mu_f.device,
            dtype=mu_f.dtype,
        )
        if f_amp.shape != mu_f.shape:
            raise ValueError(
                f"true_freq_hz shape {f_amp.shape} must match mu_f {mu_f.shape}"
            )
    else:
        raise ValueError(
            "amplitude_warmup.frequency_source must be one of "
            "'mu_f', 'center', 'oracle'"
        )

    return f_amp.detach() if detach else f_amp


def _complex_global_to_local(
    c_global: torch.Tensor,
    f_hz: torch.Tensor,
    t0: torch.Tensor,
) -> torch.Tensor:
    phase_shift = 2.0 * torch.pi * f_hz * t0[:, None]
    return c_global * torch.exp(1j * phase_shift)


def _complex_local_to_global(
    c_local: torch.Tensor,
    f_hz: torch.Tensor,
    t0: torch.Tensor,
) -> torch.Tensor:
    phase_shift = 2.0 * torch.pi * f_hz * t0[:, None]
    return c_local * torch.exp(-1j * phase_shift)


def _select_c_nn_for_local_loss(
    model_outputs,
    model,
    f_amp: torch.Tensor,
    t0: torch.Tensor,
) -> torch.Tensor:
    c_nn = model_outputs["c_nn"]
    representation = getattr(model, "amplitude_nn_representation", "segment_local")
    if representation == "segment_local":
        return c_nn
    if representation == "parent_global":
        return _complex_global_to_local(c_global=c_nn, f_hz=f_amp, t0=t0)
    raise ValueError(
        "model.amplitude_nn_representation must be one of "
        f"'segment_local', 'parent_global'; got {representation!r}"
    )


def compute_static_global_objective(
    target_windows,
    t_windows_abs,
    model_outputs,
    model,
    loss_cfg,
    noise_var_norm=None,
    amp_scale=None,
    signal_cfg=None,
    dataset_state=None,
    true_freq_hz=None,
    global_step=None,
    objective_cycles=None,
    short_num_cycles=None,
    segment_mode="prefix",
):
    """
    Strict static global-latent objective for parent long sequences.

    Args:
        target_windows: [B, M, L, 2]
        t_windows_abs: [B, M, L]
        model_outputs: dict with one global mu_f/std_f per parent, [B, K]
    """
    _validate_dataset_state_for_current_loss(dataset_state)

    if target_windows.ndim != 4 or target_windows.shape[-1] != 2:
        raise ValueError(
            f"target_windows must have shape [B, M, L, 2], got {target_windows.shape}"
        )
    if t_windows_abs.shape != target_windows.shape[:3]:
        raise ValueError(
            "t_windows_abs shape must match target_windows[:3]: "
            f"{t_windows_abs.shape} vs {target_windows.shape[:3]}"
        )

    mode, amp_prior_cfg = _static_global_amp_prior_cfg(loss_cfg)
    mu_f = model_outputs["mu_f"]
    std_f = model_outputs["std_f"]

    target_windows, t_windows_abs, num_objective_windows = _slice_windows_for_objective(
        target_windows=target_windows,
        t_windows_abs=t_windows_abs,
        objective_cycles=objective_cycles,
        short_num_cycles=short_num_cycles,
        segment_mode=segment_mode,
    )

    batch_size, num_windows, seq_len, _ = target_windows.shape
    t0 = t_windows_abs[:, 0, 0]
    t_global = (t_windows_abs - t0.view(batch_size, 1, 1)).reshape(
        batch_size,
        num_windows * seq_len,
    )
    y_complex = torch.complex(
        target_windows[..., 0],
        target_windows[..., 1],
    ).reshape(batch_size, num_windows * seq_len)

    rec_cfg = loss_cfg.get("reconstruction", {})
    s_seq = int(rec_cfg.get("sequence_posterior_samples", 1))
    include_log_const = bool(rec_cfg.get("include_log_const", False))
    use_posterior_sampling = bool(rec_cfg.get("use_posterior_sampling", True))
    normalize_by_num_points = bool(rec_cfg.get("normalize_by_num_points", False))
    amp_warmup_cfg = loss_cfg.get("amplitude_warmup", {})
    amp_warmup_enabled = bool(amp_warmup_cfg.get("enabled", False))
    amp_freq_source = amp_warmup_cfg.get("frequency_source", "mu_f")
    detach_amp_frequency = bool(amp_warmup_cfg.get("detach_frequency", True))
    if amp_warmup_enabled:
        f_amp = _select_amp_warmup_frequency(
            source=amp_freq_source,
            mu_f=mu_f,
            model=model,
            true_freq_hz=true_freq_hz,
            detach=detach_amp_frequency,
        )
    else:
        f_amp = mu_f
    c_nn_local_for_loss = None
    if "c_nn" in model_outputs:
        c_nn_local_for_loss = _select_c_nn_for_local_loss(
            model_outputs=model_outputs,
            model=model,
            f_amp=f_amp,
            t0=t0,
        )
    amp_kl_raw = torch.zeros((), device=mu_f.device, dtype=mu_f.dtype)
    if mode == "static_global_strict_elbo":
        for required_key in ("c_nn", "amp_var_nn"):
            if required_key not in model_outputs:
                raise KeyError(
                    f"static_global_strict_elbo requires model_outputs[{required_key!r}]"
                )
        if signal_cfg is None or amp_scale is None:
            raise ValueError(
                "static_global_strict_elbo requires signal_cfg and amp_scale "
                "to build p(c|f)."
            )
        if not include_log_const:
            raise ValueError(
                "Strict complex Gaussian ELBO should use include_log_const=True."
            )
        if normalize_by_num_points:
            raise ValueError(
                "Strict sequence ELBO should use normalize_by_num_points=False. "
                "If per-point logging is needed, compute it only as a diagnostic."
            )
        if not use_posterior_sampling:
            raise ValueError(
                "static_global_strict_elbo requires use_posterior_sampling=True. "
                "Do not replace q(f|Y) by mu_f."
            )

        recon_loss, amp_kl_raw, recon_diag = (
            compute_sequence_strict_global_elbo_recon_loss(
                y_complex=y_complex,
                t=t_global,
                mu_f=mu_f,
                std_f=std_f,
                amp_mu=model_outputs["c_nn"],
                amp_var=model_outputs["amp_var_nn"],
                model=model,
                noise_var_norm=noise_var_norm,
                amp_scale=amp_scale,
                t0=t0,
                signal_cfg=signal_cfg,
                amp_prior_cfg=amp_prior_cfg,
                num_samples=s_seq,
                include_log_const=include_log_const,
                normalize_by_num_points=normalize_by_num_points,
            )
        )
    elif mode == "static_global_bayesian_nnamp":
        for required_key in ("c_nn", "amp_var_nn"):
            if required_key not in model_outputs:
                raise KeyError(
                    f"static_global_bayesian_nnamp requires model_outputs[{required_key!r}]"
                )
        if signal_cfg is None or amp_scale is None:
            raise ValueError(
                "static_global_bayesian_nnamp requires signal_cfg and amp_scale "
                "to build the amplitude prior"
            )
        amp_prior_mean, amp_prior_var = build_normalized_amp_prior(
            f=mu_f,
            t0=t0,
            amp_scale=amp_scale,
            signal_cfg=signal_cfg,
            amp_prior_cfg=loss_cfg.get("amplitude_prior", {}),
        )
        if use_posterior_sampling:
            f_recon_samples = sample_sequence_frequencies(
                mu_f=mu_f,
                std_f=std_f,
                num_samples=s_seq,
                freq_lower=model.encoder.freq_lower,
                freq_upper=model.encoder.freq_upper,
            )
        else:
            f_recon_samples = mu_f.unsqueeze(0)
        recon_loss, recon_diag = compute_sequence_bayesian_nnamp_recon_loss(
            y_complex=y_complex,
            t=t_global,
            mu_f=mu_f,
            amp_mu=model_outputs["c_nn"],
            amp_var=model_outputs["amp_var_nn"],
            model=model,
            noise_var_norm=noise_var_norm,
            include_log_const=include_log_const,
            normalize_by_num_points=normalize_by_num_points,
            f_samples=f_recon_samples,
        )
        amp_kl_per_item = complex_diag_gaussian_kl(
            mu_q=model_outputs["c_nn"],
            var_q=model_outputs["amp_var_nn"],
            mu_p=amp_prior_mean,
            var_p=amp_prior_var,
        )
        amp_kl_raw = amp_kl_per_item.mean()
        recon_diag["amp_prior_var_norm_mean"] = amp_prior_var.mean()
        recon_diag["amp_uncertainty_to_prior_ratio_mean"] = (
            model_outputs["amp_var_nn"] / amp_prior_var.clamp_min(1e-8)
        ).mean()
    elif mode == "static_global_nnamp":
        if "c_nn" not in model_outputs:
            raise KeyError("static_global_nnamp requires model_outputs['c_nn']")
        recon_loss, recon_diag = compute_sequence_nnamp_recon_loss(
            y_complex=y_complex,
            t=t_global,
            mu_f=f_amp,
            c_nn=c_nn_local_for_loss,
            model=model,
            noise_var_norm=noise_var_norm,
            include_log_const=include_log_const,
            normalize_by_num_points=normalize_by_num_points,
        )
    else:
        recon_loss, recon_diag = compute_sequence_posterior_recon_loss(
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
            t0=t0,
            signal_cfg=signal_cfg,
            amp_prior_cfg=amp_prior_cfg,
            use_posterior_sampling=use_posterior_sampling,
            normalize_by_num_points=normalize_by_num_points,
        )

    freq_kl_per_item = compute_frequency_kl(
        mu_f=mu_f,
        std_f=std_f,
        model=model,
        loss_cfg=loss_cfg,
    )
    freq_kl_raw = freq_kl_per_item.sum(dim=-1).mean()
    _, outside_rate = uniform_support_penalty(
        f_samples=recon_diag["f_samples"],
        freq_lower=model.encoder.freq_lower,
        freq_upper=model.encoder.freq_upper,
    )

    recon_weight = float(loss_cfg.get("reconstruction_weight", 1.0))
    beta_freq = float(loss_cfg.get("beta_freq", 1.0))
    beta_anneal = compute_beta_anneal(loss_cfg=loss_cfg, step=global_step)
    freq_kl_weighted = beta_anneal * freq_kl_raw
    amp_kl_cfg = loss_cfg.get("amplitude_kl", {})
    beta_amp = float(amp_kl_cfg.get("beta_amp", amp_kl_cfg.get("beta", 1.0)))
    amp_kl_enabled = bool(
        amp_kl_cfg.get(
            "enabled",
            mode in ("static_global_bayesian_nnamp", "static_global_strict_elbo"),
        )
    )
    amp_kl_weighted = amp_kl_raw if amp_kl_enabled else torch.zeros_like(amp_kl_raw)
    amp_sup_cfg = loss_cfg.get("amp_supervision", {})
    amp_sup_enabled = bool(amp_sup_cfg.get("enabled", False))
    amp_sup_weight = float(amp_sup_cfg.get("weight", 1.0))
    eps = float(amp_sup_cfg.get("eps", 1e-8))
    amp_sup_loss_raw = torch.zeros((), device=mu_f.device, dtype=mu_f.dtype)
    amp_sup_target_norm = torch.zeros((), device=mu_f.device, dtype=mu_f.dtype)
    amp_sup_error_norm = torch.zeros((), device=mu_f.device, dtype=mu_f.dtype)
    if amp_sup_enabled:
        if amp_sup_cfg.get("target", "ls_at_mu_f") != "ls_at_mu_f":
            raise ValueError(
                "loss.amp_supervision.target currently only supports 'ls_at_mu_f'"
            )
        if "c_nn" not in model_outputs:
            raise KeyError("amp_supervision requires model_outputs['c_nn']")
        _, _, amp_sup_target, _ = model.solve_amplitudes_ls(
            y_complex=y_complex,
            f=f_amp,
            t=t_global,
            ridge_lambda=model.ls_ridge,
            return_condition=True,
        )
        if bool(amp_sup_cfg.get("stop_gradient_target", True)):
            amp_sup_target = amp_sup_target.detach()
        amp_sup_error = c_nn_local_for_loss - amp_sup_target
        err2 = torch.sum(torch.abs(amp_sup_error) ** 2, dim=-1)
        ref2 = torch.sum(torch.abs(amp_sup_target.detach()) ** 2, dim=-1).clamp_min(
            eps
        )
        amp_sup_loss_raw = (err2 / ref2).mean()
        amp_sup_target_norm = torch.linalg.norm(amp_sup_target, dim=-1).mean()
        amp_sup_error_norm = torch.linalg.norm(amp_sup_error, dim=-1).mean()
    amp_sup_weighted = amp_sup_weight * amp_sup_loss_raw
    loss_unscaled = (
        recon_weight * recon_loss
        + amp_sup_weighted
        + beta_freq * freq_kl_weighted
        + beta_amp * amp_kl_weighted
    )
    objective_num_points = max(int(num_windows * seq_len), 1)
    scale_modes = {
        "static_global_strict_elbo",
        "static_global_nnamp",
        "static_global_bayesian_nnamp",
    }
    optimization_loss_scale = (
        1.0 / float(objective_num_points)
        if mode in scale_modes and not normalize_by_num_points
        else 1.0
    )
    loss = loss_unscaled * optimization_loss_scale

    diagnostics = {
        "loss": loss_unscaled.detach(),
        "optimization_loss": loss.detach(),
        "loss_scale": torch.as_tensor(
            optimization_loss_scale,
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "recon_loss": recon_loss.detach(),
        "reconstruction_weight": torch.as_tensor(
            recon_weight,
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "recon_loss_per_point": (recon_loss * optimization_loss_scale).detach(),
        "freq_kl": freq_kl_weighted.detach(),
        "freq_kl_raw": freq_kl_raw.detach(),
        "amp_kl": amp_kl_weighted.detach(),
        "amp_kl_raw": amp_kl_raw.detach(),
        "amp_supervision_loss": amp_sup_weighted.detach(),
        "amp_supervision_loss_raw": amp_sup_loss_raw.detach(),
        "amp_supervision_weight": torch.as_tensor(
            amp_sup_weight,
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "amp_supervision_target_norm_mean": amp_sup_target_norm.detach(),
        "amp_supervision_error_norm_mean": amp_sup_error_norm.detach(),
        "amp_supervision_frequency_source_id": torch.as_tensor(
            {"mu_f": 0, "center": 1, "oracle": 2}[amp_freq_source],
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "beta_amp": torch.as_tensor(
            beta_amp,
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "freq_kl_beta_anneal": torch.as_tensor(
            beta_anneal,
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "freq_kl_per_harmonic_mean": freq_kl_per_item.mean(dim=0).detach(),
        "freq_prior_reg": freq_kl_weighted.detach(),
        "posterior_std_hz_mean": std_f.mean().detach(),
        "freq_sample_outside_rate": outside_rate.detach(),
        "static_global_mode": mode,
        "nn_amp_norm_mean": torch.zeros(
            (),
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "nn_amp_norm_p95": torch.zeros(
            (),
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "objective_cycles": torch.as_tensor(
            -1 if objective_cycles is None else int(objective_cycles),
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "objective_num_windows": torch.as_tensor(
            int(num_objective_windows),
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "objective_num_points": torch.as_tensor(
            objective_num_points,
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
    }
    if mode == "static_global_nnamp":
        diagnostics["nn_amp_norm_mean"] = recon_diag["nn_amp_norm_mean"].detach()
        diagnostics["nn_amp_norm_p95"] = recon_diag["nn_amp_norm_p95"].detach()

    if "log_rho2_f" in model_outputs:
        diagnostics["log_rho2_f_mean"] = model_outputs["log_rho2_f"].mean().detach()
        diagnostics["log_rho2_f_min"] = model_outputs["log_rho2_f"].min().detach()
        diagnostics["log_rho2_f_max"] = model_outputs["log_rho2_f"].max().detach()
    if "attn_weights" in model_outputs:
        attn = model_outputs["attn_weights"]
        diagnostics["attn_max_mean"] = attn.max(dim=1).values.mean().detach()
        diagnostics["attn_entropy_mean"] = (
            -(attn * torch.log(attn.clamp_min(1e-12))).sum(dim=1).mean()
        ).detach()

    diagnostics.update(
        {
            k: v.detach() if torch.is_tensor(v) else v
            for k, v in recon_diag.items()
            if k
            not in (
                "f_samples",
                "y_hat_samples",
                "c_hat_samples",
                "freq_sample_std_mean",
            )
        }
    )
    diagnostics.update(
        _summarize_dataset_state(
            dataset_state=dataset_state,
            ref_tensor=mu_f,
        )
    )

    return loss, recon_loss, freq_kl_weighted, diagnostics


def diag_gaussian_kl(mu_q, logvar_q, mu_p=0.0, var_p=1.0, eps=1e-8):
    var_q = torch.exp(logvar_q).clamp_min(eps)
    if torch.is_tensor(mu_p):
        mu_p = mu_p.to(device=mu_q.device, dtype=mu_q.dtype)
    if torch.is_tensor(var_p):
        var_p = var_p.to(device=mu_q.device, dtype=mu_q.dtype).clamp_min(eps)
    else:
        var_p = max(float(var_p), eps)
    return 0.5 * (
        torch.log(torch.as_tensor(var_p, device=mu_q.device, dtype=mu_q.dtype))
        - torch.log(var_q)
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    ).sum(dim=-1)


def complex_sequence_nll(
    y_hat,
    target_ri,
    noise_var_norm,
    include_log_const=True,
    normalize_by_num_points=True,
    eps=1e-8,
):
    if not torch.is_complex(y_hat):
        raise TypeError(f"y_hat must be complex, got {y_hat.dtype}")
    if target_ri.shape != y_hat.shape + (2,):
        raise ValueError(
            f"target_ri shape {target_ri.shape} must equal y_hat shape {y_hat.shape} + (2,)"
        )
    target = torch.complex(target_ri[..., 0], target_ri[..., 1])
    sigma2 = noise_var_norm.to(device=y_hat.device, dtype=y_hat.real.dtype).clamp_min(eps)
    view_shape = (sigma2.shape[0],) + (1,) * (y_hat.ndim - 1)
    sigma2_view = sigma2.view(view_shape)
    residual2 = torch.abs(target - y_hat) ** 2
    nll_per_item = (residual2 / sigma2_view).reshape(y_hat.shape[0], -1).sum(dim=-1)
    num_points = residual2[0].numel()
    if include_log_const:
        nll_per_item = nll_per_item + num_points * (
            torch.log(sigma2) + math.log(math.pi)
        )
    nll = nll_per_item.mean()
    if normalize_by_num_points:
        nll = nll / float(max(num_points, 1))
    mse = residual2.mean()
    return nll, mse, nll_per_item.detach()


def compute_sequential_dsae_elbo(
    batch,
    outputs,
    model,
    loss_cfg,
    global_step=None,
):
    """
    Sequential-DSAE negative ELBO for BTT patch sequences.
    """
    rec_cfg = loss_cfg.get("reconstruction", {})
    recon_nll, recon_mse, nll_per_item = complex_sequence_nll(
        y_hat=outputs["y_hat"],
        target_ri=batch["y"],
        noise_var_norm=batch["noise_var_norm"],
        include_log_const=bool(rec_cfg.get("include_log_const", True)),
        normalize_by_num_points=bool(rec_cfg.get("normalize_by_num_points", True)),
    )

    freq_kl_per_item = compute_frequency_kl(
        mu_f=outputs["mu_f"],
        std_f=outputs["std_f"],
        model=model,
        loss_cfg=loss_cfg,
    )
    freq_kl_raw = freq_kl_per_item.sum(dim=-1).mean()
    beta_anneal = compute_beta_anneal(loss_cfg=loss_cfg, step=global_step)

    logamp_prior_mean = float(loss_cfg.get("logamp_prior_mean", 0.0))
    logamp_prior_var = float(loss_cfg.get("logamp_prior_var", 1.0))
    amp_kl_per_item = diag_gaussian_kl(
        outputs["logamp_mu"],
        outputs["logamp_logvar"],
        mu_p=logamp_prior_mean,
        var_p=logamp_prior_var,
    )   # [N]，已对 K 求和
    amp_kl_raw = amp_kl_per_item.mean()

    seq_cfg = loss_cfg.get("sequential", {})
    phase_distribution = seq_cfg.get("phase_distribution", "unwrapped_gaussian")
    if phase_distribution != "unwrapped_gaussian":
        raise ValueError(
            "Only sequential.phase_distribution='unwrapped_gaussian' is implemented. "
            "Von Mises phase posteriors require circular reparameterization and KL."
        )
    z1_prior_var = float(seq_cfg.get("z1_prior_var", loss_cfg.get("z1_prior_var", 1.0)))
    delta_prior_var = float(
        seq_cfg.get("delta_prior_var", loss_cfg.get("delta_prior_var", 1e-2))
    )
    z1_kl_raw = diag_gaussian_kl(
        outputs["z1_mu"],
        outputs["z1_logvar"],
        var_p=z1_prior_var,
    ).mean()

    if outputs["delta_mu"].numel() == 0:
        delta_kl_raw = torch.zeros_like(z1_kl_raw)
    else:
        N, steps = outputs["delta_mu"].shape[0], outputs["delta_mu"].shape[1]
        delta_kl_per_step = diag_gaussian_kl(
            outputs["delta_mu"].reshape(-1, model.num_harmonics),
            outputs["delta_logvar"].reshape(-1, model.num_harmonics),
            var_p=delta_prior_var,
        )                                   # [N*steps]，已对 K 求和
        delta_kl_raw = delta_kl_per_step.reshape(N, steps).sum(dim=1).mean()

    recon_weight = float(loss_cfg.get("reconstruction_weight", 1.0))
    beta_freq = float(loss_cfg.get("beta_freq", 1.0))
    beta_amp = float(loss_cfg.get("beta_amp", loss_cfg.get("amplitude_kl", {}).get("beta_amp", 1.0)))
    beta_z1 = float(loss_cfg.get("beta_z1", 1.0))
    beta_delta = float(loss_cfg.get("beta_delta", 1.0))

    freq_kl = beta_anneal * freq_kl_raw
    loss = (
        recon_weight * recon_nll
        + beta_freq * freq_kl
        + beta_amp * amp_kl_raw
        + beta_z1 * z1_kl_raw
        + beta_delta * delta_kl_raw
    )

    if outputs["z_seq"].shape[1] > 1:
        residual = (
            outputs["z_seq"][:, 1:]
            - outputs["z_seq"][:, :-1]
            - 2.0
            * math.pi
            * outputs["f_sample"].unsqueeze(1)
            * batch["delta_s"][:, 1:].unsqueeze(-1).to(outputs["z_seq"].device)
        )
        transition_residual = residual.abs().mean()
    else:
        transition_residual = torch.zeros_like(recon_nll)

    diagnostics = {
        "loss": loss.detach(),
        "optimization_loss": loss.detach(),
        "recon_loss": recon_nll.detach(),
        "recon_nll": recon_nll.detach(),
        "recon_mse_sampled": recon_mse.detach(),
        "freq_kl": freq_kl.detach(),
        "freq_kl_raw": freq_kl_raw.detach(),
        "amp_kl": amp_kl_raw.detach(),
        "amp_kl_raw": amp_kl_raw.detach(),
        "z1_kl": z1_kl_raw.detach(),
        "delta_kl": delta_kl_raw.detach(),
        "posterior_std_hz_mean": outputs["std_f"].mean().detach(),
        "amp_post_std_mean": torch.exp(0.5 * outputs["logamp_logvar"]).mean().detach(),
        "phase_innovation_std_mean": (
            torch.exp(0.5 * outputs["delta_logvar"]).mean().detach()
            if outputs["delta_logvar"].numel() > 0
            else torch.zeros_like(recon_nll).detach()
        ),
        "transition_residual_mean": transition_residual.detach(),
        "freq_kl_beta_anneal": torch.as_tensor(
            beta_anneal,
            device=loss.device,
            dtype=loss.dtype,
        ).detach(),
        "nll_per_item_mean": nll_per_item.mean().to(loss.device).detach(),
    }
    return loss, recon_nll, freq_kl, diagnostics
