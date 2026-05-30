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
    eps=1e-8,
):
    """
    Reconstruct each sequence from sequence-level posterior frequency samples.
    """
    if f_samples is None:
        f_samples = sample_sequence_frequencies(
            mu_f=mu_f,
            std_f=std_f,
            num_samples=sequence_posterior_samples,
            freq_lower=model.encoder.freq_lower,
            freq_upper=model.encoder.freq_upper,
        )
    else:
        if f_samples.ndim != 3:
            raise ValueError(f"f_samples must be [S, B, K], got {f_samples.shape}")
        if f_samples.shape[1:] != mu_f.shape:
            raise ValueError(
                f"f_samples shape {f_samples.shape} does not match mu_f {mu_f.shape}"
            )

    y_hat_samples = []
    c_hat_samples = []
    ls_cond_samples = []

    for s in range(f_samples.shape[0]):
        f_s = f_samples[s]
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

    if noise_var_norm is None:
        noise_var = torch.ones(
            y_complex.shape[0],
            device=y_complex.device,
            dtype=sqerr.dtype,
        )
    else:
        noise_var = noise_var_norm.to(device=y_complex.device, dtype=sqerr.dtype)
        if noise_var.ndim != 1 or noise_var.shape[0] != y_complex.shape[0]:
            raise ValueError(
                f"noise_var_norm must have shape [B], got {noise_var_norm.shape}"
            )
    noise_var = noise_var.clamp_min(eps)

    recon_nll_core_per_sequence = (
        sqerr / noise_var.view(1, -1, 1)
    ).sum(dim=-1)
    recon_nll_core = recon_nll_core_per_sequence.mean()
    log_const = y_complex.shape[1] * torch.log(math.pi * noise_var).mean()
    recon_nll_full = recon_nll_core + log_const
    recon_loss = recon_nll_full if include_log_const else recon_nll_core

    amp_norm = torch.linalg.norm(c_hat_samples, dim=-1)
    diagnostics = {
        "f_samples": f_samples,
        "y_hat_samples": y_hat_samples,
        "c_hat_samples": c_hat_samples,
        "recon_mse_sampled": recon_mse,
        "recon_nll": recon_nll_core,
        "recon_nll_full": recon_nll_full,
        "noise_var_norm_mean": noise_var.mean(),
        "noise_var_norm_min": noise_var.min(),
        "noise_var_norm_max": noise_var.max(),
        "freq_sample_std_mean": f_samples.std(dim=0, unbiased=False).mean(),
        "ls_cond_mean": ls_cond_samples.mean(),
        "ls_cond_p95": torch.quantile(ls_cond_samples.reshape(-1), 0.95),
        "ls_amp_norm_mean": amp_norm.mean(),
        "ls_amp_norm_p95": torch.quantile(amp_norm.reshape(-1), 0.95),
    }

    return recon_loss, diagnostics


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


def compute_harmonic_loss(
    x_target,
    model_outputs,
    model,
    t,
    loss_cfg,
    noise_var_norm=None,
    global_step=None,
):
    """
    Args:
        x_target: [B, L, 2]
        model_outputs: dict with mu_f/std_f/logvar_f
        t: [B, L]
    """
    mu_f = model_outputs["mu_f"]
    std_f = model_outputs["std_f"]

    y_complex = torch.complex(x_target[..., 0], x_target[..., 1])

    rec_cfg = loss_cfg.get("reconstruction", {})
    s_seq = int(rec_cfg.get("sequence_posterior_samples", 1))
    include_log_const = bool(rec_cfg.get("include_log_const", False))
    recon_loss, recon_diag = compute_sequence_posterior_recon_loss(
        y_complex=y_complex,
        t=t,
        mu_f=mu_f,
        std_f=std_f,
        model=model,
        sequence_posterior_samples=s_seq,
        ridge_lambda=model.ls_ridge,
        noise_var_norm=noise_var_norm,
        include_log_const=include_log_const,
    )

    kl_cfg = loss_cfg.get("kl", {})
    kl_type = kl_cfg.get("type", "trunc_normal_to_trunc_normal")
    if kl_type == "trunc_normal_to_uniform":
        freq_kl_per_item = kl_trunc_normal_uniform(
            mu_f=mu_f,
            std_f=std_f,
            freq_lower=model.encoder.freq_lower,
            freq_upper=model.encoder.freq_upper,
        )
    elif kl_type == "trunc_normal_to_trunc_normal":
        prior_cfg = loss_cfg.get("prior", loss_cfg.get("loss_prior", {}))
        prior_mean = prior_cfg.get("mean", "center")
        if prior_mean != "center":
            raise ValueError(f"Unsupported loss prior mean={prior_mean!r}")
        prior_std_ratio = float(prior_cfg.get("std_ratio_to_half_band", 0.5))
        prior_mu_f = model.encoder.freq_mid
        prior_std_f = prior_std_ratio * model.encoder.freq_half
        freq_kl_per_item = kl_trunc_normal_trunc_normal(
            mu_q=mu_f,
            std_q=std_f,
            mu_p=prior_mu_f,
            std_p=prior_std_f,
            lower=model.encoder.freq_lower,
            upper=model.encoder.freq_upper,
        )
    else:
        raise ValueError(f"Unsupported loss.kl.type={kl_type!r}")
    freq_kl_raw = freq_kl_per_item.sum(dim=-1).mean()

    _, outside_rate = uniform_support_penalty(
        f_samples=recon_diag["f_samples"],
        freq_lower=model.encoder.freq_lower,
        freq_upper=model.encoder.freq_upper,
    )

    beta_freq = float(loss_cfg.get("beta_freq", 1.0))
    beta_anneal = compute_beta_anneal(loss_cfg=loss_cfg, step=global_step)
    freq_kl_weighted = beta_anneal * freq_kl_raw
    loss = recon_loss + beta_freq * freq_kl_weighted

    diagnostics = {
        "loss": loss.detach(),
        "recon_loss": recon_loss.detach(),
        "freq_kl": freq_kl_weighted.detach(),
        "freq_kl_raw": freq_kl_raw.detach(),
        "freq_kl_beta_anneal": torch.as_tensor(
            beta_anneal,
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).detach(),
        "freq_kl_per_harmonic_mean": freq_kl_per_item.mean(dim=0).detach(),
        "freq_prior_reg": freq_kl_weighted.detach(),
        "posterior_std_hz_mean": std_f.mean().detach(),
        "freq_sample_outside_rate": outside_rate.detach(),
    }

    if "log_rho2_f" in model_outputs:
        diagnostics["log_rho2_f_mean"] = model_outputs["log_rho2_f"].mean().detach()
        diagnostics["log_rho2_f_min"] = model_outputs["log_rho2_f"].min().detach()
        diagnostics["log_rho2_f_max"] = model_outputs["log_rho2_f"].max().detach()

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

    return loss, recon_loss, freq_kl_weighted, diagnostics
