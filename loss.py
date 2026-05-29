import torch
import torch.nn.functional as F


def complex_mse_loss(x_hat_complex, target_ri):
    if not torch.is_complex(x_hat_complex):
        raise TypeError(f"x_hat_complex must be complex, got {x_hat_complex.dtype}")
    if target_ri.ndim != 3 or target_ri.shape[-1] != 2:
        raise ValueError(f"target_ri must have shape [B, L, 2], got {target_ri.shape}")

    x_hat_ri = torch.stack([x_hat_complex.real, x_hat_complex.imag], dim=-1)
    return F.mse_loss(x_hat_ri, target_ri, reduction="mean")


def sample_sequence_frequencies(mu_f, std_f, num_samples):
    """
    Sample sequence-level frequency vectors.

    Args:
        mu_f:       [B, K]
        std_f:      [B, K]
        num_samples: S_seq

    Returns:
        f_samples: [S_seq, B, K]
    """
    if mu_f.ndim != 2 or std_f.ndim != 2:
        raise ValueError(f"mu_f/std_f must be [B, K], got {mu_f.shape}/{std_f.shape}")
    if mu_f.shape != std_f.shape:
        raise ValueError(f"mu_f and std_f shape mismatch: {mu_f.shape} vs {std_f.shape}")

    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    eps = torch.randn(
        num_samples,
        *mu_f.shape,
        device=mu_f.device,
        dtype=mu_f.dtype,
    )
    f_samples = mu_f.unsqueeze(0) + std_f.unsqueeze(0) * eps
    assert f_samples.ndim == 3
    assert f_samples.shape[1:] == mu_f.shape
    return f_samples


def compute_sequence_posterior_recon_loss(
    y_complex,
    t,
    mu_f,
    std_f,
    model,
    sequence_posterior_samples,
    ridge_lambda,
    f_samples=None,
):
    """
    Reconstruct each sequence from sequence-level posterior frequency samples.
    """
    if f_samples is None:
        f_samples = sample_sequence_frequencies(
            mu_f=mu_f,
            std_f=std_f,
            num_samples=sequence_posterior_samples,
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
    recon_loss = sqerr.mean()

    amp_norm = torch.linalg.norm(c_hat_samples, dim=-1)
    diagnostics = {
        "f_samples": f_samples,
        "y_hat_samples": y_hat_samples,
        "c_hat_samples": c_hat_samples,
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
    recon_loss, recon_diag = compute_sequence_posterior_recon_loss(
        y_complex=y_complex,
        t=t,
        mu_f=mu_f,
        std_f=std_f,
        model=model,
        sequence_posterior_samples=s_seq,
        ridge_lambda=model.ls_ridge,
    )

    freq_prior_reg, outside_rate = uniform_support_penalty(
        f_samples=recon_diag["f_samples"],
        freq_lower=model.encoder.freq_lower,
        freq_upper=model.encoder.freq_upper,
    )

    beta_freq = float(loss_cfg.get("beta_freq", 1e-5))
    loss = recon_loss + beta_freq * freq_prior_reg

    diagnostics = {
        "loss": loss.detach(),
        "recon_loss": recon_loss.detach(),
        "freq_prior_reg": freq_prior_reg.detach(),
        "freq_sample_outside_rate": outside_rate.detach(),
        **{
            k: v.detach() if torch.is_tensor(v) else v
            for k, v in recon_diag.items()
            if k not in ("f_samples", "y_hat_samples", "c_hat_samples")
        },
    }

    return loss, recon_loss, freq_prior_reg, diagnostics
