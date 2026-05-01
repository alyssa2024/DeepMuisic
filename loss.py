import math
import torch
import torch.nn.functional as F


def complex_mse_loss(x_hat_complex, target_ri):
    """
    Gaussian observation likelihood for complex signal.

    Args:
        x_hat_complex: complex tensor [B, L]
        target_ri:     real tensor [B, L, 2]
                       target_ri[..., 0] = real(x)
                       target_ri[..., 1] = imag(x)

    Returns:
        scalar MSE
    """

    if not torch.is_complex(x_hat_complex):
        raise TypeError(
            f"x_hat_complex must be complex, got {x_hat_complex.dtype}"
        )

    if target_ri.ndim != 3 or target_ri.shape[-1] != 2:
        raise ValueError(
            f"target_ri must have shape [B, L, 2], got {target_ri.shape}"
        )

    x_hat_ri = torch.stack(
        [x_hat_complex.real, x_hat_complex.imag],
        dim=-1,
    )  # [B, L, 2]

    return F.mse_loss(x_hat_ri, target_ri, reduction="mean")


def log_prob_gaussian(x, mu, logvar):
    """
    log N(x; mu, exp(logvar))

    Args:
        x, mu, logvar: [B, K]
    Returns:
        log_q: [B, K]
    """
    return -0.5 * (
        math.log(2.0 * math.pi)
        + logvar
        + (x - mu) ** 2 / torch.exp(logvar)
    )


def kl_gaussian_to_standard_normal(mu, logvar):
    return 0.5 * torch.sum(
        torch.exp(logvar) + mu ** 2 - 1.0 - logvar,
        dim=-1,
    ).mean()

def log_prob_maxwell(w, a):
    """
    Maxwell-Boltzmann log probability.

    p(w; a) = sqrt(2/pi) * w^2 / a^3 * exp(-w^2 / (2a^2))
    support: w > 0

    Args:
        w: [B, K]
        a: scalar or [K]
    Returns:
        log_p: [B, K]
    """
    eps = 1e-8

    w = torch.clamp(w, min=eps)

    a = torch.as_tensor(
        a,
        dtype=w.dtype,
        device=w.device,
    )

    log_coef = 0.5 * torch.log(
        torch.tensor(2.0 / math.pi, dtype=w.dtype, device=w.device)
    )

    log_p = (
        log_coef
        + 2.0 * torch.log(w)
        - 3.0 * torch.log(a)
        - (w ** 2) / (2.0 * a ** 2)
    )

    return log_p

def frequency_kl_gaussian_to_maxwell(mu_w, logvar_w, prior_a_w, n_samples=1):
    """
    Monte Carlo KL:
        KL[q(w|x) || p(w)]
        q(w|x) = Gaussian(mu_w, exp(logvar_w))
        p(w)   = Maxwell(prior_a_w)

    Args:
        mu_w:      [B, K]
        logvar_w:  [B, K]
        prior_a_w: scalar or [K]
    Returns:
        scalar KL
    """
    kl_samples = []

    for _ in range(n_samples):
        eps = torch.randn_like(mu_w)
        w = mu_w + torch.exp(0.5 * logvar_w) * eps
        w = torch.clamp(w, min=1e-6)

        log_q = log_prob_gaussian(w, mu_w, logvar_w)
        log_p = log_prob_maxwell(w, prior_a_w)

        kl_samples.append((log_q - log_p).sum(dim=-1))  # [B]

    kl = torch.stack(kl_samples, dim=0).mean(dim=0).mean()

    return kl

def compute_harmonic_elbo(
    x_target,
    x_hat,
    dist_params,
    beta=1e-3,
    prior_a_w=1000.0,
    use_kl_w=True,
    use_kl_a=True,
):
    """
    Negative ELBO loss.

    Args:
        x_target: [B, L, 2]
        x_hat:    complex [B, L]
        dist_params:
            posterior parameter dict

    Returns:
        loss, recon_loss, total_kl
    """

    mu_f = dist_params["mu_f"]
    logvar_f = dist_params["logvar_f"]
    mu_amp_real = dist_params["mu_amp_real"]
    logvar_amp_real = dist_params["logvar_amp_real"]
    mu_amp_imag = dist_params["mu_amp_imag"]
    logvar_amp_imag = dist_params["logvar_amp_imag"]

    recon_loss = complex_mse_loss(x_hat, x_target)

    if use_kl_w:
        kl_w = frequency_kl_gaussian_to_maxwell(
            mu_w=2.0 * torch.pi * mu_f,
            logvar_w=math.log((2.0 * math.pi) ** 2) + logvar_f,
            prior_a_w=2.0 * math.pi * torch.as_tensor(
                prior_a_w,
                dtype=mu_f.dtype,
                device=mu_f.device,
            ),
            n_samples=1,
        )
    else:
        kl_w = torch.zeros((), dtype=mu_f.dtype, device=mu_f.device)

    if use_kl_a:
        kl_a_real = kl_gaussian_to_standard_normal(mu_amp_real, logvar_amp_real)
        kl_a_imag = kl_gaussian_to_standard_normal(mu_amp_imag, logvar_amp_imag)
    else:
        kl_a_real = torch.zeros((), dtype=mu_f.dtype, device=mu_f.device)
        kl_a_imag = torch.zeros((), dtype=mu_f.dtype, device=mu_f.device)

    total_kl = kl_w + kl_a_real + kl_a_imag

    loss = recon_loss + beta * total_kl

    return loss, recon_loss, total_kl
