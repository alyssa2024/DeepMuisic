import torch
import torch.nn.functional as F
import math


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


def gaussian_kl(mu, logvar, prior_mu=0.0, prior_var=1.0):
    """
    KL[N(mu, sigma^2) || N(prior_mu, prior_var)].

    Args:
        mu:     [B, K]
        logvar: [B, K]

    Returns:
        scalar mean KL over batch
    """

    var = torch.exp(logvar)

    prior_mu = torch.as_tensor(
        prior_mu,
        dtype=mu.dtype,
        device=mu.device,
    )

    prior_var = torch.as_tensor(
        prior_var,
        dtype=mu.dtype,
        device=mu.device,
    )

    kl_per_dim = 0.5 * (
        torch.log(prior_var)
        - logvar
        + (var + (mu - prior_mu) ** 2) / prior_var
        - 1.0
    )

    return kl_per_dim.sum(dim=-1).mean()

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

def maxwell_kl(a_q, a_p):
    """
    KL[Maxwell(a_q) || Maxwell(a_p)].

    Maxwell scale parameter a > 0.

    Formula:
        KL = 3 log(a_p/a_q) + (3/2)(a_q^2/a_p^2) - 3/2

    Equivalent:
        KL = 1.5 * (r^2 - 1 - log(r^2))
        where r = a_q / a_p.

    Args:
        a_q: [B, K]
        a_p: scalar or [K]

    Returns:
        scalar mean KL over batch
    """

    a_p = torch.as_tensor(
        a_p,
        dtype=a_q.dtype,
        device=a_q.device,
    )

    ratio_sq = (a_q / a_p).pow(2)

    kl_per_dim = 1.5 * (
        ratio_sq
        - 1.0
        - torch.log(ratio_sq + 1e-8)
    )

    return kl_per_dim.sum(dim=-1).mean()


def von_mises_kl(mu_q, kappa_q, mu_p=0.0, kappa_p=0.0):
    """
    Approximate analytic KL:

        KL[VM(mu_q, kappa_q) || VM(mu_p, kappa_p)]

    Formula:
        E_q[log q - log p]
        =
        log I0(kappa_p) - log I0(kappa_q)
        + kappa_q * A(kappa_q)
        - kappa_p * A(kappa_q) * cos(mu_q - mu_p)

    where:
        A(kappa) = I1(kappa) / I0(kappa)

    If kappa_p = 0, prior is uniform over [-pi, pi).

    Args:
        mu_q:    [B, K]
        kappa_q: [B, K]
        mu_p:    scalar or [K]
        kappa_p: scalar or [K]

    Returns:
        scalar mean KL over batch
    """

    mu_p = torch.as_tensor(
        mu_p,
        dtype=mu_q.dtype,
        device=mu_q.device,
    )

    kappa_p = torch.as_tensor(
        kappa_p,
        dtype=kappa_q.dtype,
        device=kappa_q.device,
    )

    # stable log I0(kappa) using scaled Bessel i0e:
    # i0e(kappa) = exp(-abs(kappa)) I0(kappa)
    # since kappa >= 0, log I0(kappa) = kappa + log(i0e(kappa))
    i0e_q = torch.special.i0e(kappa_q)
    i1e_q = torch.special.i1e(kappa_q)

    log_i0_q = kappa_q + torch.log(i0e_q + 1e-8)

    A_q = i1e_q / (i0e_q + 1e-8)

    if torch.all(kappa_p == 0):
        log_i0_p = torch.zeros_like(kappa_q)
        prior_term = torch.zeros_like(kappa_q)
    else:
        log_i0_p = kappa_p + torch.log(torch.special.i0e(kappa_p) + 1e-8)
        prior_term = kappa_p * A_q * torch.cos(mu_q - mu_p)

    kl_per_dim = (
        log_i0_p
        - log_i0_q
        + kappa_q * A_q
        - prior_term
    )

    return kl_per_dim.sum(dim=-1).mean()


def compute_harmonic_elbo(
    x_target,
    x_hat,
    dist_params,
    beta=1e-3,
    prior_amp_real_mu=0.0,
    prior_amp_real_var=1.0,
    prior_amp_imag_mu=0.0,
    prior_amp_imag_var=1.0,
    prior_a_w=1000.0,
    use_kl_amp_real=True,
    use_kl_amp_imag=True,
    use_kl_w=True,
):
    """
    Negative ELBO loss.

    Args:
        x_target: [B, L, 2]
        x_hat:    complex [B, L]
        dist_params:
            (mu_amp_real, logvar_amp_real), (mu_amp_imag, logvar_amp_imag), (mu_f, logvar_f)

    Returns:
        loss, recon_loss, total_kl
    """

    (mu_amp_real, logvar_amp_real), (mu_amp_imag, logvar_amp_imag), (mu_f, logvar_f) = dist_params

    recon_loss = complex_mse_loss(x_hat, x_target)

    if use_kl_amp_real:
        kl_amp_real = gaussian_kl(
            mu_amp_real,
            logvar_amp_real,
            prior_mu=prior_amp_real_mu,
            prior_var=prior_amp_real_var,
        )
    else:
        kl_amp_real = torch.zeros((), dtype=mu_amp_real.dtype, device=mu_amp_real.device)

    if use_kl_amp_imag:
        kl_amp_imag = gaussian_kl(
            mu_amp_imag,
            logvar_amp_imag,
            prior_mu=prior_amp_imag_mu,
            prior_var=prior_amp_imag_var,
        )
    else:
        kl_amp_imag = torch.zeros((), dtype=mu_amp_imag.dtype, device=mu_amp_imag.device)

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

    total_kl = kl_amp_real + kl_amp_imag + kl_w

    loss = recon_loss + beta * total_kl

    return loss, recon_loss, total_kl
