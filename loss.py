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
    prior_A_mu=0.0,
    prior_A_var=1.0,
    prior_a_w=1000.0,
    prior_phi_mu=0.0,
    prior_phi_kappa=0.0,
    use_kl_A=True,
    use_kl_w=True,
    use_kl_phi=True,
):
    """
    Negative ELBO loss.

    Args:
        x_target: [B, L, 2]
        x_hat:    complex [B, L]
        dist_params:
            (mu_A, logvar_A), a_w, (mu_phi, kappa_phi)

    Returns:
        loss, recon_loss, total_kl
    """

    (mu_A, logvar_A), a_w, (mu_phi, kappa_phi) = dist_params

    recon_loss = complex_mse_loss(x_hat, x_target)

    if use_kl_A:
        kl_A = gaussian_kl(
            mu_A,
            logvar_A,
            prior_mu=prior_A_mu,
            prior_var=prior_A_var,
        )
    else:
        kl_A = torch.zeros((), dtype=mu_A.dtype, device=mu_A.device)

    if use_kl_w:
        kl_w = maxwell_kl(
            a_q=a_w,
            a_p=prior_a_w,
        )
    else:
        kl_w = torch.zeros((), dtype=a_w.dtype, device=a_w.device)

    if use_kl_phi:
        kl_phi = von_mises_kl(
            mu_q=mu_phi,
            kappa_q=kappa_phi,
            mu_p=prior_phi_mu,
            kappa_p=prior_phi_kappa,
        )
    else:
        kl_phi = torch.zeros((), dtype=kappa_phi.dtype, device=kappa_phi.device)

    total_kl = kl_A + kl_w + kl_phi

    loss = recon_loss + beta * total_kl

    return loss, recon_loss, total_kl