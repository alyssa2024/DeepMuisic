import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PhysicalHarmonicVAE(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.num_harmonics = encoder.output_dim

    def reparameterize(self, mu_A, logvar_A, mu_w, logvar_w, mu_phi, kappa_phi):
        """
        Factorized posterior:

            q(z|x) = product_k q(A_k|x) q(w_k|x) q(phi_k|x)

        A:
            Gaussian posterior.

        w:
            Gaussian posterior, then clamped to positive values.

        phi:
            Wrapped-normal approximation for differentiable sampling.
            The prior remains von Mises; this only defines the sampling path.
        """

        # Amplitude A ~ Gaussian, then mapped to non-negative domain
        std_A = torch.exp(0.5 * logvar_A)
        eps_A = torch.randn_like(std_A)
        A_raw = mu_A + std_A * eps_A
        A = F.softplus(A_raw) + 1e-8

        # Frequency w ~ Gaussian
        std_w = torch.exp(0.5 * logvar_w)
        eps_w = torch.randn_like(std_w)
        w = mu_w + std_w * eps_w

        # Frequency must stay positive
        w = torch.clamp(w, min=1e-6)

        # Phase phi: differentiable wrapped-normal approximation
        std_phi = torch.sqrt(1.0 / (kappa_phi + 1e-8))
        eps_phi = torch.randn_like(mu_phi)
        phi = mu_phi + std_phi * eps_phi

        # Wrap phase to [-pi, pi)
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))

        return A, w, phi

    def decode(self, A, w, phi, t):
        """
        Physics-informed deterministic decoder:

            x_hat(t) = sum_k A_k * exp(j * (w_k * t + phi_k))

        Args:
            A:   [B, K]
            w:   [B, K]
            phi: [B, K]
            t:   [B, L]

        Returns:
            x_hat: complex tensor [B, L]
        """

        A = A.unsqueeze(-1)  # [B, K, 1]
        w = w.unsqueeze(-1)  # [B, K, 1]
        phi = phi.unsqueeze(-1)  # [B, K, 1]

        if t.dim() == 2:
            t = t.unsqueeze(1)  # [B, 1, L]

        theta = w * t + phi  # [B, K, L]

        unit_complex = torch.polar(
            torch.ones_like(theta),
            theta,
        )  # exp(j * theta), complex [B, K, L]

        harmonics = A.to(unit_complex.dtype) * unit_complex

        x_hat = torch.sum(harmonics, dim=1)  # complex [B, L]

        return x_hat

    def forward(self, x, t, probe_ids=None):
        """
        Args:
            x:         [B, L, input_dim]
            t:         [B, L]
            probe_ids: [B, L]

        Returns:
            x_hat: complex [B, L]
            dist_params:
                (mu_A, logvar_A), (mu_w, logvar_w), (mu_phi, kappa_phi)
        """

        dist_params = self.encoder(x, probe_ids=probe_ids)
        (mu_A, logvar_A), (mu_w, logvar_w), (mu_phi, kappa_phi) = dist_params

        A, w, phi = self.reparameterize(
            mu_A,
            logvar_A,
            mu_w,
            logvar_w,
            mu_phi,
            kappa_phi,
        )

        x_hat = self.decode(A, w, phi, t)

        return x_hat, dist_params
