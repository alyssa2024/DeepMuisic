import torch
import torch.nn as nn
import numpy as np


class PhysicalHarmonicVAE(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.num_harmonics = encoder.output_dim

    def reparameterize(self, mu_A, logvar_A, a_w, mu_phi, kappa_phi):
        """
        Factorized posterior:

            q(z|x) = Π_k q(A_k|x) q(w_k|x) q(phi_k|x)

        A:
            Gaussian posterior.

        w:
            Maxwell posterior implemented as the norm of a 3D Gaussian vector.

        phi:
            Wrapped-normal approximation for differentiable sampling.
            The prior is still von Mises; this is only the posterior sampling path.
        """

        # Amplitude A ~ Gaussian
        std_A = torch.exp(0.5 * logvar_A)
        eps_A = torch.randn_like(std_A)
        A = mu_A + std_A * eps_A

        # Frequency w ~ Maxwell(a_w)
        eps_w = torch.randn(*a_w.shape, 3, device=a_w.device, dtype=a_w.dtype)
        w = a_w * torch.sqrt(torch.sum(eps_w ** 2, dim=-1) + 1e-8)

        # Phase phi: differentiable wrapped-normal approximation
        std_phi = torch.sqrt(1.0 / (kappa_phi + 1e-8))
        eps_phi = torch.randn_like(mu_phi)
        phi = mu_phi + std_phi * eps_phi

        # wrap to [-pi, pi)
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))

        return A, w, phi

    def decode(self, A, w, phi, t):
        """
        Known physical decoder:

            x_hat(t) = Σ_k A_k exp(j(w_k t + phi_k))

        Args:
            A:   [B, K]
            w:   [B, K]
            phi: [B, K]
            t:   [B, L]

        Returns:
            x_hat: complex tensor [B, L]
        """

        A = A.unsqueeze(-1)      # [B, K, 1]
        w = w.unsqueeze(-1)      # [B, K, 1]
        phi = phi.unsqueeze(-1)  # [B, K, 1]

        if t.dim() == 2:
            t = t.unsqueeze(1)   # [B, 1, L]

        theta = w * t + phi      # [B, K, L]

        harmonics = torch.polar(A, theta)  # complex [B, K, L]

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
                (mu_A, logvar_A), a_w, (mu_phi, kappa_phi)
        """

        dist_params = self.encoder(x, probe_ids=probe_ids)
        (mu_A, logvar_A), a_w, (mu_phi, kappa_phi) = dist_params

        A, w, phi = self.reparameterize(
            mu_A,
            logvar_A,
            a_w,
            mu_phi,
            kappa_phi,
        )

        x_hat = self.decode(A, w, phi, t)

        return x_hat, dist_params