import torch
import torch.nn as nn


class PhysicalHarmonicVAE(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.num_harmonics = encoder.output_dim

    def reparameterize(self, mu_amp_real, logvar_amp_real, mu_amp_imag, logvar_amp_imag, mu_f, logvar_f):
        """
        Factorized posterior:

            q(z|x) = product_k q(a_k_real|x) q(a_k_imag|x) q(f_k|x)
        """

        std_amp_real = torch.exp(0.5 * logvar_amp_real)
        eps_amp_real = torch.randn_like(std_amp_real)
        amp_real = mu_amp_real + std_amp_real * eps_amp_real

        std_amp_imag = torch.exp(0.5 * logvar_amp_imag)
        eps_amp_imag = torch.randn_like(std_amp_imag)
        amp_imag = mu_amp_imag + std_amp_imag * eps_amp_imag

        std_f = torch.exp(0.5 * logvar_f)
        eps_f = torch.randn_like(std_f)
        f = torch.clamp(mu_f + std_f * eps_f, min=1e-6)

        return amp_real, amp_imag, f

    def decode(self, amp_real, amp_imag, f, t):
        """
        Physics-informed deterministic decoder:

            x_hat(t) = sum_k (a_k_real + j a_k_imag) * exp(j * 2pi * f_k * t)

        Args:
            amp_real: [B, K]
            amp_imag: [B, K]
            f:        [B, K]
            t:        [B, L]

        Returns:
            x_hat: complex tensor [B, L]
        """

        complex_amp = torch.complex(amp_real, amp_imag).unsqueeze(-1)  # [B, K, 1]
        f = f.unsqueeze(-1)  # [B, K, 1]

        if t.dim() == 2:
            t = t.unsqueeze(1)  # [B, 1, L]

        theta = 2.0 * torch.pi * f * t  # [B, K, L]

        unit_complex = torch.polar(
            torch.ones_like(theta),
            theta,
        )  # exp(j * theta), complex [B, K, L]

        harmonics = complex_amp * unit_complex

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
                (mu_amp_real, logvar_amp_real), (mu_amp_imag, logvar_amp_imag), (mu_f, logvar_f)
        """

        dist_params = self.encoder(x, probe_ids=probe_ids)
        (mu_amp_real, logvar_amp_real), (mu_amp_imag, logvar_amp_imag), (mu_f, logvar_f) = dist_params

        amp_real, amp_imag, f = self.reparameterize(
            mu_amp_real,
            logvar_amp_real,
            mu_amp_imag,
            logvar_amp_imag,
            mu_f,
            logvar_f,
        )

        x_hat = self.decode(amp_real, amp_imag, f, t)

        return x_hat, dist_params
