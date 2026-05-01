import torch
import torch.nn as nn


class PhysicalHarmonicVAE(nn.Module):
    def __init__(self, encoder: nn.Module, ls_ridge: float = 1e-6):
        super().__init__()
        self.encoder = encoder
        self.num_harmonics = encoder.output_dim
        self.ls_ridge = ls_ridge

    def reparameterize(self, mu, logvar, clamp_min=None, noise_scale: float = 1.0):
        """
        Reparameterization with controllable sampling strength.

            z = mu + noise_scale * std * eps

        noise_scale = 0.0 gives posterior mean.
        noise_scale = 1.0 gives standard posterior sampling.
        """
        if noise_scale <= 0.0:
            sample = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample = mu + noise_scale * std * eps
        if clamp_min is not None:
            sample = torch.clamp(sample, min=clamp_min)
        return sample

    def build_dictionary(self, f, t):
        """
        Construct the complex exponential dictionary Phi(f, t).

        Args:
            f: [B, K]
            t: [B, L]

        Returns:
            Phi: complex [B, L, K]
        """

        if t.dim() != 2:
            raise ValueError(f"t must have shape [B, L], got {t.shape}")

        phase = 2.0 * torch.pi * t.unsqueeze(-1) * f.unsqueeze(1)  # [B, L, K]
        phi = torch.polar(torch.ones_like(phase), phase)
        return phi

    def solve_amplitudes_ls(self, y_complex, f, t):
        """
        Solve complex amplitudes with regularized least squares:

            a_hat = argmin_a || y - Phi(f, t) a ||_2^2 + lambda ||a||_2^2

        Args:
            y_complex: complex [B, L]
            f:         [B, K]
            t:         [B, L]

        Returns:
            amp_real:    [B, K]
            amp_imag:    [B, K]
            complex_amp: complex [B, K]
        """

        if not torch.is_complex(y_complex):
            raise TypeError(f"y_complex must be complex, got {y_complex.dtype}")

        phi = self.build_dictionary(f, t)  # [B, L, K]
        phi_h = phi.conj().transpose(-2, -1)  # [B, K, L]

        gram = phi_h @ phi  # [B, K, K]
        if self.ls_ridge > 0.0:
            eye = torch.eye(
                self.num_harmonics,
                dtype=gram.dtype,
                device=gram.device,
            ).unsqueeze(0)
            gram = gram + self.ls_ridge * eye

        rhs = phi_h @ y_complex.unsqueeze(-1)  # [B, K, 1]
        complex_amp = torch.linalg.solve(gram, rhs).squeeze(-1)  # [B, K]

        return complex_amp.real, complex_amp.imag, complex_amp

    def decode(self, amp_real, amp_imag, f, t):
        """
        Physics-informed deterministic decoder:

            x_hat(t) = sum_k (a_k_real + j a_k_imag) * exp(j * 2pi * f_k * t)
        """

        complex_amp = torch.complex(amp_real, amp_imag)
        phi = self.build_dictionary(f, t)  # [B, L, K]
        x_hat = (phi * complex_amp.unsqueeze(1)).sum(dim=-1)  # [B, L]
        return x_hat

    def infer_posteriors(
        self,
        x,
        probe_ids=None,
        sample_f=True,
        sample_a=True,
        amp_noise_scale: float = 1.0,
    ):
        context = self.encoder.encode_context(x, probe_ids=probe_ids)
        mu_f, logvar_f = self.encoder.infer_frequency_posterior(context)
        f = self.reparameterize(
            mu_f,
            logvar_f,
            clamp_min=1e-6,
            noise_scale=1.0,
        ) if sample_f else mu_f

        mu_amp_real, logvar_amp_real, mu_amp_imag, logvar_amp_imag = (
            self.encoder.infer_amplitude_posterior(context, f)
        )
        if sample_a:
            amp_real = self.reparameterize(
                mu_amp_real,
                logvar_amp_real,
                noise_scale=amp_noise_scale,
            )
            amp_imag = self.reparameterize(
                mu_amp_imag,
                logvar_amp_imag,
                noise_scale=amp_noise_scale,
            )
        else:
            amp_real = mu_amp_real
            amp_imag = mu_amp_imag

        return {
            "mu_f": mu_f,
            "logvar_f": logvar_f,
            "f": f,
            "mu_amp_real": mu_amp_real,
            "logvar_amp_real": logvar_amp_real,
            "mu_amp_imag": mu_amp_imag,
            "logvar_amp_imag": logvar_amp_imag,
            "amp_real": amp_real,
            "amp_imag": amp_imag,
            "amp_noise_scale": torch.as_tensor(
                amp_noise_scale,
                dtype=mu_f.dtype,
                device=mu_f.device,
            ),
        }

    def forward(
        self,
        x,
        t,
        probe_ids=None,
        sample_f: bool = True,
        sample_a: bool = True,
        amp_noise_scale: float = 1.0,
    ):
        """
        Args:
            x:         [B, L, input_dim]
            t:         [B, L]
            probe_ids: [B, L]
            sample_f:  whether to sample frequency posterior
            sample_a:  whether to sample amplitude posterior
            amp_noise_scale:
                gamma in c = mu_c + gamma * sigma_c * eps.
                gamma = 0.0 means posterior mean reconstruction.
                gamma = 1.0 means standard posterior sampling.

        Returns:
            x_hat: complex [B, L]
            dist_params: posterior parameter dict
        """
        dist_params = self.infer_posteriors(
            x,
            probe_ids=probe_ids,
            sample_f=sample_f,
            sample_a=sample_a,
            amp_noise_scale=amp_noise_scale,
        )
        x_hat = self.decode(
            dist_params["amp_real"],
            dist_params["amp_imag"],
            dist_params["f"],
            t,
        )
        return x_hat, dist_params
