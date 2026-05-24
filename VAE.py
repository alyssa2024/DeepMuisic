import torch
import torch.nn as nn


class PhysicalHarmonicVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        ls_ridge: float = 1e-6,
        use_amp_residual: bool = True,
        amp_residual_hidden: int = 128,
        amp_residual_gamma: float = 0.0,
        use_freq_mean_for_ls: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_harmonics = encoder.output_dim
        self.ls_ridge = ls_ridge
        self.use_amp_residual = use_amp_residual
        self.amp_residual_gamma = amp_residual_gamma
        self.use_freq_mean_for_ls = use_freq_mean_for_ls

        k = self.num_harmonics
        self.amp_residual_head = nn.Sequential(
            nn.Linear(3 * k, amp_residual_hidden),
            nn.ReLU(),
            nn.Linear(amp_residual_hidden, amp_residual_hidden),
            nn.ReLU(),
            nn.Linear(amp_residual_hidden, 2 * k),
        )
        nn.init.zeros_(self.amp_residual_head[-1].weight)
        nn.init.zeros_(self.amp_residual_head[-1].bias)

    def reparameterize(self, mu_f, logvar_f):
        """
        Posterior over frequencies only:

            q(z|x) = product_k q(f_k|x)
        """

        std_f = torch.exp(0.5 * logvar_f)
        eps_f = torch.randn_like(std_f)
        f = torch.clamp(mu_f + std_f * eps_f, min=1e-6)
        return f

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

    def predict_amp_residual(self, amp_real_ls, amp_imag_ls, f):
        """
        Predict residual correction around LS amplitudes.
        """

        if hasattr(self.encoder, "f_center") and hasattr(self.encoder, "f_band"):
            f_center = self.encoder.f_center.unsqueeze(0).to(
                dtype=f.dtype,
                device=f.device,
            )
            f_norm = (f - f_center) / float(self.encoder.f_band)
        else:
            f_norm = f

        residual_input = torch.cat([amp_real_ls, amp_imag_ls, f_norm], dim=-1)
        delta = self.amp_residual_head(residual_input)
        delta_real, delta_imag = delta.chunk(2, dim=-1)
        return delta_real, delta_imag

    def forward(self, x, t, probe_ids=None, amp_t=None, amp_target=None):
        """
        Conservative LS-centered posterior:

            f ~ q(f | x)
            a = a_LS(x, f) + gamma * r_phi(a_LS, f)

        Args:
            x:         [B, L, input_dim]
            t:         [B, L]
            probe_ids: [B, L]

        Returns:
            x_hat: complex [B, L]
            dist_params:
                (mu_f, logvar_f)
            aux:
                residual and amplitude diagnostics
        """

        mu_f, logvar_f = self.encoder(x, probe_ids=probe_ids)
        if self.use_freq_mean_for_ls:
            f = torch.clamp(mu_f, min=1e-6)
        else:
            f = self.reparameterize(mu_f, logvar_f)

        if amp_target is None or amp_t is None:
            y_ls = torch.complex(x[..., 0], x[..., 1])
            t_ls = t
        else:
            y_ls = torch.complex(amp_target[..., 0], amp_target[..., 1])
            t_ls = amp_t

        amp_real_ls, amp_imag_ls, _ = self.solve_amplitudes_ls(y_ls, f, t_ls)
        if self.use_amp_residual and self.amp_residual_gamma != 0.0:
            delta_real, delta_imag = self.predict_amp_residual(amp_real_ls, amp_imag_ls, f)
            amp_real = amp_real_ls + self.amp_residual_gamma * delta_real
            amp_imag = amp_imag_ls + self.amp_residual_gamma * delta_imag
        else:
            delta_real = torch.zeros_like(amp_real_ls)
            delta_imag = torch.zeros_like(amp_imag_ls)
            amp_real = amp_real_ls
            amp_imag = amp_imag_ls

        x_hat = self.decode(amp_real, amp_imag, f, t)
        ls_power = amp_real_ls.pow(2) + amp_imag_ls.pow(2)
        delta_power = delta_real.pow(2) + delta_imag.pow(2)
        gamma = float(self.amp_residual_gamma)

        aux = {
            "f_used": f,
            "amp_real_ls": amp_real_ls,
            "amp_imag_ls": amp_imag_ls,
            "amp_real": amp_real,
            "amp_imag": amp_imag,
            "delta_real": delta_real,
            "delta_imag": delta_imag,
            "amp_residual_norm": delta_power.mean(),
            "amp_residual_scaled_norm": (gamma ** 2) * delta_power.mean(),
            "amp_residual_rel": (
                torch.sqrt((gamma ** 2) * delta_power.mean())
                / (torch.sqrt(ls_power.mean()) + 1e-12)
            ),
        }

        return x_hat, (mu_f, logvar_f), aux
