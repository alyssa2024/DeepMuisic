import torch
import torch.nn as nn


class PhysicalHarmonicVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        ls_ridge: float = 1e-6,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_harmonics = encoder.output_dim
        self.ls_ridge = float(ls_ridge)

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
        if f.dim() != 2:
            raise ValueError(f"f must have shape [B, K], got {f.shape}")

        phase = 2.0 * torch.pi * t.unsqueeze(-1) * f.unsqueeze(1)
        return torch.polar(torch.ones_like(phase), phase)

    def solve_amplitudes_ls(
        self,
        y_complex,
        f,
        t,
        ridge_lambda=None,
        return_condition=False,
    ):
        """
        Solve complex amplitudes with regularized least squares.
        """
        if ridge_lambda is None:
            ridge_lambda = self.ls_ridge
        if not torch.is_complex(y_complex):
            raise TypeError(f"y_complex must be complex, got {y_complex.dtype}")

        phi = self.build_dictionary(f, t)
        phi_h = phi.conj().transpose(-2, -1)

        gram = phi_h @ phi
        if ridge_lambda > 0.0:
            eye = torch.eye(
                self.num_harmonics,
                dtype=gram.dtype,
                device=gram.device,
            ).unsqueeze(0)
            gram = gram + float(ridge_lambda) * eye

        rhs = phi_h @ y_complex.unsqueeze(-1)
        complex_amp = torch.linalg.solve(gram, rhs).squeeze(-1)

        if return_condition:
            cond = torch.linalg.cond(gram)
            return complex_amp.real, complex_amp.imag, complex_amp, cond

        return complex_amp.real, complex_amp.imag, complex_amp

    def decode(self, amp_real, amp_imag, f, t):
        """
        Physics-informed deterministic decoder.
        """
        complex_amp = torch.complex(amp_real, amp_imag)
        phi = self.build_dictionary(f, t)
        return (phi * complex_amp.unsqueeze(1)).sum(dim=-1)

    def forward(self, x, t=None, probe_ids=None):
        encoder_out = self.encoder(x, probe_ids=probe_ids)
        if len(encoder_out) == 4:
            mu_f, logvar_f, std_f, log_rho2_f = encoder_out
        elif len(encoder_out) == 3:
            mu_f, logvar_f, std_f = encoder_out
            log_rho2_f = None
        else:
            mu_f, logvar_f = encoder_out
            std_f = torch.exp(0.5 * logvar_f)
            log_rho2_f = None

        outputs = {
            "mu_f": mu_f,
            "std_f": std_f,
            "logvar_f": logvar_f,
        }
        if log_rho2_f is not None:
            outputs["log_rho2_f"] = log_rho2_f

        return outputs
