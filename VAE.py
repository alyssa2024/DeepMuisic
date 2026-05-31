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

    def solve_amplitudes_map(
        self,
        y_complex,
        f,
        t,
        amp_prior_mean,
        amp_prior_var,
        noise_var_norm,
        return_condition=False,
        eps=1e-12,
    ):
        """
        Solve complex amplitudes with a centered Gaussian MAP estimate.

        The prior is c ~ CN(amp_prior_mean, diag(amp_prior_var)).
        """
        if not torch.is_complex(y_complex):
            raise TypeError(f"y_complex must be complex, got {y_complex.dtype}")
        if not torch.is_complex(amp_prior_mean):
            raise TypeError(
                f"amp_prior_mean must be complex, got {amp_prior_mean.dtype}"
            )
        if f.ndim != 2:
            raise ValueError(f"f must have shape [B, K], got {f.shape}")
        if amp_prior_mean.shape != f.shape:
            raise ValueError(
                "amp_prior_mean shape must match f: "
                f"{amp_prior_mean.shape} vs {f.shape}"
            )
        if amp_prior_var.shape != f.shape:
            raise ValueError(
                "amp_prior_var shape must match f: "
                f"{amp_prior_var.shape} vs {f.shape}"
            )
        if noise_var_norm.ndim != 1 or noise_var_norm.shape[0] != f.shape[0]:
            raise ValueError(
                f"noise_var_norm must have shape [B], got {noise_var_norm.shape}"
            )

        phi = self.build_dictionary(f, t)
        phi_h = phi.conj().transpose(-2, -1)

        gram = phi_h @ phi
        rhs = phi_h @ y_complex.unsqueeze(-1)

        amp_prior_var = amp_prior_var.to(
            device=gram.device,
            dtype=gram.real.dtype,
        ).clamp_min(eps)
        noise_var_norm = noise_var_norm.to(
            device=gram.device,
            dtype=gram.real.dtype,
        ).clamp_min(eps)
        amp_prior_mean = amp_prior_mean.to(device=gram.device, dtype=gram.dtype)

        lambda_bk = noise_var_norm[:, None] / amp_prior_var
        map_mat = gram + torch.diag_embed(lambda_bk).to(dtype=gram.dtype)
        map_rhs = rhs + (lambda_bk.to(dtype=gram.dtype) * amp_prior_mean).unsqueeze(-1)

        complex_amp = torch.linalg.solve(map_mat, map_rhs).squeeze(-1)

        if return_condition:
            cond = torch.linalg.cond(map_mat)
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
