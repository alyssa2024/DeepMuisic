import math

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

    def _bayes_system(
        self,
        y_complex,
        f,
        t,
        amp_prior_mean,
        amp_prior_var,
        noise_var_norm,
        eps=1e-12,
    ):
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

        sigma2 = noise_var_norm.to(
            device=gram.device,
            dtype=gram.real.dtype,
        ).clamp_min(eps)
        tau2 = amp_prior_var.to(
            device=gram.device,
            dtype=gram.real.dtype,
        ).clamp_min(eps)
        amp_prior_mean = amp_prior_mean.to(device=gram.device, dtype=gram.dtype)

        mean_y = (phi * amp_prior_mean.unsqueeze(1)).sum(dim=-1)
        residual = y_complex - mean_y
        rhs_residual = phi_h @ residual.unsqueeze(-1)
        precision = torch.diag_embed(1.0 / tau2).to(dtype=gram.dtype)
        system = precision + gram / sigma2.view(-1, 1, 1)

        return {
            "phi": phi,
            "phi_h": phi_h,
            "gram": gram,
            "sigma2": sigma2,
            "tau2": tau2,
            "amp_prior_mean": amp_prior_mean,
            "mean_y": mean_y,
            "residual": residual,
            "rhs_residual": rhs_residual,
            "system": system,
        }

    def solve_amplitudes_bayes(
        self,
        y_complex,
        f,
        t,
        amp_prior_mean,
        amp_prior_var,
        noise_var_norm,
        return_cov_diag=False,
        return_condition=False,
        eps=1e-12,
    ):
        """
        Return the closed-form Gaussian posterior over complex amplitudes.

        The posterior mean equals the centered MAP estimate, while the
        covariance captures Stage 3B amplitude uncertainty.
        """
        system_data = self._bayes_system(
            y_complex=y_complex,
            f=f,
            t=t,
            amp_prior_mean=amp_prior_mean,
            amp_prior_var=amp_prior_var,
            noise_var_norm=noise_var_norm,
            eps=eps,
        )
        system = system_data["system"]
        sigma2 = system_data["sigma2"]
        rhs_residual = system_data["rhs_residual"]
        amp_prior_mean = system_data["amp_prior_mean"]
        tau2 = system_data["tau2"]

        correction = torch.linalg.solve(
            system,
            rhs_residual / sigma2.view(-1, 1, 1),
        ).squeeze(-1)
        post_mean = amp_prior_mean + correction

        diagnostics = {
            "amp_lambda_mean": (sigma2[:, None] / tau2).mean(),
            "amp_prior_var_norm_mean": tau2.mean(),
            "amp_post_mean_norm": torch.linalg.norm(post_mean, dim=-1).mean(),
        }

        post_var_diag = None
        if return_cov_diag:
            post_cov = torch.linalg.inv(system)
            post_var_diag = torch.real(
                torch.diagonal(post_cov, dim1=-2, dim2=-1)
            ).clamp_min(0.0)
            diagnostics["amp_post_var_trace"] = post_var_diag.sum(dim=-1).mean()
            diagnostics["amp_post_std_mean"] = torch.sqrt(
                post_var_diag.clamp_min(eps)
            ).mean()
            diagnostics["amp_uncertainty_to_prior_ratio_mean"] = (
                post_var_diag / tau2
            ).mean()

        if return_condition:
            bayes_cond = torch.linalg.cond(system)
            diagnostics["bayes_cond"] = bayes_cond
            diagnostics["bayes_cond_mean"] = bayes_cond.mean()
            diagnostics["bayes_cond_p95"] = torch.quantile(
                bayes_cond.reshape(-1),
                0.95,
            )

        return post_mean, post_var_diag, diagnostics

    def amplitude_marginal_nll(
        self,
        y_complex,
        f,
        t,
        amp_prior_mean,
        amp_prior_var,
        noise_var_norm,
        include_log_const=False,
        eps=1e-12,
    ):
        """
        Compute -log p(y | f) after marginalizing Gaussian amplitudes.
        """
        system_data = self._bayes_system(
            y_complex=y_complex,
            f=f,
            t=t,
            amp_prior_mean=amp_prior_mean,
            amp_prior_var=amp_prior_var,
            noise_var_norm=noise_var_norm,
            eps=eps,
        )
        system = system_data["system"]
        sigma2 = system_data["sigma2"]
        tau2 = system_data["tau2"]
        residual = system_data["residual"]
        rhs_residual = system_data["rhs_residual"]

        residual_norm = torch.sum(torch.abs(residual) ** 2, dim=-1) / sigma2
        system_solve = torch.linalg.solve(system, rhs_residual)
        correction = torch.real(
            rhs_residual.conj().transpose(-2, -1) @ system_solve
        ).squeeze(-1).squeeze(-1) / sigma2.pow(2)
        quad = (residual_norm - correction).clamp_min(0.0)

        seq_len = y_complex.shape[1]
        sign, logabsdet_system = torch.linalg.slogdet(system)
        logdet = (
            seq_len * torch.log(sigma2)
            + torch.sum(torch.log(tau2), dim=-1)
            + torch.real(logabsdet_system)
        )
        nll_per_sequence = quad + logdet
        if include_log_const:
            nll_per_sequence = nll_per_sequence + seq_len * math.log(math.pi)

        post_mean, post_var_diag, post_diag = self.solve_amplitudes_bayes(
            y_complex=y_complex,
            f=f,
            t=t,
            amp_prior_mean=amp_prior_mean,
            amp_prior_var=tau2,
            noise_var_norm=sigma2,
            return_cov_diag=True,
            return_condition=True,
            eps=eps,
        )
        diagnostics = {
            "marginal_nll": nll_per_sequence.mean(),
            "marginal_quad": quad.mean(),
            "marginal_logdet": logdet.mean(),
            "marginal_slogdet_sign_real_mean": torch.real(sign).mean(),
            "marginal_slogdet_sign_imag_abs_mean": torch.imag(sign).abs().mean(),
            **post_diag,
        }

        return nll_per_sequence.mean(), post_mean, post_var_diag, diagnostics

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
