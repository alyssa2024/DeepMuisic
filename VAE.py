import torch
import torch.nn as nn


class DirectFrequencyVariationalPosterior(nn.Module):
    """
    Direct single-instance variational posterior without a Transformer encoder.
    """

    is_direct_global_posterior = True

    def __init__(
        self,
        output_dim,
        freq_lower_hz,
        freq_upper_hz,
        min_log_rho2=-8.0,
        max_log_rho2=-2.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        freq_lower = torch.tensor(freq_lower_hz, dtype=torch.float32)
        freq_upper = torch.tensor(freq_upper_hz, dtype=torch.float32)
        if len(freq_lower) != self.output_dim:
            raise ValueError("freq_lower_hz length must equal output_dim")
        if torch.any(freq_upper <= freq_lower):
            raise ValueError("freq_upper_hz must exceed freq_lower_hz")
        self.register_buffer("freq_lower", freq_lower)
        self.register_buffer("freq_upper", freq_upper)
        self.register_buffer("freq_mid", 0.5 * (freq_lower + freq_upper))
        self.register_buffer("freq_half", 0.5 * (freq_upper - freq_lower))
        self.register_buffer("f_center", 0.5 * (freq_lower + freq_upper))
        self.register_buffer("f_band", 0.5 * (freq_upper - freq_lower))
        self.min_log_rho2 = float(min_log_rho2)
        self.max_log_rho2 = float(max_log_rho2)
        self.raw_mu_f = nn.Parameter(torch.zeros(self.output_dim))
        self.raw_logrho2_f = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, x=None, probe_ids=None, Cws=None):
        del x, probe_ids, Cws
        mu_unit = torch.tanh(self.raw_mu_f)
        mu_f = self.freq_mid + self.freq_half * mu_unit
        log_rho2 = self.min_log_rho2 + (
            self.max_log_rho2 - self.min_log_rho2
        ) * torch.sigmoid(self.raw_logrho2_f)
        rho = torch.exp(0.5 * log_rho2)
        std_f = self.freq_half * rho
        logvar_f = 2.0 * torch.log(std_f + 1e-12)
        return (
            mu_f.view(1, -1),
            logvar_f.view(1, -1),
            std_f.view(1, -1),
            log_rho2.view(1, -1),
        )


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

    @staticmethod
    def fuse_local_frequency_posteriors(mu_local, std_local, eps=1e-8):
        """
        Precision-weighted product of patch-local Gaussian kernels.

        Args:
            mu_local:  [G, P, K]
            std_local: [G, P, K]

        Returns:
            mu_global:  [G, K]
            std_global: [G, K]
        """
        if mu_local.ndim != 3 or std_local.ndim != 3:
            raise ValueError(
                f"mu_local/std_local must be [G, P, K], "
                f"got {mu_local.shape}/{std_local.shape}"
            )
        if mu_local.shape != std_local.shape:
            raise ValueError(
                f"mu_local and std_local shape mismatch: "
                f"{mu_local.shape} vs {std_local.shape}"
            )

        var_local = std_local.clamp_min(eps).pow(2)
        precision_local = 1.0 / var_local
        precision_global = precision_local.sum(dim=1).clamp_min(eps)
        var_global = 1.0 / precision_global
        mu_global = var_global * (precision_local * mu_local).sum(dim=1)
        std_global = torch.sqrt(var_global.clamp_min(eps))
        return mu_global, std_global

    @staticmethod
    def compute_fusion_diagnostics(std_local, eps=1e-8):
        var_local = std_local.clamp_min(eps).pow(2)
        precision = 1.0 / var_local
        weights = precision / precision.sum(dim=1, keepdim=True).clamp_min(eps)
        max_weight = weights.max(dim=1).values
        effective_patches = 1.0 / weights.pow(2).sum(dim=1).clamp_min(eps)
        entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=1)
        return {
            "fusion_weight": weights,
            "fusion_max_weight": max_weight,
            "fusion_effective_num_patches": effective_patches,
            "fusion_entropy": entropy,
        }

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
        if torch.is_tensor(ridge_lambda):
            ridge = ridge_lambda.to(device=gram.device, dtype=gram.real.dtype)
            if ridge.ndim != 1 or ridge.shape[0] != gram.shape[0]:
                raise ValueError(
                    f"ridge_lambda tensor must have shape [B], got {ridge.shape}"
                )
            eye = torch.eye(
                self.num_harmonics,
                dtype=gram.dtype,
                device=gram.device,
            ).unsqueeze(0)
            gram = gram + ridge.to(dtype=gram.dtype).view(-1, 1, 1) * eye
        elif ridge_lambda > 0.0:
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
        is_group_input = x.dim() == 4
        group_shape = None
        if is_group_input:
            group_size, patches_per_group, seq_len, input_dim = x.shape
            if group_size != 1:
                raise ValueError(
                    "single-instance profile path requires x.shape[0] == 1; "
                    f"got {group_size}"
                )
            if getattr(self.encoder, "is_direct_global_posterior", False):
                mu_f, logvar_f, std_f, log_rho2_f = self.encoder(
                    x,
                    probe_ids=probe_ids,
                )
                mu_local = mu_f[:, None, :].expand(group_size, patches_per_group, -1)
                std_local = std_f[:, None, :].expand(group_size, patches_per_group, -1)
                outputs = {
                    "mu_f": mu_f,
                    "std_f": std_f,
                    "logvar_f": logvar_f,
                    "mu_f_local": mu_local,
                    "std_f_local": std_local,
                    "logvar_f_local": logvar_f[:, None, :].expand(
                        group_size,
                        patches_per_group,
                        -1,
                    ),
                    "log_rho2_f": log_rho2_f,
                    "log_rho2_f_local": log_rho2_f[:, None, :].expand(
                        group_size,
                        patches_per_group,
                        -1,
                    ),
                }
                outputs.update(self.compute_fusion_diagnostics(std_local))
                return outputs
            group_shape = (group_size, patches_per_group)
            x_encoder = x.reshape(group_size * patches_per_group, seq_len, input_dim)
            if probe_ids is not None:
                probe_ids_encoder = probe_ids.reshape(group_size * patches_per_group, seq_len)
            else:
                probe_ids_encoder = None
        else:
            x_encoder = x
            probe_ids_encoder = probe_ids

        encoder_out = self.encoder(x_encoder, probe_ids=probe_ids_encoder)
        if len(encoder_out) == 4:
            mu_f, logvar_f, std_f, log_rho2_f = encoder_out
        elif len(encoder_out) == 3:
            mu_f, logvar_f, std_f = encoder_out
            log_rho2_f = None
        else:
            mu_f, logvar_f = encoder_out
            std_f = torch.exp(0.5 * logvar_f)
            log_rho2_f = None

        if is_group_input:
            group_size, patches_per_group = group_shape
            mu_local = mu_f.reshape(group_size, patches_per_group, -1)
            std_local = std_f.reshape(group_size, patches_per_group, -1)
            mu_global, std_global = self.fuse_local_frequency_posteriors(
                mu_local=mu_local,
                std_local=std_local,
            )
            outputs = {
                "mu_f": mu_global,
                "std_f": std_global,
                "logvar_f": 2.0 * torch.log(std_global + 1e-12),
                "mu_f_local": mu_local,
                "std_f_local": std_local,
                "logvar_f_local": logvar_f.reshape(group_size, patches_per_group, -1),
            }
            if log_rho2_f is not None:
                outputs["log_rho2_f_local"] = log_rho2_f.reshape(
                    group_size,
                    patches_per_group,
                    -1,
                )
                outputs["log_rho2_f_mean"] = outputs["log_rho2_f_local"].mean(dim=1)
            outputs.update(self.compute_fusion_diagnostics(std_local))
            return outputs

        outputs = {
            "mu_f": mu_f,
            "std_f": std_f,
            "logvar_f": logvar_f,
        }
        if log_rho2_f is not None:
            outputs["log_rho2_f"] = log_rho2_f

        return outputs
