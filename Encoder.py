import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

try:
    import gin
except ImportError:
    class _GinFallback:
        @staticmethod
        def configurable(obj):
            return obj

    gin = _GinFallback()


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class ContinuousTimePositionalEncoding(torch.nn.Module):
    """
    Continuous-time sinusoidal positional encoding.

    tau is normalized local time with shape [B, L], typically in [0, 1].
    """

    def __init__(
        self,
        d_model: int,
        num_bands: int = 64,
        trainable_proj: bool = True,
        min_freq: float = 1.0,
        max_freq: float = 10000.0,
    ):
        super().__init__()

        if num_bands < 1:
            raise ValueError(f"num_bands must be >= 1, got {num_bands}")
        if min_freq <= 0 or max_freq <= 0:
            raise ValueError("min_freq and max_freq must be positive")
        if max_freq < min_freq:
            raise ValueError("max_freq must be >= min_freq")

        self.d_model = int(d_model)
        self.num_bands = int(num_bands)
        self.trainable_proj = bool(trainable_proj)

        if num_bands == 1:
            freqs = torch.tensor([float(min_freq)], dtype=torch.float32)
        else:
            freqs = torch.exp(
                torch.linspace(
                    math.log(float(min_freq)),
                    math.log(float(max_freq)),
                    steps=num_bands,
                    dtype=torch.float32,
                )
            )
        self.register_buffer("freqs", freqs)

        raw_dim = 2 * num_bands
        if trainable_proj:
            self.proj = torch.nn.Linear(raw_dim, d_model)
        else:
            if raw_dim != d_model:
                raise ValueError(
                    "When trainable_proj=False, 2*num_bands must equal d_model. "
                    f"Got 2*num_bands={raw_dim}, d_model={d_model}."
                )
            self.proj = torch.nn.Identity()

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, L, d_model]
            tau: [B, L], normalized local time

        Returns:
            x + continuous-time positional encoding, shape [B, L, d_model]
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, L, d_model], got {x.shape}")
        if tau.ndim != 2:
            raise ValueError(f"tau must have shape [B, L], got {tau.shape}")
        if tau.shape != x.shape[:2]:
            raise ValueError(
                f"tau shape {tau.shape} must match x batch/length {x.shape[:2]}"
            )

        tau = tau.to(device=x.device, dtype=x.dtype)
        freqs = self.freqs.to(device=x.device, dtype=x.dtype)
        phase = 2.0 * math.pi * tau.unsqueeze(-1) * freqs.view(1, 1, -1)
        pe_raw = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        pe = self.proj(pe_raw)

        return x + pe


@gin.configurable
class VariationalIndependentTimeSeriesTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        hidden_dim_dense=256,
        dropout=0.0,
        max_len=5000,
        num_probes=4,
        use_standard_pe=False,
        use_time_pe=False,
        time_feature_index=-1,
        time_pe_num_bands=64,
        time_pe_trainable_proj=True,
        causal_mask=False,
        device="cpu",
        freq_lower_hz=None,
        freq_upper_hz=None,
        min_log_rho2=-12.0,
        max_log_rho2=-4.0,
        **kwargs,
    ):
        super().__init__()

        if hidden_dim % nhead != 0:
            hidden_dim = ((hidden_dim // nhead) + 1) * nhead

        self.num_harmonics = output_dim
        self.output_dim = output_dim

        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.probe_embedding = torch.nn.Embedding(num_probes, hidden_dim)

        self.use_standard_pe = bool(use_standard_pe)
        self.use_time_pe = bool(use_time_pe)
        self.time_feature_index = int(time_feature_index)

        if self.use_standard_pe and self.use_time_pe:
            raise ValueError(
                "use_standard_pe and use_time_pe should not be enabled together. "
                "Use use_time_pe=True for continuous-time positional encoding."
            )

        if self.use_standard_pe:
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        else:
            self.pos_encoder = None

        if self.use_time_pe:
            self.time_pos_encoder = ContinuousTimePositionalEncoding(
                d_model=hidden_dim,
                num_bands=int(time_pe_num_bands),
                trainable_proj=bool(time_pe_trainable_proj),
            )
        else:
            self.time_pos_encoder = None

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self._fc = torch.nn.Linear(hidden_dim, hidden_dim_dense)
        self.feature_dim = int(hidden_dim_dense)

        self._fc_f_mu = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        self._fc_f_logvar = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        torch.nn.init.zeros_(self._fc_f_mu.weight)
        torch.nn.init.zeros_(self._fc_f_mu.bias)

        self._device = device
        self._causal_mask = causal_mask
        if freq_lower_hz is None:
            freq_lower_hz = np.asarray([167.0, 341.0, 635.0, 872.0]) * 0.95
        if freq_upper_hz is None:
            freq_upper_hz = np.asarray([167.0, 341.0, 635.0, 872.0]) * 1.05
        if len(freq_lower_hz) != output_dim:
            raise ValueError(
                f"len(freq_lower_hz)={len(freq_lower_hz)} must equal output_dim={output_dim}"
            )
        if len(freq_upper_hz) != output_dim:
            raise ValueError(
                f"len(freq_upper_hz)={len(freq_upper_hz)} must equal output_dim={output_dim}"
            )
        freq_lower = torch.tensor(freq_lower_hz, dtype=torch.float32)
        freq_upper = torch.tensor(freq_upper_hz, dtype=torch.float32)
        if torch.any(freq_lower <= 0):
            raise ValueError("frequency lower bounds must be positive")
        if torch.any(freq_upper <= freq_lower):
            raise ValueError("frequency upper bounds must exceed lower bounds")

        self.register_buffer("freq_lower", freq_lower)
        self.register_buffer("freq_upper", freq_upper)
        self.register_buffer("freq_mid", 0.5 * (freq_lower + freq_upper))
        self.register_buffer("freq_half", 0.5 * (freq_upper - freq_lower))
        self.min_log_rho2 = float(min_log_rho2)
        self.max_log_rho2 = float(max_log_rho2)

        # Backward-compatible aliases for diagnostics.
        self.register_buffer("f_center", 0.5 * (freq_lower + freq_upper))
        self.register_buffer(
            "f_band",
            0.5 * (freq_upper - freq_lower),
        )

    def generate_causal_mask(self, seq_len):
        # Upper triangular mask: (seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf")).to(self._device)

    def generate_non_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.eye(seq_len)
        return mask.masked_fill(mask == 1, float("-inf")).to(self._device)

    def encode_features(self, x, probe_ids=None, Cws=None):
        """
        x:        [B, L, input_dim]
                  input_dim=7 when normalized local-time feature is used.
                  Continuous-time PE reads x[..., time_feature_index].
        probe_ids:[B, L]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        x_transformer = self.input_proj(x)

        if probe_ids is not None:
            x_transformer = x_transformer + self.probe_embedding(probe_ids)

        if self.use_standard_pe:
            x_transformer = self.pos_encoder(x_transformer)

        if self.use_time_pe:
            if not (-x.size(-1) <= self.time_feature_index < x.size(-1)):
                raise ValueError(
                    f"time_feature_index={self.time_feature_index} is invalid "
                    f"for input_dim={x.size(-1)}"
                )
            tau = x[..., self.time_feature_index]
            x_transformer = self.time_pos_encoder(x_transformer, tau)

        if self._causal_mask:
            mask = self.generate_causal_mask(seq_len)
        else:
            mask = None

        x_transformer = self.transformer_encoder(x_transformer, mask=mask)

        # Key step: mean pooling to produce one global latent per patch.
        pooled = x_transformer.mean(dim=1)  # [B, hidden_dim]

        return F.relu(self._fc(pooled))  # [B, hidden_dim_dense]

    def posterior_from_global_feature(self, h_global):
        raw_f_mu = self._fc_f_mu(h_global)
        raw_logrho2_f = self._fc_f_logvar(h_global)

        mu_unit = torch.tanh(raw_f_mu)
        mu_f = self.freq_mid + self.freq_half * mu_unit

        log_rho2 = self.min_log_rho2 + (
            self.max_log_rho2 - self.min_log_rho2
        ) * torch.sigmoid(raw_logrho2_f)
        rho = torch.exp(0.5 * log_rho2)
        std_f = self.freq_half * rho
        logvar_f = 2.0 * torch.log(std_f + 1e-12)

        return mu_f, logvar_f, std_f, log_rho2

    def forward(self, x, probe_ids=None, Cws=None):
        h = self.encode_features(x, probe_ids=probe_ids, Cws=Cws)
        return self.posterior_from_global_feature(h)


class SequentialDSAEEncoder(torch.nn.Module):
    """
    DSAE-style sequential posterior for BTT patches.

    No mean/max/attention/window pooling is used. Local patch features and the
    patch sequence are summarized by BiGRU endpoint states, matching the DSAE
    forward-last/backward-first pattern.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        local_hidden_dim=128,
        local_out_dim=128,
        context_hidden_dim=256,
        context_layers=1,
        innovation_hidden_dim=256,
        dropout=0.0,
        freq_lower_hz=None,
        freq_upper_hz=None,
        min_log_rho2=-12.0,
        max_log_rho2=-4.0,
        c_init_logvar=-4.0,
        z1_init_logvar=0.0,
        delta_init_logvar=-6.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_harmonics = int(output_dim)
        self.input_dim = int(input_dim)
        self.local_hidden_dim = int(local_hidden_dim)
        self.local_out_dim = int(local_out_dim)
        self.context_hidden_dim = int(context_hidden_dim)

        self.point_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, local_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(local_hidden_dim, local_hidden_dim),
            torch.nn.SiLU(),
        )
        self.local_rnn = torch.nn.GRU(
            input_size=local_hidden_dim,
            hidden_size=local_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.local_endpoint_proj = torch.nn.Sequential(
            torch.nn.Linear(2 * local_hidden_dim, local_out_dim),
            torch.nn.SiLU(),
        )

        self.context_rnn = torch.nn.GRU(
            input_size=local_out_dim,
            hidden_size=context_hidden_dim,
            num_layers=int(context_layers),
            dropout=float(dropout) if int(context_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.feature_dim = 2 * context_hidden_dim

        self.f_mu_head = torch.nn.Linear(self.feature_dim, self.num_harmonics)
        self.f_logrho2_head = torch.nn.Linear(self.feature_dim, self.num_harmonics)
        self.logamp_mu_head = torch.nn.Linear(self.feature_dim, self.num_harmonics)
        self.logamp_logvar_head = torch.nn.Linear(self.feature_dim, self.num_harmonics)
        self.z1_head = torch.nn.Linear(
            self.feature_dim + self.num_harmonics,
            2 * self.num_harmonics,
        )

        delta_input_dim = self.feature_dim + 2 * self.num_harmonics + 1
        self.delta_rnn = torch.nn.GRUCell(delta_input_dim, innovation_hidden_dim)
        self.delta_head = torch.nn.Linear(
            innovation_hidden_dim,
            2 * self.num_harmonics,
        )

        torch.nn.init.zeros_(self.f_mu_head.weight)
        torch.nn.init.zeros_(self.f_mu_head.bias)
        torch.nn.init.zeros_(self.f_logrho2_head.weight)
        torch.nn.init.zeros_(self.f_logrho2_head.bias)
        torch.nn.init.zeros_(self.logamp_mu_head.weight)
        torch.nn.init.zeros_(self.logamp_mu_head.bias)        # log-amp≈0 => amp≈1
        torch.nn.init.zeros_(self.logamp_logvar_head.weight)
        torch.nn.init.constant_(self.logamp_logvar_head.bias, float(c_init_logvar))
        torch.nn.init.zeros_(self.z1_head.weight)
        torch.nn.init.constant_(self.z1_head.bias[: self.num_harmonics], 0.0)
        torch.nn.init.constant_(self.z1_head.bias[self.num_harmonics :], float(z1_init_logvar))
        torch.nn.init.zeros_(self.delta_head.weight)
        torch.nn.init.constant_(self.delta_head.bias[: self.num_harmonics], 0.0)
        torch.nn.init.constant_(self.delta_head.bias[self.num_harmonics :], float(delta_init_logvar))

        if freq_lower_hz is None:
            freq_lower_hz = np.asarray([167.0, 341.0, 635.0, 872.0]) * 0.95
        if freq_upper_hz is None:
            freq_upper_hz = np.asarray([167.0, 341.0, 635.0, 872.0]) * 1.05
        if len(freq_lower_hz) != self.num_harmonics:
            raise ValueError("len(freq_lower_hz) must equal output_dim")
        if len(freq_upper_hz) != self.num_harmonics:
            raise ValueError("len(freq_upper_hz) must equal output_dim")
        freq_lower = torch.tensor(freq_lower_hz, dtype=torch.float32)
        freq_upper = torch.tensor(freq_upper_hz, dtype=torch.float32)
        if torch.any(freq_lower <= 0):
            raise ValueError("frequency lower bounds must be positive")
        if torch.any(freq_upper <= freq_lower):
            raise ValueError("frequency upper bounds must exceed lower bounds")

        self.register_buffer("freq_lower", freq_lower)
        self.register_buffer("freq_upper", freq_upper)
        self.register_buffer("freq_mid", 0.5 * (freq_lower + freq_upper))
        self.register_buffer("freq_half", 0.5 * (freq_upper - freq_lower))
        self.register_buffer("f_center", 0.5 * (freq_lower + freq_upper))
        self.register_buffer("f_band", 0.5 * (freq_upper - freq_lower))
        self.min_log_rho2 = float(min_log_rho2)
        self.max_log_rho2 = float(max_log_rho2)

    @staticmethod
    def reparameterize(mu, logvar, sample=True):
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def sample_truncated_frequency(self, mu_f, std_f, sample=True, eps=1e-6):
        """
        Inverse-CDF reparameterized sampling from the strict band-truncated
        Gaussian q(f|Y). This keeps f inside [freq_lower, freq_upper].
        """
        if not sample:
            return mu_f
        std_f = std_f.clamp_min(eps)
        lower = self.freq_lower.to(device=mu_f.device, dtype=mu_f.dtype).view(1, -1)
        upper = self.freq_upper.to(device=mu_f.device, dtype=mu_f.dtype).view(1, -1)
        alpha = (lower - mu_f) / std_f
        beta = (upper - mu_f) / std_f
        cdf_alpha = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))
        cdf_beta = 0.5 * (1.0 + torch.erf(beta / math.sqrt(2.0)))
        mass = (cdf_beta - cdf_alpha).clamp_min(eps)
        u = torch.rand_like(mu_f)
        target_cdf = (cdf_alpha + u * mass).clamp(eps, 1.0 - eps)
        normal = torch.distributions.Normal(
            torch.zeros_like(target_cdf),
            torch.ones_like(target_cdf),
        )
        z = normal.icdf(target_cdf)
        f_sample = mu_f + std_f * z
        return torch.minimum(torch.maximum(f_sample, lower), upper)

    def encode_local(self, features):
        if features.ndim != 4:
            raise ValueError(
                f"features must have shape [N, B, P, d_in], got {features.shape}"
            )
        batch_size, num_steps, points_per_step, input_dim = features.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")

        flat = features.reshape(batch_size * num_steps, points_per_step, input_dim)
        point_features = self.point_proj(flat)
        local_seq, _ = self.local_rnn(point_features)
        h_forward_last = local_seq[:, -1, : self.local_hidden_dim]
        h_backward_first = local_seq[:, 0, self.local_hidden_dim :]
        endpoints = torch.cat([h_forward_last, h_backward_first], dim=-1)
        local = self.local_endpoint_proj(endpoints)
        return local.reshape(batch_size, num_steps, self.local_out_dim)

    def encode_context(self, e_seq):
        h_bar, _ = self.context_rnn(e_seq)
        h_forward_last = h_bar[:, -1, : self.context_hidden_dim]
        h_backward_first = h_bar[:, 0, self.context_hidden_dim :]
        h_global = torch.cat([h_forward_last, h_backward_first], dim=-1)
        return h_bar, h_global

    def frequency_posterior(self, h_global):
        raw_mu = self.f_mu_head(h_global)
        raw_logrho2 = self.f_logrho2_head(h_global)
        mu_f = self.freq_mid + self.freq_half * torch.tanh(raw_mu)
        log_rho2 = self.min_log_rho2 + (
            self.max_log_rho2 - self.min_log_rho2
        ) * torch.sigmoid(raw_logrho2)
        std_f = self.freq_half * torch.exp(0.5 * log_rho2)
        logvar_f = 2.0 * torch.log(std_f + 1e-12)
        return mu_f, logvar_f, std_f, log_rho2

    def forward(self, features, delta_s, sample=True, tau=None, probe_ids=None):
        del tau, probe_ids
        if delta_s.ndim != 2:
            raise ValueError(f"delta_s must have shape [N, B], got {delta_s.shape}")

        e_seq = self.encode_local(features)
        h_bar, h_global = self.encode_context(e_seq)
        batch_size, num_steps, _ = h_bar.shape
        if delta_s.shape != (batch_size, num_steps):
            raise ValueError(
                f"delta_s shape {delta_s.shape} must match [N, B]={h_bar.shape[:2]}"
            )

        mu_f, logvar_f, std_f, log_rho2 = self.frequency_posterior(h_global)
        f_sample = self.sample_truncated_frequency(mu_f, std_f, sample=sample)

        logamp_mu = self.logamp_mu_head(h_global)
        logamp_logvar = self.logamp_logvar_head(h_global).clamp(min=-20.0, max=10.0)
        logamp_sample = self.reparameterize(logamp_mu, logamp_logvar, sample=sample)
        amp_sample = torch.exp(logamp_sample)   # [N, K] 实正
        amp_mu = torch.exp(logamp_mu)           # 解码用点估计（中位数）

        z1_params = self.z1_head(torch.cat([h_global, f_sample], dim=-1))
        z1_mu = z1_params[:, : self.num_harmonics]
        z1_logvar = z1_params[:, self.num_harmonics :].clamp(min=-20.0, max=10.0)
        z_prev = self.reparameterize(z1_mu, z1_logvar, sample=sample)

        z_seq = [z_prev]
        delta_mus = []
        delta_logvars = []
        delta_samples = []
        r = torch.zeros(
            batch_size,
            self.delta_rnn.hidden_size,
            device=features.device,
            dtype=features.dtype,
        )
        for step in range(1, num_steps):
            ds = delta_s[:, step : step + 1].to(device=features.device, dtype=features.dtype)
            delta_input = torch.cat([h_bar[:, step], f_sample, z_prev, ds], dim=-1)
            r = self.delta_rnn(delta_input, r)
            delta_params = self.delta_head(r)
            delta_mu = delta_params[:, : self.num_harmonics]
            delta_logvar = delta_params[:, self.num_harmonics :].clamp(min=-20.0, max=10.0)
            delta_z = self.reparameterize(delta_mu, delta_logvar, sample=sample)
            z_prev = z_prev + 2.0 * math.pi * f_sample * ds + delta_z
            z_seq.append(z_prev)
            delta_mus.append(delta_mu)
            delta_logvars.append(delta_logvar)
            delta_samples.append(delta_z)

        if delta_mus:
            delta_mu = torch.stack(delta_mus, dim=1)
            delta_logvar = torch.stack(delta_logvars, dim=1)
            delta_sample = torch.stack(delta_samples, dim=1)
        else:
            empty = features.new_zeros(batch_size, 0, self.num_harmonics)
            delta_mu = empty
            delta_logvar = empty
            delta_sample = empty

        return {
            "h_bar": h_bar,
            "h_global": h_global,
            "mu_f": mu_f,
            "logvar_f": logvar_f,
            "std_f": std_f,
            "log_rho2_f": log_rho2,
            "f_sample": f_sample,
            "logamp_mu": logamp_mu,
            "logamp_logvar": logamp_logvar,
            "amp_sample": amp_sample,
            "amp_mu": amp_mu,
            "z1_mu": z1_mu,
            "z1_logvar": z1_logvar,
            "z1_sample": z_seq[0],
            "delta_mu": delta_mu,
            "delta_logvar": delta_logvar,
            "delta_sample": delta_sample,
            "z_seq": torch.stack(z_seq, dim=1),
        }
