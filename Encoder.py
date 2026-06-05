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
