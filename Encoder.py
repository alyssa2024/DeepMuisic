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

        self.use_standard_pe = use_standard_pe
        if use_standard_pe:
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        else:
            self.pos_encoder = None

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

        self._fc_f_mu = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        self._fc_f_logvar = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)

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

    def forward(self, x, probe_ids=None, Cws=None):
        """
        x:        [B, L, input_dim]
        probe_ids:[B, L]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        x_transformer = self.input_proj(x)

        if probe_ids is not None:
            x_transformer = x_transformer + self.probe_embedding(probe_ids)

        if self.use_standard_pe:
            x_transformer = self.pos_encoder(x_transformer)

        if self._causal_mask:
            mask = self.generate_causal_mask(seq_len)
        else:
            mask = None

        x_transformer = self.transformer_encoder(x_transformer, mask=mask)

        # Key step: mean pooling to produce one global latent per patch.
        pooled = x_transformer.mean(dim=1)  # [B, hidden_dim]

        y = F.relu(self._fc(pooled))  # [B, hidden_dim_dense]

        raw_f_mu = self._fc_f_mu(y)
        raw_logrho2_f = self._fc_f_logvar(y)

        mu_unit = torch.tanh(raw_f_mu)
        mu_f = self.freq_mid + self.freq_half * mu_unit

        log_rho2 = torch.clamp(
            raw_logrho2_f,
            min=self.min_log_rho2,
            max=self.max_log_rho2,
        )
        rho = torch.exp(0.5 * log_rho2)
        std_f = self.freq_half * rho
        logvar_f = 2.0 * torch.log(std_f + 1e-12)

        return mu_f, logvar_f, std_f
