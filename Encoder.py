import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gin
import numpy as np


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

        self._fc_amp_real = torch.nn.Linear(hidden_dim_dense, 2 * self.num_harmonics)
        self._fc_amp_imag = torch.nn.Linear(hidden_dim_dense, 2 * self.num_harmonics)
        self._fc_f_mu = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        self._fc_f_logvar = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)

        self._device = device
        self._causal_mask = causal_mask
        self._softplus = torch.nn.Softplus(beta=1.0)

        self.register_buffer(
            "f_center",
            torch.tensor(
                [167.0, 341.0, 635.0, 872.0],
                dtype=torch.float32,
            ),
        )

        self.f_band = 150.0

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

        params_amp_real = self._fc_amp_real(y)
        mu_amp_real = params_amp_real[..., : self.num_harmonics]
        logvar_amp_real = torch.clamp(
            params_amp_real[..., self.num_harmonics :],
            min=-14.0,
            max=4.0,
        )

        params_amp_imag = self._fc_amp_imag(y)
        mu_amp_imag = params_amp_imag[..., : self.num_harmonics]
        logvar_amp_imag = torch.clamp(
            params_amp_imag[..., self.num_harmonics :],
            min=-14.0,
            max=4.0,
        )

        raw_f_mu = self._fc_f_mu(y)
        mu_f = self.f_center + self.f_band * torch.tanh(raw_f_mu)
        logvar_f = torch.clamp(self._fc_f_logvar(y), min=-14.0, max=-6.0)

        return (mu_amp_real, logvar_amp_real), (mu_amp_imag, logvar_amp_imag), (mu_f, logvar_f)
