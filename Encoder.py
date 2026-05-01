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

        self._fc_f_mu = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        self._fc_f_logvar = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        self._fc_a = torch.nn.Linear(hidden_dim_dense + self.num_harmonics, hidden_dim_dense)
        self._fc_a_real_mu = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        self._fc_a_real_logvar = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        self._fc_a_imag_mu = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)
        self._fc_a_imag_logvar = torch.nn.Linear(hidden_dim_dense, self.num_harmonics)

        self._device = device
        self._causal_mask = causal_mask
        self.register_buffer(
            "f_center",
            torch.tensor(
                [167.0, 341.0, 635.0, 872.0],
                dtype=torch.float32,
            ),
        )

        self.f_band = 15.0

    def generate_causal_mask(self, seq_len):
        # Upper triangular mask: (seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf")).to(self._device)

    def generate_non_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.eye(seq_len)
        return mask.masked_fill(mask == 1, float("-inf")).to(self._device)

    def encode_context(self, x, probe_ids=None):
        """
        x:        [B, L, input_dim]
        probe_ids:[B, L]
        """
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
        return F.relu(self._fc(pooled))  # [B, hidden_dim_dense]

    def infer_frequency_posterior(self, context):
        raw_f_mu = self._fc_f_mu(context)
        mu_f = self.f_center + self.f_band * torch.tanh(raw_f_mu)
        logvar_f = torch.clamp(self._fc_f_logvar(context), min=-14.0, max=-6.0)
        return mu_f, logvar_f

    def infer_amplitude_posterior(self, context, f):
        f_cond = (f - self.f_center.unsqueeze(0)) / self.f_band
        amp_input = torch.cat([context, f_cond], dim=-1)
        amp_hidden = F.relu(self._fc_a(amp_input))

        mu_amp_real = self._fc_a_real_mu(amp_hidden)
        logvar_amp_real = torch.clamp(
            self._fc_a_real_logvar(amp_hidden),
            min=-14.0,
            max=-4.0,
        )
        mu_amp_imag = self._fc_a_imag_mu(amp_hidden)
        logvar_amp_imag = torch.clamp(
            self._fc_a_imag_logvar(amp_hidden),
            min=-14.0,
            max=-4.0,
        )
        return mu_amp_real, logvar_amp_real, mu_amp_imag, logvar_amp_imag

    def forward(self, x, probe_ids=None, Cws=None):
        context = self.encode_context(x, probe_ids=probe_ids)
        return self.infer_frequency_posterior(context)
