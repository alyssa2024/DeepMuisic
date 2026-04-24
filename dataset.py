import numpy as np
import torch
from torch.utils.data import Dataset


def build_btt_point_features(
    x_observed,
    t_samples,
    rev_ids,
    probe_ids,
    theta_samples,
    freqs_at_samples,
    base_freq,
    n_revs,
):
    """
    构造版本 1 的 BTT token 特征。

    Args:
        x_observed:       complex array, shape [N]
        t_samples:        float array, shape [N]
        rev_ids:          int array, shape [N]
        probe_ids:        int array, shape [N]
        theta_samples:    float array, radians, shape [N]
        freqs_at_samples: float array, shape [N]
        base_freq:        float
        n_revs:           int

    Returns:
        features:  float32 array, shape [N, 6]
        t_samples: float32 array, shape [N]
        rev_ids:   int64 array, shape [N]
        probe_ids: int64 array, shape [N]
    """

    x_real = np.real(x_observed)
    x_imag = np.imag(x_observed)

    sin_theta = np.sin(theta_samples)
    cos_theta = np.cos(theta_samples)

    rev_norm = rev_ids / max(n_revs - 1, 1)
    speed_norm = freqs_at_samples / base_freq

    features = np.stack(
        [
            x_real,
            x_imag,
            sin_theta,
            cos_theta,
            rev_norm,
            speed_norm,
        ],
        axis=-1,
    ).astype(np.float32)

    return (
        features,
        t_samples.astype(np.float32),
        rev_ids.astype(np.int64),
        probe_ids.astype(np.int64),
    )

class BTTPatchDataset(Dataset):
    """
    按圈切 patch。
    输入原始点序列应按 rev_id, probe_id 顺序排列。
    """

    def __init__(
        self,
        features,
        t_samples,
        rev_ids,
        probe_ids,
        window_revs=64,
        hop_revs=32,
        num_probes=4,
    ):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.t_samples = torch.as_tensor(t_samples, dtype=torch.float32)
        self.rev_ids = torch.as_tensor(rev_ids, dtype=torch.long)
        self.probe_ids = torch.as_tensor(probe_ids, dtype=torch.long)

        self.window_revs = window_revs
        self.hop_revs = hop_revs
        self.num_probes = num_probes

        self.seq_len = window_revs * num_probes
        self.hop_len = hop_revs * num_probes

        n_points = len(features)

        if len(self.starts) == 0:
            raise ValueError(
                f"No patches generated. n_points={n_points}, "
                f"seq_len={self.seq_len}. Reduce window_revs."
            )
        
        self.starts = list(range(0, n_points - self.seq_len + 1, self.hop_len))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len

        x_feat = self.features[start:end]       # [L, 6]
        t = self.t_samples[start:end]           # [L]
        probe_id = self.probe_ids[start:end]    # [L]
        rev_id = self.rev_ids[start:end]        # [L]

        # 用于重建 loss 的目标信号：复数信号由 real/imag 组成
        target = x_feat[:, :2]                  # [L, 2]

        return x_feat, t, probe_id, rev_id, target