import numpy as np
import torch
from torch.utils.data import Dataset, Subset


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
    Build version-1 BTT token features.
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
    Slice input points into patches by revolution.
    The input raw point sequence should be ordered by (rev_id, probe_id).
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
        amp_agg_patches=1,
        amp_agg_mode="center",
    ):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.t_samples = torch.as_tensor(t_samples, dtype=torch.float32)
        self.rev_ids = torch.as_tensor(rev_ids, dtype=torch.long)
        self.probe_ids = torch.as_tensor(probe_ids, dtype=torch.long)

        self.window_revs = window_revs
        self.hop_revs = hop_revs
        self.num_probes = num_probes
        self.amp_agg_patches = int(amp_agg_patches)
        self.amp_agg_mode = amp_agg_mode
        if self.amp_agg_patches < 1:
            raise ValueError(f"amp_agg_patches must be >= 1, got {self.amp_agg_patches}")

        self.seq_len = window_revs * num_probes
        self.hop_len = hop_revs * num_probes

        n_points = len(features)

        self.starts = list(range(0, n_points - self.seq_len + 1, self.hop_len))
        if len(self.starts) == 0:
            raise ValueError(
                f"No patches generated. n_points={n_points}, "
                f"seq_len={self.seq_len}. Reduce window_revs."
            )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len

        x_feat = self.features[start:end]       # [L, 6]
        t = self.t_samples[start:end]           # [L]
        probe_id = self.probe_ids[start:end]    # [L]
        rev_id = self.rev_ids[start:end]        # [L]

        # Target signal for reconstruction loss: complex signal represented by real/imag.
        target = x_feat[:, :2]                  # [L, 2]

        # Amplitude LS aggregation window from neighboring patch starts.
        if self.amp_agg_patches == 1:
            amp_start = start
            amp_end = end
        else:
            if self.amp_agg_mode != "center":
                raise ValueError(f"Unsupported amp_agg_mode={self.amp_agg_mode}, expected 'center'.")

            half = self.amp_agg_patches // 2
            patch_start_idx = idx - half
            patch_end_idx = patch_start_idx + self.amp_agg_patches

            if patch_start_idx < 0:
                patch_start_idx = 0
                patch_end_idx = min(len(self.starts), self.amp_agg_patches)

            if patch_end_idx > len(self.starts):
                patch_end_idx = len(self.starts)
                patch_start_idx = max(0, len(self.starts) - self.amp_agg_patches)

            amp_start = self.starts[patch_start_idx]
            amp_last_start = self.starts[patch_end_idx - 1]
            amp_end = amp_last_start + self.seq_len

        amp_feat = self.features[amp_start:amp_end]       # [L_amp, 6]
        amp_t = self.t_samples[amp_start:amp_end]         # [L_amp]
        amp_target = amp_feat[:, :2]                      # [L_amp, 2]

        return x_feat, t, probe_id, rev_id, target, amp_t, amp_target


def chronological_train_val_split(dataset, val_ratio=0.2):
    """
    Split windows in chronological order and leave a guard gap so validation
    windows start strictly after the training windows end.
    """
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError(f"Need at least 2 windows for train/val split, got {n_total}.")

    n_val = max(1, int(round(n_total * val_ratio)))
    n_train_target = max(1, n_total - n_val)
    if n_train_target + n_val > n_total:
        n_val = n_total - n_train_target

    base_gap = max(0, int(np.ceil(dataset.seq_len / dataset.hop_len)) - 1)
    amp_gap = max(0, int(getattr(dataset, "amp_agg_patches", 1)) // 2)
    gap_windows = max(base_gap, amp_gap)
    max_train_end = n_total - n_val - gap_windows
    train_end = min(n_train_target, max_train_end)

    if train_end <= 0:
        raise ValueError(
            "Not enough windows to create a chronological split without overlap. "
            f"n_total={n_total}, n_val={n_val}, gap_windows={gap_windows}."
        )

    val_start = train_end + gap_windows
    train_indices = list(range(train_end))
    val_indices = list(range(val_start, n_total))

    if len(val_indices) == 0:
        raise ValueError(
            "Chronological split produced an empty validation set. "
            f"n_total={n_total}, train_end={train_end}, gap_windows={gap_windows}."
        )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        {
            "n_total": n_total,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "gap_windows": gap_windows,
            "train_end_idx": train_indices[-1],
            "val_start_idx": val_indices[0],
        },
    )
