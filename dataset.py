import numpy as np
import torch
from torch.utils.data import Dataset

from synthesis_dataset import (
    generate_one_btt_sequence,
    sample_amplitude_uniform,
    sample_frequency_uniform,
)

DATASET_MODE = {
    "iid_sequences": 0,
    "grouped_short_sequences": 1,
    "grouped_long_windows": 2,
    "fixed_param_long_sequence": 3,
}

SPLIT_ID = {
    "all": 0,
    "train": 1,
    "val": 2,
    "test": 3,
}


def _infer_dataset_mode(num_param_sets, sequences_per_param, use_long_sequence):
    if use_long_sequence and int(num_param_sets) == 1 and int(sequences_per_param) == 1:
        return DATASET_MODE["fixed_param_long_sequence"]
    if use_long_sequence:
        return DATASET_MODE["grouped_long_windows"]
    if int(num_param_sets) > 1 or int(sequences_per_param) > 1:
        return DATASET_MODE["grouped_short_sequences"]
    return DATASET_MODE["iid_sequences"]


def build_btt_point_features(
    x_observed,
    t_samples,
    rev_ids,
    probe_ids,
    theta_samples,
    freqs_at_samples,
    base_freq,
    n_revs,
    include_local_time_norm=False,
):
    """
    Build BTT token features.

    Returns:
        features:  float32 array, shape [N, 6] or [N, 7]
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

    feature_list = [
        x_real,
        x_imag,
        sin_theta,
        cos_theta,
        rev_norm,
        speed_norm,
    ]

    if include_local_time_norm:
        t0 = t_samples[0]
        duration = t_samples[-1] - t0
        local_time_norm = (t_samples - t0) / (duration + 1e-12)
        feature_list.append(local_time_norm)

    features = np.stack(feature_list, axis=-1).astype(np.float32)

    return (
        features,
        t_samples.astype(np.float32),
        rev_ids.astype(np.int64),
        probe_ids.astype(np.int64),
    )


def _build_sample_item(
    x_observed,
    t_samples,
    rev_ids,
    probe_ids,
    theta_samples,
    freqs_at_samples,
    base_freq,
    n_revs_for_feature,
    freq_hz,
    amp_real,
    amp_imag,
    noise_power,
    normalization="per_sequence_std",
    include_local_time_norm=False,
    dataset_mode=None,
    split_id=None,
    use_long_sequence=False,
    chronological_split=False,
    num_param_sets=None,
    sequences_per_param=None,
    param_group_id=None,
    parent_sequence_id=None,
    window_start_cycle=None,
    window_hop_cycles=None,
    short_num_cycles=None,
    long_sequence_num_cycles=None,
    train_ratio=None,
):
    """
    Build one model-ready sample from a full parent sequence or a sliced window.
    """

    if normalization == "per_sequence_std":
        amp_scale = float(np.std(x_observed))
        x_observed_norm = x_observed / (amp_scale + 1e-12)
    elif normalization in (None, "none"):
        amp_scale = 1.0
        x_observed_norm = x_observed
    else:
        raise ValueError(f"Unsupported normalization={normalization}")
    noise_var_norm = float(noise_power) / ((amp_scale + 1e-12) ** 2)

    local_rev_ids = rev_ids - int(rev_ids[0])
    features, t_samples, local_rev_ids, probe_ids = build_btt_point_features(
        x_observed=x_observed_norm,
        t_samples=t_samples,
        rev_ids=local_rev_ids,
        probe_ids=probe_ids,
        theta_samples=theta_samples,
        freqs_at_samples=freqs_at_samples,
        base_freq=base_freq,
        n_revs=n_revs_for_feature,
        include_local_time_norm=include_local_time_norm,
    )
    target = features[:, :2]

    item = {
        "x": torch.as_tensor(features, dtype=torch.float32),
        "t": torch.as_tensor(t_samples, dtype=torch.float32),
        "probe_ids": torch.as_tensor(probe_ids, dtype=torch.long),
        "rev_ids": torch.as_tensor(local_rev_ids, dtype=torch.long),
        "target": torch.as_tensor(target, dtype=torch.float32),
        "true_freq_hz": torch.as_tensor(freq_hz, dtype=torch.float32),
        "true_amp_real": torch.as_tensor(amp_real, dtype=torch.float32),
        "true_amp_imag": torch.as_tensor(amp_imag, dtype=torch.float32),
        "amp_scale": torch.as_tensor(amp_scale, dtype=torch.float32),
        "noise_var_norm": torch.as_tensor(noise_var_norm, dtype=torch.float32),
        "dataset_mode": torch.as_tensor(
            -1 if dataset_mode is None else int(dataset_mode),
            dtype=torch.long,
        ),
        "split_id": torch.as_tensor(
            -1 if split_id is None else int(split_id),
            dtype=torch.long,
        ),
        "use_long_sequence": torch.as_tensor(int(bool(use_long_sequence)), dtype=torch.long),
        "chronological_split": torch.as_tensor(
            int(bool(chronological_split)),
            dtype=torch.long,
        ),
        "num_param_sets": torch.as_tensor(
            -1 if num_param_sets is None else int(num_param_sets),
            dtype=torch.long,
        ),
        "sequences_per_param": torch.as_tensor(
            -1 if sequences_per_param is None else int(sequences_per_param),
            dtype=torch.long,
        ),
        "param_group_id": torch.as_tensor(
            -1 if param_group_id is None else int(param_group_id),
            dtype=torch.long,
        ),
        "parent_sequence_id": torch.as_tensor(
            -1 if parent_sequence_id is None else int(parent_sequence_id),
            dtype=torch.long,
        ),
        "window_start_cycle": torch.as_tensor(
            -1 if window_start_cycle is None else int(window_start_cycle),
            dtype=torch.long,
        ),
        "window_hop_cycles": torch.as_tensor(
            -1 if window_hop_cycles is None else int(window_hop_cycles),
            dtype=torch.long,
        ),
        "short_num_cycles": torch.as_tensor(
            -1 if short_num_cycles is None else int(short_num_cycles),
            dtype=torch.long,
        ),
        "long_sequence_num_cycles": torch.as_tensor(
            -1 if long_sequence_num_cycles is None else int(long_sequence_num_cycles),
            dtype=torch.long,
        ),
        "train_ratio_x10000": torch.as_tensor(
            -1 if train_ratio is None else int(round(float(train_ratio) * 10000)),
            dtype=torch.long,
        ),
        "is_windowed": torch.as_tensor(int(bool(use_long_sequence)), dtype=torch.long),
        "is_iid_sequence": torch.as_tensor(
            int(not bool(use_long_sequence)),
            dtype=torch.long,
        ),
    }

    return item


class BTTSequenceDataset(Dataset):
    """
    Independent short BTT sequences generated by sampling physical parameters.
    """

    def __init__(
        self,
        num_sequences,
        num_cycles,
        num_probes,
        base_freq,
        fluctuation_delta,
        probe_angles,
        freq_lower,
        freq_upper,
        amp_real_center,
        amp_imag_center,
        amp_relative_half_band,
        amp_min_half_band,
        snr_db,
        seed=0,
        normalization="per_sequence_std",
        include_local_time_norm=False,
        split="all",
    ):
        self.num_sequences = int(num_sequences)
        self.num_cycles = int(num_cycles)
        self.num_probes = int(num_probes)
        self.base_freq = float(base_freq)
        self.fluctuation_delta = float(fluctuation_delta)
        self.probe_angles = probe_angles
        self.freq_lower = np.asarray(freq_lower, dtype=np.float64)
        self.freq_upper = np.asarray(freq_upper, dtype=np.float64)
        self.amp_real_center = np.asarray(amp_real_center, dtype=np.float64)
        self.amp_imag_center = np.asarray(amp_imag_center, dtype=np.float64)
        self.amp_relative_half_band = float(amp_relative_half_band)
        self.amp_min_half_band = float(amp_min_half_band)
        self.snr_db = snr_db
        self.seed = int(seed)
        self.normalization = normalization
        self.include_local_time_norm = bool(include_local_time_norm)
        self.window_num_cycles = self.num_cycles
        self.split = split
        self.dataset_mode = DATASET_MODE["iid_sequences"]

        if self.split not in SPLIT_ID:
            raise ValueError(f"Unsupported split={self.split!r}")

        if self.freq_lower.shape != self.freq_upper.shape:
            raise ValueError("freq_lower and freq_upper must have the same shape")
        if self.amp_real_center.shape != self.freq_lower.shape:
            raise ValueError("amp_real_center must match frequency shape")
        if self.amp_imag_center.shape != self.freq_lower.shape:
            raise ValueError("amp_imag_center must match frequency shape")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))

        freq_hz = sample_frequency_uniform(
            self.freq_lower,
            self.freq_upper,
            rng,
        )
        amp_real, amp_imag = sample_amplitude_uniform(
            amp_real_center=self.amp_real_center,
            amp_imag_center=self.amp_imag_center,
            relative_half_band=self.amp_relative_half_band,
            min_half_band=self.amp_min_half_band,
            rng=rng,
        )

        sample = generate_one_btt_sequence(
            num_cycles=self.num_cycles,
            base_freq=self.base_freq,
            fluctuation_delta=self.fluctuation_delta,
            probe_angles=self.probe_angles,
            freq_hz=freq_hz,
            amp_real=amp_real,
            amp_imag=amp_imag,
            snr_db=self.snr_db,
            rng=rng,
        )

        return _build_sample_item(
            x_observed=sample["x_observed"],
            t_samples=sample["t_samples"],
            rev_ids=sample["rev_ids"],
            probe_ids=sample["probe_ids"],
            theta_samples=sample["theta_samples"],
            freqs_at_samples=sample["freqs_at_samples"],
            base_freq=self.base_freq,
            n_revs_for_feature=self.num_cycles,
            freq_hz=freq_hz,
            amp_real=amp_real,
            amp_imag=amp_imag,
            noise_power=sample["noise_power"],
            normalization=self.normalization,
            include_local_time_norm=self.include_local_time_norm,
            dataset_mode=self.dataset_mode,
            split_id=SPLIT_ID[self.split],
            use_long_sequence=False,
            chronological_split=False,
            num_param_sets=self.num_sequences,
            sequences_per_param=1,
            param_group_id=int(idx),
            parent_sequence_id=0,
            window_start_cycle=0,
            window_hop_cycles=1,
            short_num_cycles=self.num_cycles,
            long_sequence_num_cycles=self.num_cycles,
            train_ratio=None,
        )


class GroupedBTTSequenceDataset(Dataset):
    """
    General grouped synthetic BTT dataset.

    Supports many parameter groups, multiple parent sequences per group, and
    optional long parent sequences sliced into short model windows.
    """

    def __init__(
        self,
        split,
        num_param_sets,
        sequences_per_param,
        use_long_sequence,
        short_num_cycles,
        long_sequence_num_cycles,
        window_hop_cycles,
        chronological_split,
        train_ratio,
        num_probes,
        base_freq,
        fluctuation_delta,
        probe_angles,
        freq_lower,
        freq_upper,
        amp_real_center,
        amp_imag_center,
        amp_relative_half_band,
        amp_min_half_band,
        snr_db,
        seed=0,
        normalization="per_sequence_std",
        include_local_time_norm=False,
    ):
        if split not in SPLIT_ID:
            raise ValueError(f"split must be one of {sorted(SPLIT_ID)}, got {split!r}")

        self.split = split
        self.num_param_sets = int(num_param_sets)
        self.sequences_per_param = int(sequences_per_param)
        self.use_long_sequence = bool(use_long_sequence)
        self.short_num_cycles = int(short_num_cycles)
        self.long_sequence_num_cycles = int(long_sequence_num_cycles)
        self.window_hop_cycles = int(window_hop_cycles)
        self.chronological_split = bool(chronological_split)
        self.train_ratio = float(train_ratio)
        self.num_probes = int(num_probes)
        self.base_freq = float(base_freq)
        self.fluctuation_delta = float(fluctuation_delta)
        self.probe_angles = probe_angles
        self.snr_db = snr_db
        self.seed = int(seed)
        self.normalization = normalization
        self.include_local_time_norm = bool(include_local_time_norm)
        self.window_num_cycles = self.short_num_cycles
        self.dataset_mode = _infer_dataset_mode(
            num_param_sets=self.num_param_sets,
            sequences_per_param=self.sequences_per_param,
            use_long_sequence=self.use_long_sequence,
        )

        if self.num_param_sets <= 0:
            raise ValueError("num_param_sets must be positive")
        if self.sequences_per_param <= 0:
            raise ValueError("sequences_per_param must be positive")
        if self.short_num_cycles <= 0:
            raise ValueError("short_num_cycles must be positive")
        if self.window_hop_cycles <= 0:
            raise ValueError("window_hop_cycles must be positive")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0, 1)")

        if self.use_long_sequence:
            if self.long_sequence_num_cycles < self.short_num_cycles:
                raise ValueError("long_sequence_num_cycles must be >= short_num_cycles")
            self.parent_num_cycles = self.long_sequence_num_cycles
        else:
            self.parent_num_cycles = self.short_num_cycles

        if self.chronological_split and split not in ("train", "val"):
            raise ValueError(
                "Use split='train' or split='val' when chronological_split=True"
            )

        self.freq_lower = np.asarray(freq_lower, dtype=np.float64)
        self.freq_upper = np.asarray(freq_upper, dtype=np.float64)
        self.amp_real_center = np.asarray(amp_real_center, dtype=np.float64)
        self.amp_imag_center = np.asarray(amp_imag_center, dtype=np.float64)
        self.amp_relative_half_band = float(amp_relative_half_band)
        self.amp_min_half_band = float(amp_min_half_band)

        if self.freq_lower.shape != self.freq_upper.shape:
            raise ValueError("freq_lower and freq_upper must have the same shape")
        if self.amp_real_center.shape != self.freq_lower.shape:
            raise ValueError("amp_real_center must match frequency shape")
        if self.amp_imag_center.shape != self.freq_lower.shape:
            raise ValueError("amp_imag_center must match frequency shape")

        rng = np.random.default_rng(self.seed)
        self.param_freq = []
        self.param_amp_real = []
        self.param_amp_imag = []
        for _ in range(self.num_param_sets):
            freq_hz = sample_frequency_uniform(self.freq_lower, self.freq_upper, rng)
            amp_real, amp_imag = sample_amplitude_uniform(
                amp_real_center=self.amp_real_center,
                amp_imag_center=self.amp_imag_center,
                relative_half_band=self.amp_relative_half_band,
                min_half_band=self.amp_min_half_band,
                rng=rng,
            )
            self.param_freq.append(freq_hz)
            self.param_amp_real.append(amp_real)
            self.param_amp_imag.append(amp_imag)
        self.param_freq = np.asarray(self.param_freq, dtype=np.float64)
        self.param_amp_real = np.asarray(self.param_amp_real, dtype=np.float64)
        self.param_amp_imag = np.asarray(self.param_amp_imag, dtype=np.float64)

        self.parent_specs = []
        self.index = []
        self.parent_cache = {}
        for param_id in range(self.num_param_sets):
            for seq_id in range(self.sequences_per_param):
                parent_id = len(self.parent_specs)
                self.parent_specs.append((param_id, seq_id))
                if self.use_long_sequence:
                    for start_cycle in self._make_window_start_cycles(
                        total_cycles=self.long_sequence_num_cycles
                    ):
                        self.index.append((parent_id, int(start_cycle)))
                else:
                    self.index.append((parent_id, 0))

        if len(self.index) == 0:
            raise ValueError(f"No samples generated for split={split}. Check dataset config.")

    def _make_window_start_cycles(self, total_cycles):
        all_start_cycles = np.arange(
            0,
            total_cycles - self.short_num_cycles + 1,
            self.window_hop_cycles,
            dtype=np.int64,
        )

        if not self.chronological_split:
            return all_start_cycles

        split_cycle = int(np.floor(self.train_ratio * total_cycles))
        if self.split == "train":
            return all_start_cycles[
                all_start_cycles + self.short_num_cycles <= split_cycle
            ]
        if self.split == "val":
            return all_start_cycles[all_start_cycles >= split_cycle]
        raise RuntimeError("Invalid split state")

    def __len__(self):
        return len(self.index)

    def _parent_seed(self, param_id, seq_id):
        return self.seed + 1000003 * int(param_id) + 9176 * int(seq_id)

    def _generate_parent(self, parent_id):
        if self.use_long_sequence and parent_id in self.parent_cache:
            return self.parent_cache[parent_id]

        param_id, seq_id = self.parent_specs[int(parent_id)]
        parent_rng = np.random.default_rng(self._parent_seed(param_id, seq_id))
        sample = generate_one_btt_sequence(
            num_cycles=self.parent_num_cycles,
            base_freq=self.base_freq,
            fluctuation_delta=self.fluctuation_delta,
            probe_angles=self.probe_angles,
            freq_hz=self.param_freq[param_id],
            amp_real=self.param_amp_real[param_id],
            amp_imag=self.param_amp_imag[param_id],
            snr_db=self.snr_db,
            rng=parent_rng,
        )
        if self.use_long_sequence:
            self.parent_cache[parent_id] = sample
        return sample

    def __getitem__(self, idx):
        parent_id, start_cycle = self.index[int(idx)]
        param_id, seq_id = self.parent_specs[parent_id]
        sample = self._generate_parent(parent_id)

        if self.use_long_sequence:
            end_cycle = start_cycle + self.short_num_cycles
            start = start_cycle * self.num_probes
            end = end_cycle * self.num_probes
        else:
            start = 0
            end = self.short_num_cycles * self.num_probes

        return _build_sample_item(
            x_observed=sample["x_observed"][start:end],
            t_samples=sample["t_samples"][start:end],
            rev_ids=sample["rev_ids"][start:end],
            probe_ids=sample["probe_ids"][start:end],
            theta_samples=sample["theta_samples"][start:end],
            freqs_at_samples=sample["freqs_at_samples"][start:end],
            base_freq=self.base_freq,
            n_revs_for_feature=self.short_num_cycles,
            freq_hz=self.param_freq[param_id],
            amp_real=self.param_amp_real[param_id],
            amp_imag=self.param_amp_imag[param_id],
            noise_power=sample["noise_power"],
            normalization=self.normalization,
            include_local_time_norm=self.include_local_time_norm,
            dataset_mode=self.dataset_mode,
            split_id=SPLIT_ID[self.split],
            use_long_sequence=self.use_long_sequence,
            chronological_split=self.chronological_split,
            num_param_sets=self.num_param_sets,
            sequences_per_param=self.sequences_per_param,
            param_group_id=param_id,
            parent_sequence_id=seq_id,
            window_start_cycle=start_cycle,
            window_hop_cycles=self.window_hop_cycles,
            short_num_cycles=self.short_num_cycles,
            long_sequence_num_cycles=(
                self.long_sequence_num_cycles
                if self.use_long_sequence
                else self.short_num_cycles
            ),
            train_ratio=self.train_ratio,
        )
