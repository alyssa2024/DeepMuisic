"""统一数据集: 一份底层 y,t_n -> 时序视图 + 协方差视图 + 标签。

同一批数据同时支持:
    时序输入  signal[2,N] (+ t_n[N])   -> regression_cnn / classification_cnn / rope
    协方差输入 cov_channels[3,D,D]       -> resfreq
所以四种编码器在完全相同的物理样本上对照, 且天然支持在线现采 (无固定缓存)。
可选 CachedUnifiedDataset 预生成固定样本做可复现对照。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from config import grid_size, resolve_probe_angles, signal_dim as _signal_dim
from synthesis_dataset import generate_one_sample


def make_freq_grid(cfg):
    fcfg = cfg["frequency"]
    return np.linspace(fcfg["f_min"], fcfg["f_max"], grid_size(cfg)).astype(np.float32)


def gaussian_target(freqs_active_hz, grid_hz, sigma_hz):
    """DeepFreq 原生高斯核伪谱目标 [G] (回归/rope 用)。"""
    target = np.zeros(grid_hz.shape[0], dtype=np.float32)
    sigma = max(float(sigma_hz), 1e-6)
    for f in freqs_active_hz:
        target += np.exp(-((grid_hz - float(f)) ** 2) / (sigma ** 2)).astype(np.float32)
    return target


def multihot_label(freqs_active_hz, grid_hz):
    """最近网格点多热标签 [G] (分类/resfreq 用)。"""
    label = np.zeros(grid_hz.shape[0], dtype=np.float32)
    for f in freqs_active_hz:
        label[int(np.argmin(np.abs(grid_hz - float(f))))] = 1.0
    return label


def build_covariance_channels(y, cfg):
    """复信号 y[N] -> 协方差图 [3,D,D] (Re/Im/angle)。

    把 y 按 (探头, 转) 结构重排为快拍矩阵 Y[D,Q], D=block_revs*n_probes,
    Q=num_snapshots, 相邻快拍在转轴上滑窗铺满 N; R = Y Y^H / Q。
    """
    ccfg = cfg["covariance"]
    n_probe = cfg["data"]["n_probes"]
    block_revs = ccfg["block_revs"]
    q = ccfg["num_snapshots"]
    d = block_revs * n_probe
    n = len(y)
    total_revs = n // n_probe

    if total_revs < block_revs:
        raise ValueError(
            f"signal too short for covariance: total_revs={total_revs} < block_revs={block_revs}"
        )
    # 快拍在转轴上均匀铺开 (首尾对齐), 每快拍取 block_revs 连续转
    if q == 1:
        starts = [0]
    else:
        max_start = total_revs - block_revs
        starts = np.linspace(0, max_start, q).round().astype(int)

    y_rev = y.reshape(total_revs, n_probe)          # [rev, probe]
    cols = []
    for s in starts:
        block = y_rev[s : s + block_revs].reshape(-1)   # [D]
        cols.append(block)
    Y = np.stack(cols, axis=1)                      # [D, Q]

    R = (Y @ Y.conj().T) / q                        # [D, D]
    cov = np.stack([R.real, R.imag, np.angle(R)], axis=0).astype(np.float32)
    return cov


class UnifiedBTTDataset(Dataset):
    """在线现采: 每次 __getitem__ 生成一条新样本 (无固定 train/val 泄漏)。

    fixed_snr_db: 评测时固定单一 SNR; None 时训练用 (无噪声, 幅值 floor 已提供动态范围)。
    可传 snr_db 使数据带噪。
    """

    def __init__(self, cfg, size, seed=0, snr_db=None):
        self.cfg = cfg
        self.size = int(size)
        self.seed = int(seed)
        self.snr_db = snr_db
        self.probe_angles = resolve_probe_angles(cfg)
        self.grid = make_freq_grid(cfg)

    def __len__(self):
        return self.size

    def _generate(self, rng):
        dcfg = self.cfg["data"]
        scfg = self.cfg["signal"]
        # snr_db 可为: None (无噪) / 标量 (固定, 评测) / list (混训, 每样本抽一档)
        snr = self.snr_db
        if isinstance(snr, (list, tuple)):
            snr = float(snr[int(rng.integers(len(snr)))]) if len(snr) else None
        sample = generate_one_sample(
            signal_dim=_signal_dim(self.cfg),
            num_components=scfg["num_components"],
            variable_num_freq=scfg["variable_num_freq"],
            min_sep_hz=scfg["min_sep_hz"],
            f_lo=scfg["f_lo"],
            f_hi=scfg["f_hi"],
            distance=scfg["distance"],
            amplitude=scfg["amplitude"],
            floor_amplitude=scfg["floor_amplitude"],
            regime=dcfg["regime"],
            probe_angles_deg=self.probe_angles,
            f_rot_hz=dcfg["f_rot_hz"],
            speed_fluct=dcfg["speed_fluct"],
            rng=rng,
            snr_db=snr,
        )
        return sample

    def _encode(self, sample):
        y = sample["y"]
        freqs_padded = sample["freqs_hz"]
        n_freq = sample["n_freq"]
        freqs_active = freqs_padded[:n_freq]

        # 时序视图 [2,N]
        signal = np.stack([y.real, y.imag], axis=0).astype(np.float32)
        # 协方差视图 [3,D,D]
        cov = build_covariance_channels(y, self.cfg)
        # 标签
        target = gaussian_target(freqs_active, self.grid, self.cfg["frequency"]["target_sigma_hz"])
        label = multihot_label(freqs_active, self.grid)

        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "t_n": torch.as_tensor(sample["t_n"], dtype=torch.float32),
            "cov_channels": torch.as_tensor(cov, dtype=torch.float32),
            "fr_target": torch.as_tensor(target, dtype=torch.float32),
            "multihot": torch.as_tensor(label, dtype=torch.float32),
            "freqs_hz": torch.as_tensor(freqs_padded, dtype=torch.float32),
            "n_freq": torch.as_tensor(n_freq, dtype=torch.long),
        }

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        return self._encode(self._generate(rng))


class CachedUnifiedDataset(Dataset):
    """预生成 cache_size 条固定样本 (可复现对照)。底层与在线版共用编码逻辑。"""

    def __init__(self, cfg, cache_size, seed=0, snr_db=None):
        base = UnifiedBTTDataset(cfg, cache_size, seed=seed, snr_db=snr_db)
        self._items = [base[i] for i in range(cache_size)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[int(idx)]
