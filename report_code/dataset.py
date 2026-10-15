"""单窗数据集: 底层 y,t_n -> FTF 的时序视图 + anchor 标签。

**全程在线现采** (每次 __getitem__ 生成新样本): 无固定 train/val 泄漏, 也不会过拟合到
一份固定缓存 —— 曾经的 5 万条缓存对这个组合空间根本不够, 1000 步就过拟合。

FTF 只吃三样: signal[2,N] / t_n[N] (进 NUDFT 相位) / rate_summary[4] (进 FiLM)。
协方差视图 [3,D,D] (resfreq 用)、高斯核伪谱目标、多热标签都已移除 —— 那些是对照
编码器的输入/标签, 每个样本都算一遍是纯浪费。原版见 unified_project/dataset.py。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from config import grid_size, resolve_probe_angles, signal_dim as _signal_dim
from synthesis_dataset import draw_speed_fluct, draw_speed_ramp, generate_one_sample


def make_freq_grid(cfg):
    fcfg = cfg["frequency"]
    return np.linspace(fcfg["f_min"], fcfg["f_max"], grid_size(cfg)).astype(np.float32)


RATE_REF_HZ = 35.0   # 兜底值 (旧 ckpt 的 config 没有 data.rate_ref_hz 这个键)


def rate_ref_hz(cfg):
    """abs_mean 维的归一化参考转速。

    ⚠ 必须 ≈ f_rot_range 的中值。它的作用是把绝对转速压到 O(1):
        abs_mean = mean(rev_speeds)/RATE_REF - 1     (35Hz 参考 -> 25Hz:-0.29 / 45Hz:+0.29)
    参考值错了 (比如 f_rot_range 换成 [50,70] 却还用 35), abs_mean 会跑到 O(1) 之外,
    FiLM 的输入分布偏移 ⇒ 这一维实质废掉 ⇒ 网络又看不到绝对转速 (那正是 recall 卡 0.117
    的死锁根因)。build_encoder 里有断言防这个。
    """
    return float(cfg["data"].get("rate_ref_hz", RATE_REF_HZ))


def rate_summary(rev_speeds, f_rot_hz, ref_hz=RATE_REF_HZ):
    """逐圈转速 rev_speeds[n_rev] (Hz) -> 4 维窗内摘要 [rel_mean, slope, var, abs_mean]。

    前 3 维用相对转速 s = rev_speeds/f_rot - 1 (O(1), 与绝对转速解耦, 抓漂移结构):
      rel_mean = 窗内平均相对转速 (≈整体偏移, 序列内漂移量)
      slope    = 逐圈线性拟合一次项 (run-up 漂移)
      var      = 去趋势残差方差 (random walk 强度)
    第 4 维补**绝对**转速 (⭐不可省):
      abs_mean = mean(rev_speeds)/ref_hz - 1
      序列间 f_rot 变化时, 前 3 维各自除以自身 f_rot 后 rel_mean≈0 —— 30Hz 和 45Hz 的样本
      长得一模一样, 网络完全看不到绝对转速。只给前 3 维时 recall 卡 0.117 (全零谱退化解),
      补上 abs_mean 后 0.663 → 0.749。
    """
    s = rev_speeds.astype(np.float64) / f_rot_hz - 1.0
    n = len(s)
    idx = np.arange(n, dtype=np.float64)
    ic = idx - idx.mean()
    mean = s.mean()
    denom = (ic ** 2).sum()
    slope = (ic * (s - mean)).sum() / denom if denom > 0 else 0.0
    resid = s - (mean + slope * ic)
    abs_mean = rev_speeds.astype(np.float64).mean() / float(ref_hz) - 1.0
    return np.array([mean, slope, resid.var(), abs_mean], dtype=np.float32)


def film_input(cfg):
    """FiLM 吃哪种转速表示: "summary" (4维摘要) / "per_rev" (n_rev 维逐圈)。

    ⚠ 用 .get 回退到 "summary" —— 旧 ckpt 的 config.json 里没有这个键 (那时只有摘要一种),
      而它们全都是摘要训的。缺键 = 旧 ckpt = summary。
    """
    return cfg["encoder"].get("film_input", "summary")


def film_input_dim(cfg):
    """FiLM 的输入维数 (= FiLM 首层 Linear 的 in_features)。

    summary -> 4      (rate_summary 的 4 维统计摘要)
    per_rev -> n_rev  (逐圈归一化转速, 不做统计压缩)
    ⚠ 改这个会改 FiLM 首层权重形状 ⇒ 旧 ckpt 加载失败。
    """
    mode = film_input(cfg)
    if mode == "summary":
        return 4
    if mode == "per_rev":
        return int(cfg["data"]["n_rev"])
    raise ValueError(f"unknown encoder.film_input={mode!r} (要 summary / per_rev)")


class UnifiedBTTDataset(Dataset):
    """在线现采: 每次 __getitem__ 生成一条新样本 (无固定 train/val 泄漏)。

    fixed_snr_db: 评测时固定单一 SNR; None 时训练用 (无噪声, 幅值 floor 已提供动态范围)。
    可传 snr_db 使数据带噪。
    """

    def __init__(self, cfg, size, seed=0, snr_db=None):
        self.cfg = cfg
        self.ref_hz = rate_ref_hz(cfg)      # abs_mean / per_rev 的归一化参考转速
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
        # snr_db 可为: None(无噪) / 标量(固定,评测) / list(混训等概率抽一档) /
        #   dict{min,max[,low_lo,low_hi,low_frac]}(连续混训, low_frac概率偏重低段强化)
        snr = self.snr_db
        if isinstance(snr, dict):
            lf = snr.get("low_frac", 0.0)
            if lf > 0 and rng.random() < lf:          # 低SNR强化: 偏重低段
                snr = float(rng.uniform(snr.get("low_lo", snr["min"]),
                                        snr.get("low_hi", snr["min"] + 10.0)))
            else:                                     # 均匀混训主体
                snr = float(rng.uniform(snr["min"], snr["max"]))
        elif isinstance(snr, (list, tuple)):
            snr = float(snr[int(rng.integers(len(snr)))]) if len(snr) else None
        # 基础转速: f_rot_range 给定则序列间均匀抽 (每样本一个基础 f_rot), 否则固定 f_rot_hz
        frr = dcfg.get("f_rot_range")
        f_rot = float(rng.uniform(frr[0], frr[1])) if frr else float(dcfg["f_rot_hz"])
        # 频漂耦合: 混训随机抽 (对齐 SNR 混训), 含 0=无漂; 否则用固定值
        cpr = dcfg.get("drift_coupling_range")
        coupling = float(rng.uniform(cpr[0], cpr[1])) if cpr else dcfg.get("drift_coupling", 0.5)
        # 转速廓线两旋钮: 各自按 <name>_sample 开关决定逐样本抽 or 用固定值
        fmode, sfluct = draw_speed_fluct(dcfg, rng)    # speed_fluct_mode: rw 累加 / iid 不累加
        sramp = draw_speed_ramp(dcfg, f_rot, rng)    # speed_ramp_rpm 给定则按 rpm/rev 换算
        sample = generate_one_sample(
            signal_dim=_signal_dim(self.cfg),
            num_components=scfg["num_components"],
            min_sep_hz=scfg["min_sep_hz"],
            f_lo=scfg["f_lo"],
            f_hi=scfg["f_hi"],
            distance=scfg["distance"],
            amplitude=scfg["amplitude"],
            floor_amplitude=scfg["floor_amplitude"],
            k_range=scfg.get("k_range"),        # 可变 K 混训 (None -> 固定 K)
            regime=dcfg["regime"],
            probe_angles_deg=self.probe_angles,
            f_rot_hz=f_rot,
            speed_fluct=sfluct,
            fluct_mode=fmode,
            rng=rng,
            snr_db=snr,
            speed_ramp=sramp,
            freq_drift=dcfg.get("freq_drift", False),
            drift_coupling=coupling,
        )
        sample["f_rot"] = f_rot          # 该样本基础转速 (供 rate_summary 归一化)
        return sample

    def _encode(self, sample):
        y = sample["y"]
        rev_speeds = sample["rev_speeds"]
        # 时序视图 [2,N] (FTF 的 NUDFT 前端吃它 + t_n)
        signal = np.stack([y.real, y.imag], axis=0).astype(np.float32)
        # ⚠ 用该样本自身的基础转速归一化 (序列间 f_rot 不同时必须用样本级, 不能用 config
        #   的固定值 —— 那样 rel_mean 就不是"相对自己"的了)。
        f_rot = sample.get("f_rot", self.cfg["data"]["f_rot_hz"])
        # FiLM 的两种输入 (encoder.film_input 选; 两个都出, 让 encoder_forward 挑):
        #   summary [4]      窗内统计摘要 [rel_mean, slope, var, abs_mean]
        #   per_rev [n_rev]  逐圈归一化转速 (不做统计压缩, 保留逐圈细节)
        summary = rate_summary(rev_speeds, f_rot, self.ref_hz)
        rev_norm = rev_speeds.astype(np.float32) / self.ref_hz - 1.0
        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "t_n": torch.as_tensor(sample["t_n"], dtype=torch.float32),
            "rate_summary": torch.as_tensor(summary, dtype=torch.float32),        # [4]
            "rev_speeds_norm": torch.as_tensor(rev_norm, dtype=torch.float32),    # [n_rev]
            "freqs_hz": torch.as_tensor(sample["freqs_hz"], dtype=torch.float32),
            "amps": torch.as_tensor(sample["amps"], dtype=torch.float32),         # [K] 幅值反比加权用
            "n_freq": torch.as_tensor(sample["n_freq"], dtype=torch.long),
        }

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        return self._encode(self._generate(rng))
