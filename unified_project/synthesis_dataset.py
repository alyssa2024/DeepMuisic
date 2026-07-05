"""共享底层物理仿真: 非均匀 BTT 采样 + 多分量复谐波。

移植自 comparison/deepfreq_official/btt_sampling.py (物理 Hz + 非均匀 t_n),
内联 amplitude_generation, 并把频率分布做成可选:
    normal   参考 DeepFreq: 相邻间距 ~|N(0,scale*band)|+min_sep (簇状)
    uniform  参考 ResFreq : 区间内独立均匀 + 拒绝采样保 min_sep

单条样本产物 = 复信号 y[N] + 采样时刻 t_n[N] + 真频 f[K]。四种编码器的
时序视图 / 协方差视图都从这一份 y,t_n 派生 (见 dataset.py)。
"""
import numpy as np


# ------------------------------------------------------------------ #
# 幅值 (锁定为 |N(0,1)|+floor)                                        #
# ------------------------------------------------------------------ #
def amplitude_generation(shape, amplitude="normal_floor", floor_amplitude=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if amplitude == "normal_floor":
        return np.abs(rng.standard_normal(shape)) + floor_amplitude
    if amplitude == "normal":
        return np.abs(rng.standard_normal(shape))
    if amplitude == "uniform":
        return rng.random(shape) * (1 - floor_amplitude) + floor_amplitude
    raise ValueError(f"unknown amplitude={amplitude!r}")


# ------------------------------------------------------------------ #
# 频率采样 (物理 Hz, 无混叠区间不折叠)                                 #
# ------------------------------------------------------------------ #
def sample_frequencies_hz(nf, min_sep_hz, f_lo, f_hi, distance, rng, scale_frac=0.05):
    """在 [f_lo,f_hi) 内采 nf 个真频 (Hz, 已排序), 两两间隔 >= min_sep_hz。"""
    band = f_hi - f_lo

    if distance == "uniform":
        # ResFreq 式: 区间内独立均匀 + 拒绝采样保 min_sep
        for _ in range(1000):
            cand = np.sort(f_lo + rng.random(nf) * band)
            if nf < 2 or np.min(np.diff(cand)) >= min_sep_hz:
                return cand
        raise RuntimeError(
            "could not sample separated frequencies; relax min_sep_hz / widen band"
        )

    if distance == "normal":
        # DeepFreq 式: 从随机起点按 min_sep + 正态抖动逐个铺开
        for _ in range(500):
            first = f_lo + rng.random() * band
            vals = [first]
            ok = True
            for _i in range(1, nf):
                d = abs(rng.normal(scale=scale_frac * band)) + min_sep_hz
                nxt = vals[-1] + d
                if nxt >= f_hi:
                    ok = False
                    break
                vals.append(nxt)
            if ok and len(vals) == nf:
                return np.asarray(vals, dtype=np.float64)
        # 兜底: 等间距铺满
        return np.linspace(
            f_lo, min(f_hi, f_lo + (nf - 1) * min_sep_hz + 1e-6), nf
        )

    raise ValueError(f"unknown distance={distance!r} (use 'normal' or 'uniform')")


# ------------------------------------------------------------------ #
# 采样时刻 t_n (探头角 + 转速廓线)                                     #
# ------------------------------------------------------------------ #
def make_sampling_times(signal_dim, regime, probe_angles_deg, f_rot_hz, speed_fluct, rng):
    """返回采样时刻 t_n[signal_dim] (秒, 起点平移到 0)。

    uniform : 整数网格 0..N-1 (无物理几何, 对照锚点)
    probe   : 恒速标称转速, 探头角决定转内相位 (序列间几何相同)
    offset  : 整条序列一个固定偏移转速 f_rot*U[1±δ] (纯转速 OOD)
    fluct   : 逐转 i.i.d. 抖动, 每转周期 ~U[1±δ]*T_rev (转内+转间都抖)
    """
    if regime == "uniform":
        return np.arange(signal_dim, dtype=np.float64)

    angles = np.asarray(probe_angles_deg, dtype=np.float64)
    n_probe = len(angles)
    probe_phase = np.deg2rad(angles) / (2.0 * np.pi)      # in [0,1) 转数分数
    event = np.arange(signal_dim)
    probe_idx = event % n_probe
    rev_idx = event // n_probe

    if regime == "probe":
        t_rev = 1.0 / f_rot_hz
        t_n = (rev_idx.astype(np.float64) + probe_phase[probe_idx]) * t_rev
    elif regime == "offset":
        f_seq = f_rot_hz * (1.0 + speed_fluct * (2.0 * rng.random() - 1.0))
        t_rev = 1.0 / f_seq
        t_n = (rev_idx.astype(np.float64) + probe_phase[probe_idx]) * t_rev
    elif regime == "fluct":
        t_rev = 1.0 / f_rot_hz
        n_rev = int(rev_idx.max()) + 1
        periods = (1.0 + speed_fluct * (2.0 * rng.random(n_rev) - 1.0)) * t_rev
        rev_start = np.concatenate([[0.0], np.cumsum(periods)[:-1]])
        t_n = rev_start[rev_idx] + probe_phase[probe_idx] * periods[rev_idx]
    else:
        raise ValueError(f"unknown regime={regime!r}")

    return t_n - t_n[0]


# ------------------------------------------------------------------ #
# 单样本信号合成                                                       #
# ------------------------------------------------------------------ #
def generate_one_sample(
    signal_dim,
    num_components,
    variable_num_freq,
    min_sep_hz,
    f_lo,
    f_hi,
    distance,
    amplitude,
    floor_amplitude,
    regime,
    probe_angles_deg,
    f_rot_hz,
    speed_fluct,
    rng,
    snr_db=None,
):
    """生成一条 BTT 样本。

    Returns dict:
        y        : complex[signal_dim]  含噪复信号 (能量归一化到 RMS=1)
        t_n      : float[signal_dim]    采样时刻 (秒; uniform 档为整数网格)
        freqs_hz : float[K]             真频 (排序, 填充位 = -1e4)
        n_freq   : int                  实际分量数
    """
    n_freq = int(rng.integers(1, num_components + 1)) if variable_num_freq else num_components

    freqs_active = sample_frequencies_hz(
        n_freq, min_sep_hz, f_lo, f_hi, distance, rng
    )
    amps = amplitude_generation(n_freq, amplitude, floor_amplitude, rng)
    theta = rng.random(n_freq) * 2.0 * np.pi

    t_n = make_sampling_times(
        signal_dim, regime, probe_angles_deg, f_rot_hz, speed_fluct, rng
    )

    # y_n = Σ_k a_k exp(jθ_k + j2π f_k t_n)
    phase = 2j * np.pi * freqs_active[:, None] * t_n[None, :]      # [K,N]
    complex_amp = (amps * np.exp(1j * theta))[:, None]            # [K,1]
    y_clean = (complex_amp * np.exp(phase)).sum(axis=0)          # [N] complex

    noise_power = 0.0
    if snr_db is not None:
        sig_power = np.mean(np.abs(y_clean) ** 2)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(signal_dim) + 1j * rng.standard_normal(signal_dim)
        )
        y = y_clean + noise
    else:
        y = y_clean

    # 能量归一化 (与 DeepFreq 一致: 除以 RMS over 实/虚)
    rms = np.sqrt(np.mean(np.abs(y) ** 2))
    y = y / (rms + 1e-12)

    # 真频填充到固定长度 num_components (填充位 -1e4)
    freqs_padded = np.full(num_components, -1e4, dtype=np.float64)
    freqs_padded[:n_freq] = freqs_active

    return {
        "y": y.astype(np.complex64),
        "t_n": t_n.astype(np.float32),
        "freqs_hz": freqs_padded.astype(np.float32),
        "n_freq": n_freq,
    }
