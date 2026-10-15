"""跨窗序列数据: 一条 sweep 切成 M 个 (可重叠) 窗, 每窗一份 (signal, t_n)。

窗长 = data.n_rev (复用单窗编码器孔径, 避免 OOD)。序列/重叠由 config.crosswin:
    n_rev_seq : 序列总转数
    hop_rev   : 窗间步长 (转); = n_rev 非重叠, = n_rev//2 则50%重叠
    n_win = (n_rev_seq - n_rev) // hop_rev + 1
    窗 m 覆盖 rev [m*hop_rev, m*hop_rev + n_rev)

复用 synthesis_dataset 的分量/幅值采样。恒速 (regime=probe) 下:
  - 一组分量 (freqs/amps/theta) 贯穿整条序列 (相位连续, 非每窗重采);
  - 每窗真频恒等 -> per-window 标签相同 (§7 null 语境, sum 就够, 先跑通)。
  - 换 regime=fluct / 加频率漂移 -> 窗间真频/转速变, 才验时序聚合器增益。

产物 (per sample):
    signal [M,2,N]  每窗时序视图
    t_n    [M,N]    每窗采样时刻 (秒, 各窗窗内相对; 编码器只用窗内相位)
    tau    [M]      窗中心累计转数 (跨窗位置, 供聚合器 RoPE / RPM)
    f_win  [M,K]    每窗真频 (含 -1e4 填充; 恒速下 M 行相同)
    n_freq int
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from config import resolve_probe_angles, signal_dim as _signal_dim
from dataset import rate_ref_hz, rate_summary
from synthesis_dataset import (
    amplitude_generation,
    sample_frequencies_hz,
    make_sampling_times,
    drifting_phase,
    draw_speed_fluct,
    draw_speed_ramp,
)


def n_windows(cfg):
    """从 config 推导窗数。窗长=data.n_rev, 序列/步长来自 crosswin。"""
    n_rev = int(cfg["data"]["n_rev"])
    n_rev_seq = int(cfg["crosswin"]["n_rev_seq"])
    hop = int(cfg["crosswin"]["hop_rev"])
    if n_rev_seq < n_rev:
        raise ValueError(f"n_rev_seq={n_rev_seq} < window n_rev={n_rev}")
    return (n_rev_seq - n_rev) // hop + 1


def _add_noise_norm(y, snr_db, rng, amps=None):
    """干净复信号 y[N] -> 加噪 + RMS 归一化 [N] (频漂档: 信号已由全序列切段生成)。

    ⭐ SNR 按**总功率**定义 (2026-08-30 起的默认口径, 与 synthesis_dataset 同口径):
        SNR = 10·log10(Σ_k α_k² / σ_n²)  ⇒  σ_n² = Σ_k α_k² · 10^(-SNR/10)
      分量正交 ⇒ Σ_k α_k² = mean(|y|²) (整周期上交叉项抵消) ⇒ amps 给定与否结果一致:
      给定时用 Σα² (精确), 未给定时退回 mean(|y|²) (数值等价, 见下面的一致性说明)。
    ⚠⚠ 口径已从**最弱分量** (旧默认, 论文 eq:snr) 整体切换, 且**数值区间不变** ⇒
      同一个 SNR 数字现在对应的噪声**强得多** (固定真值档相差 25.8 dB, 抽样档逐序列
      不同)。2026-08-30 之前训练的 ckpt 与之前落盘的 sweeps/*.jsonl 均不可与新数据
      同表比较, 需按新口径重训/重扫。
    ⚠ y 这里是**干净**信号 (调用点都在加噪之前), 故 mean(|y|²) 就是信号功率;
      若将来有调用点传入已含噪的 y, 这个回退分支会低估 SNR —— 传 amps 可避免。
    """
    if snr_db is not None:
        if amps is not None:
            noise_p = float(np.sum(np.asarray(amps) ** 2)) / (10.0 ** (snr_db / 10.0))
        else:
            noise_p = np.mean(np.abs(y) ** 2) / (10.0 ** (snr_db / 10.0))
        y = y + np.sqrt(noise_p / 2.0) * (
            rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    rms = np.sqrt(np.mean(np.abs(y) ** 2))
    return (y / (rms + 1e-12)).astype(np.complex64)


def _one_window_signal(freqs, amps, theta, t_n, snr_db, rng):
    """给定分量与该窗采样时刻 t_n[N] -> 含噪 + RMS 归一化的复信号 [N] (恒频)。"""
    phase = 2j * np.pi * freqs[:, None] * t_n[None, :]      # [K,N]
    camp = (amps * np.exp(1j * theta))[:, None]            # [K,1]
    y = (camp * np.exp(phase)).sum(axis=0)                 # [N]
    return _add_noise_norm(y, snr_db, rng, amps=amps)


class CrossWinDataset(Dataset):
    """在线现采跨窗序列。整条序列一次生成 (逐圈转速连续演进), 逐窗切片:
      - probe 档: 逐圈恒速, 各窗几何相同 (原恒速语境, 向后兼容)。
      - fluct 档: 逐圈转速 ramp+游走, 跨窗转速自然漂移; 每窗 rev_speeds_norm 供
        编码器 FiLM per_rev 注入, 与单窗 fluct 训练同分布 (消除 OOD)。
    真频当前恒定 (不做跨窗频漂), 各窗 f_win 相同。"""

    def __init__(self, cfg, size, seed=0, snr_db=None):
        self.cfg = cfg
        self.size = int(size)
        self.seed = int(seed)
        self.snr_db = snr_db
        self.probe_angles = resolve_probe_angles(cfg)
        self.n_probe = cfg["data"]["n_probes"]
        self.n_rev = int(cfg["data"]["n_rev"])              # 窗长 (转)
        self.hop = int(cfg["crosswin"]["hop_rev"])          # 窗间步长 (转)
        self.n_win = n_windows(cfg)
        self.n_rev_seq = int(cfg["crosswin"]["n_rev_seq"])  # 序列总转数
        # ⚠⚠ 基础转速必须**逐序列抽** (与 UnifiedBTTDataset 一致), 不能固定!
        #   曾经的 bug: 这里写死 self.f_rot = cfg["data"]["f_rot_hz"] (=35), **完全忽略
        #   f_rot_range** ⇒ 跨窗数据里所有序列都是 35Hz ⇒ rate_summary 的第 4 维
        #   abs_mean = 35/35−1 ≡ 0 (恒定), 而单窗训练时它在 [−0.29,+0.29] 变化
        #   ⇒ **summary 编码器的 FiLM 输入分布不匹配**, 跨窗 recall 掉 0.035 (0.920→0.885)。
        #   (per_rev 不受影响 —— 它除以固定的 rate_ref_hz, 两个数据集算法一致 ⇒ 跨窗还涨了
        #    0.006。这就是"单窗 summary 赢 0.015, 跨窗 per_rev 反超 0.027"符号翻转的真因。)
        self.f_rot_hz = float(cfg["data"]["f_rot_hz"])          # fallback (f_rot_range=None 时)
        self.f_rot_range = cfg["data"].get("f_rot_range")       # 逐序列抽的区间, 如 [25,45]
        self.ref_hz = rate_ref_hz(cfg)     # 与单窗 dataset 共用同一个归一化参考 (别分叉)
        self.N = _signal_dim(cfg)                           # 单窗采样点 = n_probe*n_rev
        # fluct 档: 整条序列一次生成逐圈转速 (ramp+游走), 逐窗切片 -> 编码器同分布
        dcfg = cfg["data"]
        self.regime = dcfg["regime"]
        # ⚠ 两个转速旋钮**不再**在此固化成 self.* —— 它们按 <name>_sample 开关逐序列抽,
        #   必须在 _generate 里用该序列的 rng 取 (存成实例属性会让整个数据集共用一个值)。
        self.dcfg = dcfg
        # 跨窗频漂: 频率沿全序列累计转数线性漂移, 窗间中心频率不同 (聚合器缝合轨迹),
        # 窗内近恒定 (编码器合法); 耦合同编码器训练 dpr_k=ramp·coupling·(f̄_k/f_rot)。
        self.freq_drift = bool(dcfg.get("freq_drift", False))
        self.drift_coupling = float(dcfg.get("drift_coupling", 0.5))
        # 混训随机 coupling (对齐编码器混训); None 则用固定值。评测时外部覆盖固定档。
        self.drift_coupling_range = dcfg.get("drift_coupling_range")

    def __len__(self):
        return self.size

    def _generate(self, rng):
        scfg = self.cfg["signal"]
        dcfg = self.cfg["data"]
        # ⭐ k_range 给定 -> 逐样本抽 K ~ U{lo..hi} (可变 K 混训); num_components
        #   此时是**上界** = padding 长度。⚠ 抽 K 消耗 rng, 且在抽频率之前 ⇒
        #   换 k_range 后后续随机量全变, 不同 k_range 的数据集不可逐条对齐。
        K_MAX = int(scfg["num_components"])
        _kr = scfg.get("k_range")
        if _kr is not None:
            K = int(rng.integers(int(_kr[0]), min(int(_kr[1]), K_MAX) + 1))
        else:
            K = K_MAX
        freqs = sample_frequencies_hz(
            K, scfg["min_sep_hz"], scfg["f_lo"], scfg["f_hi"], scfg["distance"], rng)
        amps = amplitude_generation(K, scfg["amplitude"], scfg["floor_amplitude"], rng)
        # ⭐定性示意图用: 指定**具体**频率/幅值 (scfg["fixed_freqs"] / ["fixed_amps"])。
        #   仍照常先抽一遍再覆盖 —— 不改 rng 消耗, 采样时刻/相位/噪声与不指定时同流,
        #   这样"换一组频率"是唯一变量。长度必须与 num_components 一致。
        _ff, _fa = scfg.get("fixed_freqs"), scfg.get("fixed_amps")
        if _ff is not None:
            freqs = np.asarray(_ff, dtype=float)
            assert freqs.size == K, f"fixed_freqs 长度 {freqs.size} != num_components {K}"
        if _fa is not None:
            amps = np.asarray(_fa, dtype=float)
            assert amps.size == K, f"fixed_amps 长度 {amps.size} != num_components {K}"
        theta = rng.random(K) * 2.0 * np.pi

        snr = self.snr_db
        # ⭐ 连续区间 {"min":a,"max":b} → 逐实例 U[a,b] 均匀采样 (对齐单窗编码器的
        #    training.train_snr_choices = {"min":0,"max":50})。
        #    list/tuple → 逐实例从离散档里随机抽一个 (聚合器 phase1 的历史做法)。
        if isinstance(snr, dict):
            snr = float(rng.uniform(float(snr["min"]), float(snr["max"])))
        elif isinstance(snr, (list, tuple)):
            snr = float(snr[int(rng.integers(len(snr)))]) if len(snr) else None

        n_probe = self.n_probe
        # ⭐基础转速**逐序列抽** (与 UnifiedBTTDataset 一致; 写死会让 summary 编码器 OOD)
        frr = self.f_rot_range
        f_rot = float(rng.uniform(frr[0], frr[1])) if frr else self.f_rot_hz
        # 整条序列一次生成: 全局采样时刻 [n_probe*n_rev_seq] + 逐圈转速 [n_rev_seq]。
        # fluct 档 -> ramp+游走连续演进, 跨窗转速自然漂移; probe 档 -> 逐圈恒速。
        N_seq = n_probe * self.n_rev_seq
        # 转速廓线两旋钮逐序列抽 (同单窗 UnifiedBTTDataset, 共用 draw_speed_fluct/draw_speed_ramp 别分叉)
        fluct_mode, speed_fluct = draw_speed_fluct(dcfg, rng)   # rw 累加 / iid 不累加
        speed_ramp = draw_speed_ramp(dcfg, f_rot, rng)   # speed_ramp_rpm 给定则按 rpm/rev 换算
        t_seq, rev_speeds_seq = make_sampling_times(
            N_seq, self.regime, self.probe_angles, f_rot,
            speed_fluct, rng, speed_ramp=speed_ramp,
            fluct_mode=fluct_mode)                     # [N_seq], [n_rev_seq]

        # 全序列干净信号 y_seq (频漂: 相位积分; 恒频: f·t). 整序列一次加噪 -> 重叠区一致,
        # 且支持联合大孔径精修 (存 y_full/t_full)。
        camp = (amps * np.exp(1j * theta))[:, None]
        if self.freq_drift:
            probe_phase = np.deg2rad(np.asarray(self.probe_angles)) / (2.0 * np.pi)
            ev = np.arange(N_seq)
            rev_cont = (ev // n_probe).astype(np.float64) + probe_phase[ev % n_probe]  # [N_seq]
            rev_mid = 0.5 * (rev_cont[0] + rev_cont[-1])
            cpr = self.drift_coupling_range
            coupling = float(rng.uniform(cpr[0], cpr[1])) if cpr else self.drift_coupling
            # ⚠ 用本序列抽到的 speed_ramp: 频漂率随 ramp 一起变 (ramp 是漂移的尺度因子),
            #   即 ramp 抽样会与 coupling 抽样相乘, 叠成两层随机尺度 —— 这是刻意的。
            dpr = speed_ramp * coupling * (freqs / f_rot)                       # [K] Hz/rev
            phase_seq = drifting_phase(freqs, dpr, t_seq, rev_cont)             # [K,N_seq]
            y_seq = (camp * np.exp(1j * phase_seq)).sum(axis=0)                 # [N_seq] 干净
        else:
            phase_seq = 2.0 * np.pi * freqs[:, None] * t_seq[None, :]           # 恒频 f·t
            y_seq = (camp * np.exp(1j * phase_seq)).sum(axis=0)                 # [N_seq] 干净
        # 整序列一次加噪 (重叠区一致 + 联合精修用)
        y_seq_noisy = _add_noise_norm(y_seq, snr, rng, amps=amps)

        signals, t_ns, taus, rev_norms, f_wins, summaries = [], [], [], [], [], []
        for m in range(self.n_win):
            start_rev = m * self.hop                         # 窗 m 起始转 (支持重叠)
            r0, r1 = start_rev, start_rev + self.n_rev       # 圈区间
            p0, p1 = r0 * n_probe, r1 * n_probe              # 采样点区间 (每圈 n_probe 点)
            t_abs = t_seq[p0:p1]                             # 该窗绝对采样时刻 (相位连续)
            t_win = t_abs - t_abs[0]                         # 窗内相对时刻 (编码器只用相对相位)
            rev_win = rev_speeds_seq[r0:r1]                  # 该窗逐圈转速 [n_rev]
            if self.freq_drift:
                rev_c = 0.5 * (r0 + r1)
                f_win_m = freqs + dpr * (rev_c - rev_mid)    # [K] 该窗中心频率 (漂移轨迹)
            else:
                f_win_m = freqs                              # 恒频: 各窗同真频
            y = y_seq_noisy[p0:p1]                           # 从整序列切段 (重叠区一致)
            signals.append(np.stack([y.real, y.imag], axis=0).astype(np.float32))
            t_ns.append(t_win.astype(np.float32))
            rev_norms.append((rev_win / self.ref_hz - 1.0).astype(np.float32))   # FiLM per_rev
            # ⚠⚠ rate_summary 的前 3 维 (rel_mean/slope/var) 是**相对基准转速**的, 基准必须是
            #    **该窗自己的起始转速** rev_win[0], **不能**用序列起点的 f_rot!
            #    单窗数据集里 f_rot 就是那个窗第 0 转的转速 (实测 rev[0]/f_rot−1 = ±0.008),
            #    所以"相对本窗起点的漂移"才是编码器训练时见过的语义。
            #    曾经的 bug: 用序列起点的 f_rot ⇒ 窗 m 起始于第 m·hop 转, 那里已 ramp 了
            #    ramp·m·hop ⇒ rel_mean 随窗号线性累积。实测单窗 rel_mean 分布 0.011±0.008
            #    (p99=0.029), 而跨窗成了 0.078±0.046 (p99=**0.183**, 大 6.3 倍) ⇒ **完全跑出
            #    训练分布** ⇒ summary 编码器的 FiLM OOD, 跨窗 recall 掉 0.035。
            #    (per_rev 不受影响 —— 它除以固定的 rate_ref_hz, 无累积。这就是"单窗 summary
            #     赢, 跨窗 per_rev 反超"符号翻转的真因。)
            summaries.append(rate_summary(rev_win, float(rev_win[0]),
                                          self.ref_hz).astype(np.float32))       # FiLM summary
            taus.append(float(start_rev + self.n_rev / 2.0)) # 窗中心累计转数
            f_wins.append(np.sort(f_win_m).astype(np.float32))
        # 真值复幅 (按频率升序排列, 与 f_win/freqs 对齐), 供 LS 幅值评估
        order = np.argsort(freqs)
        camp_sorted = (amps * np.exp(1j * theta))[order]     # [K] 复幅
        return {
            "freqs": np.sort(freqs).astype(np.float32),      # 序列中心频率 (参考)
            "amp_true": np.abs(camp_sorted).astype(np.float32),   # [K] 真值幅值 |c|
            "camp_re": camp_sorted.real.astype(np.float32),  # [K] 真值复幅实部
            "camp_im": camp_sorted.imag.astype(np.float32),  # [K] 真值复幅虚部
            "f_win": np.stack(f_wins, axis=0),               # [M,K] 逐窗中心频率 (漂移轨迹)
            "signal": np.stack(signals, axis=0),             # [M,2,N]
            "t_n": np.stack(t_ns, axis=0),                   # [M,N]
            "rev_speeds_norm": np.stack(rev_norms, axis=0),  # [M,n_rev]
            "rate_summary": np.stack(summaries, axis=0),     # [M,4] FiLM 摘要 (FTF 用)
            "tau": np.asarray(taus, dtype=np.float32),       # [M]
            # 整序列 (联合大孔径精修用): 唯一采样点 + 绝对时刻
            "y_full": np.stack([y_seq_noisy.real, y_seq_noisy.imag], 0).astype(np.float32),  # [2,N_seq]
            "t_full": (t_seq - t_seq[0]).astype(np.float32), # [N_seq] 绝对时刻(相对起点)
            "snr_db": np.float32(snr if snr is not None else np.nan),  # 该样本 SNR (ELBO σ_n² 用)
            "f_rot": np.float32(f_rot),   # ⭐该序列的基础转速 (逐序列抽; 聚合器 RPM 特征用)
            "K": K,
        }

    def _encode(self, s):
        # ⭐ 一律 pad 到**上界** K_MAX = signal.num_components, 有效长度由 n_freq 标出。
        #   可变 K 时 batch 内各样本 K 不同, 不 pad 到定长无法 collate 成张量。
        #   ⚠ 下游必须按 n_freq 掩码: f_win 填充位 = -1e4, amp/camp 填充位 = 0。
        M, K = self.n_win, s["K"]
        K_MAX = int(self.cfg["signal"]["num_components"])
        f_pad = np.full((M, K_MAX), -1e4, dtype=np.float32)
        f_pad[:, :K] = s["f_win"]                            # [M,K] 逐窗中心频率 (恒频时各窗同)

        def _pad1(v):
            out = np.zeros(K_MAX, dtype=np.float32)
            out[:K] = np.asarray(v, dtype=np.float32)
            return out

        return {
            "amp_true": torch.as_tensor(_pad1(s["amp_true"])),   # [K_MAX] 真值幅值
            "camp_re": torch.as_tensor(_pad1(s["camp_re"])),     # [K_MAX]
            "camp_im": torch.as_tensor(_pad1(s["camp_im"])),     # [K_MAX]
            "signal": torch.as_tensor(s["signal"]),          # [M,2,N]
            "t_n": torch.as_tensor(s["t_n"]),                # [M,N]
            "rev_speeds_norm": torch.as_tensor(s["rev_speeds_norm"]),  # [M,n_rev]
            "rate_summary": torch.as_tensor(s["rate_summary"]),        # [M,4] FiLM 摘要
            "tau": torch.as_tensor(s["tau"]),                # [M]
            "f_win": torch.as_tensor(f_pad),                 # [M,K_MAX] (填充 -1e4)
            "y_full": torch.as_tensor(s["y_full"]),          # [2,N_seq] 整序列 (联合精修)
            "t_full": torch.as_tensor(s["t_full"]),          # [N_seq]
            "snr_db": torch.as_tensor(s["snr_db"]),          # [] 该样本 SNR (ELBO σ_n²)
            "f_rot": torch.as_tensor(s["f_rot"]),            # [] 该序列基础转速 (RPM 特征用)
            "n_freq": torch.as_tensor(K, dtype=torch.long),
        }

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        return self._encode(self._generate(rng))
