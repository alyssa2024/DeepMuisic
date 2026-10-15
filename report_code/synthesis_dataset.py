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
    if amplitude == "equal":
        # ⭐等幅音: 所有分量幅值恒为 1, 不消耗 rng。
        #   用途是**隔离噪声效应** —— 其余三档幅值随机, 而 SNR 按最弱分量定义
        #   (σ_n² = min_k α_k² · 10^(-SNR/10)), 于是同一条序列里强分量的实际 SNR
        #   可比标称高 20-30dB, 降 SNR 打不动它们, 失败模式被"弱分量被挤出 top-K"
        #   主导, 看不出噪声本身的影响。等幅时 min_k α_k = α, 标称 SNR 对每个
        #   分量都成立 ⇒ SNR 才是唯一自变量。
        #   ⚠ 仅用于诊断/对照, **不是训练档** (训练一直是 normal_floor)。
        return np.ones(shape, dtype=float)
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

    if distance == "exact_sep":
        # ⭐相邻间隔**恰好** min_sep_hz 的等间距簇 (随机整体平移)。
        #   用途: 分辨率对照。`uniform`/`normal` 的 min_sep_hz 只是**下界** ——
        #   K=2 在 490Hz 带内均匀采样, 两频平均相距 ~163Hz, min_sep 几乎从不生效,
        #   测不到"贴着分辨极限"的情形。本模式把间隔钉死, 使间隔成为唯一自变量。
        #   ⚠ 仅用于诊断/对照, **不是训练档**。
        span = (nf - 1) * min_sep_hz
        if span >= band:
            raise ValueError(
                f"exact_sep: (nf-1)*min_sep_hz={span} 超出频带宽度 {band}")
        first = f_lo + rng.random() * (band - span)
        return first + np.arange(nf, dtype=np.float64) * min_sep_hz

    raise ValueError(
        f"unknown distance={distance!r} (use 'normal' / 'uniform' / 'exact_sep')")


# ------------------------------------------------------------------ #
# 转速廓线旋钮的逐序列抽样 (单窗 / 跨窗共用, 别分叉)                     #
# ------------------------------------------------------------------ #
def draw_speed_knob(dcfg, name, rng):
    """按 config 显式开关取 speed_fluct / speed_ramp 的本序列取值。

    <name>_sample=True  -> U[<name>_range[0], <name>_range[1]] 抽一次 (整条序列共用)
    <name>_sample=False -> 固定值 dcfg[<name>] (旧行为)
    抽样档缺 <name>_range 时报错, 不静默退回固定值 (静默退回会让整批数据悄悄跑错分布)。
    """
    if not dcfg.get(f"{name}_sample", False):
        return float(dcfg.get(name, 0.0))
    rg = dcfg.get(f"{name}_range")
    if rg is None:
        raise ValueError(f"{name}_sample=True 但缺 {name}_range")
    return float(rng.uniform(float(rg[0]), float(rg[1])))


def draw_speed_ramp(dcfg, f_rot_hz, rng):
    """本序列的 speed_ramp (返回相对量, 1/rev), 由 speed_ramp_mode 选口径:

    "rel" (缺省, 旧行为): speed_ramp[_sample/_range], 相对量 1/rev, 直接用。
    "rpm"             : speed_ramp_rpm[_sample/_range], 绝对斜率 rpm/rev, 换算成相对量
                        speed_ramp = rpm_per_rev / (f_rot_hz·60)。make_sampling_times 里
                        rev_speeds = f_rot·(1+ramp·r), 故逐圈转速的绝对斜率恰为 rpm_per_rev,
                        与本序列 f_rot 无关 (实测 coast-down 主段 ≈ -0.03~-0.04 rpm/rev)。
    各分支内再按 *_sample 决定: 逐序列 U[range] 抽一次 or 用固定值。
    缺 speed_ramp_mode 的旧 config 走 "rel", rng 消耗不变, 旧数据可逐条复现。
    """
    mode = dcfg.get("speed_ramp_mode", "rel")
    if mode == "rel":
        if dcfg.get("speed_ramp_sample", False):
            lo, hi = dcfg["speed_ramp_range"]
            return float(rng.uniform(float(lo), float(hi)))
        return float(dcfg.get("speed_ramp", 0.0))
    elif mode == "rpm":
        if dcfg.get("speed_ramp_rpm_sample", False):
            lo, hi = dcfg["speed_ramp_rpm_range"]
            rpm_per_rev = float(rng.uniform(float(lo), float(hi)))
        else:
            rpm_per_rev = float(dcfg["speed_ramp_rpm"])
        return rpm_per_rev / (float(f_rot_hz) * 60.0)
    raise ValueError(f"unknown speed_ramp_mode={mode!r} (应为 'rel' / 'rpm')")


def draw_speed_fluct(dcfg, rng):
    """本序列的逐圈转速波动 (返回 (fluct_mode, σ), σ 为相对量), 由 speed_fluct_mode 选:

    "rw"  (缺省, 旧行为): speed_fluct[_sample/_range]。随机游走, 逐圈偏离**累加**:
                          rw[r] = Σ_{i≤r} σ·ε_i, 偏离 std = σ·√(r+1), 相邻圈连续。
    "iid"               : speed_fluct_iid[_sample/_range]。独立抖动, **不累加**:
                          rw[r] = σ·ε_r, 每圈围绕 f_rot·(1+ramp·r) 独立上下跳, std 恒为 σ。
    各分支内再按 *_sample 决定: 逐序列 U[range] 抽一次 or 用固定值。
    缺 speed_fluct_mode 的旧 config 走 "rw", rng 消耗不变, 旧数据可逐条复现。
    """
    mode = dcfg.get("speed_fluct_mode", "rw")
    if mode == "rw":
        if dcfg.get("speed_fluct_sample", False):
            lo, hi = dcfg["speed_fluct_range"]
            return mode, float(rng.uniform(float(lo), float(hi)))
        return mode, float(dcfg.get("speed_fluct", 0.0))
    elif mode == "iid":
        if dcfg.get("speed_fluct_iid_sample", False):
            lo, hi = dcfg["speed_fluct_iid_range"]
            return mode, float(rng.uniform(float(lo), float(hi)))
        return mode, float(dcfg["speed_fluct_iid"])
    raise ValueError(f"unknown speed_fluct_mode={mode!r} (应为 'rw' / 'iid')")


# ------------------------------------------------------------------ #
# 转速档工具 (eval / plot 脚本共用, 别在脚本里各写一套 mode 分支)          #
#   规则: 脚本里钉的 / 扫的 fluct、ramp 数值一律按**当前生效 mode** 解释,   #
#   写进该 mode 自己的键。mode 来源: 环境变量 FLUCT_MODE / RAMP_MODE >     #
#   cfg["data"] 里的 speed_fluct_mode / speed_ramp_mode > 缺省 rw / rel。  #
#   缺省 rw/rel 下所有函数与旧脚本的直接写键逐位等价。                      #
# ------------------------------------------------------------------ #
FLUCT_KEYS = {"rw": "speed_fluct", "iid": "speed_fluct_iid"}
RAMP_KEYS = {"rel": "speed_ramp", "rpm": "speed_ramp_rpm"}


def fluct_mode_of(dcfg):
    mode = dcfg.get("speed_fluct_mode", "rw")
    if mode not in FLUCT_KEYS:
        raise ValueError(f"unknown speed_fluct_mode={mode!r} (应为 'rw' / 'iid')")
    return mode


def ramp_mode_of(dcfg):
    mode = dcfg.get("speed_ramp_mode", "rel")
    if mode not in RAMP_KEYS:
        raise ValueError(f"unknown speed_ramp_mode={mode!r} (应为 'rel' / 'rpm')")
    return mode


def apply_speed_mode_env(dcfg):
    """环境变量 FLUCT_MODE / RAMP_MODE 非空则覆盖 dcfg 的两个 mode 键 (原地改, 并返回 dcfg)。"""
    import os
    fm = os.environ.get("FLUCT_MODE", "")
    rm = os.environ.get("RAMP_MODE", "")
    if fm:
        dcfg["speed_fluct_mode"] = fm
        fluct_mode_of(dcfg)
    if rm:
        dcfg["speed_ramp_mode"] = rm
        ramp_mode_of(dcfg)
    return dcfg


def active_speed_fluct(dcfg, default=0.0):
    """当前 mode 的固定 σ (忽略 *_sample)。"""
    return float(dcfg.get(FLUCT_KEYS[fluct_mode_of(dcfg)], default))


def active_speed_ramp(dcfg, default=0.0):
    """当前 mode 的固定 ramp 数值 (rel: 1/rev; rpm: rpm/rev; 忽略 *_sample)。"""
    return float(dcfg.get(RAMP_KEYS[ramp_mode_of(dcfg)], default))


def default_speed_fluct(dcfg, rw_default):
    """脚本的 fluct 缺省值: rw 档沿用脚本原来的缺省 (rw_default), iid 档取 config 的 speed_fluct_iid。
    (脚本里写死的训练档常量 0.003 是随机游走口径, 不能原样搬去当 iid 的 σ。)"""
    return float(rw_default) if fluct_mode_of(dcfg) == "rw" else active_speed_fluct(dcfg)


def default_speed_ramp(dcfg, rel_default):
    """脚本的 ramp 缺省值: rel 档沿用脚本原来的缺省, rpm 档取 config 的 speed_ramp_rpm。"""
    return float(rel_default) if ramp_mode_of(dcfg) == "rel" else active_speed_ramp(dcfg)


def pin_speed_fluct(dcfg, sigma):
    """把 σ 钉进当前 mode 的固定键, 并关该 mode 的抽样。"""
    key = FLUCT_KEYS[fluct_mode_of(dcfg)]
    dcfg[key] = float(sigma)
    dcfg[f"{key}_sample"] = False
    return dcfg


def pin_speed_ramp(dcfg, value):
    """把 ramp 钉进当前 mode 的固定键 (rel: 1/rev; rpm: rpm/rev), 并关该 mode 的抽样。"""
    key = RAMP_KEYS[ramp_mode_of(dcfg)]
    dcfg[key] = float(value)
    dcfg[f"{key}_sample"] = False
    return dcfg


def fixed_speed_knobs(dcfg, f_rot_hz):
    """直接调 make_sampling_times 的脚本用: 返回 (fluct_mode, σ, ramp 相对量 1/rev)。
    取当前 mode 的固定值 (不抽样, 不耗 rng); rpm 档按 f_rot 换算成相对量。"""
    ramp = active_speed_ramp(dcfg)
    if ramp_mode_of(dcfg) == "rpm":
        ramp = ramp / (float(f_rot_hz) * 60.0)
    return fluct_mode_of(dcfg), active_speed_fluct(dcfg), ramp


def speed_profile_tag(dcfg, fluct=None, ramp=None):
    """打印 / 记录用的转速档描述, 如 'fluct=0.003 ramp=0.0015' (缺省 mode) 或
    'fluct[iid]=0.001 ramp[rpm]=-0.035rpm/rev'。fluct/ramp 不给则取当前 mode 的固定值。"""
    fm, rm = fluct_mode_of(dcfg), ramp_mode_of(dcfg)
    fl = active_speed_fluct(dcfg) if fluct is None else float(fluct)
    rp = active_speed_ramp(dcfg) if ramp is None else float(ramp)
    fs = f"fluct={fl:g}" if fm == "rw" else f"fluct[{fm}]={fl:g}"
    rs = f"ramp={rp:g}" if rm == "rel" else f"ramp[{rm}]={rp:g}rpm/rev"
    return f"{fs} {rs}"


def speed_mode_record(dcfg):
    """写进 JSON 记录的 mode 字段。缺省 rw/rel 返回空 dict (旧记录格式不变, 旧结果可继续比对/续跑)。"""
    rec = {}
    if fluct_mode_of(dcfg) != "rw":
        rec["speed_fluct_mode"] = fluct_mode_of(dcfg)
    if ramp_mode_of(dcfg) != "rel":
        rec["speed_ramp_mode"] = ramp_mode_of(dcfg)
    return rec


def speed_mode_ref(dcfg, fallback=None):
    """扫描脚本的"转速档参考": mode 取 dcfg 自己的 (缺省 rw/rel, **不**从 fallback 继承 ——
    dcfg 为训练档常量时它就是 rw/rel 口径), 再按环境变量覆盖; iid/rpm 的数值 dcfg 没有时
    从 fallback (通常是主线 CONFIG["data"]) 取。返回新 dict, 不改入参。"""
    ref = {**(fallback or {}), **dcfg}
    ref["speed_fluct_mode"] = dcfg.get("speed_fluct_mode", "rw")
    ref["speed_ramp_mode"] = dcfg.get("speed_ramp_mode", "rel")
    return apply_speed_mode_env(ref)


def set_speed_profile(dcfg, ref, fluct, ramp):
    """把 ref 的两个 mode 抄进 dcfg, 再把 fluct / ramp 钉进对应 mode 的键 (关抽样)。
    缺省 rw/rel 下等价于旧写法 speed_fluct=..; speed_ramp=..; *_sample=False。"""
    dcfg["speed_fluct_mode"] = fluct_mode_of(ref)
    dcfg["speed_ramp_mode"] = ramp_mode_of(ref)
    pin_speed_fluct(dcfg, fluct)
    pin_speed_ramp(dcfg, ramp)
    return dcfg


def speed_mode_note(dcfg):
    """打印用 mode 提示: 缺省 rw/rel 为 "" (旧输出不变), 否则如 "[fluct_mode=iid ramp_mode=rpm] "。"""
    rec = speed_mode_record(dcfg)
    if not rec:
        return ""
    return f"[fluct_mode={fluct_mode_of(dcfg)} ramp_mode={ramp_mode_of(dcfg)}] "


def speed_mode_suffix(dcfg):
    """输出文件名 / 缓存键后缀: 缺省 rw/rel 为 "" (旧文件名不变), 否则如 "_iid" / "_rpm" / "_iid_rpm"。"""
    return "".join(f"_{v}" for v in speed_mode_record(dcfg).values())


# ------------------------------------------------------------------ #
# 采样时刻 t_n (探头角 + 转速廓线)                                     #
# ------------------------------------------------------------------ #
def make_sampling_times(signal_dim, regime, probe_angles_deg, f_rot_hz, speed_fluct,
                        rng, speed_ramp=0.0, fluct_mode="rw"):
    """返回 (采样时刻 t_n[signal_dim] 秒, 逐圈转速 rev_speeds[n_rev] Hz)。

    uniform : 整数网格 0..N-1 (无物理几何, 对照锚点)
    probe   : 恒速标称转速, 探头角决定转内相位 (序列间几何相同)
    offset  : 整条序列一个固定偏移转速 f_rot*U[1±δ] (纯转速 OOD)
    fluct   : 线性漂移(speed_ramp/rev, run-up) + 随机游走(speed_rw 累积). 逐圈转速变。
              rev_speeds[r] = f_rot * (1 + ramp*r + cumsum(rw)[r]); t_n 由逐圈周期累积。
              fluct_mode="rw": 上式 (随机游走, 累加); "iid": cumsum 换成逐圈独立 σ·ε_r (不累加)。
    rev_speeds 供编码器算转速摘要/token; 恒速档为常数序列。
    """
    n_rev = signal_dim // len(probe_angles_deg)
    if regime == "uniform":
        return (np.arange(signal_dim, dtype=np.float64),
                np.full(n_rev, f_rot_hz, dtype=np.float64))

    angles = np.asarray(probe_angles_deg, dtype=np.float64)
    n_probe = len(angles)
    probe_phase = np.deg2rad(angles) / (2.0 * np.pi)      # in [0,1) 转数分数
    event = np.arange(signal_dim)
    probe_idx = event % n_probe
    rev_idx = event // n_probe

    if regime == "probe":
        rev_speeds = np.full(n_rev, f_rot_hz, dtype=np.float64)
        t_rev = 1.0 / f_rot_hz
        t_n = (rev_idx.astype(np.float64) + probe_phase[probe_idx]) * t_rev
    elif regime == "offset":
        f_seq = f_rot_hz * (1.0 + speed_fluct * (2.0 * rng.random() - 1.0))
        rev_speeds = np.full(n_rev, f_seq, dtype=np.float64)
        t_rev = 1.0 / f_seq
        t_n = (rev_idx.astype(np.float64) + probe_phase[probe_idx]) * t_rev
    elif regime == "fluct":
        # 逐圈转速 = 标称 × (1 + 线性漂移 + 随机游走); ramp=run-up 斜率, rw=小幅抖动
        r = np.arange(n_rev, dtype=np.float64)
        if speed_fluct <= 0:
            rw = 0.0
        elif fluct_mode == "rw":
            rw = np.cumsum(speed_fluct * rng.standard_normal(n_rev))    # 随机游走 (累加)
        elif fluct_mode == "iid":
            rw = speed_fluct * rng.standard_normal(n_rev)               # 独立抖动 (不累加)
        else:
            raise ValueError(f"unknown fluct_mode={fluct_mode!r} (应为 'rw' / 'iid')")
        factor = 1.0 + speed_ramp * r + rw                        # [n_rev]
        rev_speeds = (f_rot_hz * factor).clip(min=1e-3)           # Hz, 逐圈
        periods = 1.0 / rev_speeds                                # [n_rev] 秒/圈
        rev_start = np.concatenate([[0.0], np.cumsum(periods)[:-1]])
        t_n = rev_start[rev_idx] + probe_phase[probe_idx] * periods[rev_idx]
    else:
        raise ValueError(f"unknown regime={regime!r}")

    return t_n - t_n[0], rev_speeds


# ------------------------------------------------------------------ #
# 窗内频漂相位 (真实相位积分 phase=∫f(t)dt)                             #
# ------------------------------------------------------------------ #
def drifting_phase(freqs_bar, dpr_hz_per_rev, t_n, rev_cont):
    """窗内频漂的真实相位积分。频率随累计转数线性: f_k(rev)=f̄_k+dpr_k·(rev-rev_mid)。

    相位对【时间】积分 φ_k(t)=2π∫f_k(τ)dτ。用逐采样点的累计转数 rev_cont[n] 和真实
    时间 t_n[n] 数值积分 (梯形), 不假设匀速 -> 转速漂+频漂都正确 (避开手写解析式 bug,
    见 memory btt-rotation-sidebands-fool-physics)。

    freqs_bar     [K]   窗内平均频率 (= 标签, 频漂绕 rev_mid 对称 => 时均≈f̄)
    dpr_hz_per_rev[K]   频漂率 Hz/rev
    t_n           [N]   采样时刻 (秒, 已排序升序)
    rev_cont      [N]   每采样点的连续累计转数
    返回 phase [K,N] (含 2π, 供 exp(1j·phase))。
    """
    rev_mid = 0.5 * (rev_cont[0] + rev_cont[-1])
    f_inst = freqs_bar[:, None] + dpr_hz_per_rev[:, None] * (rev_cont[None, :] - rev_mid)  # [K,N] 瞬时频率
    # φ(t_n) = 2π ∫_0^{t_n} f dτ ; 梯形数值积分 over 真实时间 t_n
    dt = np.diff(t_n)                                                # [N-1]
    favg = 0.5 * (f_inst[:, 1:] + f_inst[:, :-1])                   # [K,N-1] 相邻均值
    dphi = 2.0 * np.pi * favg * dt[None, :]                         # [K,N-1] 每段相位增量
    phase = np.concatenate([np.zeros((f_inst.shape[0], 1)), np.cumsum(dphi, axis=1)], axis=1)
    return phase                                                    # [K,N]


# ------------------------------------------------------------------ #
# 单样本信号合成                                                       #
# ------------------------------------------------------------------ #
def generate_one_sample(
    signal_dim,
    num_components,
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
    k_range=None,
    speed_ramp=0.0,
    freq_drift=False,
    drift_coupling=0.5,
    fluct_mode="rw",
):
    """生成一条 BTT 样本。

    fluct_mode: 逐圈转速波动形式, "rw"=随机游走(累加) / "iid"=独立抖动(不累加),
      透传 make_sampling_times (由 draw_speed_fluct 按 speed_fluct_mode 给出)。

    freq_drift: 窗内频漂 (真实相位积分 phase=∫f(t)dt)。频率随累计转数线性漂移,
      漂移率 dpr_k = speed_ramp·drift_coupling·(f̄_k/f_rot) (耦合转速, 异步高频漂更多)。
      标签 freqs_hz = 窗内平均频率 f̄ (整窗共享单频, 时均自洽见 verify_drift_phase)。
      频漂绕窗中心对称 -> 时均≈f̄。仅 fluct 档有意义 (需 rev 演进)。

    Returns dict:
        y          : complex[signal_dim]  含噪复信号 (能量归一化到 RMS=1)
        t_n        : float[signal_dim]    采样时刻 (秒; uniform 档为整数网格)
        rev_speeds : float[n_rev]         逐圈转速 (Hz; 供转速摘要/token)
        freqs_hz   : float[K]             真频 (=窗内平均, 排序, 填充位 = -1e4)
        n_freq     : int                  实际分量数
    """
    # ⭐ k_range 给定 -> 逐样本抽 K ~ U{lo..hi} (可变 K 混训); 否则固定 K。
    #   num_components 此时是**上界** = padding 长度, 有效长度由返回的 n_freq 标出。
    #   ⚠ 抽 K 必须用**同一个 rng**, 且在抽频率之前 —— 它消耗 rng 状态, 换 k_range
    #     会改变后续所有随机量, 故不同 k_range 的数据集不可逐条对齐 (整体分布仍可比)。
    if k_range is not None:
        lo, hi = int(k_range[0]), int(k_range[1])
        hi = min(hi, int(num_components))           # 不能超过 padding 长度
        n_freq = int(rng.integers(lo, hi + 1))      # U{lo..hi}, 含两端
    else:
        n_freq = num_components                   # 固定 K

    freqs_active = sample_frequencies_hz(
        n_freq, min_sep_hz, f_lo, f_hi, distance, rng
    )
    amps = amplitude_generation(n_freq, amplitude, floor_amplitude, rng)
    theta = rng.random(n_freq) * 2.0 * np.pi

    t_n, rev_speeds = make_sampling_times(
        signal_dim, regime, probe_angles_deg, f_rot_hz, speed_fluct, rng,
        speed_ramp=speed_ramp, fluct_mode=fluct_mode
    )

    if not freq_drift:
        # 恒频: y_n = Σ_k a_k exp(jθ_k + j2π f_k t_n)
        phase = 2.0 * np.pi * freqs_active[:, None] * t_n[None, :]     # [K,N]
    else:
        # 窗内频漂: 每采样点连续累计转数 rev_cont = rev_idx + 转内相位分数
        n_probe = len(probe_angles_deg)
        probe_phase = np.deg2rad(np.asarray(probe_angles_deg)) / (2.0 * np.pi)  # [n_probe]
        ev = np.arange(signal_dim)
        rev_cont = (ev // n_probe).astype(np.float64) + probe_phase[ev % n_probe]  # [N]
        # 频漂率 dpr_k = ramp·coupling·(f̄_k/f_rot) Hz/rev (耦合转速, 异步无EO)
        dpr = speed_ramp * drift_coupling * (freqs_active / f_rot_hz)  # [K]
        phase = drifting_phase(freqs_active, dpr, t_n, rev_cont)       # [K,N] 真实积分
    complex_amp = (amps * np.exp(1j * theta))[:, None]            # [K,1]
    y_clean = (complex_amp * np.exp(1j * phase)).sum(axis=0)     # [N] complex

    noise_power = 0.0
    if snr_db is not None:
        # ⭐ SNR 按**总功率**定义 (2026-08-30 起的默认口径):
        #     SNR = 10·log10(Σ_k α_k² / σ_n²)  ⇒  σ_n² = Σ_k α_k² · 10^(-SNR/10)
        #   分量正交 ⇒ Σ_k α_k² = mean(|y_clean|²) (整周期上交叉项抵消), 这里直接用
        #   幅值平方和, 与 crosswin_data._add_noise_norm 逐位同式。
        #   ⚠⚠ 口径已从**最弱分量** (旧默认, 论文 eq:snr) 整体切换, 且**数值区间不变**
        #     (train_snr_choices 仍 {0,50}) ⇒ 同一个 SNR 数字现在对应的噪声**强得多**:
        #     固定真值档 Σα²/minα² = 384x ⇒ 相差 25.8 dB, 抽样档逐序列不同。
        #     故 2026-08-30 之前训练的 ckpt (如 artifacts/axial_sigcalib) 与之前落盘的
        #     所有 sweeps/*.jsonl **不可与新数据同表比较**, 需按新口径重训/重扫。
        noise_power = float(np.sum(amps ** 2)) / (10.0 ** (snr_db / 10.0))
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
    # 幅值也填充 (填充位 0), 供诊断脚本分析漏检 vs 幅值/间距; 训练路径不读它
    amps_padded = np.zeros(num_components, dtype=np.float64)
    amps_padded[:n_freq] = amps

    return {
        "y": y.astype(np.complex64),
        "t_n": t_n.astype(np.float32),
        "rev_speeds": rev_speeds.astype(np.float32),
        "freqs_hz": freqs_padded.astype(np.float32),
        "amps": amps_padded.astype(np.float32),
        "n_freq": n_freq,
    }
