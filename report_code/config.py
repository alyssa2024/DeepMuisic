"""全部超参。六阶段训练与数据生成共用这一份。"""
import json

import numpy as np

# ⭐ ckpt 根目录: 新旧 SNR 口径分目录, 由 CKPT_DIR 环境变量切换 (见 ckpt_root.py)。
from ckpt_root import ARTIFACTS as _CKPT_ROOT, TB_ROOT as _TB_ROOT

CONFIG = {
    "seed": 0,

    # --- 物理数据几何 (非均匀 BTT 采样) ---
    "data": {
        "n_rev": 16,                   # 单窗转数
        "n_probes": 8,                 # 探头数 -> signal_dim = n_probes * n_rev = 128
        "probe_layout": "real8",
        "probe_angles_deg": None,
        "f_rot_hz": 35.0,              # 标称转速 (f_rot_range 给定时作 fallback)
        "f_rot_range": [25.0, 45.0],   # 每样本在此区间抽基础转速
        "rate_ref_hz": 35.0,           # 绝对转速归一化参考, 必须 ≈ f_rot_range 中值
        "regime": "fluct",             # fluct=变速 / probe=恒速
        # 转速廓线两个旋钮。各自独立控制 "是否逐序列抽样 + 抽样范围":
        #   *_sample=False -> 全体样本用固定值 speed_fluct / speed_ramp (旧行为)
        #   *_sample=True  -> 每条序列 U[range[0], range[1]] 抽一次, 整条序列内保持该值
        # ⚠ freq_drift=True 时 dpr_k = speed_ramp·coupling·(f̄_k/f_rot), 即 ramp 是频漂
        #   的尺度因子 -> ramp 抽样会与 drift_coupling_range 相乘, 叠成两层随机尺度。
        # 波动形式选择: "rw" -> 用 speed_fluct* (随机游走, 逐圈偏离累加, std=σ·√(r+1));
        #   "iid" -> 用 speed_fluct_iid* (独立抖动, 不累加, 每圈 std 恒为 σ)。
        #   σ 都是相对 f_rot 的标准差 (≥0), 每步正负由 N(0,1) 决定。*_sample 只在被选中时生效。
        "speed_fluct_mode": "rw",
        "speed_fluct": 0.003,          # 逐圈转速随机游走强度 (高斯步长 σ, 相对量)
        "speed_fluct_sample": False,    # 逐序列抽随机游走强度
        "speed_fluct_range": [0.0, 0.001],
        "speed_fluct_iid": 0.001,      # 逐圈独立抖动 σ (相对量), 仅 speed_fluct_mode="iid"
        "speed_fluct_iid_sample": False,
        "speed_fluct_iid_range": [0.0, 0.001],
        # ramp 口径选择: "rel" -> 用 speed_ramp* (相对量 1/rev); "rpm" -> 用 speed_ramp_rpm*
        #   (绝对斜率 rpm/rev, 按本序列 f_rot 换算)。两组各自的 *_sample 只在被选中时生效。
        "speed_ramp_mode": "rpm",#“"rpm"
        "speed_ramp": 0.0015,          # 逐圈转速线性 ramp / rev
        "speed_ramp_sample": False,     # 逐序列抽 ramp 斜率
        "speed_ramp_range": [0.0, 0.0015],
        # 绝对斜率版 ramp (rpm/rev, 负=减速), 仅 speed_ramp_mode="rpm" 时生效
        #   (见 synthesis_dataset.draw_speed_ramp)。
        #   实测 coast-down (REAL_GT) 主减速段 -0.031 ~ -0.039 rpm/rev, 全程平均 -0.030。
        "speed_ramp_rpm": -0.035,
        "speed_ramp_rpm_sample": True,
        "speed_ramp_rpm_range": [-0.04, 0.04],
        "freq_drift": False,           # 窗内频漂 (全局关闭: 六阶段与推理评测统一恒频)
        "drift_coupling": 0.5,         # 仅 freq_drift=True 时生效
        "drift_coupling_range": [0.0, 5.0],
    },

    # --- 信号 (多分量) ---
    "signal": {
        # ⭐ num_components = **K 的上界** (padding 长度)。k_range 给定时逐样本抽
        #   K ~ U{k_range[0]..k_range[1]}, 真频/真幅填到前 n_freq 位, 其余填充
        #   (freqs=-1e4, amps=0), 由 batch["n_freq"] 标出有效长度。
        #   ⚠ 下游必须按 n_freq 掩码, 不能假定整行都有效 —— loss.build_anchor_targets
        #     与 train_freq_axis 的 recall 都已按 n_freq 掩码, 故可变 K 是支持的。
        #   k_range=None -> 固定 K = num_components (旧行为)。
        "num_components": 15,          # 分量数 K (k_range=None 时即固定 K; 给 k_range 时 = 上界/padding 长度, 须 ≥ k_range[1])
        "k_range": [1, 15],            # 逐样本 K ~ U{lo..hi}; None 则固定 K
        "min_sep_hz": 4.0,             # 分量最小间隔
        "f_lo": 5.0,                   # 真频采样下界 (网格 0-500, 留 5Hz 边距)
        "f_hi": 495.0,                 # 真频采样上界
        "distance": "uniform",         # 区间内独立均匀 + 拒绝采样保 min_sep
        "amplitude": "normal_floor",   # |N(0,1)| + floor_amplitude
        "floor_amplitude": 0.1,
    },

    # --- 频率网格 (anchor 的 bin = token) ---
    "frequency": {
        "grid_step_hz": 2.0,
        "f_min": 0.0,                  # 网格下界 (0-500 -> G=251)
        "f_max": 500.0,
    },

    # --- 编码器: FTF (频率-token Transformer, anchor 三头) ---
    "encoder": {
        "type": "nudft_ftf_anchor",
        "rate_inject": True,           # FiLM 转速注入
        "film_input": "summary",       # summary(4维摘要) / per_rev(n_rev 维逐圈)
        "ftf": {
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "ffn_dim": 256,
            "dropout": 0.0,
        },
        "anchor": {
            # 正文 (eq:softlabel): σ_obj = 1 **band**, 高斯写在 band 索引域 (g-b_k)。
            # ⚠ 单位是 band 不是 Hz: 设 obj_sigma_hz=1.0 会被 /Δ=2 换成 0.5 band (窄一半)。
            "obj_sigma_hz": None,      # obj 软标签宽度 (物理 Hz; None 则用 obj_sigma_bins)
            "obj_sigma_bins": 1.0,     # = 正文 σ_obj = 1 band
        },
    },

    # --- 训练 (阶段一/二: 单窗频率轴) ---
    "training": {
        "batch_size": 256,
        "num_workers": 12,             # Windows/CPU 上须设 0
        # ⭐⭐ SNR 口径 = **总功率** (2026-08-30 起): σ_n² = Σ_k α_k² · 10^(-SNR/10)。
        #   旧口径为**最弱分量** (σ_n² = min_k α_k²·10^(-SNR/10), 论文 eq:snr)。
        #   ⚠ 切换时**数值区间保持不变** ⇒ 同一个数字对应的噪声比旧口径强得多:
        #     固定真值档差 25.8 dB; K=6 抽样档差 11.2..32.8 dB (中位 19.7, 逐序列不同)。
        #   ⚠⚠ 故 2026-08-30 之前训练的 ckpt (artifacts/axial_sigcalib 等) 与之前落盘的
        #     sweeps/*.jsonl 都是**旧口径**产物, 不能与新数据同表比较, 需重训 + 重扫。
        #   实现点 (改口径必须同步全部, 否则 CRB 与实际噪声对不上):
        #     synthesis_dataset.generate_one_sample / crosswin_data._add_noise_norm
        #     infer_fft_uniform / eval.recon_bound / eval.scan_*._seq_crb*
        "train_snr_choices": {"min": 0.0, "max": 50.0},
        "stage1": {
            "iters": 60000,
            "lr": 3e-4,
            "anneal_start": 40000,
            "anneal_min_factor": 0.01,
        },
        "stage2": {
            # 6000 步时 val_loss/σcorr/σ-err 三者都还在降 (被步数截断, best.pt 落在末步)。
            # 20000 步才真收敛: Δval → 2e-4, σcorr 锁 0.595, σ/err 1.81。
            "iters": 20000,
            "lr": 3e-4,
            "anneal_start": 6000,      # 前 6000 步恒定, 之后 cosine 退火
            "anneal_min_factor": 0.01,
        },
    },

    # --- 评测 (单窗) ---
    "eval": {
        "eval_every": 2000,
        "eval_batches": 8,             # 验证集 = batch_size × 该值 = 2048 单窗
        "monitor_snr_db": 20.0,        # 训练期监控档
        "eval_snr_db": [-5.0, 0.0, 5.0, 10.0, 20.0, 30.0],   # 分档评估档位 (六阶段共用)
        # 训练**结束**的分档评估 (metrics.json per_snr) 额外报 recall@(K+dL): 逐样本取
        #   前 n_freq+dL 个峰。0 = 原 recall/allk 键 (选模型用); 其余键名 recall@K+dL / allk@K+dL。
        #   训练过程中的 log 只算 dL=0, 不受此项影响。
        "final_recall_dL": [0, 2],
        "basin_hz": 2.0,               # 命中容差
        "nms_distance_hz": 2.7,        # 取峰 NMS 最小间距 (-> min_dist=1 band)。
                                       # 正文未规定取峰细节, 对齐 artifacts_noweight。
    },

    # --- 跨窗 (阶段三~六) ---
    "crosswin": {
        "n_rev_seq": 40,               # 序列总转数 (2026-09-26 由 104 改; 旧结果见 artifacts_total/)
        "hop_rev": 8,                  # 窗间步长 -> 50% 重叠, n_win=(40-16)//8+1=4, 窗 0/8/16/24 恰好铺满
        # 时间轴层结构 (逐 bin RoPE attention; 层数 = 频率轴层数, 交替一对一)
        "time_axis": {
            "n_heads": 4,
            "ffn_dim": 256,
        },
        "train": {
            "iters": 8000,
            "batch_size": 32,
            "lr": 1e-3,
            "anneal_start": 4800,
            "anneal_min_factor": 0.01,
            "beta_gnll": 0.0,          # GNLL 权重 (0 -> σ 头冻结)
        },
        "eval": {
            "eval_every": 500,
            "monitor_snr_db": 20.0,    # 训练期监控档 (与 eval.monitor_snr_db 同义)
            "eval_size": 256,          # 每次评测的序列数
        },
    },

    "checkpoint": {
        # ⭐ 新旧 SNR 口径的模型分目录 (见 ckpt_root.py)。默认 artifacts_total/ = 新
        #   口径 (总功率); 旧口径 (最弱分量) 模型原地留在 artifacts/, 不搬迁 ——
        #   它是 eval/sweeps_weakest/ 那批结果的来源, 搬走就无法复现。
        # ⚠ 两套 ckpt 目录结构与文件名完全相同 (axial_sigcalib/best.pt 等), 混放
        #   事后无从分辨; 且旧 ckpt 配新口径数据等于严重外推 ⇒ 必须配套使用。
        #   回用旧模型: CKPT_DIR=artifacts python <script>.py
        "dir": _CKPT_ROOT,
    },

    "tb_dir": _TB_ROOT,        # report_code/tf-logs (数据盘), 见 ckpt_root.py
}


PROBE_LAYOUTS = {
    "real8": [0.0, 9.47, 29.00, 43.12, 70.08, 80.66, 94.06, 112.54],
}


def resolve_probe_angles(cfg):
    """探头周向角 (度)。"""
    layout = cfg["data"]["probe_layout"]
    if layout == "custom":
        ang = cfg["data"]["probe_angles_deg"]
        if not ang:
            raise ValueError("probe_layout='custom' requires probe_angles_deg")
        return list(ang)
    if layout not in PROBE_LAYOUTS:
        raise ValueError(f"unknown probe_layout={layout!r}")
    return list(PROBE_LAYOUTS[layout])


def grid_size(cfg):
    """网格 bin 数 G = (f_max - f_min)/Δ + 1 (格心点数, 含两端)。"""
    fcfg = cfg["frequency"]
    return int(round((fcfg["f_max"] - fcfg["f_min"]) / fcfg["grid_step_hz"])) + 1


def signal_dim(cfg):
    """采样点数 N = n_probes * n_rev。"""
    d = cfg["data"]
    return int(d["n_probes"]) * int(d["n_rev"])


def load_ckpt_config(path):
    """读 ckpt 产物目录里的 config.json (训练时落盘的那份)。"""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
