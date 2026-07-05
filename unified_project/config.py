"""统一项目集中配置。

一份底层物理数据(非均匀 BTT 采样 + 多分量谐波), 四种局部编码器共用:
    regression_cnn  DeepFreq 原生伪谱回归 (2,N)->fr_size, 高斯核 MSE
    classification_cnn  DeepFreq 主干 + 逐位置多热分类头 (2,N)->grid_size, BCE
    resfreq  协方差图 [3,D,D]->grid_size 多热, BCE
    rope  RoPE-Transformer 伪谱回归 (2,N)+t_n->fr_size, 高斯核 MSE (big 档)

切换编码器只改 CONFIG["encoder"]["type"], 数据零改动。所有数据/信号/网格/
协方差/编码器参数都在这里, 不写死在别处。
"""

CONFIG = {
    "seed": 0,

    # --- 共享底层数据几何 (非均匀 BTT 采样) ---
    # 短窗对比 (四编码器共用一份数据): 时序单窗 >16 转会崩 (相位积分误差累积),
    # 故 n_rev=16 是所有编码器的公共上限。协方差在这 16 转内用小 block_revs 自洽
    # (见 covariance 段)。频段/幅度/K/转速仍对齐 normal_floor 实验。
    "data": {
        "n_rev": 16,                  # 转数 (signal_dim = n_probes * n_rev; 16=时序单窗上限)
        "n_probes": 5,                 # 探头数 (real8; unified 协方差 D=block_revs*n_probes)
        "probe_layout": "paper5",       # real8 / paper5 / custom(读 probe_angles_deg)
        "probe_angles_deg": None,      # probe_layout="custom" 时用这个显式角度列表
        "f_rot_hz": 100.0,             # 标称转速 (Hz) — 原文 100Hz 恒速
        "regime": "probe",             # probe=恒速无波动 (对齐原文, ResFreq 主场)
        "speed_fluct": 0.0,            # 恒速 -> 无波动
    },

    # --- 信号 (多分量谐波; 幅值统一 |N(0,1)|+floor) ---
    # 对齐原文: K=8 固定, 50-450Hz, min_sep6, uniform 分布 (ResFreq 式区间均匀+拒采)。
    "signal": {
        "num_components": 8,           # 最大分量数 K (原文 K=8)
        "variable_num_freq": False,    # 固定 8 分量 (原文非可变)
        "min_sep_hz": 6.0,             # 分量最小间隔 (Hz) — 原文 6Hz
        "f_lo": 50.0,                  # 真频区间下界 (Hz) — 原文 50Hz
        "f_hi": 450.0,                 # 真频区间上界 (Hz) — 原文 450Hz
        # 频率分布:
        #   normal  参考 DeepFreq: 相邻间距 ~ |N(0,scale*band)|+min_sep (簇状)
        #   uniform 参考 ResFreq: 区间内独立均匀 + 拒绝采样保 min_sep
        "distance": "uniform",         # 原文 ResFreq 式区间均匀 + 拒绝采样
        "amplitude": "normal_floor",   # |N(0,1)|+floor_amplitude (已锁定)
        "floor_amplitude": 0.1,
    },

    # --- resfreq 协方差视图: 快拍分块 (从共享 y 重排) ---
    # D = block_revs * n_probes (协方差图边长); Q = num_snapshots (快拍数)。
    # 短窗自洽 (n_rev=16): 时序单窗>16转会崩, 故协方差也必须在16转内成立。
    #   block_revs=2 -> D=2*n_probes (小), 让 Q=13 个快拍在 [0,14] 铺开不重叠;
    #   Q=13>=D 保协方差满秩 (Q<D 会天然低秩退化)。不再是原文 D=64 (那需104转,
    #   时序会崩), 而是四编码器共用同一份 16转短窗数据的自洽几何。
    "covariance": {
        "block_revs": 2,               # 每快拍覆盖的转数 -> D = 2 * n_probes
        "num_snapshots": 13,           # 快拍数 Q (Q>=D 保满秩; 13个快拍在16转内铺开)
        "channels": ["real", "imag", "angle"],
    },

    # --- 频率网格 (分类 / resfreq 多热 + eval basin) ---
    # 对齐原文: 50-450Hz, 2Hz 网格 -> 201 格。
    "frequency": {
        "grid_step_hz": 2.0,           # 网格间距 (Hz) — 原文 2Hz
        "f_min": 50.0,                 # 网格下界 (Hz)
        "f_max": 450.0,                # 网格上界 (Hz)
        "target_sigma_hz": 1.0,        # 回归/rope 高斯核目标宽度 (Hz)
    },

    # --- 编码器 ---
    "encoder": {
        # 选择训练哪些编码器 (一次跑逐个训练选中的; 顺序即训练顺序)。
        # 想只训某个就留一个; 想全训就四个都列上; 注释掉不训的。
        "train": [
            "regression_cnn",
            "classification_cnn",
            "resfreq",
            "rope",
        ],
        # type: 单选 fallback (--encoder 覆盖它; 仅当 train 为空时用作单跑目标)。
        "type": "regression_cnn",      # regression_cnn/classification_cnn/resfreq/rope

        # DeepFreq CNN 主干 (回归 & 分类共用)
        "cnn": {
            "n_filters": 64,
            "n_layers": 20,
            "inner_dim": 125,
            "kernel_size": 3,
            "upsampling": 8,           # fr_size = inner_dim * upsampling (回归)
            "kernel_out": 25,
        },

        # RoPE-Transformer (big 档: d256 / nl4 / ff512)
        "rope": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "ffn_dim": 512,
            "base": 10000.0,
            "t_scale": None,           # None -> 用 f_rot_hz (秒*f_rot=转数, 使相位 O(1))
            # 桥接/上采样头与 DeepFreq 同构
            "n_filters": 64,
            "inner_dim": 125,
            "upsampling": 8,
            "kernel_out": 25,
        },

        # ResFreq 协方差网络
        "resfreq": {
            "upsample_channels": 8,
            "stage_channels": [64, 128, 256, 512],
            "residual_blocks_per_stage": 2,
        },
    },

    # --- 损失 (按 encoder.type 分派) ---
    "loss": {
        # 回归/rope: 高斯核伪谱 MSE (target_sigma_hz 在 frequency 段)
        # 分类/resfreq: 多热 BCEWithLogits
        "bce_pos_weight": None,        # 分类/resfreq 正样本权重 (None=1)
    },

    # --- 训练 ---
    "training": {
        "iters": 30000,
        "batch_size": 256,
        "lr": 3e-4,
        "grad_clip": 0.0,
        "device": "cpu",              # "cuda" if available
        # DataLoader 并行 worker 数。在线现采下数据生成是瓶颈(比GPU慢~15×),
        # 多 worker 并行现采让 GPU 吃满 (0=主进程串行, 慢)。GPU 训练建议 8-16。
        "num_workers": 12,
        # 混训 SNR (对齐 resfreq_reproduction 原文): 每样本抽 U{choices} dB。
        # None 或 [] -> 无噪训练 (靠 floor 提供动态范围, unified 旧默认)。
        "train_snr_choices": [0.0, 2.5, 5.0, 7.5],
        # 在线现采 (每 step 新批, 无固定 train/val 泄漏); 可选固定缓存做可复现对照
        "cache": {
            "enabled": False,
            "cache_size": 50000,       # enabled=True 时预生成的固定样本数
        },
    },

    # --- 评测 (mixed-train / separated-eval 协议) ---
    "eval": {
        "eval_every": 2000,
        "eval_batches": 8,
        # 训练期监控只看单档 SNR (省时, 训练日志只打这档 recall + train/val loss);
        # 训练结束后再跑 eval_snr_db 全档评估存 metrics.json。
        "monitor_snr_db": 20.0,
        "monitor_recall_L": 5,         # 训练期打 recall@这个L
        "eval_snr_db": [-5.0, 0.0, 5.0, 10.0, 20.0, 30.0],
        "basin_hz": 2.0,               # recall@L 命中带宽 (±Hz)
        "recall_L": [1, 3, 5, 10],
        "nms_distance_hz": 2.7,        # 峰值 NMS 间距
        "success_tol_hz": 1.0,         # freq MAE 成功率阈值
    },

    "checkpoint": {
        "dir": "unified_project/artifacts",
        "save_every": 2000,
    },
}


def resolve_probe_angles(cfg):
    """返回探头角(度) numpy 数组。"""
    import numpy as np

    layouts = {
        "real8": [0.0, 9.47, 29.00, 43.12, 70.08, 80.66, 94.06, 112.54],
        "paper5": [0.0, 120.0, 150.0, 180.0, 240.0],
    }
    layout = cfg["data"]["probe_layout"]
    if layout == "custom":
        angles = cfg["data"]["probe_angles_deg"]
        if angles is None:
            raise ValueError("probe_layout='custom' requires probe_angles_deg")
    else:
        if layout not in layouts:
            raise ValueError(f"unknown probe_layout={layout!r}")
        angles = layouts[layout]
    return np.asarray(angles, dtype=np.float64)


def signal_dim(cfg):
    """采样点数 N = n_probes * n_rev (从转数派生, 与探头数解耦地表达序列长度)。"""
    d = cfg["data"]
    return int(d["n_probes"]) * int(d["n_rev"])


def grid_size(cfg):
    """频率网格点数 G。"""
    fcfg = cfg["frequency"]
    return int(round((fcfg["f_max"] - fcfg["f_min"]) / fcfg["grid_step_hz"])) + 1
