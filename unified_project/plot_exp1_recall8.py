"""实验一 (5探头/100Hz/K=8): 四编码器 recall@8 随 SNR 变化。
横轴 SNR(dB), 纵轴 recall@8。画风对齐 btt_amortized_vi_project/plot_conclusion_*.py。
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
JSON = ROOT / "unified_project" / "artifacts_5pro_100hzFr_8fre" / "recall_8_10_12.json"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BLUE = "#2f6fbb"
ORANGE = "#d8842a"
GREEN = "#2e8b57"
RED = "#c9504b"
GRAY = "#6b7280"

# 频率网格 (50-450Hz, 2Hz) 与命中带宽 (basin_hz=2)
GRID_G = 201
BASIN_HZ = 2.0
GRID_STEP = 2.0


def random_recall(L):
    """网格均匀随机撒 L 点时, 单个真频被 ±basin 覆盖的期望概率 (解析下界)。"""
    cells = 2 * BASIN_HZ / GRID_STEP + 1     # basin 覆盖的网格格数
    p_hit = cells / GRID_G                    # 一个随机点命中的概率
    return 1.0 - (1.0 - p_hit) ** L

# 编码器 -> (显示名, 颜色, 标记)
ENCODERS = {
    "regression_cnn": ("regression_cnn", BLUE, "o"),
    "classification_cnn": ("classification_cnn", ORANGE, "s"),
    "rope": ("rope", GREEN, "^"),
    "resfreq": ("resfreq", RED, "D"),
}


def load_recall8():
    data = json.loads(JSON.read_text(encoding="utf-8"))
    curves = {}
    for enc in ENCODERS:
        per = data[enc]
        snrs = sorted(float(s) for s in per)
        vals = [per[str(s) if str(s) in per else f"{s:.1f}"]["recall@8"] for s in snrs]
        curves[enc] = (np.array(snrs), np.array(vals))
    return curves


def main():
    curves = load_recall8()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for enc, (name, color, marker) in ENCODERS.items():
        snr, rec = curves[enc]
        ax.plot(snr, rec, marker=marker, linewidth=2.1, color=color, label=name)

    rand = random_recall(8)
    ax.axhline(rand, color=GRAY, linestyle="--", linewidth=1.5,
               label=f"random top-8 = {rand:.3f}")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("recall@8")
    ax.set_title("Experiment 1 (5 probes / 100Hz / K=8): recall@8 vs SNR")
    any_snr = next(iter(curves.values()))[0]
    ax.set_xticks(any_snr)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "exp1_recall8_vs_snr.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
