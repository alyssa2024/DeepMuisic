"""弱分量专项测试(对齐表 §3.3 推后项 + 用户 2026-06-27 提出的缺口)。

此前 B0/B1/§13.2 全用各向同性等幅复高斯(各分量同量级)。本脚本测【强+弱】:
1 强分量 + 1 弱分量,扫幅值比 [0, -10, -20, -30] dB,看连续精修 + 幅值 LS 在弱分量上:
  - 强/弱分量各自的 频率 MAE;
  - 幅值相对误差 |â-a|/|a|;
  - marginal vs profile_map 是否在弱分量拉开差距(marginal 的 logdet/幅值先验理论上更稳)。

oracle 初值(真频对齐网格),K=2 已知。直接合成信号(绕过数据集固定幅值),完全可控。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_BTT = _ROOT / "btt_amortized_vi_project"
sys.path.insert(0, str(_HERE))
for p in (str(_BTT), str(_BTT / "legacy_reuse" / "current_project")):
    if p not in sys.path:
        sys.path.insert(0, str(p))

from btt_oracle_continuous_refinement import _load_runmod, newton_refine  # noqa: E402

# 显式按路径加载,避免 runmod 注册的同名 synthesis_dataset(无 speed_profile)覆盖 sys.modules
import importlib.util as _ilu  # noqa: E402
_sds = _BTT / "legacy_reuse" / "current_project" / "synthesis_dataset.py"
_spec = _ilu.spec_from_file_location("synthesis_dataset_weak", _sds)
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
generate_one_btt_sequence = _mod.generate_one_btt_sequence


def amp_ls(y, t, freqs, mask, ridge=1e-3):
    """给定频率,闭式 LS 解复幅值。y:[N], freqs:[K] -> [K] complex。"""
    phase = 2.0 * torch.pi * t.unsqueeze(-1) * freqs.unsqueeze(0)
    Phi = torch.polar(torch.ones_like(phase), phase) * mask.unsqueeze(-1)
    yw = y * mask
    PhiH = Phi.conj().transpose(-2, -1)
    gram = PhiH @ Phi
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    a = torch.linalg.solve(gram + ridge * eye, PhiH @ yw.unsqueeze(-1)).squeeze(-1)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=60)
    ap.add_argument("--num_cycles", type=int, default=16)
    ap.add_argument("--snr_db", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--spacing_hz", type=float, default=40.0,
                    help="两分量间距(Hz),取大间距隔离弱分量效应,不混近邻问题")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    runmod = _load_runmod()
    dc = runmod.CONFIG["data"]
    base_freq = dc["base_freq"]; probes = dc["probes"]
    gs = (1000.0 - 1.0) / 512.0

    ratios_db = [0.0, -10.0, -20.0, -30.0]
    f1_center = 300.0  # 强分量
    print(f"=== 弱分量测试 N={cli.num_samples} c{cli.num_cycles} snr{cli.snr_db}dB spacing{cli.spacing_hz}Hz ===")
    print(f"{'amp_ratio_dB':>13}{'mode':>10}{'f_strong_MAE':>13}{'f_weak_MAE':>12}"
          f"{'a_strong_relerr':>16}{'a_weak_relerr':>15}")

    for mode in ("marginal", "profile_map"):
        recon = "point_ls" if mode == "profile_map" else "marginal"
        dec = runmod.JointConstantFrequencyDecoder(recon_mode=recon, ridge=1e-3)
        for rdb in ratios_db:
            scale = 10 ** (rdb / 20.0)  # 弱分量幅值缩放
            rng = np.random.default_rng(cli.seed)
            fs_e, fw_e, as_e, aw_e = [], [], [], []
            for _ in range(cli.num_samples):
                f1 = f1_center + rng.uniform(-5, 5)
                f2 = f1 + cli.spacing_hz
                # 强分量幅值 ~1e-3 量级(同数据集),弱分量按 ratio 缩放
                a1 = complex(6e-4, 3e-4)
                a2 = complex(6e-4 * scale, 3e-4 * scale)
                s = generate_one_btt_sequence(
                    num_cycles=cli.num_cycles, base_freq=base_freq, fluctuation_delta=dc.get("fluctuation_delta", 0.0),
                    probe_angles=probes, freq_hz=[f1, f2],
                    amp_real=[a1.real, a2.real], amp_imag=[a1.imag, a2.imag],
                    snr_db=cli.snr_db, rng=rng, speed_profile="constant", speed_change=0.0)
                y = s["x_observed"].astype(np.complex128)
                norm = np.std(y) + 1e-12
                y = y / norm
                t = s["t_samples"].astype(np.float64)
                yv = torch.tensor(y, dtype=torch.complex64)
                tv = torch.tensor(t, dtype=torch.float32)
                mask = torch.ones_like(tv)
                nv = torch.tensor([float(s["noise_power"]) / norm**2], dtype=torch.float32)
                tf = np.array([f1, f2])
                a_true = np.array([a1, a2]) / norm  # 归一化后真幅值
                f0 = torch.tensor(np.round((tf - 1.0) / gs) * gs + 1.0, dtype=torch.float32)
                f_ref = newton_refine(dec, yv.view(1, -1), tv.view(1, -1), f0,
                                      mask.view(1, -1), nv, mode, steps=cli.steps)
                a_hat = amp_ls(yv, tv, f_ref, mask).numpy()
                # 按频率升序对应(f1<f2 恒成立, spacing>0)
                order = np.argsort(f_ref.numpy())
                fr = f_ref.numpy()[order]; ah = a_hat[order]
                fs_e.append(abs(fr[0] - tf[0])); fw_e.append(abs(fr[1] - tf[1]))
                as_e.append(abs(ah[0] - a_true[0]) / (abs(a_true[0]) + 1e-12))
                aw_e.append(abs(ah[1] - a_true[1]) / (abs(a_true[1]) + 1e-12))
            print(f"{rdb:>13.0f}{mode:>10}{np.mean(fs_e):>13.4f}{np.mean(fw_e):>12.4f}"
                  f"{np.mean(as_e):>16.4f}{np.mean(aw_e):>15.4f}")


if __name__ == "__main__":
    main()
