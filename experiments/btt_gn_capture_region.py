"""B1 (plan_final §13.3): GN 捕获域扫描。

给真频加【受控初值误差】,测阻尼牛顿/GN 的:收敛成功率、最终 MAE、发散率。
据此确定 FR 候选必须达到的最大初值误差(对照 plan.md §3 实测 FR 初值误差
mean 0.52 / p95 1.11 / max 1.84 Hz)。

与 B0 共用 newton_refine;唯一区别:初值 = 真频 + offset(单分量先做,隔离捕获域,
不混多分量耦合)。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from btt_oracle_continuous_refinement import _load_runmod, newton_refine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["marginal", "profile_map"], default="marginal")
    ap.add_argument("--num_samples", type=int, default=60)
    ap.add_argument("--num_cycles", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--profile_ridge", type=float, default=1e-3)
    ap.add_argument("--success_tol_hz", type=float, default=0.25,
                    help="收敛成功判据: 最终误差 < tol")
    ap.add_argument("--diverge_tol_hz", type=float, default=2.0,
                    help="发散判据: 最终误差 > tol (跑到邻 bin/远处)")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    offsets = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

    runmod = _load_runmod()
    args = runmod.parse_args([
        "--candidate_support_mode", "basin", "--candidate_mode", "profile_ls",
        "--basin_min_half_width_hz", "8.0", "--global_bins", "512", "--top_j", "24",
        "--num_cycles", str(cli.num_cycles), "--train_samples", str(cli.num_samples),
        "--profile_ridge", str(cli.profile_ridge),
        "--recon_mode", "point_ls" if cli.mode == "profile_map" else "marginal",
    ])
    ds = runmod.LegacyFourComponentCandidateDataset("train", cli.num_samples, cli.seed, args)
    decoder = runmod.JointConstantFrequencyDecoder(
        recon_mode="point_ls" if cli.mode == "profile_map" else "marginal",
        ridge=cli.profile_ridge)

    insts = []
    for i in range(cli.num_samples):
        inst = ds[i]
        tgt = inst["target"].numpy()
        insts.append({
            "y": torch.tensor(tgt[:, 0] + 1j * tgt[:, 1], dtype=torch.complex64).view(1, -1),
            "t": torch.tensor(inst["t"].numpy(), dtype=torch.float32).view(1, -1),
            "mask": torch.tensor(inst["mask"].numpy().astype(np.float32)).view(1, -1),
            "tf": inst["true_freq_hz"].numpy().astype(np.float64),
            "nv": torch.tensor([float(inst["noise_var_norm"])], dtype=torch.float32),
        })

    print(f"=== B1 GN 捕获域扫描 [{cli.mode}] N={cli.num_samples} c{cli.num_cycles} steps={cli.steps} ===")
    print(f"{'offset_hz':>10}{'final_MAE':>11}{'success%':>10}{'diverge%':>10}")
    for off in offsets:
        fin_err, succ, div = [], 0, 0
        n = 0
        for s in insts:
            tf = s["tf"]
            # 给【所有】分量加同向 offset (单参数捕获域: 看 GN 能否从 off 拉回)
            f0 = torch.tensor(tf + off, dtype=torch.float32).clamp_min(1e-6)
            f_ref = newton_refine(decoder, s["y"], s["t"], f0, s["mask"], s["nv"],
                                  cli.mode, steps=cli.steps).numpy()
            fr = np.sort(f_ref); tt = np.sort(tf)
            e = np.abs(fr - tt)
            fin_err.append(e)
            for ei in e:
                n += 1
                if ei < cli.success_tol_hz:
                    succ += 1
                if ei > cli.diverge_tol_hz:
                    div += 1
        fe = np.concatenate(fin_err)
        print(f"{off:>10.2f}{fe.mean():>11.4f}{100.0*succ/n:>9.1f}%{100.0*div/n:>9.1f}%")

    print("\n判读: FR 实测初值误差 mean 0.52/p95 1.11/max 1.84 Hz (plan.md §3)。")
    print("若 |offset|<=1.0 的 success% 高、diverge% 低 → FR 主体在捕获域内;sub-bin 插值兜尾部。")


if __name__ == "__main__":
    main()
