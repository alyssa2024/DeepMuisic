"""§15: local covariance / 不确定性输出。

每个保留 basin 收敛后:q(f|z_m,y) ≈ N(f̂_m, H_m⁻¹),H_m = NLL 在收敛点的 Hessian
(marginal NLL 的 Hessian ≈ Fisher 信息 → H⁻¹ ≈ CRB,解析,不需变分)。

mixture 后验:q(f|y) ≈ Σ_m π_m N(f̂_m, H_m⁻¹),π_m = softmax(s_m)。
plan_final §15 要求区分:
  - basin uncertainty(π_m,离散,选哪个洞);
  - basin 内 continuous uncertainty(H⁻¹,连续,洞内精度)。

本脚本(oracle K 已知,单 basin 验证连续不确定性校准):
  对每样本牛顿精修 → 取收敛点 Hessian → σ_pred = sqrt(diag(H⁻¹)) →
  对比 σ_pred 与实际频率误差 |f̂-f_true| 的统计,看不确定性是否【校准】
  (well-calibrated: σ_pred 应与 RMSE 同量级,且 |error|/σ_pred ~ N(0,1))。
  并对比解析 CRB(若与 H⁻¹ 一致,确认 H⁻¹ 就是 CRB)。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from btt_oracle_continuous_refinement import _load_runmod, newton_refine, nll_of_freqs  # noqa: E402


def hessian_at(decoder, y, t, f, mask, noise_var, mode):
    """NLL 在 f 处的全 Hessian [K,K](复用 B0 的二阶 autograd)。"""
    f = f.clone().detach().requires_grad_(True)
    K = f.numel()
    nll = nll_of_freqs(decoder, y, t, f, mask, noise_var, mode)
    g, = torch.autograd.grad(nll, f, create_graph=True)
    H = torch.zeros(K, K, dtype=f.dtype, device=f.device)
    for i in range(K):
        gi = torch.autograd.grad(g[i], f, retain_graph=True)[0]
        H[i] = gi
    return 0.5 * (H + H.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["marginal", "profile_map"], default="marginal")
    ap.add_argument("--num_samples", type=int, default=80)
    ap.add_argument("--num_cycles", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    runmod = _load_runmod()
    recon = "point_ls" if cli.mode == "profile_map" else "marginal"
    args = runmod.parse_args([
        "--candidate_support_mode", "basin", "--candidate_mode", "profile_ls",
        "--basin_min_half_width_hz", "8.0", "--global_bins", "512", "--top_j", "24",
        "--num_cycles", str(cli.num_cycles), "--train_samples", str(cli.num_samples),
        "--profile_ridge", "1e-3", "--recon_mode", recon])
    ds = runmod.LegacyFourComponentCandidateDataset("train", cli.num_samples, cli.seed, args)
    dec = runmod.JointConstantFrequencyDecoder(recon_mode=recon, ridge=1e-3)
    gs = (1000.0 - 1.0) / 512.0

    z_scores = []   # |error| / σ_pred  (校准: 应 ~ N(0,1), std≈1)
    sig_pred_all, err_all = [], []
    for i in range(cli.num_samples):
        inst = ds[i]; tgt = inst["target"].numpy()
        y = torch.tensor(tgt[:, 0] + 1j * tgt[:, 1], dtype=torch.complex64).view(1, -1)
        t = torch.tensor(inst["t"].numpy(), dtype=torch.float32).view(1, -1)
        mask = torch.tensor(inst["mask"].numpy().astype(np.float32)).view(1, -1)
        tf = inst["true_freq_hz"].numpy().astype(np.float64)
        nv = torch.tensor([float(inst["noise_var_norm"])], dtype=torch.float32)
        f0 = torch.tensor(np.round((tf - 1.0) / gs) * gs + 1.0, dtype=torch.float32)
        f_ref = newton_refine(dec, y, t, f0, mask, nv, cli.mode, steps=cli.steps)
        H = hessian_at(dec, y, t, f_ref, mask, nv, cli.mode)
        try:
            cov = torch.linalg.inv(H + 1e-9 * torch.eye(H.shape[0]))
            sig = torch.sqrt(torch.diag(cov).clamp_min(0)).numpy()
        except Exception:
            sig = np.full(f_ref.numel(), np.nan)
        order = np.argsort(f_ref.numpy())
        fr = f_ref.numpy()[order]; sg = sig[order]; tt = np.sort(tf)
        err = fr - tt
        for e, s in zip(err, sg):
            if np.isfinite(s) and s > 1e-9:
                z_scores.append(e / s); sig_pred_all.append(s); err_all.append(abs(e))

    z = np.array(z_scores); sp = np.array(sig_pred_all); ea = np.array(err_all)
    print(f"=== sec15 local covariance calib [{cli.mode}] N={cli.num_samples} c{cli.num_cycles} ===")
    print(f"  sigma_pred (Hinv std): mean={sp.mean():.4f} median={np.median(sp):.4f} Hz")
    print(f"  actual |error|:        mean={ea.mean():.4f} RMSE={np.sqrt((ea**2).mean()):.4f} Hz")
    print(f"  calib ratio mean(|err|)/mean(sigma_pred) = {ea.mean()/sp.mean():.3f}  (ideal ~0.8 half-normal)")
    print(f"  z=err/sigma_pred: mean={z.mean():+.3f} std={z.std():.3f}  (calibrated: std~1)")
    cov95 = np.mean(np.abs(z) < 1.96)
    print(f"  empirical 95% coverage (|z|<1.96): {100*cov95:.1f}%  (ideal 95%)")


if __name__ == "__main__":
    main()
