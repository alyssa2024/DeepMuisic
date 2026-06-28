"""B0 (plan_final §13.1 / §16): oracle basin 初值 + marginal/profile-MAP 目标 +
Gauss-Newton/牛顿 在【绝对频率】上的连续精修。验证能否复现 ~0.11 Hz。

与此前 `comparison/probe_delta_optim.py` 的区别(对齐表标注的"半验证"→"正式 B0"):
  - 那个用 Adam 优化 delta=(f-cand)/radius 的【δ 参数化】;
  - 这里直接在【绝对频率 f_k (Hz)】上做【阻尼牛顿/GN】,符合 plan_final B0 口径。

不依赖 encoder(oracle):用真频最近的候选/basin 中心作初值。K 已知。
复用现有 marginal/profile solver(JointConstantFrequencyDecoder),其频率 NLL 对
freqs_hz 全程可微 → 用 autograd 求一阶梯度 + Hessian 做牛顿步,无需手推 Jacobian。

报告三档:
  - init(=oracle basin 中心, 即 raw 初值精度);
  - newton(阻尼牛顿精修后);
  - 与 CRB 对照(若可得)。
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_BTT = _ROOT / "btt_amortized_vi_project"
for p in (str(_ROOT), str(_BTT), str(_BTT / "legacy_reuse" / "current_project")):
    if p not in sys.path:
        sys.path.insert(0, str(p))


def _load_runmod():
    spec = importlib.util.spec_from_file_location(
        "run_legacy_four_vi_b0", _BTT / "run_legacy_four_component_candidate_vi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nll_of_freqs(decoder, y, t, freqs, mask, noise_var, mode):
    """单样本频率 NLL (标量), freqs: [K] requires_grad。返回标量便于 autograd 求导。"""
    f = freqs.view(1, -1)
    if mode == "marginal":
        return decoder.marginal_nll_per_sample(y, t, f, mask, noise_var).sum()
    else:  # profile_map / point_ls
        return decoder.reconstruction_nll_per_sample(y, t, f, mask, noise_var).sum()


def newton_refine(decoder, y, t, f0, mask, noise_var, mode,
                  steps=8, damping=1e-3, max_step_hz=2.0):
    """阻尼牛顿/GN: 在绝对频率上精修。每步用 autograd 求 grad + Hessian,解 (H+λI)d=-g。"""
    f = f0.clone().detach().requires_grad_(True)
    K = f.numel()
    for _ in range(steps):
        nll = nll_of_freqs(decoder, y, t, f, mask, noise_var, mode)
        g, = torch.autograd.grad(nll, f, create_graph=True)
        # 全 Hessian (K 小, 直接逐列求二阶导)
        H = torch.zeros(K, K, dtype=f.dtype, device=f.device)
        for i in range(K):
            gi = torch.autograd.grad(g[i], f, retain_graph=True)[0]
            H[i] = gi
        H = 0.5 * (H + H.T)
        # 阻尼 (Levenberg): (H + λ·diag) d = -g
        lam = damping * (torch.diag(H).abs().mean() + 1e-12)
        A = H + lam * torch.eye(K, dtype=f.dtype, device=f.device)
        try:
            d = torch.linalg.solve(A, -g.detach())
        except Exception:
            d = -g.detach() / (torch.diag(A) + 1e-9)
        d = d.clamp(-max_step_hz, max_step_hz)
        with torch.no_grad():
            f_new = (f + d).clamp_min(1e-6)
        f = f_new.detach().requires_grad_(True)
    return f.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["marginal", "profile_map"], default="marginal")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--num_cycles", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--profile_ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    runmod = _load_runmod()
    # oracle: candidate_mode 用 alias_hybrid (greedy) 仅为生成候选/label; 我们只取 label 指向的
    # basin 中心做初值 —— 但 B0 是 oracle, 直接用【真频最近的 1.95Hz 网格点】当 basin 初值更纯。
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

    grid_spacing = (1000.0 - 1.0) / 512.0  # ~1.951 Hz, 模拟 FR 网格
    init_err, ref_err = [], []
    for i in range(cli.num_samples):
        inst = ds[i]
        tgt = inst["target"].numpy()
        y = torch.tensor(tgt[:, 0] + 1j * tgt[:, 1], dtype=torch.complex64).view(1, -1)
        t = torch.tensor(inst["t"].numpy(), dtype=torch.float32).view(1, -1)
        mask = torch.tensor(inst["mask"].numpy().astype(np.float32)).view(1, -1)
        tf = inst["true_freq_hz"].numpy().astype(np.float64)
        nv = torch.tensor([float(inst["noise_var_norm"])], dtype=torch.float32)
        # oracle basin 初值: 真频对齐到最近 1.95Hz 网格 (= FR 峰能给的最好初值)
        f0 = torch.tensor(np.round((tf - 1.0) / grid_spacing) * grid_spacing + 1.0,
                          dtype=torch.float32)
        init_err.append(np.abs(f0.numpy() - tf))
        f_ref = newton_refine(decoder, y, t, f0, mask, nv, cli.mode, steps=cli.steps)
        # 与真频按排序对应 (K 已知, 升序匹配)
        fr = np.sort(f_ref.numpy()); tt = np.sort(tf)
        ref_err.append(np.abs(fr - tt))
    init_err = np.concatenate(init_err); ref_err = np.concatenate(ref_err)
    print(f"=== B0 oracle 连续精修 [{cli.mode}] N={cli.num_samples} c{cli.num_cycles} steps={cli.steps} ===")
    print(f"  init (oracle basin, ~FR网格): MAE={init_err.mean():.4f}  RMSE={np.sqrt((init_err**2).mean()):.4f} Hz")
    print(f"  newton 精修后:                MAE={ref_err.mean():.4f}  RMSE={np.sqrt((ref_err**2).mean()):.4f} Hz")
    print(f"  目标 ~0.11 (自由参数侧证); LSF 0.127; CRB ~0.099")


if __name__ == "__main__":
    main()
