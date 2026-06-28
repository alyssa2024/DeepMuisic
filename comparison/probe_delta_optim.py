"""判定实验 2：recon 梯度能不能真的把 delta_mu 推到真频？

probe_delta_curvature 已证明 recon 对 delta 曲率很强、最小值在真频附近。但训练里
delta_mu 不动。这里绕过整个 encoder，把 delta_mu 设成可训练的自由参数（每个样本、
每个分量一个标量），只用 cartesian_topk recon 梯度去优化它，看它能否收敛到真频对应
的 delta。

- 若收敛到真频 -> recon 梯度路径完全 OK，问题在训练配置（encoder 的 delta_head 学
  不动 / 被别的项压制 / lr 太小 / qz 不够尖峰把梯度糊掉）。
- 若不收敛 -> cartesian_topk 的梯度组装本身有问题。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_BTT = _HERE.parent / "btt_amortized_vi_project"
for p in (str(_HERE), str(_HERE / "deepfreq_official"),
          str(_BTT), str(_BTT / "legacy_reuse" / "current_project")):
    if p not in sys.path:
        sys.path.insert(0, str(p))

from deepfreq_btt_geomrobust import build_front  # noqa: E402
from run_frfront_elbo import _fr_pool_factory, _load_runmod  # noqa: E402


def main():
    runmod = _load_runmod()
    ck = torch.load("comparison/artifacts/geomrobust_nudft_speedrand/geomrobust_fr.pt",
                    map_location="cpu", weights_only=False)
    from argparse import Namespace
    fa = Namespace(**ck["args"])
    fr_model = build_front(fa, int(fa.num_cycles) * 4)
    fr_model.load_state_dict(ck["model"]); fr_model.eval()
    grid_hz = np.linspace(float(fa.global_fmin_hz), float(fa.global_fmax_hz),
                          int(fa.fr_size), endpoint=False)
    runmod.make_greedy_basin_pool_candidates = _fr_pool_factory(fr_model, grid_hz, 1.0)

    argv = ["--candidate_support_mode", "basin", "--candidate_mode", "profile_ls",
            "--basin_min_half_width_hz", "8.0", "--global_bins", "512", "--top_j", "24",
            "--num_cycles", "16", "--train_samples", "8"]
    args = runmod.parse_args(argv)
    ds = runmod.LegacyFourComponentCandidateDataset("train", 8, 0, args)
    dec = runmod.JointConstantFrequencyDecoder(recon_mode="point_ls", ridge=1e-3)

    N = 4
    insts = [ds[i] for i in range(N)]
    K = insts[0]["candidate_freq_hz"].shape[0]
    # 用每个样本里离真频最近的候选当 basin center（模拟 Stage1 完美选择）
    cand = torch.stack([insts[i]["candidate_freq_hz"] for i in range(N)])    # [N,K,J]
    rad = torch.stack([insts[i]["local_radius_hz"] for i in range(N)])
    true_f = torch.stack([insts[i]["true_freq_hz"].to(torch.float32) for i in range(N)])  # [N,K]
    near = (cand - true_f.unsqueeze(-1)).abs().argmin(dim=-1)                # [N,K]
    base = torch.gather(cand, 2, near.unsqueeze(-1)).squeeze(-1)             # [N,K] 选中候选频率
    radK = torch.gather(rad, 2, near.unsqueeze(-1)).squeeze(-1)             # [N,K]
    y = torch.stack([torch.complex(insts[i]["target"][:, 0], insts[i]["target"][:, 1])
                     for i in range(N)]).to(torch.complex64)               # [N,L]
    t = torch.stack([insts[i]["t"].to(torch.float32) for i in range(N)])
    mask = torch.stack([insts[i]["mask"].to(torch.float32) for i in range(N)])
    nv = torch.tensor([float(insts[i]["noise_var_norm"]) for i in range(N)], dtype=torch.float32)

    target_delta = ((true_f - base) / radK).clamp(-1, 1)
    raw_off = (base - true_f).abs().mean().item()
    print(f"raw freq off |base-true| mean = {raw_off:.4f} Hz")
    print(f"target_delta:\n{target_delta.numpy().round(3)}")

    # delta_mu 作为自由参数（pre-tanh），初始 0 -> tanh=0 -> 频率=候选中心(=raw)
    raw_mu = torch.zeros(N, K, requires_grad=True)
    opt = torch.optim.Adam([raw_mu], lr=0.05)
    for step in range(400):
        opt.zero_grad()
        delta = torch.tanh(raw_mu)
        freqs = (base + radK * delta).clamp_min(1e-6)     # [N,K] 全 K 个分量一起精修
        nll = dec.reconstruction_nll_per_sample(y, t, freqs, mask, nv).mean()
        nll.backward()
        opt.step()
        if step % 50 == 0 or step == 399:
            with torch.no_grad():
                d = torch.tanh(raw_mu)
                f = (base + radK * d)
                mae = (f - true_f).abs().mean().item()
                dmae = (d - target_delta).abs().mean().item()
                print(f"step{step:3d} recon={nll.item():8.2f} freq_mae={mae:.4f} "
                      f"delta_err={dmae:.4f} |grad|={raw_mu.grad.abs().mean().item():.4e}")
    print("\n最终 delta:\n", torch.tanh(raw_mu).detach().numpy().round(3))
    print("应到 target_delta:\n", target_delta.numpy().round(3))


if __name__ == "__main__":
    main()
