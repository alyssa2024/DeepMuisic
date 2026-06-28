"""Decisive test: does the recon NLL actually have CURVATURE in delta?

delta collapses to the logvar clamp (-7.4 ~ -8) under BOTH kl=0 and marginal, and
freq_mae stays frozen at raw 0.50. That points to recon being FLAT in delta (no
downhill direction) rather than a KL/amplitude-prior issue.

Here we bypass the encoder entirely: take real BTT samples, get FR candidates, pick
the candidate nearest each true freq, then sweep a MANUAL delta on the matched
component (others held at their nearest candidate) and plot recon NLL vs the
resulting frequency offset. If NLL has a clear min at the true freq -> landscape is
fine, problem is optimization/encoder. If NLL is flat -> the recon objective itself
cannot localize delta (amplitude/other-combos wash it out).
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
    fr_pool = _fr_pool_factory(fr_model, grid_hz, 1.0)

    # Build a small dataset the SAME way the trainer does.
    argv = ["--candidate_support_mode", "basin", "--candidate_mode", "profile_ls",
            "--basin_min_half_width_hz", "8.0", "--global_bins", "512", "--top_j", "24",
            "--num_cycles", "16", "--train_samples", "8"]
    runmod.make_greedy_basin_pool_candidates = fr_pool
    args = runmod.parse_args(argv)
    ds = runmod.LegacyFourComponentCandidateDataset("train", 8, 0, args)

    # point_ls 与 marginal 两个解码器
    dec_pls = runmod.JointConstantFrequencyDecoder(recon_mode="point_ls", ridge=1e-3)
    dec_mrg = runmod.JointConstantFrequencyDecoder(recon_mode="marginal", amp_prior_var_scale=1.0)

    deltas = np.linspace(-1.0, 1.0, 41)
    for si in range(4):
        inst = ds[si]
        tgt = inst["target"].numpy()
        y = (tgt[:, 0].astype(np.float64) + 1j * tgt[:, 1].astype(np.float64))
        y = torch.as_tensor(y, dtype=torch.complex64)[None]
        t = torch.as_tensor(inst["t"].numpy(), dtype=torch.float32)[None]
        mask = torch.as_tensor(inst["mask"].numpy().astype(np.float32), dtype=torch.float32)[None]
        cand = inst["candidate_freq_hz"].to(torch.float32)        # [K,J]
        rad = inst["local_radius_hz"].to(torch.float32)
        true_f = inst["true_freq_hz"].numpy().astype(np.float64)
        nv = torch.as_tensor([float(inst["noise_var_norm"])], dtype=torch.float32)
        K = cand.shape[0]
        # nearest candidate per true component
        near_idx = [(cand[k] - true_f[k]).abs().argmin().item() for k in range(K)]
        base = torch.tensor([cand[k, near_idx[k]] for k in range(K)])  # [K]
        radK = torch.tensor([rad[k, near_idx[k]] for k in range(K)])
        raw_off = (base.numpy() - true_f)
        print(f"\n=== sample {si} true={np.round(true_f,2)} "
              f"nearest_cand={np.round(base.numpy(),2)} raw_off={np.round(raw_off,3)} "
              f"radius={np.round(radK.numpy(),3)} ===")
        # sweep delta on component 0, others fixed at base
        comp = 0
        pls, mrg = [], []
        for dv in deltas:
            f = base.clone()
            f[comp] = base[comp] + radK[comp] * dv
            f = f.clamp_min(1e-6)[None]  # [1,K]
            with torch.no_grad():
                n1 = dec_pls.reconstruction_nll_per_sample(y, t, f, mask, nv).item()
                n2 = dec_mrg.reconstruction_nll_per_sample(y, t, f, mask, nv).item()
            pls.append(n1); mrg.append(n2)
        pls = np.array(pls); mrg = np.array(mrg)
        # offset that the true freq corresponds to (delta needed to hit truth)
        true_delta = (true_f[comp] - base[comp].item()) / max(radK[comp].item(), 1e-9)
        amin_pls = deltas[pls.argmin()]; amin_mrg = deltas[mrg.argmin()]
        print(f"  comp0 true_delta={true_delta:+.3f}  "
              f"argmin_delta point_ls={amin_pls:+.3f} marginal={amin_mrg:+.3f}")
        print(f"  point_ls NLL range [{pls.min():.2f},{pls.max():.2f}] "
              f"span={pls.max()-pls.min():.3f}  (flat? {pls.max()-pls.min()<1e-2})")
        print(f"  marginal NLL range [{mrg.min():.2f},{mrg.max():.2f}] "
              f"span={mrg.max()-mrg.min():.3f}")
        # show the curve coarsely
        idx = np.linspace(0, len(deltas)-1, 9).astype(int)
        print("  delta:   " + " ".join(f"{deltas[i]:+.2f}" for i in idx))
        print("  pls-min: " + " ".join(f"{pls[i]-pls.min():5.2f}" for i in idx))
        print("  mrg-min: " + " ".join(f"{mrg[i]-mrg.min():5.2f}" for i in idx))


if __name__ == "__main__":
    main()
