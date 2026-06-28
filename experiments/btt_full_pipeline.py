"""§14 完整 hybrid 推理 pipeline(Batch 4 主线)。

串联已就绪的三块(均不重训):
  FR-front 候选生成 → q(z) selection 选 top-L → 每候选【联合牛顿】精修(=B0 主力 solver)
  → s_m = -RSS/σ² + γ·log q 打分 → 报告 physics-only / encoder-only / hybrid 三路。

K 已知(=4)。复用:
  - FR-front ckpt + _fr_pool_factory (候选)
  - 8 维 q(z) Stage1 ckpt (selection, top1=1.0)
  - newton_refine (B0 solver) + JointConstantFrequencyDecoder (marginal NLL / RSS)

三路定义(plan_final §14,不得只报 hybrid):
  - physics-only: 只用 FR_score 排序选候选 (不用 encoder)
  - encoder-only: 只用 q(z) log 概率选候选
  - hybrid:       s_m = -RSS/σ² + γ·log q  (精修后 RSS + 选择概率)
全部都接【联合牛顿精修】,差异仅在"选哪个候选"。
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
_CMP = _ROOT / "comparison"
sys.path.insert(0, str(_HERE))
for p in (str(_ROOT), str(_CMP), str(_CMP / "deepfreq_official"),
          str(_BTT), str(_BTT / "legacy_reuse" / "current_project")):
    if p not in sys.path:
        sys.path.insert(0, str(p))

from btt_oracle_continuous_refinement import _load_runmod, newton_refine  # noqa: E402
from deepfreq_btt_geomrobust import build_front  # noqa: E402
from run_frfront_elbo import _fr_pool_factory  # noqa: E402


def build_encoder_from_ckpt(runmod, args, num_components, ckpt_path, device="cpu"):
    enc = runmod.FourComponentCandidateEncoder(
        input_dim=8 if args.include_local_time else 6, cand_dim=6,
        num_components=num_components, hidden_dim=args.hidden_dim, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        hidden_dim_dense=args.hidden_dim_dense, dropout=0.0,
        logvar_param=args.delta_logvar_param, logvar_min=args.delta_logvar_min,
        logvar_max=args.delta_logvar_max, selection_mode=args.selection_mode,
        num_probes=int(runmod.CONFIG["data"]["num_probes"]),
        use_probe_embedding=args.use_probe_embedding, time_encoding=args.time_encoding,
        time_num_freqs=args.time_num_freqs, time_min_hz=args.time_min_hz,
        time_max_hz=args.time_max_hz).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc.load_state_dict(sd); enc.eval()
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr_checkpoint", default="comparison/artifacts/geomrobust_nudft_speedrand/geomrobust_fr.pt")
    ap.add_argument("--stage1_ckpt", default="comparison/artifacts/frfront_elbo_localtime_stage1/four_component_candidate_encoder.pt")
    ap.add_argument("--num_samples", type=int, default=60)
    ap.add_argument("--num_cycles", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--mode", choices=["marginal", "profile_map"], default="marginal")
    ap.add_argument("--snr_db", type=float, default=20.0,
                    help="降 SNR 是难场景:K=4 同频率中心(encoder 不 OOD),只增噪声。"
                         "看 physics-only(FR峰高选)何时崩 vs encoder/hybrid 是否更稳。")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    runmod = _load_runmod()
    ck = torch.load(cli.fr_checkpoint, map_location="cpu", weights_only=False)
    from argparse import Namespace
    fa = Namespace(**ck["args"])
    fr_model = build_front(fa, int(fa.num_cycles) * 4); fr_model.load_state_dict(ck["model"]); fr_model.eval()
    grid = np.linspace(float(fa.global_fmin_hz), float(fa.global_fmax_hz), int(fa.fr_size), endpoint=False)
    runmod.make_greedy_basin_pool_candidates = _fr_pool_factory(fr_model, grid, 1.0)

    recon = "point_ls" if cli.mode == "profile_map" else "marginal"
    args = runmod.parse_args([
        "--candidate_support_mode", "basin", "--candidate_mode", "profile_ls",
        "--basin_min_half_width_hz", "8.0", "--global_bins", "512", "--top_j", "24",
        "--num_cycles", str(cli.num_cycles), "--train_samples", str(cli.num_samples),
        "--snr_db", str(cli.snr_db),
        "--include_local_time", "--profile_ridge", "1e-3", "--recon_mode", recon])
    ds = runmod.LegacyFourComponentCandidateDataset("train", cli.num_samples, cli.seed, args)
    K = ds.num_components
    enc = build_encoder_from_ckpt(runmod, args, K, cli.stage1_ckpt)
    dec = runmod.JointConstantFrequencyDecoder(recon_mode=recon, ridge=1e-3)

    routes = {"physics": [], "encoder": [], "hybrid": [], "raw_sel": []}
    for i in range(cli.num_samples):
        inst = ds[i]
        x = inst["x"].unsqueeze(0); mask = inst["mask"].unsqueeze(0)
        cf = inst["candidate_features"].unsqueeze(0); t = inst["t"].unsqueeze(0)
        cand = inst["candidate_freq_hz"]            # [K, J]
        tgt = inst["target"].numpy()
        y = torch.tensor(tgt[:, 0] + 1j * tgt[:, 1], dtype=torch.complex64)
        tt = torch.tensor(inst["t"].numpy(), dtype=torch.float32)
        m1 = torch.tensor(inst["mask"].numpy().astype(np.float32))
        tf = inst["true_freq_hz"].numpy().astype(np.float64)
        nv = torch.tensor([float(inst["noise_var_norm"])], dtype=torch.float32)
        pid = inst.get("probe_ids", None)
        pid = pid.unsqueeze(0) if pid is not None else None
        with torch.no_grad():
            out = enc(x, mask, cf, probe_ids=pid, t=t)
            logq = torch.log_softmax(out["z_logits"], dim=-1)[0]  # [K, J]
        fr_score = cf[0, :, :, 1]                   # FR_score 特征通道(归一化峰高)

        # 每个 slot 选 1 个候选(K 已知)。三路差异在"选哪个 j"
        def refine_and_rss(sel_idx):
            f0 = torch.tensor([float(cand[k, sel_idx[k]]) for k in range(K)], dtype=torch.float32)
            f_ref = newton_refine(dec, y.view(1, -1), tt.view(1, -1), f0, m1.view(1, -1), nv, cli.mode, steps=cli.steps)
            rss = dec.reconstruction_nll_per_sample(y.view(1, -1), tt.view(1, -1), f_ref.view(1, -1), m1.view(1, -1), nv).item()
            err = np.abs(np.sort(f_ref.numpy()) - np.sort(tf))
            return f_ref, rss, err

        # 每 slot 选 1 候选, 但 K 个 slot 不能撞同一候选 -> 贪婪去重分配:
        # 按 score 降序遍历 (slot, j), 占用未被占的 (slot, j), 直到每个 slot 有一个不同候选频率。
        def assign_distinct(score):  # score: [K, J] tensor
            sc = score.clone()
            sel = [None] * K
            used_freq = []
            flat = [(float(sc[k, j]), k, j) for k in range(K) for j in range(sc.shape[1])]
            flat.sort(reverse=True)
            for _, k, j in flat:
                if sel[k] is not None:
                    continue
                fj = float(cand[k, j])
                if any(abs(fj - uf) < 2.0 for uf in used_freq):  # 已占用近邻频率, 跳过
                    continue
                sel[k] = j; used_freq.append(fj)
                if all(s is not None for s in sel):
                    break
            for k in range(K):  # 兜底
                if sel[k] is None:
                    sel[k] = int(sc[k].argmax())
            return sel

        sel_phys = assign_distinct(fr_score)
        sel_enc = assign_distinct(logq)
        _, _, e_phys = refine_and_rss(sel_phys)
        _, rss_enc, e_enc = refine_and_rss(sel_enc)
        raw = np.abs(np.sort(np.array([float(cand[k, sel_enc[k]]) for k in range(K)])) - np.sort(tf))
        # hybrid: s_m = -RSS/σ² + γ·logq。对 encoder 的 top-2 候选/slot 组合精修后按 s_m 选最优。
        # 简化实现: 比较 "encoder 选" vs "physics 选" 两套, 按 -RSS/σ²+γ·sum(logq) 取优。
        _, rss_phys, _ = refine_and_rss(sel_phys)
        sig2 = float(nv.item())
        s_enc = -rss_enc / max(sig2, 1e-12) + cli.gamma * sum(float(logq[k, sel_enc[k]]) for k in range(K))
        s_phys = -rss_phys / max(sig2, 1e-12) + cli.gamma * sum(float(logq[k, sel_phys[k]]) for k in range(K))
        e_hyb = e_enc if s_enc >= s_phys else e_phys

        routes["physics"].append(e_phys); routes["encoder"].append(e_enc)
        routes["hybrid"].append(e_hyb); routes["raw_sel"].append(raw)

    print(f"=== §14 pipeline [{cli.mode}] N={cli.num_samples} c{cli.num_cycles} K={K} steps={cli.steps} ===")
    for name in ("physics", "encoder", "hybrid", "raw_sel"):
        e = np.concatenate(routes[name])
        print(f"  {name:>10}: freq MAE={e.mean():.4f}  RMSE={np.sqrt((e**2).mean()):.4f} Hz")
    print("  (raw_sel = encoder 选中候选未精修; 其余为精修后)")


if __name__ == "__main__":
    main()
