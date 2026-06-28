"""判定对照: 原 greedy 候选 (不 monkeypatch FR) 跑 production 配置的 unsup ELBO,
看 delta 在「候选 raw 已 ~0.16」时到底动不动。

production 实测: raw_freq_mae=0.1625 -> 精修后 0.1693 (delta 几乎没起作用甚至帮倒忙)。
这里复刻 production 的精确 args, 从 production checkpoint 继续训, 打印每 epoch 的
freq_mae vs raw_freq_mae 与 delta_logvar_mean:

  - 若 freq_mae 始终 ≈ raw (delta 不动) -> 证实 ELBO 的局部精修在 production 里
    从未真正工作过, 0.169 全靠 greedy 候选本身。这正面回应「ELBO 做不了训练损失」。
  - 若 freq_mae 明显 < raw -> ELBO 精修确实有效, 那 FR 路线 0.50 卡住是 FR 候选特有
    的障碍 (网格量化), 而非 ELBO 本身。
"""
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BTT = _HERE.parent / "btt_amortized_vi_project"
for p in (str(_HERE), str(_BTT), str(_BTT / "legacy_reuse" / "current_project")):
    if p not in sys.path:
        sys.path.insert(0, str(p))


def _load_runmod():
    spec = importlib.util.spec_from_file_location(
        "run_legacy_four_vi_greedy", _BTT / "run_legacy_four_component_candidate_vi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod_checkpoint", default=str(
        _BTT / "artifacts/legacy_four_component_candidate_vi/remote_gpu_archive/"
        "c16_production_repulsion/four_component_candidate_encoder.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out_dir", default="comparison/artifacts/greedy_control_stage2")
    cli = ap.parse_args()

    runmod = _load_runmod()  # 注意: 不 monkeypatch, 用原生 greedy 候选

    # 精确复刻 production 的 unsup ELBO 配置 (来自 metrics.json["args"])
    argv = [
        "--candidate_support_mode", "basin",
        "--candidate_mode", "alias_hybrid",        # ← 原生 greedy
        "--basin_min_half_width_hz", "8.0",
        "--basin_half_width_bins", "1.0",
        "--basin_candidate_min_gap_bins", "2",
        "--global_bins", "512", "--top_j", "24",
        "--num_cycles", "16", "--train_samples", "2000",
        "--snr_db", "20.0",
        "--unsupervised_elbo_train",
        "--elbo_frequency_source", "cartesian_topk", "--cartesian_topk", "3",
        "--recon_weight", "0.01",
        "--sample_mode", "mean",
        "--local_nll_weight", "0.0",
        "--local_kl_weight", "0.001",
        "--z_loss_type", "none", "--z_loss_weight", "0.0",
        "--physics_prior_weight", "0.0",
        "--z_repulsion_weight", "1.0",
        "--local_radius_hz", "0.25",
        "--profile_ridge", "1e-3",   # 防奇异 Gram (production 用 1e-5 但本 seed 会崩)
        "--lr", "0.001", "--batch_size", "64",
        "--coverage_tol_hz", "1.0",
        "--init_encoder_checkpoint", cli.prod_checkpoint,
        "--epochs", str(cli.epochs),
        "--out_dir", cli.out_dir,
    ]
    print("=== greedy 对照: production 配置 unsup ELBO (原生 greedy 候选) ===")
    runmod.train(runmod.parse_args(argv))
    print(f"[done] -> {cli.out_dir}")


if __name__ == "__main__":
    main()
