import argparse
import copy
import json
from pathlib import Path

import numpy as np

import main as train_main
from config import CONFIG


def deep_update(d, u):
    d = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d


def safe_name(x):
    return str(x).replace("-", "m").replace(".", "p")


PROBE_CONFIGS = {
    3: {
        "num_probes": 3,
        "probes": [0, 360 * 1 / 7, 360 * 3 / 7],
    },
    4: {
        "num_probes": 4,
        "probes": [0, 360 * 1 / 13, 360 * 4 / 13, 360 * 6 / 13],
    },
    5: {
        "num_probes": 5,
        "probes": [0, 360 * 1 / 19, 360 * 2 / 19, 360 * 6 / 19, 360 * 9 / 19],
    },
    6: {
        "num_probes": 6,
        "probes": [0, 360 * 1 / 27, 360 * 2 / 27, 360 * 6 / 27, 360 * 10 / 27, 360 * 13 / 27],
    },
}


K_CONFIGS = {
    1: {
        "freqs_hz": [341.0],
        "amp_real_m": [0.0005403],
        "amp_imag_m": [0.0008415],
    },
    2: {
        "freqs_hz": [167.0, 341.0],
        "amp_real_m": [0.0006, 0.0005403],
        "amp_imag_m": [0.0, 0.0008415],
    },
    3: {
        "freqs_hz": [167.0, 341.0, 635.0],
        "amp_real_m": [0.0006, 0.0005403, -0.0003329],
        "amp_imag_m": [0.0, 0.0008415, 0.0007274],
    },
    4: {
        "freqs_hz": [167.0, 341.0, 635.0, 872.0],
        "amp_real_m": [0.0006, 0.0005403, -0.0003329, -0.0008910],
        "amp_imag_m": [0.0, 0.0008415, 0.0007274, 0.0001270],
    },
    5: {
        "freqs_hz": [167.0, 341.0, 635.0, 872.0, 930.0],
        "amp_real_m": [0.0006, 0.0005403, -0.0003329, -0.0008910, 0.0003000],
        "amp_imag_m": [0.0, 0.0008415, 0.0007274, 0.0001270, -0.0002000],
    },
    6: {
        "freqs_hz": [120.0, 167.0, 341.0, 635.0, 872.0, 930.0],
        "amp_real_m": [0.0003000, 0.0006, 0.0005403, -0.0003329, -0.0008910, 0.0003000],
        "amp_imag_m": [0.0001000, 0.0, 0.0008415, 0.0007274, 0.0001270, -0.0002000],
    },
}


def build_experiment_grid(experiment, base_cfg):
    if experiment == "exp1_snr":
        values = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 30]
        return [("snr_db", v, {"signal": {"snr_db": v}}) for v in values]

    if experiment == "exp2_n_revs":
        values = [1000, 2000, 5000, 10000, 20000, 40000]
        return [("n_revs", v, {"data": {"n_revs": v}}) for v in values]

    if experiment == "exp3a_window_revs":
        values = [4, 8, 16, 32, 64, 128]
        return [
            ("window_revs", v, {"data": {"window_revs": v, "hop_revs": max(1, v // 4)}})
            for v in values
        ]

    if experiment == "exp3b_num_probes":
        return [
            ("num_probes", p, {"data": {"num_probes": cfg["num_probes"], "probes": cfg["probes"]}})
            for p, cfg in PROBE_CONFIGS.items()
        ]

    if experiment == "exp4_num_harmonics":
        grid = []
        for k, cfg in K_CONFIGS.items():
            grid.append(
                (
                    "K",
                    k,
                    {
                        "data": {"num_harmonics": k},
                        "signal": {
                            "freqs_hz": cfg["freqs_hz"],
                            "amp_real_m": cfg["amp_real_m"],
                            "amp_imag_m": cfg["amp_imag_m"],
                        },
                        "prior": {
                            "f_center_hz": cfg["freqs_hz"],
                            "f_band_hz": base_cfg.get("prior", {}).get("f_band_hz", 15.0),
                        },
                    },
                )
            )
        return grid

    if experiment == "exp5a_prior_center_shift":
        ratios = [-0.10, -0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
        true_freqs = np.array(base_cfg["signal"]["freqs_hz"], dtype=float)
        grid = []
        for r in ratios:
            shifted = (true_freqs * (1.0 + r)).tolist()
            grid.append(("center_shift_ratio", r, {"prior": {"f_center_hz": shifted}}))
        return grid

    if experiment == "exp5b_prior_band":
        values = [5, 10, 15, 30, 50, 100]
        return [("f_band_hz", v, {"prior": {"f_band_hz": v}}) for v in values]

    if experiment == "exp5c_speed_fluctuation":
        values = [0.0, 0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.025, 0.05]
        return [("fluctuation_delta", v, {"data": {"fluctuation_delta": v}}) for v in values]

    raise ValueError(f"Unknown experiment: {experiment}")


def run_grid(args):
    result_root = Path(args.result_root) / args.model_name / args.experiment
    result_root.mkdir(parents=True, exist_ok=True)

    base_cfg = copy.deepcopy(CONFIG)
    base_cfg.setdefault("loss", {})
    base_cfg["loss"]["freq_success_tol_hz"] = args.freq_tol_hz
    base_cfg["loss"]["amp_success_tol_m"] = args.amp_tol_m

    base_cfg.setdefault("prior", {})
    base_cfg["prior"].setdefault("f_center_hz", base_cfg["signal"]["freqs_hz"])
    base_cfg["prior"].setdefault("f_band_hz", 15.0)

    grid = build_experiment_grid(args.experiment, base_cfg)
    seeds = [int(s) for s in args.seeds.split(",")]

    for factor_name, factor_value, overrides in grid:
        for seed in seeds:
            cfg = deep_update(base_cfg, overrides)
            cfg["seed"] = seed
            cfg["model_name"] = args.model_name

            run_dir = result_root / f"{factor_name}_{safe_name(factor_value)}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"[SKIP] {metrics_path}")
                continue

            cfg["run_dir"] = str(run_dir)
            cfg.setdefault("checkpoint", {})
            cfg.setdefault("logging", {})

            cfg["checkpoint"]["dir"] = str(run_dir / "checkpoints")
            cfg["checkpoint"]["name"] = "latest.pt"
            cfg["checkpoint"]["resume_from"] = None

            cfg["logging"]["tensorboard_dir"] = str(run_dir / "tensorboard")
            cfg["logging"]["curve_dir"] = str(run_dir / "curves")

            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            print(
                f"[RUN] model={args.model_name} "
                f"experiment={args.experiment} "
                f"{factor_name}={factor_value} seed={seed}"
            )

            train_main.CONFIG = cfg
            train_main.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--result_root", type=str, default="/content/drive/MyDrive/deepmusic_results/v3")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--freq_tol_hz", type=float, default=1.0)
    parser.add_argument("--amp_tol_m", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_grid(args)
