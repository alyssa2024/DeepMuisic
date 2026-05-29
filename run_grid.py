import argparse
import copy
import json
from pathlib import Path

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
    3: {"num_probes": 3, "probes": [0, 360 * 1 / 7, 360 * 3 / 7]},
    4: {"num_probes": 4, "probes": [0, 360 * 1 / 13, 360 * 4 / 13, 360 * 6 / 13]},
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
        "center_hz": [341.0],
        "amp_real_center_m": [0.0005403],
        "amp_imag_center_m": [0.0008415],
    },
    2: {
        "center_hz": [167.0, 341.0],
        "amp_real_center_m": [0.0006, 0.0005403],
        "amp_imag_center_m": [0.0, 0.0008415],
    },
    3: {
        "center_hz": [167.0, 341.0, 635.0],
        "amp_real_center_m": [0.0006, 0.0005403, -0.0003329],
        "amp_imag_center_m": [0.0, 0.0008415, 0.0007274],
    },
    4: {
        "center_hz": [167.0, 341.0, 635.0, 872.0],
        "amp_real_center_m": [0.0006, 0.0005403, -0.0003329, -0.0008910],
        "amp_imag_center_m": [0.0, 0.0008415, 0.0007274, 0.0001270],
    },
}


def build_experiment_grid(experiment, base_cfg):
    if experiment == "exp1_snr":
        values = base_cfg.get("experiment", {}).get("snr_values", [-5, 0, 5, 10, 15, 20])
        return [("snr_db", v, {"signal": {"snr_db": v}}) for v in values]

    if experiment == "exp2_num_cycles":
        values = [4, 8, 16, 32, 64]
        return [("num_cycles", v, {"data": {"num_cycles": v}}) for v in values]

    if experiment == "exp3_num_probes":
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
                        "frequency": {"center_hz": cfg["center_hz"]},
                        "signal": {
                            "amp_real_center_m": cfg["amp_real_center_m"],
                            "amp_imag_center_m": cfg["amp_imag_center_m"],
                        },
                    },
                )
            )
        return grid

    if experiment == "exp5_relative_half_band":
        values = [0.01, 0.02, 0.05, 0.08, 0.10]
        return [("relative_half_band", v, {"frequency": {"relative_half_band": v}}) for v in values]

    if experiment == "exp6_sequence_posterior_samples":
        values = [1, 2, 4]
        return [
            (
                "sequence_posterior_samples",
                v,
                {"loss": {"reconstruction": {"sequence_posterior_samples": v}}},
            )
            for v in values
        ]

    if experiment == "exp7_amp_prior_band":
        values = [0.0, 0.1, 0.2, 0.4]
        return [
            (
                "amp_relative_half_band",
                v,
                {"signal": {"amp_data_prior": {"relative_half_band": v}}},
            )
            for v in values
        ]

    raise ValueError(f"Unknown experiment: {experiment}")


def run_grid(args):
    result_root = Path(args.result_root) / args.model_name / args.experiment
    result_root.mkdir(parents=True, exist_ok=True)

    base_cfg = copy.deepcopy(CONFIG)
    base_cfg.setdefault("training", {})
    base_cfg["training"]["early_stopping"] = {
        "enabled": True,
        "monitor": args.monitor,
        "mode": "min",
        "patience": 3,
        "min_delta": 0.0,
    }
    base_cfg.setdefault("loss", {})
    base_cfg["loss"].setdefault("success", {})
    base_cfg["loss"]["success"]["freq_relative_tol"] = args.freq_relative_tol
    base_cfg["loss"]["success"]["amp_relative_tol"] = args.amp_relative_tol
    base_cfg["loss"]["success"]["complex_coeff_relative_tol"] = args.complex_coeff_relative_tol

    grid = build_experiment_grid(args.experiment, base_cfg)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [int(s) for s in base_cfg.get("experiment", {}).get("seeds", [0])]

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
    parser.add_argument("--result_root", type=str, default="artifacts/v3/prior_sampling")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--freq_relative_tol", type=float, default=0.02)
    parser.add_argument("--amp_relative_tol", type=float, default=0.05)
    parser.add_argument("--complex_coeff_relative_tol", type=float, default=0.05)
    parser.add_argument("--monitor", type=str, default="freq_rmse_hz_mean")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_grid(args)
