import argparse
import copy
import json
from pathlib import Path

import main as train_main
from config import CONFIG


NUM_CYCLES_VALUES = [4, 8, 12, 16, 32]


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


def build_sweep():
    return (
        "num_cycles",
        "num_cycles",
        NUM_CYCLES_VALUES,
        lambda v: {"data": {"num_cycles": v}},
    )


def prepare_base_config(args):
    base_cfg = copy.deepcopy(CONFIG)
    base_cfg.setdefault("training", {})
    base_cfg["training"]["early_stopping"] = {
        "enabled": True,
        "monitor": args.monitor,
        "mode": "min",
        "patience": args.early_patience,
        "min_delta": args.early_min_delta,
    }
    base_cfg.setdefault("loss", {})
    base_cfg["loss"].setdefault("success", {})
    base_cfg["loss"]["success"]["freq_relative_tol"] = args.freq_relative_tol
    base_cfg["loss"]["success"]["amp_relative_tol"] = args.amp_relative_tol
    base_cfg["loss"]["success"]["complex_coeff_relative_tol"] = (
        args.complex_coeff_relative_tol
    )
    return base_cfg


def run_one(cfg, run_dir, force, resume_from=None, resume_each=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    resume_requested = resume_each or resume_from is not None
    if metrics_path.exists() and not force and not resume_requested:
        print(f"[SKIP] {metrics_path}")
        return

    cfg["run_dir"] = str(run_dir)
    cfg.setdefault("checkpoint", {})
    cfg.setdefault("logging", {})
    cfg["checkpoint"]["dir"] = str(run_dir / "checkpoints")
    cfg["checkpoint"]["name"] = "latest.pt"
    if resume_each:
        run_checkpoint = run_dir / "checkpoints" / "latest.pt"
        if run_checkpoint.exists():
            cfg["checkpoint"]["resume_from"] = str(run_checkpoint)
        else:
            cfg["checkpoint"]["resume_from"] = None
            print(f"[WARN] No checkpoint for this run, starting fresh: {run_checkpoint}")
    else:
        cfg["checkpoint"]["resume_from"] = resume_from
    cfg["logging"]["tensorboard_dir"] = str(run_dir / "tensorboard")
    cfg["logging"]["curve_dir"] = str(run_dir / "curves")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    train_main.CONFIG = cfg
    train_main.main()


def run_sweep(args):
    base_cfg = prepare_base_config(args)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [int(s) for s in base_cfg.get("experiment", {}).get("seeds", [0])]

    result_root = Path(args.result_root) / args.model_name
    sweep_name, factor_name, values, override_fn = build_sweep()

    for value in values:
        for seed in seeds:
            cfg = deep_update(base_cfg, override_fn(value))
            cfg["seed"] = seed
            cfg["model_name"] = args.model_name
            cfg["sweep_name"] = sweep_name
            cfg["sweep_factor"] = factor_name
            cfg["sweep_value"] = value

            run_dir = (
                result_root
                / sweep_name
                / f"{factor_name}_{safe_name(value)}"
                / f"seed_{seed}"
            )
            print(
                f"[RUN] model={args.model_name} "
                f"sweep={sweep_name} {factor_name}={value} seed={seed}"
            )
            run_one(
                cfg,
                run_dir,
                force=args.force,
                resume_from=args.resume_from,
                resume_each=args.resume_each,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run stage-1 sweep for short sequence length."
    )
    parser.add_argument("--model_name", type=str, default="stage1_prior_sampling")
    parser.add_argument("--result_root", type=str, default="artifacts/v3/stage1_sweep")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--freq_relative_tol", type=float, default=0.02)
    parser.add_argument("--amp_relative_tol", type=float, default=0.05)
    parser.add_argument("--complex_coeff_relative_tol", type=float, default=0.05)
    parser.add_argument("--monitor", type=str, default="freq_rmse_hz_mean")
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--early_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional checkpoint path to resume every sweep run from.",
    )
    parser.add_argument(
        "--resume_each",
        action="store_true",
        help="Resume each sweep run from its own run_dir/checkpoints/latest.pt if present.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_sweep(args)
