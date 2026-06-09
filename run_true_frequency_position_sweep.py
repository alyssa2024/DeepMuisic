import argparse
import copy
import csv
import json
import os

from config import CONFIG
import main as train_main
from synthesis_dataset import compute_frequency_support


BASE_CONFIG = copy.deepcopy(CONFIG)
DEFAULT_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def reset_config():
    CONFIG.clear()
    CONFIG.update(copy.deepcopy(BASE_CONFIG))


def position_name(position):
    return f"true_freq_pos_{str(float(position)).replace('.', 'p')}"


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def true_frequency_at_position(freq_lower, freq_upper, position):
    if position < 0.0 or position > 1.0:
        raise ValueError("true frequency position must be in [0, 1]")
    return [
        float(lower + position * (upper - lower))
        for lower, upper in zip(freq_lower, freq_upper)
    ]


def frequency_support_from_config():
    freq_cfg = CONFIG["frequency"]
    absolute_half_band = freq_cfg.get("absolute_half_band_hz")
    if absolute_half_band is not None:
        return compute_frequency_support(
            freq_center_hz=freq_cfg["center_hz"],
            absolute_half_band_hz=absolute_half_band,
        )
    return compute_frequency_support(
        freq_center_hz=freq_cfg["center_hz"],
        relative_half_band=freq_cfg["relative_half_band"],
    )


def configure_run(run_dir, seed, position):
    CONFIG["seed"] = int(seed)
    CONFIG["run_dir"] = run_dir
    CONFIG["sweep_name"] = "true_frequency_position"
    CONFIG["sweep_value"] = float(position)
    CONFIG["signal"]["single_instance_parameter_source"] = "fixed"

    freq_lower, freq_upper, _freq_center, _freq_half = frequency_support_from_config()
    true_freq_hz = true_frequency_at_position(freq_lower, freq_upper, position)
    CONFIG["signal"]["true_frequency_hz"] = true_freq_hz

    CONFIG["checkpoint"]["dir"] = os.path.join(run_dir, "checkpoints")
    CONFIG["checkpoint"]["name"] = "latest.pt"
    CONFIG["checkpoint"]["resume_from"] = None
    CONFIG["logging"]["tensorboard_dir"] = os.path.join(run_dir, "tensorboard")
    CONFIG["logging"]["curve_dir"] = os.path.join(run_dir, "curves")
    return true_freq_hz


def read_metrics(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Expected metrics.json not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("last_metrics", payload)


def summarize_run(position, seed, run_dir, true_freq_hz, metrics):
    joint_success = metrics.get("joint_amp_freq_success_rate_mean", "")
    row = {
        "true_frequency_position": float(position),
        "seed": int(seed),
        "run_dir": run_dir,
        "freq_rmse_hz_mean": metrics.get("freq_rmse_hz_mean", ""),
        "freq_nrmse_band_mean": metrics.get("freq_nrmse_band_mean", ""),
        "val_recon_btt_mse": metrics.get("val_recon_btt_mse", ""),
        "test_recon_btt_mse": metrics.get("test_recon_btt_mse", ""),
        "joint_amp_freq_success_rate_mean": metrics.get(
            "joint_amp_freq_success_rate_mean",
            "",
        ),
        "failure": (
            "" if joint_success == "" else int(float(joint_success) < 1.0)
        ),
        "global_freq_coverage_68": metrics.get("global_freq_coverage_68", ""),
        "global_freq_coverage_95": metrics.get("global_freq_coverage_95", ""),
        "posterior_std_hz_mean": metrics.get("posterior_std_hz_mean", ""),
        "fusion_effective_num_patches_mean": metrics.get(
            "fusion_effective_num_patches_mean",
            "",
        ),
    }
    for idx, value in enumerate(true_freq_hz, start=1):
        row[f"true_freq_h{idx}_hz"] = float(value)
        row[f"freq_rmse_h{idx}_hz"] = metrics.get(f"freq_rmse_h{idx}_hz", "")
        row[f"val_amp_mape_h{idx}"] = metrics.get(f"val_amp_mape_h{idx}", "")
    return row


def _numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        values.append(float(value))
    return values


def _median(values):
    if not values:
        return ""
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def aggregate_by_position(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["true_frequency_position"], []).append(row)

    aggregate_rows = []
    for position in sorted(grouped):
        group = grouped[position]
        rmse_values = _numeric_values(group, "freq_rmse_hz_mean")
        failures = _numeric_values(group, "failure")
        coverage68 = _numeric_values(group, "global_freq_coverage_68")
        coverage95 = _numeric_values(group, "global_freq_coverage_95")

        best = None
        worst = None
        valid_rmse_rows = [
            row for row in group if row.get("freq_rmse_hz_mean", "") != ""
        ]
        if valid_rmse_rows:
            best = min(valid_rmse_rows, key=lambda row: float(row["freq_rmse_hz_mean"]))
            worst = max(valid_rmse_rows, key=lambda row: float(row["freq_rmse_hz_mean"]))

        aggregate_rows.append(
            {
                "true_frequency_position": position,
                "num_seeds": len(group),
                "median_freq_rmse_hz_mean": _median(rmse_values),
                "failure_rate": (
                    "" if not failures else sum(failures) / float(len(failures))
                ),
                "best_seed": "" if best is None else int(best["seed"]),
                "best_freq_rmse_hz_mean": (
                    "" if best is None else best["freq_rmse_hz_mean"]
                ),
                "worst_seed": "" if worst is None else int(worst["seed"]),
                "worst_freq_rmse_hz_mean": (
                    "" if worst is None else worst["freq_rmse_hz_mean"]
                ),
                "mean_coverage_68": (
                    "" if not coverage68 else sum(coverage68) / float(len(coverage68))
                ),
                "mean_coverage_95": (
                    "" if not coverage95 else sum(coverage95) / float(len(coverage95))
                ),
                "median_coverage_95": _median(coverage95),
            }
        )
    return aggregate_rows


def _write_rows_csv(rows, csv_path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(rows, root):
    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, "true_frequency_position_summary.json")
    csv_path = os.path.join(root, "true_frequency_position_summary.csv")
    aggregate_json_path = os.path.join(
        root,
        "true_frequency_position_aggregate_summary.json",
    )
    aggregate_csv_path = os.path.join(
        root,
        "true_frequency_position_aggregate_summary.csv",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _write_rows_csv(rows, csv_path)

    aggregate_rows = aggregate_by_position(rows)
    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_rows, f, indent=2)
    _write_rows_csv(aggregate_rows, aggregate_csv_path)

    print(f"Saved summary JSON: {json_path}")
    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved aggregate summary JSON: {aggregate_json_path}")
    print(f"Saved aggregate summary CSV: {aggregate_csv_path}")


def run(positions, seeds, root):
    rows = []
    for position in positions:
        for seed in seeds:
            reset_config()
            run_dir = os.path.join(root, position_name(position), f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            true_freq_hz = configure_run(
                run_dir=run_dir,
                seed=seed,
                position=position,
            )
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(CONFIG, f, indent=2)

            print("=" * 80)
            print(
                "Running true-frequency position sweep, "
                f"position={position:g}, seed={seed}, "
                f"true_freq_hz={true_freq_hz}, run_dir={run_dir}"
            )
            print("=" * 80)

            train_main.main()
            metrics = read_metrics(run_dir)
            row = summarize_run(
                position=position,
                seed=seed,
                run_dir=run_dir,
                true_freq_hz=true_freq_hz,
                metrics=metrics,
            )
            rows.append(row)
            write_summary(rows, root)

    write_summary(rows, root)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep fixed true frequency positions inside each search band. "
            "For position r, f*_k = l_k + r * (u_k - l_k)."
        )
    )
    parser.add_argument(
        "--positions",
        default=",".join(str(x) for x in DEFAULT_POSITIONS),
        help=(
            "Comma-separated true-frequency positions in [0,1]. "
            "Default: 0.1,0.3,0.5,0.7,0.9"
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seeds. Default: 0,1,2,3,4",
    )
    parser.add_argument(
        "--root",
        default="artifacts/profile_LS_ELBO/true_frequency_position_sweep",
        help="Root directory for sweep artifacts.",
    )
    args = parser.parse_args()
    run(
        positions=parse_float_list(args.positions),
        seeds=parse_int_list(args.seeds),
        root=args.root,
    )


if __name__ == "__main__":
    main()
