import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_factor_dir(s):
    parts = s.split("_")
    raw = parts[-1]
    name = "_".join(parts[:-1])
    raw = raw.replace("m", "-").replace("p", ".")
    try:
        value = float(raw)
    except ValueError:
        value = raw
    return name, value


def collect_results(result_root, metric_source="last"):
    result_root = Path(result_root)
    rows = []

    for metrics_path in result_root.glob("**/metrics.json"):
        run_dir = metrics_path.parent

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        config_path = run_dir / "config.json"
        cfg = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        rel = metrics_path.relative_to(result_root).parts
        model_name = rel[0] if len(rel) > 0 else cfg.get("model_name", "unknown")
        experiment = rel[1] if len(rel) > 1 else "unknown"
        factor_dir = rel[2] if len(rel) > 2 else "unknown"
        seed_dir = rel[3] if len(rel) > 3 else "seed_unknown"

        factor_name, x_value = parse_factor_dir(factor_dir)

        row = {
            "model_name": cfg.get("model_name", model_name),
            "experiment": experiment,
            "factor_name": factor_name,
            "x_value": x_value,
            "seed_dir": seed_dir,
            "run_dir": str(run_dir),
        }
        if metric_source == "best" and isinstance(metrics.get("best_metrics"), dict):
            row.update(metrics["best_metrics"])
            # Keep summary metadata from the top-level payload for reference.
            for k in (
                "best_epoch_by_freq_rmse",
                "best_freq_rmse_hz",
                "last_freq_rmse_hz",
                "freq_rmse_degradation_ratio",
                "early_stopped",
                "early_stop_epoch",
                "early_stopping_patience",
                "early_stopping_monitor",
            ):
                if k in metrics:
                    row[k] = metrics[k]
        else:
            row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


PLOT_META = {
    "exp1_snr": ("SNR (dB)", "SNR robustness"),
    "exp2_num_cycles": ("Cycles per sequence", "Sequence length"),
    "exp3_num_probes": ("Number of probes", "Probe number"),
    "exp4_num_harmonics": ("Number of sinusoids K", "Number of sinusoids"),
    "exp5_relative_half_band": ("Relative half band", "Frequency search band"),
    "exp6_sequence_posterior_samples": ("Posterior samples per sequence", "Posterior sampling"),
    "exp7_amp_prior_band": ("Amplitude relative half band", "Amplitude prior band"),
}


def plot_metric(df, experiment, metric, ylabel, out_dir):
    sub = df[df["experiment"] == experiment].copy()
    if sub.empty:
        print(f"[WARN] No data for {experiment}")
        return
    if metric not in sub.columns:
        print(f"[WARN] Metric {metric} missing for {experiment}")
        return

    grouped = sub.groupby(["model_name", "x_value"])[metric].agg(["mean", "std"]).reset_index().sort_values("x_value")

    xlabel, title_prefix = PLOT_META.get(experiment, ("Sweep value", experiment))

    plt.figure(figsize=(7, 4.5))
    for model_name, g in grouped.groupby("model_name"):
        plt.errorbar(
            g["x_value"],
            g["mean"],
            yerr=g["std"].fillna(0.0),
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=model_name,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{experiment}_{metric}.png"
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[SAVE] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=str, default="/content/drive/MyDrive/deepmusic_results/v3")
    parser.add_argument("--metric_source", type=str, choices=["last", "best"], default="last")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    df = collect_results(result_root, metric_source=args.metric_source)

    if df.empty:
        print("[WARN] No metrics.json found.")
        return

    csv_path = result_root / f"all_results_{args.metric_source}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")

    plot_dir = result_root / "figures"

    experiments = sorted(df["experiment"].unique())
    for exp in experiments:
        plot_metric(df, experiment=exp, metric="freq_rmse_hz_mean", ylabel="Frequency RMSE (Hz)", out_dir=plot_dir)
        plot_metric(df, experiment=exp, metric="freq_success_rate_mean", ylabel="Frequency success rate", out_dir=plot_dir)
        plot_metric(df, experiment=exp, metric="amp_success_rate_mean", ylabel="Amplitude success rate", out_dir=plot_dir)
        plot_metric(df, experiment=exp, metric="joint_success_rate_mean", ylabel="Joint success rate", out_dir=plot_dir)


if __name__ == "__main__":
    main()
