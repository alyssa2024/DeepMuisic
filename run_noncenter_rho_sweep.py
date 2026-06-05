import copy
import csv
import json
import os

from config import CONFIG
import main as train_main


BASE_CONFIG = copy.deepcopy(CONFIG)
RHO_VALUES = [0.0, 0.05, -0.05, 0.1, -0.1]


def rho_value_name(rho):
    text = f"{float(rho):.6g}".replace("-", "m").replace(".", "p")
    return f"rho_{text}"


def rho_vector_for_value(rho, num_harmonics):
    return [float(rho)] * int(num_harmonics)


def reset_config():
    CONFIG.clear()
    CONFIG.update(copy.deepcopy(BASE_CONFIG))


def read_metrics(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Expected metrics.json not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_summary(rows, root):
    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, "noncenter_rho_sweep_summary.json")
    csv_path = os.path.join(root, "noncenter_rho_sweep_summary.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

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

    print(f"Saved summary JSON: {json_path}")
    print(f"Saved summary CSV: {csv_path}")


def summarize_run(rho, rho_k, run_dir, metrics):
    return {
        "rho": float(rho),
        "rho_k": json.dumps(rho_k),
        "seed": metrics.get("seed", CONFIG.get("seed", "")),
        "final_eval_checkpoint": metrics.get("final_eval_checkpoint", ""),
        "early_stopped": metrics.get("early_stopped", ""),
        "early_stop_epoch": metrics.get("early_stop_epoch", ""),
        "best_epoch_by_monitor": metrics.get("best_epoch_by_monitor", ""),
        "recon_mse_mean": metrics.get("recon_mse_mean", ""),
        "freq_rmse_hz_mean": metrics.get("freq_rmse_hz_mean", ""),
        "center_baseline_freq_rmse_hz": metrics.get(
            "center_baseline_freq_rmse_hz",
            "",
        ),
        "amp_mape_mean": metrics.get("amp_mape_mean", ""),
        "joint_amp_freq_success_rate_mean": metrics.get(
            "joint_amp_freq_success_rate_mean",
            "",
        ),
        "poe_invalid_precision_rate": metrics.get("poe_invalid_precision_rate", ""),
        "poe_precision_raw_min": metrics.get("poe_precision_raw_min", ""),
        "poe_precision_raw_mean": metrics.get("poe_precision_raw_mean", ""),
        "poe_mu_before_clamp_mean_abs_err_to_prior": metrics.get(
            "poe_mu_before_clamp_mean_abs_err_to_prior",
            "",
        ),
        "poe_mu_after_clamp_mean_abs_err_to_prior": metrics.get(
            "poe_mu_after_clamp_mean_abs_err_to_prior",
            "",
        ),
        "poe_window_std_mean": metrics.get("poe_window_std_mean", ""),
        "poe_global_std_mean": metrics.get("poe_global_std_mean", ""),
        "run_dir": run_dir,
    }


def run():
    root = os.path.join("artifacts", "v5", "noncenter_rho_sweep")
    rows = []

    for rho in RHO_VALUES:
        reset_config()

        num_harmonics = CONFIG["data"]["num_harmonics"]
        rho_k = rho_vector_for_value(rho, num_harmonics)
        case_name = rho_value_name(rho)
        run_dir = os.path.join(root, case_name)

        CONFIG["frequency"]["rho_k"] = rho_k
        CONFIG["run_dir"] = run_dir
        CONFIG["sweep_name"] = "noncenter_rho"
        CONFIG["sweep_value"] = float(rho)

        CONFIG["checkpoint"]["dir"] = os.path.join(run_dir, "checkpoints")
        CONFIG["checkpoint"]["name"] = "latest.pt"
        CONFIG["checkpoint"]["resume_from"] = None
        CONFIG["logging"]["tensorboard_dir"] = os.path.join(run_dir, "tensorboard")
        CONFIG["logging"]["curve_dir"] = os.path.join(run_dir, "curves")

        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2)

        print("=" * 80)
        print(
            "Running non-center rho sweep, "
            f"case={case_name}, rho={rho:g}, rho_k={rho_k}, "
            f"curriculum_enabled="
            f"{CONFIG['training']['objective_curriculum']['enabled']}, "
            f"run_dir={run_dir}"
        )
        print("=" * 80)

        train_main.CONFIG = CONFIG
        train_main.main()

        metrics = read_metrics(run_dir)
        row = summarize_run(
            rho=rho,
            rho_k=rho_k,
            run_dir=run_dir,
            metrics=metrics,
        )
        rows.append(row)
        write_summary(rows, root)

        print(
            "[SUMMARY] "
            f"case={case_name} "
            f"recon_mse={row['recon_mse_mean']} "
            f"freq_rmse_hz={row['freq_rmse_hz_mean']} "
            f"amp_mape={row['amp_mape_mean']} "
            f"joint_success={row['joint_amp_freq_success_rate_mean']}"
        )

    write_summary(rows, root)


if __name__ == "__main__":
    run()
