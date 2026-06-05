import copy
import csv
import json
import os

from config import CONFIG
import main as train_main


BASE_CONFIG = copy.deepcopy(CONFIG)


RHO_VALUES = [0.05, -0.05, 0.1, -0.1]
SEEDS = [0]


PASS_THRESHOLDS = {
    "freq_rmse_hz_mean": 0.01,
    "recon_mse_mean": 0.02,
    "amp_mape_mean": 0.01,
    "joint_amp_freq_success_rate_mean": 1.0,
}


def rho_value_name(rho):
    text = f"{float(rho):.6g}".replace("-", "m").replace(".", "p")
    return f"rho_{text}"


def rho_vector_for_value(rho, num_harmonics):
    return [float(rho)] * int(num_harmonics)


def reset_config():
    CONFIG.clear()
    CONFIG.update(copy.deepcopy(BASE_CONFIG))


def apply_v5_deterministic_curriculum_sanity():
    CONFIG["loss"]["beta_freq"] = 1.0
    CONFIG["loss"]["reconstruction"].update(
        {
            "include_log_const": True,
            "use_posterior_sampling": True,
            "sequence_posterior_samples": 2,
            "sample_at_train": True,
            "eval_at_mean": True,
            "normalize_by_num_points": True,
        }
    )
    CONFIG["loss"]["kl"].update(
        {
            "enabled": True,
            "type": "trunc_normal_to_trunc_normal",
            "warmup_steps": 0,
            "reuse_reconstruction_samples": False,
        }
    )
    CONFIG["loss"]["amplitude_kl"] = {
        "enabled": True,
        "beta_amp": 1.0,
    }
    CONFIG["loss"]["elbo"].update(
        {
            "mode": "static_global_bayesian_nnamp",
            "state_aware": True,
            "strict_long_sequence_elbo": True,
            "global_time_origin": "parent",
        }
    )
    CONFIG["amplitude_nn"] = {
        "enabled": True,
        "type": "complex_gaussian",
        "output_domain": "normalized",
        "activation": "tanh",
        "amp_scale_norm": 1.0,
        "init_scale": 0.05,
        "min_logvar": -12.0,
        "max_logvar": 0.0,
        "init_logvar": -4.0,
    }

    CONFIG["training"]["epochs"] = 190
    CONFIG["training"]["objective_curriculum"] = {
        "enabled": True,
        "cycles": [4, 8, 16, 32, 64, 256, 1000, 10000],
        "epochs_per_stage": [20, 20, 20, 20, 20, 20, 30, 40],
        "lr_per_stage": [1e-4, 1e-4, 1e-4, 5e-5, 3e-5, 1e-5, 3e-6, 1e-6],
        "segment_mode": "prefix",
        "apply_to_encoder": True,
    }
    CONFIG["training"]["early_stopping"] = {
        "enabled": True,
        "monitor": "recon_mse_mean",
        "mode": "min",
        "patience": 5,
        "min_delta": 1e-5,
    }


def apply_fixed_parent_dataset():
    train_dataset = CONFIG["data"].setdefault("train_dataset", {})
    train_dataset.update(
        {
            "num_param_sets": 1,
            "sequences_per_param": 1,
            "use_long_sequence": True,
            "long_sequence_num_cycles": 10000,
            "sequence_num_cycles": 4,
            "window_hop_cycles": 4,
            "chronological_split": True,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "return_global_parent": True,
        }
    )
    test_dataset = CONFIG["data"].setdefault("test_dataset", {})
    test_dataset.update(
        {
            "num_param_sets": 1,
            "sequences_per_param": 1,
            "use_long_sequence": True,
            "long_sequence_num_cycles": 10000,
            "sequence_num_cycles": 4,
            "window_hop_cycles": 4,
            "chronological_split": True,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "return_global_parent": True,
        }
    )
    CONFIG["data"]["batch_size"] = 1


def is_pass(metrics):
    return (
        float(metrics.get("freq_rmse_hz_mean", float("inf")))
        < PASS_THRESHOLDS["freq_rmse_hz_mean"]
        and float(metrics.get("recon_mse_mean", float("inf")))
        <= PASS_THRESHOLDS["recon_mse_mean"]
        and float(metrics.get("amp_mape_mean", float("inf")))
        < PASS_THRESHOLDS["amp_mape_mean"]
        and float(metrics.get("joint_amp_freq_success_rate_mean", 0.0))
        >= PASS_THRESHOLDS["joint_amp_freq_success_rate_mean"]
    )


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


def run():
    root = "artifacts/v5/noncenter_rho_sweep"
    rows = []

    for rho in RHO_VALUES:
        for seed in SEEDS:
            reset_config()
            apply_v5_deterministic_curriculum_sanity()
            apply_fixed_parent_dataset()

            num_harmonics = int(CONFIG["data"]["num_harmonics"])
            rho_k = rho_vector_for_value(rho, num_harmonics)
            case_name = rho_value_name(rho)
            run_dir = os.path.join(root, case_name, f"seed_{seed}")

            CONFIG["seed"] = int(seed)
            CONFIG["frequency"]["rho_k"] = rho_k
            CONFIG["run_dir"] = run_dir
            CONFIG["sweep_name"] = "noncenter_rho"
            CONFIG["sweep_value"] = float(rho)

            CONFIG["checkpoint"]["dir"] = os.path.join(run_dir, "checkpoints")
            CONFIG["checkpoint"]["name"] = "latest.pt"
            CONFIG["checkpoint"]["save_every"] = 20
            CONFIG["checkpoint"]["resume_from"] = None
            CONFIG["logging"]["tensorboard_dir"] = os.path.join(run_dir, "tensorboard")
            CONFIG["logging"]["curve_dir"] = os.path.join(run_dir, "curves")

            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(CONFIG, f, indent=2)

            print("=" * 80)
            print(
                "Running non-center rho sanity check, "
                f"case={case_name}, seed={seed}, rho={rho:g}, rho_k={rho_k}, "
                f"beta_freq={CONFIG['loss']['beta_freq']}, "
                f"kl_enabled={CONFIG['loss']['kl']['enabled']}, "
                f"use_posterior_sampling="
                f"{CONFIG['loss']['reconstruction']['use_posterior_sampling']}, "
                f"normalize_by_num_points="
                f"{CONFIG['loss']['reconstruction']['normalize_by_num_points']}, "
                f"run_dir={run_dir}"
            )
            print("=" * 80)

            train_main.main()

            metrics = read_metrics(run_dir)
            passed = is_pass(metrics)
            row = {
                "case": case_name,
                "seed": seed,
                "rho": float(rho),
                "rho_k": json.dumps(rho_k),
                "passed": passed,
                "final_eval_checkpoint": metrics.get("final_eval_checkpoint", ""),
                "best_checkpoint_path": metrics.get("best_checkpoint_path", ""),
                "final_best_checkpoint_path": metrics.get(
                    "final_best_checkpoint_path",
                    "",
                ),
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
                "amp_success_rate_mean": metrics.get("amp_success_rate_mean", ""),
                "joint_amp_freq_success_rate_mean": metrics.get(
                    "joint_amp_freq_success_rate_mean",
                    "",
                ),
                "run_dir": run_dir,
            }
            rows.append(row)
            write_summary(rows, root)

            print(
                "[SUMMARY] "
                f"case={case_name} seed={seed} passed={passed} "
                f"final_eval_checkpoint={row['final_eval_checkpoint']} "
                f"recon_mse={row['recon_mse_mean']} "
                f"freq_rmse_hz={row['freq_rmse_hz_mean']} "
                f"amp_mape={row['amp_mape_mean']} "
                f"joint_success={row['joint_amp_freq_success_rate_mean']}"
            )

    write_summary(rows, root)


if __name__ == "__main__":
    run()
