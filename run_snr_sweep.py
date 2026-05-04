import copy
import os

from config import CONFIG
import main as train_main


BASE_CONFIG = copy.deepcopy(CONFIG)


SNR_VALUES = [-5, 0, 5, 10, 15, 20]
SEEDS = [0]


def snr_name(snr_db: int) -> str:
    if snr_db < 0:
        return f"snr_db_m{abs(snr_db)}"
    return f"snr_db_{snr_db}"


def reset_config():
    CONFIG.clear()
    CONFIG.update(copy.deepcopy(BASE_CONFIG))


def run():
    root = "artifacts/v3/complex_ls/exp1_snr"

    for snr_db in SNR_VALUES:
        for seed in SEEDS:
            reset_config()

            run_dir = os.path.join(root, snr_name(snr_db), f"seed_{seed}")

            CONFIG["seed"] = seed
            CONFIG["signal"]["snr_db"] = snr_db
            CONFIG["run_dir"] = run_dir

            CONFIG["model"]["use_amp_residual"] = False
            CONFIG["model"]["amp_residual_gamma"] = 0.0
            CONFIG["loss"]["residual_weight"] = 0.0

            CONFIG["training"]["early_stopping"] = {
                "enabled": True,
                "monitor": "freq_rmse_hz",
                "mode": "min",
                "patience": 3,
                "min_delta": 0.0,
            }

            CONFIG["checkpoint"]["dir"] = os.path.join(run_dir, "checkpoints")
            CONFIG["checkpoint"]["name"] = "latest.pt"
            CONFIG["checkpoint"]["save_every"] = 20
            CONFIG["checkpoint"]["resume_from"] = None
            CONFIG["logging"]["tensorboard_dir"] = os.path.join(run_dir, "tensorboard")
            CONFIG["logging"]["curve_dir"] = os.path.join(run_dir, "curves")

            os.makedirs(run_dir, exist_ok=True)

            print("=" * 80)
            print(f"Running SNR={snr_db} dB, seed={seed}, run_dir={run_dir}")
            print("=" * 80)

            train_main.main()


if __name__ == "__main__":
    run()
