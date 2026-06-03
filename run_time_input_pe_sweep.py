import copy
import json
import os

from config import CONFIG
import main as train_main


BASE_CONFIG = copy.deepcopy(CONFIG)


VARIANTS = [
    {
        "name": "no_time_no_pe",
        "input_dim": 6,
        "include_local_time_norm": False,
        "use_standard_pe": False,
        "use_time_pe": False,
    },
    {
        "name": "time_no_pe",
        "input_dim": 7,
        "include_local_time_norm": True,
        "use_standard_pe": False,
        "use_time_pe": False,
    },
    {
        "name": "time_time_pe",
        "input_dim": 7,
        "include_local_time_norm": True,
        "use_standard_pe": False,
        "use_time_pe": True,
    },
]

SEEDS = [0]


def reset_config():
    CONFIG.clear()
    CONFIG.update(copy.deepcopy(BASE_CONFIG))


def apply_variant(variant):
    CONFIG["data"]["input_dim"] = int(variant["input_dim"])
    CONFIG["data"]["include_local_time_norm"] = bool(
        variant["include_local_time_norm"]
    )

    CONFIG["model"]["use_standard_pe"] = bool(variant["use_standard_pe"])
    CONFIG["model"]["use_time_pe"] = bool(variant["use_time_pe"])
    CONFIG["model"].setdefault("time_feature_index", -1)
    CONFIG["model"].setdefault("time_pe_num_bands", 64)
    CONFIG["model"].setdefault("time_pe_trainable_proj", True)


def run():
    root = "artifacts/v4/time_input_pe_sweep"

    for variant in VARIANTS:
        for seed in SEEDS:
            reset_config()
            apply_variant(variant)

            run_dir = os.path.join(root, variant["name"], f"seed_{seed}")

            CONFIG["seed"] = seed
            CONFIG["run_dir"] = run_dir
            CONFIG["sweep_name"] = "time_input_pe"
            CONFIG["sweep_value"] = variant["name"]

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
                "Running "
                f"variant={variant['name']}, seed={seed}, "
                f"input_dim={CONFIG['data']['input_dim']}, "
                f"include_local_time_norm={CONFIG['data']['include_local_time_norm']}, "
                f"use_standard_pe={CONFIG['model']['use_standard_pe']}, "
                f"use_time_pe={CONFIG['model']['use_time_pe']}, "
                f"run_dir={run_dir}"
            )
            print("=" * 80)

            train_main.main()


if __name__ == "__main__":
    run()
