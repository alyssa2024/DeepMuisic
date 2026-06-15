"""
Centralized project configuration.

Edit values here, then run `main.py`.
"""

CONFIG = {
    "seed": 42,
    "frequency_model": {
        "type": "static_global",
        "interpretation": "first_order_approximation",
    },
    "data": {
        "input_dim": 7,
        "num_harmonics": 4,
        "num_probes": 4,
        "base_freq": 150.0,
        "fluctuation_delta": 0.001,
        "probes": [0, 28, 111.08, 166.15],
        "train_dataset": {
            "num_param_sets": 1000,
            "sequences_per_param": 5,
            "long_sequence_num_cycles": 32,
            "sequence_num_cycles": 8,
        },
        "val_dataset": {
            "num_param_sets": 200,
            "sequences_per_param": 5,
            "long_sequence_num_cycles": 32,
            "sequence_num_cycles": 8,
        },
        "test_dataset": {
            "num_param_sets": 200,
            "sequences_per_param": 5,
            "long_sequence_num_cycles": 32,
            "sequence_num_cycles": 8,
        },
        "batch_size": 8,
        "normalization": "per_sequence_std",
    },
    "signal": {
        "amp_real_center_m": [0.0006, 0.0005403, -0.0003329, -0.0008910],
        "amp_imag_center_m": [0.0004, 0.0008415, 0.0007274, 0.0001270],
        "rho_amp_real_k": [0.0, 0.0, 0.0, 0.0],
        "rho_amp_imag_k": [0.0, 0.0, 0.0, 0.0],
        "amp_data_prior": {
            # Single-carrier phase model: amplitude is a real non-negative
            # magnitude (uniform in [0, 1]) and the phase is uniform in
            # [0, 2*pi). The decoder phase is carried entirely by z.
            "type": "uniform_polar",
            "magnitude_min": 0.0,
            "magnitude_max": 1.0,
            "phase_min": 0.0,
            "phase_max": 6.283185307179586,  # 2*pi
        },
        "snr_db": 20,
    },
    "frequency": {
        "center_hz": [217.0, 341.0, 635.0, 872.0],
        "relative_half_band": 0.05,
        "posterior": {
            "min_log_rho2": -12.0,
            "max_log_rho2": -4.0,
        },
        "loss_prior": {
            "mean": "center",
            "std_ratio_to_half_band": 0.5,
        },
    },
    "model": {
        "variant": "sequential_dsae",
        "local_hidden_dim": 128,
        "local_out_dim": 128,
        "context_hidden_dim": 256,
        "context_layers": 1,
        "innovation_hidden_dim": 256,
        "dropout": 0.0,
        "endpoint_summary": "forward_last_backward_first",
        "c_init_logvar": -4.0,
        "z1_init_logvar": 0.0,
        "delta_init_logvar": -6.0,
    },
    "loss": {
        "beta_freq": 1.0,
        "beta_amp": 1.0,
        "beta_z1": 1.0,
        "beta_delta": 1.0,
        "logamp_prior_mean": 0.0,
        "logamp_prior_var": 1.0,
        "reconstruction_weight": 1.0,
        "reconstruction": {
            "include_log_const": True,
            "normalize_by_num_points": False,
        },
        "sequential": {
            "z1_prior_var": 1.0,
            "delta_prior_var": 0.01,
        },
        "kl": {
            "enabled": True,
            "type": "trunc_normal_to_trunc_normal",
            "warmup_steps": 1000,
        },
    },
    "training": {
        "epochs": 200,
        "lr": 2e-5,
        "freeze_encoder_train_amp_head_only": False,
        "amp_head_train_mode": "bias_only",
        "amp_head_lr": 1e-2,
        "objective_curriculum": {
            "enabled": False,
            "cycles": [32],
            "epochs_per_stage": [200],
            "lr_per_stage": [2e-5],
            "segment_mode": "prefix",
            "apply_to_encoder": True,
        },
        "lr_schedule": {
            "type": "constant",
        },
        "early_stopping": {
            "enabled": False,
            "monitor": "loss",
            "mode": "min",
            "patience": 10,
            "min_delta": 1e-4,
        },
    },
    "eval": {
        "eval_every": 5,
        "dense_factor": 4,
        "target_recon_mse": 0.1,
    },
    "checkpoint": {
        "dir": "checkpoints",
        "name": "latest.pt",
        "save_every": 20,
        "resume_from": None,
    },
    "logging": {
        "enable_tensorboard": True,
        "tensorboard_dir": "artifacts/tensorboard",
        "save_curves": True,
        "curve_dir": "artifacts/curves",
        "curve_every": 1,
    },
    "experiment": {
        "snr_values": [-5, 0, 5, 10, 15, 20],
        "seeds": [0],
    },
}
