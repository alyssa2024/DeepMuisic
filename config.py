"""
Centralized project configuration.

Edit values here, then run `main.py`.
"""

CONFIG = {
    "seed": 42,
    "data": {
        "input_dim": 6,
        "num_harmonics": 4,
        "num_probes": 4,
        "base_freq": 150.0,
        "fluctuation_delta": 0.001,
        "probes": [0, 28, 111.08, 166.15],
        "num_cycles": 4,
        "num_train_sequences": 10000,
        "num_val_sequences": 2000,
        "num_test_sequences": 2000,
        "batch_size": 16,
        "normalization": "per_sequence_std",
    },
    "signal": {
        "amp_real_center_m": [0.0006, 0.0005403, -0.0003329, -0.0008910],
        "amp_imag_center_m": [0.0, 0.0008415, 0.0007274, 0.0001270],
        "amp_data_prior": {
            "type": "independent_uniform",
            "relative_half_band": 0.2,
            "min_half_band_m": 1e-5,
        },
        "snr_db": 20,
    },
    "frequency": {
        "center_hz": [167.0, 341.0, 635.0, 872.0],
        "relative_half_band": 0.05,
        "data_prior": {
            "type": "uniform",
        },
        "model_search": {
            "type": "relative_band",
        },
        "posterior": {
            "type": "truncated_normal",
            "scale_parameterization": "sigmoid_bound",
            "min_log_rho2": -12.0,
            "max_log_rho2": -7.0,
        },
        "loss_prior": {
            "type": "truncated_normal",
            "mean": "center",
            "std_ratio_to_half_band": 0.5,
        },
    },
    "model": {
        "hidden_dim": 128,
        "nhead": 8,
        "num_layers": 2,
        "dim_feedforward": 256,
        "hidden_dim_dense": 256,
        "use_standard_pe": False,
        "ls_ridge": 1e-5,
    },
    "loss": {
        "beta_freq": 1.0,
        "reconstruction": {
            "type": "complex_gaussian_nll",
            "include_log_const": False,
            "use_posterior_sampling": True,
            "sequence_posterior_samples": 2,
            "sample_at_train": True,
            "eval_at_mean": True,
        },
        "kl": {
            "enabled": True,
            "type": "trunc_normal_to_trunc_normal",
            "warmup_steps": 10000,
            "reuse_reconstruction_samples": False,
        },
        "success": {
            "freq_relative_tol": 0.02,
            "amp_relative_tol": 0.05,
            "complex_coeff_relative_tol": 0.05,
        },
    },
    "training": {
        "epochs": 150,
        "lr": 1e-4,
        "lr_schedule": {
            "type": "warmup_cosine",
            "warmup_steps": 1000,
            "min_lr": 1e-6,
        },
        "grad_clip": {
            "enabled": False,
            "max_norm": 1.0,
        },
        "early_stopping": {
            "enabled": True,
            "monitor": "recon_mse_mean",
            "mode": "min",
            "patience": 3,
            "min_delta": 1e-6,
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
