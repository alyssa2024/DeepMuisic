"""
Centralized project configuration.

Edit values here, then run `main.py`.
"""

CONFIG = {
    "seed": 42,
    "data": {
        "dataset_type": "single_fixed_long_sequence",
        "input_dim": 6,
        "num_harmonics": 4,
        "num_probes": 4,
        "base_freq": 150.0,
        "fluctuation_delta": 0.001,
        "probes": [0, 28, 111.08, 166.15],
        "patch_num_cycles": 4,
        "num_train_patches": 256,
        "num_val_patches": 64,
        "num_test_patches": 64,
        "patch_hop_cycles": 4,
        "allow_patch_overlap": False,
        "loader_batch_size": 1,
        "normalization": "group_std",
    },
    "signal": {
        "single_instance_parameter_source": "fixed",
        "true_frequency_hz": [167.0, 341.0, 635.0, 872.0],
        "true_amp_real": [0.6, 0.5403, -0.3329, -0.8910],
        "true_amp_imag": [0.0, 0.8415, 0.7274, 0.1270],
        "amp_real_center_m": [0.6, 0.5403, -0.3329, -0.8910],
        "amp_imag_center_m": [0.0, 0.8415, 0.7274, 0.1270],
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
            "min_log_rho2": -8.0,
            "max_log_rho2": -2.0,
        },
        "loss_prior": {
            "type": "uniform",
        },
    },
    "model": {
        "inference_parameterization": "encoder_fusion",
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
        "reduction": "sum",
        "reconstruction": {
            "type": "complex_gaussian_nll",
            "include_log_const": True,
            "use_posterior_sampling": True,
            "sequence_posterior_samples": 4,
            "sample_at_train": True,
            "eval_at_mean": True,
        },
        "kl": {
            "enabled": True,
            "type": "trunc_normal_to_uniform",
            "warmup_steps": 0,
            "reuse_reconstruction_samples": False,
        },
        "amplitude_prior": {
            "type": "complex_isotropic_gaussian",
            "tau2_norm": 1.0,
            "include_map_prior_penalty": False,
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
            "warmup_steps": 10,
            "min_lr": 1e-6,
        },
        "grad_clip": {
            "enabled": False,
            "max_norm": 1.0,
        },
        "early_stopping": {
            "enabled": True,
            "monitor": "val_recon_btt_mse",
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
