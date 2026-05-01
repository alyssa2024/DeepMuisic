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
        "base_freq": 150.0,  # Theoretical maximum detectable frequency is about 975 Hz
        "fluctuation_delta": 0.001,
        "probes": [0, 28, 111.08, 166.15],  # Physical installation angles (deg)
        "n_revs": 20000,
        "window_revs": 8,
        "hop_revs": 2,
        "batch_size": 16,
    },
    "signal": {
        "freqs_hz": [167.0, 341.0, 635.0, 872.0],
        "amp_real_m": [0.0006, 0.0005403, -0.0003329, -0.0008910],
        "amp_imag_m": [0.0, 0.0008415, 0.0007274, 0.0001270],
        "snr_db": 20,
    },
    "model": {
        "hidden_dim": 128,
        "nhead": 8,
        "num_layers": 2,
        "dim_feedforward": 256,
        "hidden_dim_dense": 256,
        "use_standard_pe": False,
    },
    "training": {
        "epochs": 100,
        "lr": 1e-4,
        "max_grad_norm": 0.1,
    },
    "eval": {
        "val_ratio": 0.2,
        "eval_every": 5,
        "dense_factor": 4,
        "target_recon_btt_mse": 0.1,
        "split_seed": 42,
    },
    "loss": {
        "beta": 1e-5,
        "use_kl_w": True,
    },
    "checkpoint": {
        "dir": "checkpoints",
        "name": "latest.pt",
        "save_every": 20,
        "resume_from": None,  # Example: "checkpoints/latest.pt"
    },
    "logging": {
        "enable_tensorboard": True,
        "tensorboard_dir": "artifacts/tensorboard",
        "save_curves": True,
        "curve_dir": "artifacts/curves",
        "curve_every": 1,
    },
}
