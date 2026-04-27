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
        "n_revs": 200,
        "window_revs": 8,
        "hop_revs": 2,
        "batch_size": 8,
    },
    "signal": {
        "freqs_hz": [167.0, 341.0, 635.0, 872.0],
        "amplitudes_m": [0.0006, 0.0010, 0.0008, 0.0009],
        "phases_rad": [0.0, 1.0, 2.0, 3.0],
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
        "epochs": 500,
        "lr": 1e-4,
        "max_grad_norm": 0.1,
    },
    "loss": {
        "beta": 1e-5,
        "prior_A_mu": 0.0,
        "prior_A_var": 1.0,
        "prior_phi_mu": 0.0,
        "prior_phi_kappa": 5.0,
        "use_kl_A": True,
        "use_kl_w": True,
        "use_kl_phi": True,
    },
    "checkpoint": {
        "dir": "checkpoints",
        "name": "latest.pt",
        "save_every": 10,
        "resume_from": None,  # Example: "checkpoints/latest.pt"
    },
}
