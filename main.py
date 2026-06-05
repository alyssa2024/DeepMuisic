import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from batch_utils import extract_dataset_state
from dataset import BTTSequenceDataset, GroupedBTTSequenceDataset
from Encoder import VariationalIndependentTimeSeriesTransformer
from eval import evaluate_model
from loss import compute_static_global_objective
from synthesis_dataset import compute_frequency_support
from VAE import PhysicalHarmonicVAE

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    total_steps,
    nonfinite_steps,
    grad_clip_triggered_steps,
    epoch_to_target,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "total_steps": total_steps,
            "nonfinite_steps": nonfinite_steps,
            "grad_clip_triggered_steps": grad_clip_triggered_steps,
            "epoch_to_target": epoch_to_target,
        },
        path,
    )


def _log_scalar(writer, tag, value, step):
    if writer is not None:
        writer.add_scalar(tag, float(value), int(step))


def _append_history(history, key, step, value):
    history.setdefault(key, []).append((int(step), float(value)))


def _param_norm(params):
    vals = [p.detach().flatten() for p in params]
    if not vals:
        return 0.0
    return torch.linalg.norm(torch.cat(vals)).item()


def _grad_norm(params):
    vals = [
        p.grad.detach().flatten()
        for p in params
        if p.grad is not None
    ]
    if not vals:
        return 0.0
    return torch.linalg.norm(torch.cat(vals)).item()


def _resolve_lr_schedule(train_cfg, steps_per_epoch):
    schedule_cfg = train_cfg.get("lr_schedule", {})
    schedule_type = schedule_cfg.get("type", "warmup_cosine")
    total_steps = int(schedule_cfg.get("total_steps", train_cfg["epochs"] * steps_per_epoch))
    if "warmup_steps" in schedule_cfg:
        warmup_steps = int(schedule_cfg["warmup_steps"])
    else:
        warmup_ratio = float(schedule_cfg.get("warmup_ratio", 0.0))
        if warmup_ratio < 0 or warmup_ratio >= 1:
            raise ValueError("training.lr_schedule.warmup_ratio must be in [0, 1)")
        warmup_steps = int(total_steps * warmup_ratio)
    min_lr = float(schedule_cfg.get("min_lr", 0.0))

    if schedule_type not in ("constant", "warmup_cosine"):
        raise ValueError(f"Unsupported training.lr_schedule.type={schedule_type}")
    if total_steps <= 0:
        raise ValueError("training.lr_schedule.total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("training.lr_schedule.warmup_steps must be non-negative")
    if warmup_steps >= total_steps:
        warmup_steps = max(total_steps - 1, 0)
    if min_lr < 0:
        raise ValueError("training.lr_schedule.min_lr must be non-negative")
    return schedule_type, total_steps, warmup_steps, min_lr


def _compute_learning_rate(base_lr, step, schedule_type, total_steps, warmup_steps, min_lr):
    if schedule_type == "constant":
        return float(base_lr)

    step = min(max(int(step), 1), int(total_steps))
    base_lr = float(base_lr)
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    cosine_steps = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / cosine_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_decay


def _resolve_objective_curriculum(train_cfg):
    cfg = train_cfg.get("objective_curriculum", {})
    if not bool(cfg.get("enabled", False)):
        return {
            "enabled": False,
            "cycles": None,
            "epochs_per_stage": None,
            "segment_mode": "prefix",
            "apply_to_encoder": False,
        }

    cycles = [int(x) for x in cfg.get("cycles", [])]
    epochs_per_stage = [int(x) for x in cfg.get("epochs_per_stage", [])]
    lr_per_stage_raw = cfg.get("lr_per_stage", None)
    lr_per_stage = (
        None
        if lr_per_stage_raw is None
        else [float(x) for x in lr_per_stage_raw]
    )

    if not cycles:
        raise ValueError("training.objective_curriculum.cycles must be non-empty")
    if len(epochs_per_stage) != len(cycles):
        raise ValueError(
            "training.objective_curriculum.epochs_per_stage must have same length as cycles"
        )
    if lr_per_stage is not None and len(lr_per_stage) != len(cycles):
        raise ValueError(
            "training.objective_curriculum.lr_per_stage must have same length as cycles"
        )
    if any(c <= 0 for c in cycles):
        raise ValueError("objective curriculum cycles must be positive")
    if any(e <= 0 for e in epochs_per_stage):
        raise ValueError("objective curriculum epochs_per_stage must be positive")
    if lr_per_stage is not None and any(lr <= 0.0 for lr in lr_per_stage):
        raise ValueError("objective curriculum lr_per_stage values must be positive")

    segment_mode = cfg.get("segment_mode", "prefix")
    if segment_mode != "prefix":
        raise ValueError(
            "Only training.objective_curriculum.segment_mode='prefix' is supported, "
            f"got {segment_mode!r}"
        )

    return {
        "enabled": True,
        "cycles": cycles,
        "epochs_per_stage": epochs_per_stage,
        "lr_per_stage": lr_per_stage,
        "segment_mode": segment_mode,
        "apply_to_encoder": bool(cfg.get("apply_to_encoder", True)),
    }


def _objective_cycles_for_epoch(curriculum_cfg, epoch):
    if not curriculum_cfg["enabled"]:
        return None, -1

    remaining = int(epoch)
    for idx, (cycles, n_epochs) in enumerate(
        zip(curriculum_cfg["cycles"], curriculum_cfg["epochs_per_stage"])
    ):
        if remaining < n_epochs:
            return int(cycles), int(idx)
        remaining -= n_epochs

    return int(curriculum_cfg["cycles"][-1]), len(curriculum_cfg["cycles"]) - 1


def _num_windows_for_objective(objective_cycles, short_num_cycles, total_windows):
    if objective_cycles is None:
        return total_windows
    short_num_cycles = int(short_num_cycles)
    if short_num_cycles <= 0:
        raise ValueError("short_num_cycles must be positive")
    return min(
        int(total_windows),
        max(1, int(math.ceil(int(objective_cycles) / short_num_cycles))),
    )


def _base_lr_for_curriculum_stage(curriculum_cfg, curriculum_stage, default_lr):
    lr_per_stage = curriculum_cfg.get("lr_per_stage")
    if not curriculum_cfg["enabled"] or lr_per_stage is None:
        return float(default_lr)
    if curriculum_stage < 0:
        return float(default_lr)
    stage = min(int(curriculum_stage), len(lr_per_stage) - 1)
    return float(lr_per_stage[stage])


def _set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _save_training_curves(history, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    os.makedirs(output_dir, exist_ok=True)
    figures = {
        "train_loss.png": [
            ("train/loss", "loss"),
            ("train/recon", "recon"),
        ],
        "eval_metrics.png": [
            ("eval/loss", "loss"),
            ("eval/recon_mse_mean", "recon_mean"),
            ("eval/recon_mse_sampled", "recon_sampled"),
            ("eval/freq_rmse_hz_mean", "freq_rmse"),
            ("eval/freq_nrmse_band_mean", "freq_nrmse"),
        ],
        "eval_success.png": [
            ("eval/freq_success_rate_mean", "freq_success"),
            ("eval/amp_success_rate_mean", "amp_success"),
            ("eval/joint_amp_freq_success_rate_mean", "joint_amp_freq"),
            ("eval/complex_coeff_success_rate", "complex_success"),
        ],
        "eval_coefficients.png": [
            ("eval/amp_mape_mean", "amp_mape"),
            ("eval/phase_circ_mae_rad", "phase_mae"),
            ("eval/complex_coeff_rel_err_mean", "complex_rel_err"),
        ],
    }

    for filename, series in figures.items():
        available = [
            (history[key], label)
            for key, label in series
            if key in history and history[key]
        ]
        if not available:
            continue

        plt.figure(figsize=(10, 5))
        for points, label in available:
            xs = [step for step, _ in points]
            ys = [value for _, value in points]
            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(filename.replace(".png", "").replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=160)
        plt.close()

    return True


def _build_metrics_payload(
    last_metrics,
    best_metrics,
    best_epoch,
    early_stopped,
    early_stop_epoch,
    early_patience,
    early_monitor,
):
    if last_metrics is None:
        return None

    payload = dict(last_metrics)
    payload["last_metrics"] = dict(last_metrics)
    payload["best_metrics"] = dict(best_metrics) if best_metrics is not None else None
    payload["best_epoch_by_monitor"] = best_epoch
    payload["early_stopping_monitor"] = str(early_monitor)
    payload["best_monitor_value"] = (
        best_metrics.get(early_monitor) if best_metrics is not None else None
    )
    payload["last_monitor_value"] = last_metrics.get(early_monitor)
    payload["early_stopped"] = bool(early_stopped)
    payload["early_stop_epoch"] = early_stop_epoch
    payload["early_stopping_patience"] = int(early_patience)
    return payload


def _build_dataset(
    num_sequences,
    seed,
    data_cfg,
    signal_cfg,
    freq_lower,
    freq_upper,
    split="all",
    frequency_rho=None,
    amp_real_rho=None,
    amp_imag_rho=None,
):
    amp_prior_cfg = signal_cfg["amp_data_prior"]
    return BTTSequenceDataset(
        num_sequences=num_sequences,
        num_cycles=data_cfg["num_cycles"],
        num_probes=data_cfg["num_probes"],
        base_freq=data_cfg["base_freq"],
        fluctuation_delta=data_cfg["fluctuation_delta"],
        probe_angles=data_cfg["probes"],
        freq_lower=freq_lower,
        freq_upper=freq_upper,
        amp_real_center=signal_cfg["amp_real_center_m"],
        amp_imag_center=signal_cfg["amp_imag_center_m"],
        amp_relative_half_band=amp_prior_cfg["relative_half_band"],
        amp_min_half_band=amp_prior_cfg["min_half_band_m"],
        snr_db=signal_cfg["snr_db"],
        seed=seed,
        normalization=data_cfg.get("normalization", "per_sequence_std"),
        include_local_time_norm=data_cfg.get("include_local_time_norm", False),
        frequency_rho=frequency_rho,
        amp_real_rho=amp_real_rho,
        amp_imag_rho=amp_imag_rho,
        split=split,
    )


def _build_grouped_dataset(
    dataset_cfg,
    split,
    seed,
    data_cfg,
    signal_cfg,
    freq_lower,
    freq_upper,
    frequency_rho=None,
    amp_real_rho=None,
    amp_imag_rho=None,
):
    amp_prior_cfg = signal_cfg["amp_data_prior"]
    short_num_cycles = dataset_cfg.get("sequence_num_cycles", data_cfg["num_cycles"])
    return GroupedBTTSequenceDataset(
        split=split,
        num_param_sets=dataset_cfg["num_param_sets"],
        sequences_per_param=dataset_cfg["sequences_per_param"],
        use_long_sequence=dataset_cfg.get("use_long_sequence", False),
        short_num_cycles=short_num_cycles,
        long_sequence_num_cycles=dataset_cfg.get(
            "long_sequence_num_cycles",
            short_num_cycles,
        ),
        window_hop_cycles=dataset_cfg.get("window_hop_cycles", 1),
        chronological_split=dataset_cfg.get("chronological_split", False),
        train_ratio=dataset_cfg.get("train_ratio", 0.6),
        val_ratio=dataset_cfg.get("val_ratio", 0.2),
        num_probes=data_cfg["num_probes"],
        base_freq=data_cfg["base_freq"],
        fluctuation_delta=data_cfg["fluctuation_delta"],
        probe_angles=data_cfg["probes"],
        freq_lower=freq_lower,
        freq_upper=freq_upper,
        amp_real_center=signal_cfg["amp_real_center_m"],
        amp_imag_center=signal_cfg["amp_imag_center_m"],
        amp_relative_half_band=amp_prior_cfg["relative_half_band"],
        amp_min_half_band=amp_prior_cfg["min_half_band_m"],
        snr_db=signal_cfg["snr_db"],
        seed=seed,
        normalization=data_cfg.get("normalization", "per_sequence_std"),
        include_local_time_norm=data_cfg.get("include_local_time_norm", False),
        return_global_parent=dataset_cfg.get("return_global_parent", True),
        frequency_rho=frequency_rho,
        amp_real_rho=amp_real_rho,
        amp_imag_rho=amp_imag_rho,
    )


def main():
    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    freq_cfg = CONFIG["frequency"]
    model_cfg = CONFIG["model"]
    train_cfg = CONFIG["training"]
    loss_cfg = dict(CONFIG["loss"])
    loss_cfg["prior"] = dict(freq_cfg.get("loss_prior", {}))
    eval_cfg = CONFIG.get("eval", {})
    seed = CONFIG.get("seed", 42)
    run_dir = CONFIG.get("run_dir", ".")
    os.makedirs(run_dir, exist_ok=True)
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    freq_lower, freq_upper, freq_center, freq_half_band = compute_frequency_support(
        freq_center_hz=freq_cfg["center_hz"],
        relative_half_band=freq_cfg["relative_half_band"],
    )
    frequency_rho = freq_cfg.get("rho_k", None)
    amp_real_rho = signal_cfg.get("rho_amp_real_k", None)
    amp_imag_rho = signal_cfg.get("rho_amp_imag_k", None)
    print(f"Frequency centers: {freq_center}")
    print(f"Frequency half bands: {freq_half_band}")

    train_dataset_cfg = data_cfg.get("train_dataset", None)
    test_dataset_cfg = data_cfg.get("test_dataset", None)
    if train_dataset_cfg is None:
        raise ValueError("Stage 1 requires data.train_dataset with return_global_parent=True")
    if not bool(train_dataset_cfg.get("return_global_parent", True)):
        raise ValueError("Stage 1 requires data.train_dataset.return_global_parent=True")

    if train_dataset_cfg.get("chronological_split", False):
        train_set = _build_grouped_dataset(
            dataset_cfg=train_dataset_cfg,
            split="train",
            seed=seed,
            data_cfg=data_cfg,
            signal_cfg=signal_cfg,
            freq_lower=freq_lower,
            freq_upper=freq_upper,
            frequency_rho=frequency_rho,
            amp_real_rho=amp_real_rho,
            amp_imag_rho=amp_imag_rho,
        )
        val_set = _build_grouped_dataset(
            dataset_cfg=train_dataset_cfg,
            split="val",
            seed=seed,
            data_cfg=data_cfg,
            signal_cfg=signal_cfg,
            freq_lower=freq_lower,
            freq_upper=freq_upper,
            frequency_rho=frequency_rho,
            amp_real_rho=amp_real_rho,
            amp_imag_rho=amp_imag_rho,
        )
        test_set = _build_grouped_dataset(
            dataset_cfg=train_dataset_cfg,
            split="test",
            seed=seed,
            data_cfg=data_cfg,
            signal_cfg=signal_cfg,
            freq_lower=freq_lower,
            freq_upper=freq_upper,
            frequency_rho=frequency_rho,
            amp_real_rho=amp_real_rho,
            amp_imag_rho=amp_imag_rho,
        )
    else:
        train_set = _build_grouped_dataset(
            dataset_cfg=train_dataset_cfg,
            split="train",
            seed=seed,
            data_cfg=data_cfg,
            signal_cfg=signal_cfg,
            freq_lower=freq_lower,
            freq_upper=freq_upper,
            frequency_rho=frequency_rho,
            amp_real_rho=amp_real_rho,
            amp_imag_rho=amp_imag_rho,
        )
        val_dataset_cfg = data_cfg.get(
            "val_dataset",
            {
                **train_dataset_cfg,
                "num_param_sets": data_cfg.get("num_val_sequences", 2000),
                "sequences_per_param": 1,
                "use_long_sequence": True,
                "chronological_split": False,
                "sequence_num_cycles": data_cfg["num_cycles"],
                "long_sequence_num_cycles": train_dataset_cfg.get(
                    "long_sequence_num_cycles",
                    data_cfg["num_cycles"],
                ),
                "window_hop_cycles": train_dataset_cfg.get("window_hop_cycles", data_cfg["num_cycles"]),
                "return_global_parent": True,
            },
        )
        val_set = _build_grouped_dataset(
            dataset_cfg=val_dataset_cfg,
            split="val",
            seed=seed + 100000,
            data_cfg=data_cfg,
            signal_cfg=signal_cfg,
            freq_lower=freq_lower,
            freq_upper=freq_upper,
            frequency_rho=frequency_rho,
            amp_real_rho=amp_real_rho,
            amp_imag_rho=amp_imag_rho,
        )

    if not train_dataset_cfg.get("chronological_split", False):
        if test_dataset_cfg is not None:
            test_set = _build_grouped_dataset(
                dataset_cfg=test_dataset_cfg,
                split="test",
                seed=seed + 200000,
                data_cfg=data_cfg,
                signal_cfg=signal_cfg,
                freq_lower=freq_lower,
                freq_upper=freq_upper,
                frequency_rho=frequency_rho,
                amp_real_rho=amp_real_rho,
                amp_imag_rho=amp_imag_rho,
            )
        else:
            test_set = None

    train_loader = DataLoader(
        train_set,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            drop_last=False,
        )
    window_num_cycles = getattr(train_set, "window_num_cycles", data_cfg["num_cycles"])
    print(
        "Datasets: "
        f"train_sequences={len(train_set)}, val_sequences={len(val_set)}, "
        f"test_sequences={len(test_set) if test_set is not None else 0}, "
        f"sequence_length={window_num_cycles * data_cfg['num_probes']}"
    )

    posterior_cfg = freq_cfg.get("posterior", {})
    encoder = VariationalIndependentTimeSeriesTransformer(
        input_dim=data_cfg["input_dim"],
        output_dim=data_cfg["num_harmonics"],
        hidden_dim=model_cfg["hidden_dim"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        hidden_dim_dense=model_cfg["hidden_dim_dense"],
        num_probes=data_cfg["num_probes"],
        use_standard_pe=model_cfg.get("use_standard_pe", False),
        use_time_pe=model_cfg.get("use_time_pe", False),
        time_feature_index=model_cfg.get("time_feature_index", -1),
        time_pe_num_bands=model_cfg.get("time_pe_num_bands", 64),
        time_pe_trainable_proj=model_cfg.get("time_pe_trainable_proj", True),
        device=device,
        freq_lower_hz=freq_lower,
        freq_upper_hz=freq_upper,
        min_log_rho2=posterior_cfg.get("min_log_rho2", -12.0),
        max_log_rho2=posterior_cfg.get("max_log_rho2", -4.0),
    )
    model = PhysicalHarmonicVAE(
        encoder=encoder,
        ls_ridge=model_cfg.get("ls_ridge", 1e-6),
        use_window_position_embedding=model_cfg.get(
            "global_aggregation",
            {},
        ).get("use_window_position_embedding", True),
        amplitude_nn_cfg=CONFIG.get("amplitude_nn", {}),
    ).to(device)
    base_lr = float(train_cfg["lr"])
    amp_head_only = bool(train_cfg.get("freeze_encoder_train_amp_head_only", False))
    amp_head_train_mode = train_cfg.get("amp_head_train_mode", "head")
    if amp_head_only:
        if amp_head_train_mode not in ("head", "bias_only"):
            raise ValueError(
                "training.amp_head_train_mode must be one of "
                "'head', 'bias_only'; "
                f"got {amp_head_train_mode!r}"
            )
        for _, p in model.named_parameters():
            p.requires_grad = False
        if amp_head_train_mode == "bias_only":
            model.amp_head.bias.requires_grad = True
        else:
            for name, p in model.named_parameters():
                if name.startswith("amp_head"):
                    p.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters selected for amp_head-only training")
        base_lr = float(train_cfg.get("amp_head_lr", 1e-3))
        print(
            "Training only amp_head parameters: "
            f"mode={amp_head_train_mode}, lr={base_lr:g}, "
            f"num_tensors={len(trainable_params)}"
        )
    else:
        trainable_params = list(model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=base_lr)
    curriculum_cfg = _resolve_objective_curriculum(train_cfg)
    if curriculum_cfg["enabled"]:
        total_curriculum_epochs = sum(curriculum_cfg["epochs_per_stage"])
        if int(train_cfg["epochs"]) < total_curriculum_epochs:
            raise ValueError(
                f"training.epochs={train_cfg['epochs']} is smaller than "
                f"objective curriculum total epochs={total_curriculum_epochs}"
            )
        print(
            "Objective curriculum: "
            f"cycles={curriculum_cfg['cycles']}, "
            f"epochs_per_stage={curriculum_cfg['epochs_per_stage']}, "
            f"lr_per_stage={curriculum_cfg['lr_per_stage']}, "
            f"segment_mode={curriculum_cfg['segment_mode']}, "
            f"apply_to_encoder={curriculum_cfg['apply_to_encoder']}"
        )
    short_num_cycles = int(train_dataset_cfg.get(
        "sequence_num_cycles",
        data_cfg["num_cycles"],
    ))
    (
        lr_schedule_type,
        lr_total_steps,
        lr_warmup_steps,
        lr_min,
    ) = _resolve_lr_schedule(train_cfg, steps_per_epoch=len(train_loader))
    print(
        "LR schedule: "
        f"type={lr_schedule_type}, base_lr={base_lr:g}, min_lr={lr_min:g}, "
        f"warmup_steps={lr_warmup_steps}, total_steps={lr_total_steps}"
    )

    eval_every = int(eval_cfg.get("eval_every", 5))
    target_recon = float(eval_cfg.get("target_recon_mse", 0.1))
    dense_factor = int(eval_cfg.get("dense_factor", 4))

    early_cfg = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    early_monitor = early_cfg.get("monitor", "recon_mse_mean")
    early_patience = int(early_cfg.get("patience", 3))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_mode = early_cfg.get("mode", "min")
    if early_mode not in ("min", "max"):
        raise ValueError(f"Unsupported early_stopping.mode={early_mode}")
    best_monitor_value = float("inf") if early_mode == "min" else -float("inf")
    epochs_without_improvement = 0
    early_stopped = False
    early_stop_epoch = None
    final_metrics = None
    best_metrics = None
    best_epoch = None
    best_ckpt_path = None

    ckpt_cfg = CONFIG.get("checkpoint", {})
    ckpt_dir = ckpt_cfg.get("dir", "checkpoints")
    ckpt_name = ckpt_cfg.get("name", "latest.pt")
    ckpt_save_every = int(ckpt_cfg.get("save_every", 20))
    ckpt_resume_from = ckpt_cfg.get("resume_from", None)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_cfg = CONFIG.get("logging", {})
    enable_tensorboard = log_cfg.get("enable_tensorboard", True)
    tensorboard_dir = log_cfg.get("tensorboard_dir", "artifacts/tensorboard")
    save_curves = log_cfg.get("save_curves", True)
    curve_dir = log_cfg.get("curve_dir", "artifacts/curves")
    curve_every = max(int(log_cfg.get("curve_every", 1)), 1)
    writer = None
    if enable_tensorboard and SummaryWriter is not None:
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard log dir: {tensorboard_dir}")

    start_epoch = 0
    total_steps = 0
    nonfinite_steps = 0
    grad_clip_triggered_steps = 0
    epoch_to_target = None
    if ckpt_resume_from:
        if not os.path.exists(ckpt_resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_resume_from}")
        checkpoint = torch.load(ckpt_resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        total_steps = int(checkpoint.get("total_steps", 0))
        nonfinite_steps = int(checkpoint.get("nonfinite_steps", 0))
        grad_clip_triggered_steps = int(checkpoint.get("grad_clip_triggered_steps", 0))
        epoch_to_target = checkpoint.get("epoch_to_target", None)
        print(f"Resumed from checkpoint: {ckpt_resume_from}")

    history = {}
    try:
        for epoch in range(start_epoch, train_cfg["epochs"]):
            model.train()
            objective_cycles, curriculum_stage = _objective_cycles_for_epoch(
                curriculum_cfg,
                epoch,
            )
            train_sums = {
                "loss": 0.0,
                "optimization_loss": 0.0,
                "loss_scale": 0.0,
                "recon": 0.0,
                "recon_loss_per_point": 0.0,
                "recon_mse_sampled": 0.0,
                "recon_nll_full": 0.0,
                "freq_kl": 0.0,
                "freq_kl_raw": 0.0,
                "amp_kl": 0.0,
                "amp_kl_raw": 0.0,
                "amp_supervision_loss": 0.0,
                "amp_supervision_loss_raw": 0.0,
                "amp_supervision_weight": 0.0,
                "amp_supervision_target_norm_mean": 0.0,
                "amp_supervision_error_norm_mean": 0.0,
                "amp_supervision_frequency_source_id": 0.0,
                "amp_head_grad_norm": 0.0,
                "amp_head_param_norm_before": 0.0,
                "amp_head_param_norm_after": 0.0,
                "amp_head_update_norm": 0.0,
                "c_nn_norm_mean": 0.0,
                "freq_kl_beta_anneal": 0.0,
                "freq_prior_reg": 0.0,
                "posterior_std_hz_mean": 0.0,
                "freq_sample_outside_rate": 0.0,
                "noise_var_norm_mean": 0.0,
                "ls_cond_mean": 0.0,
                "ls_cond_p95": 0.0,
                "ls_amp_norm_mean": 0.0,
                "ls_amp_norm_p95": 0.0,
                "amp_prior_quad": 0.0,
                "amp_lambda_mean": 0.0,
                "amp_lambda_min": 0.0,
                "amp_lambda_max": 0.0,
                "amp_prior_var_norm_mean": 0.0,
                "map_amp_norm_mean": 0.0,
                "map_amp_norm_p95": 0.0,
                "nn_amp_norm_mean": 0.0,
                "nn_amp_norm_p95": 0.0,
                "marginal_nll": 0.0,
                "marginal_quad": 0.0,
                "marginal_logdet": 0.0,
                "amp_post_var_trace": 0.0,
                "amp_post_std_mean": 0.0,
                "amp_uncertainty_to_prior_ratio_mean": 0.0,
                "objective_cycles": 0.0,
                "objective_stage": 0.0,
                "objective_num_windows": 0.0,
                "objective_num_points": 0.0,
            }
            train_batches = 0

            for batch in train_loader:
                x_batch = batch["x_windows"].to(device)
                t_batch = batch["t_windows"].to(device)
                probe_ids = batch["probe_ids_windows"].to(device)
                target_batch = batch["target_windows"].to(device)
                window_start_cycle = batch["window_start_cycle"].to(device)
                noise_var_norm = batch["noise_var_norm"].to(device)
                amp_scale = batch["amp_scale"].to(device)
                true_freq_hz = batch["true_freq_hz"].to(device)
                dataset_state = extract_dataset_state(batch, device)

                objective_num_windows = _num_windows_for_objective(
                    objective_cycles=objective_cycles,
                    short_num_cycles=short_num_cycles,
                    total_windows=x_batch.shape[1],
                )
                if curriculum_cfg["enabled"] and curriculum_cfg["apply_to_encoder"]:
                    x_model = x_batch[:, :objective_num_windows]
                    probe_model = probe_ids[:, :objective_num_windows]
                    window_start_model = window_start_cycle[:, :objective_num_windows]
                else:
                    x_model = x_batch
                    probe_model = probe_ids
                    window_start_model = window_start_cycle

                optimizer.zero_grad()
                model_outputs = model.forward_global(
                    x_model,
                    probe_ids_windows=probe_model,
                    window_start_cycle=window_start_model,
                )
                loss, recon, freq_kl, loss_diag = compute_static_global_objective(
                    target_windows=target_batch,
                    t_windows_abs=t_batch,
                    model_outputs=model_outputs,
                    model=model,
                    loss_cfg=loss_cfg,
                    noise_var_norm=noise_var_norm,
                    amp_scale=amp_scale,
                    signal_cfg=signal_cfg,
                    dataset_state=dataset_state,
                    true_freq_hz=true_freq_hz,
                    global_step=total_steps + 1,
                    objective_cycles=objective_cycles,
                    short_num_cycles=short_num_cycles,
                    segment_mode=curriculum_cfg["segment_mode"],
                )

                if not torch.isfinite(loss):
                    nonfinite_steps += 1
                    continue

                loss.backward()
                amp_head_params = list(model.amp_head.parameters())
                amp_head_grad_norm = _grad_norm(amp_head_params)
                amp_head_param_norm_before = _param_norm(amp_head_params)
                c_nn_norm_mean = 0.0
                if "c_nn" in model_outputs:
                    c_nn_norm_mean = torch.linalg.norm(
                        model_outputs["c_nn"],
                        dim=-1,
                    ).mean().item()
                grad_clip_cfg = train_cfg.get("grad_clip", {})
                if grad_clip_cfg.get("enabled", False):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(grad_clip_cfg.get("max_norm", 1.0)),
                    )
                    if torch.isfinite(grad_norm) and grad_norm > float(
                        grad_clip_cfg.get("max_norm", 1.0)
                    ):
                        grad_clip_triggered_steps += 1
                else:
                    grad_norm = None

                stage_base_lr = _base_lr_for_curriculum_stage(
                    curriculum_cfg=curriculum_cfg,
                    curriculum_stage=curriculum_stage,
                    default_lr=base_lr,
                )
                if amp_head_only:
                    stage_base_lr = base_lr
                step_lr = _compute_learning_rate(
                    base_lr=stage_base_lr,
                    step=total_steps + 1,
                    schedule_type=lr_schedule_type,
                    total_steps=lr_total_steps,
                    warmup_steps=lr_warmup_steps,
                    min_lr=lr_min,
                )
                _set_optimizer_lr(optimizer, step_lr)
                optimizer.step()
                amp_head_param_norm_after = _param_norm(amp_head_params)
                amp_head_update_norm = abs(
                    amp_head_param_norm_after - amp_head_param_norm_before
                )
                total_steps += 1
                train_batches += 1

                train_sums["loss"] += float(loss_diag["loss"].item())
                train_sums["optimization_loss"] += loss.item()
                train_sums["loss_scale"] += float(loss_diag["loss_scale"].item())
                train_sums["recon"] += recon.item()
                train_sums["recon_loss_per_point"] += float(
                    loss_diag["recon_loss_per_point"].item()
                )
                train_sums["recon_mse_sampled"] += float(loss_diag["recon_mse_sampled"].item())
                train_sums["recon_nll_full"] += float(loss_diag["recon_nll_full"].item())
                train_sums["freq_kl"] += float(loss_diag["freq_kl"].item())
                train_sums["freq_kl_raw"] += float(loss_diag["freq_kl_raw"].item())
                train_sums["amp_kl"] += float(loss_diag["amp_kl"].item())
                train_sums["amp_kl_raw"] += float(loss_diag["amp_kl_raw"].item())
                train_sums["amp_supervision_loss"] += float(
                    loss_diag["amp_supervision_loss"].item()
                )
                train_sums["amp_supervision_loss_raw"] += float(
                    loss_diag["amp_supervision_loss_raw"].item()
                )
                train_sums["amp_supervision_weight"] += float(
                    loss_diag["amp_supervision_weight"].item()
                )
                train_sums["amp_supervision_target_norm_mean"] += float(
                    loss_diag["amp_supervision_target_norm_mean"].item()
                )
                train_sums["amp_supervision_error_norm_mean"] += float(
                    loss_diag["amp_supervision_error_norm_mean"].item()
                )
                train_sums["amp_supervision_frequency_source_id"] += float(
                    loss_diag["amp_supervision_frequency_source_id"].item()
                )
                train_sums["amp_head_grad_norm"] += amp_head_grad_norm
                train_sums["amp_head_param_norm_before"] += amp_head_param_norm_before
                train_sums["amp_head_param_norm_after"] += amp_head_param_norm_after
                train_sums["amp_head_update_norm"] += amp_head_update_norm
                train_sums["c_nn_norm_mean"] += c_nn_norm_mean
                train_sums["freq_kl_beta_anneal"] += float(
                    loss_diag["freq_kl_beta_anneal"].item()
                )
                train_sums["freq_prior_reg"] += float(loss_diag["freq_prior_reg"].item())
                for key in (
                    "posterior_std_hz_mean",
                    "freq_sample_outside_rate",
                    "noise_var_norm_mean",
                    "ls_cond_mean",
                    "ls_cond_p95",
                    "ls_amp_norm_mean",
                    "ls_amp_norm_p95",
                    "amp_prior_quad",
                    "amp_lambda_mean",
                    "amp_lambda_min",
                    "amp_lambda_max",
                    "amp_prior_var_norm_mean",
                    "map_amp_norm_mean",
                    "map_amp_norm_p95",
                    "nn_amp_norm_mean",
                    "nn_amp_norm_p95",
                    "marginal_nll",
                    "marginal_quad",
                    "marginal_logdet",
                    "amp_post_var_trace",
                    "amp_post_std_mean",
                    "amp_uncertainty_to_prior_ratio_mean",
                    "objective_cycles",
                    "objective_num_windows",
                    "objective_num_points",
                ):
                    train_sums[key] += float(loss_diag[key].item())
                train_sums["objective_stage"] += float(curriculum_stage)
                for key, value in loss_diag.items():
                    if not key.startswith("data_state/"):
                        continue
                    if torch.is_tensor(value):
                        value = value.item()
                    train_sums.setdefault(key, 0.0)
                    train_sums[key] += float(value)

                _log_scalar(writer, "train_step/loss", loss_diag["loss"].item(), total_steps)
                _log_scalar(writer, "train_step/optimization_loss", loss.item(), total_steps)
                _log_scalar(
                    writer,
                    "train_step/loss_scale",
                    loss_diag["loss_scale"].item(),
                    total_steps,
                )
                _log_scalar(writer, "train_step/recon", recon.item(), total_steps)
                _log_scalar(
                    writer,
                    "train_step/recon_loss_per_point",
                    loss_diag["recon_loss_per_point"].item(),
                    total_steps,
                )
                _log_scalar(
                    writer,
                    "train_step/recon_mse_sampled",
                    loss_diag["recon_mse_sampled"].item(),
                    total_steps,
                )
                _log_scalar(
                    writer,
                    "train_step/recon_nll_full",
                    loss_diag["recon_nll_full"].item(),
                    total_steps,
                )
                _log_scalar(writer, "train_step/freq_kl", loss_diag["freq_kl"].item(), total_steps)
                _log_scalar(
                    writer,
                    "train_step/freq_kl_raw",
                    loss_diag["freq_kl_raw"].item(),
                    total_steps,
                )
                _log_scalar(
                    writer,
                    "train_step/freq_kl_beta_anneal",
                    loss_diag["freq_kl_beta_anneal"].item(),
                    total_steps,
                )
                _log_scalar(
                    writer,
                    "train_step/freq_prior_reg",
                    loss_diag["freq_prior_reg"].item(),
                    total_steps,
                )
                _log_scalar(
                    writer,
                    "train_step/amp_supervision_loss",
                    loss_diag["amp_supervision_loss"].item(),
                    total_steps,
                )
                _log_scalar(
                    writer,
                    "train_step/amp_supervision_error_norm_mean",
                    loss_diag["amp_supervision_error_norm_mean"].item(),
                    total_steps,
                )
                _log_scalar(writer, "train_step/amp_head_grad_norm", amp_head_grad_norm, total_steps)
                _log_scalar(
                    writer,
                    "train_step/amp_head_update_norm",
                    amp_head_update_norm,
                    total_steps,
                )
                _log_scalar(writer, "train_step/c_nn_norm_mean", c_nn_norm_mean, total_steps)
                if grad_norm is not None and torch.isfinite(grad_norm):
                    _log_scalar(writer, "train_step/grad_norm", grad_norm.item(), total_steps)
                _log_scalar(writer, "train_step/lr", step_lr, total_steps)
                _log_scalar(
                    writer,
                    "train_step/objective_cycles",
                    loss_diag["objective_cycles"].item(),
                    total_steps,
                )
                _log_scalar(
                    writer,
                    "train_step/objective_num_windows",
                    loss_diag["objective_num_windows"].item(),
                    total_steps,
                )

            if train_batches == 0:
                print(f"epoch={epoch:04d} no valid training batches")
                continue

            train_means = {
                key: value / max(train_batches, 1) for key, value in train_sums.items()
            }
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_means["lr"] = current_lr
            for key, value in train_means.items():
                _append_history(history, f"train/{key}", epoch + 1, value)
                _log_scalar(writer, f"train_epoch/{key}", value, epoch + 1)

            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_means['loss']:.6f} "
                f"opt_loss={train_means['optimization_loss']:.6f} "
                f"train_recon_nll={train_means['recon']:.6f} "
                f"train_recon_mse={train_means['recon_mse_sampled']:.6f} "
                f"obj_cycles={train_means['objective_cycles']:.0f} "
                f"obj_stage={train_means['objective_stage']:.0f} "
                f"obj_windows={train_means['objective_num_windows']:.0f} "
                f"freq_kl={train_means['freq_kl']:.6f} "
                f"freq_kl_raw={train_means['freq_kl_raw']:.6f} "
                f"amp_kl={train_means['amp_kl']:.6f} "
                f"amp_kl_raw={train_means['amp_kl_raw']:.6f} "
                f"kl_anneal={train_means['freq_kl_beta_anneal']:.4f} "
                f"posterior_std={train_means['posterior_std_hz_mean']:.4f} "
                f"outside={train_means['freq_sample_outside_rate']:.4f} "
                f"ls_cond={train_means['ls_cond_mean']:.3e} "
                f"marginal_quad={train_means['marginal_quad']:.3e} "
                f"marginal_logdet={train_means['marginal_logdet']:.3e} "
                f"amp_post_std={train_means['amp_post_std_mean']:.3e} "
                f"amp_prior_quad={train_means['amp_prior_quad']:.3e} "
                f"map_amp_norm={train_means['map_amp_norm_mean']:.3e} "
                f"nn_amp_norm={train_means['nn_amp_norm_mean']:.3e} "
                f"lr={current_lr:.3e}"
            )

            need_eval = (
                epoch == 0
                or (epoch + 1) % eval_every == 0
                or epoch == (train_cfg["epochs"] - 1)
            )
            if need_eval:
                val_metrics = evaluate_model(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    loss_cfg=loss_cfg,
                    signal_cfg=signal_cfg,
                    dense_factor=dense_factor,
                )

                if epoch_to_target is None and val_metrics["recon_mse_mean"] <= target_recon:
                    epoch_to_target = epoch + 1

                final_metrics = dict(val_metrics)
                final_metrics["epoch"] = epoch + 1
                final_metrics["total_steps"] = total_steps
                final_metrics["seed"] = seed
                final_metrics["epoch_to_target"] = epoch_to_target

                for key, value in val_metrics.items():
                    _append_history(history, f"eval/{key}", epoch + 1, value)
                    _log_scalar(writer, f"eval/{key}", value, epoch + 1)

                if early_monitor not in val_metrics:
                    raise KeyError(
                        f"early_stopping.monitor={early_monitor!r} is not in val_metrics. "
                        f"Available keys: {sorted(val_metrics.keys())}"
                    )
                monitor_value = float(val_metrics[early_monitor])
                improved = (
                    monitor_value < (best_monitor_value - early_min_delta)
                    if early_mode == "min"
                    else monitor_value > (best_monitor_value + early_min_delta)
                )

                if improved:
                    best_monitor_value = monitor_value
                    best_metrics = dict(final_metrics)
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    best_ckpt = os.path.join(
                        ckpt_dir,
                        f"best_{str(early_monitor).replace('/', '_')}.pt",
                    )
                    best_ckpt_path = best_ckpt
                    save_checkpoint(
                        path=best_ckpt,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        total_steps=total_steps,
                        nonfinite_steps=nonfinite_steps,
                        grad_clip_triggered_steps=grad_clip_triggered_steps,
                        epoch_to_target=epoch_to_target,
                    )
                    print(f"Saved best checkpoint by {early_monitor}: {best_ckpt}")
                else:
                    epochs_without_improvement += 1

                grad_clip_ratio = grad_clip_triggered_steps / max(total_steps, 1)

                print(
                    "[EVAL] "
                    f"epoch={epoch:04d} "
                    f"loss={val_metrics['loss']:.6f} "
                    f"freq_kl={val_metrics['freq_kl']:.6f} "
                    f"freq_kl_raw={val_metrics.get('freq_kl_raw', val_metrics['freq_kl']):.6f} "
                    f"amp_kl={val_metrics.get('amp_kl', 0.0):.6f} "
                    f"amp_kl_raw={val_metrics.get('amp_kl_raw', 0.0):.6f} "
                    f"recon_nll={val_metrics['recon_nll_sampled']:.6f} "
                    f"recon_mse_mean={val_metrics['recon_mse_mean']:.6f} "
                    f"recon_mse_sampled={val_metrics['recon_mse_sampled']:.6f} "
                    f"freq_rmse_hz_mean={val_metrics['freq_rmse_hz_mean']:.4f} "
                    f"freq_nrmse_band_mean={val_metrics['freq_nrmse_band_mean']:.4f} "
                    f"freq_success={val_metrics['freq_success_rate_mean']:.4f} "
                    f"amp_mape={val_metrics['amp_mape_mean']:.4f} "
                    f"amp_success={val_metrics['amp_success_rate_mean']:.4f} "
                    f"joint_amp_freq_success={val_metrics['joint_amp_freq_success_rate_mean']:.4f} "
                    f"phase_mae={val_metrics['phase_circ_mae_rad']:.4f} "
                    f"complex_rel_err={val_metrics['complex_coeff_rel_err_mean']:.4f} "
                    f"complex_success={val_metrics['complex_coeff_success_rate']:.4f} "
                    f"posterior_std={val_metrics['posterior_std_hz_mean']:.4f} "
                    f"amp_post_std={val_metrics.get('amp_post_std_norm_mean', 0.0):.3e} "
                    f"marginal_quad={val_metrics.get('marginal_quad', 0.0):.3e} "
                    f"marginal_logdet={val_metrics.get('marginal_logdet', 0.0):.3e} "
                    f"outside={val_metrics['freq_sample_outside_rate']:.4f} "
                    f"ls_cond_p95={val_metrics['ls_cond_p95']:.3e} "
                    f"ls_amp_norm_p95={val_metrics['ls_amp_norm_p95']:.3e} "
                    f"grad_clip_ratio={grad_clip_ratio:.6f} "
                    f"epoch_to_target={epoch_to_target if epoch_to_target is not None else -1}"
                )

                if save_curves and ((epoch + 1) % curve_every == 0):
                    if _save_training_curves(history, curve_dir):
                        print(f"Updated curve images in: {curve_dir}")

                metrics_path = os.path.join(run_dir, "metrics.json")
                metrics_to_save = _build_metrics_payload(
                    last_metrics=final_metrics,
                    best_metrics=best_metrics,
                    best_epoch=best_epoch,
                    early_stopped=early_stopped,
                    early_stop_epoch=early_stop_epoch,
                    early_patience=early_patience,
                    early_monitor=early_monitor,
                )
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_to_save, f, indent=2)
                print(f"Saved metrics: {metrics_path}")

                if early_enabled and epochs_without_improvement >= early_patience:
                    early_stopped = True
                    early_stop_epoch = epoch + 1
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}: "
                        f"{early_monitor} did not improve for {early_patience} eval checks."
                    )
                    break

            should_save_ckpt = (
                (epoch + 1) % ckpt_save_every == 0
                or epoch == (train_cfg["epochs"] - 1)
            )
            if should_save_ckpt:
                latest_ckpt = os.path.join(ckpt_dir, ckpt_name)
                epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch + 1:04d}.pt")
                save_checkpoint(
                    path=latest_ckpt,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    total_steps=total_steps,
                    nonfinite_steps=nonfinite_steps,
                    grad_clip_triggered_steps=grad_clip_triggered_steps,
                    epoch_to_target=epoch_to_target,
                )
                save_checkpoint(
                    path=epoch_ckpt,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    total_steps=total_steps,
                    nonfinite_steps=nonfinite_steps,
                    grad_clip_triggered_steps=grad_clip_triggered_steps,
                    epoch_to_target=epoch_to_target,
                )
                print(f"Saved checkpoint: {latest_ckpt}")

        final_eval_checkpoint = "last"
        if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            final_eval_checkpoint = "best"
            best_checkpoint_epoch = int(checkpoint.get("epoch", -1)) + 1
            print(f"Loaded best checkpoint for final evaluation: {best_ckpt_path}")
            final_best_ckpt = os.path.join(ckpt_dir, "final_best.pt")
            save_checkpoint(
                path=final_best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=best_checkpoint_epoch - 1,
                total_steps=total_steps,
                nonfinite_steps=nonfinite_steps,
                grad_clip_triggered_steps=grad_clip_triggered_steps,
                epoch_to_target=epoch_to_target,
            )
            print(f"Saved final best checkpoint: {final_best_ckpt}")

            best_final_metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                loss_cfg=loss_cfg,
                signal_cfg=signal_cfg,
                dense_factor=dense_factor,
            )
            final_metrics = dict(best_final_metrics)
            final_metrics["epoch"] = best_epoch if best_epoch is not None else best_checkpoint_epoch
            final_metrics["total_steps"] = total_steps
            final_metrics["seed"] = seed
            final_metrics["epoch_to_target"] = epoch_to_target
            final_metrics["final_eval_checkpoint"] = final_eval_checkpoint
            final_metrics["best_checkpoint_path"] = best_ckpt_path
            final_metrics["final_best_checkpoint_path"] = final_best_ckpt
            if best_metrics is None:
                best_metrics = dict(final_metrics)
                best_epoch = final_metrics["epoch"]

            metrics_path = os.path.join(run_dir, "metrics.json")
            metrics_to_save = _build_metrics_payload(
                last_metrics=final_metrics,
                best_metrics=best_metrics,
                best_epoch=best_epoch,
                early_stopped=early_stopped,
                early_stop_epoch=early_stop_epoch,
                early_patience=early_patience,
                early_monitor=early_monitor,
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_to_save, f, indent=2)
            print(
                "[FINAL] "
                f"checkpoint={final_eval_checkpoint} "
                f"epoch={final_metrics['epoch']} "
                f"loss={final_metrics['loss']:.6f} "
                f"recon_mse_mean={final_metrics['recon_mse_mean']:.6f} "
                f"freq_rmse_hz_mean={final_metrics['freq_rmse_hz_mean']:.4f} "
                f"amp_mape={final_metrics['amp_mape_mean']:.4f}"
            )
            print(f"Saved final best metrics: {metrics_path}")
        elif final_metrics is not None:
            final_metrics["final_eval_checkpoint"] = final_eval_checkpoint

        if bool(eval_cfg.get("evaluate_test_dataset", False)) and test_loader is not None:
            eval_checkpoint = final_eval_checkpoint
            test_checkpoint = eval_cfg.get("test_checkpoint", "best")
            if (
                test_checkpoint == "best"
                and best_ckpt_path is not None
                and os.path.exists(best_ckpt_path)
            ):
                checkpoint = torch.load(best_ckpt_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                eval_checkpoint = "best"

            test_metrics = evaluate_model(
                model=model,
                dataloader=test_loader,
                device=device,
                loss_cfg=loss_cfg,
                signal_cfg=signal_cfg,
                dense_factor=dense_factor,
            )
            test_metrics["epoch"] = epoch + 1
            test_metrics["total_steps"] = total_steps
            test_metrics["seed"] = seed
            test_metrics["test_sequences"] = len(test_set)
            test_metrics["eval_checkpoint"] = eval_checkpoint

            for key, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    _log_scalar(writer, f"test/{key}", value, total_steps)

            test_metrics_path = os.path.join(
                run_dir,
                eval_cfg.get("test_metrics_name", "test_metrics.json"),
            )
            with open(test_metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
            print(
                "[TEST] "
                f"checkpoint={eval_checkpoint} "
                f"loss={test_metrics['loss']:.6f} "
                f"recon_mse_mean={test_metrics['recon_mse_mean']:.6f} "
                f"freq_rmse_hz_mean={test_metrics['freq_rmse_hz_mean']:.4f} "
                f"freq_success={test_metrics['freq_success_rate_mean']:.4f} "
                f"amp_mape={test_metrics['amp_mape_mean']:.4f} "
                f"test_sequences={len(test_set)}"
            )
            print(f"Saved test metrics: {test_metrics_path}")
    finally:
        if final_metrics is not None:
            metrics_path = os.path.join(run_dir, "metrics.json")
            metrics_to_save = _build_metrics_payload(
                last_metrics=final_metrics,
                best_metrics=best_metrics,
                best_epoch=best_epoch,
                early_stopped=early_stopped,
                early_stop_epoch=early_stop_epoch,
                early_patience=early_patience,
                early_monitor=early_monitor,
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"Saved final metrics: {metrics_path}")
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
