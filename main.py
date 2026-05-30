import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import BTTSequenceDataset
from Encoder import VariationalIndependentTimeSeriesTransformer
from eval import evaluate_model
from loss import compute_harmonic_loss
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
        raise ValueError("training.lr_schedule.warmup_steps must be smaller than total_steps")
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
            ("train/freq_kl", "freq_kl"),
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


def _build_dataset(num_sequences, seed, data_cfg, signal_cfg, freq_lower, freq_upper):
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
    )


def main():
    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    freq_cfg = CONFIG["frequency"]
    model_cfg = CONFIG["model"]
    train_cfg = CONFIG["training"]
    loss_cfg = CONFIG["loss"]
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
    print(f"Frequency centers: {freq_center}")
    print(f"Frequency half bands: {freq_half_band}")

    train_set = _build_dataset(
        num_sequences=data_cfg["num_train_sequences"],
        seed=seed,
        data_cfg=data_cfg,
        signal_cfg=signal_cfg,
        freq_lower=freq_lower,
        freq_upper=freq_upper,
    )
    val_set = _build_dataset(
        num_sequences=data_cfg["num_val_sequences"],
        seed=seed + 100000,
        data_cfg=data_cfg,
        signal_cfg=signal_cfg,
        freq_lower=freq_lower,
        freq_upper=freq_upper,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    print(
        "Datasets: "
        f"train_sequences={len(train_set)}, val_sequences={len(val_set)}, "
        f"sequence_length={data_cfg['num_cycles'] * data_cfg['num_probes']}"
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
        use_standard_pe=model_cfg["use_standard_pe"],
        device=device,
        freq_lower_hz=freq_lower,
        freq_upper_hz=freq_upper,
        min_log_rho2=posterior_cfg.get("min_log_rho2", -12.0),
        max_log_rho2=posterior_cfg.get("max_log_rho2", -4.0),
    )
    model = PhysicalHarmonicVAE(
        encoder=encoder,
        ls_ridge=model_cfg.get("ls_ridge", 1e-6),
    ).to(device)
    base_lr = float(train_cfg["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
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
            train_sums = {
                "loss": 0.0,
                "recon": 0.0,
                "freq_kl": 0.0,
                "freq_prior_reg": 0.0,
                "posterior_std_hz_mean": 0.0,
                "freq_sample_outside_rate": 0.0,
                "ls_cond_mean": 0.0,
                "ls_cond_p95": 0.0,
                "ls_amp_norm_mean": 0.0,
                "ls_amp_norm_p95": 0.0,
            }
            train_batches = 0

            for batch in train_loader:
                x_batch = batch["x"].to(device)
                t_batch = batch["t"].to(device)
                probe_ids = batch["probe_ids"].to(device)
                target_batch = batch["target"].to(device)
                t_local = t_batch - t_batch[:, :1]

                optimizer.zero_grad()
                model_outputs = model(x_batch, t_local, probe_ids=probe_ids)
                loss, recon, freq_kl, loss_diag = compute_harmonic_loss(
                    x_target=target_batch,
                    model_outputs=model_outputs,
                    model=model,
                    t=t_local,
                    loss_cfg=loss_cfg,
                )

                if not torch.isfinite(loss):
                    nonfinite_steps += 1
                    continue

                loss.backward()
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

                step_lr = _compute_learning_rate(
                    base_lr=base_lr,
                    step=total_steps + 1,
                    schedule_type=lr_schedule_type,
                    total_steps=lr_total_steps,
                    warmup_steps=lr_warmup_steps,
                    min_lr=lr_min,
                )
                _set_optimizer_lr(optimizer, step_lr)
                optimizer.step()
                total_steps += 1
                train_batches += 1

                train_sums["loss"] += loss.item()
                train_sums["recon"] += recon.item()
                train_sums["freq_kl"] += freq_kl.item()
                train_sums["freq_prior_reg"] += freq_kl.item()
                for key in (
                    "posterior_std_hz_mean",
                    "freq_sample_outside_rate",
                    "ls_cond_mean",
                    "ls_cond_p95",
                    "ls_amp_norm_mean",
                    "ls_amp_norm_p95",
                ):
                    train_sums[key] += float(loss_diag[key].item())

                _log_scalar(writer, "train_step/loss", loss.item(), total_steps)
                _log_scalar(writer, "train_step/recon", recon.item(), total_steps)
                _log_scalar(writer, "train_step/freq_kl", freq_kl.item(), total_steps)
                _log_scalar(writer, "train_step/freq_prior_reg", freq_kl.item(), total_steps)
                if grad_norm is not None and torch.isfinite(grad_norm):
                    _log_scalar(writer, "train_step/grad_norm", grad_norm.item(), total_steps)
                _log_scalar(writer, "train_step/lr", step_lr, total_steps)

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
                f"train_recon={train_means['recon']:.6f} "
                f"freq_kl={train_means['freq_kl']:.6f} "
                f"freq_prior_reg={train_means['freq_prior_reg']:.6f} "
                f"posterior_std={train_means['posterior_std_hz_mean']:.4f} "
                f"outside={train_means['freq_sample_outside_rate']:.4f} "
                f"ls_cond={train_means['ls_cond_mean']:.3e} "
                f"ls_amp_norm={train_means['ls_amp_norm_mean']:.3e} "
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
