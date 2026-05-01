import os
from torch.utils.data import DataLoader
import torch
import numpy as np

from dataset import (
    BTTPatchDataset,
    build_btt_point_features,
    chronological_train_val_split,
)
from Encoder import VariationalIndependentTimeSeriesTransformer
from VAE import PhysicalHarmonicVAE
from loss import compute_harmonic_elbo
from eval import evaluate_model
from synthesis_dataset import (
    simulate_fluctuating_speed_btt,
    generate_complex_harmonic_displacement,
)
from config import CONFIG

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def build_prior_a_w(freqs_hz):
    """Build Maxwell scale prior from target frequencies."""
    true_w = 2 * np.pi * np.array(freqs_hz)
    return true_w / (2.0 * np.sqrt(2.0 / np.pi))


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
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_steps": total_steps,
        "nonfinite_steps": nonfinite_steps,
        "grad_clip_triggered_steps": grad_clip_triggered_steps,
        "epoch_to_target": epoch_to_target,
    }
    torch.save(state, path)


def _append_history(history, key, step, value):
    history.setdefault(key, []).append((int(step), float(value)))


def _log_scalar(writer, tag, value, step):
    if writer is None:
        return
    writer.add_scalar(tag, float(value), int(step))


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
            ("train/kl", "kl"),
        ],
        "eval_metrics.png": [
            ("eval/loss", "loss"),
            ("eval/recon_btt_mse", "recon_btt_mse"),
            ("eval/recon_dense_mse", "recon_dense_mse"),
        ],
        "eval_harmonics.png": [
            ("eval/freq_mae_hz", "freq_mae_hz"),
            ("eval/amp_mape", "amp_mape"),
            ("eval/phase_circ_mae_rad", "phase_circ_mae_rad"),
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

        plt.xlabel("Step / Epoch")
        plt.ylabel("Value")
        plt.title(filename.replace(".png", "").replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=160)
        plt.close()

    return True


def main():
    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    model_cfg = CONFIG["model"]
    train_cfg = CONFIG["training"]
    loss_cfg = CONFIG["loss"]
    seed = CONFIG.get("seed", 42)
    set_global_seed(seed)

    prior_a_w = build_prior_a_w(signal_cfg["freqs_hz"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    )

    model = PhysicalHarmonicVAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    t_samples, freqs_per_rev, rev_ids, probe_ids, theta_samples, freqs_at_samples = (
        simulate_fluctuating_speed_btt(
            n_revs=data_cfg["n_revs"],
            base_freq_x=data_cfg["base_freq"],
            delta=data_cfg["fluctuation_delta"],
            probe_angles=data_cfg["probes"],
        )
    )

    x_observed, _ = generate_complex_harmonic_displacement(
        t=t_samples,
        freqs=signal_cfg["freqs_hz"],
        amp_real=signal_cfg["amp_real_m"],
        amp_imag=signal_cfg["amp_imag_m"],
        snr_db=signal_cfg["snr_db"],
    )

    amp_scale = np.std(x_observed)
    x_observed_norm = x_observed / (amp_scale + 1e-12)

    features, t_samples, rev_ids, probe_ids = build_btt_point_features(
        x_observed=x_observed_norm,
        t_samples=t_samples,
        rev_ids=rev_ids,
        probe_ids=probe_ids,
        theta_samples=theta_samples,
        freqs_at_samples=freqs_at_samples,
        base_freq=data_cfg["base_freq"],
        n_revs=data_cfg["n_revs"],
    )

    dataset = BTTPatchDataset(
        features=features,
        t_samples=t_samples,
        rev_ids=rev_ids,
        probe_ids=probe_ids,
        window_revs=data_cfg["window_revs"],
        hop_revs=data_cfg["hop_revs"],
        num_probes=data_cfg["num_probes"],
    )

    eval_cfg = CONFIG.get("eval", {})
    val_ratio = eval_cfg.get("val_ratio", 0.2)
    eval_every = eval_cfg.get("eval_every", 10)
    dense_factor = eval_cfg.get("dense_factor", 4)
    target_recon = eval_cfg.get("target_recon_btt_mse", 0.1)
    ckpt_cfg = CONFIG.get("checkpoint", {})
    log_cfg = CONFIG.get("logging", {})
    ckpt_dir = ckpt_cfg.get("dir", "checkpoints")
    ckpt_save_every = ckpt_cfg.get("save_every", 10)
    ckpt_resume_from = ckpt_cfg.get("resume_from", None)
    ckpt_name = ckpt_cfg.get("name", "latest.pt")
    enable_tensorboard = log_cfg.get("enable_tensorboard", True)
    tensorboard_dir = log_cfg.get("tensorboard_dir", "artifacts/tensorboard")
    save_curves = log_cfg.get("save_curves", True)
    curve_dir = log_cfg.get("curve_dir", "artifacts/curves")
    curve_every = max(int(log_cfg.get("curve_every", 1)), 1)
    if ckpt_save_every <= 0:
        raise ValueError(f"checkpoint.save_every must be > 0, got {ckpt_save_every}")

    train_set, val_set, split_info = chronological_train_val_split(
        dataset,
        val_ratio=val_ratio,
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
        "Dataset split: "
        f"total={split_info['n_total']}, "
        f"train={split_info['n_train']}, "
        f"val={split_info['n_val']}, "
        f"gap_windows={split_info['gap_windows']}, "
        f"train_end_idx={split_info['train_end_idx']}, "
        f"val_start_idx={split_info['val_start_idx']}, "
        f"eval_every={eval_every}"
    )

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
        print(
            f"Resumed from checkpoint: {ckpt_resume_from} "
            f"(start_epoch={start_epoch}, total_steps={total_steps})"
        )

    if start_epoch >= train_cfg["epochs"]:
        print(
            f"No training needed: start_epoch={start_epoch} >= epochs={train_cfg['epochs']}. "
            f"Increase training.epochs or use an earlier checkpoint."
        )
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    history = {}

    writer = None
    if enable_tensorboard:
        if SummaryWriter is None:
            print("TensorBoard logging disabled: torch.utils.tensorboard is unavailable.")
        else:
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"TensorBoard log dir: {tensorboard_dir}")

    if save_curves:
        os.makedirs(curve_dir, exist_ok=True)
        print(f"Curve output dir: {curve_dir}")

    try:
        for epoch in range(start_epoch, train_cfg["epochs"]):
            model.train()
            last_dist_params = None
            last_loss = None
            last_recon = None
            last_kl = None
            train_loss_sum = 0.0
            train_recon_sum = 0.0
            train_kl_sum = 0.0
            train_batches = 0

            for x_batch, t_batch, probe_ids, rev_ids, target_batch in train_loader:
                x_batch = x_batch.to(device)  # [B, L, 6]
                t_batch = t_batch.to(device)  # [B, L]
                probe_ids = probe_ids.to(device)  # [B, L]
                target_batch = target_batch.to(device)  # [B, L, 2]
                t_local = t_batch - t_batch[:, :1]

                optimizer.zero_grad()

                x_hat, dist_params = model(
                    x_batch,
                    t_local,
                    probe_ids=probe_ids,
                )

                loss, recon, kl = compute_harmonic_elbo(
                    x_target=target_batch,
                    x_hat=x_hat,
                    dist_params=dist_params,
                    beta=loss_cfg["beta"],
                    prior_a_w=prior_a_w,
                    use_kl_w=loss_cfg["use_kl_w"],
                )

                if not torch.isfinite(loss):
                    nonfinite_steps += 1
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=train_cfg["max_grad_norm"],
                )
                if torch.isfinite(grad_norm) and grad_norm > train_cfg["max_grad_norm"]:
                    grad_clip_triggered_steps += 1

                optimizer.step()
                total_steps += 1
                last_dist_params = dist_params
                last_loss = loss
                last_recon = recon
                last_kl = kl
                train_loss_sum += loss.item()
                train_recon_sum += recon.item()
                train_kl_sum += kl.item()
                train_batches += 1

                _log_scalar(writer, "train_step/loss", loss.item(), total_steps)
                _log_scalar(writer, "train_step/recon", recon.item(), total_steps)
                _log_scalar(writer, "train_step/kl", kl.item(), total_steps)
                if torch.isfinite(grad_norm):
                    _log_scalar(writer, "train_step/grad_norm", grad_norm.item(), total_steps)

            with torch.no_grad():
                if last_dist_params is None:
                    print(f"epoch={epoch:04d} no valid training step (all batches non-finite or dropped).")
                else:
                    mu_f, _ = last_dist_params
                    train_loss_mean = train_loss_sum / max(train_batches, 1)
                    train_recon_mean = train_recon_sum / max(train_batches, 1)
                    train_kl_mean = train_kl_sum / max(train_batches, 1)

                    _append_history(history, "train/loss", epoch + 1, train_loss_mean)
                    _append_history(history, "train/recon", epoch + 1, train_recon_mean)
                    _append_history(history, "train/kl", epoch + 1, train_kl_mean)
                    _log_scalar(writer, "train_epoch/loss", train_loss_mean, epoch + 1)
                    _log_scalar(writer, "train_epoch/recon", train_recon_mean, epoch + 1)
                    _log_scalar(writer, "train_epoch/kl", train_kl_mean, epoch + 1)
                    _log_scalar(writer, "train_epoch/f_mean", mu_f.mean().item(), epoch + 1)

                    print(
                        f"epoch={epoch:04d} "
                        f"train_loss={train_loss_mean:.6f} "
                        f"train_recon={train_recon_mean:.6f} "
                        f"train_kl={train_kl_mean:.6f} | "
                        f"f_mean={mu_f.mean().item():.4f}"
                    )
                    print("f_mean per harmonic:", mu_f.mean(dim=0).detach().cpu().numpy())
                    f_offset = mu_f - model.encoder.f_center[None, :]
                    f_ratio = f_offset / model.encoder.f_band
                    print("f_offset per harmonic:", f_offset.mean(dim=0).detach().cpu().numpy())
                    print("f_ratio per harmonic:", f_ratio.mean(dim=0).detach().cpu().numpy())

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
                    true_freqs_hz=signal_cfg["freqs_hz"],
                    true_amp_real=signal_cfg["amp_real_m"],
                    true_amp_imag=signal_cfg["amp_imag_m"],
                    amp_scale=amp_scale,
                    prior_a_w=prior_a_w,
                    loss_cfg=loss_cfg,
                    dense_factor=dense_factor,
                )

                if epoch_to_target is None and val_metrics["recon_btt_mse"] <= target_recon:
                    epoch_to_target = epoch + 1

                grad_clip_ratio = (
                    grad_clip_triggered_steps / max(total_steps, 1)
                )
                train_nan_or_inf_rate = nonfinite_steps / max(total_steps + nonfinite_steps, 1)

                for key, value in val_metrics.items():
                    if key == "nan_or_inf_rate":
                        _append_history(history, f"eval/{key}", epoch + 1, value)
                        _log_scalar(writer, f"eval/{key}", value, epoch + 1)
                        continue
                    _append_history(history, f"eval/{key}", epoch + 1, value)
                    _log_scalar(writer, f"eval/{key}", value, epoch + 1)

                _append_history(history, "train/grad_clip_ratio", epoch + 1, grad_clip_ratio)
                _append_history(history, "train/nan_or_inf_rate", epoch + 1, train_nan_or_inf_rate)
                _log_scalar(writer, "train/grad_clip_ratio", grad_clip_ratio, epoch + 1)
                _log_scalar(writer, "train/nan_or_inf_rate", train_nan_or_inf_rate, epoch + 1)

                print(
                    "[EVAL] "
                    f"epoch={epoch:04d} "
                    f"loss={val_metrics['loss']:.6f} "
                    f"recon_btt_mse={val_metrics['recon_btt_mse']:.6f} "
                    f"recon_btt_mse_det={val_metrics['recon_btt_mse_det']:.6f} "
                    f"recon_dense_mse={val_metrics['recon_dense_mse']:.6f} "
                    f"freq_mae_hz={val_metrics['freq_mae_hz']:.4f} "
                    f"complex_coeff_rel_err={val_metrics['complex_coeff_rel_err']:.4f} "
                    f"amp_mape={val_metrics['amp_mape']:.4f} "
                    f"phase_circ_mae_rad={val_metrics['phase_circ_mae_rad']:.4f} "
                    f"total_kl={val_metrics['total_kl']:.6f} "
                    f"kl_w={val_metrics['kl_w']:.6f} "
                    f"patch_freq_std_hz={val_metrics['patch_freq_std_hz']:.4f} "
                    f"harmonic_order_consistency={val_metrics['harmonic_order_consistency']:.4f} "
                    f"val_nan_or_inf_rate={val_metrics['nan_or_inf_rate']:.6f} "
                    f"train_nan_or_inf_rate={train_nan_or_inf_rate:.6f} "
                    f"grad_clip_ratio={grad_clip_ratio:.6f} "
                    f"epoch_to_target={epoch_to_target if epoch_to_target is not None else -1}"
                )

                if save_curves and ((epoch + 1) % curve_every == 0):
                    curves_saved = _save_training_curves(history, curve_dir)
                    if curves_saved:
                        print(f"Updated curve images in: {curve_dir}")

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
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
