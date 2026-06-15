import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import BTTSequentialPatchDataset
from Encoder import SequentialDSAEEncoder
from loss import compute_sequential_dsae_elbo
from synthesis_dataset import compute_frequency_support
from VAE import SequentialPhysicalHarmonicVAE

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def save_checkpoint(path, model, optimizer, epoch, total_steps):
    torch.save(
        {
            "epoch": int(epoch),
            "total_steps": int(total_steps),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def _log_scalar(writer, tag, value, step):
    if writer is not None:
        writer.add_scalar(tag, float(value), int(step))


def _resolve_lr_schedule(train_cfg, steps_per_epoch):
    schedule_cfg = train_cfg.get("lr_schedule", {})
    schedule_type = schedule_cfg.get("type", "constant")
    total_steps = int(schedule_cfg.get("total_steps", train_cfg["epochs"] * steps_per_epoch))
    warmup_steps = int(schedule_cfg.get("warmup_steps", 0))
    min_lr = float(schedule_cfg.get("min_lr", 0.0))
    if schedule_type not in ("constant", "warmup_cosine"):
        raise ValueError(f"Unsupported training.lr_schedule.type={schedule_type}")
    return schedule_type, total_steps, warmup_steps, min_lr


def _compute_learning_rate(base_lr, step, schedule_type, total_steps, warmup_steps, min_lr):
    if schedule_type == "constant":
        return float(base_lr)
    step = min(max(int(step), 1), int(total_steps))
    if warmup_steps > 0 and step <= warmup_steps:
        return float(base_lr) * step / warmup_steps
    cosine_steps = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / cosine_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (float(base_lr) - min_lr) * cosine_decay


def build_dataset(dataset_cfg, split, seed, data_cfg, signal_cfg, freq_lower, freq_upper):
    amp_prior_cfg = signal_cfg["amp_data_prior"]
    use_fixed_amp_rho = amp_prior_cfg.get("type") == "fixed_rho"
    amp_sampling = "polar" if amp_prior_cfg.get("type") == "uniform_polar" else "cartesian"
    return BTTSequentialPatchDataset(
        split=split,
        num_param_sets=dataset_cfg["num_param_sets"],
        sequences_per_param=dataset_cfg["sequences_per_param"],
        sequence_num_cycles=dataset_cfg["long_sequence_num_cycles"],
        patch_num_cycles=dataset_cfg["sequence_num_cycles"],
        num_probes=data_cfg["num_probes"],
        base_freq=data_cfg["base_freq"],
        fluctuation_delta=data_cfg["fluctuation_delta"],
        probe_angles=data_cfg["probes"],
        freq_lower=freq_lower,
        freq_upper=freq_upper,
        amp_real_center=signal_cfg["amp_real_center_m"],
        amp_imag_center=signal_cfg["amp_imag_center_m"],
        amp_relative_half_band=amp_prior_cfg.get("relative_half_band", 0.2),
        amp_min_half_band=amp_prior_cfg.get("min_half_band_m", 1e-5),
        snr_db=signal_cfg["snr_db"],
        seed=seed,
        normalization=data_cfg.get("normalization", "per_sequence_std"),
        frequency_rho=None,
        amp_real_rho=signal_cfg.get("rho_amp_real_k") if use_fixed_amp_rho else None,
        amp_imag_rho=signal_cfg.get("rho_amp_imag_k") if use_fixed_amp_rho else None,
        amp_sampling=amp_sampling,
        amp_magnitude_min=amp_prior_cfg.get("magnitude_min", 0.0),
        amp_magnitude_max=amp_prior_cfg.get("magnitude_max", 1.0),
        amp_phase_min=amp_prior_cfg.get("phase_min", 0.0),
        amp_phase_max=amp_prior_cfg.get("phase_max", 2.0 * math.pi),
    )


def build_model(data_cfg, model_cfg, freq_cfg, freq_lower, freq_upper, device):
    posterior_cfg = freq_cfg.get("posterior", {})
    encoder = SequentialDSAEEncoder(
        input_dim=data_cfg["input_dim"],
        output_dim=data_cfg["num_harmonics"],
        local_hidden_dim=model_cfg.get("local_hidden_dim", 128),
        local_out_dim=model_cfg.get("local_out_dim", 128),
        context_hidden_dim=model_cfg.get("context_hidden_dim", 256),
        context_layers=model_cfg.get("context_layers", 1),
        innovation_hidden_dim=model_cfg.get("innovation_hidden_dim", 256),
        dropout=model_cfg.get("dropout", 0.0),
        freq_lower_hz=freq_lower,
        freq_upper_hz=freq_upper,
        min_log_rho2=posterior_cfg.get("min_log_rho2", -12.0),
        max_log_rho2=posterior_cfg.get("max_log_rho2", -4.0),
        c_init_logvar=model_cfg.get("c_init_logvar", -4.0),
        z1_init_logvar=model_cfg.get("z1_init_logvar", 0.0),
        delta_init_logvar=model_cfg.get("delta_init_logvar", -6.0),
    )
    return SequentialPhysicalHarmonicVAE(encoder=encoder).to(device)


@torch.no_grad()
def evaluate(model, loader, loss_cfg, device, max_batches=None):
    model.eval()
    sums = {}
    batches = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, sample=False)
        loss, _, _, diagnostics = compute_sequential_dsae_elbo(
            batch=batch,
            outputs=outputs,
            model=model,
            loss_cfg=loss_cfg,
            global_step=None,
        )
        freq_mae = (outputs["mu_f"] - batch["true_freq_hz"]).abs().mean()
        diagnostics = dict(diagnostics)
        diagnostics["freq_mae_hz"] = freq_mae.detach()
        diagnostics["loss"] = loss.detach()
        for key, value in diagnostics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
        if max_batches is not None and batches >= int(max_batches):
            break
    denom = max(batches, 1)
    return {key: value / denom for key, value in sums.items()}


def main():
    data_cfg = CONFIG["data"]
    signal_cfg = CONFIG["signal"]
    freq_cfg = CONFIG["frequency"]
    model_cfg = CONFIG["model"]
    train_cfg = CONFIG["training"]
    loss_cfg = dict(CONFIG["loss"])
    loss_cfg["prior"] = dict(freq_cfg.get("loss_prior", {}))
    loss_cfg["sequential"] = dict(loss_cfg.get("sequential", {}))
    seed = int(CONFIG.get("seed", 42))
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if model_cfg.get("endpoint_summary") != "forward_last_backward_first":
        raise ValueError("Only DSAE endpoint summary is supported; pooling is disabled.")

    freq_lower, freq_upper, freq_center, freq_half_band = compute_frequency_support(
        freq_center_hz=freq_cfg["center_hz"],
        relative_half_band=freq_cfg["relative_half_band"],
    )
    print(f"Frequency centers: {freq_center}")
    print(f"Frequency half bands: {freq_half_band}")

    train_set = build_dataset(
        dataset_cfg=data_cfg["train_dataset"],
        split="train",
        seed=seed,
        data_cfg=data_cfg,
        signal_cfg=signal_cfg,
        freq_lower=freq_lower,
        freq_upper=freq_upper,
    )
    val_set = build_dataset(
        dataset_cfg=data_cfg.get("val_dataset", data_cfg["train_dataset"]),
        split="val",
        seed=seed + 100000,
        data_cfg=data_cfg,
        signal_cfg=signal_cfg,
        freq_lower=freq_lower,
        freq_upper=freq_upper,
    )
    test_cfg = data_cfg.get("test_dataset")
    test_set = (
        build_dataset(
            dataset_cfg=test_cfg,
            split="test",
            seed=seed + 200000,
            data_cfg=data_cfg,
            signal_cfg=signal_cfg,
            freq_lower=freq_lower,
            freq_upper=freq_upper,
        )
        if test_cfg is not None
        else None
    )

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
    test_loader = (
        DataLoader(test_set, batch_size=data_cfg["batch_size"], shuffle=False)
        if test_set is not None
        else None
    )

    num_patches = (
        data_cfg["train_dataset"]["long_sequence_num_cycles"]
        // data_cfg["train_dataset"]["sequence_num_cycles"]
    )
    points_per_patch = data_cfg["train_dataset"]["sequence_num_cycles"] * data_cfg["num_probes"]
    print(
        "Datasets: "
        f"train={len(train_set)}, val={len(val_set)}, "
        f"test={len(test_set) if test_set is not None else 0}, "
        f"patches_per_sequence={num_patches}, points_per_patch={points_per_patch}"
    )

    model = build_model(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        freq_cfg=freq_cfg,
        freq_lower=freq_lower,
        freq_upper=freq_upper,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    lr_schedule_type, lr_total_steps, lr_warmup_steps, lr_min = _resolve_lr_schedule(
        train_cfg,
        steps_per_epoch=len(train_loader),
    )

    ckpt_cfg = CONFIG.get("checkpoint", {})
    ckpt_dir = ckpt_cfg.get("dir", "checkpoints")
    ckpt_name = ckpt_cfg.get("name", "latest.pt")
    ckpt_save_every = int(ckpt_cfg.get("save_every", 20))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    writer = None
    log_cfg = CONFIG.get("logging", {})
    if log_cfg.get("enable_tensorboard", True) and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=log_cfg.get("tensorboard_dir", "artifacts/tensorboard"))

    total_steps = 0
    best_val = float("inf")
    best_metrics = None
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        train_sums = {}
        train_batches = 0
        for batch in train_loader:
            total_steps += 1
            batch = move_batch_to_device(batch, device)
            step_lr = _compute_learning_rate(
                base_lr=train_cfg["lr"],
                step=total_steps,
                schedule_type=lr_schedule_type,
                total_steps=lr_total_steps,
                warmup_steps=lr_warmup_steps,
                min_lr=lr_min,
            )
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            optimizer.zero_grad()
            outputs = model(batch, sample=True)
            loss, _, _, diagnostics = compute_sequential_dsae_elbo(
                batch=batch,
                outputs=outputs,
                model=model,
                loss_cfg=loss_cfg,
                global_step=total_steps,
            )
            if not torch.isfinite(loss):
                print(f"Skipping non-finite loss at step {total_steps}")
                continue
            loss.backward()
            grad_cfg = train_cfg.get("grad_clip", {})
            if grad_cfg.get("enabled", False):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(grad_cfg.get("max_norm", 1.0)),
                )
            optimizer.step()

            for key, value in diagnostics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    train_sums[key] = train_sums.get(key, 0.0) + float(value.detach().cpu())
            train_batches += 1

        train_metrics = {
            key: value / max(train_batches, 1)
            for key, value in train_sums.items()
        }
        val_metrics = evaluate(model, val_loader, loss_cfg, device)
        val_loss = val_metrics.get("loss", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            best_metrics = dict(val_metrics)
            save_checkpoint(
                path=os.path.join(ckpt_dir, "best.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                total_steps=total_steps,
            )

        for key, value in train_metrics.items():
            _log_scalar(writer, f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            _log_scalar(writer, f"val/{key}", value, epoch)

        print(
            f"Epoch {epoch + 1:04d}: "
            f"train_loss={train_metrics.get('loss', float('nan')):.6g}, "
            f"val_loss={val_metrics.get('loss', float('nan')):.6g}, "
            f"val_freq_mae_hz={val_metrics.get('freq_mae_hz', float('nan')):.6g}"
        )

        if (epoch + 1) % ckpt_save_every == 0:
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                total_steps=total_steps,
            )

    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg["epochs"]) - 1,
        total_steps=total_steps,
    )

    final_payload = {"best_val": best_val, "best_metrics": best_metrics}
    if test_loader is not None:
        final_payload["test_metrics"] = evaluate(model, test_loader, loss_cfg, device)
    metrics_path = os.path.join(ckpt_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved metrics to {metrics_path}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
