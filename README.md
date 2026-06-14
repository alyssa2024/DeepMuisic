# DeepMuisic: Physics-Informed VAE for BTT Harmonic Inference

This repository implements a physics-informed variational model for Blade Tip Timing (BTT) signals in PyTorch.
The model infers a posterior over harmonic frequencies from irregular samples, estimates complex amplitudes with closed-form least-squares / MAP-style solvers, and reconstructs the complex displacement with a deterministic physical decoder.

Physical model:

\[
x_t = \sum_{k=1}^{K}(a_{k,\mathrm{real}} + j a_{k,\mathrm{imag}})\exp\left(j 2\pi f_k t\right)
\]

## 1. Repository Structure

```text
DeepMuisic-v5/
|- README.md                         # Project overview and usage notes
|- config.py                         # Centralized experiment, data, model, loss, logging, and checkpoint config
|- main.py                           # Main training/evaluation entry point
|- batch_utils.py                    # Dataset-state extraction helpers used by training, eval, and loss code
|- synthesis_dataset.py              # Synthetic BTT sampling, harmonic signal generation, and prior sampling utilities
|- dataset.py                        # Sequence/window dataset builders, normalization, feature construction, splits
|- Encoder.py                        # Variational Transformer encoder and posterior heads
|- VAE.py                            # Physical harmonic VAE wrapper and deterministic decoder
|- loss.py                           # Static/global objectives, reconstruction losses, priors, and KL utilities
|- eval.py                           # Validation metrics and model evaluation loop
|- diagnose_checkpoint.py            # Checkpoint diagnostic script for sequence-level posterior behavior
|- plot_results.py                   # Sweep-result collection and plotting utility
|- run_grid.py                       # General experiment-grid runner
|- run_noncenter_rho_sweep.py        # Non-center rho sweep runner
|- run_snr_sweep.py                  # SNR sweep runner
|- run_stage1_sweep.py               # Stage-1 sweep runner
|- run_time_input_pe_sweep.py        # Time-input / positional-encoding sweep runner
|- colab_experiments.ipynb           # Colab experiment notebook
|- eval.ipynb                        # Evaluation notebook
|- .vscode/                          # Local VS Code settings
`- __pycache__/                      # Generated Python bytecode cache
```

Runtime outputs are created on demand and are not present in a fresh checkout:

- `checkpoints/` for model checkpoints, controlled by `checkpoint.dir`
- `artifacts/tensorboard/` for TensorBoard logs when `logging.enable_tensorboard=True`
- `artifacts/curves/` for learning-curve PNGs when `logging.save_curves=True`

## 2. Method Overview

- `Encoder.py`
  - `VariationalIndependentTimeSeriesTransformer`
  - Encodes BTT sequence/window tokens and outputs bounded frequency posterior parameters:
    - `mu_f`
    - `logvar_f`
    - `std_f`
    - `log_rho2_f`
- `VAE.py`
  - `PhysicalHarmonicVAE`
  - Aggregates window-level frequency posteriors, solves complex amplitudes, and decodes with the fixed harmonic equation (complex-valued output).
- `loss.py`
  - `compute_static_global_objective()`
  - Sequence posterior reconstruction loss, optional amplitude prior terms, and frequency KL utilities.
- `eval.py`
  - `evaluate_model()`
  - Reports reconstruction and parameter-error metrics.

This is a sequence/window-level PI-VAE with static/global harmonic parameters, not a full dynamical state-transition VAE.

## 3. Data Pipeline

- `synthesis_dataset.py`
  - Generates fluctuating-speed BTT sampling times.
  - Synthesizes multi-harmonic complex displacement with optional complex Gaussian noise.
- `dataset.py`
  - Builds 6-D or 7-D point features:
    1. `x_real`
    2. `x_imag`
    3. `sin(theta)`
    4. `cos(theta)`
    5. `rev_norm`
    6. `speed_norm`
    7. `local_time_norm` when `data.include_local_time_norm=True`
  - Creates sequence/window datasets (`BTTSequenceDataset`, `GroupedBTTSequenceDataset`).
  - Supports chronological train/val/test slicing for long-sequence experiments.

## 4. Configuration

All experiment settings are centralized in `config.py`, including:

- frequency model metadata (`frequency_model`)
- data generation (`data`, `signal`)
- frequency supports and posterior settings (`frequency`)
- model size (`model`)
- training hyperparameters (`training`)
- evaluation settings (`eval`)
- loss priors/weights (`loss`)
- checkpoint settings (`checkpoint`)
- logging settings (`logging`)
- sweep values (`experiment`)

## 5. Quick Start

### 5.1 Requirements

Core training and evaluation use:

- `torch`
- `numpy`

Optional utilities use:

- `matplotlib` for synthetic-data visualization and saved training curves
- `pandas` for `plot_results.py`
- TensorBoard support through `torch.utils.tensorboard`

`Encoder.py` can use `gin` when it is installed, but it also includes a fallback so `gin` is not required for the default scripts.

### 5.2 Run Training + Eval

```bash
python main.py
```

`main.py` will:

1. synthesize data,
2. build sequence/window datasets,
3. split train/val/test chronologically when `data.train_dataset.chronological_split=True`,
4. train the PI-VAE,
5. run periodic validation via `evaluate_model()`,
6. save metrics and checkpoints.

Training also writes:

- validation metrics to `metrics.json` in `run_dir` (the project root by default)
- TensorBoard scalars to `artifacts/tensorboard` when `logging.enable_tensorboard=True`
- PNG learning curves to `artifacts/curves` when `logging.save_curves=True`
- checkpoints to `checkpoints/`, including `latest.pt`, periodic `epoch_*.pt`, and best/final-best checkpoints when available

If TensorBoard is available in your environment, you can inspect logs with:

```bash
tensorboard --logdir artifacts/tensorboard
```

### 5.3 Optional: Visualize Synthetic Data

```bash
python synthesis_dataset.py
```

### 5.4 Optional: Run Sweeps and Plot Results

The repository includes several sweep runners for common experiments:

```bash
python run_snr_sweep.py
python run_grid.py --help
python run_stage1_sweep.py --help
python run_noncenter_rho_sweep.py
python run_time_input_pe_sweep.py
```

`run_snr_sweep.py`, `run_noncenter_rho_sweep.py`, and `run_time_input_pe_sweep.py` use fixed settings in the script and start running immediately.

After a sweep has written `metrics.json` files, use:

```bash
python plot_results.py --help
```

### 5.5 Optional: Diagnose a Checkpoint

```bash
python diagnose_checkpoint.py --checkpoint checkpoints/latest.pt
```

## 6. Main Eval Metrics

`eval.py` reports metrics such as:

- `recon_mse_mean`
- `recon_mse_sampled`
- `recon_nll_sampled`
- `freq_rmse_hz_mean`
- `freq_mae_hz_mean`
- `freq_nrmse_band_mean`
- `freq_success_rate_mean`
- `amp_mape_mean`
- `complex_coeff_rel_err_mean`
- `complex_coeff_success_rate`
- `phase_circ_mae_rad`
- `posterior_std_hz_mean`
- `ls_cond_p95`
- `harmonic_order_consistency`

## 7. Notes

- The current formulation assumes sequence/window-level quasi-stationary harmonic parameters.
- For strongly time-varying transients, further sequential/dynamical latent modeling may be needed.
