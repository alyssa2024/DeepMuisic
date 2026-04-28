# DeepMuisic: Physics-Informed VAE for BTT Harmonic Inference

This repository implements a physics-informed variational model for Blade Tip Timing (BTT) signals in PyTorch.
The model infers harmonic parameters (amplitude, angular frequency, phase) from irregular samples and reconstructs the complex displacement with a deterministic physical decoder.

Physical model:

\[
x_t = \sum_{k=1}^{K}\alpha_k \exp\left(j(\omega_k t+\phi_k)\right)
\]

## 1. Repository Structure

```text
DeepMuisic/
|- main.py              # Training entry: synthesize data, build dataloader, train model
|- config.py            # Centralized experiment configuration
|- synthesis_dataset.py # Synthetic BTT sampling + harmonic signal generation
|- dataset.py           # Feature builder, patch dataset, chronological train/val split
|- Encoder.py           # Variational Transformer encoder
|- VAE.py               # Reparameterization + physics decoder
|- loss.py              # ELBO and KL utilities
|- eval.py              # Validation metrics and model evaluation
`- README.md
```

## 2. Method Overview

- `Encoder.py`
  - `VariationalIndependentTimeSeriesTransformer`
  - Encodes a patch sequence and outputs posterior parameters:
    - `q(A|x): mu_A, logvar_A`
    - `q(w|x): mu_w, logvar_w`
    - `q(phi|x): mu_phi, kappa_phi`
- `VAE.py`
  - `PhysicalHarmonicVAE`
  - Reparameterizes latent variables and decodes with the fixed harmonic equation (complex-valued output).
  - Amplitude is mapped to non-negative values (`softplus`).
- `loss.py`
  - `compute_harmonic_elbo()`
  - Reconstruction term (complex MSE over real/imag channels) + KL regularization terms.
- `eval.py`
  - `evaluate_model()`
  - Reports reconstruction and parameter-error metrics.

This is a patch-level PI-VAE (constant latent per patch), not a full dynamical state-transition VAE.

## 3. Data Pipeline

- `synthesis_dataset.py`
  - Generates fluctuating-speed BTT sampling times.
  - Synthesizes multi-harmonic complex displacement with optional complex Gaussian noise.
- `dataset.py`
  - Builds 6-D point features:
    1. `x_real`
    2. `x_imag`
    3. `sin(theta)`
    4. `cos(theta)`
    5. `rev_norm`
    6. `speed_norm`
  - Creates patch samples (`BTTPatchDataset`).
  - Splits train/val chronologically (`chronological_train_val_split`) with a guard gap to reduce overlap leakage across split boundary.

## 4. Configuration

All experiment settings are centralized in `config.py`, including:

- data generation (`data`, `signal`)
- model size (`model`)
- training hyperparameters (`training`)
- evaluation settings (`eval`)
- loss priors/weights (`loss`)
- checkpoint settings (`checkpoint`)

## 5. Quick Start

### 5.1 Requirements

This project is designed to run in the default Google Colab environment.
In most cases, no extra setup is needed before running `main.py`.

If Colab reports a missing package (for example `gin`), install only the missing one and rerun.

### 5.2 Run Training + Eval

```bash
python main.py
```

`main.py` will:

1. synthesize data,
2. build patch datasets,
3. split train/val chronologically,
4. train the PI-VAE,
5. run periodic validation via `evaluate_model()`,
6. save checkpoints.

### 5.3 Optional: Visualize Synthetic Data

```bash
python synthesis_dataset.py
```

## 6. Main Eval Metrics

`eval.py` reports metrics such as:

- `recon_btt_mse`
- `recon_btt_mse_det`
- `recon_dense_mse`
- `freq_mae_hz`
- `amp_mape`
- `phase_circ_mae_rad`
- `patch_freq_std_hz`
- `harmonic_order_consistency`

## 7. Notes

- The current formulation assumes patch-local quasi-stationary harmonic parameters.
- For strongly time-varying transients, further sequential/dynamical latent modeling may be needed.
