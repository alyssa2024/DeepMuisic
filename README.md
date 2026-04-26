# DeepMuisic: Physics-Informed Self-Supervised VAE for BTT Signal Modeling

This repository is a PyTorch implementation of a Physics-Informed VAE (PI-VAE) for Blade Tip Timing (BTT) style signals.  
The model learns harmonic parameters (amplitude, angular frequency, phase) from irregularly sampled observations, then reconstructs displacement with a deterministic physics decoder.

## 1. Project Goal

- Input: irregular BTT sequences with probe/time/context features.
- Output: posterior parameters of harmonic latent variables and reconstructed complex displacement.
- Key idea: the decoder is fixed by physics, not learned as a generic neural network.

Physical model:

\[
x_t=\sum_{k=1}^{K}\alpha_k \exp\left(j(\omega_k t+\phi_k)\right)
\]

## 2. Method-to-Code Mapping

- Encoder (variational Transformer): `Encoder.py`
  - Class: `VariationalIndependentTimeSeriesTransformer`
  - Outputs:
    - `q(A_k|x)`: `mu_A`, `logvar_A`
    - `q(w_k|x)`: `mu_w`, `logvar_w`
    - `q(phi_k|x)`: `mu_phi`, `kappa_phi`

- Decoder (physics-based, parameter-free): `VAE.py`
  - Class: `PhysicalHarmonicVAE`
  - `decode()` directly applies the multi-harmonic equation to generate complex `x_hat`.

- Loss (negative ELBO): `loss.py`
  - Function: `compute_harmonic_elbo()`
  - Terms:
    - Reconstruction loss (complex MSE over real/imag channels)
    - KL regularization:
      - Gaussian KL for amplitude
      - Gaussian-to-Maxwell KL for frequency
      - von Mises KL for phase

- Data simulation and patching: `synthesis_dataset.py`, `dataset.py`
  - Synthetic irregular BTT data generation
  - Patch dataset building for training windows

## 3. Repository Structure

```text
DeepMuisic/
|- main.py                  # Training entry: synthesize data, build dataloader, train model
|- Encoder.py               # Variational Transformer encoder
|- VAE.py                   # Reparameterization + physics decoder
|- loss.py                  # ELBO and KL utilities
|- dataset.py               # Feature builder and patch dataset
|- synthesis_dataset.py     # Main synthetic data pipeline
|- synthesis_data_deltaT.py # Alternative delta-time synthetic script
|- test_totation_speed.py   # Speed fluctuation visualization/test script
`- notes.txt                # Research notes
```

## 4. Quick Start

### 4.1 Dependencies

Recommended: Python 3.10+

Install:

```bash
pip install torch numpy matplotlib gin-config
```

### 4.2 Run Training

```bash
python main.py
```

Current hyperparameters are set directly in `main.py` (for example `num_harmonics`, `epochs`, window settings, and synthetic signal frequencies).

## 5. Feature Design

`build_btt_point_features()` currently builds 6-D token features:

1. `x_real`
2. `x_imag`
3. `sin(theta)`
4. `cos(theta)`
5. `rev_norm`
6. `speed_norm`

`BTTPatchDataset` returns:

- `x_feat`: `[L, 6]`
- `t`: `[L]`
- `probe_id`: `[L]`
- `rev_id`: `[L]`
- `target`: `[L, 2]` (real/imag target for reconstruction)


