# DeepMUSIC Sequential-DSAE

This repository implements a **DSAE-style physics-informed sequential VAE** for Blade Tip Timing (BTT) harmonic inference.

The model separates the latent representation of a BTT sequence into:

- **static/global latent variables**: frequency `f` and complex amplitude `c`
- **dynamic latent variables**: phase states `z_1:B` along ordered patches
- **fixed physics decoder**: a BTT sinusoidal observation model

The implementation is intentionally close to the Disentangled Sequential Autoencoder style: the sequence is read by recurrent encoders, global variables are inferred from BiRNN endpoint states, and dynamic variables are inferred recurrently. **Mean, max, attention, and window pooling over time are not used in the main path.**

---

## 1. Physical Model

For patch `b` and local sample `p`, the decoder is fixed as:

```math
\hat y_{b,p}
=
\sum_{k=1}^{K}
c_k
\exp
\left(
j\left[z_{k,b}+2\pi f_k\tau_{b,p}\right]
\right)
```

where:

- `f_k` is the global frequency of harmonic `k`
- `c_k` is the global complex amplitude of harmonic `k`
- `z_{k,b}` is the phase state of harmonic `k` at the start of patch `b`
- `tau_{b,p}` is local time inside patch `b`

The phase state is modeled as an **unwrapped Gaussian phase state**. It lives on
the real line during inference and transition, and only the decoder phase is
reduced modulo `2*pi` for numerical stability.

The phase state evolves as:

```math
z_b=z_{b-1}+2\pi f\Delta s_b+\delta z_b
```

The innovation `delta z_b` absorbs timestamp error, speed fluctuation, probe error, and other non-ideal effects.

---

## 2. Network Architecture

### 2.1 Local Encoder

Each BTT parent sequence is split into ordered patches. With the default configuration:

```text
1000 parameter sets
5 sequences per parameter set
32 cycles per sequence
8 cycles per patch
4 ordered patches per sequence
4 probes per cycle
32 samples per patch
```

The batch shape is:

```text
features: [batch, B, P, d_in]
y:        [batch, B, P, 2]
tau:      [batch, B, P]
delta_s:  [batch, B]
```

The local encoder maps each patch to a local feature:

```text
features[:, b] -> e_b
```

Patch points are encoded with an MLP and a bidirectional GRU endpoint summary:

```text
point features
-> MLP
-> BiGRU
-> concat(forward_last, backward_first)
-> e_b
```

No pooling is used.

### 2.2 Context Encoder

Patch features form an ordered sequence:

```text
e_1:B = [e_1, ..., e_B]
```

The context encoder reads the full sequence:

```text
e_1:B
-> BiGRU
-> h_bar_1:B
```

The global sequence representation is DSAE-style endpoint summary:

```text
h_global = concat(forward_last, backward_first)
```

This replaces attention/global/window pooling.

### 2.3 Approximate Posterior for Static Latents

The global frequency posterior is inferred from `h_global`:

```math
q(f\mid Y)
```

Current implementation uses a band-constrained truncated-normal style parameterization:

```text
h_global -> mu_f, std_f
```

The global complex amplitude posterior is also inferred from `h_global`:

```math
q(c\mid Y)
```

implemented with real/imaginary Gaussian parameters:

```text
h_global -> c_mu_real, c_mu_imag, c_logvar_real, c_logvar_imag
```

### 2.4 Approximate Posterior for Dynamic Latents

The initial phase posterior is:

```math
q(z_1\mid f,Y)
```

For later patches, the model infers phase innovations:

```math
q(\delta z_b\mid z_{b-1}, f, Y)
```

The recurrent inference path is:

```text
for b = 2,...,B:
    GRU_z([h_bar_b, f, z_{b-1}, delta_s_b])
        -> delta_mu_b, delta_logvar_b

    delta_z_b = reparameterize(delta_mu_b, delta_logvar_b)
    z_b = z_{b-1} + 2*pi*f*delta_s_b + delta_z_b
```

The model never directly predicts a free independent `z_b`.

---

## 3. Priors

The generative prior is:

```math
p(f,c,z_1,\delta z_{2:B})
=
p(f)p(c)p(z_1)\prod_{b=2}^{B}p(\delta z_b)
```

Default priors:

```text
p(f): frequency band prior
p(c): diagonal complex Gaussian
p(z_1): unwrapped diagonal Gaussian
p(delta_z_b): unwrapped diagonal Gaussian physics innovation prior
```

Unlike the original image/video DSAE, the dynamic prior is not a learned LSTM prior. The BTT model already has an explicit physical transition:

```math
z_b=z_{b-1}+2\pi f\Delta s_b+\delta z_b
```

`z` could also be modeled with a Von Mises posterior, but that requires circular
reparameterization and circular KL terms. The current implementation uses the
unwrapped Gaussian variant.

---

## 4. Training Objective

The model minimizes the negative ELBO:

```math
\mathcal J
=
-\mathbb E_q[\log p(Y\mid f,c,z_{1:B})]
+
KL[q(f\mid Y)\|p(f)]
+
KL[q(c\mid Y)\|p(c)]
+
KL[q(z_1\mid f,Y)\|p(z_1)]
+
\sum_{b=2}^{B}
KL[q(\delta z_b\mid z_{b-1},f,Y)\|p(\delta z_b)]
```

The reconstruction term is complex Gaussian NLL:

```math
p(Y\mid f,c,z)
=
\prod_{b,p}
\mathcal{CN}(y_{b,p};\hat y_{b,p},\sigma_y^2)
```

---

## 5. Repository Structure

```text
DeepMuisic-v6/
|- main.py              # Sequential-DSAE training entry
|- config.py            # Experiment and model configuration
|- synthesis_dataset.py # Synthetic BTT sampling and harmonic signal generation
|- dataset.py           # Sequential patch dataset and legacy datasets
|- Encoder.py           # SequentialDSAEEncoder plus legacy encoder
|- VAE.py               # Sequential physics VAE plus legacy utilities
|- loss.py              # Sequential ELBO and KL utilities
|- eval.py              # Sequential and legacy evaluation helpers
|- plot_results.py      # Plotting helpers
|- docs/                # Refactor plans and architecture notes
`- README.md
```

---

## 6. Data Flow

```text
BTTSequentialPatchDataset
    -> features, y, tau, s, delta_s

SequentialDSAEEncoder
    -> LocalEncoder
    -> ContextEncoder
    -> q(f | Y), q(c | Y)
    -> q(z_1 | f,Y)
    -> q(delta_z_b | z_{b-1}, f,Y)
    -> z_seq

SequentialPhysicalHarmonicVAE
    -> fixed BTT decoder
    -> y_hat

compute_sequential_dsae_elbo
    -> reconstruction NLL
    -> KL_f + KL_c + KL_z1 + KL_delta
```

---

## 7. Quick Start

Run training:

```bash
python main.py
```

Training will:

1. synthesize multi-parameter BTT sequences,
2. build ordered patch-sequence batches,
3. train `SequentialPhysicalHarmonicVAE`,
4. evaluate on validation data,
5. save checkpoints and metrics.

Default outputs:

```text
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/metrics.json
```

TensorBoard logging is controlled by `logging.enable_tensorboard` in `config.py`.

---

## 8. Important Configuration

Main data settings in `config.py`:

```text
num_param_sets = 1000
sequences_per_param = 5
long_sequence_num_cycles = 32
sequence_num_cycles = 8
```

Main model setting:

```text
model.variant = sequential_dsae
model.endpoint_summary = forward_last_backward_first
```

The endpoint summary setting is intentional: pooling over time is disabled to preserve sequence information.

Main phase-latent setting:

```text
loss.sequential.phase_distribution = unwrapped_gaussian
loss.sequential.z1_prior_var = (2*pi)^2
loss.sequential.delta_prior_var = 0.01
```

---

## 9. Main Metrics

Sequential evaluation reports:

- reconstruction NLL
- reconstruction MSE
- frequency MAE
- frequency KL
- amplitude KL
- initial phase KL
- phase innovation KL
- phase transition residual
- posterior standard deviations

The transition residual is:

```math
z_b-z_{b-1}-2\pi f\Delta s_b
```

and is useful even when ground-truth phase is unavailable.

---

## 10. Notes

- `PhysicalHarmonicVAE`, `VariationalIndependentTimeSeriesTransformer`, and static/global objectives remain in the repository as legacy utilities/reference paths.
- The main training path is `BTTSequentialPatchDataset -> SequentialDSAEEncoder -> SequentialPhysicalHarmonicVAE -> compute_sequential_dsae_elbo`.
- The current frequency posterior is single-band per harmonic. A future extension can add alias-band mixture posteriors.
