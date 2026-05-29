import numpy as np
import matplotlib.pyplot as plt

# In this simplified simulator, actual sampling time equals ideal sampling time.


def simulate_fluctuating_speed_btt(
    n_revs=100,
    base_freq_x=150.0,
    delta=0.018,
    probe_angles=[0, 45, 100, 185],
    rng=None,
):
    """
    Generate non-uniform BTT sampling times under fluctuating rotational speed.

    Returns:
        t_samples:        Sampling times, shape [N]
        freqs_per_rev:    Rotation frequency per revolution, shape [n_revs]
        rev_ids:          Revolution index for each sample, shape [N]
        probe_ids:        Probe index for each sample, shape [N]
        theta_samples:    Probe angle (radians) for each sample, shape [N]
        freqs_at_samples: Rotation frequency at each sample, shape [N]
    """
    probe_angles_rad = np.radians(probe_angles)

    # 1) Generate per-revolution rotational frequency (Hz)
    rng = np.random.default_rng() if rng is None else rng
    fluctuations = rng.uniform(-delta, delta, n_revs)
    freqs_per_rev = base_freq_x * (1 + fluctuations)

    # 2) Build absolute start time for each revolution
    time_per_rev = 1.0 / freqs_per_rev
    rev_start_times = np.concatenate(([0], np.cumsum(time_per_rev[:-1])))

    # 3) Compute exact sampling time for every probe in every revolution
    t_samples = []
    rev_ids = []
    probe_ids = []
    theta_samples = []
    freqs_at_samples = []

    for r in range(n_revs):
        f_r = freqs_per_rev[r]
        t_start = rev_start_times[r]

        for probe_idx, theta in enumerate(probe_angles_rad):
            t_to_probe = (theta / (2 * np.pi)) * (1.0 / f_r)
            t_sample = t_start + t_to_probe

            t_samples.append(t_sample)
            rev_ids.append(r)
            probe_ids.append(probe_idx)
            theta_samples.append(theta)
            freqs_at_samples.append(f_r)

    return (
        np.array(t_samples),
        freqs_per_rev,
        np.array(rev_ids),
        np.array(probe_ids),
        np.array(theta_samples),
        np.array(freqs_at_samples),
    )


def generate_complex_harmonic_displacement(
    t,
    freqs,
    amp_real,
    amp_imag,
    snr_db=None,
    rng=None,
):
    """
    Generate complex blade displacement with a multi-harmonic model.

    Model:
        x_t = sum_k (a_k_real + j a_k_imag) * exp(j * 2pi * f_k * t)
    """
    # Initialize complex displacement
    x_t = np.zeros(len(t), dtype=complex)

    # Sum K harmonic components
    for f_k, a_real_k, a_imag_k in zip(freqs, amp_real, amp_imag):
        complex_amp_k = a_real_k + 1j * a_imag_k
        x_t += complex_amp_k * np.exp(1j * (2 * np.pi * f_k * t))

    # Add complex Gaussian white noise if requested
    if snr_db is not None:
        rng = np.random.default_rng() if rng is None else rng
        # Signal power
        sig_power = np.mean(np.abs(x_t) ** 2)
        # Noise power from target SNR
        noise_power = sig_power / (10 ** (snr_db / 10))

        # Complex Gaussian noise (real and imag each take half variance)
        noise_real = np.sqrt(noise_power / 2) * rng.standard_normal(len(t))
        noise_imag = np.sqrt(noise_power / 2) * rng.standard_normal(len(t))
        noise = noise_real + 1j * noise_imag

        x_t_noisy = x_t + noise
    else:
        x_t_noisy = x_t

    return x_t_noisy, x_t


def compute_frequency_support(freq_center_hz, relative_half_band):
    centers = np.asarray(freq_center_hz, dtype=np.float64)
    rel = np.asarray(relative_half_band, dtype=np.float64)

    if rel.ndim == 0:
        rel = np.full_like(centers, float(rel))

    if rel.shape != centers.shape:
        raise ValueError(
            f"relative_half_band shape {rel.shape} must be scalar or match centers {centers.shape}"
        )

    half_band = rel * centers
    lower = centers - half_band
    upper = centers + half_band

    if np.any(lower <= 0):
        raise ValueError("frequency lower bound must be positive")

    if np.any(upper[:-1] >= lower[1:]):
        raise ValueError(
            f"overlapping frequency supports: upper[:-1]={upper[:-1]}, "
            f"lower[1:]={lower[1:]}"
        )

    return lower, upper, centers, half_band


def sample_frequency_uniform(freq_lower, freq_upper, rng):
    freq_lower = np.asarray(freq_lower, dtype=np.float64)
    freq_upper = np.asarray(freq_upper, dtype=np.float64)
    return rng.uniform(freq_lower, freq_upper)


def sample_amplitude_uniform(
    amp_real_center,
    amp_imag_center,
    relative_half_band,
    min_half_band,
    rng,
):
    amp_real_center = np.asarray(amp_real_center, dtype=np.float64)
    amp_imag_center = np.asarray(amp_imag_center, dtype=np.float64)

    real_half = relative_half_band * np.maximum(
        np.abs(amp_real_center),
        min_half_band,
    )
    imag_half = relative_half_band * np.maximum(
        np.abs(amp_imag_center),
        min_half_band,
    )

    amp_real = rng.uniform(
        amp_real_center - real_half,
        amp_real_center + real_half,
    )
    amp_imag = rng.uniform(
        amp_imag_center - imag_half,
        amp_imag_center + imag_half,
    )

    return amp_real, amp_imag


def generate_one_btt_sequence(
    num_cycles,
    base_freq,
    fluctuation_delta,
    probe_angles,
    freq_hz,
    amp_real,
    amp_imag,
    snr_db,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng
    t_samples, freqs_per_rev, rev_ids, probe_ids, theta_samples, freqs_at_samples = (
        simulate_fluctuating_speed_btt(
            n_revs=num_cycles,
            base_freq_x=base_freq,
            delta=fluctuation_delta,
            probe_angles=probe_angles,
            rng=rng,
        )
    )

    x_observed, x_clean = generate_complex_harmonic_displacement(
        t=t_samples,
        freqs=freq_hz,
        amp_real=amp_real,
        amp_imag=amp_imag,
        snr_db=snr_db,
        rng=rng,
    )

    return {
        "x_observed": x_observed,
        "x_clean": x_clean,
        "t_samples": t_samples,
        "rev_ids": rev_ids,
        "probe_ids": probe_ids,
        "theta_samples": theta_samples,
        "freqs_at_samples": freqs_at_samples,
        "freq_hz": freq_hz,
        "amp_real": amp_real,
        "amp_imag": amp_imag,
    }


# ==========================================
# Example run and visualization
# ==========================================
if __name__ == "__main__":
    # 1) Rotation and sampling configuration
    base_x = 150.0  # Theoretical maximum detectable frequency is about 975 Hz
    fluctuation_delta = 0.001
    probes = [0, 28, 111.08, 166.15]  # Physical installation angles of probes
    n_revs = 200

    # Generate sampling times
    t_samples, freqs_per_rev, rev_ids, probe_ids, theta_samples, freqs_at_samples = simulate_fluctuating_speed_btt(
        n_revs=n_revs,
        base_freq_x=base_x,
        delta=fluctuation_delta,
        probe_angles=probes,
    )

    # 2) Harmonic displacement parameters
    # K = 4 components
    f_k = [167.0, 341.0, 635.0, 872.0]  # Frequency (Hz)
    amp_real_k = [0.0006, 0.0005403, -0.0003329, -0.0008910]
    amp_imag_k = [0.0, 0.0008415, 0.0007274, 0.0001270]
    SNR = 20  # Signal-to-noise ratio (dB)

    # Generate complex displacement signal x_t
    x_t_observed, x_t_true = generate_complex_harmonic_displacement(
        t=t_samples,
        freqs=f_k,
        amp_real=amp_real_k,
        amp_imag=amp_imag_k,
        snr_db=SNR,
    )

    # ==========================================
    # Visualization (for complex data, plot real part or magnitude)
    # ==========================================
    plt.figure(figsize=(12, 6))

    # Plot the real part of the first 40 samples (~10 revolutions)
    plot_points = 40
    plt.plot(
        t_samples[:plot_points],
        np.real(x_t_observed[:plot_points]),
        "ro",
        label="Measured Real(x_t) with Noise",
    )
    plt.plot(
        t_samples[:plot_points],
        np.real(x_t_true[:plot_points]),
        "b--",
        alpha=0.6,
        label="True Real(x_t)",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement Amplitude (Real Part)")
    plt.title("Synthetic BTT Data based on Complex Harmonic Model")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Generated {len(t_samples)} samples in total.")
    print(f"x_t data type: {x_t_observed.dtype}")
    print(f"Sample data [0]: {x_t_observed[0]:.4f}")
