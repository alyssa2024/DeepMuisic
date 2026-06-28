import numpy as np


def harmonic_rad(t, omega_rad_s, amplitude=2.0, phase=0.0):
    return amplitude * np.cos(float(omega_rad_s) * np.asarray(t) + float(phase))


def linear_chirp_hz(t, f0_hz, f1_hz, t0=None, t1=None, amplitude=1.0, phase=0.0):
    t = np.asarray(t, dtype=np.float64)
    if t0 is None:
        t0 = float(t.min())
    if t1 is None:
        t1 = float(t.max())
    duration = max(float(t1 - t0), 1e-12)
    slope = (float(f1_hz) - float(f0_hz)) / duration
    tau = t - float(t0)
    phase_cycles = float(f0_hz) * tau + 0.5 * slope * tau * tau
    y = amplitude * np.cos(2.0 * np.pi * phase_cycles + phase)
    f_true = float(f0_hz) + slope * tau
    return y, f_true


def rom_harmonic_hz(events, a0_hz, a1_per_rev, amplitude=1.0, phase=0.0):
    t = np.asarray(events["t"], dtype=np.float64)
    theta = np.asarray(events["theta"], dtype=np.float64)
    phase_cycles = float(a0_hz) * t + float(a1_per_rev) * theta / (2.0 * np.pi)
    y = amplitude * np.cos(2.0 * np.pi * phase_cycles + phase)
    f_true = float(a0_hz) + float(a1_per_rev) * np.asarray(events["rot_hz"], dtype=np.float64)
    return y, f_true


def add_noise(y_clean, snr_db, rng):
    y_clean = np.asarray(y_clean, dtype=np.float64)
    signal_power = float(np.mean(y_clean**2))
    if signal_power <= 0.0:
        return y_clean.copy(), 0.0
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    return y_clean + np.sqrt(noise_power) * rng.standard_normal(y_clean.shape), noise_power

