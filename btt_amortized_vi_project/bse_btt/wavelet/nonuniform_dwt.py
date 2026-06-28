import numpy as np


def harmonic_matching(t, y, omega_grid):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    omega_grid = np.asarray(omega_grid, dtype=np.float64)
    values = np.empty_like(omega_grid)
    for start in range(0, len(omega_grid), 2048):
        omega = omega_grid[start : start + 2048]
        phase = np.exp(-1j * omega[:, None] * t[None, :])
        values[start : start + len(omega)] = np.abs(phase @ y)
    return values


def nonuniform_gabor_spectrogram(t, y, u_grid, omega_grid, window_s=None, normalize=True):
    """Windowed nonuniform Fourier/Gabor spectrogram on raw BTT TOA samples.

    This is the numerical workhorse used by the reproduction scripts. It keeps
    the key BSE requirement: sum directly on nonuniform arrival times, with no
    interpolation onto a uniform grid.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64)
    omega_grid = np.asarray(omega_grid, dtype=np.float64)
    if window_s is None:
        window_s = max(float(t.max() - t.min()) / 3.0, 1e-6)

    out = np.empty((len(u_grid), len(omega_grid)), dtype=np.float64)
    for i, u in enumerate(u_grid):
        w = np.exp(-0.5 * ((t - u) / float(window_s)) ** 2)
        yw = y * w
        for start in range(0, len(omega_grid), 1024):
            omega = omega_grid[start : start + 1024]
            values = np.exp(-1j * omega[:, None] * (t[None, :] - u)) @ yw
            out[i, start : start + len(omega)] = np.abs(values)
        if normalize:
            denom = float(np.max(out[i]))
            if denom > 0.0:
                out[i] /= denom
    return out

