import numpy as np


def probe_angles_rad(degrees):
    return np.deg2rad(np.asarray(degrees, dtype=np.float64))


def paper_probe_angles_rad():
    return probe_angles_rad([69.2, 108.2, 144.5, 195.7])


def real_probe_angles_rad():
    return probe_angles_rad([69.25, 108.25, 144.50, 195.75])


def constant_speed_events(num_revs, probe_angles, omega_rot_rad_s):
    probe_angles = np.asarray(probe_angles, dtype=np.float64)
    rev_ids = np.repeat(np.arange(num_revs, dtype=np.int64), len(probe_angles))
    probe_ids = np.tile(np.arange(len(probe_angles), dtype=np.int64), num_revs)
    theta = 2.0 * np.pi * rev_ids + probe_angles[probe_ids]
    t = theta / float(omega_rot_rad_s)
    order = np.argsort(t)
    return {
        "t": t[order],
        "theta": theta[order],
        "omega_rot": np.full_like(t, float(omega_rot_rad_s))[order],
        "rot_hz": np.full_like(t, float(omega_rot_rad_s) / (2.0 * np.pi))[order],
        "rev_ids": rev_ids[order],
        "probe_ids": probe_ids[order],
    }


def linear_speed_events(num_revs, probe_angles, base_rot_hz, speed_change, profile="up"):
    probe_angles = np.asarray(probe_angles, dtype=np.float64)
    total_theta = 2.0 * np.pi * float(num_revs)
    omega_mid = 2.0 * np.pi * float(base_rot_hz)
    sign = 1.0 if profile == "up" else -1.0
    omega_start = omega_mid * (1.0 - sign * speed_change / 2.0)
    omega_end = omega_mid * (1.0 + sign * speed_change / 2.0)
    beta = (omega_end**2 - omega_start**2) / (2.0 * total_theta)

    rev_ids = np.repeat(np.arange(num_revs, dtype=np.int64), len(probe_angles))
    probe_ids = np.tile(np.arange(len(probe_angles), dtype=np.int64), num_revs)
    theta = 2.0 * np.pi * rev_ids + probe_angles[probe_ids]

    if abs(beta) < 1e-14:
        omega = np.full_like(theta, omega_start)
        t = theta / omega_start
    else:
        omega = np.sqrt(np.maximum(omega_start**2 + 2.0 * beta * theta, 1e-14))
        t = (omega - omega_start) / beta

    order = np.argsort(t)
    return {
        "t": t[order],
        "theta": theta[order],
        "omega_rot": omega[order],
        "rot_hz": omega[order] / (2.0 * np.pi),
        "rev_ids": rev_ids[order],
        "probe_ids": probe_ids[order],
        "omega_start": omega_start,
        "omega_end": omega_end,
        "beta": beta,
    }


def local_rot_hz_at_times(events, u_grid):
    return np.interp(u_grid, events["t"], events["rot_hz"])


def local_rot_omega_at_times(events, u_grid):
    return 2.0 * np.pi * local_rot_hz_at_times(events, u_grid)


def interference_coefficients(k_values, probe_angles):
    k_values = np.asarray(k_values, dtype=np.int64)
    probe_angles = np.asarray(probe_angles, dtype=np.float64)
    phase = np.exp(-1j * k_values[:, None] * probe_angles[None, :])
    return phase.mean(axis=1)

