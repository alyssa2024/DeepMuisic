import numpy as np

from bse_btt.btt.sampling import interference_coefficients


def alias_template(
    omega_axis,
    omega_true,
    omega_rot,
    probe_angles,
    k_min=-8,
    k_max=8,
    sigma_omega=20.0,
    normalize=True,
):
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    k_values = np.arange(int(k_min), int(k_max) + 1, dtype=np.int64)
    coeff = interference_coefficients(k_values, probe_angles)
    template = np.zeros_like(omega_axis, dtype=np.float64)
    for k, bk in zip(k_values, coeff):
        weight = abs(bk)
        centers = (float(omega_true) + k * float(omega_rot), -float(omega_true) + k * float(omega_rot))
        for center in centers:
            template += weight * np.exp(-0.5 * ((omega_axis - center) / float(sigma_omega)) ** 2)
    if normalize:
        norm = float(np.linalg.norm(template))
        if norm > 0.0:
            template /= norm
    return template

