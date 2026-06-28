import numpy as np

from bse_btt.wavelet.theoretical_spectrum import alias_template


def bse_log_likelihood(
    observed_spectrogram,
    omega_axis,
    candidate_omega,
    rot_omega_by_time,
    probe_angles,
    k_min=-8,
    k_max=8,
    sigma_alias=20.0,
    sigma_scale=0.1,
):
    observed = np.asarray(observed_spectrogram, dtype=np.float64)
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    candidate_omega = np.asarray(candidate_omega, dtype=np.float64)
    rot_omega_by_time = np.asarray(rot_omega_by_time, dtype=np.float64)
    energies = np.empty((observed.shape[0], len(candidate_omega)), dtype=np.float64)

    for i in range(observed.shape[0]):
        w = observed[i]
        for j, omega in enumerate(candidate_omega):
            tmpl = alias_template(
                omega_axis,
                omega,
                rot_omega_by_time[i],
                probe_angles,
                k_min=k_min,
                k_max=k_max,
                sigma_omega=sigma_alias,
                normalize=False,
            )
            denom = float(np.dot(tmpl, tmpl)) + 1e-12
            amp = max(float(np.dot(tmpl, w)) / denom, 0.0)
            resid = w - amp * tmpl
            energies[i, j] = float(np.dot(resid, resid))

    sigma2 = float(sigma_scale) * float(np.median(energies) + 1e-12)
    log_like = -energies / (2.0 * sigma2)
    log_like -= np.max(log_like, axis=1, keepdims=True)
    return log_like, energies

