#实际采样时间不等于理想采样时间
import numpy as np

def simulate_btt_ideal_times(
    n_revs=100,
    base_freq_x=150.0,
    delta=0.018,
    probe_angles=[0, 45, 100, 185]
):
    """
    生成理想 BTT 到达时间。
    此时还没有考虑叶尖振动造成的到达时间偏移。
    """
    probe_angles_rad = np.radians(probe_angles)

    # 每圈转频 Hz
    fluctuations = np.random.uniform(-delta, delta, n_revs)
    freqs_per_rev = base_freq_x * (1 + fluctuations)

    # 每圈周期
    time_per_rev = 1.0 / freqs_per_rev

    # 每圈起始时间
    rev_start_times = np.concatenate(([0], np.cumsum(time_per_rev[:-1])))

    t_ideal = []
    rev_ids = []
    probe_ids = []
    theta_list = []
    freq_list = []

    for r in range(n_revs):
        f_r = freqs_per_rev[r]
        T_r = 1.0 / f_r
        t_start = rev_start_times[r]

        for i, theta in enumerate(probe_angles_rad):
            t_to_probe = (theta / (2 * np.pi)) * T_r
            t_sample = t_start + t_to_probe

            t_ideal.append(t_sample)
            rev_ids.append(r)
            probe_ids.append(i)
            theta_list.append(theta)
            freq_list.append(f_r)

    return (
        np.array(t_ideal),
        np.array(freq_list),
        np.array(rev_ids),
        np.array(probe_ids),
        np.array(theta_list),
        freqs_per_rev
    )

def generate_complex_harmonic_displacement(t, freqs, alphas, phis):
    """
    生成复数谐波位移。
    """
    x_t = np.zeros(len(t), dtype=complex)

    for f_k, alpha_k, phi_k in zip(freqs, alphas, phis):
        omega_k = 2 * np.pi * f_k
        x_t += alpha_k * np.exp(1j * (omega_k * t + phi_k))

    return x_t


def convert_displacement_to_time_shift(
    displacement,
    freqs_at_samples,
    blade_radius=0.5,
    sign=-1.0
):
    """
    将叶尖位移转换为到达时间偏移。

    displacement: 实值位移 u(t)，单位应与 blade_radius 一致，例如 m
    freqs_at_samples: 每个采样点对应的转频 Hz
    blade_radius: 叶片半径 R
    sign:
        -1 表示沿旋转方向的正位移导致提前到达
        +1 表示正位移导致滞后到达
    """
    omega_rot = 2 * np.pi * freqs_at_samples  # rad/s

    delta_t = sign * displacement / (blade_radius * omega_rot)

    return delta_t

def add_real_gaussian_noise(y, snr_db):
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(y))
    return y + noise

if __name__ == "__main__":
    # 这里只放测试合成数据是否正常的代码
  # 1. 参数
  base_x = 150.0
  fluctuation_delta = 0.001
  probes = [0, 28, 111.08, 166.15]
  n_revs = 200

  f_k = [167.0, 341.0, 635.0, 872.0]
  alpha_k = [0.0006, 0.0010, 0.0008, 0.0009]  # 建议用米级小位移
  phi_k = [0.0, 1.0, 2.0, 3.0]

  blade_radius = 0.5
  SNR = 20

  # 2. 理想到达时间
  (
      t_ideal,
      freqs_at_samples,
      rev_ids,
      probe_ids,
      theta_samples,
      freqs_per_rev
  ) = simulate_btt_ideal_times(
      n_revs=n_revs,
      base_freq_x=base_x,
      delta=fluctuation_delta,
      probe_angles=probes
  )

  # 3. 在理想到达时间处生成叶尖位移
  x_complex_ideal = generate_complex_harmonic_displacement(
      t=t_ideal,
      freqs=f_k,
      alphas=alpha_k,
      phis=phi_k
  )

  # BTT 时间偏移通常需要实值位移
  u_ideal = np.real(x_complex_ideal)

  # 4. 位移 -> 到达时间偏移
  delta_t = convert_displacement_to_time_shift(
      displacement=u_ideal,
      freqs_at_samples=freqs_at_samples,
      blade_radius=blade_radius,
      sign=-1.0
  )

  # 5. 实际到达时间
  t_actual = t_ideal + delta_t

  # 6. 可选：在实际到达时间重新计算观测位移
  x_complex_actual = generate_complex_harmonic_displacement(
      t=t_actual,
      freqs=f_k,
      alphas=alpha_k,
      phis=phi_k
  )

  u_actual = np.real(x_complex_actual)

  y_observed = add_real_gaussian_noise(u_actual, SNR)


  ## 注意：在实际应用中，传感器测量的到达时间可能还会有额外的噪声，这里我们暂时不考虑。
  # time_noise_std = 1e-7  # 根据传感器精度设置
  # delta_t_observed = delta_t + time_noise_std * np.random.randn(len(delta_t))
  # t_observed = t_ideal + delta_t_observed

