import numpy as np
import matplotlib.pyplot as plt
# 实际采样时间等于理想采样时间

def simulate_fluctuating_speed_btt(
    n_revs=100,           
    base_freq_x=150.0,    
    delta=0.018,          
    probe_angles=[0, 45, 100, 185] 
):
    """
    生成带有随机波动的转速及非均匀采样时间 t
    返回:
        t_samples: 采样时间数组
        freqs_per_rev: 每圈的转速数组
        rev_ids: 每个采样点对应的圈号
        probe_ids: 每个采样点对应的探头编号
        theta_samples: 每个采样点对应的角度 (弧度)
        freqs_at_samples: 每个采样点对应的转速
    """
    probe_angles_rad = np.radians(probe_angles)
    
    # 1. 生成每一圈的实际转速 (Hz)
    fluctuations = np.random.uniform(-delta, delta, n_revs)
    freqs_per_rev = base_freq_x * (1 + fluctuations)
    
    # 2. 计算每圈的时间并累加得到绝对时间
    time_per_rev = 1.0 / freqs_per_rev
    rev_start_times = np.concatenate(([0], np.cumsum(time_per_rev[:-1])))
    
    # 3. 计算每个探头的确切采样时刻 t
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
        np.array(freqs_at_samples)
    )

def generate_complex_harmonic_displacement(t, freqs, alphas, phis, snr_db=None):
    """
    根据复指数谐波模型生成叶片位移信号 x_t
    模型: x_t = sum(alpha_k * exp(j * (omega_k * t + phi_k)))
    """
    # 初始化为复数数组
    x_t = np.zeros(len(t), dtype=complex)
    
    # 叠加 K 个谐波分量
    for f_k, alpha_k, phi_k in zip(freqs, alphas, phis):
        omega_k = 2 * np.pi * f_k  # 将频率(Hz)转换为角频率(rad/s)
        x_t += alpha_k * np.exp(1j * (omega_k * t + phi_k))
        
    # 添加复高斯白噪声
    if snr_db is not None:
        # 计算信号功率
        sig_power = np.mean(np.abs(x_t)**2)
        # 根据 SNR 计算噪声功率
        noise_power = sig_power / (10**(snr_db / 10))
        
        # 生成复高斯噪声 (实部和虚部各自承担一半的方差)
        noise_real = np.sqrt(noise_power / 2) * np.random.randn(len(t))
        noise_imag = np.sqrt(noise_power / 2) * np.random.randn(len(t))
        noise = noise_real + 1j * noise_imag
        
        x_t_noisy = x_t + noise
    else:
        x_t_noisy = x_t
        
    return x_t_noisy, x_t

# ==========================================
# 示例运行与数据合成
# ==========================================

# 1. 转速与采样时间参数 (参考上传文献参数)
base_x = 150.0 #理论上能检测到的最高频率是975Hz
fluctuation_delta = 0.001 
probes = [0, 28, 111.08, 166.15] # 探头物理安装角度
n_revs = 200

# 生成时间 t
t_samples, freqs_per_rev, rev_ids, probe_ids, theta_samples, freqs_at_samples = simulate_fluctuating_speed_btt(
    n_revs=n_revs, 
    base_freq_x=base_x, 
    delta=fluctuation_delta,
    probe_angles=probes
)

# 2. 谐波位移参数
# K=4 个分量
f_k = [167.0, 341.0, 635.0, 872.0]  # 频率 (Hz)
alpha_k = [0.0006, 0.0010, 0.0008, 0.0009] # 幅度 (m)
phi_k = [0.0, 1.0, 2.0, 3.0]        # 初始相位 (rad)
SNR = 20                            # 信噪比 (dB)

# 生成复数位移信号 x_t
x_t_observed, x_t_true = generate_complex_harmonic_displacement(
    t=t_samples,
    freqs=f_k,
    alphas=alpha_k,
    phis=phi_k,
    snr_db=SNR
)

# ==========================================
# 可视化 (由于是复数信号，通常绘制其实部或幅值)
# ==========================================
plt.figure(figsize=(12, 6))

# 绘制前 40 个采样点（约 10 圈的数据）的实部
plot_points = 40
plt.plot(t_samples[:plot_points], np.real(x_t_observed[:plot_points]), 'ro', label='Measured Real(x_t) with Noise')
plt.plot(t_samples[:plot_points], np.real(x_t_true[:plot_points]), 'b--', alpha=0.6, label='True Real(x_t)')

plt.xlabel('Time (s)')
plt.ylabel('Displacement Amplitude (Real Part)')
plt.title('Synthetic BTT Data based on Complex Harmonic Model')
plt.legend()
plt.grid(True)
plt.show()

print(f"Generated {len(t_samples)} samples in total.")
print(f"x_t data type: {x_t_observed.dtype}")
print(f"Sample data [0]: {x_t_observed[0]:.4f}")