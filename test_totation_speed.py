import numpy as np
import matplotlib.pyplot as plt

def simulate_fluctuating_speed_btt(
    n_revs=100,           # 转圈数 (Nr)
    base_freq_x=150.0,    # 基准转速 x (Hz) - 参考论文参数
    delta=0.018,          # 波动系数 δ (例如 1.8% = 0.018)
    probe_angles=[0, 45, 100, 185] # 探头安装角度 (度)
):
    """
    模拟带有均匀分布波动的转速，并计算 BTT 探头的采样时间
    """
    probe_angles_rad = np.radians(probe_angles)
    num_probes = len(probe_angles)
    
    # 1. 生成每一圈的实际转速 (Hz)
    # 均匀分布在 (1 - delta)*x 到 (1 + delta)*x 之间
    # 等价于 base_freq_x * (1 + uniform(-delta, delta))
    fluctuations = np.random.uniform(-delta, delta, n_revs)
    freqs_per_rev = base_freq_x * (1 + fluctuations)
    
    # 计算每一圈完整旋转所需的时间 (周期 T = 1 / f)
    time_per_rev = 1.0 / freqs_per_rev
    
    # 计算每圈开始的绝对时间 (累加前面的周期)
    # 第 0 圈的开始时间是 0
    rev_start_times = np.concatenate(([0], np.cumsum(time_per_rev[:-1])))
    
    # 2. 计算每个探头的采样时间 t_rj
    t = []
    # 记录每个采样点对应的瞬时转速 (用于后续位移模型或参考)
    instant_freqs = [] 
    
    for r in range(n_revs):
        f_r = freqs_per_rev[r]
        t_start = rev_start_times[r]
        
        for theta in probe_angles_rad:
            # 当前圈到达角度 theta 所需的时间
            # 假设在一整圈内转速是恒定的，等于 f_r
            t_to_probe = (theta / (2 * np.pi)) * (1.0 / f_r)
            t_sample = t_start + t_to_probe
            
            t.append(t_sample)
            instant_freqs.append(f_r)
            
    return np.array(t), np.array(instant_freqs), freqs_per_rev

# --- 示例运行与可视化 ---
base_x = 110.0 # 基准频率 110 Hz
fluctuation_delta = 0.001 # 1.8% 的波动
base_x_rpm = base_x * 60 # 转速转换为 RPM (每分钟转数)

# 生成时间与转速数据
t_samples, freq_samples, freqs_per_rev = simulate_fluctuating_speed_btt(
    n_revs=200, 
    base_freq_x=base_x, 
    delta=fluctuation_delta,
    probe_angles=[0, 28, 111.08, 166.15]
)

# 可视化转速波动
plt.figure(figsize=(12, 5))

# 绘制每一圈的转速
plt.plot(range(len(freqs_per_rev)), freqs_per_rev *60, 'b.-', alpha=0.7, label='Actual Frequency per Revolution')
plt.axhline(base_x_rpm, color='r', linestyle='--', label=f'Base Frequency x = {base_x_rpm} RPM')
plt.axhline(base_x_rpm * (1 + fluctuation_delta), color='g', linestyle=':', label='Upper Bound (+0.1%)')
plt.axhline(base_x_rpm * (1 - fluctuation_delta), color='g', linestyle=':', label='Lower Bound (-%)')

plt.xlabel('Revolution Number')
plt.ylabel('Rotational Frequency (RPM)')
plt.title(f'Simulated Rotational Speed Fluctuation ($\delta$ = {fluctuation_delta*100}%)')
plt.legend()
plt.grid(True)
plt.show()
