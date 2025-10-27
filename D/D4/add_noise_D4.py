import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d

# SNR設定
TARGET_SNR_DB = 5

# D4パラメータ：非線形時間遅延
DELAY_AMPLITUDE = 50  # 遅延の最大振幅（サンプル）
DELAY_FREQ = 3  # 遅延の周波数（Hz）
SAMPLING_RATE = 6000  # サンプリングレート

print("Loading original data...")
df_original = pd.read_csv('../../1_sample.csv')
original_signal = df_original['Voltage'].values
time_points = df_original['Time_Point'].values
N = len(original_signal)

print(f"Original signal: {N} points")
print(f"Target SNR: {TARGET_SNR_DB} dB")

def calculate_noise_scale(signal, noise, target_snr_db):
    """SNRに基づいてノイズスケールを計算"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    target_snr_linear = 10 ** (target_snr_db / 10)
    target_noise_power = signal_power / target_snr_linear
    scale = np.sqrt(target_noise_power / noise_power)
    return scale

# D4. 非線形時間遅延
print("\n[D4] Generating Nonlinear Time Delay...")
# 時間軸（秒）
t = time_points / SAMPLING_RATE

# 非線形遅延を生成（正弦波ベース）
time_delay = DELAY_AMPLITUDE * np.sin(2 * np.pi * DELAY_FREQ * t)

# 遅延を適用した時刻
delayed_time = time_points + time_delay

# 範囲内に収める
delayed_time = np.clip(delayed_time, time_points[0], time_points[-1])

# 元の信号を補間して遅延後の時刻でサンプリング
interpolator = interp1d(time_points, original_signal, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
delayed_signal = interpolator(delayed_time)

# ノイズ成分を抽出
delay_noise = delayed_signal - original_signal

scale = calculate_noise_scale(original_signal, delay_noise, TARGET_SNR_DB)
delay_noise_scaled = delay_noise * scale
noisy_D4 = original_signal + delay_noise_scaled

print(f"  Time delay range: {time_delay.min():.2f} to {time_delay.max():.2f} samples")

# CSV保存
df_D4 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_D4
})
df_D4.to_csv('1_sample_D4_nonlinear_time_delay.csv', index=False)
print("  Saved: 1_sample_D4_nonlinear_time_delay.csv")

# グラフ作成
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 10))

# 上段：オリジナルのみ
ax1.plot(time_points, original_signal, linewidth=0.7, color='blue', alpha=0.8)
ax1.set_ylabel('Voltage', fontsize=14)
ax1.set_title('Original Signal', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-7e-13, 5e-13)
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useOffset=False, useMathText=False)

# 下段：ノイズ付き
ax2.plot(time_points, noisy_D4, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (Nonlinear Time Delay)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-9e-13, 7e-13)
# 整数目盛りを強制
formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
formatter.set_useOffset(False)
ax2.yaxis.set_major_formatter(formatter)
# データ範囲から適切な整数目盛りを設定
y_min, y_max = ax2.get_ylim()
order = np.floor(np.log10(max(abs(y_min), abs(y_max))))
scale = 10 ** order
n_ticks = 7
tick_vals = np.linspace(y_min / scale, y_max / scale, n_ticks)
tick_vals = np.round(tick_vals).astype(int) * scale
ax2.set_yticks(np.unique(tick_vals))

plt.tight_layout()
plt.savefig('1_sample_D4_nonlinear_time_delay.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_D4_nonlinear_time_delay.png")

print("\n✓ D4 Complete!")
