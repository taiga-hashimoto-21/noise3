import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# SNR設定
TARGET_SNR_DB = 5

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

# A6. 帯域限定ガウスノイズ
print("\n[A6] Generating Band-limited Gaussian Noise...")
fs = 1.0
lowcut = 0.1
highcut = 0.4

sos = signal.butter(4, [lowcut, highcut], btype='band', output='sos')
white_for_band = np.random.normal(0, 1, N)
bandlimited_noise = signal.sosfilt(sos, white_for_band)

scale = calculate_noise_scale(original_signal, bandlimited_noise, TARGET_SNR_DB)
bandlimited_noise_scaled = bandlimited_noise * scale
noisy_A6 = original_signal + bandlimited_noise_scaled

# CSV保存
df_A6 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_A6
})
df_A6.to_csv('1_sample_A6_band_limited_gaussian.csv', index=False)
print("  Saved: 1_sample_A6_band_limited_gaussian.csv")

# グラフ作成
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 10))

# 上段：オリジナルのみ
ax1.plot(time_points, original_signal, linewidth=0.7, color='blue', alpha=0.8)
ax1.set_ylabel('Voltage', fontsize=14)
ax1.set_title('Original Signal', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-7e-13, 5e-13)
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useOffset=False)

# 下段：ノイズ付き
ax2.plot(time_points, noisy_A6, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (Band-limited Gaussian)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-9e-13, 7e-13)
ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useOffset=False)

plt.tight_layout()
plt.savefig('1_sample_A6_band_limited_gaussian.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_A6_band_limited_gaussian.png")

print("\n✓ A6 Complete!")
