import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# A1. ホワイトノイズ（White Gaussian Noise）
print("\n[A1] Generating White Gaussian Noise...")
white_noise = np.random.normal(0, 1, N)
scale = calculate_noise_scale(original_signal, white_noise, TARGET_SNR_DB)
white_noise_scaled = white_noise * scale
noisy_A1 = original_signal + white_noise_scaled

# CSV保存
df_A1 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_A1
})
df_A1.to_csv('1_sample_A1_white_gaussian.csv', index=False)
print("  Saved: 1_sample_A1_white_gaussian.csv")

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
ax2.plot(time_points, noisy_A1, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (White Gaussian)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-9e-13, 7e-13)
ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useOffset=False)

plt.tight_layout()
plt.savefig('1_sample_A1_white_gaussian.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_A1_white_gaussian.png")

print("\n✓ A1 Complete!")
