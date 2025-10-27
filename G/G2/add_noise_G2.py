import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# SNR設定
TARGET_SNR_DB = 5

# G2パラメータ：周期バースト
BURST_PERIOD = 1000  # バースト周期（サンプル）
BURST_DURATION = 100  # バースト持続時間（サンプル）
BURST_AMPLITUDE = 3.0  # バースト振幅（信号の標準偏差の倍数）

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

# G2. 周期バースト
print("\n[G2] Generating Periodic Bursts...")
# バースト信号を初期化
burst_noise = np.zeros(N)

# 信号の標準偏差を計算
signal_std = np.std(original_signal)

# 周期的にバーストを追加
num_bursts = 0
for start_pos in range(0, N, BURST_PERIOD):
    end_pos = min(start_pos + BURST_DURATION, N)
    # ガウスノイズベースのバーストを生成
    burst_noise[start_pos:end_pos] = BURST_AMPLITUDE * signal_std * np.random.randn(end_pos - start_pos)
    num_bursts += 1

# バーストを適用
burst_signal = original_signal + burst_noise

# ノイズ成分を抽出（既にburst_noiseがノイズ成分）
scale = calculate_noise_scale(original_signal, burst_noise, TARGET_SNR_DB)
burst_noise_scaled = burst_noise * scale
noisy_G2 = original_signal + burst_noise_scaled

print(f"  Burst period: {BURST_PERIOD} samples")
print(f"  Burst duration: {BURST_DURATION} samples")
print(f"  Number of bursts: {num_bursts}")

# CSV保存
df_G2 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_G2
})
df_G2.to_csv('1_sample_G2_periodic_bursts.csv', index=False)
print("  Saved: 1_sample_G2_periodic_bursts.csv")

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
ax2.plot(time_points, noisy_G2, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (Periodic Bursts)', fontsize=12, fontweight='bold')
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
plt.savefig('1_sample_G2_periodic_bursts.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_G2_periodic_bursts.png")

print("\n✓ G2 Complete!")
