import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# SNR設定
TARGET_SNR_DB = 5

# C3パラメータ：ブロック単位ゲインシフト
BLOCK_SIZE = 500  # ブロックサイズ（サンプル）
GAIN_SHIFT_RANGE = 0.3  # ゲインシフト範囲（±30%）

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

# C3. ブロック単位ゲインシフト
print("\n[C3] Generating Block-wise Gain Shift...")
# ブロックごとにランダムなゲインを割り当て
num_blocks = int(np.ceil(N / BLOCK_SIZE))
block_gains = 1.0 + GAIN_SHIFT_RANGE * (2 * np.random.rand(num_blocks) - 1)

# 各サンプルにブロックゲインを適用
gain_array = np.zeros(N)
for i in range(num_blocks):
    start_idx = i * BLOCK_SIZE
    end_idx = min((i + 1) * BLOCK_SIZE, N)
    gain_array[start_idx:end_idx] = block_gains[i]

# ゲインシフトを適用
shifted_signal = original_signal * gain_array

# ノイズ成分を抽出
shift_noise = shifted_signal - original_signal

scale = calculate_noise_scale(original_signal, shift_noise, TARGET_SNR_DB)
shift_noise_scaled = shift_noise * scale
noisy_C3 = original_signal + shift_noise_scaled

print(f"  Number of blocks: {num_blocks}")
print(f"  Gain range: {block_gains.min():.3f} to {block_gains.max():.3f}")

# CSV保存
df_C3 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_C3
})
df_C3.to_csv('1_sample_C3_block_wise_gain_shift.csv', index=False)
print("  Saved: 1_sample_C3_block_wise_gain_shift.csv")

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
ax2.plot(time_points, noisy_C3, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (Block-wise Gain Shift)', fontsize=12, fontweight='bold')
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
plt.savefig('1_sample_C3_block_wise_gain_shift.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_C3_block_wise_gain_shift.png")

print("\n✓ C3 Complete!")
