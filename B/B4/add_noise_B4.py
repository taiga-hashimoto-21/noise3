import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# SNR設定
TARGET_SNR_DB = 5

# B4パラメータ：ランダムテレグラフノイズ
RTN_SWITCH_PROBABILITY = 0.005  # 状態遷移確率（0.5%）
RTN_AMPLITUDE = 2.0  # ±Aの振幅

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

# B4. ランダムテレグラフノイズ
print("\n[B4] Generating Random Telegraph Noise (RTN)...")
# 初期状態：+A or -A
rtn_state = np.random.choice([-RTN_AMPLITUDE, RTN_AMPLITUDE])
rtn_noise = np.zeros(N)

# 各時点で確率的に状態遷移
for i in range(N):
    if np.random.rand() < RTN_SWITCH_PROBABILITY:
        rtn_state = -rtn_state  # 状態反転
    rtn_noise[i] = rtn_state

# 信号の標準偏差でスケーリング
signal_std = np.std(original_signal)
rtn_noise_scaled = rtn_noise * signal_std

scale = calculate_noise_scale(original_signal, rtn_noise_scaled, TARGET_SNR_DB)
rtn_noise_scaled = rtn_noise_scaled * scale
noisy_B4 = original_signal + rtn_noise_scaled

# 状態遷移回数をカウント
switches = np.sum(np.abs(np.diff(rtn_noise)) > 0)
print(f"  State transitions: {switches}")

# CSV保存
df_B4 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_B4
})
df_B4.to_csv('1_sample_B4_random_telegraph_noise_rtn.csv', index=False)
print("  Saved: 1_sample_B4_random_telegraph_noise_rtn.csv")

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
ax2.plot(time_points, noisy_B4, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (Random Telegraph Noise)', fontsize=12, fontweight='bold')
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
plt.savefig('1_sample_B4_random_telegraph_noise_rtn.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_B4_random_telegraph_noise_rtn.png")

print("\n✓ B4 Complete!")
