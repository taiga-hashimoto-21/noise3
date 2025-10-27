import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# SNR設定
TARGET_SNR_DB = 5

# D3パラメータ：重複/スキップ
DUP_SKIP_PROBABILITY = 0.01  # 重複/スキップ確率（1%）

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

# D3. 重複/スキップ
print("\n[D3] Generating Duplicate/Skip...")
# ランダムな位置で重複またはスキップを発生
dup_skip_signal = original_signal.copy()
num_dups = 0
num_skips = 0

for i in range(N):
    if np.random.rand() < DUP_SKIP_PROBABILITY:
        if np.random.rand() < 0.5 and i > 0:
            # 重複: 前のサンプルを使用
            dup_skip_signal[i] = dup_skip_signal[i-1]
            num_dups += 1
        elif i < N - 1:
            # スキップ: 次のサンプルを使用
            dup_skip_signal[i] = original_signal[min(i+1, N-1)]
            num_skips += 1

# ノイズ成分を抽出
dup_skip_noise = dup_skip_signal - original_signal

scale = calculate_noise_scale(original_signal, dup_skip_noise, TARGET_SNR_DB)
dup_skip_noise_scaled = dup_skip_noise * scale
noisy_D3 = original_signal + dup_skip_noise_scaled

print(f"  Duplicates: {num_dups}, Skips: {num_skips}")

# CSV保存
df_D3 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_D3
})
df_D3.to_csv('1_sample_D3_duplicate_skip.csv', index=False)
print("  Saved: 1_sample_D3_duplicate_skip.csv")

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
ax2.plot(time_points, noisy_D3, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (Duplicate/Skip)', fontsize=12, fontweight='bold')
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
plt.savefig('1_sample_D3_duplicate_skip.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_D3_duplicate_skip.png")

print("\n✓ D3 Complete!")
