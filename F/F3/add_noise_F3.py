import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# SNR設定
TARGET_SNR_DB = 5

# F3パラメータ：帯域ノッチ（Bandstop/Notch）
NOTCH_CENTER_FREQ = 100  # ノッチ中心周波数（Hz）
NOTCH_BANDWIDTH = 20  # ノッチ帯域幅（Hz）
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

# F3. 帯域ノッチ（Bandstop/Notch）
print("\n[F3] Generating Bandstop/Notch...")
# FFTを実行
fft_signal = np.fft.fft(original_signal)
freqs = np.fft.fftfreq(N, d=1/SAMPLING_RATE)

# ノッチフィルタを適用する周波数範囲を決定
notch_low = NOTCH_CENTER_FREQ - NOTCH_BANDWIDTH / 2
notch_high = NOTCH_CENTER_FREQ + NOTCH_BANDWIDTH / 2

# ノッチフィルタを適用（指定帯域をゼロ化）
fft_notched = fft_signal.copy()
mask = (np.abs(freqs) >= notch_low) & (np.abs(freqs) <= notch_high)
fft_notched[mask] = 0

# 逆FFTで時間領域に戻す
notched_signal = np.fft.ifft(fft_notched).real

# ノイズ成分を抽出
notch_noise = notched_signal - original_signal

scale = calculate_noise_scale(original_signal, notch_noise, TARGET_SNR_DB)
notch_noise_scaled = notch_noise * scale
noisy_F3 = original_signal + notch_noise_scaled

print(f"  Notch center frequency: {NOTCH_CENTER_FREQ} Hz")
print(f"  Notch bandwidth: {NOTCH_BANDWIDTH} Hz ({notch_low}-{notch_high} Hz)")
print(f"  Affected frequency bins: {np.sum(mask)}")

# CSV保存
df_F3 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_F3
})
df_F3.to_csv('1_sample_F3_bandstop_notch.csv', index=False)
print("  Saved: 1_sample_F3_bandstop_notch.csv")

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
ax2.plot(time_points, noisy_F3, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (Bandstop/Notch)', fontsize=12, fontweight='bold')
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
plt.savefig('1_sample_F3_bandstop_notch.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_F3_bandstop_notch.png")

print("\n✓ F3 Complete!")
