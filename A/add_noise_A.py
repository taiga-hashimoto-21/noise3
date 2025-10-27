import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# SNR設定（デフォルト）
TARGET_SNR_DB = 5

print("Loading original data...")
df_original = pd.read_csv('../1_sample.csv')
original_signal = df_original['Voltage'].values
time_points = df_original['Time_Point'].values
N = len(original_signal)

print(f"Original signal: {N} points")
print(f"Target SNR: {TARGET_SNR_DB} dB")

# SNRに基づいてノイズレベルを計算
def calculate_noise_scale(signal, noise, target_snr_db):
    """
    信号とノイズからSNRを達成するためのスケーリング係数を計算
    SNR(dB) = 10 * log10(P_signal / P_noise)
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    # 目標SNRから必要なノイズパワーを計算
    target_snr_linear = 10 ** (target_snr_db / 10)
    target_noise_power = signal_power / target_snr_linear

    # スケーリング係数
    scale = np.sqrt(target_noise_power / noise_power)
    return scale

# グラフ作成関数
def save_plot(original, noisy, filename, noise_type):
    """上段：オリジナル（青）、下段：ノイズ付き（赤）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 10))

    # 上段：オリジナルのみ
    ax1.plot(time_points, original, linewidth=0.7, color='blue', alpha=0.8)
    ax1.set_ylabel('Voltage', fontsize=14)
    ax1.set_title('Original Signal', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # 下段：ノイズ付き
    ax2.plot(time_points, noisy, linewidth=0.7, color='red', alpha=0.8)
    ax2.set_xlabel('Time (sampling points)', fontsize=14)
    ax2.set_ylabel('Voltage', fontsize=14)
    ax2.set_title(f'Noisy Signal ({noise_type})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# ============================================
# A1. ホワイトノイズ（White Gaussian Noise）
# ============================================
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

# グラフ保存
save_plot(original_signal, noisy_A1, '1_sample_A1_white_gaussian.png', 'White Gaussian')

# ============================================
# A2. ピンクノイズ（1/f）
# ============================================
print("\n[A2] Generating Pink Noise (1/f)...")
# FFTベースで生成
white = np.random.normal(0, 1, N)
fft_white = np.fft.rfft(white)
freqs = np.fft.rfftfreq(N)

# 1/f特性を適用（DC成分は除外）
pink_fft = fft_white.copy()
pink_fft[1:] /= np.sqrt(freqs[1:])  # 1/√f = 1/f^0.5 でパワーが1/f
pink_noise = np.fft.irfft(pink_fft, n=N)

scale = calculate_noise_scale(original_signal, pink_noise, TARGET_SNR_DB)
pink_noise_scaled = pink_noise * scale
noisy_A2 = original_signal + pink_noise_scaled

df_A2 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_A2
})
df_A2.to_csv('1_sample_A2_pink_noise_1_over_f.csv', index=False)
print("  Saved: 1_sample_A2_pink_noise_1_over_f.csv")

save_plot(original_signal, noisy_A2, '1_sample_A2_pink_noise_1_over_f.png', 'Pink Noise (1/f)')

# ============================================
# A3. ブラウンノイズ（1/f²）
# ============================================
print("\n[A3] Generating Brown Noise (1/f²)...")
brown_fft = fft_white.copy()
brown_fft[1:] /= freqs[1:]  # 1/f でパワーが1/f²
brown_noise = np.fft.irfft(brown_fft, n=N)

scale = calculate_noise_scale(original_signal, brown_noise, TARGET_SNR_DB)
brown_noise_scaled = brown_noise * scale
noisy_A3 = original_signal + brown_noise_scaled

df_A3 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_A3
})
df_A3.to_csv('1_sample_A3_brown_noise_1_over_f2.csv', index=False)
print("  Saved: 1_sample_A3_brown_noise_1_over_f2.csv")

save_plot(original_signal, noisy_A3, '1_sample_A3_brown_noise_1_over_f2.png', 'Brown Noise (1/f²)')

# ============================================
# A4. ブルーノイズ（f）
# ============================================
print("\n[A4] Generating Blue Noise (f)...")
blue_fft = fft_white.copy()
blue_fft[1:] *= np.sqrt(freqs[1:])  # √f でパワーがf
blue_noise = np.fft.irfft(blue_fft, n=N)

scale = calculate_noise_scale(original_signal, blue_noise, TARGET_SNR_DB)
blue_noise_scaled = blue_noise * scale
noisy_A4 = original_signal + blue_noise_scaled

df_A4 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_A4
})
df_A4.to_csv('1_sample_A4_blue_noise_f.csv', index=False)
print("  Saved: 1_sample_A4_blue_noise_f.csv")

save_plot(original_signal, noisy_A4, '1_sample_A4_blue_noise_f.png', 'Blue Noise (f)')

# ============================================
# A5. バイオレットノイズ（f²）
# ============================================
print("\n[A5] Generating Violet Noise (f²)...")
violet_fft = fft_white.copy()
violet_fft[1:] *= freqs[1:]  # f でパワーがf²
violet_noise = np.fft.irfft(violet_fft, n=N)

scale = calculate_noise_scale(original_signal, violet_noise, TARGET_SNR_DB)
violet_noise_scaled = violet_noise * scale
noisy_A5 = original_signal + violet_noise_scaled

df_A5 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_A5
})
df_A5.to_csv('1_sample_A5_violet_noise_f2.csv', index=False)
print("  Saved: 1_sample_A5_violet_noise_f2.csv")

save_plot(original_signal, noisy_A5, '1_sample_A5_violet_noise_f2.png', 'Violet Noise (f²)')

# ============================================
# A6. 帯域限定ガウスノイズ
# ============================================
print("\n[A6] Generating Band-limited Gaussian Noise...")
# バンドパス: サンプリング周波数の10%〜40%あたり
fs = 1.0  # 正規化周波数
lowcut = 0.1
highcut = 0.4

# バターワースバンドパスフィルタ
sos = signal.butter(4, [lowcut, highcut], btype='band', output='sos')
white_for_band = np.random.normal(0, 1, N)
bandlimited_noise = signal.sosfilt(sos, white_for_band)

scale = calculate_noise_scale(original_signal, bandlimited_noise, TARGET_SNR_DB)
bandlimited_noise_scaled = bandlimited_noise * scale
noisy_A6 = original_signal + bandlimited_noise_scaled

df_A6 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_A6
})
df_A6.to_csv('1_sample_A6_band_limited_gaussian.csv', index=False)
print("  Saved: 1_sample_A6_band_limited_gaussian.csv")

save_plot(original_signal, noisy_A6, '1_sample_A6_band_limited_gaussian.png', 'Band-limited Gaussian')

print("\n✓ All Category A noises generated successfully!")
print("  - 6 CSV files")
print("  - 6 PNG files")
