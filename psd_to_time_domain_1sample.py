import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Loading pickle file...")
with open('/Users/taiga/Desktop/noise/data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

# 最初の1サンプルを取得
x = data['x'][0, 0, :].cpu().numpy()  # (3000,)
print(f"Loaded 1 sample with {x.shape[0]} frequency points")
print(f"PSD data range: {x.min():.2e} to {x.max():.2e}")

# PSDから時間領域信号に変換
print("\nConverting PSD to time-domain signal...")
psd = x  # (3000,) - PSDデータ

# PSDから振幅スペクトルを推定
amplitude_spectrum = np.sqrt(psd * len(psd))

# ランダムな位相を追加
random_phase = np.random.uniform(0, 2*np.pi, len(amplitude_spectrum))

# 複素スペクトルを構築
complex_spectrum = amplitude_spectrum * np.exp(1j * random_phase)

# エルミート対称性を保証
full_spectrum = np.zeros(len(psd) * 2, dtype=complex)
full_spectrum[0] = complex_spectrum[0].real
full_spectrum[1:len(psd)] = complex_spectrum[1:]
full_spectrum[len(psd)+1:] = np.conj(complex_spectrum[-1:0:-1])

# 逆フーリエ変換
time_signal = np.fft.ifft(full_spectrum).real

print(f"Time-domain signal shape: {time_signal.shape}")
print(f"Time-domain data range: {time_signal.min():.2e} to {time_signal.max():.2e}")

# CSVに保存
print("\nSaving to CSV...")
df = pd.DataFrame({
    'Time_Point': np.arange(len(time_signal)),
    'Voltage': time_signal
})
df.to_csv('/Users/taiga/Desktop/noise/1_sample_time_domain.csv', index=False)
print(f"Saved {len(time_signal):,} time points to 1_sample_time_domain.csv")

# グラフ作成
print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=(20, 6))

time_points = df['Time_Point'].values
voltage = df['Voltage'].values

ax.plot(time_points, voltage, linewidth=0.7, color='blue', alpha=0.8)
ax.set_xlabel('Time (sampling points)', fontsize=14)
ax.set_ylabel('Voltage', fontsize=14)
ax.set_title('1 Sample Time-Domain Signal (6,000 points)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('/Users/taiga/Desktop/noise/1_sample_time_domain.png', dpi=150, bbox_inches='tight')
print("Saved: 1_sample_time_domain.png")

# 統計情報
print(f"\nStatistics:")
print(f"  Total points: {len(time_signal):,}")
print(f"  Min voltage: {voltage.min():.4e}")
print(f"  Max voltage: {voltage.max():.4e}")
print(f"  Mean voltage: {voltage.mean():.4e}")
print(f"  Std voltage: {voltage.std():.4e}")

print("\n✓ Complete!")
print(f"  - CSV: 1_sample_time_domain.csv ({len(time_signal):,} points)")
print(f"  - PNG: 1_sample_time_domain.png")
