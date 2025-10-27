import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Loading pickle file...")
with open('/Users/taiga/Desktop/noise/data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

# 最初の10サンプルを取得
x = data['x'][:10, 0, :].cpu().numpy()  # (10, 3000)
print(f"Loaded {x.shape[0]} samples with {x.shape[1]} frequency points each")
print(f"PSD data range: {x.min():.2e} to {x.max():.2e}")

# PSDから時間領域信号に変換
print("\nConverting PSD to time-domain signals...")
time_domain_signals = []

for i in range(len(x)):
    psd = x[i]  # (3000,) - PSDデータ

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

    time_domain_signals.append(time_signal)
    print(f"  Processed sample {i}")

time_domain_signals = np.array(time_domain_signals)  # (10, 6000)
print(f"\nTime-domain signals shape: {time_domain_signals.shape}")
print(f"Time-domain data range: {time_domain_signals.min():.2e} to {time_domain_signals.max():.2e}")

# CSVに保存（全10サンプルを連結）
print("\nSaving to CSV...")
concatenated = time_domain_signals.flatten()  # (60000,)
df = pd.DataFrame({
    'Time_Point': np.arange(len(concatenated)),
    'Voltage': concatenated
})
df.to_csv('/Users/taiga/Desktop/noise/10_samples_time_domain.csv', index=False)
print(f"Saved {len(concatenated):,} time points to 10_samples_time_domain.csv")

# グラフ作成
print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=(20, 6))

time_points = df['Time_Point'].values
voltage = df['Voltage'].values

ax.plot(time_points, voltage, linewidth=0.5, color='blue', alpha=0.8)
ax.set_xlabel('Time (sampling points)', fontsize=14)
ax.set_ylabel('Voltage', fontsize=14)
ax.set_title('10 Samples Time-Domain Signal (60,000 points)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('/Users/taiga/Desktop/noise/10_samples_time_domain.png', dpi=150, bbox_inches='tight')
print("Saved: 10_samples_time_domain.png")

# 統計情報
print(f"\nStatistics:")
print(f"  Number of samples: {len(time_domain_signals)}")
print(f"  Points per sample: {len(time_domain_signals[0])}")
print(f"  Total points: {len(concatenated):,}")
print(f"  Min voltage: {voltage.min():.4e}")
print(f"  Max voltage: {voltage.max():.4e}")
print(f"  Mean voltage: {voltage.mean():.4e}")
print(f"  Std voltage: {voltage.std():.4e}")

print("\n✓ Complete!")
print(f"  - CSV: 10_samples_time_domain.csv ({len(concatenated):,} points)")
print(f"  - PNG: 10_samples_time_domain.png")
