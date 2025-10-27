import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Loading pickle file...")
with open('/Users/taiga/Desktop/noise/data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

# 最初の100サンプルを取得
x = data['x'][:100, 0, :].cpu().numpy()  # (100, 3000)
print(f"Loaded {x.shape[0]} samples with {x.shape[1]} frequency points each")
print(f"PSD data range: {x.min():.2e} to {x.max():.2e}")

# PSDから時間領域信号に変換
print("\nConverting PSD to time-domain signals...")
time_domain_signals = []

for i in range(len(x)):
    psd = x[i]  # (3000,) - PSDデータ

    # PSDから振幅スペクトルを推定
    # PSD ∝ |FFT|² なので、振幅 = sqrt(PSD)
    # スケーリング係数を追加して実際の振幅に近づける
    amplitude_spectrum = np.sqrt(psd * len(psd))

    # ランダムな位相を追加（PSDには位相情報がないため）
    random_phase = np.random.uniform(0, 2*np.pi, len(amplitude_spectrum))

    # 複素スペクトルを構築
    complex_spectrum = amplitude_spectrum * np.exp(1j * random_phase)

    # エルミート対称性を保証（実数信号を得るため）
    # 正の周波数成分のみを使用し、負の周波数は共役で構築
    full_spectrum = np.zeros(len(psd) * 2, dtype=complex)
    full_spectrum[0] = complex_spectrum[0].real  # DC成分は実数
    full_spectrum[1:len(psd)] = complex_spectrum[1:]
    full_spectrum[len(psd)+1:] = np.conj(complex_spectrum[-1:0:-1])

    # 逆フーリエ変換
    time_signal = np.fft.ifft(full_spectrum).real

    time_domain_signals.append(time_signal)

    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{len(x)} samples")

time_domain_signals = np.array(time_domain_signals)  # (100, 6000)
print(f"\nTime-domain signals shape: {time_domain_signals.shape}")
print(f"Time-domain data range: {time_domain_signals.min():.2e} to {time_domain_signals.max():.2e}")

# CSVに保存（全100サンプルを連結）
print("\nSaving to CSV...")
concatenated = time_domain_signals.flatten()  # (600000,)
df = pd.DataFrame({
    'Time_Point': np.arange(len(concatenated)),
    'Voltage': concatenated
})
df.to_csv('/Users/taiga/Desktop/noise/100_samples_time_domain.csv', index=False)
print(f"Saved {len(concatenated)} time points to 100_samples_time_domain.csv")

# 可視化1: 元のPSDデータ（周波数ドメイン）
print("\nCreating PSD visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# サンプル0のPSD
ax = axes[0, 0]
frequencies = np.arange(len(x[0]))
ax.loglog(frequencies[1:], x[0][1:], linewidth=1.5, color='blue')
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('PSD (A²/Hz)', fontsize=11)
ax.set_title('Sample #0 - PSD (Original Data)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

# 最初の5サンプルのPSD
ax = axes[0, 1]
for i in range(5):
    ax.loglog(frequencies[1:], x[i][1:], linewidth=1, alpha=0.7, label=f'Sample {i}')
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('PSD (A²/Hz)', fontsize=11)
ax.set_title('First 5 Samples - PSD Comparison', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# PSDの平均
ax = axes[1, 0]
mean_psd = np.mean(x, axis=0)
std_psd = np.std(x, axis=0)
ax.loglog(frequencies[1:], mean_psd[1:], linewidth=2, color='red', label='Mean')
ax.fill_between(frequencies[1:],
                 mean_psd[1:] - std_psd[1:],
                 mean_psd[1:] + std_psd[1:],
                 alpha=0.3, color='red', label='±1 std')
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('PSD (A²/Hz)', fontsize=11)
ax.set_title('Average PSD of 100 Samples', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# PSDヒストグラム（特定周波数での分布）
ax = axes[1, 1]
freq_idx = 100  # 周波数インデックス100での分布
ax.hist(x[:, freq_idx], bins=30, color='green', alpha=0.7, edgecolor='black')
ax.set_xlabel(f'PSD value at frequency index {freq_idx}', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Distribution of PSD Values at f={freq_idx} Hz', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/taiga/Desktop/noise/psd_visualization.png', dpi=150, bbox_inches='tight')
print("Saved psd_visualization.png")

# 可視化2: 変換後の時間領域信号
print("\nCreating time-domain visualization...")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# サンプル0の時間領域波形
ax = axes[0, 0]
time_points = np.arange(len(time_domain_signals[0]))
ax.plot(time_points, time_domain_signals[0], linewidth=0.5, color='blue')
ax.set_xlabel('Time (sampling points)', fontsize=11)
ax.set_ylabel('Voltage', fontsize=11)
ax.set_title('Sample #0 - Time Domain (Converted from PSD)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# サンプル0の最初の1000ポイント（ズーム）
ax = axes[0, 1]
ax.plot(time_points[:1000], time_domain_signals[0][:1000], linewidth=0.8, color='blue')
ax.set_xlabel('Time (sampling points)', fontsize=11)
ax.set_ylabel('Voltage', fontsize=11)
ax.set_title('Sample #0 - First 1000 Points (Zoomed)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 最初の5サンプルの重ね合わせ
ax = axes[1, 0]
for i in range(5):
    ax.plot(time_points[:2000], time_domain_signals[i][:2000],
            linewidth=0.6, alpha=0.7, label=f'Sample {i}')
ax.set_xlabel('Time (sampling points)', fontsize=11)
ax.set_ylabel('Voltage', fontsize=11)
ax.set_title('First 5 Samples - Overlay (First 2000 points)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 全100サンプルの連結波形（最初の10000ポイント）
ax = axes[1, 1]
concat_points = np.arange(10000)
ax.plot(concat_points, concatenated[:10000], linewidth=0.3, color='purple')
ax.set_xlabel('Time (sampling points)', fontsize=11)
ax.set_ylabel('Voltage', fontsize=11)
ax.set_title('Concatenated Signal - First 10,000 Points', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 振幅のヒストグラム
ax = axes[2, 0]
ax.hist(time_domain_signals.flatten(), bins=50, color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Voltage', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution of Time-Domain Amplitudes', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 統計情報
ax = axes[2, 1]
ax.axis('off')
stats_text = f"""
Time-Domain Signal Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Number of samples: {len(time_domain_signals)}
Points per sample: {len(time_domain_signals[0])}
Total data points: {len(concatenated):,}

Amplitude Range:
  Min: {time_domain_signals.min():.4e}
  Max: {time_domain_signals.max():.4e}
  Mean: {time_domain_signals.mean():.4e}
  Std: {time_domain_signals.std():.4e}

Original PSD Data:
  Min: {x.min():.4e}
  Max: {x.max():.4e}
"""
ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig('/Users/taiga/Desktop/noise/time_domain_visualization.png', dpi=150, bbox_inches='tight')
print("Saved time_domain_visualization.png")

print("\n✓ Complete!")
print(f"  - CSV: 100_samples_time_domain.csv ({len(concatenated):,} points)")
print(f"  - PSD visualization: psd_visualization.png")
print(f"  - Time-domain visualization: time_domain_visualization.png")
