import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# SNR設定
TARGET_SNR_DB = 5

# B5パラメータ：ポアソンショットノイズ
POISSON_RATE = 10  # 平均10イベント/秒
SHOT_AMPLITUDE = 5.0  # ショットイベントの振幅
SHOT_DECAY = 100  # 減衰定数（サンプルポイント）
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

# B5. ポアソンショットノイズ
print("\n[B5] Generating Poisson Shot Noise...")
# 時間軸（秒）
t_total = N / SAMPLING_RATE
# ポアソン過程でイベント時刻を生成
num_events = np.random.poisson(POISSON_RATE * t_total)
event_times = np.random.uniform(0, N, num_events).astype(int)

# ショットノイズ生成（各イベントから指数減衰）
shot_noise = np.zeros(N)
for event_time in event_times:
    for i in range(event_time, N):
        decay = np.exp(-(i - event_time) / SHOT_DECAY)
        shot_noise[i] += SHOT_AMPLITUDE * decay

# 信号の標準偏差でスケーリング
signal_std = np.std(original_signal)
shot_noise_scaled = shot_noise * signal_std / np.std(shot_noise) if np.std(shot_noise) > 0 else shot_noise

scale = calculate_noise_scale(original_signal, shot_noise_scaled, TARGET_SNR_DB)
shot_noise_scaled = shot_noise_scaled * scale
noisy_B5 = original_signal + shot_noise_scaled

print(f"  Poisson events: {num_events} over {t_total:.2f}s")

# CSV保存
df_B5 = pd.DataFrame({
    'Time_Point': time_points,
    'Voltage': noisy_B5
})
df_B5.to_csv('1_sample_B5_poisson_shot_noise.csv', index=False)
print("  Saved: 1_sample_B5_poisson_shot_noise.csv")

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
ax2.plot(time_points, noisy_B5, linewidth=0.7, color='red', alpha=0.8)
ax2.set_xlabel('Time (sampling points)', fontsize=14)
ax2.set_ylabel('Voltage', fontsize=14)
ax2.set_title('Noisy Signal (Poisson Shot Noise)', fontsize=12, fontweight='bold')
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
plt.savefig('1_sample_B5_poisson_shot_noise.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_sample_B5_poisson_shot_noise.png")

print("\n✓ B5 Complete!")
