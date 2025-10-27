import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading CSV file...")
df = pd.read_csv('/Users/taiga/Desktop/noise/1_sample_time_domain.csv')

print(f"Loaded {len(df):,} data points")
print(f"Data range: {df['Voltage'].min():.4e} to {df['Voltage'].max():.4e}")

# グラフ作成（縦横比 3:2）
print("\nCreating adjusted visualization...")
fig, ax = plt.subplots(figsize=(15, 10))  # 3:2の比率

time_points = df['Time_Point'].values
voltage = df['Voltage'].values

ax.plot(time_points, voltage, linewidth=0.7, color='blue', alpha=0.8)
ax.set_xlabel('Time (sampling points)', fontsize=14)
ax.set_ylabel('Voltage', fontsize=14)
# タイトルを削除
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('/Users/taiga/Desktop/noise/1_sample_time_domain.png', dpi=150, bbox_inches='tight')
print("Saved: 1_sample_time_domain.png (updated)")

print("\n✓ Complete!")
