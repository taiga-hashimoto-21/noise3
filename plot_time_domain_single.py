import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading CSV file...")
df = pd.read_csv('/Users/taiga/Desktop/noise/100_samples_time_domain.csv')

print(f"Loaded {len(df):,} data points")
print(f"Data range: {df['Voltage'].min():.4e} to {df['Voltage'].max():.4e}")

# 1つのグラフにすべてのデータをプロット
print("\nCreating single graph...")
fig, ax = plt.subplots(figsize=(20, 6))

time_points = df['Time_Point'].values
voltage = df['Voltage'].values

ax.plot(time_points, voltage, linewidth=0.3, color='blue', alpha=0.8)
ax.set_xlabel('Time (sampling points)', fontsize=14)
ax.set_ylabel('Voltage', fontsize=14)
ax.set_title('100 Samples Time-Domain Signal (600,000 points)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('/Users/taiga/Desktop/noise/100_samples_time_domain_single.png', dpi=150, bbox_inches='tight')
print("Saved: 100_samples_time_domain_single.png")

# 統計情報も表示
print(f"\nStatistics:")
print(f"  Total points: {len(df):,}")
print(f"  Min voltage: {voltage.min():.4e}")
print(f"  Max voltage: {voltage.max():.4e}")
print(f"  Mean voltage: {voltage.mean():.4e}")
print(f"  Std voltage: {voltage.std():.4e}")

print("\n✓ Complete!")
