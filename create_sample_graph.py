"""
data_lowF_noise.pickleから1サンプルを読み込んでグラフを作成するスクリプト
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

# データ読み込み
print("データを読み込んでいます...")
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

# 1サンプル目を取得
x_sample = data['x'][0]  # shape: (3000,)

# PyTorchテンソルの場合numpy配列に変換
if hasattr(x_sample, 'numpy'):
    x_sample = x_sample.numpy()
elif hasattr(x_sample, 'cpu'):
    x_sample = x_sample.cpu().numpy()

print(f"サンプルの形状: {x_sample.shape}")
print(f"最小値: {np.min(x_sample):.4f}")
print(f"最大値: {np.max(x_sample):.4f}")
print(f"平均値: {np.mean(x_sample):.4f}")
print(f"標準偏差: {np.std(x_sample):.4f}")

# グラフ作成
plt.figure(figsize=(14, 5))
plt.plot(x_sample, linewidth=0.5, color='blue', alpha=0.8)
plt.xlabel('Time (sampling points)', fontsize=12)
plt.ylabel('Voltage', fontsize=12)
plt.title('Low Frequency Noise Waveform - Sample #0 (Clean)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# 保存
output_path = 'sample_0_clean.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nグラフを保存しました: {output_path}")

# 表示（GUIがある環境なら）
# plt.show()
