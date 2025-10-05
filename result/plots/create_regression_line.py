import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

df = pd.read_parquet("/content/drive/MyDrive/modality_diff.parquet")

# Pick whichever column exists
col = "mean_diff" if "mean_diff" in df.columns else "modality_diff"

# Compute per-layer mean
layer_means = df.groupby("layer")[col].mean()
layers = layer_means.index.to_numpy()
mean_diffs = layer_means.values

# Plot
plt.figure(figsize=(10,5))
plt.plot(layers, mean_diffs, marker='o', linewidth=2, label='Mean Difference per Layer')
plt.title("Mean Difference Across Layers (LLaVA Model)", fontsize=14)
plt.xlabel("Layer (0–31)")
plt.ylabel("Mean Difference")
plt.xticks(range(0, 32))
plt.grid(True, linestyle='--', alpha=0.6)

# Regression line
slope, intercept, r, p, se = linregress(layers, mean_diffs)
plt.plot(layers, intercept + slope * layers, 'r--', label=f'Linear Fit (R²={r**2:.3f})')
plt.legend()
plt.tight_layout()

output_path = "/content/drive/MyDrive/plots/mean_diff_plots_png/mean_difference_vs_layers.png"
plt.savefig(output_path, dpi=300)
plt.show()
