import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# 1. Load Data and Recreate Feature Names List
df = pd.read_csv('../data_extraction/data/final_df.csv')

feature_names = [c for c in df.columns if
                 c not in ['target', 'pitcher', 'new_before_tj',
                           'player_name', 'league_level', 'game_date']]

# 2. Extract All Metrics from the Attention Hotspot (Patch 2)
patch_2_metrics = feature_names[32:48]

# 3. Focus pada subset 'Cedera' (Target 1)
injured_df = df[df['target'] == 1].copy()

results = []
for metric in patch_2_metrics:
    valid_data = injured_df[[metric, 'new_before_tj']].dropna()

    if len(valid_data) > 30:
        r_val, p_val = pearsonr(valid_data[metric], valid_data['new_before_tj'])
        results.append({'Metric': metric, 'r': r_val, 'p': p_val})

# Convert ke DataFrame dan urutkan berdasarkan nilai korelasi
results_df = pd.DataFrame(results).sort_values(by='r', ascending=True)

# 4. Visualisasi
plt.figure(figsize=(12, 8))

# Pewarnaan: Hijau kebiruan (signifikan), Abu-abu (tidak signifikan)
colors = ['teal' if p < 0.05 else 'lightgray' for p in results_df['p']]

bars = plt.barh(results_df['Metric'], results_df['r'],
                color=colors, edgecolor='black', alpha=0.8)

# Label p-value di sisi kanan (tetap)
x_min, x_max = plt.xlim()
label_x = x_max - (x_max - x_min) * 0.02

for i, bar in enumerate(bars):
    p_val = results_df.iloc[i]['p']

    plt.text(label_x,
             bar.get_y() + bar.get_height() / 2,
             f'p={p_val:.4f}',
             va='center',
             ha='right',
             fontsize=9,
             fontweight='bold')

# Garis referensi nol
plt.axvline(0, color='black', linewidth=1)

# Label dan judul (Bahasa Indonesia)
plt.xlabel("Koefisien Korelasi Pearson ($r$)", fontsize=12)
plt.title("Gambar 11: Analisis Arah Fisik\n(Metrik Patch 2 terhadap Hari Sebelum Cedera)",
          fontsize=14, fontweight='bold')

# Grid
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Legenda
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='teal', lw=4, label='Signifikan ($p < 0{,}05$)'),
    Line2D([0], [0], color='lightgray', lw=4, label='Tidak signifikan')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig("analisis_korelasi_pearson_patch2.png", dpi=300)
plt.show()

# 5. Ringkasan hasil signifikan
significant_results = results_df[results_df['p'] < 0.05].sort_values(by='p')

print("\n--- Ringkasan Hasil Signifikan (p < 0.05) ---")
print(significant_results)