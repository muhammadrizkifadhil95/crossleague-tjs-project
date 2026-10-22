import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. DATA PREPARATION
# =========================
df = pd.read_csv('../data_extraction/data/final_df.csv')

injured_df = df[df['target'] == 1].copy()
injured_df['bin'] = (injured_df['new_before_tj'] // 5) * 5

# =========================
# 2. METRICS CONFIG
# =========================
metrics = [
    ('diff_pfx_z_CH', 'Perubahan Vertikal (CH)'),
    ('diff_pfx_z_SI', 'Perubahan Vertikal (SI)'),
    ('diff_pfx_z_FF', 'Perubahan Vertikal (FF)')
]

colors = {
    'MLB': '#002D72',
    'AAA': '#D50032'
}

# =========================
# 3. PLOTTING FUNCTION (MANUAL LOOP)
# =========================
for metric, metric_label in metrics:

    fig, ax = plt.subplots(figsize=(10, 5))

    for level in ['MLB', 'AAA']:
        subset = injured_df[injured_df['league_level'] == level]

        binned_stats = (
            subset.groupby('bin')[metric]
            .agg(['mean', 'sem'])
            .reset_index()
            .sort_values('bin')
        )

        ax.plot(
            binned_stats['bin'],
            binned_stats['mean'],
            marker='o',
            linewidth=2,
            label=level,
            color=colors[level]
        )

        ax.fill_between(
            binned_stats['bin'],
            binned_stats['mean'] - binned_stats['sem'],
            binned_stats['mean'] + binned_stats['sem'],
            alpha=0.15,
            color=colors[level]
        )

    # =========================
    # 4. FORMATTING (JOURNAL STYLE)
    # =========================
    ax.set_title(f"{metric_label} (MLB vs AAA)", fontsize=12)
    ax.set_xlabel("Hari Sebelum TJS", fontsize=11)
    ax.set_ylabel("Perubahan Metrik", fontsize=11)

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title="Level Liga")

    ax.invert_xaxis()  # Injury timeline direction

    # =========================
    # 5. SAVE FIGURE
    # =========================
    filename = f"{metric}_mlb_vs_aaa.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()