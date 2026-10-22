import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Target 1 (Injured) Aggregate Statistics (Mean across 10 seeds)
# Replace these with your actual Target 1 averages from your logs
data_t1 = {
    'Window': ['60-Day', '60-Day', '80-Day', '80-Day'],
    'Metric': ['Recall (T1)', 'F1-Score (T1)', 'Recall (T1)', 'F1-Score (T1)'],
    'Mean_Score': [0.33, 0.36, 0.41, 0.43],
    'Std_Dev': [0.18660117898877, 0.10555093557141, 0.13477388471065, 0.074054034326294]
}

df_sens_t1 = pd.DataFrame(data_t1)

# 2. Plotting the Comparison (Section 4.2.7)
plt.figure(figsize=(10, 7))
ax = sns.barplot(data=df_sens_t1, x='Window', y='Mean_Score', hue='Metric', palette='rocket')

# Add error bars for Stability (Std Dev)
x_pos = [-0.2, 0.2, 0.8, 1.2]
plt.errorbar(x=x_pos, y=df_sens_t1['Mean_Score'], yerr=df_sens_t1['Std_Dev'],
             fmt='none', c='black', capsize=5)

plt.title("Target 1 (Cedera) Perbandingan Sensitivitas", fontsize=14, fontweight='bold')
plt.ylabel("Performance Score (Target 1 Only)")
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Annotation for the Seed 700 Exception
plt.annotate('Seed 700 Peak\n(Recall 0.78)', xy=(0, 0.44), xytext=(-0.35, 0.80),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10, bbox=dict(boxstyle="round", fc="0.95"))

plt.tight_layout()
plt.show()