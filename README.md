# crossleague-tjs-project

# Cross-League UCL Injury Risk Prediction in Baseball Pitchers

**A machine learning research project exploring Ulnar Collateral Ligament (UCL) injury risk in professional baseball pitchers using Statcast data, a Vision Transformer (ViT), and temporal sensitivity analysis.**

## Overview

Ulnar Collateral Ligament (UCL) injuries are a major concern in baseball, particularly among pitchers, and severe cases may require Tommy John Surgery (TJS). Identifying patterns associated with injury risk is a potential application of data-driven sports analytics.

This project investigates whether pitching data from Major League Baseball (MLB) and Triple-A (AAA) can be used to predict UCL injury risk using a Vision Transformer (ViT). Building upon the approach presented by Kang et al. (2025), the project incorporates more recent MLB data, adds AAA data, and evaluates how the observation window affects injury prediction performance.

The research compares five temporal windows — **20, 40, 60, 80, and 100 days** — to examine how the amount of available pre-injury data influences model performance.

## Research Objectives

- **Cross-league data integration:** Incorporate pitching data from MLB and AAA to explore injury prediction across different levels of professional baseball.
- **Temporal sensitivity analysis:** Compare five observation windows to evaluate how the temporal scope of the input data affects prediction performance.
- **Vision Transformer modeling:** Apply a ViT-based classification approach to structured pitching data.
- **Model interpretation:** Examine model attention patterns and Pearson correlations to investigate which pitching metrics are associated with injury labels.

## Dataset

The project uses Statcast pitching data covering the following periods:

| League | Data period |
|---|---|
| Major League Baseball (MLB) | 2017–2025 |
| Triple-A (AAA) | 2023–2025 |

The data pipeline combines pitching records with injury labels and pitcher-level information. Feature preparation includes pitcher-specific deviations from career averages, handedness normalization, physical measurements, missing-value handling, and outlier filtering.

The resulting dataset is used to construct model inputs for the temporal sensitivity experiments.

**Important:** The available injury cases are imbalanced between leagues, particularly in AAA. Consequently, cross-league results should be interpreted with this limitation in mind.

## Methodology

The project follows a data-processing, model-training, and evaluation pipeline.

1. **Data extraction:** Retrieve MLB and AAA pitching data using the project’s Statcast extraction scripts.
2. **Data integration:** Merge pitching records with injury labels and relevant pitcher information.
3. **Feature engineering and preprocessing:** Prepare pitching metrics, calculate pitcher-specific deviations, handle missing values, and normalize relevant attributes.
4. **Temporal window construction:** Prepare model inputs using 20-, 40-, 60-, 80-, and 100-day observation windows.
5. **Model training:** Train a Vision Transformer classifier with weighted binary cross-entropy to account for class imbalance.
6. **Performance evaluation:** Compare classification metrics and ROC-AUC across temporal windows and experimental runs.
7. **Interpretability analysis:** Examine attention patterns and Pearson correlations to explore relationships between pitching metrics and injury labels.

### Temporal Sensitivity Analysis

A central component of this research is the comparison of five observation windows. The experiments are meant to investigate how prediction performance varies with the amount of pre-injury data available.

The analysis considers injury-class recall, precision, F1-score, and ROC-AUC, alongside other classification metrics.

### Key Findings

- **The 80-day window achieved the highest mean injury-class recall (0.41) and mean F1-score (0.44)** among the tested windows.
- **The 60-day window achieved the highest peak injury-class recall (0.78)** in an individual experimental run. This is a peak result, not its average performance.
- Mean ROC-AUC ranged from 0.76 to 0.78, indicating that the tested windows produced broadly similar average discrimination performance.
- The results illustrate the importance of evaluating multiple temporal windows rather than selecting a window based on a single experimental run.

These findings describe the experiments conducted in this project; they do not establish a clinically validated injury-warning system.

## Introducing AAA Data

An important aspect of this project is the inclusion of Triple-A (AAA) data alongside Major League Baseball (MLB). The objective is not simply to maximize predictive performance on MLB data, but to investigate injury-risk patterns across different levels of professional baseball.

The results need to be interpreted in this context. MLB-only experiments produced stronger and more clearly differentiated performance across temporal windows, whereas the combined MLB–AAA experiments showed more similar average performance across the five observation periods.

One possible explanation is the difference in biomechanical consistency between the two competitive environments. As explored in this research, MLB pitchers exhibit relatively stable mechanical baselines, making deviations from their usual pitching patterns easier to examine. AAA pitchers, by comparison, exhibit greater biomechanical variability, potentially introducing additional complexity when the model attempts to identify patterns associated with injury risk across both populations.

**This variability is part of what makes cross-league analysis meaningful.** Integrating AAA data broadens the scope of the research beyond the more established MLB environment, but it also introduces challenges that may not be reflected in MLB-only experiments. Consequently, similar performance across temporal windows should not automatically be interpreted as a lack of value in the broader dataset.

This interpretation must remain provisional. The number of injured AAA pitchers in the dataset was very small, limiting the strength of conclusions about league-specific injury patterns. Further data would be needed to establish whether the observed biomechanical differences consistently explain the performance characteristics of the combined model.

## Model Interpretability

The project combines Vision Transformer attention analysis with Pearson correlation analysis to investigate patterns in the pitching metrics.

The attention analysis identified regions associated with pitch-movement and pitch-location deviation features, including vertical movement (`diff_pfx_z_*`) and plate-location metrics (`diff_plate_x_*` and `diff_plate_z_*`).

Pearson correlation analysis also identified statistically significant associations between some features and injury labels. However, the reported correlations were generally small.

These analyses provide exploratory insights into the model and dataset. Attention patterns and correlations should not be interpreted as proof of causal relationships or as evidence that an individual pitching metric directly causes a UCL injury.

## Repository Structure

```text
crossleague-tjs-project/
├── classification/
│   ├── Prepfortrain.py
│   ├── ViT_v2.py
│   ├── classification_train_v2.py
│   ├── cosmicannealing.py
│   ├── sensitivity_analysis.py
│   ├── global_attention.py
│   ├── holistic_analysis.py
│   ├── pearson_correlation.py
│   └── Visualization.py
├── data_extraction/
│   ├── Pybaseball_extract_MLB.py
│   ├── Pybaseball_extract_AAA.py
│   ├── minor_league_statcast_database.py
│   ├── sc_search_minor_leagues.py
│   ├── df_merge.py
│   └── final_audit.py
├── results/
│   └── _FINAL/
├── source_data/
├── .gitignore
└── README.md
```

The main directories serve the following purposes:

- **`data_extraction/`** — Scripts for retrieving, combining, and auditing MLB and AAA data.
- **`classification/`** — Data preparation, ViT training, temporal sensitivity experiments, visualization, attention analysis, and correlation analysis.
- **`results/_FINAL/`** — Final research outputs and experimental results.
- **`source_data/`** — Source data used by the research pipeline, subject to applicable data-access and redistribution restrictions.

The filenames above describe the main components; consult each script for its specific inputs, configuration, and execution requirements.

## Tools and Technologies

- **Python** — Data processing, experimentation, and analysis.
- **pybaseball / Statcast** — Baseball pitching data retrieval.
- **PyTorch and timm** — Deep learning and Vision Transformer implementation.
- **pandas and NumPy** — Data manipulation and numerical processing.
- **scikit-learn** — Classification evaluation and statistical analysis.
- **Matplotlib** — Result visualization.

*These technologies reflect the project’s implementation; compatible package versions and environment requirements should be established from the scripts and the environment used for the experiments.*

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/muhammadrizkifadhil95/crossleague-tjs-project.git
cd crossleague-tjs-project
```

### 2. Prepare the Python environment

Use a Python environment with the dependencies required by the project scripts. Review the imports and configuration in the `classification/` and `data_extraction/` directories before installing packages.

A pinned dependency file should be used if one is provided in a future release. The repository's scripts may require additional configuration or locally available data files.

### 3. Prepare the data

Run the appropriate extraction and preprocessing scripts after configuring the required data sources and paths. Some source datasets may need to be obtained separately, and their availability may depend on the original provider.

### 4. Run the experiments

Review the training and analysis scripts in `classification/`. Configure the input paths, output directories, and experiment settings before running the model or reproducing the reported analyses.

**Reproducibility note:** Exact reproduction of the thesis results requires the appropriate datasets, compatible dependencies, and matching experimental configurations.

## Limitations

- The number of injured AAA pitchers available for analysis was limited, restricting the strength of cross-league conclusions.
- Injury-class recall was lower on average than healthy-class recall, reflecting the challenge of identifying injury cases in an imbalanced dataset.
- The temporal-window results represent experimental findings and do not establish a universally optimal early-warning period.
- The model and interpretability analyses are research tools, not clinically validated diagnostic or injury-prevention systems.

## Reference

Kang et al. (2025). *Data-driven approaches for predicting Tommy John Surgery risk in Major League Baseball pitchers.*

This project builds upon the referenced research approach by incorporating more recent MLB data, adding AAA data, and investigating temporal sensitivity across multiple observation windows.

## Author

**Muhammad Rizki Fadhil**

Undergraduate research project — Informatics, Universitas Multimedia Nusantara.

[GitHub Repository](https://github.com/muhammadrizkifadhil95/crossleague-tjs-project)
