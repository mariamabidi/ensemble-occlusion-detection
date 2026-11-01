# ECG Arrhythmia Classification 🫀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> Automated ECG arrhythmia detection using ensemble learning and consensus feature aggregation

A machine learning pipeline achieving **95.35% accuracy** for binary ECG classification (Normal vs. Arrhythmia) through advanced feature selection and ensemble methods.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Best Accuracy** | 95.35% (Stacking RF) |
| **Best Specificity** | 96.75% (Stacking Logistic) |
| **Feature Reduction** | 38.7% (163 → 100) |
| **Dataset Size** | 10,646 ECG recordings |
| **Workload Reduction** | 70% fewer manual reviews |

---

## ✨ Features

- ✅ **5 Feature Selection Methods**: Random Forest, Mutual Information, Chi-Square, ANOVA, RFE
- ✅ **5 Aggregation Techniques**: Mean Rank, Mean Weight, RRA, Threshold Algorithm, MedRank
- ✅ **13 ML Classifiers**: From LDA to Gradient Boosting
- ✅ **5 Ensemble Methods**: Hard/Soft Voting, Weighted Averaging, Stacking
- ✅ **Professional Visualizations**: Publication-ready plots
- ✅ **Modular Design**: Easy to extend and customize

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your files in the correct structure:

```
data/raw/
├── Diagnostics.xlsx          # Patient information
└── ECGDataDenoised/          # ECG CSV files
    ├── JS00001.csv
    ├── JS00002.csv
    └── ...
```

### 3. Run Pipeline

```bash
# Run complete pipeline (50-90 minutes)
python scripts/run_full_pipeline.py

# Or run individual steps
python scripts/run_preprocessing.py
python scripts/run_feature_selection.py
python scripts/run_training.py
python scripts/run_testing.py
python scripts/run_ensemble.py
python scripts/run_visualization.py
```

---

## 📁 Project Structure

```
ecg-arrhythmia-classification/
├── data/
│   ├── raw/                  # Input data
│   └── processed/            # Processed datasets
├── src/
│   ├── preprocessing.py      # Data preprocessing
│   ├── feature_selection.py  # Feature ranking
│   ├── feature_aggregation.py # Consensus features
│   ├── model_training.py     # Train classifiers
│   ├── model_testing.py      # Evaluate models
│   ├── ensemble_methods.py   # Ensemble techniques
│   └── visualization.py      # Create plots
├── scripts/
│   └── run_*.py             # Executable scripts
├── results/                  # Generated outputs
└── requirements.txt          # Dependencies
```

---

## 📊 Results

### Top 5 Models

| Rank | Model | Features | Accuracy | Specificity |
|------|-------|----------|----------|-------------|
| 1 | Stacking (RF) | All | **95.35%** | 95.90% |
| 2 | Stacking (Logistic) | All | **95.31%** | **96.75%** |
| 3 | Gradient Boosting | RRA | 94.55% | 96.81% |
| 4 | Gradient Boosting | All | 94.46% | 96.94% |
| 5 | Gradient Boosting | Mean Weight | 94.46% | 96.94% |

### Ensemble Methods

```
Method                  Accuracy    F1-Score    Sensitivity    Specificity
────────────────────────────────────────────────────────────────────────
Stacking (RF)           95.35%      95.38%      95.35%         95.90%
Stacking (Logistic)     95.31%      95.31%      95.31%         96.75%
Hard Voting             93.24%      93.16%      93.24%         96.68%
Weighted Averaging      93.10%      93.05%      93.10%         96.10%
Soft Voting             92.58%      92.52%      92.58%         95.84%
```

---

## 📧 Contact

**Mariam Abidi**  
📧 ma6267@rit.edu  
🎓 Rochester Institute of Technology  
👨‍🏫 Advisor: Dr. Linwei Wang

🔗 **Project**: [github.com/yourusername/ecg-arrhythmia-classification](https://github.com/yourusername/ecg-arrhythmia-classification)

---
