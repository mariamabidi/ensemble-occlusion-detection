# ECG Arrhythmia Classification using Ensemble Learning 🫀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Consensus Feature Aggregation in Ensemble Learning for ECG-Based Arrhythmia Detection in Clinical Decision Making**

A comprehensive machine learning pipeline for automated ECG arrhythmia classification achieving **95.35% accuracy** through consensus feature aggregation and ensemble learning methods.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project implements a robust machine learning framework for binary ECG classification (Normal vs. Arrhythmia) using:

- **Multiple Feature Selection Methods**: Random Forest, Mutual Information, Chi-Square, ANOVA F-test, RFE
- **Feature Aggregation Techniques**: Mean Rank, Mean Weight, RRA, Threshold Algorithm, MedRank
- **13 Machine Learning Classifiers**: From linear models to ensemble methods
- **Advanced Ensemble Methods**: Hard/Soft Voting, Weighted Averaging, Stacking

### 🏥 Clinical Significance

- **High Accuracy**: 95.35% with stacking ensemble
- **Excellent Specificity**: 96.75% (minimal false positives)
- **Strong Sensitivity**: 95.31% (detects 95%+ of arrhythmias)
- **Workload Reduction**: 70% reduction in manual ECG review

---

## 🏆 Key Results

| Metric | Value | Model |
|--------|-------|-------|
| **Best Accuracy** | 95.35% | Stacking (Random Forest) |
| **Best Specificity** | 96.75% | Stacking (Logistic Regression) |
| **Best Individual Model** | 94.55% | RRA + Gradient Boosting |
| **Feature Reduction** | 38.7% | 163 → 100 features |
| **Dataset Size** | 10,646 | ECG recordings |

### 📊 Performance Summary

```
┌─────────────────────────┬──────────┬──────────┬─────────────┬─────────────┐
│ Method                  │ Accuracy │ F1-Score │ Sensitivity │ Specificity │
├─────────────────────────┼──────────┼──────────┼─────────────┼─────────────┤
│ Stacking (RF)          │  95.35%  │  95.38%  │   95.35%    │   95.90%    │
│ Stacking (Logistic)    │  95.31%  │  95.31%  │   95.31%    │   96.75%    │
│ RRA + Gradient Boost   │  94.55%  │  94.53%  │   94.55%    │   96.81%    │
│ Hard Voting            │  93.24%  │  93.16%  │   93.24%    │   96.68%    │
└─────────────────────────┴──────────┴──────────┴─────────────┴─────────────┘
```

---

## ✨ Features

### 🔬 Comprehensive Pipeline

- ✅ **Automated Feature Extraction**: 163 features from 12-lead ECG signals
- ✅ **5 Feature Selection Methods**: Multiple ranking approaches
- ✅ **5 Aggregation Techniques**: Consensus feature identification
- ✅ **13 ML Classifiers**: Extensive model comparison
- ✅ **5 Ensemble Methods**: Advanced prediction aggregation
- ✅ **Professional Visualizations**: Publication-ready plots
- ✅ **Modular Design**: Easy to extend and customize

### 📈 Advanced Features

- Feature importance analysis with clinical interpretation
- Class imbalance handling (72% Normal, 28% Arrhythmia)
- Cross-validation and stratified splitting
- Model performance tracking and comparison
- Automated hyperparameter optimization
- Comprehensive evaluation metrics

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- 5GB free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import numpy, pandas, sklearn; print('✅ Installation successful!')"
```

---

## ⚡ Quick Start

### 1. Prepare Your Data

Place your data files in the correct directories:

```
data/
├── raw/
│   ├── Diagnostics.xlsx          # Patient diagnostics
│   └── ECGDataDenoised/          # ECG CSV files
│       ├── JS00001.csv
│       ├── JS00002.csv
│       └── ...
```

### 2. Run Complete Pipeline

```bash
# Run all steps in sequence
python scripts/run_full_pipeline.py
```

This will execute:
1. ✅ Data preprocessing (15-30 min)
2. ✅ Feature selection (5-10 min)
3. ✅ Feature aggregation (1-2 min)
4. ✅ Model training (20-40 min)
5. ✅ Model testing (5-10 min)
6. ✅ Ensemble methods (2-5 min)

**Total time: ~50-90 minutes**

### 3. Generate Visualizations

```bash
python scripts/run_visualization.py
```

### 4. View Results

```bash
# Check test results
cat results/test_results/test_set_results.csv

# Check ensemble results
cat results/ensemble_results/ensemble_comparison.csv
```

---

## 📁 Project Structure

```
ecg-arrhythmia-classification/
│
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   │   ├── Diagnostics.xlsx
│   │   └── ECGDataDenoised/
│   └── processed/                 # Processed datasets
│       ├── train_set.csv
│       ├── val_set.csv
│       ├── test_set.csv
│       └── scaler.pkl
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── config.py                  # Configuration settings
│   ├── utils.py                   # Utility functions
│   ├── preprocessing.py           # Data preprocessing
│   ├── feature_selection.py       # Feature selection methods
│   ├── feature_aggregation.py     # Aggregation techniques
│   ├── model_training.py          # Model training
│   ├── model_testing.py           # Model evaluation
│   ├── ensemble_methods.py        # Ensemble methods
│   └── visualization.py           # Visualization tools
│
├── scripts/                       # Executable scripts
│   ├── run_preprocessing.py
│   ├── run_feature_selection.py
│   ├── run_aggregation.py
│   ├── run_training.py
│   ├── run_testing.py
│   ├── run_ensemble.py
│   ├── run_visualization.py
│   └── run_full_pipeline.py       # Complete pipeline
│
├── results/                       # Generated results
│   ├── feature_selection/
│   ├── feature_aggregation/
│   ├── trained_models/
│   ├── test_results/
│   └── ensemble_results/
│
├── notebooks/                     # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```

---

## 📖 Usage

### Run Individual Pipeline Steps

```bash
# Step 1: Preprocessing
python scripts/run_preprocessing.py

# Step 2: Feature Selection
python scripts/run_feature_selection.py

# Step 3: Feature Aggregation
python scripts/run_aggregation.py

# Step 4: Model Training
python scripts/run_training.py

# Step 5: Model Testing
python scripts/run_testing.py

# Step 6: Ensemble Methods
python scripts/run_ensemble.py

# Visualization
python scripts/run_visualization.py
```

### Use as Python Module

```python
from src.preprocessing import create_final_dataset, prepare_for_training
from src.model_training import evaluate_classifiers_on_aggregated_features
from src.visualization import create_all_visualizations

# Preprocess data
dataset = create_final_dataset()
data = prepare_for_training()

# Train models
results = evaluate_classifiers_on_aggregated_features(
    X=data['X_train'],
    y=data['y_train'],
    feature_names=data['feature_names']
)

# Create visualizations
create_all_visualizations()
```

### Customize Configuration

Edit `src/config.py` to modify settings:

```python
# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
TOP_N_FEATURES = 100

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
```

---

## 🔬 Methodology

### 1. Data Preprocessing

- **Feature Extraction**: 13 statistical features per lead (mean, std, quartiles, skewness, kurtosis, RMS, energy)
- **Clinical Features**: Age, Gender, Heart Rates, QRS Duration, QT Interval
- **Normalization**: StandardScaler (zero mean, unit variance)
- **Handling Missing Values**: Mean imputation
- **Train/Val/Test Split**: 72% / 8% / 20% (stratified)

### 2. Feature Selection Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| **Random Forest** | Gini impurity-based importance | O(n log n) |
| **Mutual Information** | Information-theoretic dependency | O(n²) |
| **Chi-Square** | Statistical independence test | O(n) |
| **ANOVA F-test** | Variance between classes | O(n) |
| **RFE (SVM)** | Recursive feature elimination | O(n²) |

### 3. Feature Aggregation Techniques

- **Mean Rank**: Average rank position
- **Mean Weight**: Average importance scores
- **Robust Rank Aggregation (RRA)**: Statistical p-value ranking ⭐
- **Threshold Algorithm**: Top-k selection
- **MedRank**: Consensus-based (20% threshold)

### 4. Machine Learning Models

**Linear Models**: LDA, QDA  
**Tree Ensembles**: Random Forest, Gradient Boosting, Extra Trees, AdaBoost, Bagging  
**Probabilistic**: Gaussian NB, Bernoulli NB  
**Instance-Based**: K-Nearest Neighbors  
**Neural Networks**: Multi-Layer Perceptron  
**Single Trees**: Decision Tree, Extra Tree  

### 5. Ensemble Methods

- **Hard Voting**: Majority vote
- **Soft Voting**: Average probabilities
- **Weighted Averaging**: Performance-based weights
- **Stacking (Logistic)**: Logistic regression meta-learner
- **Stacking (Random Forest)**: RF meta-learner ⭐

---

## 📊 Results

### Top 10 Individual Models (Test Set)

| Rank | Model | Features | Accuracy | F1-Score | Specificity |
|------|-------|----------|----------|----------|-------------|
| 1 | Gradient Boosting | RRA | 94.55% | 94.53% | 96.81% |
| 2 | Gradient Boosting | All | 94.46% | 94.42% | 96.94% |
| 3 | Gradient Boosting | Mean Weight | 94.46% | 94.42% | 96.94% |
| 4 | Gradient Boosting | Mean Rank | 94.23% | 94.21% | 96.36% |
| 5 | Random Forest | Mean Rank | 94.04% | 94.03% | 95.97% |
| 6 | Random Forest | All | 93.99% | 93.96% | 96.55% |
| 7 | Random Forest | Mean Weight | 93.94% | 93.91% | 96.49% |
| 8 | Random Forest | RRA | 93.71% | 93.69% | 95.97% |
| 9 | MLP | RRA | 93.62% | 93.62% | 95.51% |
| 10 | Extra Trees | Mean Rank | 93.52% | 93.48% | 96.23% |

### Ensemble Methods Comparison

| Method | Accuracy | F1-Score | Sensitivity | Specificity |
|--------|----------|----------|-------------|-------------|
| **Stacking (RF)** ⭐ | **95.35%** | **95.38%** | **95.35%** | **95.90%** |
| **Stacking (Logistic)** | **95.31%** | **95.31%** | **95.31%** | **96.75%** |
| Hard Voting | 93.24% | 93.16% | 93.24% | 96.68% |
| Weighted Averaging | 93.10% | 93.05% | 93.10% | 96.10% |
| Soft Voting | 92.58% | 92.52% | 92.58% | 95.84% |

### Feature Importance (Top 10)

1. **AtrialRate** - Primary rhythm indicator
2. **VentricularRate** - Heart rate variability
3. **V3_q25** - Precordial lead quartile
4. **V5_q25** - Lateral wall activity
5. **V4_q25** - Mid-precordial signal
6. **V2_q25** - Septal activity
7. **II_median** - Inferior lead measure
8. **V5_iqr** - Signal variability
9. **V1_iqr** - Right ventricular activity
10. **QTInterval** - Repolarization duration

---

## 📈 Visualization

The project includes comprehensive visualization tools:

### Generated Plots

1. **Accuracy vs. Sensitivity by Aggregation Method**
   - 6 subplots (one per feature set)
   - Shows classifier performance across different feature selections
   
2. **Specificity vs. F1-Score by Classifier**
   - 13 subplots (one per classifier)
   - Compares aggregation methods for each classifier

3. **Ensemble Methods Bar Plot**
   - Sensitivity and Specificity comparison
   - Easy identification of best ensemble technique

4. **Top 10 Models Comparison**
   - Bubble chart with size = sensitivity
   - Clear visualization of performance hierarchy

### Example

```python
from src.visualization import (
    plot_aggregation_accuracy_sensitivity,
    plot_classifier_specificity_f1score,
    plot_ensemble_comparison,
    create_all_visualizations
)

# Create all visualizations at once
create_all_visualizations()

# Or create specific plots
plot_aggregation_accuracy_sensitivity()
plot_classifier_specificity_f1score()
plot_ensemble_comparison()
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification

# Create development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black src/ scripts/
```

### Areas for Contribution

- 🔬 **Multi-class classification** (specific arrhythmia types)
- 🧠 **Deep learning integration** (CNN, LSTM features)
- ⚡ **Real-time inference optimization**
- 📊 **Additional visualization types**
- 🩺 **External dataset validation**
- 📚 **Documentation improvements**
- 🐛 **Bug fixes and optimizations**

---

## 📧 Contact

**Mariam Abidi**  
📧 Email: ma6267@rit.edu  
🎓 Rochester Institute of Technology  
👨‍🏫 Advisor: Dr. Linwei Wang

**Project Links:**
- 🔗 GitHub: [github.com/yourusername/ecg-arrhythmia-classification](https://github.com/yourusername/ecg-arrhythmia-classification)
- 📊 Research Poster: [Link to poster]
- 📄 Technical Report: [Link to report]

---

## 🙏 Acknowledgments

- **Dr. Linwei Wang** for guidance and supervision
- **Rochester Institute of Technology** for computational resources
- **ECG Dataset Contributors** for providing the clinical data
- **Open Source Community** for the amazing ML libraries (scikit-learn, pandas, matplotlib)

---

## 📌 Related Work

- [PhysioNet ECG Databases](https://physionet.org/)
- [A large scale 12-lead electrocardiogram database for arrhythmia study]([https://physionet.org/content/mitdb/](https://physionet.org/content/ecg-arrhythmia/1.0.0/))
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [ECG Signal Processing](https://github.com/topics/ecg-signal-processing)

---

## 🔖 Keywords

`ecg-classification` `arrhythmia-detection` `ensemble-learning` `feature-selection` `feature-aggregation` `machine-learning` `gradient-boosting` `random-forest` `medical-ai` `cardiac-diagnosis` `healthcare-ml` `clinical-decision-support` `signal-processing` `biomedical-engineering`

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Mariam Abidi](https://github.com/mariamabidi)

</div>
