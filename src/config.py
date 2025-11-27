"""
Configuration file for ECG Arrhythmia Classification
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURE_SELECTION_DIR = RESULTS_DIR / "feature_selection"
FEATURE_AGGREGATION_DIR = RESULTS_DIR / "feature_aggregation"
TRAINED_MODELS_DIR = RESULTS_DIR / "trained_models"
TEST_RESULTS_DIR = RESULTS_DIR / "test_results"
ENSEMBLE_RESULTS_DIR = RESULTS_DIR / "ensemble_results"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR,
                  FEATURE_SELECTION_DIR, FEATURE_AGGREGATION_DIR,
                  TRAINED_MODELS_DIR, TEST_RESULTS_DIR, ENSEMBLE_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
DIAGNOSTICS_PATH = RAW_DATA_DIR / "Diagnostics.xlsx"
ECG_DATA_DIR = RAW_DATA_DIR / "ECGDataDenoised"
DATASET_PATH = PROCESSED_DATA_DIR / "ecg_arrhythmia_dataset.csv"
TRAIN_SET_PATH = PROCESSED_DATA_DIR / "train_set.csv"
VAL_SET_PATH = PROCESSED_DATA_DIR / "val_set.csv"
TEST_SET_PATH = PROCESSED_DATA_DIR / "test_set.csv"
SCALER_PATH = PROCESSED_DATA_DIR / "scaler.pkl"

# Rhythm classifications
NORMAL_RHYTHMS = ['SR', 'SB', 'ST', 'SI', 'SA', 'SAAWR']
ARRHYTHMIA_RHYTHMS = ['AFIB', 'AF', 'SVT', 'AT', 'AVNRT', 'AVRT']

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
NORMALIZE = True

# Feature selection parameters
TOP_N_FEATURES = 100

# Training parameters
SAVE_INTERVAL = 500

# Display parameters
DISPLAY_TOP_N = 10