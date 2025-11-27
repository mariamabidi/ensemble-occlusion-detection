"""
Script to run ensemble methods step
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble_methods import run_all_ensemble_methods
from src.config import *

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 6: ENSEMBLE METHODS")
    print("="*70)

    # Load datasets
    df_train = pd.read_csv(TRAIN_SET_PATH)
    df_val = pd.read_csv(VAL_SET_PATH)
    df_test = pd.read_csv(TEST_SET_PATH)

    y_train = df_train['label'].values
    X_train = df_train.drop(columns=['label'])

    y_val = df_val['label'].values
    X_val = df_val.drop(columns=['label'])

    y_test = df_test['label'].values
    X_test = df_test.drop(columns=['label'])

    # Run ensemble methods
    results, aggregator = run_all_ensemble_methods(
        X_train, y_train,
        X_test, y_test,
        X_val, y_val
    )

    print("\n✅ Ensemble methods complete!")