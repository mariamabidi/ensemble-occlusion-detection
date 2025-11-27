"""
Script to run feature selection step
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_selection import rank_features
from src.config import *

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE SELECTION")
    print("=" * 70)

    # Load training data
    df_train = pd.read_csv(TRAIN_SET_PATH)

    feature_cols = [col for col in df_train.columns if col != "label"]
    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values

    # Rank features
    rankings = rank_features(X_train, y_train, feature_cols, FEATURE_SELECTION_DIR)

    print("\n✅ Feature selection complete!")
    print(f"📁 Rankings saved to: {FEATURE_SELECTION_DIR}")