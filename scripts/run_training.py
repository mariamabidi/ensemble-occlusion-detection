"""
Script to run model training step
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_training import evaluate_classifiers_on_aggregated_features
from src.config import *

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)

    # Load training data
    df_train = pd.read_csv(TRAIN_SET_PATH)

    feature_cols = [col for col in df_train.columns if col != "label"]
    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values

    # Train models
    results_df = evaluate_classifiers_on_aggregated_features(
        X=X_train,
        y=y_train,
        feature_names=feature_cols,
        agg_results_folder=FEATURE_AGGREGATION_DIR,
        top_n=TOP_N_FEATURES
    )

    # Display best model
    best_model = results_df.loc[results_df["Accuracy"].idxmax()]

    print("\n🏆 Best Performing Model:")
    print(f"   - Aggregation Method: {best_model['Aggregation Method']}")
    print(f"   - Classifier: {best_model['Classifier']}")
    print(f"   - Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   - F1-Score: {best_model['F1-score']:.4f}")

    print("\n✅ Model training complete!")
    print(f"📁 Models saved to: {TRAINED_MODELS_DIR}")