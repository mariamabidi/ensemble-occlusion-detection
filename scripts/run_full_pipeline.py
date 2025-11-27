"""
Script to run the complete pipeline
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import create_final_dataset, prepare_for_training
from src.feature_selection import rank_features
from src.feature_aggregation import run_all_aggregations
from src.model_training import evaluate_classifiers_on_aggregated_features
from src.model_testing import (load_test_validation_data, test_all_models,
                               display_top_models)
from src.ensemble_methods import run_all_ensemble_methods
from src.config import *
from src.utils import print_section_header


def main():
    print_section_header("ECG ARRHYTHMIA CLASSIFICATION - FULL PIPELINE")

    # STEP 1: Preprocessing
    print_section_header("STEP 1/6: PREPROCESSING")
    df_final = create_final_dataset(
        ecg_folder=ECG_DATA_DIR,
        diagnostics_path=DIAGNOSTICS_PATH,
        output_csv=DATASET_PATH
    )

    data = prepare_for_training(
        csv_path=DATASET_PATH,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        normalize=NORMALIZE,
        save_splits=True
    )

    X_train = data['X_train']
    y_train = data['y_train']
    feature_names = data['feature_names']

    # STEP 2: Feature Selection
    print_section_header("STEP 2/6: FEATURE SELECTION")
    rankings = rank_features(X_train, y_train, feature_names, FEATURE_SELECTION_DIR)

    # STEP 3: Feature Aggregation
    print_section_header("STEP 3/6: FEATURE AGGREGATION")
    run_all_aggregations(top_n=DISPLAY_TOP_N)

    # STEP 4: Model Training
    print_section_header("STEP 4/6: MODEL TRAINING")
    results_df = evaluate_classifiers_on_aggregated_features(
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        agg_results_folder=FEATURE_AGGREGATION_DIR,
        top_n=TOP_N_FEATURES
    )

    best_model = results_df.loc[results_df["Accuracy"].idxmax()]
    print(f"\n🏆 Best Model: {best_model['Classifier']} "
          f"({best_model['Aggregation Method']}) - Acc: {best_model['Accuracy']:.4f}")

    # STEP 5: Model Testing
    print_section_header("STEP 5/6: MODEL TESTING")
    X_test, y_test, X_val, y_val, _ = load_test_validation_data()

    df_test_results, df_val_results = test_all_models(
        X_test, y_test, X_val, y_val
    )

    display_top_models(df_test_results, top_n=DISPLAY_TOP_N, metric='Accuracy')

    # STEP 6: Ensemble Methods
    print_section_header("STEP 6/6: ENSEMBLE METHODS")

    df_train = pd.read_csv(TRAIN_SET_PATH)
    X_train_df = df_train.drop(columns=['label'])
    y_train = df_train['label'].values

    ensemble_results, aggregator = run_all_ensemble_methods(
        X_train_df, y_train,
        X_test, y_test,
        X_val, y_val
    )

    print_section_header("PIPELINE COMPLETE!")
    print("\n📁 Results Summary:")
    print(f"   - Preprocessed data: {PROCESSED_DATA_DIR}")
    print(f"   - Feature selection: {FEATURE_SELECTION_DIR}")
    print(f"   - Feature aggregation: {FEATURE_AGGREGATION_DIR}")
    print(f"   - Trained models: {TRAINED_MODELS_DIR}")
    print(f"   - Test results: {TEST_RESULTS_DIR}")
    print(f"   - Ensemble results: {ENSEMBLE_RESULTS_DIR}")
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()