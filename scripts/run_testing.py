"""
Script to run model testing step
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_testing import (load_test_validation_data, test_all_models,
                               display_top_models)
from src.config import DISPLAY_TOP_N

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 5: MODEL TESTING")
    print("="*70)

    # Load data
    X_test, y_test, X_val, y_val, feature_names = load_test_validation_data()

    # Test all models
    df_test_results, df_val_results = test_all_models(
        X_test, y_test, X_val, y_val
    )

    # Display top models
    display_top_models(df_test_results, top_n=DISPLAY_TOP_N, metric='Accuracy')
    display_top_models(df_test_results, top_n=DISPLAY_TOP_N, metric='F1-Score')

    print("\n✅ Model testing complete!")