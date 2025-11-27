"""
Script to run preprocessing step
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import create_final_dataset, prepare_for_training
from src.config import *

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING ECG DATA")
    print("="*70)

    # Create dataset
    df_final = create_final_dataset(
        ecg_folder=ECG_DATA_DIR,
        diagnostics_path=DIAGNOSTICS_PATH,
        output_csv=DATASET_PATH,
        save_interval=SAVE_INTERVAL
    )

    # Prepare for training
    data = prepare_for_training(
        csv_path=DATASET_PATH,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        normalize=NORMALIZE,
        save_splits=True
    )

    print("\n✅ Preprocessing complete!")
    print(f"📁 Train set: {TRAIN_SET_PATH}")
    print(f"📁 Val set: {VAL_SET_PATH}")
    print(f"📁 Test set: {TEST_SET_PATH}")