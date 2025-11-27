"""
ECG Data Preprocessing Module
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

from .config import *
from .utils import print_section_header

warnings.filterwarnings('ignore')


def load_and_label_diagnostics(diagnostics_path=DIAGNOSTICS_PATH):
    """
    Load diagnostics and create binary labels
    0 = Normal (Sinus rhythms)
    1 = Arrhythmia (Non-sinus rhythms)
    """
    print_section_header("LOADING DIAGNOSTICS")
    print(f"\n📂 Loading {diagnostics_path}...")
    df = pd.read_excel(diagnostics_path)
    print(f"   Total records: {len(df)}")

    def assign_label(rhythm):
        rhythm = rhythm.strip()
        if rhythm in NORMAL_RHYTHMS:
            return 0
        elif rhythm in ARRHYTHMIA_RHYTHMS:
            return 1
        else:
            return None

    df['label'] = df['Rhythm'].apply(assign_label)

    before_count = len(df)
    df = df[df['label'].notna()].copy()
    after_count = len(df)

    print(f"   Records with valid labels: {after_count}")
    print(f"   Records removed (unknown rhythm): {before_count - after_count}")

    print(f"\n📊 Label Distribution:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Normal" if label == 0 else "Arrhythmia"
        print(f"   {int(label)} ({label_name}): {count} patients ({count / len(df) * 100:.1f}%)")

    return df


def extract_ecg_features(ecg_file_path):
    """Extract statistical features from ECG CSV file"""
    try:
        df_ecg = pd.read_csv(ecg_file_path, header=None)
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        if df_ecg.shape[1] != 12:
            print(f"Warning: {ecg_file_path} has {df_ecg.shape[1]} leads instead of 12")
            return None

        features = {}

        for idx, lead_name in enumerate(lead_names):
            signal = df_ecg.iloc[:, idx].values

            features[f'{lead_name}_mean'] = np.mean(signal)
            features[f'{lead_name}_std'] = np.std(signal)
            features[f'{lead_name}_min'] = np.min(signal)
            features[f'{lead_name}_max'] = np.max(signal)
            features[f'{lead_name}_median'] = np.median(signal)
            features[f'{lead_name}_range'] = np.ptp(signal)

            q25, q75 = np.percentile(signal, [25, 75])
            features[f'{lead_name}_q25'] = q25
            features[f'{lead_name}_q75'] = q75
            features[f'{lead_name}_iqr'] = q75 - q25

            features[f'{lead_name}_skew'] = pd.Series(signal).skew()
            features[f'{lead_name}_kurtosis'] = pd.Series(signal).kurtosis()

            features[f'{lead_name}_rms'] = np.sqrt(np.mean(signal ** 2))
            features[f'{lead_name}_energy'] = np.sum(signal ** 2)

        return features

    except Exception as e:
        print(f"Error processing {ecg_file_path}: {e}")
        return None


def create_final_dataset(ecg_folder=ECG_DATA_DIR, diagnostics_path=DIAGNOSTICS_PATH,
                         output_csv=DATASET_PATH, save_interval=SAVE_INTERVAL):
    """Create final dataset with features and binary labels"""

    df_diag = load_and_label_diagnostics(diagnostics_path)

    ecg_path = Path(ecg_folder)
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG folder not found: {ecg_folder}")

    print(f"\n📁 ECG folder found: {ecg_folder}")

    all_data = []
    processed = 0
    skipped = 0

    print(f"\n🔄 Processing {len(df_diag)} ECG files...")
    print("=" * 70)

    for idx, row in tqdm(df_diag.iterrows(), total=len(df_diag), desc="Processing"):
        filename = row['FileName']
        ecg_file = ecg_path / f"{filename}.csv"

        if not ecg_file.exists():
            skipped += 1
            continue

        features = extract_ecg_features(ecg_file)

        if features is None:
            skipped += 1
            continue

        features['FileName'] = filename
        features['PatientAge'] = row['PatientAge']
        features['Gender'] = row['Gender']
        features['Rhythm'] = row['Rhythm']
        features['VentricularRate'] = row['VentricularRate']
        features['AtrialRate'] = row['AtrialRate']
        features['QRSDuration'] = row['QRSDuration']
        features['QTInterval'] = row['QTInterval']
        features['QTCorrected'] = row['QTCorrected']
        features['label'] = int(row['label'])

        all_data.append(features)
        processed += 1

        if processed % save_interval == 0:
            temp_df = pd.DataFrame(all_data)
            temp_df.to_csv(output_csv, index=False)
            print(f"\n💾 Saved progress: {processed} records processed")

    df_final = pd.DataFrame(all_data)
    df_final.to_csv(output_csv, index=False)

    print_section_header("PREPROCESSING COMPLETE!")
    print(f"✓ Successfully processed: {processed} patients")
    print(f"✗ Skipped (file not found): {skipped} patients")
    print(f"📄 Output file: {output_csv}")
    print(f"📊 Final dataset shape: {df_final.shape}")

    return df_final


def prepare_for_training(csv_path=DATASET_PATH, test_size=TEST_SIZE,
                         val_size=VAL_SIZE, random_state=RANDOM_SEED,
                         normalize=NORMALIZE, save_splits=True):
    """Prepare dataset for ML model training"""

    print_section_header("PREPARING DATASET FOR MODEL TRAINING")

    print(f"\n📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Dataset shape: {df.shape}")

    exclude_cols = ['FileName', 'Rhythm', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['label'].values

    if 'Gender' in X.columns:
        X['Gender'] = X['Gender'].map({'MALE': 1, 'FEMALE': 0})
        print(f"   ✓ Gender encoded: MALE=1, FEMALE=0")

    X = X.values

    # Check for missing values
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    if nan_count > 0 or inf_count > 0:
        print(f"   ⚠️  Replacing NaN/Inf with column mean...")
        X = pd.DataFrame(X, columns=feature_cols)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        X = X.values

    # Split datasets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    print(f"\n📊 Split Sizes:")
    print(f"   Training:   {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test:       {len(X_test)} samples")

    # Normalize features
    scaler = None
    if normalize:
        print(f"\n🔧 Normalizing features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        print(f"   ✓ Features normalized")

    # Save splits
    if save_splits:
        print(f"\n💾 Saving train/val/test splits...")

        pd.DataFrame(X_train, columns=feature_cols).assign(label=y_train).to_csv(
            TRAIN_SET_PATH, index=False)
        pd.DataFrame(X_val, columns=feature_cols).assign(label=y_val).to_csv(
            VAL_SET_PATH, index=False)
        pd.DataFrame(X_test, columns=feature_cols).assign(label=y_test).to_csv(
            TEST_SET_PATH, index=False)

        if scaler is not None:
            joblib.dump(scaler, SCALER_PATH)

        print(f"   ✓ Saved splits and scaler")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler
    }