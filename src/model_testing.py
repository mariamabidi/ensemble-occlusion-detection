"""
Model Testing Module
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_auc_score)
from tabulate import tabulate

from .config import (TRAINED_MODELS_DIR, TEST_RESULTS_DIR,
                     TRAIN_SET_PATH, VAL_SET_PATH, TEST_SET_PATH)
from .utils import print_section_header


def load_test_validation_data():
    """Load test and validation datasets"""
    print_section_header("LOADING TEST AND VALIDATION DATA")

    df_test = pd.read_csv(TEST_SET_PATH)
    y_test = df_test['label'].values
    X_test = df_test.drop(columns=['label'])

    df_val = pd.read_csv(VAL_SET_PATH)
    y_val = df_val['label'].values
    X_val = df_val.drop(columns=['label'])

    feature_names = X_test.columns.tolist()

    print(f"\n✅ Test set: {X_test.shape[0]} samples")
    print(f"✅ Val set:  {X_val.shape[0]} samples")

    return X_test, y_test, X_val, y_val, feature_names


def test_single_model(model_path, X_test, y_test, selected_features=None):
    """Test a single trained model"""
    try:
        model = joblib.load(model_path)

        if selected_features is not None:
            available_features = [f for f in selected_features if f in X_test.columns]
            if len(available_features) == 0:
                return None
            X_test_subset = X_test[available_features]
        else:
            X_test_subset = X_test

        y_pred = model.predict(X_test_subset)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        try:
            y_prob = model.predict_proba(X_test_subset)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            auc = None

        return {
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'roc_auc': auc
        }

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None


def test_all_models(X_test, y_test, X_val, y_val):
    """Test all trained models"""
    print_section_header("TESTING ALL TRAINED MODELS")

    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    test_results = []
    val_results = []

    models_path = Path(TRAINED_MODELS_DIR)
    aggregation_folders = [f for f in models_path.iterdir() if f.is_dir()]

    for agg_folder in aggregation_folders:
        agg_method = agg_folder.name
        print(f"\n{'─' * 70}")
        print(f"Testing: {agg_method}")

        feature_list_path = agg_folder / "selected_features.txt"
        selected_features = None

        if feature_list_path.exists():
            with open(feature_list_path, 'r') as f:
                selected_features = [line.strip() for line in f.readlines()]

        model_files = list(agg_folder.glob("*.joblib"))

        for model_file in model_files:
            classifier_name = model_file.stem

            # Test set
            test_metrics = test_single_model(model_file, X_test, y_test, selected_features)
            if test_metrics:
                test_results.append({
                    'Aggregation Method': agg_method,
                    'Classifier': classifier_name,
                    'Accuracy': test_metrics['accuracy'],
                    'F1-Score': test_metrics['f1_score'],
                    'Precision': test_metrics['precision'],
                    'Recall': test_metrics['recall'],
                    'Specificity': test_metrics['specificity'],
                    'ROC-AUC': test_metrics['roc_auc']
                })
                print(f"   ✅ {classifier_name}: Acc={test_metrics['accuracy']:.4f}")

            # Validation set
            val_metrics = test_single_model(model_file, X_val, y_val, selected_features)
            if val_metrics:
                val_results.append({
                    'Aggregation Method': agg_method,
                    'Classifier': classifier_name,
                    'Accuracy': val_metrics['accuracy'],
                    'F1-Score': val_metrics['f1_score'],
                    'Precision': val_metrics['precision'],
                    'Recall': val_metrics['recall'],
                    'Specificity': val_metrics['specificity'],
                    'ROC-AUC': val_metrics['roc_auc']
                })

    df_test_results = pd.DataFrame(test_results)
    df_val_results = pd.DataFrame(val_results)

    df_test_results.to_csv(TEST_RESULTS_DIR / 'test_set_results.csv', index=False)
    df_val_results.to_csv(TEST_RESULTS_DIR / 'val_set_results.csv', index=False)

    print_section_header("TESTING COMPLETE")

    return df_test_results, df_val_results


def display_top_models(df_results, top_n=10, metric='Accuracy'):
    """Display top performing models"""
    print(f"\n🏆 TOP {top_n} MODELS (by {metric})\n")

    df_sorted = df_results.sort_values(metric, ascending=False).head(top_n)
    display_df = df_sorted[['Aggregation Method', 'Classifier', 'Accuracy',
                            'F1-Score', 'Precision', 'Recall', 'Specificity']].copy()

    for col in ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Specificity']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    print(tabulate(display_df, headers='keys', tablefmt='fancy_grid', showindex=False))