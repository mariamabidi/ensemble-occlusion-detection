"""
Model Training Module
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from .config import (FEATURE_AGGREGATION_DIR, TRAINED_MODELS_DIR,
                     RANDOM_SEED, TOP_N_FEATURES)
from .utils import get_classifiers, print_section_header


def evaluate_classifiers_on_aggregated_features(X, y, feature_names,
                                                agg_results_folder=FEATURE_AGGREGATION_DIR,
                                                top_n=TOP_N_FEATURES):
    """Train and evaluate classifiers on aggregated features"""

    classifiers = get_classifiers(RANDOM_SEED)
    results = []

    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()

    # Baseline: ALL features
    print_section_header("TRAINING ON ALL FEATURES (BASELINE)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    baseline_dir = TRAINED_MODELS_DIR / "all_features"
    baseline_dir.mkdir(exist_ok=True)

    for name, model in classifiers.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            sens = recall_score(y_test, y_pred, average="weighted")

            try:
                y_prob = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            except:
                auc = None

            joblib.dump(model, baseline_dir / f"{name}.joblib")

            results.append({
                "Aggregation Method": "all_features",
                "Classifier": name,
                "Accuracy": acc,
                "Sensitivity": sens,
                "F1-score": f1,
                "ROC-AUC": auc,
                "Num Features": X_df.shape[1]
            })

            print(f"✅ {name}: Acc={acc:.4f}")
        except Exception as e:
            print(f"⚠️ Error training {name}: {e}")

    # Aggregated features
    aggregation_files = [
        "mean_rank.csv",
        "mean_weight.csv",
        "rra.csv",
        "threshold_algorithm.csv",
        "medrank.csv"
    ]

    for file in aggregation_files:
        agg_path = Path(agg_results_folder) / file
        if not agg_path.exists():
            continue

        print_section_header(f"TRAINING ON {file}")

        agg_df = pd.read_csv(agg_path)
        top_features = agg_df["Feature"].head(top_n).values.tolist()
        available_features = [f for f in top_features if f in X_df.columns]

        if len(available_features) == 0:
            continue

        X_subset = X_df[available_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        method_name = file.replace(".csv", "")
        method_dir = TRAINED_MODELS_DIR / method_name
        method_dir.mkdir(exist_ok=True)

        # Save feature list
        with open(method_dir / "selected_features.txt", 'w') as f:
            f.write('\n'.join(available_features))

        for name, model in classifiers.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                sens = recall_score(y_test, y_pred, average="weighted")

                try:
                    y_prob = model.predict_proba(X_test)
                    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                except:
                    auc = None

                joblib.dump(model, method_dir / f"{name}.joblib")

                results.append({
                    "Aggregation Method": method_name,
                    "Classifier": name,
                    "Accuracy": acc,
                    "Sensitivity": sens,
                    "F1-score": f1,
                    "ROC-AUC": auc,
                    "Num Features": len(available_features)
                })

                print(f"   ✅ {name}: Acc={acc:.4f}")
            except Exception as e:
                print(f"   ⚠️ Error: {e}")

    results_df = pd.DataFrame(results)
    output_path = Path(agg_results_folder) / "model_performance_summary.csv"
    results_df.to_csv(output_path, index=False)

    print_section_header("TRAINING COMPLETE")
    print(f"📊 Results saved to: {output_path}")

    return results_df