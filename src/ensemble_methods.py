"""
Ensemble Methods Module
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, confusion_matrix, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import TRAINED_MODELS_DIR, ENSEMBLE_RESULTS_DIR, RANDOM_SEED
from .utils import print_section_header, print_subsection_header


class PredictionAggregator:
    """Aggregates predictions from multiple trained models"""

    def __init__(self, models_root=TRAINED_MODELS_DIR):
        self.models_root = models_root
        self.aggregation_methods = {}
        self.feature_subsets = {}

    def load_all_models(self):
        """Load all trained models"""
        print_section_header("LOADING ALL TRAINED MODELS")

        models_path = Path(self.models_root)
        aggregation_folders = [f for f in models_path.iterdir() if f.is_dir()]

        for agg_folder in aggregation_folders:
            agg_method = agg_folder.name
            print(f"\n📁 Loading from: {agg_method}")

            feature_list_path = agg_folder / "selected_features.txt"
            if feature_list_path.exists():
                with open(feature_list_path, 'r') as f:
                    self.feature_subsets[agg_method] = [line.strip() for line in f.readlines()]
                print(f"   ✅ Loaded {len(self.feature_subsets[agg_method])} selected features")
            else:
                self.feature_subsets[agg_method] = None
                print(f"   ✅ Using all features")

            self.aggregation_methods[agg_method] = {}
            model_files = list(agg_folder.glob("*.joblib"))

            for model_file in model_files:
                classifier_name = model_file.stem
                try:
                    model = joblib.load(model_file)
                    self.aggregation_methods[agg_method][classifier_name] = model
                    print(f"   ✅ Loaded: {classifier_name}")
                except Exception as e:
                    print(f"   ❌ Failed to load {classifier_name}: {e}")

        print(f"\n{'='*70}")
        print(f"✅ Loaded {len(self.aggregation_methods)} aggregation methods")
        print(f"{'='*70}")

    def get_predictions(self, X, agg_method, classifier_name, return_proba=True):
        """Get predictions from a specific model"""
        try:
            model = self.aggregation_methods[agg_method][classifier_name]
            features = self.feature_subsets[agg_method]

            if features is not None:
                available_features = [f for f in features if f in X.columns]
                if len(available_features) == 0:
                    return None
                X_subset = X[available_features]
            else:
                X_subset = X

            if return_proba and hasattr(model, 'predict_proba'):
                return model.predict_proba(X_subset)
            else:
                return model.predict(X_subset)

        except Exception as e:
            return None

    def hard_voting(self, X, agg_method=None, classifiers=None):
        """
        Hard Voting: Each model votes for a class, majority wins
        """
        if agg_method is None:
            agg_methods = list(self.aggregation_methods.keys())
        else:
            agg_methods = [agg_method]

        all_predictions = []

        for agg in agg_methods:
            if classifiers is None:
                clf_names = list(self.aggregation_methods[agg].keys())
            else:
                clf_names = classifiers

            for clf in clf_names:
                pred = self.get_predictions(X, agg, clf, return_proba=False)
                if pred is not None:
                    all_predictions.append(pred)

        if len(all_predictions) == 0:
            return None

        stacked = np.column_stack(all_predictions)
        final_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=1, arr=stacked
        )

        return final_pred

    def soft_voting(self, X, agg_method=None, classifiers=None):
        """
        Soft Voting: Average predicted probabilities, then take argmax
        """
        if agg_method is None:
            agg_methods = list(self.aggregation_methods.keys())
        else:
            agg_methods = [agg_method]

        all_probabilities = []

        for agg in agg_methods:
            if classifiers is None:
                clf_names = list(self.aggregation_methods[agg].keys())
            else:
                clf_names = classifiers

            for clf in clf_names:
                proba = self.get_predictions(X, agg, clf, return_proba=True)
                if proba is not None:
                    all_probabilities.append(proba)

        if len(all_probabilities) == 0:
            return None, None

        avg_proba = np.mean(all_probabilities, axis=0)
        final_pred = np.argmax(avg_proba, axis=1)

        return final_pred, avg_proba

    def weighted_averaging(self, X, y, agg_method=None, classifiers=None,
                          validation_X=None, validation_y=None):
        """
        Weighted Averaging: Weight models by their validation performance
        """
        if agg_method is None:
            agg_methods = list(self.aggregation_methods.keys())
        else:
            agg_methods = [agg_method]

        # Calculate weights using validation set or training set
        if validation_X is not None and validation_y is not None:
            X_weight = validation_X
            y_weight = validation_y
        else:
            X_weight = X
            y_weight = y

        weights = []
        all_probabilities = []
        model_names = []

        for agg in agg_methods:
            if classifiers is None:
                clf_names = list(self.aggregation_methods[agg].keys())
            else:
                clf_names = classifiers

            for clf in clf_names:
                proba = self.get_predictions(X_weight, agg, clf, return_proba=True)
                if proba is not None:
                    # Calculate weight based on accuracy
                    pred = np.argmax(proba, axis=1)
                    acc = accuracy_score(y_weight, pred)
                    weights.append(acc)
                    model_names.append(f"{agg}_{clf}")

                    # Get predictions on actual data
                    test_proba = self.get_predictions(X, agg, clf, return_proba=True)
                    if test_proba is not None:
                        all_probabilities.append(test_proba)

        if len(all_probabilities) == 0:
            return None, None, None

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average of probabilities
        weighted_proba = np.zeros_like(all_probabilities[0])
        for i, proba in enumerate(all_probabilities):
            weighted_proba += weights[i] * proba

        final_pred = np.argmax(weighted_proba, axis=1)

        # Create weights dictionary
        weights_dict = dict(zip(model_names, weights))

        return final_pred, weighted_proba, weights_dict

    def stacking(self, X_train, y_train, X_test, agg_method=None,
                classifiers=None, meta_learner='logistic'):
        """
        Stacking: Train a meta-learner on base model predictions

        Parameters:
        - X_train: training features
        - y_train: training labels
        - X_test: test features
        - agg_method: specific aggregation method (if None, uses all)
        - classifiers: list of classifier names (if None, uses all)
        - meta_learner: 'logistic' or 'rf' (random forest)

        Returns:
        - final predictions, meta-model
        """
        if agg_method is None:
            agg_methods = list(self.aggregation_methods.keys())
        else:
            agg_methods = [agg_method]

        # Get base model predictions on training set
        base_train_preds = []

        for agg in agg_methods:
            if classifiers is None:
                clf_names = list(self.aggregation_methods[agg].keys())
            else:
                clf_names = classifiers

            for clf in clf_names:
                proba = self.get_predictions(X_train, agg, clf, return_proba=True)
                if proba is not None:
                    base_train_preds.append(proba)

        if len(base_train_preds) == 0:
            return None, None

        # Stack base predictions
        X_meta_train = np.column_stack([pred.flatten() if pred.ndim == 1
                                       else pred for pred in base_train_preds])

        # Train meta-learner
        if meta_learner == 'logistic':
            meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        elif meta_learner == 'rf':
            meta_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner}")

        meta_model.fit(X_meta_train, y_train)

        # Get base model predictions on test set
        base_test_preds = []

        for agg in agg_methods:
            if classifiers is None:
                clf_names = list(self.aggregation_methods[agg].keys())
            else:
                clf_names = classifiers

            for clf in clf_names:
                proba = self.get_predictions(X_test, agg, clf, return_proba=True)
                if proba is not None:
                    base_test_preds.append(proba)

        # Stack test predictions
        X_meta_test = np.column_stack([pred.flatten() if pred.ndim == 1
                                      else pred for pred in base_test_preds])

        # Meta-model prediction
        final_pred = meta_model.predict(X_meta_test)

        return final_pred, meta_model


def evaluate_ensemble(y_true, y_pred, y_proba=None, method_name="Ensemble"):
    """
    Evaluate ensemble predictions
    """
    metrics = {
        'Method': method_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted')
    }

    # Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        metrics['Specificity'] = None

    # ROC-AUC
    if y_proba is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except:
            metrics['ROC-AUC'] = None
    else:
        metrics['ROC-AUC'] = None

    return metrics


def run_all_ensemble_methods(X_train, y_train, X_test, y_test, X_val, y_val):
    """Run all ensemble methods and compare results"""
    print_section_header("PREDICTION AGGREGATION PIPELINE")

    os.makedirs(ENSEMBLE_RESULTS_DIR, exist_ok=True)

    aggregator = PredictionAggregator()
    aggregator.load_all_models()

    results = []

    # 1. Hard Voting
    print_subsection_header("1️⃣  HARD VOTING")
    pred_hard = aggregator.hard_voting(X_test)
    if pred_hard is not None:
        metrics = evaluate_ensemble(y_test, pred_hard, method_name="Hard Voting")
        results.append(metrics)
        print(f"✅ Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")

    # 2. Soft Voting
    print_subsection_header("2️⃣  SOFT VOTING")
    pred_soft, proba_soft = aggregator.soft_voting(X_test)
    if pred_soft is not None:
        metrics = evaluate_ensemble(y_test, pred_soft, proba_soft, method_name="Soft Voting")
        results.append(metrics)
        print(f"✅ Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")

    # 3. Weighted Averaging
    print_subsection_header("3️⃣  WEIGHTED AVERAGING")
    pred_weighted, proba_weighted, weights = aggregator.weighted_averaging(
        X_test, y_test, validation_X=X_val, validation_y=y_val
    )
    if pred_weighted is not None:
        metrics = evaluate_ensemble(y_test, pred_weighted, proba_weighted,
                                   method_name="Weighted Averaging")
        results.append(metrics)
        print(f"✅ Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")

        # Save weights
        weights_df = pd.DataFrame(list(weights.items()), columns=['Model', 'Weight'])
        weights_df = weights_df.sort_values('Weight', ascending=False)
        weights_path = os.path.join(ENSEMBLE_RESULTS_DIR, 'model_weights.csv')
        weights_df.to_csv(weights_path, index=False)
        print(f"💾 Saved model weights to: {weights_path}")

    # 4. Stacking with Logistic Regression
    print_subsection_header("4️⃣  STACKING (Logistic Regression)")
    pred_stack_lr, meta_lr = aggregator.stacking(
        X_train, y_train, X_test, meta_learner='logistic'
    )
    if pred_stack_lr is not None:
        metrics = evaluate_ensemble(y_test, pred_stack_lr,
                                   method_name="Stacking (Logistic)")
        results.append(metrics)
        print(f"✅ Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")

        # Save meta-model
        meta_path = os.path.join(ENSEMBLE_RESULTS_DIR, 'meta_model_logistic.joblib')
        joblib.dump(meta_lr, meta_path)
        print(f"💾 Saved meta-model to: {meta_path}")

    # 5. Stacking with Random Forest
    print_subsection_header("5️⃣  STACKING (Random Forest)")
    pred_stack_rf, meta_rf = aggregator.stacking(
        X_train, y_train, X_test, meta_learner='rf'
    )
    if pred_stack_rf is not None:
        metrics = evaluate_ensemble(y_test, pred_stack_rf,
                                   method_name="Stacking (RF)")
        results.append(metrics)
        print(f"✅ Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")

        # Save meta-model
        meta_path = os.path.join(ENSEMBLE_RESULTS_DIR, 'meta_model_rf.joblib')
        joblib.dump(meta_rf, meta_path)
        print(f"💾 Saved meta-model to: {meta_path}")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    results_path = os.path.join(ENSEMBLE_RESULTS_DIR, 'ensemble_comparison.csv')
    df_results.to_csv(results_path, index=False)

    # Display results
    print_section_header("ENSEMBLE METHODS COMPARISON")
    from tabulate import tabulate
    display_df = df_results.copy()
    for col in ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Specificity']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    print(tabulate(display_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    print_section_header("ENSEMBLE AGGREGATION COMPLETE")
    print(f"📁 Results saved in '{ENSEMBLE_RESULTS_DIR}/' folder")

    return df_results, aggregator