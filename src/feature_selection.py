"""
Feature Selection Module
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, RFE
from sklearn.svm import SVC

from .config import FEATURE_SELECTION_DIR
from .utils import ensure_dir, print_section_header


def rank_features(X, y, feature_names, output_dir=FEATURE_SELECTION_DIR):
    """Rank features using multiple methods"""
    ensure_dir(output_dir)
    rankings = {}

    print_section_header("FEATURE RANKING")

    # Random Forest
    print("\n1️⃣ Random Forest Feature Importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rankings["RandomForest"] = sorted(
        zip(feature_names, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    # Mutual Information
    print("2️⃣ Mutual Information...")
    mi = mutual_info_classif(X, y, random_state=42)
    rankings["MutualInfo"] = sorted(
        zip(feature_names, mi), key=lambda x: x[1], reverse=True
    )

    # Chi-Square
    print("3️⃣ Chi-Square...")
    X_pos = np.abs(X)
    chi2_scores, _ = chi2(X_pos, y)
    rankings["ChiSquare"] = sorted(
        zip(feature_names, chi2_scores), key=lambda x: x[1], reverse=True
    )

    # ANOVA F-value
    print("4️⃣ ANOVA F-test...")
    f_scores, _ = f_classif(X, y)
    rankings["ANOVA"] = sorted(
        zip(feature_names, f_scores), key=lambda x: x[1], reverse=True
    )

    # RFE
    print("5️⃣ RFE with SVM...")
    svc = SVC(kernel="linear", random_state=42)
    rfe = RFE(estimator=svc, n_features_to_select=10, step=0.2)
    rfe.fit(X, y)
    rankings["RFE"] = sorted(
        zip(feature_names, rfe.ranking_), key=lambda x: x[1]
    )

    # Save rankings
    for method, ranked_list in rankings.items():
        df_rank = pd.DataFrame(ranked_list, columns=["Feature", "Score/Rank"])
        df_rank.to_csv(f"{output_dir}/{method}_ranking.csv", index=False)
        print(f"   ✅ Saved {method} ranking")

    return rankings