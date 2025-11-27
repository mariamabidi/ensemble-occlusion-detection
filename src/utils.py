"""
Utility functions for ECG Arrhythmia Classification
"""
import os
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


def ensure_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_classifiers(seed=42):
    """
    Get dictionary of all classifiers to train

    Parameters:
    - seed: random seed for reproducibility

    Returns:
    - Dictionary of classifier names and instances
    """
    classifiers = {
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "Bagging": BaggingClassifier(random_state=seed),
        "Extra Trees Ensemble": ExtraTreesClassifier(random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(random_state=seed),
        "BNB": BernoulliNB(),
        "GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(random_state=seed, max_iter=500),
        "DTC": DecisionTreeClassifier(random_state=seed),
        "ETC": ExtraTreeClassifier(random_state=seed),
    }
    return classifiers


def print_section_header(title, width=70):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection_header(title, width=70):
    """Print a formatted subsection header"""
    print("\n" + "─" * width)
    print(title)
    print("─" * width)