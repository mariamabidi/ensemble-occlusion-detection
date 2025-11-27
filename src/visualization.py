"""
Visualization Module for ECG Classification Results
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .config import TEST_RESULTS_DIR, ENSEMBLE_RESULTS_DIR
from .utils import print_section_header


def plot_aggregation_accuracy_sensitivity(test_results_path=None, output_folder=None):
    """
    Create scatter plots: Accuracy vs Sensitivity for each aggregation method
    Each plot shows different classifiers as different colored points

    Parameters:
    - test_results_path: path to test_set_results.csv
    - output_folder: folder to save plots
    """
    if test_results_path is None:
        test_results_path = TEST_RESULTS_DIR / 'test_set_results.csv'

    if output_folder is None:
        output_folder = TEST_RESULTS_DIR

    print_section_header("CREATING ACCURACY VS SENSITIVITY PLOTS (BY AGGREGATION METHOD)")

    # Load test results
    df = pd.read_csv(test_results_path)

    # Get unique aggregation methods and classifiers
    agg_methods = sorted(df['Aggregation Method'].unique())
    classifiers = sorted(df['Classifier'].unique())

    # Create color map for classifiers
    colors = plt.cm.tab20(np.linspace(0, 1, len(classifiers)))
    color_map = dict(zip(classifiers, colors))

    # Calculate grid size
    n_methods = len(agg_methods)
    n_cols = 3  # 3 columns
    n_rows = int(np.ceil(n_methods / n_cols))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    fig.suptitle('Accuracy vs Sensitivity by Aggregation Method',
                 fontsize=20, fontweight='bold', y=0.995)

    # Flatten axes for easier iteration
    if n_methods > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for idx, agg_method in enumerate(agg_methods):
        ax = axes_flat[idx]

        # Filter data for this aggregation method
        method_data = df[df['Aggregation Method'] == agg_method]

        # Plot each classifier
        for classifier in classifiers:
            clf_data = method_data[method_data['Classifier'] == classifier]
            if len(clf_data) > 0:
                ax.scatter(
                    clf_data['Accuracy'],
                    clf_data['Recall'],  # Recall = Sensitivity
                    label=classifier,
                    color=color_map[classifier],
                    s=200,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=1.5
                )

        # Add diagonal reference line (equal accuracy and sensitivity)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal Line')

        # Formatting
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sensitivity (Recall)', fontsize=12, fontweight='bold')
        ax.set_title(f'{agg_method}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set axis limits with padding
        if len(method_data) > 0:
            acc_min, acc_max = method_data['Accuracy'].min(), method_data['Accuracy'].max()
            sens_min, sens_max = method_data['Recall'].min(), method_data['Recall'].max()

            acc_range = acc_max - acc_min
            sens_range = sens_max - sens_min

            ax.set_xlim(
                max(0.0, acc_min - 0.05 * acc_range),
                min(1.0, acc_max + 0.05 * acc_range)
            )
            ax.set_ylim(
                max(0.0, sens_min - 0.05 * sens_range),
                min(1.0, sens_max + 0.05 * sens_range)
            )

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='lower right', ncol=2)

    # Hide unused subplots
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    # Save plot
    output_path = Path(output_folder) / 'accuracy_vs_sensitivity_by_aggregation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.show()

    return fig


def plot_classifier_specificity_f1score(test_results_path=None, output_folder=None):
    """
    Create scatter plots: Specificity vs F1-Score for each classifier
    Each plot shows different aggregation methods as different colored points

    Parameters:
    - test_results_path: path to test_set_results.csv
    - output_folder: folder to save plots
    """
    if test_results_path is None:
        test_results_path = TEST_RESULTS_DIR / 'test_set_results.csv'

    if output_folder is None:
        output_folder = TEST_RESULTS_DIR

    print_section_header("CREATING SPECIFICITY VS F1-SCORE PLOTS (BY CLASSIFIER)")

    # Load test results
    df = pd.read_csv(test_results_path)

    # Get unique classifiers and aggregation methods
    classifiers = sorted(df['Classifier'].unique())
    agg_methods = sorted(df['Aggregation Method'].unique())

    # Create color map for aggregation methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(agg_methods)))
    color_map = dict(zip(agg_methods, colors))

    # Calculate grid size
    n_classifiers = len(classifiers)
    n_cols = 4  # 4 columns
    n_rows = int(np.ceil(n_classifiers / n_cols))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5.5*n_rows))
    fig.suptitle('Specificity vs F1-Score by Classifier',
                 fontsize=20, fontweight='bold', y=0.998)

    # Flatten axes for easier iteration
    if n_classifiers > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for idx, classifier in enumerate(classifiers):
        ax = axes_flat[idx]

        # Filter data for this classifier
        clf_data = df[df['Classifier'] == classifier]

        # Plot each aggregation method
        for agg_method in agg_methods:
            method_data = clf_data[clf_data['Aggregation Method'] == agg_method]
            if len(method_data) > 0:
                ax.scatter(
                    method_data['Specificity'],
                    method_data['F1-Score'],
                    label=agg_method,
                    color=color_map[agg_method],
                    s=200,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=1.5
                )

        # Add diagonal reference line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal Line')

        # Formatting
        ax.set_xlabel('Specificity', fontsize=11, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{classifier}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set axis limits with padding
        if len(clf_data) > 0:
            spec_min, spec_max = clf_data['Specificity'].min(), clf_data['Specificity'].max()
            f1_min, f1_max = clf_data['F1-Score'].min(), clf_data['F1-Score'].max()

            spec_range = spec_max - spec_min
            f1_range = f1_max - f1_min

            ax.set_xlim(
                max(0.0, spec_min - 0.05 * spec_range),
                min(1.0, spec_max + 0.05 * spec_range)
            )
            ax.set_ylim(
                max(0.0, f1_min - 0.05 * f1_range),
                min(1.0, f1_max + 0.05 * f1_range)
            )

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right', ncol=1)

    # Hide unused subplots
    for idx in range(n_classifiers, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    # Save plot
    output_path = Path(output_folder) / 'specificity_vs_f1score_by_classifier.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.show()

    return fig


def plot_ensemble_comparison(ensemble_results_path=None, output_folder=None):
    """
    Create grouped bar plot comparing ensemble methods by Sensitivity and F1-Score

    Parameters:
    - ensemble_results_path: path to ensemble_comparison.csv
    - output_folder: folder to save plots
    """
    if ensemble_results_path is None:
        ensemble_results_path = ENSEMBLE_RESULTS_DIR / 'ensemble_comparison.csv'

    if output_folder is None:
        output_folder = ENSEMBLE_RESULTS_DIR

    print_section_header("CREATING ENSEMBLE COMPARISON BAR PLOT")

    # Load ensemble results
    df = pd.read_csv(ensemble_results_path)

    # Sort by F1-Score for better visualization
    df_sorted = df.sort_values('F1-Score', ascending=False)

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.suptitle('Prediction Aggregation Methods: Sensitivity & F1-Score Comparison',
                 fontsize=18, fontweight='bold', y=0.96)

    # Set up bar positions
    x = np.arange(len(df_sorted))
    width = 0.35  # Width of bars

    # Define colors
    color_sensitivity = '#1f77b4'  # Blue
    color_f1score = '#ff7f0e'  # Orange

    # Create grouped bars
    bars1 = ax.bar(x - width / 2,
                   df_sorted['Recall'] * 100,  # Sensitivity
                   width,
                   label='Sensitivity (Recall)',
                   color=color_sensitivity,
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=1.5)

    bars2 = ax.bar(x + width / 2,
                   df_sorted['F1-Score'] * 100,  # F1-Score
                   width,
                   label='F1-Score',
                   color=color_f1score,
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=1.5)

    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_xlabel('Ensemble Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Sensitivity and F1-Score by Ensemble Method', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['Method'], rotation=45, ha='right', fontsize=11)
    ax.set_ylim([90, 100])  # Focus on the relevant range
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=95, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='95% Threshold')
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()

    # Save plot
    output_path = Path(output_folder) / 'ensemble_methods_sensitivity_f1score_barplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.show()

    return fig


def create_all_visualizations(test_results_path=None, ensemble_results_path=None):
    """
    Create all visualization plots

    Parameters:
    - test_results_path: path to test_set_results.csv
    - ensemble_results_path: path to ensemble_comparison.csv
    """
    print_section_header("CREATING ALL VISUALIZATIONS")

    # Set default paths
    if test_results_path is None:
        test_results_path = TEST_RESULTS_DIR / 'test_set_results.csv'

    if ensemble_results_path is None:
        ensemble_results_path = ENSEMBLE_RESULTS_DIR / 'ensemble_comparison.csv'

    # Check if files exist
    if not Path(test_results_path).exists():
        print(f"❌ Test results not found: {test_results_path}")
        print("   Please run model testing first!")
        return

    # Create plots
    try:
        # Plot 1: Accuracy vs Sensitivity by Aggregation Method
        fig1 = plot_aggregation_accuracy_sensitivity(test_results_path, TEST_RESULTS_DIR)

        # Plot 2: Specificity vs F1-Score by Classifier
        fig2 = plot_classifier_specificity_f1score(test_results_path, TEST_RESULTS_DIR)

        # Plot 3: Ensemble comparison (if ensemble results exist)
        if Path(ensemble_results_path).exists():
            fig3 = plot_ensemble_comparison(ensemble_results_path, ENSEMBLE_RESULTS_DIR)
        else:
            print(f"\n⚠️  Ensemble results not found: {ensemble_results_path}")
            print("   Skipping ensemble visualization.")

        print_section_header("ALL VISUALIZATIONS COMPLETE")
        print(f"\n📁 Plots saved to:")
        print(f"   - {TEST_RESULTS_DIR}")
        print(f"   - {ENSEMBLE_RESULTS_DIR}")

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


def plot_top_models_comparison(test_results_path=None, top_n=10, output_folder=None):
    """
    Create a comparison plot of top N models

    Parameters:
    - test_results_path: path to test_set_results.csv
    - top_n: number of top models to display
    - output_folder: folder to save plot
    """
    if test_results_path is None:
        test_results_path = TEST_RESULTS_DIR / 'test_set_results.csv'

    if output_folder is None:
        output_folder = TEST_RESULTS_DIR

    print(f"\nCreating top {top_n} models comparison...")

    # Load test results
    df = pd.read_csv(test_results_path)

    # Get top N models by accuracy
    top_models = df.nlargest(top_n, 'Accuracy')

    # Create labels
    top_models['Model_Label'] = top_models['Classifier'] + '\n(' + top_models['Aggregation Method'] + ')'

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create scatter plot
    scatter = ax.scatter(
        top_models['Accuracy'],
        top_models['F1-Score'],
        s=top_models['Recall'] * 500,  # Size based on sensitivity
        c=range(len(top_models)),
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=2
    )

    # Add labels for each point
    for idx, row in top_models.iterrows():
        ax.annotate(
            f"{row['Classifier']}\n({row['Aggregation Method']})",
            (row['Accuracy'], row['F1-Score']),
            fontsize=8,
            ha='center'
        )

    ax.set_xlabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {top_n} Models: Accuracy vs F1-Score\n(Size = Sensitivity)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rank', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_path = Path(output_folder) / f'top_{top_n}_models_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    # When run directly, create all visualizations
    create_all_visualizations()