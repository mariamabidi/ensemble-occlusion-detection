"""
Script to create all visualizations for ECG classification results
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    create_all_visualizations,
    plot_aggregation_accuracy_sensitivity,
    plot_classifier_specificity_f1score,
    plot_ensemble_comparison,
    plot_top_models_comparison
)
from src.config import TEST_RESULTS_DIR, ENSEMBLE_RESULTS_DIR
from src.utils import print_section_header


def main():
    print_section_header("VISUALIZATION PIPELINE")

    print("\n📊 Available Visualizations:")
    print("   1. Accuracy vs Sensitivity (by Aggregation Method)")
    print("   2. Specificity vs F1-Score (by Classifier)")
    print("   3. Ensemble Methods Comparison")
    print("   4. Top Models Comparison")
    print("   5. Create All Visualizations")

    choice = input("\nEnter your choice (1-5) or press Enter for all: ").strip()

    test_results = TEST_RESULTS_DIR / 'test_set_results.csv'
    ensemble_results = ENSEMBLE_RESULTS_DIR / 'ensemble_comparison.csv'

    # Check if test results exist
    if not test_results.exists():
        print(f"\n❌ Error: Test results not found at {test_results}")
        print("   Please run model testing first:")
        print("   python scripts/run_testing.py")
        return

    try:
        if choice == '1':
            print_section_header("CREATING ACCURACY VS SENSITIVITY PLOTS")
            plot_aggregation_accuracy_sensitivity(test_results, TEST_RESULTS_DIR)

        elif choice == '2':
            print_section_header("CREATING SPECIFICITY VS F1-SCORE PLOTS")
            plot_classifier_specificity_f1score(test_results, TEST_RESULTS_DIR)

        elif choice == '3':
            if ensemble_results.exists():
                print_section_header("CREATING ENSEMBLE COMPARISON PLOTS")
                plot_ensemble_comparison(ensemble_results, ENSEMBLE_RESULTS_DIR)
            else:
                print(f"\n❌ Error: Ensemble results not found at {ensemble_results}")
                print("   Please run ensemble methods first:")
                print("   python scripts/run_ensemble.py")

        elif choice == '4':
            print_section_header("CREATING TOP MODELS COMPARISON")
            top_n = input("Enter number of top models to display (default=10): ").strip()
            top_n = int(top_n) if top_n else 10
            plot_top_models_comparison(test_results, top_n, TEST_RESULTS_DIR)

        else:  # choice == '5' or empty
            print_section_header("CREATING ALL VISUALIZATIONS")
            create_all_visualizations(test_results, ensemble_results)

            # Also create top models comparison
            print("\n" + "─" * 70)
            plot_top_models_comparison(test_results, 10, TEST_RESULTS_DIR)

        print_section_header("VISUALIZATION COMPLETE")
        print("\n✅ All plots have been saved!")
        print(f"\n📁 Check the following folders:")
        print(f"   - {TEST_RESULTS_DIR}")
        print(f"   - {ENSEMBLE_RESULTS_DIR}")

    except Exception as e:
        print(f"\n❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()