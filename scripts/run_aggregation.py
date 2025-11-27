"""
Script to run feature aggregation step
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_aggregation import run_all_aggregations
from src.config import DISPLAY_TOP_N

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 3: FEATURE AGGREGATION")
    print("="*70)

    run_all_aggregations(top_n=DISPLAY_TOP_N)

    print("\n✅ Feature aggregation complete!")