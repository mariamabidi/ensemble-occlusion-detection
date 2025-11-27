"""
Feature Aggregation Module
"""
import os
import glob
import numpy as np
import pandas as pd
from tabulate import tabulate

from .config import FEATURE_AGGREGATION_DIR, FEATURE_SELECTION_DIR
from .utils import print_section_header


def mean_rank_aggregation(ranking_folder=FEATURE_SELECTION_DIR,
                          output_path=FEATURE_AGGREGATION_DIR, top_n=15):
    """Aggregate features using mean rank"""
    os.makedirs(output_path, exist_ok=True)

    files = glob.glob(f"{ranking_folder}/*.csv")
    rank_dfs = []

    for file in files:
        df = pd.read_csv(file)
        df["Rank"] = df.index + 1
        df = df[["Feature", "Rank"]]
        rank_dfs.append(df)

    combined = rank_dfs[0]
    for df in rank_dfs[1:]:
        combined = pd.merge(combined, df, on="Feature", how="outer", suffixes=(None, "_x"))

    rank_cols = [col for col in combined.columns if "Rank" in col]
    combined["MeanRank"] = combined[rank_cols].mean(axis=1)
    combined = combined.sort_values("MeanRank").reset_index(drop=True)
    combined["RankedPosition"] = combined.index + 1

    combined.to_csv(f"{output_path}/mean_rank.csv", index=False)

    print("\n📊 Top Features by Mean Rank:")
    table = combined[["RankedPosition", "Feature", "MeanRank"]].head(top_n)
    print(tabulate(table, headers="keys", tablefmt="fancy_grid", showindex=False))

    return combined


def mean_weight_aggregation(ranking_folder=FEATURE_SELECTION_DIR,
                            output_path=FEATURE_AGGREGATION_DIR, top_n=15):
    """Aggregate features using mean weight"""
    os.makedirs(output_path, exist_ok=True)

    files = glob.glob(f"{ranking_folder}/*.csv")
    dfs = [pd.read_csv(file) for file in files]

    combined = dfs[0]
    for df in dfs[1:]:
        combined = pd.merge(combined, df, on="Feature", how="outer", suffixes=(None, "_x"))

    score_cols = [col for col in combined.columns if "Score" in col]
    combined["MeanWeight"] = combined[score_cols].mean(axis=1)
    combined = combined.sort_values("MeanWeight", ascending=False).reset_index(drop=True)
    combined["RankedPosition"] = combined.index + 1

    combined.to_csv(f"{output_path}/mean_weight.csv", index=False)

    print("\n📊 Top Features by Mean Weight:")
    table = combined[["RankedPosition", "Feature", "MeanWeight"]].head(top_n)
    print(tabulate(table, headers="keys", tablefmt="fancy_grid", showindex=False))

    return combined


def robust_rank_aggregation(ranking_folder=FEATURE_SELECTION_DIR,
                            output_path=FEATURE_AGGREGATION_DIR, top_n=15):
    """Robust Rank Aggregation (RRA)"""
    os.makedirs(output_path, exist_ok=True)

    files = glob.glob(f"{ranking_folder}/*.csv")
    rank_dfs = []

    for file in files:
        df = pd.read_csv(file)
        df["Rank"] = df.index + 1
        rank_dfs.append(df[["Feature", "Rank"]])

    combined = rank_dfs[0]
    for df in rank_dfs[1:]:
        combined = pd.merge(combined, df, on="Feature", how="outer", suffixes=(None, "_x"))

    rank_cols = [col for col in combined.columns if "Rank" in col]
    n_features = len(combined)

    for col in rank_cols:
        combined[col] = combined[col] / n_features

    combined["RRA_pvalue"] = combined[rank_cols].mean(axis=1)
    combined["RRA_pvalue"] = combined["RRA_pvalue"].rank() / n_features
    combined = combined.sort_values("RRA_pvalue").reset_index(drop=True)
    combined["RankedPosition"] = combined.index + 1

    combined.to_csv(f"{output_path}/rra.csv", index=False)

    print("\n📊 Top Features by RRA:")
    table = combined[["RankedPosition", "Feature", "RRA_pvalue"]].head(top_n)
    print(tabulate(table, headers="keys", tablefmt="fancy_grid", showindex=False))

    return combined


def threshold_algorithm(ranking_folder=FEATURE_SELECTION_DIR,
                        output_path=FEATURE_AGGREGATION_DIR, k=100, top_n=100):
    """Threshold Algorithm"""
    os.makedirs(output_path, exist_ok=True)

    files = glob.glob(f"{ranking_folder}/*.csv")
    dfs = []

    for file in files:
        df = pd.read_csv(file)
        df["Score"] = df["Score/Rank"].rank(ascending=False)
        dfs.append(df[["Feature", "Score"]])

    combined = dfs[0]
    for df in dfs[1:]:
        combined = pd.merge(combined, df, on="Feature", how="outer", suffixes=(None, "_x"))

    score_cols = [col for col in combined.columns if "Score" in col]
    combined["MeanScore"] = combined[score_cols].mean(axis=1)
    combined = combined.sort_values("MeanScore", ascending=False).reset_index(drop=True)

    k = k or int(np.mean([len(df) for df in dfs]) / 5)
    threshold = combined["MeanScore"].iloc[k - 1]

    top_features = combined[combined["MeanScore"] >= threshold].copy()
    top_features["RankedPosition"] = range(1, len(top_features) + 1)

    top_features.to_csv(f"{output_path}/threshold_algorithm.csv", index=False)

    print("\n📊 Top Features by Threshold Algorithm:")
    table = top_features[["RankedPosition", "Feature", "MeanScore"]].head(top_n)
    print(tabulate(table, headers="keys", tablefmt="fancy_grid", showindex=False))

    return top_features


def medrank_aggregation(ranking_folder=FEATURE_SELECTION_DIR,
                        output_path=FEATURE_AGGREGATION_DIR,
                        min_presence=0.2, top_n=15):
    """MedRank Algorithm"""
    os.makedirs(output_path, exist_ok=True)

    files = glob.glob(f"{ranking_folder}/*.csv")
    dfs = [pd.read_csv(file) for file in files]
    n_lists = len(dfs)

    feature_counts = {}
    for df in dfs:
        for f in df["Feature"].values:
            feature_counts[f] = feature_counts.get(f, 0) + 1

    medrank_df = pd.DataFrame(list(feature_counts.items()),
                              columns=["Feature", "Frequency"])
    medrank_df["PresenceFraction"] = medrank_df["Frequency"] / n_lists

    selected = medrank_df[medrank_df["PresenceFraction"] >= min_presence].copy()
    selected = selected.sort_values("PresenceFraction", ascending=False).reset_index(drop=True)
    selected["RankedPosition"] = selected.index + 1

    selected.to_csv(f"{output_path}/medrank.csv", index=False)

    print("\n📊 Top Features by MedRank:")
    table = selected[["RankedPosition", "Feature", "PresenceFraction"]].head(top_n)
    print(tabulate(table, headers="keys", tablefmt="fancy_grid", showindex=False))

    return selected


def run_all_aggregations(top_n=10):
    """Run all aggregation methods"""
    print_section_header("FEATURE AGGREGATION")

    mean_rank_aggregation(top_n=top_n)
    mean_weight_aggregation(top_n=top_n)
    robust_rank_aggregation(top_n=top_n)
    threshold_algorithm(top_n=top_n)
    medrank_aggregation(top_n=top_n)

    print("\n✅ All aggregation methods complete!")