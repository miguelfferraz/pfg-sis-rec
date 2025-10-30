from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StatsAnalyzer:
    @staticmethod
    def generate_comparison_report(filtering_report: Dict) -> str:
        original = filtering_report["original_stats"]
        final = filtering_report["final_stats"]
        reduction = filtering_report["reduction_summary"]

        report = f"""
COLD START FILTERING REPORT
{'='*50}

Original Dataset:
  Ratings: {original['ratings_count']:,}
  Users: {original['users_count']:,}
  Items: {original['items_count']:,}
  Sparsity: {original['sparsity']*100:.2f}%
  Density: {original['density']*100:.2f}%

Filtered Dataset:
  Ratings: {final['ratings_count']:,}
  Users: {final['users_count']:,}
  Items: {final['items_count']:,}
  Sparsity: {final['sparsity']*100:.2f}%
  Density: {final['density']*100:.2f}%

Reduction Summary:
  Ratings removed: {reduction['ratings_reduction']*100:.1f}%
  Users removed: {reduction['users_reduction']*100:.1f}%
  Items removed: {reduction['items_reduction']*100:.1f}%
  Sparsity change: {reduction['sparsity_change']*100:.2f}% points

Iterations performed: {filtering_report['iterations_performed']}
"""
        return report

    @staticmethod
    def plot_filtering_impact(
        original_df: pd.DataFrame, filtered_df: pd.DataFrame, filtering_report: Dict, save_path: str = None
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        ax1 = axes[0, 0]
        original_user_counts = original_df["user_id"].value_counts()
        filtered_user_counts = filtered_df["user_id"].value_counts()

        bins = np.logspace(0, 3, 20)
        ax1.hist(original_user_counts, bins=bins, alpha=0.7, label="Original", color="lightblue")
        ax1.hist(filtered_user_counts, bins=bins, alpha=0.7, label="Filtered", color="lightcoral")
        ax1.set_xscale("log")
        ax1.set_xlabel("Ratings per User")
        ax1.set_ylabel("Number of Users")
        ax1.set_title("User Activity Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        original_item_counts = original_df["item_id"].value_counts()
        filtered_item_counts = filtered_df["item_id"].value_counts()

        ax2.hist(original_item_counts, bins=bins, alpha=0.7, label="Original", color="lightblue")
        ax2.hist(filtered_item_counts, bins=bins, alpha=0.7, label="Filtered", color="lightcoral")
        ax2.set_xscale("log")
        ax2.set_xlabel("Ratings per Item")
        ax2.set_ylabel("Number of Items")
        ax2.set_title("Item Popularity Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        original_ratings = original_df["rating"].value_counts().sort_index()
        filtered_ratings = filtered_df["rating"].value_counts().sort_index()

        x = np.arange(len(original_ratings))
        width = 0.35

        ax3.bar(x - width / 2, original_ratings.values, width, label="Original", color="lightblue", alpha=0.7)
        ax3.bar(x + width / 2, filtered_ratings.values, width, label="Filtered", color="lightcoral", alpha=0.7)
        ax3.set_xlabel("Rating Value")
        ax3.set_ylabel("Count")
        ax3.set_title("Rating Value Distribution")
        ax3.set_xticks(x)
        ax3.set_xticklabels(original_ratings.index)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        original_stats = filtering_report["original_stats"]
        final_stats = filtering_report["final_stats"]

        metrics = ["Ratings", "Users", "Items"]
        original_values = [
            original_stats["ratings_count"],
            original_stats["users_count"],
            original_stats["items_count"],
        ]
        final_values = [final_stats["ratings_count"], final_stats["users_count"], final_stats["items_count"]]

        x = np.arange(len(metrics))
        width = 0.35

        ax4.bar(x - width / 2, original_values, width, label="Original", color="lightblue", alpha=0.7)
        ax4.bar(x + width / 2, final_values, width, label="Filtered", color="lightcoral", alpha=0.7)
        ax4.set_xlabel("Metric")
        ax4.set_ylabel("Count")
        ax4.set_title("Dataset Size Comparison")
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Filtering analysis plot saved: {save_path}")

        plt.show()
