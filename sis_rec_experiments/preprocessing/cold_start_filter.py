from typing import Dict, Tuple

import pandas as pd


class ColdStartFilter:
    def __init__(
        self,
        min_user_ratings: int = 20,
        min_item_ratings: int = 10,
        max_user_ratings: int = 500,
        max_iterations: int = 3,
    ):
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings
        self.max_user_ratings = max_user_ratings
        self.max_iterations = max_iterations

    def filter_dataset(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        original_stats = self._get_dataset_stats(ratings_df)
        filtered_df = ratings_df.copy()

        iteration_stats = []

        for iteration in range(self.max_iterations):
            initial_size = len(filtered_df)

            filtered_df = self._filter_users(filtered_df)
            filtered_df = self._filter_items(filtered_df)

            iteration_stats.append(
                {
                    "iteration": iteration,
                    "ratings_count": len(filtered_df),
                    "users_count": filtered_df["user_id"].nunique(),
                    "items_count": filtered_df["item_id"].nunique(),
                }
            )

            if len(filtered_df) == initial_size:
                break

        final_stats = self._get_dataset_stats(filtered_df)

        filtering_report = {
            "original_stats": original_stats,
            "final_stats": final_stats,
            "iterations_performed": len(iteration_stats),
            "iteration_details": iteration_stats,
            "reduction_summary": {
                "ratings_reduction": (original_stats["ratings_count"] - final_stats["ratings_count"])
                / original_stats["ratings_count"],
                "users_reduction": (original_stats["users_count"] - final_stats["users_count"])
                / original_stats["users_count"],
                "items_reduction": (original_stats["items_count"] - final_stats["items_count"])
                / original_stats["items_count"],
                "sparsity_change": final_stats["sparsity"] - original_stats["sparsity"],
            },
        }

        return filtered_df, filtering_report

    def _filter_users(self, df: pd.DataFrame) -> pd.DataFrame:
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[(user_counts >= self.min_user_ratings) & (user_counts <= self.max_user_ratings)].index
        return df[df["user_id"].isin(valid_users)]

    def _filter_items(self, df: pd.DataFrame) -> pd.DataFrame:
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_ratings].index
        return df[df["item_id"].isin(valid_items)]

    def _get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        n_users = df["user_id"].nunique()
        n_items = df["item_id"].nunique()
        n_ratings = len(df)
        sparsity = 1 - (n_ratings / (n_users * n_items))

        return {
            "ratings_count": n_ratings,
            "users_count": n_users,
            "items_count": n_items,
            "sparsity": sparsity,
            "density": 1 - sparsity,
            "avg_ratings_per_user": n_ratings / n_users if n_users > 0 else 0,
            "avg_ratings_per_item": n_ratings / n_items if n_items > 0 else 0,
        }
