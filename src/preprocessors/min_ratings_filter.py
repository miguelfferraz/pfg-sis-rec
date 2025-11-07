import pandas as pd

from src.preprocessors.base_preprocessor import BasePreprocessor


class MinRatingsFilter(BasePreprocessor):
    def process(
        self, ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        min_user_ratings = self.params.get("min_user_ratings", 0)
        min_item_ratings = self.params.get("min_item_ratings", 0)

        filtered_ratings = ratings_df.copy()

        if min_user_ratings > 0:
            user_counts = filtered_ratings["user_id"].value_counts()
            valid_users = user_counts[user_counts >= min_user_ratings].index
            filtered_ratings = filtered_ratings[filtered_ratings["user_id"].isin(valid_users)]

        if min_item_ratings > 0:
            item_counts = filtered_ratings["item_id"].value_counts()
            valid_items = item_counts[item_counts >= min_item_ratings].index
            filtered_ratings = filtered_ratings[filtered_ratings["item_id"].isin(valid_items)]

        valid_user_ids = filtered_ratings["user_id"].unique()
        valid_item_ids = filtered_ratings["item_id"].unique()

        filtered_users = users_df[users_df["user_id"].isin(valid_user_ids)]
        filtered_items = items_df[items_df["item_id"].isin(valid_item_ids)]

        return filtered_ratings, filtered_users, filtered_items
