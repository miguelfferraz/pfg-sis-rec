import pandas as pd

from src.preprocessors.base_preprocessor import BasePreprocessor


class RatingNormalizer(BasePreprocessor):
    def process(
        self, ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        method = self.params.get("method", "z_score")
        normalized_ratings = ratings_df.copy()

        if method == "z_score":
            mean = normalized_ratings["rating"].mean()
            std = normalized_ratings["rating"].std()
            if std > 0:
                normalized_ratings["rating"] = (normalized_ratings["rating"] - mean) / std
        elif method == "min_max":
            min_val = normalized_ratings["rating"].min()
            max_val = normalized_ratings["rating"].max()
            if max_val > min_val:
                normalized_ratings["rating"] = (normalized_ratings["rating"] - min_val) / (max_val - min_val)

        return normalized_ratings, users_df, items_df
