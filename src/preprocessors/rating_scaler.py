import pandas as pd

from src.preprocessors.base_preprocessor import BasePreprocessor


class RatingScaler(BasePreprocessor):
    def process(
        self, ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from_min = self.params.get("from_min", 0)
        from_max = self.params.get("from_max", 10)
        to_min = self.params.get("to_min", 1)
        to_max = self.params.get("to_max", 5)

        scaled_ratings = ratings_df.copy()

        from_range = from_max - from_min
        to_range = to_max - to_min

        if from_range > 0:
            scaled_ratings["rating"] = to_min + ((scaled_ratings["rating"] - from_min) / from_range) * to_range

        return scaled_ratings, users_df, items_df
