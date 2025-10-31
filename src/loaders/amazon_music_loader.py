import json

import pandas as pd

from src.loaders.base_loader import BaseDatasetLoader


class AmazonMusicLoader(BaseDatasetLoader):
    """
    This dataset contains reviews of musical products from Amazon,
    including ratings and metadata of the products.
    """

    DEFAULT_RATING_SCALE = (1.0, 5.0)

    def _load_ratings(self) -> pd.DataFrame:
        json_file = self.dataset_path / "Digital_Music_5.json"

        if not json_file.exists():
            raise FileNotFoundError(f"Rating file not found: {json_file}")

        ratings_data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    ratings_data.append(
                        {
                            "user_id": review["reviewerID"],
                            "item_id": review["asin"],
                            "rating": float(review["overall"]),
                            "timestamp": review.get("unixReviewTime", None),
                            "helpful": review.get("helpful", [0, 0]),
                        }
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

        return pd.DataFrame(ratings_data)

    def _load_users(self) -> pd.DataFrame:
        if self._ratings_df is None:
            self._ratings_df = self._load_ratings()

        users_data = []
        for _, row in (
            self._ratings_df.groupby("user_id")
            .agg({"rating": ["count", "mean"], "timestamp": "min"})
            .reset_index()
            .iterrows()
        ):
            users_data.append(
                {
                    "user_id": row["user_id"],
                    "total_ratings": row[("rating", "count")],
                    "avg_rating": row[("rating", "mean")],
                    "first_review": row[("timestamp", "min")],
                }
            )

        return pd.DataFrame(users_data)

    def _load_items(self) -> pd.DataFrame:
        csv_file = self.dataset_path / "amazon_music_metadata.csv"

        if not csv_file.exists():
            if self._ratings_df is None:
                self._ratings_df = self._load_ratings()

            items_data = []
            for _, row in (
                self._ratings_df.groupby("item_id").agg({"rating": ["count", "mean"]}).reset_index().iterrows()
            ):
                items_data.append(
                    {
                        "item_id": row["item_id"],
                        "total_ratings": row[("rating", "count")],
                        "avg_rating": row[("rating", "mean")],
                    }
                )

            return pd.DataFrame(items_data)

        try:
            return pd.read_csv(csv_file)
        except Exception:
            return pd.DataFrame()
