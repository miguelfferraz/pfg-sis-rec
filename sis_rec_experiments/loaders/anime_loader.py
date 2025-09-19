from typing import Any, Dict, Optional

import pandas as pd

from sis_rec_experiments.loaders.base_loader import BaseDatasetLoader


class AnimeLoader(BaseDatasetLoader):
    """
    This dataset contains ratings of anime from MyAnimeList.
    Includes explicit ratings and access history.
    """

    DEFAULT_RATING_SCALE = (1.0, 10.0)

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.history_df: Optional[pd.DataFrame] = None

    def load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "anime_ratings.dat"

        if not ratings_file.exists():
            raise FileNotFoundError(f"Rating file not found: {ratings_file}")

        self.ratings_df = pd.read_csv(
            ratings_file, sep="\t", header=0, dtype={"User_ID": "int32", "Anime_ID": "int32", "Feedback": "float32"}
        )

        self.ratings_df = self.ratings_df.rename(
            columns={"User_ID": "user_id", "Anime_ID": "item_id", "Feedback": "rating"}
        )

        return self.ratings_df

    def load_metadata(self) -> pd.DataFrame:
        info_file = self.dataset_path / "anime_info.dat"

        if info_file.exists():
            try:
                self.metadata_df = pd.read_csv(info_file, sep="\t")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()

        return self.metadata_df

    def load_history(self) -> pd.DataFrame:
        history_file = self.dataset_path / "anime_history.dat"

        if not history_file.exists():
            raise FileNotFoundError(f"History file not found: {history_file}")

        self.history_df = pd.read_csv(
            history_file, sep="\t", header=0, dtype={"User_ID": "int32", "Anime_ID": "int32", "Feedback": "int8"}
        )

        self.history_df = self.history_df.rename(
            columns={"User_ID": "user_id", "Anime_ID": "item_id", "Feedback": "accessed"}
        )

        return self.history_df

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()

        if self.history_df is None:
            self.load_history()

        basic_stats = self.get_basic_stats()

        history_stats = {}
        if self.history_df is not None and not self.history_df.empty:
            history_stats = {
                "total_history_records": len(self.history_df),
                "unique_users_history": self.history_df["user_id"].nunique(),
                "unique_items_history": self.history_df["item_id"].nunique(),
                "avg_items_per_user_history": len(self.history_df) / max(self.history_df["user_id"].nunique(), 1),
            }

        info = {
            **basic_stats,
            **history_stats,
            "dataset_name": "Anime Recommendations",
            "dataset_type": "Entertainment Ratings",
            "domain": "Anime/Manga",
            "has_metadata": self.metadata_df is not None and not self.metadata_df.empty,
            "rating_scale": (1.0, 10.0),
            "data_format": "DAT",
        }

        return info
