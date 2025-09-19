from typing import Any, Dict, Optional

import pandas as pd

from sis_rec_experiments.loaders.base_loader import BaseDatasetLoader


class BookCrossingLoader(BaseDatasetLoader):
    """
    This dataset contains ratings of books from the Book-Crossing community.
    Includes explicit ratings, access history and demographic information of users.
    """
    DEFAULT_RATING_SCALE = (1.0, 10.0)

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.history_df: Optional[pd.DataFrame] = None
        self.users_info_df: Optional[pd.DataFrame] = None

    def load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "book_ratings.dat"

        if not ratings_file.exists():
            raise FileNotFoundError(f"Rating file not found: {ratings_file}")

        self.ratings_df = pd.read_csv(
            ratings_file, sep="\t", header=0, dtype={"user": "int32", "item": "int32", "rating": "float32"}
        )

        self.ratings_df = self.ratings_df.rename(columns={"user": "user_id", "item": "item_id"})

        return self.ratings_df

    def load_metadata(self) -> pd.DataFrame:
        info_file = self.dataset_path / "items_info.dat"

        if info_file.exists():
            try:
                self.metadata_df = pd.read_csv(info_file, sep="\t", encoding="utf-8", on_bad_lines="skip")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()

        return self.metadata_df

    def load_history(self) -> pd.DataFrame:
        history_file = self.dataset_path / "book_history.dat"

        if not history_file.exists():
            raise FileNotFoundError(f"History file not found: {history_file}")

        self.history_df = pd.read_csv(
            history_file, sep="\t", header=0, dtype={"user": "int32", "item": "int32", "accessed": "int8"}
        )

        self.history_df = self.history_df.rename(columns={"user": "user_id", "item": "item_id"})

        return self.history_df

    def load_users_info(self) -> pd.DataFrame:
        users_file = self.dataset_path / "users_info.dat"

        if users_file.exists():
            try:
                self.users_info_df = pd.read_csv(
                    users_file,
                    sep=r"\s+", # regex for multiple spaces
                    encoding="utf-8",
                    on_bad_lines="skip",
                    engine="python",
                )

                if "User-ID" in self.users_info_df.columns:
                    self.users_info_df["User-ID"] = pd.to_numeric(
                        self.users_info_df["User-ID"], errors="coerce"
                    ).astype("Int32")
                if "Age" in self.users_info_df.columns:
                    self.users_info_df["Age"] = pd.to_numeric(self.users_info_df["Age"], errors="coerce")

                self.users_info_df = self.users_info_df.rename(columns={"User-ID": "user_id"})
            except Exception as e:
                print(f"Error loading users info: {e}")
                self.users_info_df = pd.DataFrame()
        else:
            self.users_info_df = pd.DataFrame()

        return self.users_info_df

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()

        if self.history_df is None:
            self.load_history()
        if self.users_info_df is None:
            self.load_users_info()

        basic_stats = self.get_basic_stats()

        history_stats = {}
        if self.history_df is not None and not self.history_df.empty:
            history_stats = {
                "total_history_records": len(self.history_df),
                "unique_users_history": self.history_df["user_id"].nunique(),
                "unique_items_history": self.history_df["item_id"].nunique(),
                "avg_items_per_user_history": len(self.history_df) / max(self.history_df["user_id"].nunique(), 1),
            }

        users_stats = {}
        if self.users_info_df is not None and not self.users_info_df.empty:
            valid_ages = self.users_info_df["Age"][
                (self.users_info_df["Age"] >= 5) & (self.users_info_df["Age"] <= 100)
            ]

            users_stats = {
                "total_users_info": len(self.users_info_df),
                "users_with_age": len(valid_ages),
                "avg_user_age": valid_ages.mean() if len(valid_ages) > 0 else None,
                "min_user_age": valid_ages.min() if len(valid_ages) > 0 else None,
                "max_user_age": valid_ages.max() if len(valid_ages) > 0 else None,
            }

        info = {
            **basic_stats,
            **history_stats,
            **users_stats,
            "dataset_name": "Book Crossing",
            "dataset_type": "Book Ratings",
            "domain": "Books/Literature",
            "has_metadata": self.metadata_df is not None and not self.metadata_df.empty,
            "rating_scale": (1.0, 10.0),
            "data_format": "DAT (Tab-separated)",
        }

        return info
