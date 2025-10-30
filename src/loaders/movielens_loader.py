from typing import Any, Dict

import pandas as pd

from src.loaders.base_loader import BaseDatasetLoader


class MovieLensLoader(BaseDatasetLoader):
    """
    This dataset contains movie ratings from MovieLens 100k.
    Classic dataset with 100,000 ratings from 943 users on 1682 movies.
    """

    DEFAULT_RATING_SCALE = (1.0, 5.0)

    def load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "u.data"

        if not ratings_file.exists():
            raise FileNotFoundError(f"Rating file not found: {ratings_file}")

        self.ratings_df = pd.read_csv(
            ratings_file,
            sep="\t",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype={"user_id": "int32", "item_id": "int32", "rating": "float32", "timestamp": "int64"},
        )

        return self.ratings_df

    def load_metadata(self) -> pd.DataFrame:
        movies_file = self.dataset_path / "u.item"

        if movies_file.exists():
            try:
                self.metadata_df = pd.read_csv(
                    movies_file, sep="|", header=None, encoding="latin-1", on_bad_lines="skip"
                )

                if len(self.metadata_df.columns) >= 2:
                    self.metadata_df = self.metadata_df.rename(columns={0: "item_id", 1: "title", 2: "release_date"})
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()

        return self.metadata_df

    def load_user_demographics(self) -> pd.DataFrame:
        users_file = self.dataset_path / "u.user"

        if not users_file.exists():
            raise FileNotFoundError(f"User demographics file not found: {users_file}")

        demographics_df = pd.read_csv(
            users_file,
            sep="|",
            header=None,
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            dtype={"user_id": "int32", "age": "int32", "gender": "str", "occupation": "str", "zip_code": "str"},
        )

        return demographics_df

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()

        basic_stats = self.get_basic_stats()

        info = {
            **basic_stats,
            "dataset_name": "MovieLens 100k",
            "dataset_type": "Movie Ratings",
            "domain": "Movies/Entertainment",
            "has_metadata": self.metadata_df is not None and not self.metadata_df.empty,
            "has_demographics": True,
            "rating_scale": (1.0, 5.0),
            "data_format": "TAB (Tab-separated)",
            "description": "Classic MovieLens dataset with 100k ratings and user demographics",
        }

        return info
