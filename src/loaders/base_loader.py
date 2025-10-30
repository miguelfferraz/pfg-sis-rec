from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from surprise import Dataset, Reader


class BaseDatasetLoader(ABC):
    DEFAULT_RATING_SCALE: tuple = None

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.ratings_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self._surprise_dataset: Optional[Dataset] = None
        self._validate_path()

    def _validate_path(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

    @abstractmethod
    def load_ratings(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        pass

    def to_surprise_dataset(self, rating_scale=None, reader=None) -> Dataset:
        self._validate_ratings_loaded()

        if self._surprise_dataset is not None:
            return self._surprise_dataset

        if rating_scale is None:
            if self.DEFAULT_RATING_SCALE is None:
                raise NotImplementedError(f"The class {self.__class__.__name__} must define DEFAULT_RATING_SCALE")
            rating_scale = self.DEFAULT_RATING_SCALE

        if reader is None:
            reader = Reader(rating_scale=rating_scale)

        surprise_data = self.ratings_df[["user_id", "item_id", "rating"]].copy()
        self._surprise_dataset = Dataset.load_from_df(surprise_data, reader)

        return self._surprise_dataset

    def get_basic_stats(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()

        stats = {
            "total_ratings": len(self.ratings_df),
            "unique_users": self.ratings_df["user_id"].nunique(),
            "unique_items": self.ratings_df["item_id"].nunique(),
            "rating_min": self.ratings_df["rating"].min(),
            "rating_max": self.ratings_df["rating"].max(),
            "rating_mean": self.ratings_df["rating"].mean(),
            "rating_std": self.ratings_df["rating"].std(),
            "sparsity": 1
            - (len(self.ratings_df) / (self.ratings_df["user_id"].nunique() * self.ratings_df["item_id"].nunique())),
        }

        return stats

    def _validate_ratings_loaded(self) -> None:
        if self.ratings_df is None or self.ratings_df.empty:
            raise ValueError("Ratings not loaded. Execute load_ratings() first.")
