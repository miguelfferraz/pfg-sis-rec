from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from surprise import Dataset, Reader


class BaseDatasetLoader(ABC):
    DEFAULT_RATING_SCALE: tuple = None

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._ratings_df: Optional[pd.DataFrame] = None
        self._users_df: Optional[pd.DataFrame] = None
        self._items_df: Optional[pd.DataFrame] = None
        self._surprise_dataset: Optional[Dataset] = None
        self._validate_path()

    def _validate_path(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

    @abstractmethod
    def _load_ratings(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _load_users(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _load_items(self) -> pd.DataFrame:
        pass

    def get_ratings(self) -> Dataset:
        if self._surprise_dataset is not None:
            return self._surprise_dataset

        if self._ratings_df is None:
            self._ratings_df = self._load_ratings()

        if self.DEFAULT_RATING_SCALE is None:
            raise NotImplementedError(f"The class {self.__class__.__name__} must define DEFAULT_RATING_SCALE")

        reader = Reader(rating_scale=self.DEFAULT_RATING_SCALE)
        surprise_data = self._ratings_df[["user_id", "item_id", "rating"]].copy()
        self._surprise_dataset = Dataset.load_from_df(surprise_data, reader)

        return self._surprise_dataset

    def get_users(self) -> pd.DataFrame:
        if self._users_df is None:
            self._users_df = self._load_users()
        return self._users_df

    def get_items(self) -> pd.DataFrame:
        if self._items_df is None:
            self._items_df = self._load_items()
        return self._items_df
