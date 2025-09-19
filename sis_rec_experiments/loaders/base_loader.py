from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class BaseDatasetLoader(ABC):
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.ratings_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self._validate_path()

    def _validate_path(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path não encontrado: {self.dataset_path}")

    @abstractmethod
    def load_ratings(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        pass

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
