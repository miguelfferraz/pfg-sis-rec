from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
from surprise import Dataset


class BasePreprocessor(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def process(
        self, ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

