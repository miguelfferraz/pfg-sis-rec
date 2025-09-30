from typing import Dict, Type

from .algorithms.baseline_models import BaselineOnlyModel
from .algorithms.matrix_factorization import SVDModel
from .algorithms.advanced_matrix_factorization import SVDppModel, NMFModel
from .algorithms.knn_models import KNNBasicModel, KNNWithMeansModel, KNNWithZScoreModel, KNNBaselineModel
from .base_model import BaseModel


class ModelFactory:
    _models: Dict[str, Type[BaseModel]] = {
        "baseline": BaselineOnlyModel,
        
        "svd": SVDModel,
        "svdpp": SVDppModel,
        "nmf": NMFModel,
        
        "knn_basic": KNNBasicModel,
        "knn_means": KNNWithMeansModel,
        "knn_zscore": KNNWithZScoreModel,
        "knn_baseline": KNNBaselineModel,
    }

    @classmethod
    def create_model(cls, model_name: str, **params) -> BaseModel:
        if model_name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        
        model_class = cls._models[model_name]
        return model_class(**params)

    @classmethod
    def get_available_models(cls) -> list:
        return list(cls._models.keys())
