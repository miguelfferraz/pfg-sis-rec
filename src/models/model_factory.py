from typing import Dict, Type

from src.models.advanced_matrix_factorization.nmf_model import NMFModel
from src.models.advanced_matrix_factorization.svd_pp_model import SVDppModel
from src.models.base_model import BaseModel
from src.models.baseline.baseline_model import BaselineOnlyModel
from src.models.knn.knn_baseline_model import KNNBaselineModel
from src.models.knn.knn_basic_model import KNNBasicModel
from src.models.knn.knn_with_means_model import KNNWithMeansModel
from src.models.knn.knn_with_z_score_model import KNNWithZScoreModel
from src.models.matrix_factorization.svd_model import SVDModel


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
