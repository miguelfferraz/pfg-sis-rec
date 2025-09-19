from typing import Dict, Type

from .algorithms.baseline_models import BaselineOnlyModel
from .base_model import BaseModel


class ModelFactory:
    _models: Dict[str, Type[BaseModel]] = {
        "baseline": BaselineOnlyModel,
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
