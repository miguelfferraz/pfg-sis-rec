from surprise import KNNWithMeans

from src.models.base_model import BaseModel


class KNNWithMeansModel(BaseModel):
    """KNN that considers the average of user ratings."""

    def __init__(self, **params):
        default_params = {"k": 40, "min_k": 1, "sim_options": {"name": "cosine", "user_based": True}}
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = KNNWithMeans(**self.params)

    def get_name(self) -> str:
        user_based = self.params.get("sim_options", {}).get("user_based", True)
        return f"KNNWithMeans_{'User' if user_based else 'Item'}"
