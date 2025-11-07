from surprise import KNNWithZScore

from src.models.base_model import BaseModel


class KNNWithZScoreModel(BaseModel):
    """
    KNN that normalizes ratings using z-score.
    Z-score normalization is a technique that scales the ratings to have a mean of 0 and a standard deviation of 1.
    """

    def __init__(self, **params):
        default_params = {"k": 40, "min_k": 1, "sim_options": {"name": "cosine", "user_based": True}}
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = KNNWithZScore(**self.params)

    def get_name(self) -> str:
        user_based = self.params.get("sim_options", {}).get("user_based", True)
        return f"KNNWithZScore_{'User' if user_based else 'Item'}"
