from surprise import KNNBaseline

from src.models.base_model import BaseModel


class KNNBaselineModel(BaseModel):
    """
    KNN that considers user and item baselines.
    This model is a combination of KNNBasic and BaselineOnly.
    It uses the KNNBasic model to find the nearest neighbors and the BaselineOnly model to predict the ratings.
    """

    def __init__(self, **params):
        default_params = {
            "k": 40,
            "min_k": 1,
            "sim_options": {"name": "cosine", "user_based": True},
            "bsl_options": {"method": "als", "n_epochs": 5, "reg_u": 15, "reg_i": 10},
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = KNNBaseline(**self.params)

    def get_name(self) -> str:
        user_based = self.params.get("sim_options", {}).get("user_based", True)
        return f"KNNBaseline_{'User' if user_based else 'Item'}"
