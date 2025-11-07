from surprise import KNNBasic

from src.models.base_model import BaseModel


class KNNBasicModel(BaseModel):
    """
    Basic KNN using similarity between users or items.
    This model is a simple KNN model that uses the cosine similarity between users or items to predict the ratings.
    """

    def __init__(self, **params):
        default_params = {
            "k": 40,
            "min_k": 1,
            "sim_options": {
                "name": "cosine",
                "user_based": True,
            },
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = KNNBasic(**self.params)

    def get_name(self) -> str:
        user_based = self.params.get("sim_options", {}).get("user_based", True)
        return f"KNNBasic_{'User' if user_based else 'Item'}"
