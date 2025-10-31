from surprise import SVDpp

from src.models.base_model import BaseModel


class SVDppModel(BaseModel):
    """
    SVD++ (SVD Plus Plus) - SVD extension that considers implicit feedback.
    Takes into account not only explicit ratings, but also which items
    the user has rated (implicit feedback).
    """

    def __init__(self, **params):
        default_params = {
            "n_factors": 20,  # Number of latent factors (lower than SVD due to complexity)
            "n_epochs": 20,  # Training epochs
            "lr_all": 0.007,  # Learning rate
            "reg_all": 0.02,  # Regularization
            "cache_ratings": True,  # Cache for better performance
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = SVDpp(**self.params)

    def get_name(self) -> str:
        return "SVD++"
