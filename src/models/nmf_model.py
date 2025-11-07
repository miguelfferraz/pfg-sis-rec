from surprise import NMF

from src.models.base_model import BaseModel


class NMFModel(BaseModel):
    """
    Non-negative Matrix Factorization (NMF) - Non-negative matrix factorization.
    Ensures that all latent factors are non-negative, which can be
    interpreted as "additive" characteristics of users and items.
    """

    def __init__(self, **params):
        default_params = {
            "n_factors": 15,  # Number of latent factors
            "n_epochs": 50,  # More epochs are needed for convergence
            "biased": False,  # NMF traditionally doesn't use bias
            "reg_pu": 0.06,  # Regularization for user factors
            "reg_qi": 0.06,  # Regularization for item factors
            "init_low": 0,  # Minimum initialization (non-negative)
            "init_high": 1,  # Maximum initialization
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = NMF(**self.params)

    def get_name(self) -> str:
        return "NMF"
