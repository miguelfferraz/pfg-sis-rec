from surprise import SVD

from ..base_model import BaseModel


class SVDModel(BaseModel):
    def __init__(self, **params):
        default_params = {
            'n_factors': 100,
            'n_epochs': 20,
            'lr_all': 0.005,
            'reg_all': 0.02,
            'biased': True
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = SVD(**self.params)

    def get_name(self) -> str:
        return "SVD"
