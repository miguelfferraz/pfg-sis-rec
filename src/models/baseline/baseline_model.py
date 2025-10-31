from surprise import BaselineOnly

from src.models.base_model import BaseModel


class BaselineOnlyModel(BaseModel):
    def __init__(self, **params):
        default_params = {"bsl_options": {"method": "als", "n_epochs": 10, "reg_u": 15, "reg_i": 10}}
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = BaselineOnly(**self.params)

    def get_name(self) -> str:
        return "BaselineOnly"
