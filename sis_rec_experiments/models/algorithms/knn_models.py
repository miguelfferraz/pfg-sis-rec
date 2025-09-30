from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline

from ..base_model import BaseModel


class KNNBasicModel(BaseModel):
    """KNN básico usando similaridade entre usuários ou itens."""
    
    def __init__(self, **params):
        default_params = {
            'k': 40,                    # Número de vizinhos
            'min_k': 1,                 # Mínimo de vizinhos
            'sim_options': {
                'name': 'cosine',       # Métrica de similaridade: cosine, msd, pearson
                'user_based': True      # True: user-based, False: item-based
            }
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = KNNBasic(**self.params)

    def get_name(self) -> str:
        user_based = self.params.get('sim_options', {}).get('user_based', True)
        return f"KNNBasic_{'User' if user_based else 'Item'}"


class KNNWithMeansModel(BaseModel):
    """KNN que considera a média dos ratings dos usuários."""
    
    def __init__(self, **params):
        default_params = {
            'k': 40,
            'min_k': 1,
            'sim_options': {
                'name': 'cosine',
                'user_based': True
            }
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = KNNWithMeans(**self.params)

    def get_name(self) -> str:
        user_based = self.params.get('sim_options', {}).get('user_based', True)
        return f"KNNWithMeans_{'User' if user_based else 'Item'}"


class KNNWithZScoreModel(BaseModel):
    """KNN que normaliza ratings usando z-score."""
    
    def __init__(self, **params):
        default_params = {
            'k': 40,
            'min_k': 1,
            'sim_options': {
                'name': 'cosine',
                'user_based': True
            }
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = KNNWithZScore(**self.params)

    def get_name(self) -> str:
        user_based = self.params.get('sim_options', {}).get('user_based', True)
        return f"KNNWithZScore_{'User' if user_based else 'Item'}"


class KNNBaselineModel(BaseModel):
    """KNN que considera as baselines dos usuários e itens."""
    
    def __init__(self, **params):
        default_params = {
            'k': 40,
            'min_k': 1,
            'sim_options': {
                'name': 'cosine',
                'user_based': True
            },
            'bsl_options': {
                'method': 'als',
                'n_epochs': 5,
                'reg_u': 15,
                'reg_i': 10
            }
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = KNNBaseline(**self.params)

    def get_name(self) -> str:
        user_based = self.params.get('sim_options', {}).get('user_based', True)
        return f"KNNBaseline_{'User' if user_based else 'Item'}"
