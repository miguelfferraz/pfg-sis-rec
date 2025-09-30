from surprise import SVDpp, NMF

from ..base_model import BaseModel


class SVDppModel(BaseModel):
    """
    SVD++ (SVD Plus Plus) - Extensão do SVD que considera feedback implícito.
    Leva em conta não apenas os ratings explícitos, mas também quais itens
    o usuário avaliou (feedback implícito).
    """
    
    def __init__(self, **params):
        default_params = {
            'n_factors': 20,        # Número de fatores latentes (menor que SVD por ser mais complexo)
            'n_epochs': 20,         # Épocas de treinamento
            'lr_all': 0.007,        # Taxa de aprendizado
            'reg_all': 0.02,        # Regularização
            'cache_ratings': True   # Cache para melhor performance
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = SVDpp(**self.params)

    def get_name(self) -> str:
        return "SVD++"


class NMFModel(BaseModel):
    """
    Non-negative Matrix Factorization (NMF) - Fatoração matricial não-negativa.
    Garante que todos os fatores latentes sejam não-negativos, o que pode
    ser interpretado como características "aditivas" dos usuários e itens.
    """
    
    def __init__(self, **params):
        default_params = {
            'n_factors': 15,        # Número de fatores latentes
            'n_epochs': 50,         # Mais épocas são necessárias para convergência
            'biased': False,        # NMF tradicionalmente não usa bias
            'reg_pu': 0.06,         # Regularização para fatores de usuário
            'reg_qi': 0.06,         # Regularização para fatores de item
            'init_low': 0,          # Inicialização mínima (não-negativo)
            'init_high': 1          # Inicialização máxima
        }
        default_params.update(params)
        super().__init__(**default_params)

    def _build_algorithm(self):
        self.algorithm = NMF(**self.params)

    def get_name(self) -> str:
        return "NMF"
