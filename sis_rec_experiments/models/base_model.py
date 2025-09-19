from abc import ABC, abstractmethod
from typing import Any, Dict

from surprise import Dataset


class BaseModel(ABC):
    def __init__(self, **params):
        self.params = params
        self.algorithm = None
        self._build_algorithm()

    @abstractmethod
    def _build_algorithm(self):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def fit(self, trainset):
        return self.algorithm.fit(trainset)

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        return self.algorithm.predict(uid, iid, r_ui, clip, verbose)

    def test(self, testset, verbose=False):
        return self.algorithm.test(testset, verbose)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()
