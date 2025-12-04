import pytest

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


class TestModelFactory:
    def test_create_model_valid_names(self):
        for model_name in ModelFactory.get_available_models():
            model = ModelFactory.create_model(model_name)
            assert isinstance(model, BaseModel)
            assert model.get_name() is not None

    def test_create_model_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown model: invalid"):
            ModelFactory.create_model("invalid")

    def test_create_model_with_params(self):
        model = ModelFactory.create_model("svd", n_factors=50, n_epochs=10)
        params = model.get_params()
        assert params["n_factors"] == 50
        assert params["n_epochs"] == 10

    def test_get_available_models_returns_list(self):
        models = ModelFactory.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_all_registered_models_exist(self):
        expected_models = [
            "baseline",
            "svd",
            "svdpp",
            "nmf",
            "knn_basic",
            "knn_means",
            "knn_zscore",
            "knn_baseline",
        ]
        available_models = ModelFactory.get_available_models()

        for model_name in expected_models:
            assert model_name in available_models

    def test_models_have_different_names(self):
        model_names = set()
        for model_name in ModelFactory.get_available_models():
            model = ModelFactory.create_model(model_name)
            name = model.get_name()
            assert name not in model_names
            model_names.add(name)
