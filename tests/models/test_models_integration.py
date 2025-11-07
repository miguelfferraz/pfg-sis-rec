import pytest

from src.models.model_factory import ModelFactory


class TestModelsIntegration:
    @pytest.mark.parametrize("model_name", ModelFactory.get_available_models())
    def test_model_creation_and_interface(self, model_name):
        model = ModelFactory.create_model(model_name)

        assert model is not None
        assert hasattr(model, "algorithm")
        assert hasattr(model, "params")
        assert callable(model.get_name)
        assert callable(model.fit)
        assert callable(model.predict)
        assert callable(model.test)
        assert callable(model.get_params)

    @pytest.mark.parametrize("model_name", ModelFactory.get_available_models())
    def test_model_has_valid_name(self, model_name):
        model = ModelFactory.create_model(model_name)
        name = model.get_name()

        assert isinstance(name, str)
        assert len(name) > 0

    @pytest.mark.parametrize("model_name", ModelFactory.get_available_models())
    def test_model_params_are_accessible(self, model_name):
        model = ModelFactory.create_model(model_name)
        params = model.get_params()

        assert isinstance(params, dict)

    def test_models_have_unique_names(self):
        model_names = set()
        for model_name in ModelFactory.get_available_models():
            model = ModelFactory.create_model(model_name)
            name = model.get_name()
            assert name not in model_names, f"Duplicate model name: {name}"
            model_names.add(name)

    def test_models_accept_custom_parameters(self):
        test_cases = [
            ("svd", {"n_factors": 50}),
            ("baseline", {"bsl_options": {"method": "sgd"}}),
            ("knn_basic", {"k": 20}),
        ]

        for model_name, custom_params in test_cases:
            model = ModelFactory.create_model(model_name, **custom_params)
            params = model.get_params()

            for key, value in custom_params.items():
                assert key in params
                assert params[key] == value

    def test_model_algorithm_is_initialized(self):
        for model_name in ModelFactory.get_available_models():
            model = ModelFactory.create_model(model_name)
            assert model.algorithm is not None
