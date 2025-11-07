from unittest.mock import Mock

import pytest

from src.models.base_model import BaseModel


class TestBaseModel:
    def test_base_model_is_abstract(self):
        with pytest.raises(TypeError):
            BaseModel()

    def test_base_model_initialization(self):
        class TestModel(BaseModel):
            def _build_algorithm(self):
                self.algorithm = Mock()

            def get_name(self):
                return "TestModel"

        params = {"param1": "value1", "param2": 42}
        model = TestModel(**params)

        assert model.params == params
        assert model.algorithm is not None

    def test_fit_delegates_to_algorithm(self, mock_trainset):
        class TestModel(BaseModel):
            def _build_algorithm(self):
                self.algorithm = Mock()

            def get_name(self):
                return "TestModel"

        model = TestModel()
        result = model.fit(mock_trainset)

        model.algorithm.fit.assert_called_once_with(mock_trainset)
        assert result == model.algorithm.fit.return_value

    def test_predict_delegates_to_algorithm(self):
        class TestModel(BaseModel):
            def _build_algorithm(self):
                self.algorithm = Mock()

            def get_name(self):
                return "TestModel"

        model = TestModel()
        result = model.predict("user1", "item1", 4.0, True, False)

        model.algorithm.predict.assert_called_once_with("user1", "item1", 4.0, True, False)
        assert result == model.algorithm.predict.return_value

    def test_predict_with_default_parameters(self):
        class TestModel(BaseModel):
            def _build_algorithm(self):
                self.algorithm = Mock()

            def get_name(self):
                return "TestModel"

        model = TestModel()
        result = model.predict("user1", "item1")

        model.algorithm.predict.assert_called_once_with("user1", "item1", None, True, False)
        assert result == model.algorithm.predict.return_value

    def test_test_delegates_to_algorithm(self, mock_testset):
        class TestModel(BaseModel):
            def _build_algorithm(self):
                self.algorithm = Mock()

            def get_name(self):
                return "TestModel"

        model = TestModel()
        result = model.test(mock_testset, True)

        model.algorithm.test.assert_called_once_with(mock_testset, True)
        assert result == model.algorithm.test.return_value

    def test_test_with_default_verbose(self, mock_testset):
        class TestModel(BaseModel):
            def _build_algorithm(self):
                self.algorithm = Mock()

            def get_name(self):
                return "TestModel"

        model = TestModel()
        result = model.test(mock_testset)

        model.algorithm.test.assert_called_once_with(mock_testset, False)
        assert result == model.algorithm.test.return_value

    def test_get_params_returns_copy(self):
        class TestModel(BaseModel):
            def _build_algorithm(self):
                self.algorithm = Mock()

            def get_name(self):
                return "TestModel"

        params = {"param1": "value1"}
        model = TestModel(**params)
        returned_params = model.get_params()

        assert returned_params == params
        assert returned_params is not model.params

        returned_params["new_param"] = "new_value"
        assert "new_param" not in model.params
