import pandas as pd
import pytest

from src.loaders.loader_factory import LoaderFactory


@pytest.fixture
def movielens_loader():
    factory = LoaderFactory()
    return factory.create_loader("movielens")


@pytest.fixture
def sample_predictions_df():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3] * 2,
            "item_id": [10, 20, 10, 30, 20, 30] * 2,
            "rating_true": [5.0, 3.0, 4.0, 2.0, 5.0, 1.0] * 2,
            "rating_pred": [4.5, 3.5, 4.2, 2.5, 4.8, 1.5] * 2,
            "fold": [0] * 6 + [1] * 6,
            "model_name": ["svd"] * 12,
        }
    )


@pytest.fixture
def fairness_config():
    return {
        "user_attributes": [{"name": "gender", "type": "categorical"}, {"name": "age", "type": "categorical"}],
        "item_attributes": [{"name": "genre", "type": "categorical"}],
        "positive_threshold": 4.0,
    }
