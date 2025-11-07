import pytest


@pytest.fixture
def sample_config():
    return {
        "dataset": {"name": "movielens", "path": "src/datasets/extracted/movielens"},
        "preprocessing": [{"type": "min_ratings_filter", "params": {"min_user_ratings": 5, "min_item_ratings": 3}}],
        "training": {
            "models": [{"name": "svd", "params": {"n_factors": 10, "n_epochs": 5}}],
            "validation": {"type": "train_test_split", "test_size": 0.2, "metrics": ["rmse", "mae"]},
        },
        "output": {"save_path": "tests/pipeline/test_results"},
    }


@pytest.fixture
def sample_config_cross_validate():
    return {
        "dataset": {"name": "movielens"},
        "training": {
            "models": [{"name": "baseline", "params": {}}],
            "validation": {"type": "cross_validate", "cv": 3, "metrics": ["rmse"]},
        },
        "output": {"save_path": "tests/pipeline/test_results_cv"},
    }


@pytest.fixture
def minimal_config():
    return {"dataset": {"name": "movielens"}, "training": {"models": [{"name": "baseline", "params": {}}]}}
