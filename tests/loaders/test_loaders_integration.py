import pandas as pd
import pytest
from surprise import Dataset

from src.loaders import create_loader


class TestLoadersIntegration:

    @pytest.mark.parametrize("dataset_name", ["amazonmusic", "anime", "bookcrossing", "movielens", "movielens1m"])
    def test_loader_creation(self, dataset_name):
        try:
            loader = create_loader(dataset_name)
            assert loader is not None
        except FileNotFoundError:
            pytest.skip(f"Dataset {dataset_name} files not found")

    @pytest.mark.parametrize("dataset_name", ["amazonmusic", "anime", "bookcrossing", "movielens", "movielens1m"])
    def test_get_ratings_returns_surprise_dataset(self, dataset_name):
        try:
            loader = create_loader(dataset_name)
            ratings = loader.get_ratings()
            assert isinstance(ratings, Dataset)
        except FileNotFoundError:
            pytest.skip(f"Dataset {dataset_name} files not found")

    @pytest.mark.parametrize("dataset_name", ["amazonmusic", "anime", "bookcrossing", "movielens", "movielens1m"])
    def test_get_users_returns_dataframe(self, dataset_name):
        try:
            loader = create_loader(dataset_name)
            users = loader.get_users()
            assert isinstance(users, pd.DataFrame)
            assert "user_id" in users.columns
            assert len(users) > 0
        except FileNotFoundError:
            pytest.skip(f"Dataset {dataset_name} files not found")

    @pytest.mark.parametrize("dataset_name", ["amazonmusic", "anime", "bookcrossing", "movielens", "movielens1m"])
    def test_get_items_returns_dataframe(self, dataset_name):
        try:
            loader = create_loader(dataset_name)
            items = loader.get_items()
            assert isinstance(items, pd.DataFrame)
            assert "item_id" in items.columns
            assert len(items) > 0
        except FileNotFoundError:
            pytest.skip(f"Dataset {dataset_name} files not found")

    def test_interface_consistency(self):
        dataset_names = ["amazonmusic", "anime", "bookcrossing", "movielens", "movielens1m"]

        for dataset_name in dataset_names:
            try:
                loader = create_loader(dataset_name)

                assert hasattr(loader, "get_ratings")
                assert hasattr(loader, "get_users")
                assert hasattr(loader, "get_items")

                assert callable(loader.get_ratings)
                assert callable(loader.get_users)
                assert callable(loader.get_items)

            except FileNotFoundError:
                pytest.skip(f"Dataset {dataset_name} files not found")
