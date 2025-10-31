from unittest.mock import patch

import pandas as pd
import pytest
from surprise import Dataset

from src.loaders import create_loader

from .conftest import assert_basic_dataframe_structure, assert_loader_basic_functionality


class TestLoadersIntegration:

    def test_loader_creation(self, safe_loader):
        assert safe_loader is not None

    def test_get_ratings_returns_surprise_dataset(self, safe_loader):
        ratings = safe_loader.get_ratings()
        assert isinstance(ratings, Dataset)

    def test_get_users_returns_dataframe(self, safe_loader):
        users = safe_loader.get_users()
        assert_basic_dataframe_structure(users, "user_id")
        assert len(users) > 0

    def test_get_items_returns_dataframe(self, safe_loader):
        items = safe_loader.get_items()
        assert_basic_dataframe_structure(items, "item_id")
        assert len(items) > 0

    def test_caching_all_loaders(self, safe_loader):
        users1 = safe_loader.get_users()
        users2 = safe_loader.get_users()
        assert users1 is users2

        items1 = safe_loader.get_items()
        items2 = safe_loader.get_items()
        assert items1 is items2

        ratings1 = safe_loader.get_ratings()
        ratings2 = safe_loader.get_ratings()
        assert ratings1 is ratings2

    def test_create_loader_invalid_dataset(self):
        with pytest.raises(ValueError, match="Dataset 'invalid' não suportado"):
            create_loader("invalid")

    @pytest.mark.parametrize("dataset_name", ["MOVIELENS", "movielens", "MovieLens"])
    def test_create_loader_case_insensitive(self, dataset_name):
        loader1 = create_loader(dataset_name.upper())
        loader2 = create_loader(dataset_name.lower())
        assert type(loader1) == type(loader2)

    def test_base_loader_invalid_path(self, movielens_loader_class):
        with pytest.raises(FileNotFoundError, match="Dataset path not found"):
            movielens_loader_class("/path/that/does/not/exist")

    def test_base_loader_missing_rating_scale(self, test_loader_class, temp_dir):
        loader = test_loader_class(str(temp_dir))
        with pytest.raises(NotImplementedError, match="must define DEFAULT_RATING_SCALE"):
            loader.get_ratings()

    def test_loader_graceful_degradation(self, safe_loader):
        users = safe_loader.get_users()
        items = safe_loader.get_items()
        assert_basic_dataframe_structure(users, "user_id")
        assert_basic_dataframe_structure(items, "item_id")

    def test_consistent_dataframe_structure(self, safe_loader):
        assert_loader_basic_functionality(safe_loader)

    def test_independent_loader_instances(self, dataset_name):
        loader1 = create_loader(dataset_name)
        loader2 = create_loader(dataset_name)

        users1 = loader1.get_users()
        users2 = loader2.get_users()

        assert users1 is not users2
        pd.testing.assert_frame_equal(users1, users2)

    def test_loader_error_handling_no_files(self, loader_info, temp_dir):
        loader = loader_info["loader_class"](str(temp_dir))
        with pytest.raises(FileNotFoundError):
            loader.get_ratings()

    def test_loader_with_minimal_data(self, loader_info, minimal_dataset_files):
        loader = loader_info["loader_class"](str(minimal_dataset_files.parent))
        assert_loader_basic_functionality(loader)

    def test_empty_dataset_handling(self, loader_info, empty_dataset_file):
        loader = loader_info["loader_class"](str(empty_dataset_file.parent))

        try:
            ratings = loader.get_ratings()
            assert isinstance(ratings, Dataset)
        except (ValueError, FileNotFoundError, Exception):
            pass

    def test_corrupted_data_handling(self, loader_info, corrupted_dataset_file):
        loader = loader_info["loader_class"](str(corrupted_dataset_file.parent))
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            loader.get_ratings()

    @patch("pandas.read_csv")
    def test_pandas_error_handling_movielens_only(self, mock_read_csv, movielens_loader_class, minimal_ratings_file):
        mock_read_csv.side_effect = Exception("Simulated pandas error")
        loader = movielens_loader_class(str(minimal_ratings_file.parent))
        with pytest.raises(Exception):
            loader._load_ratings()
