import pandas as pd
import pytest


class TestSensitiveVariables:

    def test_movielens_users_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("movielens")
            users = loader.get_users()

            assert isinstance(users, pd.DataFrame)
            assert "user_id" in users.columns
            assert "gender" in users.columns
            assert "age" in users.columns
            assert "occupation" in users.columns

            assert users["gender"].dtype in ["int32", "int64"]
            assert users["age"].dtype in ["int32", "int64"]
            assert users["occupation"].dtype in ["int32", "int64"]

            user_mappings = loader.get_user_mappings()
            assert "gender" in user_mappings
            assert "occupation" in user_mappings

            assert loader.decode_user_variable("gender", 0) == "M"
            assert loader.decode_user_variable("gender", 1) == "F"

        except FileNotFoundError:
            pytest.skip("MovieLens dataset files not found")

    def test_movielens_items_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("movielens")
            items = loader.get_items()

            assert isinstance(items, pd.DataFrame)
            assert "item_id" in items.columns
            assert "primary_genre" in items.columns

            assert items["primary_genre"].dtype in ["int32", "int64"]

            item_mappings = loader.get_item_mappings()
            assert "primary_genre" in item_mappings

        except FileNotFoundError:
            pytest.skip("MovieLens dataset files not found")

    def test_mappings_consistency(self, loader_factory):
        try:
            loader = loader_factory.create_loader("movielens")
            users = loader.get_users()

            user_mappings = loader.get_user_mappings()

            for col in ["gender", "occupation"]:
                if col in users.columns:
                    unique_values = users[col].unique()
                    mapping_values = set(user_mappings[col].values())
                    for val in unique_values:
                        if not pd.isna(val):
                            assert val in mapping_values, f"Value {val} not found in {col} mapping"

        except FileNotFoundError:
            pytest.skip("MovieLens dataset files not found")

    def test_decode_invalid_variable(self, loader_factory):
        try:
            loader = loader_factory.create_loader("movielens")
            loader.get_users()

            result = loader.decode_user_variable("invalid_var", 0)
            assert result == 0

            result = loader.decode_user_variable("gender", 999)
            assert result == 999

        except FileNotFoundError:
            pytest.skip("MovieLens dataset files not found")

    def test_movielens1m_users_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("movielens1m")
            users = loader.get_users()

            assert isinstance(users, pd.DataFrame)
            assert "user_id" in users.columns
            assert "gender" in users.columns
            assert "age" in users.columns
            assert "occupation" in users.columns

            assert users["gender"].dtype in ["int32", "int64"]
            assert users["age"].dtype in ["int32", "int64"]
            assert users["occupation"].dtype in ["int32", "int64"]

            user_mappings = loader.get_user_mappings()
            assert "gender" in user_mappings
            assert "occupation" in user_mappings

            assert loader.decode_user_variable("gender", 0) in ["M", "F"]
            assert loader.decode_user_variable("gender", 1) in ["M", "F"]

        except FileNotFoundError:
            pytest.skip("MovieLens 1M dataset files not found")

    def test_movielens1m_items_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("movielens1m")
            items = loader.get_items()

            assert isinstance(items, pd.DataFrame)
            assert "item_id" in items.columns
            assert "primary_genre" in items.columns

            assert items["primary_genre"].dtype in ["int32", "int64"]

            item_mappings = loader.get_item_mappings()
            assert "primary_genre" in item_mappings

        except FileNotFoundError:
            pytest.skip("MovieLens 1M dataset files not found")

    def test_amazonmusic_users_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("amazonmusic")
            users = loader.get_users()

            assert isinstance(users, pd.DataFrame)
            assert "user_id" in users.columns
            assert "activity_level" in users.columns
            assert "rating_behavior" in users.columns
            assert "helpfulness_level" in users.columns

            assert users["activity_level"].dtype in ["int32", "int64"]
            assert users["rating_behavior"].dtype in ["int32", "int64"]
            assert users["helpfulness_level"].dtype in ["int32", "int64"]

            user_mappings = loader.get_user_mappings()
            assert "activity_level" in user_mappings
            assert "rating_behavior" in user_mappings
            assert "helpfulness_level" in user_mappings

        except FileNotFoundError:
            pytest.skip("Amazon Music dataset files not found")

    def test_amazonmusic_items_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("amazonmusic")
            items = loader.get_items()

            assert isinstance(items, pd.DataFrame)
            assert "item_id" in items.columns
            assert "primary_genre" in items.columns

            assert items["primary_genre"].dtype in ["int32", "int64"]

            item_mappings = loader.get_item_mappings()
            assert "primary_genre" in item_mappings

        except FileNotFoundError:
            pytest.skip("Amazon Music dataset files not found")

    def test_anime_users_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("anime")
            users = loader.get_users()

            assert isinstance(users, pd.DataFrame)
            assert "user_id" in users.columns
            assert "activity_level" in users.columns
            assert "rating_behavior" in users.columns
            assert "engagement_level" in users.columns

            assert users["activity_level"].dtype in ["int32", "int64"]
            assert users["rating_behavior"].dtype in ["int32", "int64"]
            assert users["engagement_level"].dtype in ["int32", "int64"]

            user_mappings = loader.get_user_mappings()
            assert "activity_level" in user_mappings
            assert "rating_behavior" in user_mappings
            assert "engagement_level" in user_mappings

        except FileNotFoundError:
            pytest.skip("Anime dataset files not found")

    def test_anime_items_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("anime")
            items = loader.get_items()

            assert isinstance(items, pd.DataFrame)
            assert "item_id" in items.columns
            assert "primary_genre" in items.columns
            assert "anime_type" in items.columns
            assert "episode_category" in items.columns

            assert items["primary_genre"].dtype in ["int32", "int64"]
            assert items["anime_type"].dtype in ["int32", "int64"]
            assert items["episode_category"].dtype in ["int32", "int64"]

            item_mappings = loader.get_item_mappings()
            assert "primary_genre" in item_mappings
            assert "anime_type" in item_mappings
            assert "episode_category" in item_mappings

        except FileNotFoundError:
            pytest.skip("Anime dataset files not found")

    def test_bookcrossing_users_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("bookcrossing")
            users = loader.get_users()

            assert isinstance(users, pd.DataFrame)
            assert "user_id" in users.columns
            assert "country" in users.columns
            assert "age" in users.columns
            assert "activity_level" in users.columns
            assert "rating_behavior" in users.columns

            assert users["country"].dtype in ["int32", "int64"]
            assert users["age"].dtype in ["int32", "int64", "float32", "float64"]
            assert users["activity_level"].dtype in ["int32", "int64"]
            assert users["rating_behavior"].dtype in ["int32", "int64"]

            user_mappings = loader.get_user_mappings()
            assert "country" in user_mappings
            assert "activity_level" in user_mappings
            assert "rating_behavior" in user_mappings

        except FileNotFoundError:
            pytest.skip("BookCrossing dataset files not found")

    def test_bookcrossing_items_sensitive_variables(self, loader_factory):
        try:
            loader = loader_factory.create_loader("bookcrossing")
            items = loader.get_items()

            assert isinstance(items, pd.DataFrame)
            assert "item_id" in items.columns
            assert "publication_year" in items.columns
            assert "publisher_book_count" in items.columns

            assert items["publication_year"].dtype in ["int32", "int64", "float32", "float64"]
            assert items["publisher_book_count"].dtype in ["int32", "int64", "float32", "float64"]

            loader.get_item_mappings()

        except FileNotFoundError:
            pytest.skip("BookCrossing dataset files not found")
