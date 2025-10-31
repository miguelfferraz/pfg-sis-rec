import pandas as pd
import pytest

from src.preprocessors.min_ratings_filter import MinRatingsFilter
from src.preprocessors.rating_normalizer import RatingNormalizer


class TestPreprocessors:
    @pytest.fixture
    def sample_ratings_df(self):
        return pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
                "item_id": [1, 2, 3, 1, 2, 1, 2, 3, 4, 5],
                "rating": [5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )

    @pytest.fixture
    def sample_users_df(self):
        return pd.DataFrame({"user_id": [1, 2, 3, 4], "age": [25, 30, 35, 40]})

    @pytest.fixture
    def sample_items_df(self):
        return pd.DataFrame({"item_id": [1, 2, 3, 4, 5], "genre": [0, 1, 0, 1, 0]})

    def test_min_ratings_filter_users(self, sample_ratings_df, sample_users_df, sample_items_df):
        filter_processor = MinRatingsFilter({"min_user_ratings": 3})
        filtered_ratings, filtered_users, filtered_items = filter_processor.process(
            sample_ratings_df, sample_users_df, sample_items_df
        )

        assert len(filtered_ratings) == 7
        assert 1 in filtered_ratings["user_id"].values
        assert 3 in filtered_ratings["user_id"].values
        assert 2 not in filtered_ratings["user_id"].values
        assert 4 not in filtered_ratings["user_id"].values

    def test_min_ratings_filter_items(self, sample_ratings_df, sample_users_df, sample_items_df):
        filter_processor = MinRatingsFilter({"min_item_ratings": 3})
        filtered_ratings, filtered_users, filtered_items = filter_processor.process(
            sample_ratings_df, sample_users_df, sample_items_df
        )

        assert len(filtered_ratings) == 6
        assert 1 in filtered_ratings["item_id"].values
        assert 2 in filtered_ratings["item_id"].values
        assert 4 not in filtered_ratings["item_id"].values
        assert 5 not in filtered_ratings["item_id"].values

    def test_min_ratings_filter_both(self, sample_ratings_df, sample_users_df, sample_items_df):
        filter_processor = MinRatingsFilter({"min_user_ratings": 3, "min_item_ratings": 2})
        filtered_ratings, filtered_users, filtered_items = filter_processor.process(
            sample_ratings_df, sample_users_df, sample_items_df
        )

        assert len(filtered_ratings) == 6

    def test_rating_normalizer_z_score(self, sample_ratings_df, sample_users_df, sample_items_df):
        normalizer = RatingNormalizer({"method": "z_score"})
        normalized_ratings, _, _ = normalizer.process(sample_ratings_df, sample_users_df, sample_items_df)

        mean = normalized_ratings["rating"].mean()
        std = normalized_ratings["rating"].std()

        assert abs(mean) < 1e-10
        assert abs(std - 1.0) < 1e-10

    def test_rating_normalizer_min_max(self, sample_ratings_df, sample_users_df, sample_items_df):
        normalizer = RatingNormalizer({"method": "min_max"})
        normalized_ratings, _, _ = normalizer.process(sample_ratings_df, sample_users_df, sample_items_df)

        min_val = normalized_ratings["rating"].min()
        max_val = normalized_ratings["rating"].max()

        assert min_val == 0.0
        assert max_val == 1.0

    def test_rating_normalizer_default(self, sample_ratings_df, sample_users_df, sample_items_df):
        normalizer = RatingNormalizer({})
        normalized_ratings, _, _ = normalizer.process(sample_ratings_df, sample_users_df, sample_items_df)

        mean = normalized_ratings["rating"].mean()
        assert abs(mean) < 1e-10
