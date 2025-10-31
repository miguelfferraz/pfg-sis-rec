import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from src.loaders.base_loader import BaseDatasetLoader
from src.loaders.loader_factory import LoaderFactory


@pytest.fixture
def loader_factory():
    return LoaderFactory()


@pytest.fixture(params=["movielens", "movielens1m", "amazonmusic", "anime", "bookcrossing"])
def dataset_name(request):
    return request.param


@pytest.fixture
def safe_loader(dataset_name, loader_factory):
    try:
        return loader_factory.create_loader(dataset_name)
    except FileNotFoundError:
        pytest.skip(f"Dataset {dataset_name} files not found")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(
    params=[
        ("movielens", "MovieLensLoader", "u.data"),
        ("movielens1m", "MovieLens1MLoader", "ratings.dat"),
        ("amazonmusic", "AmazonMusicLoader", "Digital_Music_5.json"),
        ("anime", "AnimeLoader", "anime_ratings.dat"),
        ("bookcrossing", "BookCrossingLoader", "book_ratings.dat"),
    ]
)
def loader_info(request):
    dataset_name, class_name, main_file = request.param

    if class_name == "MovieLensLoader":
        from src.loaders.movielens_loader import MovieLensLoader

        loader_class = MovieLensLoader
    elif class_name == "MovieLens1MLoader":
        from src.loaders.movielens_1m_loader import MovieLens1MLoader

        loader_class = MovieLens1MLoader
    elif class_name == "AmazonMusicLoader":
        from src.loaders.amazon_music_loader import AmazonMusicLoader

        loader_class = AmazonMusicLoader
    elif class_name == "AnimeLoader":
        from src.loaders.anime_loader import AnimeLoader

        loader_class = AnimeLoader
    elif class_name == "BookCrossingLoader":
        from src.loaders.book_crossing_loader import BookCrossingLoader

        loader_class = BookCrossingLoader

    return {
        "dataset_name": dataset_name,
        "class_name": class_name,
        "loader_class": loader_class,
        "main_file": main_file,
    }


@pytest.fixture
def movielens_loader_class():
    from src.loaders.movielens_loader import MovieLensLoader

    return MovieLensLoader


@pytest.fixture
def minimal_dataset_files(temp_dir, loader_info):
    main_file = temp_dir / loader_info["main_file"]

    if loader_info["dataset_name"] in ["movielens", "movielens1m"]:
        if loader_info["main_file"] == "u.data":
            main_file.write_text("1\t1\t5\t123456789\n2\t2\t4\t123456790\n")
        else:
            main_file.write_text("1::1::5::123456789\n2::2::4::123456790\n")

    elif loader_info["dataset_name"] == "amazonmusic":
        main_file.write_text(
            '{"reviewerID": "A1", "asin": "B1", "overall": 5.0, "unixReviewTime": 123456789, "helpful": [1, 1]}\n'
            '{"reviewerID": "A2", "asin": "B2", "overall": 4.0, "unixReviewTime": 123456790, "helpful": [0, 1]}\n'
        )

    elif loader_info["dataset_name"] == "anime":
        main_file.write_text("User_ID\tAnime_ID\tFeedback\n1\t1\t8.0\n2\t2\t7.0\n")

        history_file = temp_dir / "anime_history.dat"
        history_file.write_text("User_ID\tAnime_ID\tFeedback\n1\t1\t1\n2\t2\t1\n")

        info_file = temp_dir / "anime_info.dat"
        info_file.write_text("anime_ids\tgenre\ttype\tepisodes\n1\tAction\tTV\t12\n2\tComedy\tMovie\t1\n")

    elif loader_info["dataset_name"] == "bookcrossing":
        main_file.write_text("user\titem\trating\n1\t1\t8.0\n2\t2\t7.0\n")

        users_file = temp_dir / "users_info.dat"
        users_file.write_text(
            "User-ID\tLocation\tAge\n1\tsan francisco, california, usa\t25\n2\tnew york, new york, usa\t30\n"
        )

        items_file = temp_dir / "items_info.dat"
        items_file.write_text(
            "Book_ID\tISBN\tBook-Title\tBook-Author\tYear-Of-Publication\tPublisher\tImage-URL-S\tImage-URL-M\tImage-URL-L\n1\t123456789\tTest Book 1\tTest Author 1\t2000\tTest Publisher\turl1\turl2\turl3\n2\t987654321\tTest Book 2\tTest Author 2\t2001\tTest Publisher\turl1\turl2\turl3\n"
        )

    return main_file


@pytest.fixture
def empty_dataset_file(temp_dir, loader_info):
    main_file = temp_dir / loader_info["main_file"]
    main_file.write_text("")
    return main_file


@pytest.fixture
def corrupted_dataset_file(temp_dir, loader_info):
    main_file = temp_dir / loader_info["main_file"]
    main_file.write_text("invalid\tdata\there\n")
    return main_file


@pytest.fixture
def minimal_ratings_file(temp_dir):
    ratings_file = temp_dir / "u.data"
    ratings_file.write_text("1\t1\t5\t123456789\n2\t2\t4\t123456790\n")
    return ratings_file


@pytest.fixture
def empty_ratings_file(temp_dir):
    ratings_file = temp_dir / "u.data"
    ratings_file.write_text("")
    return ratings_file


@pytest.fixture
def corrupted_ratings_file(temp_dir):
    ratings_file = temp_dir / "u.data"
    ratings_file.write_text("invalid\tdata\there\n")
    return ratings_file


@pytest.fixture
def test_loader_class():
    class TestLoader(BaseDatasetLoader):
        def _load_ratings(self):
            return pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [5.0]})

        def _load_users(self):
            return pd.DataFrame({"user_id": [1]})

        def _load_items(self):
            return pd.DataFrame({"item_id": [1]})

    return TestLoader


@pytest.fixture
def complete_test_loader_class():
    class CompleteTestLoader(BaseDatasetLoader):
        DEFAULT_RATING_SCALE = (1.0, 5.0)

        def _load_ratings(self):
            return pd.DataFrame({"user_id": [1, 2, 1], "item_id": [1, 1, 2], "rating": [5.0, 4.0, 3.0]})

        def _load_users(self):
            return pd.DataFrame({"user_id": [1, 2]})

        def _load_items(self):
            return pd.DataFrame({"item_id": [1, 2]})

    return CompleteTestLoader


def assert_basic_dataframe_structure(df: pd.DataFrame, required_column: str):
    assert isinstance(df, pd.DataFrame)
    assert required_column in df.columns
    if len(df) > 0:
        assert df[required_column].notna().all()


def assert_loader_basic_functionality(loader: BaseDatasetLoader):
    from surprise import Dataset

    users = loader.get_users()
    items = loader.get_items()
    ratings = loader.get_ratings()

    assert_basic_dataframe_structure(users, "user_id")
    assert_basic_dataframe_structure(items, "item_id")
    assert isinstance(ratings, Dataset)
